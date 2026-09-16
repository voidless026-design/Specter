"""The ingest path - one place that takes a fetched document all the way in.

Chunk it, route it to a namespace, store it, embed it, enrich it. Every way
into the knowledge base (`ev learn`, the GUI's LEARN panel, the feed loop)
goes through here, so they can't drift apart.

Ordering matters and is deliberate: the document is stored *first*, then
embedded, then enriched. Storing is what makes it findable by keyword
straight away; embedding adds semantic search; enrichment adds summaries and
facts. Each step is resumable on its own, so an interrupt anywhere leaves a
store that still works, just with less of it indexed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ev_assistant import namespaces
from ev_assistant.chunking import chunk_rows
from ev_assistant.store import Store

logger = logging.getLogger(__name__)


@dataclass
class LearnResult:
    """What happened to one document."""

    title: str
    source: str
    document_id: int | None = None
    chunks: int = 0
    skipped: bool = False

    @property
    def added(self) -> bool:
        return self.document_id is not None


@dataclass
class LearnReport:
    """What happened to a whole crawl."""

    documents: list[LearnResult] = field(default_factory=list)
    embedded: int = 0
    enriched: int = 0

    @property
    def pages(self) -> int:
        return len(self.documents)

    @property
    def added(self) -> int:
        return sum(1 for d in self.documents if d.added)

    @property
    def skipped(self) -> int:
        return sum(1 for d in self.documents if d.skipped)

    @property
    def chunks(self) -> int:
        return sum(d.chunks for d in self.documents)


class Library:
    """Everything that puts knowledge into the store."""

    def __init__(self, cfg, store: Store | None = None, embedder=None, enricher=None):
        self.cfg = cfg
        self.store = store or Store(cfg.store_path)
        self._embedder = embedder
        self._enricher = enricher

    @property
    def embedder(self):
        if self._embedder is None:
            from ev_assistant.embeddings import Embedder

            self._embedder = Embedder(self.cfg, self.store)
        return self._embedder

    @property
    def enricher(self):
        if self._enricher is None:
            from ev_assistant.enrich import Enricher

            self._enricher = Enricher(self.cfg, self.store)
        return self._enricher

    def add(self, title: str, text: str, source: str, *, source_type: str = "",
            namespace: str = "", published_at: float | None = None,
            force: bool = False) -> LearnResult:
        """Chunk and store one document. Does not embed - that comes after."""
        source_type = source_type or _sniff_type(source)
        namespace = namespace or namespaces.route(source_type, source,
                                                  getattr(self.cfg, "namespace_rules", None))
        chunks = chunk_rows(text, title=title, source_uri=source, source_type=source_type)
        if not chunks:
            return LearnResult(title=title, source=source, skipped=True)
        document_id = self.store.add_document(
            source_uri=source, source_type=source_type, title=title, text=text,
            chunks=chunks, namespace=namespace, published_at=published_at,
            token_count=sum(c["token_count"] for c in chunks), force=force,
        )
        if document_id is None:
            return LearnResult(title=title, source=source, skipped=True)
        return LearnResult(title=title, source=source, document_id=document_id,
                           chunks=len(chunks))

    def index(self, progress=None) -> int:
        """Embed everything waiting. Returns how many chunks were indexed."""
        try:
            return self.embedder.index_pending(progress=progress).embedded
        except Exception as e:
            # A missing model shouldn't fail a learn - keyword search works
            # now, and `ev reindex` picks the vectors up later.
            logger.warning("Couldn't build vectors yet: %s", e)
            return 0

    def enrich(self, limit: int | None = None) -> int:
        """Summarise and extract facts for whatever is waiting."""
        try:
            return self.enricher.drain(limit=limit).documents
        except Exception as e:
            logger.warning("Couldn't enrich yet: %s", e)
            return 0

    def learn(self, source: str, *, depth: int = 0, max_pages: int = 20,
              same_domain: bool = True, force: bool = False, on_page=None,
              enrich: bool = True) -> LearnReport:
        """Fetch a source (crawling if asked), store, index and enrich it."""
        from ev_assistant import ingest

        report = LearnReport()
        for doc in ingest.crawl(source, depth=depth, max_pages=max_pages,
                                same_domain=same_domain):
            outcome = self.add(doc.title, doc.text, doc.source, force=force)
            report.documents.append(outcome)
            if on_page is not None:
                on_page(outcome)
        report.embedded = self.index()
        if enrich:
            report.enriched = self.enrich()
        return report


def _sniff_type(source: str) -> str:
    source = (source or "").lower()
    if source.startswith("wikipedia:"):
        return "wikipedia"
    if source.startswith(("http://", "https://")):
        return "web"
    if source.endswith(".pdf"):
        return "pdf"
    if source.endswith((".epub",)):
        return "epub"
    if "/" in source or "\\" in source:
        return "file"
    return "note"
