"""Moving an existing knowledge base into the retrieval store.

Everything taught with `ev learn` before the retrieval layer existed lives in
``knowledge.sqlite3`` as a flat FTS table of passages. The new store wants
documents with chunks, namespaces and vectors. This carries the old data
across so nobody's library disappears the day they upgrade.

Passages move one-to-one into chunks rather than being reassembled and
re-chunked. The old chunker overlapped its output, so stitching passages back
together would duplicate a sentence at every boundary and then chunk *that* -
worse than leaving them as they are. They were already sized sensibly, so
they stand up fine as chunks; ``ev reindex`` can rebuild properly from source
later if you want the improved chunking.

Idempotent: the store's content hash means running this twice adds nothing.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ev_assistant import namespaces
from ev_assistant.store import Store

logger = logging.getLogger(__name__)


class MigrationReport:
    def __init__(self):
        self.documents = 0
        self.chunks = 0
        self.skipped = 0

    def __repr__(self) -> str:
        return (f"MigrationReport(documents={self.documents}, chunks={self.chunks}, "
                f"skipped={self.skipped})")

    def __bool__(self) -> bool:
        return self.documents > 0


def _source_type(source: str) -> str:
    source = (source or "").lower()
    if source.startswith("wikipedia:"):
        return "wikipedia"
    if source.startswith(("http://", "https://")):
        return "web"
    if source.endswith(".pdf"):
        return "pdf"
    if "/" in source or "\\" in source:
        return "file"
    return "note"


def read_passages(path: Path) -> dict[tuple[str, str], list[str]]:
    """{(title, source): [passage, ...]} from an old knowledge database."""
    if not Path(path).is_file():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()}
        if "passages" not in tables:
            return {}
        rows = conn.execute(
            "SELECT title, source, text FROM passages ORDER BY rowid"
        ).fetchall()
    except sqlite3.DatabaseError as e:
        logger.warning("Couldn't read the old knowledge base at %s: %s", path, e)
        return {}
    finally:
        conn.close()

    grouped: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        text = (row["text"] or "").strip()
        if text:
            grouped.setdefault((row["title"] or "Untitled", row["source"] or ""), []).append(text)
    return grouped


def migrate(knowledge_path: Path, store: Store) -> MigrationReport:
    """Copy an old knowledge base into `store`. Safe to run repeatedly."""
    report = MigrationReport()
    for (title, source), passages in read_passages(knowledge_path).items():
        text = "\n\n".join(passages)
        source_type = _source_type(source)
        document_id = store.add_document(
            source_uri=source or f"legacy:{title}",
            source_type=source_type,
            title=title,
            text=text,
            namespace=namespaces.route(source_type, source),
            chunks=[{"text": p, "ordinal": i} for i, p in enumerate(passages)],
        )
        if document_id is None:
            report.skipped += 1       # already in the store
            continue
        report.documents += 1
        report.chunks += len(passages)
    if report.documents:
        logger.info("Migrated %s documents (%s passages) from the old knowledge base",
                    report.documents, report.chunks)
    return report


def migrate_if_needed(cfg, store: Store) -> MigrationReport:
    """Run the migration once, when the store is empty and old data exists."""
    if store.stats()["documents"]:
        return MigrationReport()
    if store.get_meta("legacy_migrated"):
        return MigrationReport()
    report = migrate(cfg.knowledge_path, store)
    store.set_meta("legacy_migrated", "1")
    return report
