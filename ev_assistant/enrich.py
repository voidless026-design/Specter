"""Ingest-time enrichment - summaries and atomic facts.

Two things get added to every document after it lands, both optional and
both paid for in model calls:

- A **summary** of three to five sentences on ``documents.summary``. It makes
  a cheap first-pass filter, and it lets E.V. cite a source without dragging
  full chunks into the context window.
- **Atomic facts** in the ``facts`` table: one subject-predicate statement
  per row. For a factual question these retrieve and inject far more
  efficiently than prose - a row saying "Tungsten melts at 3422 C" beats a
  500-token paragraph that happens to contain it.

Both run on a background queue, because ingest must not wait on a model. A
crawl of 200 Wikipedia pages should finish at network speed and enrich
afterwards, not take 200 model round-trips inline.

``documents.enrichment_status`` is the bookmark, so this is resumable the
same way embedding is, and a document that genuinely yields no facts is
marked done rather than retried forever.

On structured output: the spec asks for structured LLM output, and Claude
supports it natively - but E.V. also runs on Ollama and on whatever
OpenAI-compatible endpoint you point her at, where support is uneven. So the
model is asked for JSON and the reply is parsed defensively. A model that
wraps its JSON in prose or a code fence still works; one that returns
nothing usable costs a document its facts, not the run.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading

from ev_assistant.store import ENRICH_DONE, ENRICH_FAILED, Store

logger = logging.getLogger(__name__)

SUMMARY_MAX_TOKENS = 320
FACTS_MAX_TOKENS = 1024
DEFAULT_MAX_FACTS = 12
DEFAULT_ENRICH_CHARS = 6000
DEFAULT_BATCH = 4

# A fenced or prose-wrapped JSON array - models add framing whatever you ask.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

_SUMMARY_SYSTEM = (
    "Summarise the document in three to five plain sentences. Say what it is "
    "about and what someone could learn from it. Use only what the document "
    "says - do not add outside knowledge and do not speculate. "
    "Reply with the summary only: no preamble, no bullet points, no heading."
)

_FACTS_SYSTEM = (
    "Extract the standalone factual claims from the document.\n"
    "Each fact must be a single self-contained sentence that makes sense with "
    "no other context - name the subject explicitly instead of saying 'it' or "
    "'this'.\n"
    "Only extract what the document actually states. Do not infer, generalise, "
    "or add anything you happen to know.\n"
    'Reply with a JSON array and nothing else. Each element: {"statement": '
    '"<the fact>", "subject": "<what it is about>", "confidence": <0.0-1.0, '
    "how clearly the document states it>}.\n"
    "If the document contains no clear factual claims, reply with []."
)


class EnrichmentSettings:
    """What enrichment should do, read off config once."""

    def __init__(self, cfg=None):
        self.summaries = bool(getattr(cfg, "enrich_summaries", True))
        self.facts = bool(getattr(cfg, "enrich_facts", True))
        self.max_facts = int(getattr(cfg, "max_facts_per_document", DEFAULT_MAX_FACTS))
        self.max_chars = int(getattr(cfg, "enrich_chars", DEFAULT_ENRICH_CHARS))
        self.batch_size = max(1, int(getattr(cfg, "enrich_batch_size", DEFAULT_BATCH)))

    @property
    def enabled(self) -> bool:
        return self.summaries or self.facts

    def describe(self) -> str:
        parts = [name for name, on in
                 (("summaries", self.summaries), ("facts", self.facts)) if on]
        return " + ".join(parts) or "nothing (disabled)"


# ---------------------------------------------------------------------------
# The two model calls
# ---------------------------------------------------------------------------


def summarize(text: str, title: str = "", cfg=None, complete=None) -> str:
    """Three to five sentences, or "" if no brain answered."""
    if not text.strip() or cfg is None or complete is None:
        return ""
    prompt = f"Title: {title}\n\n{text}" if title else text
    answer = complete(cfg, _SUMMARY_SYSTEM, prompt, SUMMARY_MAX_TOKENS)
    return (answer or "").strip()


def parse_facts(reply: str, max_facts: int = DEFAULT_MAX_FACTS) -> list[dict]:
    """Pull a fact list out of whatever the model actually sent back.

    Tolerates code fences and surrounding prose. Anything malformed is
    dropped rather than stored, because a garbled fact is worse than a
    missing one - it retrieves and gets injected as if it were true.
    """
    if not reply:
        return []
    cleaned = _FENCE_RE.sub("", reply).strip()
    payload = None
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        match = _JSON_ARRAY_RE.search(cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.debug("Fact extraction returned unparseable JSON: %.200s", reply)
                return []
    if isinstance(payload, dict):
        # Some models wrap the array: {"facts": [...]}.
        for key in ("facts", "statements", "results", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []

    facts: list[dict] = []
    seen: set[str] = set()
    for item in payload:
        if isinstance(item, str):
            item = {"statement": item}
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or item.get("fact") or "").strip()
        if len(statement) < 8 or len(statement) > 500:
            continue
        key = statement.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        facts.append({
            "statement": statement,
            "subject": str(item.get("subject") or "").strip()[:200],
            "confidence": min(1.0, max(0.0, confidence)),
        })
        if len(facts) >= max_facts:
            break
    return facts


def extract_facts(text: str, title: str = "", cfg=None, complete=None,
                  max_facts: int = DEFAULT_MAX_FACTS) -> list[dict] | None:
    """Atomic subject-predicate claims.

    Returns None when no brain answered at all, and [] when one did and found
    nothing worth recording. The caller needs the difference: the first means
    try again later, the second means this document is finished.
    """
    if not text.strip() or cfg is None or complete is None:
        return None
    prompt = f"Title: {title}\n\n{text}" if title else text
    reply = complete(cfg, _FACTS_SYSTEM, prompt, FACTS_MAX_TOKENS)
    if reply is None:
        return None
    return parse_facts(reply, max_facts)


# ---------------------------------------------------------------------------
# The background queue
# ---------------------------------------------------------------------------


class EnrichReport:
    """What one enrichment pass got through."""

    def __init__(self):
        self.summarized = 0
        self.facts_added = 0
        self.documents = 0
        self.failed = 0
        self.skipped = 0

    def __repr__(self) -> str:
        return (f"EnrichReport(documents={self.documents}, summarized={self.summarized}, "
                f"facts_added={self.facts_added}, failed={self.failed}, "
                f"skipped={self.skipped})")


class Enricher:
    """Summaries and facts, off the ingest path.

    `enqueue()` is non-blocking by design: ingest hands over a document id and
    carries on. `drain()` is the synchronous version, for the CLI and tests.
    """

    def __init__(self, cfg, store: Store, complete=None):
        self.cfg = cfg
        self.store = store
        self.settings = EnrichmentSettings(cfg)
        if complete is None:
            from ev_assistant import providers
            complete = providers.complete
        self.complete = complete
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- one document --------------------------------------------------

    def enrich_document(self, document_id: int, report: EnrichReport | None = None
                        ) -> EnrichReport:
        """Summarise and extract facts for one document."""
        report = report or EnrichReport()
        document = self.store.get_document(document_id)
        if document is None:
            report.skipped += 1
            return report
        if not self.settings.enabled:
            # Disabled is not "pending forever" - mark it done and move on.
            self.store.mark_enriched(document_id)
            report.skipped += 1
            return report

        text = self.store.document_text(document_id, self.settings.max_chars)
        if not text.strip():
            self.store.mark_enriched(document_id)
            report.skipped += 1
            return report

        did_something = False
        try:
            if self.settings.summaries:
                summary = summarize(text, document.title, self.cfg, self.complete)
                if summary:
                    self.store.set_summary(document_id, summary)
                    report.summarized += 1
                    did_something = True
            if self.settings.facts:
                facts = extract_facts(text, document.title, self.cfg, self.complete,
                                      self.settings.max_facts)
                # None means nothing answered; [] means it answered "no facts
                # here", which is a finished document, not a failed one.
                if facts is not None:
                    did_something = True
                    if facts:
                        self.store.replace_facts(document_id, facts)
                        report.facts_added += len(facts)
        except Exception as e:
            logger.warning("Enriching document %s failed: %s", document_id, e)
            self.store.mark_enriched(document_id, ENRICH_FAILED)
            report.failed += 1
            return report

        if not did_something:
            # No brain reachable. Leave it pending so a later run can try.
            logger.debug("No brain available to enrich document %s; leaving it queued",
                         document_id)
            report.skipped += 1
            return report

        self.store.mark_enriched(document_id, ENRICH_DONE)
        report.documents += 1
        return report

    # -- batches -------------------------------------------------------

    def drain(self, limit: int | None = None, stop: threading.Event | None = None
              ) -> EnrichReport:
        """Work through the pending backlog now. Returns what it managed.

        Stops early if a pass makes no progress, so an unreachable brain
        doesn't spin the whole backlog.
        """
        report = EnrichReport()
        if not self.settings.enabled:
            return report
        done = 0
        while True:
            if stop is not None and stop.is_set():
                break
            want = self.settings.batch_size
            if limit is not None:
                if done >= limit:
                    break
                want = min(want, limit - done)
            batch = self.store.pending_enrichment(limit=want)
            if not batch:
                break
            before = report.documents + report.failed
            for document in batch:
                self.enrich_document(document.id, report)
                done += 1
            if report.documents + report.failed == before:
                # Nothing in that batch could be enriched; another pass over
                # the same documents would do the same thing.
                logger.info("Enrichment made no progress - stopping this pass")
                break
        return report

    # -- background thread ---------------------------------------------

    def enqueue(self, document_id: int) -> None:
        """Hand a document over without waiting for it. Never blocks ingest."""
        self._queue.put(document_id)

    def start(self) -> None:
        if self._thread is not None or not self.settings.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ev-enrich", daemon=True)
        self._thread.start()
        logger.info("Enrichment worker started (%s)", self.settings.describe())

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._queue.put(None)   # wake the worker so it sees the stop flag
        self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        # Anything left pending from a previous run comes first.
        try:
            self.drain(stop=self._stop)
        except Exception:
            logger.exception("Enrichment backlog pass failed")
        while not self._stop.is_set():
            try:
                document_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if document_id is None:
                break
            try:
                self.enrich_document(document_id)
            except Exception:
                logger.exception("Enriching document %s failed", document_id)
            finally:
                self._queue.task_done()

    def __enter__(self) -> "Enricher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
