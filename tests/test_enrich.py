"""Phase 6 of the retrieval spec: ingest-time summaries and atomic facts.

No network and no model: every test drives enrichment with a fake `complete`
so the parsing, bookkeeping and threading are exercised deterministically.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from ev_assistant.enrich import (
    EnrichmentSettings,
    Enricher,
    extract_facts,
    parse_facts,
    summarize,
)
from ev_assistant.store import ENRICH_DONE, ENRICH_FAILED, ENRICH_PENDING, Store

SUMMARY = "This document explains water purification. It covers boiling and filtering."
FACTS_JSON = json.dumps([
    {"statement": "Boiling water for one minute makes it safe to drink.",
     "subject": "water purification", "confidence": 0.95},
    {"statement": "Cloudy water should be filtered before boiling.",
     "subject": "water purification", "confidence": 0.8},
])


def scripted(*answers):
    """A `complete` that returns each answer in turn, then repeats the last."""
    calls = []

    def complete(cfg, system, user, max_tokens=256):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        index = min(len(calls) - 1, len(answers) - 1)
        return answers[index]

    complete.calls = calls
    return complete


def by_system(summary=None, facts=None):
    """A `complete` that answers based on which prompt it was given."""
    calls = []

    def complete(cfg, system, user, max_tokens=256):
        calls.append(system)
        return summary if "Summarise" in system else facts

    complete.calls = calls
    return complete


def add_doc(store: Store, *, uri="notes://a", title="Water", text="Boil the water.") -> int:
    return store.add_document(
        source_uri=uri, source_type="note", title=title, text=text,
        chunks=[{"text": text, "ordinal": 0}],
    )


# -- fact parsing ----------------------------------------------------------


def test_parses_a_clean_json_array():
    facts = parse_facts(FACTS_JSON)
    assert len(facts) == 2
    assert facts[0]["statement"].startswith("Boiling water")
    assert facts[0]["subject"] == "water purification"
    assert facts[0]["confidence"] == 0.95


def test_parses_through_a_code_fence():
    assert len(parse_facts(f"```json\n{FACTS_JSON}\n```")) == 2


def test_parses_through_surrounding_prose():
    reply = f"Sure! Here are the facts I found:\n\n{FACTS_JSON}\n\nHope that helps."
    assert len(parse_facts(reply)) == 2


def test_parses_a_wrapped_object():
    assert len(parse_facts(json.dumps({"facts": json.loads(FACTS_JSON)}))) == 2


def test_parses_a_bare_list_of_strings():
    facts = parse_facts(json.dumps(["Water boils at 100 degrees at sea level."]))
    assert facts[0]["statement"].startswith("Water boils")
    assert facts[0]["subject"] == ""
    assert 0.0 <= facts[0]["confidence"] <= 1.0


def test_an_empty_array_is_a_valid_answer():
    assert parse_facts("[]") == []
    assert parse_facts("```json\n[]\n```") == []


def test_garbage_is_dropped_rather_than_stored():
    # A garbled fact is worse than a missing one - it gets injected as truth.
    assert parse_facts("") == []
    assert parse_facts("I'm not sure what you want.") == []
    assert parse_facts("[{unclosed") == []
    assert parse_facts(json.dumps([{"statement": "short"}])) == []
    assert parse_facts(json.dumps([{"nothing": "useful"}])) == []
    assert parse_facts(json.dumps(["x" * 900])) == []


def test_confidence_is_clamped_and_defaulted():
    facts = parse_facts(json.dumps([
        {"statement": "A statement that is long enough.", "confidence": 5},
        {"statement": "Another statement long enough.", "confidence": "nonsense"},
        {"statement": "A third statement long enough.", "confidence": -2},
    ]))
    assert [f["confidence"] for f in facts] == [1.0, 0.7, 0.0]


def test_duplicate_statements_are_collapsed():
    facts = parse_facts(json.dumps([
        {"statement": "Boiling water kills pathogens."},
        {"statement": "boiling water kills pathogens."},
    ]))
    assert len(facts) == 1


def test_facts_are_capped():
    many = [{"statement": f"Statement number {i} is long enough."} for i in range(50)]
    assert len(parse_facts(json.dumps(many), max_facts=5)) == 5


# -- the two calls ---------------------------------------------------------


def test_summarize_returns_the_model_text():
    complete = scripted(SUMMARY)
    assert summarize("some text", "Water", cfg=object(), complete=complete) == SUMMARY
    assert "Water" in complete.calls[0]["user"]
    assert "Summarise" in complete.calls[0]["system"]


def test_summarize_without_a_brain_or_text():
    assert summarize("some text") == ""
    assert summarize("", "T", cfg=object(), complete=scripted(SUMMARY)) == ""
    assert summarize("text", "T", cfg=object(), complete=scripted(None)) == ""


def test_extract_facts_distinguishes_no_brain_from_no_facts():
    # None means "try again later"; [] means "this document is finished".
    assert extract_facts("text", "T", cfg=object(), complete=scripted(None)) is None
    assert extract_facts("text") is None
    assert extract_facts("text", "T", cfg=object(), complete=scripted("[]")) == []


# -- settings --------------------------------------------------------------


def test_settings_read_from_config(cfg):
    cfg.enrich_summaries = False
    cfg.enrich_facts = True
    cfg.max_facts_per_document = 3
    settings = EnrichmentSettings(cfg)
    assert settings.enabled
    assert settings.describe() == "facts"
    assert settings.max_facts == 3

    cfg.enrich_facts = False
    off = EnrichmentSettings(cfg)
    assert not off.enabled
    assert "disabled" in off.describe()


# -- one document ----------------------------------------------------------


def test_enriching_a_document_stores_a_summary_and_facts(cfg, store):
    doc_id = add_doc(store)
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, FACTS_JSON))

    report = enricher.enrich_document(doc_id)

    assert report.documents == 1
    assert report.summarized == 1
    assert report.facts_added == 2
    assert store.get_document(doc_id).summary == SUMMARY
    assert len(store.facts_for(doc_id)) == 2
    assert store.get_document(doc_id).enrichment_status == ENRICH_DONE


def test_a_document_with_no_facts_is_still_finished(cfg, store):
    doc_id = add_doc(store)
    Enricher(cfg, store, complete=by_system(SUMMARY, "[]")).enrich_document(doc_id)

    assert store.get_document(doc_id).enrichment_status == ENRICH_DONE
    assert store.facts_for(doc_id) == []


def test_without_a_brain_the_document_stays_queued(cfg, store):
    doc_id = add_doc(store)
    report = Enricher(cfg, store, complete=scripted(None)).enrich_document(doc_id)

    assert report.skipped == 1
    assert report.documents == 0
    assert store.get_document(doc_id).enrichment_status == ENRICH_PENDING


def test_re_enrichment_replaces_facts_instead_of_doubling_them(cfg, store):
    doc_id = add_doc(store)
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, FACTS_JSON))
    enricher.enrich_document(doc_id)
    enricher.enrich_document(doc_id)

    assert len(store.facts_for(doc_id)) == 2


def test_a_raising_store_marks_the_document_failed(cfg, store, monkeypatch):
    doc_id = add_doc(store)
    monkeypatch.setattr(store, "set_summary",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    report = Enricher(cfg, store, complete=by_system(SUMMARY, FACTS_JSON)).enrich_document(doc_id)

    assert report.failed == 1
    assert store.get_document(doc_id).enrichment_status == ENRICH_FAILED


def test_a_missing_document_is_skipped(cfg, store):
    assert Enricher(cfg, store, complete=scripted(SUMMARY)).enrich_document(999).skipped == 1


def test_disabled_enrichment_marks_documents_done_not_pending_forever(cfg, store):
    cfg.enrich_summaries = False
    cfg.enrich_facts = False
    doc_id = add_doc(store)

    Enricher(cfg, store, complete=scripted(SUMMARY)).enrich_document(doc_id)

    assert store.get_document(doc_id).enrichment_status == ENRICH_DONE
    assert store.get_document(doc_id).summary == ""


def test_each_half_can_be_switched_off_independently(cfg, store):
    cfg.enrich_facts = False
    doc_id = add_doc(store)
    complete = by_system(SUMMARY, FACTS_JSON)
    Enricher(cfg, store, complete=complete).enrich_document(doc_id)

    assert store.get_document(doc_id).summary == SUMMARY
    assert store.facts_for(doc_id) == []
    assert all("Summarise" in system for system in complete.calls)  # no facts call made


def test_long_documents_are_truncated_at_a_paragraph_break(cfg, store):
    cfg.enrich_chars = 400
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(20))
    doc_id = store.add_document(
        source_uri="notes://long", source_type="note", title="Long", text=text,
        chunks=[{"text": p, "ordinal": i} for i, p in enumerate(text.split("\n\n"))],
    )
    sent = []

    def capture(cfg_, system, user, max_tokens=256):
        sent.append(user)
        return SUMMARY if "Summarise" in system else "[]"

    Enricher(cfg, store, complete=capture).enrich_document(doc_id)

    shown = sent[0]
    assert len(shown) <= 400 + len("Title: Long\n\n")
    assert "Paragraph 19" not in shown   # it really was cut short
    assert shown.endswith("word")        # and never mid-word


def test_truncation_falls_back_to_a_word_boundary(cfg, store):
    # One paragraph longer than the whole budget: there is no break to use.
    one_long_paragraph = "supercalifragilistic " * 200
    doc_id = store.add_document(
        source_uri="notes://one", source_type="note", title="One", text=one_long_paragraph,
        chunks=[{"text": one_long_paragraph, "ordinal": 0}],
    )
    shown = store.document_text(doc_id, 300)

    assert len(shown) <= 300
    assert shown.endswith("supercalifragilistic")   # not "supercalifrag"


# -- batches ---------------------------------------------------------------


def test_drain_works_through_the_backlog(cfg, store):
    for i in range(5):
        add_doc(store, uri=f"notes://{i}", text=f"Document number {i} about water.")
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, "[]"))

    report = enricher.drain()

    assert report.documents == 5
    assert store.pending_enrichment() == []
    assert store.stats()["enriched_documents"] == 5
    assert store.stats()["pending_enrichment"] == 0


def test_drain_respects_a_limit(cfg, store):
    for i in range(6):
        add_doc(store, uri=f"notes://{i}", text=f"Document number {i} about water.")
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, "[]"))

    assert enricher.drain(limit=2).documents == 2
    assert len(store.pending_enrichment(limit=10)) == 4


def test_drain_stops_instead_of_spinning_when_nothing_can_be_enriched(cfg, store):
    for i in range(20):
        add_doc(store, uri=f"notes://{i}", text=f"Document number {i} about water.")
    complete = scripted(None)
    enricher = Enricher(cfg, store, complete=complete)

    report = enricher.drain()

    assert report.documents == 0
    # One batch attempted, then it gave up rather than grinding the backlog.
    assert len(complete.calls) <= 2 * enricher.settings.batch_size
    assert len(store.pending_enrichment(limit=50)) == 20


def test_drain_is_a_noop_when_disabled(cfg, store):
    cfg.enrich_summaries = cfg.enrich_facts = False
    add_doc(store)
    complete = scripted(SUMMARY)
    assert Enricher(cfg, store, complete=complete).drain().documents == 0
    assert complete.calls == []


def test_drain_can_be_stopped(cfg, store):
    for i in range(20):
        add_doc(store, uri=f"notes://{i}", text=f"Document number {i} about water.")
    stop = threading.Event()
    stop.set()
    assert Enricher(cfg, store, complete=by_system(SUMMARY, "[]")).drain(stop=stop).documents == 0


def test_reset_enrichment_requeues(cfg, store):
    add_doc(store, uri="notes://a", text="First document about water.")
    add_doc(store, uri="notes://b", text="Second document about knots.", title="Knots")
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, "[]"))
    enricher.drain()

    assert store.reset_enrichment() == 2
    assert len(store.pending_enrichment()) == 2


# -- the background worker -------------------------------------------------


def test_enqueue_does_not_block_ingest(cfg, store):
    """The whole point: handing over a document must be instant."""
    doc_ids = [add_doc(store, uri=f"notes://{i}", text=f"Document {i} about water.")
               for i in range(10)]

    def slow(cfg_, system, user, max_tokens=256):
        time.sleep(0.05)
        return SUMMARY if "Summarise" in system else "[]"

    enricher = Enricher(cfg, store, complete=slow)
    enricher.settings.facts = False

    started = time.monotonic()
    with enricher:
        for doc_id in doc_ids:
            enricher.enqueue(doc_id)
        handover = time.monotonic() - started
        # 10 documents at 50ms each is half a second of model time; handing
        # them over must not cost anything like that.
        assert handover < 0.1

        deadline = time.monotonic() + 10
        while store.stats()["pending_enrichment"] and time.monotonic() < deadline:
            time.sleep(0.05)

    assert store.stats()["pending_enrichment"] == 0
    assert store.get_document(doc_ids[0]).summary == SUMMARY


def test_the_worker_picks_up_a_backlog_on_start(cfg, store):
    doc_id = add_doc(store)
    enricher = Enricher(cfg, store, complete=by_system(SUMMARY, "[]"))

    with enricher:
        deadline = time.monotonic() + 10
        while store.stats()["pending_enrichment"] and time.monotonic() < deadline:
            time.sleep(0.05)

    assert store.get_document(doc_id).summary == SUMMARY


def test_stopping_a_worker_that_never_started_is_fine(cfg, store):
    enricher = Enricher(cfg, store, complete=scripted(SUMMARY))
    enricher.stop()
    enricher.stop()


def test_a_disabled_enricher_starts_no_thread(cfg, store):
    cfg.enrich_summaries = cfg.enrich_facts = False
    enricher = Enricher(cfg, store, complete=scripted(SUMMARY))
    enricher.start()
    assert enricher._thread is None


def test_a_raising_document_does_not_kill_the_worker(cfg, store):
    good = add_doc(store, uri="notes://good", text="A good document about water.")

    seen = []

    def flaky(cfg_, system, user, max_tokens=256):
        seen.append(user)
        if "explode" in user:
            raise RuntimeError("boom")
        return SUMMARY if "Summarise" in system else "[]"

    bad = add_doc(store, uri="notes://bad", text="This one will explode on purpose.")
    enricher = Enricher(cfg, store, complete=flaky)

    with enricher:
        enricher.enqueue(bad)
        enricher.enqueue(good)
        deadline = time.monotonic() + 10
        while store.get_document(good).summary == "" and time.monotonic() < deadline:
            time.sleep(0.05)

    assert store.get_document(good).summary == SUMMARY
    assert store.get_document(bad).enrichment_status == ENRICH_FAILED


# -- schema migration ------------------------------------------------------


def test_a_v1_store_gains_the_enrichment_column(tmp_path):
    """An existing store must survive the upgrade with its data intact."""
    import sqlite3

    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, source_uri TEXT NOT NULL, source_type TEXT NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'reference', title TEXT NOT NULL DEFAULT '',
            content_hash TEXT UNIQUE NOT NULL, fetched_at REAL NOT NULL, published_at REAL,
            license TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
            token_count INTEGER NOT NULL DEFAULT 0);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO documents (source_uri, source_type, title, content_hash, fetched_at)
            VALUES ('notes://old', 'note', 'Older doc', 'abc123', 1700000000.0);
    """)
    conn.commit()
    conn.close()

    store = Store(path)

    assert store.get_meta("schema_version") == "2"
    document = store.get_document(1)
    assert document.title == "Older doc"
    assert document.enrichment_status == ENRICH_PENDING
    assert len(store.pending_enrichment()) == 1


def test_migration_runs_twice_without_complaint(tmp_path):
    path = tmp_path / "s.sqlite3"
    first = Store(path)
    doc_id = add_doc(first)
    first.mark_enriched(doc_id)

    second = Store(path)
    assert second.get_document(doc_id).enrichment_status == ENRICH_DONE
    assert second.pending_enrichment() == []
