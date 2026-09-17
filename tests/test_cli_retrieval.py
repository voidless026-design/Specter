"""Phase 11 of the retrieval spec: the CLI surface and observability."""

from __future__ import annotations

import argparse
import json

import pytest

from ev_assistant import probe
from ev_assistant.chunking import chunk_rows
from ev_assistant.cli import (
    _fmt_size,
    build_parser,
    cmd_reindex,
    cmd_retrieve,
    cmd_search,
    cmd_stats,
)
from ev_assistant.embeddings import Embedder, HashingBackend
from ev_assistant.store import Store, vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")

DOCS = [
    ("Knots", "notes://knots",
     "The bowline forms a fixed loop at the end of a rope that will not slip "
     "under load. The clove hitch secures a rope to a post."),
    ("Water", "notes://water",
     "Boiling is the most reliable way to make water safe to drink. "
     "Hold a rolling boil for one full minute."),
]


@pytest.fixture
def stocked(cfg, monkeypatch):
    """A real store behind the CLI's load_config, wired to a hashing embedder."""
    cfg.reranker_backend = "lexical"
    store = Store(cfg.store_path)
    for title, uri, text in DOCS:
        store.add_document(source_uri=uri, source_type="note", title=title, text=text,
                           chunks=chunk_rows(text, title=title))
    Embedder(cfg, store, backend=HashingBackend(512)).index_pending()

    monkeypatch.setattr("ev_assistant.cli.load_config", lambda: cfg)
    monkeypatch.setattr("ev_assistant.embeddings.build_backend",
                        lambda c: HashingBackend(512))
    return store


def args(**kwargs) -> argparse.Namespace:
    defaults = dict(query="", limit=8, namespace=None, floor=None, width=150, json=False,
                    explain=False, all=False, model_changed=False, enrich=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# -- argument parsing ------------------------------------------------------


@pytest.mark.parametrize("argv,command", [
    (["search", "water"], "cmd_search"),
    (["retrieve", "water", "--explain"], "cmd_retrieve"),
    (["retrieve", "water", "--json"], "cmd_retrieve"),
    (["reindex"], "cmd_reindex"),
    (["reindex", "--all", "--namespace", "personal"], "cmd_reindex"),
    (["reindex", "--model-changed", "--enrich"], "cmd_reindex"),
    (["stats"], "cmd_stats"),
    (["eval", "--floor", "0.3"], "cmd_eval"),
])
def test_every_new_command_parses(argv, command):
    parsed = build_parser().parse_args(argv)
    assert parsed.func.__name__ == command


def test_search_options():
    parsed = build_parser().parse_args(["search", "q", "-n", "3", "--namespace", "code",
                                        "--floor", "0.4"])
    assert parsed.limit == 3
    assert parsed.namespace == "code"
    assert parsed.floor == 0.4


def test_fmt_size():
    assert _fmt_size(500) == "500 B"
    assert _fmt_size(2048) == "2.0 KB"
    assert _fmt_size(5 * 1024 * 1024) == "5.0 MB"


# -- ev search -------------------------------------------------------------


@needs_vec
def test_search_prints_ranked_results(stocked, capsys):
    cmd_search(args(query="which knot makes a fixed loop", limit=3))

    out = capsys.readouterr().out
    assert "Knots" in out
    assert "[1]" in out
    assert "result(s) in" in out
    assert "floor" in out


@needs_vec
def test_search_says_nothing_clearly(stocked, capsys):
    cmd_search(args(query="what is the capital city of Peru"))

    out = capsys.readouterr().out
    assert "Nothing relevant" in out
    assert "below_threshold" in out
    # And explains that this is a real answer, not a malfunction.
    assert "empty result is a real answer" in out


@needs_vec
def test_search_explains_a_skipped_question(stocked, capsys):
    cmd_search(args(query="thanks!"))
    assert "doesn't need the knowledge base" in capsys.readouterr().out


@needs_vec
def test_search_on_an_empty_store(cfg, monkeypatch, capsys):
    monkeypatch.setattr("ev_assistant.cli.load_config", lambda: cfg)
    Store(cfg.store_path)
    cmd_search(args(query="anything at all"))
    assert "empty" in capsys.readouterr().out.lower()


@needs_vec
def test_search_floor_override_changes_the_outcome(stocked, capsys):
    cmd_search(args(query="which knot makes a fixed loop", floor=0.99))
    assert "Nothing relevant" in capsys.readouterr().out

    cmd_search(args(query="which knot makes a fixed loop", floor=0.01))
    assert "Knots" in capsys.readouterr().out


# -- ev retrieve --explain -------------------------------------------------


@needs_vec
def test_retrieve_explains_every_stage(stocked, capsys):
    cmd_retrieve(args(query="which knot makes a fixed loop", limit=4))

    out = capsys.readouterr().out
    for section in ("Question:", "Retrieve:", "Namespaces:", "Variants:",
                    "Candidates:", "Reranker:", "Timings:"):
        assert section in out, f"missing {section}"
    # The per-candidate table, with each stage's score.
    for column in ("rrf", "dense", "sparse", "rerank", "score", "channels"):
        assert column in out
    assert "cleared the relevance floor" in out


@needs_vec
def test_retrieve_shows_why_a_question_was_skipped(stocked, capsys):
    cmd_retrieve(args(query="thanks!"))
    out = capsys.readouterr().out
    assert "Retrieve:  False" in out
    assert "chitchat" in out


@needs_vec
def test_retrieve_json_is_machine_readable(stocked, capsys):
    cmd_retrieve(args(query="which knot makes a fixed loop", json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "ok"
    assert payload["trace"]["floor"] > 0
    assert payload["trace"]["reranker"]
    candidate = payload["candidates"][0]
    assert {"chunk_id", "rrf", "dense", "sparse", "rerank", "score", "survived",
            "channels", "title"} <= set(candidate)
    assert isinstance(candidate["survived"], bool)


# -- ev reindex ------------------------------------------------------------


@needs_vec
def test_reindex_with_nothing_pending_says_so(stocked, capsys):
    cmd_reindex(args())
    assert "Nothing to index" in capsys.readouterr().out


@needs_vec
def test_reindex_all_rebuilds_everything(stocked, capsys):
    before = stocked.stats()["embedded_chunks"]
    assert before > 0

    cmd_reindex(args(all=True))

    out = capsys.readouterr().out
    assert f"Requeued {before}" in out
    assert "Indexed" in out
    assert stocked.stats()["embedded_chunks"] == before
    assert stocked.stats()["pending_chunks"] == 0


@needs_vec
def test_reindex_can_target_one_namespace(stocked, capsys):
    cmd_reindex(args(namespace="reference"))
    out = capsys.readouterr().out
    assert "namespace reference" in out


@needs_vec
def test_reindex_model_changed_drops_the_vector_index(stocked, capsys):
    stocked.set_meta("embedding_model", "st:some/other-model")

    cmd_reindex(args(all=True, model_changed=True))

    out = capsys.readouterr().out
    assert "Dropped the vector index" in out
    assert stocked.get_meta("embedding_model") == "hashing:512"


@needs_vec
def test_reindex_reports_a_model_mismatch_instead_of_crashing(stocked, capsys, monkeypatch):
    # A store built with one model, config now naming another, vectors present.
    stocked.set_meta("embedding_model", "st:some/other-model")
    monkeypatch.setattr("ev_assistant.embeddings.build_backend",
                        lambda c: HashingBackend(512))

    with pytest.raises(SystemExit) as exit_info:
        cmd_reindex(args())

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "can't share an index" in err
    assert "--model-changed" in err


@needs_vec
def test_reindex_enrich_requeues_documents(stocked, capsys, monkeypatch):
    monkeypatch.setattr("ev_assistant.enrich.Enricher.drain",
                        lambda self, **kw: __import__("ev_assistant.enrich",
                                                     fromlist=["EnrichReport"]).EnrichReport())
    cmd_reindex(args(enrich=True))
    out = capsys.readouterr().out
    assert "Requeued" in out and "summaries and facts" in out


# -- ev stats --------------------------------------------------------------


@needs_vec
def test_stats_reports_the_store(stocked, capsys):
    cmd_stats(args())

    out = capsys.readouterr().out
    assert "2 documents" in out
    assert "Embedding coverage" in out
    assert "100%" in out
    assert "By namespace" in out
    assert "By source type" in out
    assert "vector search: available" in out


@needs_vec
def test_stats_flags_unembedded_chunks(stocked, capsys):
    stocked.reset_embeddings()
    cmd_stats(args())

    out = capsys.readouterr().out
    assert "waiting - run `ev reindex`" in out


@needs_vec
def test_stats_flags_pending_enrichment(stocked, capsys):
    cmd_stats(args())
    assert "reindex --enrich" in capsys.readouterr().out


def test_stats_on_an_empty_store(cfg, monkeypatch, capsys):
    monkeypatch.setattr("ev_assistant.cli.load_config", lambda: cfg)
    Store(cfg.store_path)
    cmd_stats(args())

    out = capsys.readouterr().out
    assert "0 documents" in out


# -- the probe cache -------------------------------------------------------


def test_a_failure_is_remembered_then_expires(cfg):
    assert probe.recently_failed(cfg, "st:some/model") is False

    probe.remember_failure(cfg, "st:some/model")

    assert probe.recently_failed(cfg, "st:some/model") is True
    assert probe.recently_failed(cfg, "st:other/model") is False
    # A short TTL means an old failure stops counting.
    assert probe.recently_failed(cfg, "st:some/model", ttl=0.0) is False


def test_forget_clears_one_or_all(cfg):
    probe.remember_failure(cfg, "a")
    probe.remember_failure(cfg, "b")

    probe.forget(cfg, "a")
    assert not probe.recently_failed(cfg, "a")
    assert probe.recently_failed(cfg, "b")

    probe.forget(cfg)
    assert not probe.recently_failed(cfg, "b")


def test_a_corrupt_probe_cache_is_ignored_not_fatal(cfg):
    (cfg.data_dir / probe.FILENAME).write_text("{not json", encoding="utf-8")
    assert probe.recently_failed(cfg, "anything") is False
    probe.remember_failure(cfg, "anything")   # and it repairs itself
    assert probe.recently_failed(cfg, "anything") is True


def test_the_probe_cache_skips_a_recently_failed_backend(cfg, monkeypatch):
    from ev_assistant import embeddings

    attempts = []

    def counting_available(self):
        attempts.append(self.name)
        return False

    monkeypatch.setattr(embeddings.SentenceTransformerBackend, "available", counting_available)
    monkeypatch.setattr(embeddings.OpenAIEmbeddingBackend, "available", lambda self: False)
    cfg.embedding_backend = "auto"

    embeddings.build_backend(cfg)
    first = len(attempts)
    embeddings.build_backend(cfg)

    assert first == 1
    assert len(attempts) == 1, "the failed backend was probed again"


def test_the_probe_cache_also_guards_the_reranker(cfg, monkeypatch):
    from ev_assistant import rerank

    attempts = []
    monkeypatch.setattr(rerank.CrossEncoderReranker, "available",
                        lambda self: attempts.append(self.name) or False)
    monkeypatch.setattr(rerank.CohereReranker, "available", lambda self: False)
    cfg.reranker_backend = "auto"

    rerank.build_reranker(cfg)
    rerank.build_reranker(cfg)

    assert len(attempts) == 1


def test_a_config_without_a_data_dir_degrades_quietly():
    class Bare:
        pass

    bare = Bare()
    assert probe.recently_failed(bare, "x") is False
    probe.remember_failure(bare, "x")     # must not raise
    probe.forget(bare)


# -- observability ---------------------------------------------------------


@needs_vec
def test_retrieval_logs_per_stage_timings_at_debug(cfg, stocked, caplog):
    from ev_assistant.rerank import LexicalReranker
    from ev_assistant.retrieval import Retriever

    retriever = Retriever(cfg, stocked, embedder=Embedder(cfg, stocked, HashingBackend(512)),
                          reranker=LexicalReranker(), complete=lambda *a, **k: None)

    with caplog.at_level("DEBUG", logger="ev_assistant.retrieval"):
        retriever.retrieve("which knot makes a fixed loop", k=3)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "search=" in logged and "rerank=" in logged
    assert "total=" in logged


@needs_vec
def test_an_abstention_is_logged_with_the_reason(cfg, stocked, caplog):
    from ev_assistant.rerank import LexicalReranker
    from ev_assistant.retrieval import Retriever

    retriever = Retriever(cfg, stocked, embedder=Embedder(cfg, stocked, HashingBackend(512)),
                          reranker=LexicalReranker(), complete=lambda *a, **k: None)

    with caplog.at_level("DEBUG", logger="ev_assistant.retrieval"):
        retriever.retrieve("what is the capital city of Peru", k=3)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "nothing cleared the floor" in logged
