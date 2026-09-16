"""Phase 9, part two: the ingest path everything writes through."""

from __future__ import annotations

import pytest

from ev_assistant.library import Library, _sniff_type
from ev_assistant.namespaces import CODE, NEWS, PERSONAL, REFERENCE
from ev_assistant.store import vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")

WATER = ("Water purification",
         "Boiling is the most reliable way to make water safe to drink. "
         "Bring water to a rolling boil and hold it for one full minute. "
         "Cloudy water should be filtered through cloth before boiling.")


@pytest.fixture
def library(cfg, store):
    from ev_assistant.embeddings import Embedder, HashingBackend

    cfg.reranker_backend = "lexical"
    return Library(cfg, store, embedder=Embedder(cfg, store, backend=HashingBackend(512)))


def test_sniff_type_from_the_source():
    assert _sniff_type("wikipedia:Water") == "wikipedia"
    assert _sniff_type("https://example.org/a") == "web"
    assert _sniff_type("/home/me/notes.pdf") == "pdf"
    assert _sniff_type("/home/me/notes.txt") == "file"
    assert _sniff_type("typed:note") == "note"


def test_adding_a_document_chunks_and_routes_it(library, store):
    outcome = library.add(*WATER[::-1][::-1], source="wikipedia:Water")

    assert outcome.added
    assert outcome.chunks >= 1
    document = store.get_document(outcome.document_id)
    assert document.namespace == REFERENCE
    assert document.source_type == "wikipedia"
    assert document.token_count > 0


@pytest.mark.parametrize("source,expected", [
    ("wikipedia:Water", REFERENCE),
    ("https://hnrss.org/frontpage", REFERENCE),
    ("/home/me/main.py", CODE),
    ("/home/me/diary.txt", PERSONAL),
])
def test_routing_happens_on_the_way_in(library, store, source, expected):
    outcome = library.add("T", "Some text about a thing worth storing.", source)
    assert store.get_document(outcome.document_id).namespace == expected


def test_an_explicit_namespace_wins(library, store):
    outcome = library.add("T", "Some text.", "wikipedia:X", namespace=NEWS)
    assert store.get_document(outcome.document_id).namespace == NEWS


def test_the_same_document_twice_is_skipped(library):
    first = library.add(WATER[0], WATER[1], "wikipedia:Water")
    second = library.add(WATER[0], WATER[1], "wikipedia:Water")

    assert first.added
    assert second.skipped
    assert not second.added


def test_force_replaces_it(library, store):
    library.add(WATER[0], WATER[1], "wikipedia:Water")
    forced = library.add(WATER[0], WATER[1], "wikipedia:Water", force=True)

    assert forced.added
    assert store.stats()["documents"] == 1


def test_empty_text_is_skipped_not_stored(library, store):
    assert library.add("Empty", "   ", "notes://empty").skipped
    assert store.stats()["documents"] == 0


@needs_vec
def test_indexing_follows_storing(library, store):
    library.add(WATER[0], WATER[1], "wikipedia:Water")
    assert store.stats()["pending_chunks"] >= 1

    indexed = library.index()

    assert indexed >= 1
    assert store.stats()["pending_chunks"] == 0
    assert store.stats()["embedded_chunks"] == indexed


def test_a_missing_embedding_model_does_not_fail_the_learn(cfg, store):
    class Broken:
        def index_pending(self, **kwargs):
            raise RuntimeError("no model on this machine")

    library = Library(cfg, store, embedder=Broken())
    outcome = library.add(WATER[0], WATER[1], "wikipedia:Water")

    assert outcome.added          # stored, and findable by keyword
    assert library.index() == 0   # just not vectorised yet
    assert store.stats()["pending_chunks"] >= 1


def test_enrichment_failing_does_not_fail_the_learn(cfg, store):
    class Broken:
        def drain(self, **kwargs):
            raise RuntimeError("no brain")

    library = Library(cfg, store, enricher=Broken())
    library.add(WATER[0], WATER[1], "wikipedia:Water")
    assert library.enrich() == 0


@needs_vec
def test_learn_stores_indexes_and_reports(cfg, store, library, monkeypatch):
    from ev_assistant import ingest, library as library_module

    pages = [
        ingest.Fetched(title="Water purification", text=WATER[1], source="wikipedia:Water"),
        ingest.Fetched(title="Knots", text="The bowline forms a fixed loop that will not slip.",
                       source="wikipedia:Knot"),
    ]
    monkeypatch.setattr("ev_assistant.ingest.crawl", lambda *a, **k: iter(pages))
    seen = []

    report = library.learn("Water purification", on_page=seen.append, enrich=False)

    assert report.pages == 2
    assert report.added == 2
    assert report.skipped == 0
    assert report.chunks >= 2
    assert report.embedded >= 2
    assert [o.title for o in seen] == ["Water purification", "Knots"]
    assert store.stats()["documents"] == 2


@needs_vec
def test_learned_documents_are_immediately_retrievable(cfg, store, library, monkeypatch):
    from ev_assistant import ingest
    from ev_assistant.rerank import LexicalReranker
    from ev_assistant.retrieval import Retriever

    monkeypatch.setattr("ev_assistant.ingest.crawl", lambda *a, **k: iter([
        ingest.Fetched(title=WATER[0], text=WATER[1], source="wikipedia:Water"),
    ]))
    library.learn("Water purification", enrich=False)

    retriever = Retriever(cfg, store, embedder=library.embedder, reranker=LexicalReranker(),
                          complete=lambda *a, **k: None)
    result = retriever.retrieve("how do I make water safe to drink", k=3)

    assert result.chunks
    assert result.chunks[0].document.title == "Water purification"


@needs_vec
def test_the_whole_path_from_learn_to_a_cited_answer(cfg, memory, store, library, monkeypatch):
    """learn -> store -> index -> retrieve -> context -> brain -> citation."""
    from ev_assistant import ingest
    from ev_assistant.brain import Brain
    from ev_assistant.rerank import LexicalReranker
    from ev_assistant.retrieval import Retriever
    import ev_assistant.brain as brain_module

    monkeypatch.setattr("ev_assistant.ingest.crawl", lambda *a, **k: iter([
        ingest.Fetched(title=WATER[0], text=WATER[1], source="wikipedia:Water"),
    ]))
    library.learn("Water purification", enrich=False)

    class Provider:
        name = "test"

        def __init__(self):
            self.context = ""

        def available(self, cfg_):
            return True

        def respond(self, cfg_, context, history, user_text, tools, executor):
            self.context = context
            return "Boil it for one full minute [S1]."

    provider = Provider()
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])
    retriever = Retriever(cfg, store, embedder=library.embedder, reranker=LexicalReranker(),
                          complete=lambda *a, **k: None)
    brain = Brain(cfg, memory, store=store, retriever=retriever)

    reply = brain.respond_detailed("how do I make water safe to drink")

    assert "rolling boil" in provider.context
    assert reply.text == "Boil it for one full minute [S1]."
    assert reply.spoken == "Boil it for one full minute."
    assert reply.citations[0]["label"] == "Wikipedia - Water purification"
    assert reply.retrieval["reason"] == "ok"
