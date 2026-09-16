"""Phase 9 of the retrieval spec: brain integration, citations, migration."""

from __future__ import annotations

import sqlite3

import pytest

from ev_assistant import namespaces
from ev_assistant.brain import Brain, BrainReply, cited_tags, strip_citations
from ev_assistant.chunking import chunk_rows
from ev_assistant.context import HEADER, NOTHING_RELEVANT
from ev_assistant.embeddings import Embedder, HashingBackend
from ev_assistant.knowledge import Knowledge
from ev_assistant.migrate import migrate, migrate_if_needed, read_passages
from ev_assistant.personality import build_persona_text
from ev_assistant.rerank import LexicalReranker
from ev_assistant.retrieval import Retriever
from ev_assistant.store import Store, vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")


class RecordingProvider:
    """A provider that captures the context it was handed."""

    name = "recording"

    def __init__(self, reply="Right then."):
        self.reply = reply
        self.contexts: list[str] = []

    def available(self, cfg):
        return True

    def respond(self, cfg, context, history, user_text, tools, executor):
        self.contexts.append(context)
        return self.reply


@pytest.fixture
def stocked_store(cfg, store):
    cfg.reranker_backend = "lexical"
    for title, uri, source_type, text in [
        ("Water purification", "wikipedia:Water", "wikipedia",
         "Boiling is the most reliable way to make water safe to drink. "
         "Bring water to a rolling boil and hold it for one full minute."),
        ("Shed notes", "notes://shed", "note",
         "The blue tarp is in the shed behind the mower."),
    ]:
        store.add_document(source_uri=uri, source_type=source_type, title=title, text=text,
                           namespace=namespaces.route(source_type, uri),
                           chunks=chunk_rows(text, title=title))
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()
    return store, embedder


def make_brain(cfg, memory, stocked_store, provider):
    store, embedder = stocked_store
    retriever = Retriever(cfg, store, embedder=embedder, reranker=LexicalReranker(),
                          complete=lambda *a, **k: None)
    brain = Brain(cfg, memory, store=store, retriever=retriever)
    import ev_assistant.brain as brain_module

    return brain, brain_module, provider


# -- citation handling -----------------------------------------------------


def test_strip_citations_leaves_clean_prose():
    assert strip_citations("Boil it for a minute [S1].") == "Boil it for a minute."
    assert strip_citations("Two sources agree [S1] [S2].") == "Two sources agree."
    assert strip_citations("[S1] Boiling works.") == "Boiling works."


def test_strip_citations_is_a_noop_without_tags():
    assert strip_citations("Just an answer.") == "Just an answer."
    assert strip_citations("") == ""


def test_strip_citations_does_not_eat_ordinary_brackets():
    assert strip_citations("Use the [red] valve.") == "Use the [red] valve."


def test_cited_tags_reports_what_was_actually_used():
    assert cited_tags("Facts [S2] and more [S1] and again [S2].") == ["[S2]", "[S1]"]
    assert cited_tags("nothing here") == []


def test_brain_reply_separates_text_from_speech():
    reply = BrainReply(text="Boil it [S1].", sources=[
        {"tag": "[S1]", "label": "Wikipedia - Water"},
        {"tag": "[S2]", "label": "Your notes - Shed"},
    ])
    assert reply.text == "Boil it [S1]."
    assert reply.spoken == "Boil it."
    # Citations are what the reply used, not everything it was offered.
    assert [c["tag"] for c in reply.citations] == ["[S1]"]


# -- the system prompt -----------------------------------------------------


def test_the_persona_explains_the_retrieved_block(cfg):
    persona = build_persona_text(cfg)

    assert "RETRIEVED FROM YOUR KNOWLEDGE BASE" in persona
    assert "[S1]" in persona
    assert "cite" in persona.lower()


def test_the_persona_sets_the_conflict_and_abstention_rules(cfg):
    persona = build_persona_text(cfg).lower()

    assert "trust the retrieved material" in persona
    assert "own knowledge" in persona
    assert "never imply you read something you didn't" in persona


def test_the_persona_distinguishes_speaking_from_writing(cfg):
    persona = build_persona_text(cfg).lower()
    assert "speaking aloud" in persona
    assert "leave the tag out" in persona


# -- context reaching the provider -----------------------------------------


@needs_vec
def test_the_retrieved_block_reaches_the_provider(cfg, memory, stocked_store, monkeypatch):
    provider = RecordingProvider("Boil it for a minute [S1].")
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    reply = brain.respond_detailed("how do I make water safe to drink")

    context = provider.contexts[0]
    assert HEADER in context
    assert "[S1] Wikipedia - Water purification" in context
    assert reply.sources
    assert reply.citations[0]["label"] == "Wikipedia - Water purification"


@needs_vec
def test_an_uncovered_question_tells_the_brain_the_store_had_nothing(
    cfg, memory, stocked_store, monkeypatch
):
    provider = RecordingProvider("No idea, and not from your notes.")
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    reply = brain.respond_detailed("what is the capital city of Mongolia")

    assert NOTHING_RELEVANT in provider.contexts[0]
    assert reply.sources == []
    assert reply.retrieval["reason"] == "below_threshold"


@needs_vec
def test_a_question_that_needs_no_store_gets_no_block(cfg, memory, stocked_store, monkeypatch):
    provider = RecordingProvider("No worries.")
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    reply = brain.respond_detailed("thanks!")

    assert HEADER not in provider.contexts[0]
    assert NOTHING_RELEVANT not in provider.contexts[0]
    assert reply.retrieval["reason"] == "skipped"


@needs_vec
def test_memory_facts_still_reach_the_brain(cfg, memory, stocked_store, monkeypatch):
    memory.add_fact("weather", "It is raining in Melbourne", external_id="w1")
    provider = RecordingProvider()
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    brain.respond_detailed("what is the weather doing in Melbourne")

    assert "Melbourne" in provider.contexts[0]


@needs_vec
def test_the_trace_explains_the_decision(cfg, memory, stocked_store, monkeypatch):
    provider = RecordingProvider()
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    trace = brain.respond_detailed("how do I make water safe to drink").retrieval

    assert trace["reason"] == "ok"
    assert trace["chunks"] >= 1
    assert trace["reranker"]
    assert trace["total_ms"] >= 0
    assert trace["context_tokens"] > 0
    assert any(v["kind"] == "question" for v in trace["variants"])


@needs_vec
def test_the_exchange_is_recorded_in_memory(cfg, memory, stocked_store, monkeypatch):
    provider = RecordingProvider("Boil it [S1].")
    brain, brain_module, _ = make_brain(cfg, memory, stocked_store, provider)
    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    brain.respond_detailed("how do I make water safe to drink")

    turns = memory.recent_turns(limit=4)
    assert turns[-2]["role"] == "user"
    assert turns[-1]["content"] == "Boil it [S1]."


def test_retrieval_failing_does_not_cost_the_user_an_answer(cfg, memory, monkeypatch):
    class Exploding:
        def retrieve(self, *a, **k):
            raise RuntimeError("the store caught fire")

    provider = RecordingProvider("Here is an answer anyway.")
    brain = Brain(cfg, memory, retriever=Exploding(), store=object())
    import ev_assistant.brain as brain_module

    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])

    reply = brain.respond_detailed("how do I purify water")

    assert reply.text == "Here is an answer anyway."
    assert reply.retrieval == {"error": "retrieval failed"}


def test_respond_still_returns_a_plain_string(cfg, memory, monkeypatch):
    provider = RecordingProvider("Plain reply.")
    brain = Brain(cfg, memory, retriever=_NullRetriever(), store=object())
    import ev_assistant.brain as brain_module

    monkeypatch.setattr(brain_module, "build_chain", lambda c: [provider])
    assert brain.respond("anything at all") == "Plain reply."


class _NullRetriever:
    def retrieve(self, *a, **k):
        from ev_assistant.retrieval import SKIPPED, RetrievalResult

        return RetrievalResult(reason=SKIPPED)


# -- migrating an existing knowledge base ----------------------------------


def test_reads_passages_from_an_old_knowledge_base(tmp_path):
    knowledge = Knowledge(tmp_path / "knowledge.sqlite3")
    knowledge.add_document("Water", "wikipedia:Water", "Boil the water for one minute.")
    knowledge.add_document("Shed", "notes://shed", "The blue tarp is behind the mower.")

    grouped = read_passages(tmp_path / "knowledge.sqlite3")

    assert ("Water", "wikipedia:Water") in grouped
    assert ("Shed", "notes://shed") in grouped
    assert "Boil the water" in grouped[("Water", "wikipedia:Water")][0]


def test_reading_a_missing_or_foreign_database_is_harmless(tmp_path):
    assert read_passages(tmp_path / "nope.sqlite3") == {}

    other = tmp_path / "other.sqlite3"
    sqlite3.connect(other).execute("CREATE TABLE unrelated (x INT)")
    assert read_passages(other) == {}


def test_migration_carries_documents_across(cfg, tmp_path):
    knowledge = Knowledge(cfg.knowledge_path)
    long_text = " ".join(f"Sentence number {i} about purifying water." for i in range(80))
    knowledge.add_document("Water", "wikipedia:Water", long_text)
    knowledge.add_document("Shed", "notes://shed", "The blue tarp is behind the mower.")
    store = Store(cfg.store_path)

    report = migrate(cfg.knowledge_path, store)

    assert report.documents == 2
    assert report.chunks >= 2
    assert bool(report)
    stats = store.stats()
    assert stats["documents"] == 2
    assert stats["chunks"] == report.chunks
    # Routing is applied on the way in, not left as the default.
    assert stats["by_namespace"].get("personal") == 1
    assert stats["by_namespace"].get("reference") == 1


def test_migration_is_idempotent(cfg):
    knowledge = Knowledge(cfg.knowledge_path)
    knowledge.add_document("Water", "wikipedia:Water", "Boil the water for one minute.")
    store = Store(cfg.store_path)

    first = migrate(cfg.knowledge_path, store)
    second = migrate(cfg.knowledge_path, store)

    assert first.documents == 1
    assert second.documents == 0
    assert second.skipped == 1
    assert store.stats()["documents"] == 1


def test_migrate_if_needed_runs_once_and_only_on_an_empty_store(cfg):
    knowledge = Knowledge(cfg.knowledge_path)
    knowledge.add_document("Water", "wikipedia:Water", "Boil the water for one minute.")
    store = Store(cfg.store_path)

    assert migrate_if_needed(cfg, store).documents == 1
    assert store.get_meta("legacy_migrated") == "1"
    # A second call does nothing, even after the marker is cleared, because
    # the store is no longer empty.
    assert migrate_if_needed(cfg, store).documents == 0


def test_migrate_if_needed_skips_a_store_that_already_has_documents(cfg, store):
    knowledge = Knowledge(cfg.knowledge_path)
    knowledge.add_document("Water", "wikipedia:Water", "Boil the water for one minute.")
    store.add_document(source_uri="notes://x", source_type="note", title="X",
                       text="already here", chunks=[{"text": "already here", "ordinal": 0}])

    assert migrate_if_needed(cfg, store).documents == 0
    assert store.stats()["documents"] == 1


def test_migration_with_nothing_to_migrate(cfg):
    store = Store(cfg.store_path)
    assert migrate(cfg.knowledge_path, store).documents == 0


@needs_vec
def test_migrated_documents_are_retrievable(cfg):
    knowledge = Knowledge(cfg.knowledge_path)
    knowledge.add_document(
        "Water purification", "wikipedia:Water",
        "Boiling is the most reliable way to make water safe to drink. "
        "Bring it to a rolling boil and hold it for one full minute.")
    store = Store(cfg.store_path)
    migrate(cfg.knowledge_path, store)

    cfg.reranker_backend = "lexical"
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    assert embedder.index_pending().embedded >= 1

    retriever = Retriever(cfg, store, embedder=embedder, reranker=LexicalReranker(),
                          complete=lambda *a, **k: None)
    result = retriever.retrieve("how do I make water safe to drink", k=3)

    assert result.chunks
    assert result.chunks[0].document.title == "Water purification"
