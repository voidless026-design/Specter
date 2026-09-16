"""Phase 7, part two: the hybrid retrieval pipeline.

Runs on the hashing embedder and a scripted reranker so the scores are
deterministic and no model is downloaded.
"""

from __future__ import annotations

import calendar

import pytest

from ev_assistant import namespaces
from ev_assistant.chunking import chunk_rows
from ev_assistant.embeddings import Embedder, HashingBackend
from ev_assistant.namespaces import CODE, PERSONAL, REFERENCE
from ev_assistant.query import Filters, build_plan
from ev_assistant.rerank import BaseReranker, LexicalReranker
from ev_assistant.retrieval import (
    BELOW_THRESHOLD,
    EMPTY_STORE,
    NO_CANDIDATES,
    OK,
    SKIPPED,
    Retriever,
    reciprocal_rank_fusion,
)
from ev_assistant.store import vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")

CORPUS = [
    ("Water purification", "wikipedia:Water", "wikipedia",
     "Boiling is the most reliable way to make water safe to drink. "
     "Bring water to a rolling boil and hold it for one full minute. "
     "Cloudy water should be filtered through cloth before boiling."),
    ("Knots", "wikipedia:Knot", "wikipedia",
     "The bowline forms a fixed loop that will not slip under load. "
     "The clove hitch secures a rope to a post."),
    ("Fire making", "wikipedia:Fire", "wikipedia",
     "Gather tinder, kindling and fuel wood before striking a spark. "
     "Dry grass and birch bark make excellent tinder."),
    ("Shed notes", "notes://shed", "note",
     "The blue tarp is in the shed behind the mower. "
     "I moved the water filter into the garage in March."),
    ("parser.py", "/home/me/parser.py", "code",
     "def parse_query(text):\n    return text.split()\n"),
]


class ScriptedReranker(BaseReranker):
    """Returns a score per passage from a lookup, so floors are exact."""

    name = "scripted"
    default_floor = 0.5

    def __init__(self, scores: dict[str, float], fallback: float = 0.0):
        self.scores = scores
        self.fallback = fallback
        self.calls: list[tuple[str, int]] = []

    def available(self) -> bool:
        return True

    def score(self, query, passages):
        self.calls.append((query, len(passages)))
        out = []
        for passage in passages:
            value = self.fallback
            for marker, score in self.scores.items():
                if marker.lower() in passage.lower():
                    value = score
                    break
            out.append(value)
        return out


@pytest.fixture
def corpus(cfg, store):
    """A small store with documents across several namespaces, indexed."""
    cfg.reranker_backend = "lexical"
    for title, uri, source_type, text in CORPUS:
        store.add_document(
            source_uri=uri, source_type=source_type, title=title, text=text,
            namespace=namespaces.route(source_type, uri),
            chunks=chunk_rows(text, title=title, source_uri=uri),
        )
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()
    return embedder


def retriever(cfg, store, corpus, reranker=None, **kw):
    r = Retriever(cfg, store, embedder=corpus, reranker=reranker or LexicalReranker(),
                  complete=lambda *a, **k: None)
    for key, value in kw.items():
        setattr(r, key, value)
    return r


# -- reciprocal rank fusion ------------------------------------------------


def test_rrf_rewards_appearing_in_several_lists():
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3], "sparse": [3, 4, 5]}, k=60)
    # 3 is in both lists, so it beats 1 which is only first in one.
    assert fused[3][0] > fused[1][0]


def test_rrf_scores_follow_rank():
    fused = reciprocal_rank_fusion({"a": [10, 20, 30]}, k=60)
    assert fused[10][0] == pytest.approx(1 / 61)
    assert fused[20][0] == pytest.approx(1 / 62)
    assert fused[10][0] > fused[20][0] > fused[30][0]


def test_rrf_records_the_best_rank_per_list():
    fused = reciprocal_rank_fusion({"a": [7, 7, 1], "b": [1]}, k=60)
    assert fused[7][1] == {"a": 1}
    assert fused[1][1] == {"a": 3, "b": 1}


def test_rrf_needs_no_score_normalisation():
    # The whole point: two channels on wildly different scales fuse fine,
    # because only rank is used.
    fused = reciprocal_rank_fusion({"bm25": [5, 6], "cosine": [6, 5]}, k=60)
    assert fused[5][0] == pytest.approx(fused[6][0])


def test_rrf_on_nothing():
    assert reciprocal_rank_fusion({}) == {}
    assert reciprocal_rank_fusion({"a": []}) == {}


def test_rrf_k_flattens_rank_influence():
    tight = reciprocal_rank_fusion({"a": [1, 2]}, k=1)
    loose = reciprocal_rank_fusion({"a": [1, 2]}, k=1000)
    assert tight[1][0] / tight[2][0] > loose[1][0] / loose[2][0]


# -- the happy path --------------------------------------------------------


@needs_vec
def test_retrieves_the_right_chunk(cfg, store, corpus):
    result = retriever(cfg, store, corpus).retrieve("how do I make water safe to drink", k=3)

    assert result.reason == OK
    assert result
    assert result.chunks[0].document.title == "Water purification"
    assert result.chunks[0].score > 0


@needs_vec
def test_both_channels_contribute(cfg, store, corpus):
    result = retriever(cfg, store, corpus).retrieve("rolling boil for one minute", k=5)

    assert result.trace.counts["dense_variants"] >= 1
    assert result.trace.counts["sparse_variants"] >= 1
    assert result.trace.counts["fused"] > 0


@needs_vec
def test_keyword_only_still_works_without_vectors(cfg, store, corpus, monkeypatch):
    # The keyword-only install: sqlite-vec missing, everything else the same.
    monkeypatch.setattr(store, "vec_available", False)
    result = retriever(cfg, store, corpus).retrieve("bowline fixed loop", k=3)

    assert result.reason == OK
    assert result.chunks[0].document.title == "Knots"
    assert result.trace.counts["dense_variants"] == 0


@needs_vec
def test_the_trace_records_every_stage(cfg, store, corpus):
    result = retriever(cfg, store, corpus).retrieve("how do I purify water", k=3)

    for stage in ("plan", "embed", "search", "fuse", "shortlist", "rerank"):
        assert stage in result.trace.timings_ms
    assert result.trace.total_ms > 0
    assert result.trace.reranker
    assert "chunks" in result.explain()


# -- abstention ------------------------------------------------------------


@needs_vec
def test_nothing_above_the_floor_returns_nothing(cfg, store, corpus):
    # An empty result is a correct answer.
    r = retriever(cfg, store, corpus, reranker=ScriptedReranker({}, fallback=0.1))
    cfg.relevance_floor = 0.5

    result = r.retrieve("what is the capital of Mongolia", k=3)

    assert result.reason == BELOW_THRESHOLD
    assert result.chunks == []
    assert result.empty
    assert not result
    assert result.trace.dropped_below_floor > 0


@needs_vec
def test_a_real_question_about_an_uncovered_topic_abstains(cfg, store, corpus):
    result = retriever(cfg, store, corpus).retrieve(
        "what is the capital city of Mongolia", k=3)
    assert result.reason == BELOW_THRESHOLD
    assert result.chunks == []


@needs_vec
def test_the_floor_is_configurable(cfg, store, corpus):
    r = retriever(cfg, store, corpus, reranker=ScriptedReranker({"boiling": 0.4}, fallback=0.0))

    cfg.relevance_floor = 0.3
    assert r.retrieve("boiling water safely", k=3).chunks

    cfg.relevance_floor = 0.6
    assert r.retrieve("boiling water safely", k=3).chunks == []


def test_a_question_that_needs_no_store_never_touches_it(cfg, store, corpus):
    result = retriever(cfg, store, corpus).retrieve("thanks!", k=3)

    assert result.reason == SKIPPED
    assert result.chunks == []
    assert "skipped" in result.explain().lower()


def test_an_empty_store_says_so(cfg, store):
    r = Retriever(cfg, store, embedder=None, reranker=LexicalReranker(),
                  complete=lambda *a, **k: None)
    result = r.retrieve("how do I purify water", k=3)

    assert result.reason == EMPTY_STORE
    assert result.chunks == []


@needs_vec
def test_no_candidates_is_distinct_from_below_threshold(cfg, store, corpus, monkeypatch):
    r = retriever(cfg, store, corpus)
    monkeypatch.setattr(r, "_dense", lambda *a, **k: [])
    monkeypatch.setattr(r, "_sparse", lambda *a, **k: [])

    assert r.retrieve("how do I purify water", k=3).reason == NO_CANDIDATES


# -- namespaces: ordering, never admission ---------------------------------


@needs_vec
def test_personal_outranks_reference_on_an_equal_match(cfg, store, corpus):
    r = retriever(cfg, store, corpus,
                  reranker=ScriptedReranker({"water": 0.9}, fallback=0.0))
    result = r.retrieve("water storage and filtering", k=5)

    titles = [c.document.title for c in result.chunks]
    assert titles[0] == "Shed notes"          # personal, boosted above
    assert "Water purification" in titles     # reference, still present


@needs_vec
def test_a_namespace_boost_cannot_buy_a_chunk_past_the_floor(cfg, store, corpus):
    """The invariant: boosts reorder, they never admit."""
    cfg.relevance_floor = 0.5
    # The personal chunk scores below the floor; its 1.25x boost would lift
    # the weighted score to 0.5 exactly, but the floor tests the raw score.
    r = retriever(cfg, store, corpus,
                  reranker=ScriptedReranker({"blue tarp": 0.4}, fallback=0.0))

    result = r.retrieve("where is the blue tarp", k=5)

    assert result.chunks == []
    assert result.reason == BELOW_THRESHOLD


@needs_vec
def test_a_hard_namespace_restriction_excludes_others(cfg, store, corpus):
    r = retriever(cfg, store, corpus)
    result = r.retrieve("parse_query function in the codebase", k=5,
                        restrict_namespaces=True)

    assert result.plan.namespace_plan.restrict
    for chunk in result.chunks:
        assert chunk.document.namespace in (CODE, PERSONAL)


@needs_vec
def test_explicit_namespaces_override_the_plan(cfg, store, corpus):
    r = retriever(cfg, store, corpus)
    result = r.retrieve("water purification", k=5, namespaces=[REFERENCE], restrict_namespaces=True)

    for chunk in result.chunks:
        assert chunk.document.namespace in (REFERENCE, PERSONAL)


@needs_vec
def test_filtered_search_over_fetches_instead_of_coming_back_empty(cfg, store, corpus):
    """Regression guard for a real sqlite-vec trap.

    vec0 picks its top-k *before* any join filter, so a naive filtered KNN
    can return nothing even when matches exist. Verified against the
    extension; this keeps the over-fetch in place.
    """
    r = retriever(cfg, store, corpus)
    plan = namespaces.plan([REFERENCE], restrict=True, cfg=cfg)
    blob = corpus.embed_query("water purification boiling")

    # One slot, restricted to a namespace that isn't the nearest neighbour.
    hits = r._dense(blob, 1, plan, None)

    assert hits, "filtered KNN came back empty - the over-fetch is gone"


# -- filters ---------------------------------------------------------------


@needs_vec
def test_temporal_filters_narrow_the_search(cfg, store):
    cfg.reranker_backend = "lexical"
    old = calendar.timegm((2015, 1, 1, 0, 0, 0, 0, 1, 0))
    new = calendar.timegm((2025, 1, 1, 0, 0, 0, 0, 1, 0))
    for title, when in (("Old report", old), ("New report", new)):
        text = f"{title}: the quarterly water usage figures were published."
        store.add_document(source_uri=f"web://{title}", source_type="web", title=title,
                           text=text, published_at=when,
                           chunks=chunk_rows(text, title=title))
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()
    r = retriever(cfg, store, embedder)

    after_2020 = calendar.timegm((2020, 1, 1, 0, 0, 0, 0, 1, 0))
    result = r.retrieve("quarterly water usage figures", k=5,
                        filters=Filters(published_after=after_2020))

    titles = [c.document.title for c in result.chunks]
    assert "New report" in titles
    assert "Old report" not in titles


@needs_vec
def test_documents_without_a_date_survive_a_temporal_filter(cfg, store, corpus):
    # Most of the store has no published_at; a date filter must not erase it.
    r = retriever(cfg, store, corpus)
    after = calendar.timegm((2020, 1, 1, 0, 0, 0, 0, 1, 0))

    result = r.retrieve("how do I purify water", k=3, filters=Filters(published_after=after))

    assert result.chunks


# -- facts -----------------------------------------------------------------


@needs_vec
def test_facts_matching_an_entity_come_back(cfg, store, corpus):
    doc_id = store.get_chunk(1).document_id
    store.add_facts(doc_id, [
        {"statement": "Tungsten melts at 3422 degrees Celsius.", "subject": "Tungsten",
         "confidence": 0.9},
    ])
    r = retriever(cfg, store, corpus)

    result = r.retrieve("what does Tungsten melt at", k=3)

    assert any("3422" in f.statement for f in result.facts)


@needs_vec
def test_facts_survive_an_abstention(cfg, store, corpus):
    """An exact entity match is evidence even when no chunk clears the floor."""
    doc_id = store.get_chunk(1).document_id
    store.add_facts(doc_id, [
        {"statement": "Tungsten melts at 3422 degrees Celsius.", "subject": "Tungsten"},
    ])
    cfg.relevance_floor = 0.99
    r = retriever(cfg, store, corpus, reranker=ScriptedReranker({}, fallback=0.1))

    result = r.retrieve("what does Tungsten melt at", k=3)

    assert result.reason == BELOW_THRESHOLD
    assert result.chunks == []
    assert result.facts
    assert bool(result) is True   # not empty - there is still something to say


@needs_vec
def test_no_entities_means_no_fact_lookup(cfg, store, corpus):
    assert retriever(cfg, store, corpus).retrieve("how do i boil water", k=3).facts == []


# -- diversification and neighbours ----------------------------------------


@needs_vec
def test_one_document_cannot_monopolise_the_results(cfg, store):
    cfg.reranker_backend = "lexical"
    long_text = "\n\n".join(
        f"Section {i}. Water purification matters because water carries disease. "
        + "Boil the water thoroughly every single time you collect it. " * 8
        for i in range(12)
    )
    store.add_document(source_uri="wikipedia:Long", source_type="wikipedia", title="Long",
                       text=long_text, chunks=chunk_rows(long_text, title="Long"))
    short = "Boiling water makes it safe to drink in an emergency."
    store.add_document(source_uri="notes://short", source_type="note", title="Short",
                       text=short, chunks=chunk_rows(short, title="Short"))
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()
    r = retriever(cfg, store, embedder)
    r.max_per_document = 2

    result = r.retrieve("boil water to purify it", k=8)

    from collections import Counter
    counts = Counter(c.document.title for c in result.chunks)
    assert counts["Long"] <= 2
    assert "Short" in counts


@needs_vec
def test_neighbour_expansion_restores_continuity(cfg, store):
    cfg.reranker_backend = "lexical"
    text = "\n\n".join(f"Paragraph {i} about purifying water carefully and thoroughly."
                       for i in range(6))
    store.add_document(source_uri="wikipedia:Multi", source_type="wikipedia", title="Multi",
                       text=text, chunks=chunk_rows(text, title="Multi", target_tokens=12))
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()

    without = retriever(cfg, store, embedder, neighbor_window=0, max_per_document=99)
    with_neighbours = retriever(cfg, store, embedder, neighbor_window=1, max_per_document=99)

    plain = without.retrieve("purifying water thoroughly", k=2)
    expanded = with_neighbours.retrieve("purifying water thoroughly", k=2)

    assert len(expanded.chunks) > len(plain.chunks)
    # Neighbours sort below the chunk that earned its place.
    assert expanded.chunks[0].score >= max(c.score for c in expanded.chunks[1:])


@needs_vec
def test_k_bounds_the_result(cfg, store, corpus):
    r = retriever(cfg, store, corpus, max_per_document=99)
    assert len(r.retrieve("purify water", k=1).chunks) <= 1
    assert len(r.retrieve("purify water", k=2).chunks) <= 2


# -- resilience ------------------------------------------------------------


@needs_vec
def test_a_failing_reranker_falls_back_instead_of_crashing(cfg, store, corpus):
    class Broken(BaseReranker):
        name = "broken"
        default_floor = 0.0

        def available(self):
            return True

        def score(self, query, passages):
            raise RuntimeError("model went away mid-query")

    result = retriever(cfg, store, corpus, reranker=Broken()).retrieve(
        "how do I purify water", k=3)

    assert result.reason == OK
    assert result.chunks
    assert "lexical" in result.trace.reranker


@needs_vec
def test_a_malformed_keyword_query_does_not_take_the_search_down(cfg, store, corpus):
    r = retriever(cfg, store, corpus)
    # Straight into the channel, bypassing fts_match_query's sanitising.
    assert r._sparse('NEAR("unclosed', 5, namespaces.plan(), None) == []


@needs_vec
def test_an_unembeddable_query_degrades_to_keyword_search(cfg, store, corpus, monkeypatch):
    def boom(texts):
        raise RuntimeError("embedding model went away")

    monkeypatch.setattr(corpus, "embed_queries", boom)
    result = retriever(cfg, store, corpus).retrieve("bowline fixed loop", k=3)

    assert result.reason == OK
    assert result.chunks[0].document.title == "Knots"


@needs_vec
def test_a_precomputed_plan_is_reused(cfg, store, corpus):
    plan = build_plan("how do I purify water")
    result = retriever(cfg, store, corpus).retrieve("ignored", k=3, plan=plan)
    assert result.plan is plan


@needs_vec
def test_retrieval_survives_junk_queries(cfg, store, corpus):
    r = retriever(cfg, store, corpus)
    for junk in ["", "   ", "?????", "🔥🔥🔥", "a" * 3000, 'unbalanced " quote']:
        result = r.retrieve(junk, k=3)
        assert result.reason in (OK, SKIPPED, BELOW_THRESHOLD, NO_CANDIDATES)
