"""Phase 8 of the retrieval spec: context assembly."""

from __future__ import annotations

import pytest

from ev_assistant.chunking import estimate_tokens
from ev_assistant.context import (
    DEFAULT_BUDGET_TOKENS,
    FACTS_HEADER,
    HEADER,
    NOTHING_RELEVANT,
    build,
    citation_map,
    edge_order,
    format_facts,
    source_label,
    truncate_to_sentence,
)
from ev_assistant.retrieval import (
    BELOW_THRESHOLD,
    EMPTY_STORE,
    NO_CANDIDATES,
    OK,
    SKIPPED,
    RetrievalResult,
)
from ev_assistant.store import Chunk, Document, Fact

SENTENCES = [
    "Boiling is the most reliable way to make water safe to drink.",
    "Bring the water to a rolling boil and hold it for one full minute.",
    "Above two thousand metres, hold the boil for three minutes instead.",
    "Cloudy water should be filtered through cloth before it is boiled.",
]


def document(doc_id=1, title="Water purification", source_type="wikipedia",
             uri="wikipedia:Water") -> Document:
    return Document(id=doc_id, source_uri=uri, source_type=source_type,
                    namespace="reference", title=title, content_hash=f"h{doc_id}",
                    fetched_at=0.0)


def chunk(chunk_id=1, text=None, score=0.9, doc=None) -> Chunk:
    c = Chunk(id=chunk_id, document_id=(doc or document()).id, ordinal=0,
              text=text if text is not None else " ".join(SENTENCES))
    c.document = doc or document()
    c.score = score
    return c


def fact(statement="Boiling water for one minute makes it safe.", confidence=0.9) -> Fact:
    return Fact(id=1, document_id=1, statement=statement, subject="water",
                confidence=confidence, asserted_at=0.0)


def result(chunks=None, facts=None, reason=OK) -> RetrievalResult:
    return RetrievalResult(chunks=chunks or [], facts=facts or [], reason=reason)


# -- source labels ---------------------------------------------------------


@pytest.mark.parametrize("source_type,expected", [
    ("wikipedia", "Wikipedia"), ("web", "Web"), ("pdf", "PDF"), ("note", "Your notes"),
    ("code", "Code"), ("conversation", "Earlier conversation"), ("feed", "News feed"),
])
def test_source_label_reads_naturally(source_type, expected):
    label = source_label(chunk(doc=document(source_type=source_type, title="Thing")))
    assert label == f"{expected} - Thing"


def test_source_label_handles_an_unknown_type():
    assert source_label(chunk(doc=document(source_type="gopher"))) .startswith("Gopher - ")


def test_source_label_falls_back_when_metadata_is_thin():
    bare = Chunk(id=1, document_id=1, ordinal=0, text="x")
    assert source_label(bare) == "Unknown source"

    untitled = chunk(doc=document(title="", uri="wikipedia:Thing"))
    assert "wikipedia:Thing" in source_label(untitled)


# -- truncation ------------------------------------------------------------


def test_text_within_budget_is_untouched():
    text = " ".join(SENTENCES)
    assert truncate_to_sentence(text, 1000) == text


def test_truncation_stops_at_a_sentence_boundary():
    text = " ".join(SENTENCES)
    trimmed = truncate_to_sentence(text, estimate_tokens(SENTENCES[0]) + 2)

    assert trimmed.endswith(".")
    assert trimmed in text
    assert SENTENCES[0] in trimmed
    assert SENTENCES[2] not in trimmed


def test_truncation_never_cuts_a_word_in_half():
    # A single sentence longer than the whole budget is the hard case.
    runaway = "supercalifragilistic " * 200
    trimmed = truncate_to_sentence(runaway, 20)

    assert estimate_tokens(trimmed) <= 24     # budget plus the ellipsis
    assert "supercalifrag " not in trimmed
    assert trimmed.endswith("...")


def test_truncation_keeps_whole_sentences_only():
    """At every budget, the result is exactly the first N sentences."""
    text = " ".join(SENTENCES)
    prefixes = {" ".join(SENTENCES[:n]) for n in range(1, len(SENTENCES) + 1)}

    for budget in range(10, 120, 3):
        trimmed = truncate_to_sentence(text, budget)
        if trimmed.endswith("..."):
            continue    # the first sentence alone was over budget
        assert trimmed in prefixes, f"budget {budget} produced a partial sentence"


# -- facts -----------------------------------------------------------------


def test_facts_are_listed_most_confident_first():
    text, tokens = format_facts([
        fact("Water freezes at zero degrees.", 0.4),
        fact("Water boils at one hundred degrees.", 0.95),
    ], 500)

    assert text.startswith(FACTS_HEADER)
    assert text.index("boils") < text.index("freezes")
    assert tokens > 0


def test_facts_respect_their_budget():
    many = [fact(f"Fact number {i} is recorded here in full.", 0.9) for i in range(200)]
    text, tokens = format_facts(many, 60)

    assert tokens <= 60
    assert 0 < text.count("\n- ") < 200


def test_no_facts_no_section():
    assert format_facts([], 500) == ("", 0)
    assert format_facts([fact()], 0) == ("", 0)


# -- edge ordering ---------------------------------------------------------


def test_edge_order_puts_the_best_at_both_ends():
    # Attention is strongest at the edges, so the runner-up goes last.
    assert edge_order(["a", "b", "c", "d", "e"]) == ["a", "c", "d", "e", "b"]


def test_edge_order_leaves_short_lists_alone():
    assert edge_order([]) == []
    assert edge_order(["a"]) == ["a"]
    assert edge_order(["a", "b"]) == ["a", "b"]


def test_edge_order_keeps_every_item():
    items = list("abcdefg")
    assert sorted(edge_order(items)) == sorted(items)


# -- assembly --------------------------------------------------------------


def test_a_block_is_tagged_and_labelled(cfg):
    block = build(result([chunk(1, score=0.9)]), cfg)

    assert block.text.startswith(HEADER)
    assert "[S1] Wikipedia - Water purification" in block.text
    assert block.sources[0]["tag"] == "[S1]"
    assert block.cited
    assert bool(block)


def test_tags_are_sequential_and_unique(cfg):
    chunks = [chunk(i, score=1.0 - i / 10,
                    doc=document(doc_id=i, title=f"Doc {i}")) for i in range(1, 6)]
    block = build(result(chunks), cfg)

    tags = [s["tag"] for s in block.sources]
    assert tags == ["[S1]", "[S2]", "[S3]", "[S4]", "[S5]"]
    for tag in tags:
        assert block.text.count(tag) == 1


def test_the_best_chunk_opens_and_the_runner_up_closes(cfg):
    chunks = [chunk(i, text=f"Body of document {i}. " * 5, score=1.0 - i / 10,
                    doc=document(doc_id=i, title=f"Doc {i}")) for i in range(1, 6)]
    block = build(result(chunks), cfg)

    assert block.text.index("[S1]") < block.text.index("[S3]")
    assert block.text.rindex("[S2]") > block.text.rindex("[S5]")


def test_facts_sit_above_the_prose(cfg):
    block = build(result([chunk(1)], facts=[fact()]), cfg)

    assert FACTS_HEADER in block.text
    assert block.text.index(FACTS_HEADER) < block.text.index("[S1]")


def test_the_budget_is_a_hard_ceiling(cfg):
    cfg.context_budget_tokens = 200
    chunks = [chunk(i, text=" ".join(SENTENCES) * 3, score=1.0 - i / 20,
                    doc=document(doc_id=i, title=f"Doc {i}")) for i in range(1, 21)]

    block = build(result(chunks), cfg)

    assert block.tokens <= 200 * 1.15   # header and tag overhead
    assert block.dropped > 0
    assert len(block.sources) < 20


def test_a_truncated_chunk_still_ends_on_a_sentence(cfg):
    cfg.context_budget_tokens = 90
    block = build(result([chunk(1, text=" ".join(SENTENCES) * 4)]), cfg)

    body = block.text.split("\n", 2)[-1]
    assert block.truncated == 1
    assert body.rstrip().endswith((".", "!", "?", "..."))


def test_the_default_budget_is_generous_enough_for_a_normal_answer(cfg):
    chunks = [chunk(i, score=1.0 - i / 20, doc=document(doc_id=i, title=f"Doc {i}"))
              for i in range(1, 9)]
    block = build(result(chunks), cfg)

    assert cfg.context_budget_tokens == DEFAULT_BUDGET_TOKENS
    assert len(block.sources) == 8
    assert block.dropped == 0
    assert block.truncated == 0


# -- saying nothing --------------------------------------------------------


@pytest.mark.parametrize("reason", [BELOW_THRESHOLD, EMPTY_STORE, NO_CANDIDATES])
def test_an_empty_result_says_so_explicitly(cfg, reason):
    # A silent absence is indistinguishable from never having searched.
    block = build(result(reason=reason), cfg)

    assert block.text == NOTHING_RELEVANT
    assert "searched" in block.text
    assert "own knowledge" in block.text
    assert block.reason == reason
    assert not block.cited
    assert block.sources == []


def test_a_skipped_question_gets_no_block_at_all(cfg):
    block = build(result(reason=SKIPPED), cfg)

    assert block.text == ""
    assert not block
    assert block.reason == SKIPPED


def test_facts_alone_are_enough_to_build_a_block(cfg):
    block = build(result(facts=[fact()], reason=BELOW_THRESHOLD), cfg)

    assert block.text != NOTHING_RELEVANT
    assert FACTS_HEADER in block.text
    assert "[S1]" not in block.text


def test_a_budget_too_small_for_anything_falls_back_to_saying_nothing(cfg):
    cfg.context_budget_tokens = 12
    block = build(result([chunk(1, text=" ".join(SENTENCES) * 10)]), cfg)

    assert block.text == NOTHING_RELEVANT
    assert block.reason == BELOW_THRESHOLD


# -- citations -------------------------------------------------------------


def test_citation_map_turns_tags_back_into_names(cfg):
    chunks = [
        chunk(1, score=0.9, doc=document(1, "Water purification")),
        chunk(2, score=0.8, doc=document(2, "Shed notes", "note", "notes://shed")),
    ]
    block = build(result(chunks), cfg)

    assert citation_map(block) == {
        "[S1]": "Wikipedia - Water purification",
        "[S2]": "Your notes - Shed notes",
    }


def test_sources_carry_enough_to_answer_where_did_you_get_that(cfg):
    block = build(result([chunk(1, score=0.77)]), cfg)
    source = block.sources[0]

    assert source["source_uri"] == "wikipedia:Water"
    assert source["score"] == pytest.approx(0.77)
    assert source["chunk_id"] == 1


def test_describe_summarises_the_block(cfg):
    cfg.context_budget_tokens = 200
    chunks = [chunk(i, text=" ".join(SENTENCES) * 3, score=1.0 - i / 20,
                    doc=document(doc_id=i, title=f"Doc {i}")) for i in range(1, 21)]
    assert "dropped" in build(result(chunks), cfg).describe()
    assert "sources" in build(result([chunk(1)]), cfg).describe()


# -- end to end ------------------------------------------------------------


def test_a_real_retrieval_assembles_cleanly(cfg, store):
    from ev_assistant import namespaces
    from ev_assistant.chunking import chunk_rows
    from ev_assistant.embeddings import Embedder, HashingBackend
    from ev_assistant.rerank import LexicalReranker
    from ev_assistant.retrieval import Retriever
    from ev_assistant.store import vec_supported

    if not vec_supported():
        pytest.skip("sqlite-vec not installed")

    cfg.reranker_backend = "lexical"
    for title, uri, source_type, text in [
        ("Water purification", "wikipedia:Water", "wikipedia",
         "Boiling is the most reliable way to make water safe to drink. "
         "Bring water to a rolling boil and hold it for one full minute."),
        ("Shed notes", "notes://shed", "note",
         "The blue tarp is in the shed. I moved the water filter to the garage."),
    ]:
        store.add_document(source_uri=uri, source_type=source_type, title=title, text=text,
                           namespace=namespaces.route(source_type, uri),
                           chunks=chunk_rows(text, title=title))
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    embedder.index_pending()
    retriever = Retriever(cfg, store, embedder=embedder, reranker=LexicalReranker(),
                          complete=lambda *a, **k: None)

    block = build(retriever.retrieve("how do I make water safe to drink", k=3), cfg)

    assert block.text.startswith(HEADER)
    assert "Water purification" in block.text
    assert block.tokens <= cfg.context_budget_tokens
    assert block.sources

    nothing = build(retriever.retrieve("what is the capital city of Mongolia", k=3), cfg)
    assert nothing.text == NOTHING_RELEVANT
