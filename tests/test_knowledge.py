from __future__ import annotations

from ev_assistant.knowledge import Knowledge, chunk_text


def test_chunk_text_splits_long_content():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(5))
    chunks = chunk_text(text, target_chars=400)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_add_and_count(knowledge):
    added = knowledge.add_document("Fire making", "notes", "Gather dry tinder. Strike a spark. Build slowly.")
    assert added >= 1
    assert knowledge.passage_count() == added


def test_search_finds_relevant_passage(knowledge):
    knowledge.add_document("Water", "notes", "Boil water for one minute to purify it before drinking.")
    knowledge.add_document("Shelter", "notes", "Build a lean-to against wind and rain using branches.")

    results = knowledge.search("how do I purify water", limit=2)

    assert results
    assert "boil" in results[0].text.lower()


def test_search_empty_query_returns_nothing(knowledge):
    knowledge.add_document("X", "notes", "something")
    assert knowledge.search("", limit=3) == []


def test_format_context_labels_sources(knowledge):
    knowledge.add_document("Knots", "notes", "The bowline makes a fixed loop that won't slip.")
    context = knowledge.format_context("bowline knot")
    assert "Knots" in context
    assert "bowline" in context.lower()


def test_sources_lists_titles(knowledge):
    knowledge.add_document("Alpha", "notes", "content one")
    knowledge.add_document("Beta", "notes", "content two")
    assert set(knowledge.sources()) == {"Alpha", "Beta"}
