from __future__ import annotations

from ev_assistant.knowledge import chunk_text


def test_chunks_carry_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 200 for i in range(4))
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Each chunk after the first starts with the tail of the previous one.
    tail = chunks[0][-40:]
    assert tail.split()[-1] in chunks[1]


def test_zero_overlap_is_respected():
    text = "\n\n".join(f"Para {i} " + "word " * 200 for i in range(3))
    no_overlap = chunk_text(text, target_tokens=100, overlap_tokens=0)
    with_overlap = chunk_text(text, target_tokens=100, overlap_tokens=30)
    assert len(with_overlap[1]) > len(no_overlap[1])


def test_dedup_skips_identical_content(knowledge):
    first = knowledge.add_document("Fire", "notes", "Dry tinder, then a spark, then kindling.")
    second = knowledge.add_document("Fire", "notes", "Dry tinder, then a spark, then kindling.")
    assert first > 0
    assert second == 0  # identical content -> no duplicate passages
    assert knowledge.passage_count() == first


def test_dedup_is_content_based_not_title_based(knowledge):
    knowledge.add_document("A", "s1", "Same body text here.")
    again = knowledge.add_document("Different title", "s2", "Same body text here.")
    assert again == 0


def test_different_content_is_stored(knowledge):
    knowledge.add_document("A", "s1", "First body.")
    n = knowledge.add_document("B", "s2", "Completely different body.")
    assert n > 0
    assert knowledge.document_count() == 2


def test_force_reingests(knowledge):
    knowledge.add_document("Fire", "notes", "Tinder and spark.")
    forced = knowledge.add_document("Fire", "notes", "Tinder and spark.", force=True)
    assert forced > 0


def test_empty_text_is_ignored(knowledge):
    assert knowledge.add_document("Nothing", "notes", "   ") == 0
    assert knowledge.passage_count() == 0


def test_knows_source(knowledge):
    knowledge.add_document("Water", "wikipedia:Water", "Boil it.")
    assert knowledge.knows_source("wikipedia:Water")
    assert not knowledge.knows_source("wikipedia:Fire")
