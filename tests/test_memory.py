from __future__ import annotations


def test_add_fact_dedups_by_external_id(memory):
    assert memory.add_fact("feed:x", "Headline one", tags="news", external_id="guid-1") is True
    assert memory.add_fact("feed:x", "Headline one (updated)", tags="news", external_id="guid-1") is False
    assert memory.fact_count() == 1


def test_add_fact_without_external_id_never_dedups(memory):
    memory.add_fact("weather", "Sunny, 25C")
    memory.add_fact("weather", "Sunny, 25C")
    assert memory.fact_count() == 2


def test_search_facts_prefers_keyword_overlap(memory):
    memory.add_fact("feed:tech", "New Python release ships faster interpreter", tags="news", external_id="a")
    memory.add_fact("feed:sports", "Local team wins championship game", tags="news", external_id="b")

    results = memory.search_facts("tell me about the python release", limit=1)

    assert len(results) == 1
    assert "Python" in results[0].text


def test_search_facts_fills_remaining_slots_with_recent(memory):
    for i in range(3):
        memory.add_fact("weather", f"Reading {i}", external_id=f"w{i}")

    results = memory.search_facts("completely unrelated query with no overlap", limit=2)

    assert len(results) == 2


def test_format_context_empty_when_no_facts(memory):
    assert memory.format_context("anything") == ""


def test_format_context_lists_facts(memory):
    memory.add_fact("weather", "Sunny and 25C", external_id="w1")
    context = memory.format_context("weather")
    assert "Sunny and 25C" in context
    assert context.startswith("Things E.V. currently knows")


def test_conversation_turns_round_trip_in_order(memory):
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi there")
    memory.add_turn("user", "how are you")

    turns = memory.recent_turns(limit=10)

    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert turns[0]["content"] == "hello"
    assert turns[-1]["content"] == "how are you"


def test_recent_turns_respects_limit_and_stays_in_order(memory):
    for i in range(5):
        memory.add_turn("user", f"message {i}")

    turns = memory.recent_turns(limit=2)

    assert [t["content"] for t in turns] == ["message 3", "message 4"]
