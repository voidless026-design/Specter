"""Phase 5 of the retrieval spec: query understanding.

Every test runs on the rules-only path unless it explicitly passes a fake
`complete`, so the suite needs no brain, no network and no microphone - which
is also the path E.V. takes on a machine with nothing reachable.
"""

from __future__ import annotations

import calendar
import sqlite3
import time

import pytest

from ev_assistant.namespaces import CODE, NEWS, PERSONAL
from ev_assistant.query import (
    Filters,
    RetrievalPlan,
    build_plan,
    build_variants,
    classify,
    classify_by_rules,
    extract_entities,
    extract_filters,
    extract_temporal,
    fts_match_query,
    hyde,
    keyword_query,
    needs_rewrite,
    rewrite,
    salient_topic,
)

NOW = calendar.timegm((2026, 6, 15, 12, 0, 0, 0, 1, 0))


def fake_complete(answer):
    """A stand-in for providers.complete that always says `answer`."""
    calls = []

    def complete(cfg, system, user, max_tokens=256):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return answer

    complete.calls = calls
    return complete


# -- classification --------------------------------------------------------


@pytest.mark.parametrize("text", [
    "hey", "hello", "thanks", "how are you", "good morning", "never mind", "bye",
])
def test_chitchat_skips_retrieval(text):
    assert classify_by_rules(text) == (False, "chitchat")


@pytest.mark.parametrize("text", [
    "what's 17 times 3", "calculate 45 / 9", "how much is 2 + 2", "12 * 8",
])
def test_arithmetic_skips_retrieval(text):
    needed, _ = classify_by_rules(text)
    assert needed is False


@pytest.mark.parametrize("text", [
    "open firefox", "close spotify", "turn the volume up", "play the next track",
    "lock the screen", "disable that gnome extension",
])
def test_device_commands_skip_retrieval(text):
    assert classify_by_rules(text) == (False, "device command")


def test_a_question_that_looks_like_a_command_still_retrieves():
    # "how do I open a locked file?" is a question, not an instruction.
    assert classify_by_rules("open a stuck jar lid, how?") != (False, "device command")


@pytest.mark.parametrize("text", ["what time is it", "what's the date", "what day is it today"])
def test_clock_questions_skip_retrieval(text):
    assert classify_by_rules(text) == (False, "clock")


def test_empty_and_tiny_input_skips():
    assert classify_by_rules("") == (False, "empty")
    assert classify_by_rules("   ") == (False, "empty")
    assert classify_by_rules("okay sure")[0] is False


@pytest.mark.parametrize("text", [
    "what did I tell you about the shed",
    "check my notes on water storage",
    "where is that in the codebase",
    "what do you know about my car",
])
def test_explicit_store_references_always_retrieve(text):
    needed, reason = classify_by_rules(text)
    assert needed is True
    assert reason == "names the store"


def test_store_reference_beats_the_chitchat_rule():
    # "ok, what did I tell you about the shed" opens like chitchat.
    assert classify_by_rules("ok what did I tell you about the shed")[0] is True


def test_rules_defer_when_unsure():
    assert classify_by_rules("how do I purify water in the wild") is None


def test_the_model_decides_when_rules_are_unsure():
    needed, _, by = classify("how do I purify water", cfg=object(),
                             complete=fake_complete("SKIP"))
    assert needed is False
    assert by == "model"

    needed, _, by = classify("how do I purify water", cfg=object(),
                             complete=fake_complete("SEARCH"))
    assert needed is True
    assert by == "model"


def test_rules_are_not_overridden_by_the_model():
    complete = fake_complete("SEARCH")
    needed, _, by = classify("hello", cfg=object(), complete=complete)
    assert needed is False
    assert by == "rules"
    assert complete.calls == []   # and the cheap call was never made


def test_an_unreachable_or_confused_model_falls_back_to_retrieving():
    # An unnecessary lookup costs latency; the relevance floor costs nothing.
    assert classify("how do I purify water", cfg=object(),
                    complete=fake_complete(None))[0] is True
    assert classify("how do I purify water", cfg=object(),
                    complete=fake_complete("maybe?"))[0] is True
    assert classify("how do I purify water")[2] == "default"


def test_the_classifier_call_is_kept_tiny():
    complete = fake_complete("SKIP")
    classify("how do I purify water", cfg=object(), complete=complete)
    assert complete.calls[0]["max_tokens"] <= 8
    assert "SEARCH" in complete.calls[0]["system"]


# -- rewriting -------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "what about its melting point?",
    "and how about hers",
    "is that still true",
    "what about them",
])
def test_dangling_questions_are_flagged_for_rewrite(text):
    assert needs_rewrite(text)


@pytest.mark.parametrize("text", [
    "how do I purify water in the wild",
    "what is the melting point of tungsten",
])
def test_self_contained_questions_are_not(text):
    assert not needs_rewrite(text)


def test_a_long_question_carries_its_own_context():
    long_one = ("given everything we covered about the solar array and its inverter "
                "sizing, what wire gauge should the run from the panels use")
    assert not needs_rewrite(long_one)


def test_rewrite_uses_the_model_when_there_is_history():
    complete = fake_complete("What is the melting point of tungsten?")
    text, done = rewrite("what about its melting point?",
                         [{"role": "user", "content": "tell me about tungsten"}],
                         cfg=object(), complete=complete)
    assert done
    assert text == "What is the melting point of tungsten?"
    assert "tungsten" in complete.calls[0]["user"]


def test_rewrite_strips_a_chatty_model_reply():
    complete = fake_complete('  "What is the melting point of tungsten?"\nHope that helps! ')
    text, done = rewrite("what about its melting point?",
                         [{"role": "user", "content": "tungsten"}],
                         cfg=object(), complete=complete)
    assert text == "What is the melting point of tungsten?"
    assert done


def test_rewrite_is_skipped_without_history_or_without_a_model():
    assert rewrite("what about its melting point?", None) == \
        ("what about its melting point?", False)
    assert rewrite("what about its melting point?",
                   [{"role": "user", "content": "tungsten"}]) == \
        ("what about its melting point?", False)


def test_salient_topic_prefers_a_proper_noun():
    history = [{"role": "user", "content": "tell me about Tungsten please"}]
    assert salient_topic(history) == "Tungsten"


def test_salient_topic_falls_back_to_the_longest_content_word():
    history = [{"role": "user", "content": "how does electroplating actually work"}]
    assert salient_topic(history) == "electroplating"


def test_salient_topic_walks_back_through_history():
    history = [
        {"role": "user", "content": "tell me about Tungsten"},
        {"role": "user", "content": "ok"},
    ]
    assert salient_topic(history) == "Tungsten"
    assert salient_topic([]) == ""


# -- variants --------------------------------------------------------------


def test_keyword_query_drops_stopwords_and_duplicates():
    assert keyword_query("how do I purify the water and the water filter") == \
        "purify water filter"


def test_keyword_query_keeps_identifiers_intact():
    assert "parse_query" in keyword_query("where is the parse_query function")
    assert "module.py" in keyword_query("what is in module.py")


def test_fts_match_query_is_valid_sql_for_awkward_input():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(text)")
    conn.execute("INSERT INTO t(text) VALUES ('boil water to purify it')")
    awkward = [
        'how do I "purify water" safely?',
        "NEAR AND OR NOT",
        "what is C++ vs C#?",
        "don't * me ^ok: 1",
        'search for "unclosed quote',
        "tungsten (W) at 3422 degrees",
        "O'Brien's file: /etc/passwd",
        "café résumé naïve",
        "x" * 300,
    ]
    for question in awkward:
        match = fts_match_query(question)
        assert match, f"no match expression for {question!r}"
        conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (match,)).fetchone()


def test_fts_match_query_is_empty_when_there_is_nothing_to_match():
    assert fts_match_query("a the of and") == ""
    assert fts_match_query("") == ""


def test_fts_match_query_handles_non_ascii_words_whole():
    assert fts_match_query("café ünïcödé") == '"café" OR "ünïcödé"'


def test_variants_cover_the_dense_and_sparse_channels():
    variants = build_variants("how do I purify water in the wild")
    kinds = {v.kind for v in variants}
    assert "question" in kinds
    assert "keyword" in kinds
    assert any(v.channel in ("dense", "both") for v in variants)
    assert any(v.channel in ("sparse", "both") for v in variants)


def test_a_question_that_is_all_content_words_serves_both_channels():
    variants = build_variants("tungsten melting point")
    assert len(variants) == 1
    assert variants[0].channel == "both"


def test_the_topic_variant_stands_in_for_a_rewrite():
    variants = build_variants("what about its melting point?", topic="Tungsten")
    context = [v for v in variants if v.kind == "context"]
    assert context and context[0].text.startswith("Tungsten ")


def test_the_topic_variant_is_skipped_when_already_present():
    variants = build_variants("what is tungsten's melting point", topic="tungsten")
    assert not [v for v in variants if v.kind == "context"]


def test_hyde_adds_an_answer_shaped_variant():
    passage = "Tungsten melts at 3422 degrees Celsius, the highest of any metal."
    variants = build_variants("melting point of tungsten", cfg=object(),
                              complete=fake_complete(passage))
    hyde_variants = [v for v in variants if v.kind == "hyde"]
    assert hyde_variants[0].text == passage
    assert hyde_variants[0].channel == "dense"


def test_hyde_is_absent_without_a_brain():
    assert hyde("anything") == ""
    assert hyde("anything", cfg=object(), complete=fake_complete(None)) == ""
    assert not [v for v in build_variants("melting point of tungsten") if v.kind == "hyde"]


def test_variants_are_capped():
    variants = build_variants("what about its melting point?", topic="Tungsten",
                              cfg=object(), complete=fake_complete("A paragraph."))
    assert len(variants) <= 4


# -- filters ---------------------------------------------------------------


def test_relative_periods_become_a_lower_bound():
    after, before = extract_temporal("what happened last week", NOW)
    assert before is None
    assert NOW - 15 * 86400 < after < NOW


def test_last_n_units():
    after, _ = extract_temporal("anything from the last 3 months", NOW)
    assert after == pytest.approx(NOW - 3 * 31 * 86400)


def test_since_and_before_a_year():
    after, before = extract_temporal("articles since 2023", NOW)
    assert after == calendar.timegm((2023, 1, 1, 0, 0, 0, 0, 1, 0))
    assert before is None

    after, before = extract_temporal("anything before 2020", NOW)
    assert before == calendar.timegm((2020, 1, 1, 0, 0, 0, 0, 1, 0))


def test_in_a_year_bounds_both_ends():
    after, before = extract_temporal("what happened in 1969", NOW)
    assert after == calendar.timegm((1969, 1, 1, 0, 0, 0, 0, 1, 0))
    assert before == calendar.timegm((1970, 1, 1, 0, 0, 0, 0, 1, 0))


def test_a_timeless_question_has_no_bounds():
    assert extract_temporal("how do I purify water", NOW) == (None, None)


def test_entities_pick_up_names_quotes_and_identifiers():
    found = extract_entities('what does "cold soak" mean for the Tungsten Carbide drill')
    assert "cold soak" in found
    assert any("Tungsten" in e for e in found)

    code = extract_entities("why does parse_query raise in module.py")
    assert "parse_query" in code


def test_a_sentence_initial_capital_is_not_an_entity():
    assert extract_entities("Water is safe once boiled") == []


def test_entities_are_capped_and_deduplicated():
    found = extract_entities(" ".join(f"Name{i} Name{i}" for i in range(20)))
    assert len(found) <= 8
    assert len(found) == len(set(found))


def test_extract_filters_combines_both():
    filters = extract_filters('anything about "solar panels" since 2023', NOW)
    assert filters.temporal
    assert "solar panels" in filters.entities
    assert "2023" in filters.describe()


def test_filters_describe_nothing_when_empty():
    assert Filters().describe() == "none"
    assert Filters().temporal is False


# -- whole plans -----------------------------------------------------------


def test_a_skipped_plan_carries_no_variants():
    plan = build_plan("thanks!")
    assert plan.needs_retrieval is False
    assert plan.variants == []
    assert plan.reason == "chitchat"


def test_a_full_plan_has_everything_retrieval_needs():
    plan = build_plan("what did I tell you about my solar panels since 2023", now=NOW)

    assert plan.needs_retrieval
    assert plan.variants
    assert plan.namespace_plan.weight(PERSONAL) > plan.namespace_plan.weight(NEWS)
    assert plan.filters.published_after
    assert plan.texts("sparse")
    assert plan.texts("dense")


def test_plan_routes_namespaces_from_the_question():
    assert build_plan("where is the parse function in the codebase").namespace_plan.routed \
        == [CODE]
    assert NEWS in build_plan("what are the latest headlines").namespace_plan.routed


def test_plan_can_hard_restrict_when_the_caller_asks():
    plan = build_plan("where is the parse function in the codebase",
                      restrict_namespaces=True)
    assert plan.namespace_plan.restrict
    assert not plan.namespace_plan.allows("reference")


def test_plan_rewrites_with_history_and_a_model():
    plan = build_plan(
        "what about its melting point?",
        history=[{"role": "user", "content": "tell me about tungsten"}],
        cfg=object(), complete=fake_complete("What is the melting point of tungsten?"),
    )
    assert plan.rewritten
    assert plan.question == "What is the melting point of tungsten?"
    # A real rewrite makes the blunt topic variant unnecessary.
    assert not [v for v in plan.variants if v.kind == "context"]


def test_plan_falls_back_to_a_topic_variant_with_no_model():
    plan = build_plan("what about its melting point?",
                      history=[{"role": "user", "content": "tell me about Tungsten"}])
    assert plan.rewritten is False
    assert [v for v in plan.variants if v.kind == "context"]


def test_texts_filters_by_channel():
    plan = build_plan("how do I purify water in the wild")
    assert set(plan.texts()) >= set(plan.texts("dense"))
    assert all(isinstance(t, str) and t for t in plan.texts("sparse"))


def test_logging_a_plan_never_raises(caplog):
    build_plan("what did I tell you about the shed").log()
    build_plan("thanks").log()
    assert caplog or True  # the assertion is that neither call raised


def test_plan_survives_junk_input():
    for junk in ["", "   ", "?????", "🔥🔥🔥", "a" * 5000]:
        plan = build_plan(junk)
        assert isinstance(plan, RetrievalPlan)
        if plan.needs_retrieval:
            assert plan.variants
