"""Phase 4 of the retrieval spec: namespaces, routing, and soft filtering."""

from __future__ import annotations

import pytest

from ev_assistant.namespaces import (
    CODE,
    DEFAULT_NAMESPACE,
    NEWS,
    PERSONAL,
    REFERENCE,
    NamespacePlan,
    detect,
    domain,
    is_valid,
    looks_like_code,
    normalize,
    plan,
    plan_for_query,
    route,
)


# -- naming ----------------------------------------------------------------


def test_normalize_canonicalises():
    assert normalize("  Personal ") == PERSONAL
    assert normalize("Domain:Organic Chem") == "domain:organic-chem"


def test_normalize_falls_back_on_blank_or_malformed():
    assert normalize("") == DEFAULT_NAMESPACE
    assert normalize(None) == DEFAULT_NAMESPACE
    assert normalize("has/slash") == DEFAULT_NAMESPACE
    assert normalize("9leading-digit") == DEFAULT_NAMESPACE


def test_is_valid():
    assert is_valid("personal")
    assert is_valid("domain:chem-101")
    assert not is_valid("domain:")
    assert not is_valid("has space")
    assert not is_valid("")


def test_domain_builds_a_slug():
    assert domain("Organic Chemistry") == "domain:organic-chemistry"
    assert domain("  Chem 101!  ") == "domain:chem-101"
    with pytest.raises(ValueError):
        domain("   ")


# -- ingest-time routing ---------------------------------------------------


@pytest.mark.parametrize("source_type,expected", [
    ("wikipedia", REFERENCE),
    ("web", REFERENCE),
    ("pdf", REFERENCE),
    ("feed", NEWS),
    ("conversation", PERSONAL),
    ("voice", PERSONAL),
    ("file", PERSONAL),
    ("code", CODE),
    ("something-unheard-of", DEFAULT_NAMESPACE),
    ("", DEFAULT_NAMESPACE),
])
def test_route_by_source_type(source_type, expected):
    assert route(source_type) == expected


def test_source_shape_beats_source_type():
    # A .py file fetched over the web is still code.
    assert route("web", "https://example.org/thing.py") == CODE
    assert route("web", "https://github.com/me/repo") == CODE
    assert route("file", "/home/me/project/main.rs") == CODE


def test_looks_like_code():
    assert looks_like_code("/home/me/x.py")
    assert looks_like_code("https://github.com/me/repo/blob/main/a.go?raw=1")
    assert not looks_like_code("/home/me/notes.md")
    assert not looks_like_code("")


def test_config_rules_win_over_everything():
    rules = {"wikipedia.org": "domain:chem", "type:feed": "personal"}
    assert route("web", "https://en.wikipedia.org/wiki/Tungsten", rules) == "domain:chem"
    assert route("feed", "https://hnrss.org/frontpage", rules) == PERSONAL
    # A rule that doesn't match leaves the defaults alone.
    assert route("wikipedia", "wikipedia:Water", {"example.com": "code"}) == REFERENCE


def test_a_human_typed_rule_target_is_slugified():
    # Someone writing `"example.org" = "my stuff"` in config means it.
    assert route("web", "https://example.org/x", {"example.org": "My Stuff"}) == "my-stuff"


def test_an_unusable_rule_target_falls_back_rather_than_poisoning_the_store():
    assert route("web", "https://example.org/x", {"example.org": "///"}) == DEFAULT_NAMESPACE


def test_routed_documents_round_trip_through_the_store(store):
    for source_type, uri, text in [
        ("wikipedia", "wikipedia:Tungsten", "tungsten is a metal"),
        ("feed", "https://hnrss.org/frontpage", "todays headlines"),
        ("file", "/home/me/main.py", "def main(): pass"),
        ("conversation", "conversation", "you told me your name"),
    ]:
        store.add_document(
            source_uri=uri, source_type=source_type, title="T", text=text,
            namespace=route(source_type, uri), chunks=[{"text": text, "ordinal": 0}],
        )
    assert store.stats()["by_namespace"] == {REFERENCE: 1, NEWS: 1, CODE: 1, PERSONAL: 1}


# -- query hints -----------------------------------------------------------


@pytest.mark.parametrize("question,expected", [
    ("what did I tell you about my laptop", PERSONAL),
    ("check my notes on the shed", PERSONAL),
    ("where is the parse function in the codebase", CODE),
    ("why does this traceback happen", CODE),
    ("what are the latest headlines", NEWS),
    ("what happened this week", NEWS),
    ("what is the history of tungsten according to wikipedia", REFERENCE),
])
def test_detect_finds_the_cue(question, expected):
    assert expected in detect(question)


def test_detect_returns_nothing_for_a_plain_question():
    assert detect("how do I purify water") == []
    assert detect("") == []


def test_detect_orders_by_where_the_cue_appears():
    found = detect("in my notes, what are the latest headlines")
    assert found[0] == PERSONAL
    assert NEWS in found


# -- planning --------------------------------------------------------------


def test_personal_is_always_boosted_even_when_the_query_routed_elsewhere():
    p = plan([CODE])
    assert p.weight(PERSONAL) > p.weight(CODE)
    assert p.weight(CODE) > p.weight(REFERENCE)


def test_unrouted_namespaces_are_demoted_not_excluded():
    p = plan([CODE])
    assert p.allows(REFERENCE)
    assert 0 < p.weight(REFERENCE) < p.weight(CODE)
    assert p.sql_filter() == ("", [])


def test_a_plan_with_no_routing_still_favours_personal():
    p = plan()
    assert p.routed == []
    assert p.weight(PERSONAL) > p.weight(REFERENCE)
    assert p.allows("domain:anything")


def test_restrict_hard_filters_only_when_asked():
    soft = plan([CODE])
    assert soft.allowed is None

    hard = plan([CODE], restrict=True)
    assert hard.restrict
    assert hard.allows(CODE)
    assert not hard.allows(REFERENCE)


def test_restriction_never_locks_personal_out():
    # "personal is always searched" outranks the caller's filter.
    hard = plan([CODE], restrict=True)
    assert hard.allows(PERSONAL)
    clause, params = hard.sql_filter()
    assert clause == "d.namespace IN (?,?)"
    assert set(params) == {CODE, PERSONAL}


def test_the_eval_harness_can_opt_out_of_the_personal_boost():
    p = plan([CODE], restrict=True, include_personal=False)
    assert not p.allows(PERSONAL)
    assert p.weight(PERSONAL) == p.weight(REFERENCE)


def test_restrict_with_nothing_routed_is_not_a_filter():
    # Restricting to "no namespaces" would find nothing at all.
    p = plan([], restrict=True)
    assert p.restrict is False
    assert p.allowed is None


def test_plan_deduplicates_and_drops_junk():
    p = plan([CODE, "CODE", "not a namespace", CODE])
    assert p.routed == [CODE]


def test_plan_reads_weights_from_config(cfg):
    cfg.personal_namespace_boost = 3.0
    cfg.routed_namespace_boost = 2.0
    cfg.other_namespace_weight = 0.1
    p = plan([CODE], cfg=cfg)
    assert p.weight(PERSONAL) == 3.0
    assert p.weight(CODE) == 2.0
    assert p.weight(REFERENCE) == 0.1


def test_plan_for_query_wires_detection_to_planning():
    p = plan_for_query("where is the parse function in the codebase")
    assert p.routed == [CODE]
    assert p.weight(CODE) > p.weight(REFERENCE)


def test_sql_filter_targets_the_given_column():
    clause, params = plan([NEWS], restrict=True).sql_filter("documents.namespace")
    assert clause.startswith("documents.namespace IN (")
    assert set(params) == {NEWS, PERSONAL}


def test_plan_describes_itself_for_the_log():
    assert "boosting" in plan([CODE]).describe()
    assert "restricted to" in plan([CODE], restrict=True).describe()
    assert "nothing in particular" in plan().describe()


def test_weight_normalises_its_argument():
    p = plan([CODE])
    assert p.weight("  CODE  ") == p.weight(CODE)
    assert p.weight(None) == p.weight(REFERENCE)


def test_namespace_plan_is_usable_standalone():
    p = NamespacePlan(routed=[CODE], weights={CODE: 2.0}, other_weight=0.5)
    assert p.weight(CODE) == 2.0
    assert p.weight(REFERENCE) == 0.5
    assert p.allows(REFERENCE)
