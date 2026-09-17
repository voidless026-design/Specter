"""Phase 10 of the retrieval spec: the eval harness itself.

These test the harness, not retrieval quality - the quality numbers come
from running `ev eval`. A metric that is computed wrongly is worse than no
metric, so the arithmetic gets the same scrutiny as the code it measures.
"""

from __future__ import annotations

import json

import pytest

from ev_assistant.evaluate import (
    ADVERSARIAL_DEPTH,
    CORPUS_DIR,
    GOLDEN_PATH,
    QuestionOutcome,
    _stem,
    build_corpus_store,
    compare,
    format_report,
    load_golden,
    run,
    save_results,
    summarise,
)
from ev_assistant.store import vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")


def outcome(bucket, *, returned=(), expect=(), avoid=(), abstained=False, rank=None,
            oid="x", rewritten=False, timings=None) -> QuestionOutcome:
    return QuestionOutcome(
        id=oid, bucket=bucket, question="q", returned=list(returned), expect=list(expect),
        avoid=list(avoid), abstained=abstained, rank=rank, rewritten=rewritten,
        timings_ms=timings or {},
    )


# -- the golden set itself -------------------------------------------------


def test_the_golden_set_is_well_formed():
    questions = load_golden()

    assert len(questions) >= 60, "the spec asks for at least 60 questions"
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in questions:
        assert q["bucket"] in {"store", "brain", "followup", "adversarial"}
        assert q["question"].strip()


def test_every_bucket_is_represented():
    buckets = {q["bucket"] for q in load_golden()}
    assert buckets == {"store", "brain", "followup", "adversarial"}


def test_answerable_questions_name_a_document_that_exists():
    stems = {p.stem for p in CORPUS_DIR.iterdir() if p.is_file()}
    for q in load_golden():
        for name in q.get("expect", []) + q.get("avoid", []):
            assert name in stems, f"{q['id']} names a missing corpus document: {name}"


def test_brain_questions_claim_no_document():
    # If a brain-only question expected a document it would be a store question.
    for q in load_golden():
        if q["bucket"] == "brain":
            assert not q.get("expect")


def test_followups_carry_the_history_they_depend_on():
    for q in load_golden():
        if q["bucket"] == "followup":
            assert q.get("history"), f"{q['id']} has no history to resolve against"
            assert q.get("expect")


def test_adversarial_questions_name_a_decoy():
    for q in load_golden():
        if q["bucket"] == "adversarial":
            assert q.get("avoid"), f"{q['id']} has no decoy, so it tests nothing"
            assert q.get("expect")
            assert not set(q["expect"]) & set(q["avoid"])


def test_the_corpus_is_substantial():
    files = [p for p in CORPUS_DIR.iterdir() if p.is_file()]
    assert len(files) >= 15
    assert sum(len(p.read_text(encoding="utf-8").split()) for p in files) > 2000


def test_the_golden_file_is_valid_json():
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["version"] >= 1
    assert payload["description"]


# -- per-question scoring --------------------------------------------------


def test_hit_respects_depth():
    o = outcome("store", returned=["a", "b", "c", "d", "e", "f"], expect=["f"])
    assert not o.hit(5)
    assert o.hit(10)


def test_a_store_question_passes_only_on_a_hit():
    assert outcome("store", returned=["a"], expect=["a"]).passed
    assert not outcome("store", returned=["b"], expect=["a"]).passed
    assert not outcome("store", returned=[], expect=["a"], abstained=True).passed


def test_a_brain_question_passes_only_by_abstaining():
    assert outcome("brain", abstained=True).passed
    assert not outcome("brain", returned=["a"]).passed


def test_an_adversarial_question_needs_the_answer_and_no_decoy():
    assert outcome("adversarial", returned=["right"], expect=["right"],
                   avoid=["wrong"]).passed
    # The answer is there, but so is the decoy.
    assert not outcome("adversarial", returned=["right", "wrong"], expect=["right"],
                       avoid=["wrong"]).passed
    # The decoy is deep enough not to matter.
    deep = ["right", "a", "b", "wrong"]
    assert outcome("adversarial", returned=deep, expect=["right"], avoid=["wrong"]).passed


def test_decoyed_only_counts_the_top_of_the_list():
    padding = ["x"] * ADVERSARIAL_DEPTH
    assert not outcome("adversarial", returned=padding + ["wrong"], avoid=["wrong"]).decoyed


# -- summary arithmetic ----------------------------------------------------


def test_recall_and_mrr():
    outcomes = [
        outcome("store", returned=["a"], expect=["a"], rank=1),
        outcome("store", returned=["x", "y", "a"], expect=["a"], rank=3),
        outcome("store", returned=["x"], expect=["a"]),
    ]
    summary = summarise(outcomes)

    assert summary["store"]["recall@5"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["store"]["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-3)


def test_abstention_precision_punishes_abstaining_on_answerable_questions():
    # Two correct abstentions, one wrong one: precision 2/3.
    outcomes = [
        outcome("brain", abstained=True),
        outcome("brain", abstained=True),
        outcome("store", expect=["a"], abstained=True),
    ]
    summary = summarise(outcomes)

    assert summary["brain"]["abstention_precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["brain"]["abstention_recall"] == 1.0
    assert summary["store"]["abstained_wrongly"] == 1


def test_abstention_recall_counts_the_ones_it_should_have_caught():
    outcomes = [
        outcome("brain", abstained=True),
        outcome("brain", returned=["a"]),
        outcome("brain", returned=["b"]),
    ]
    summary = summarise(outcomes)

    assert summary["brain"]["abstention_recall"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["brain"]["abstention_precision"] == 1.0
    assert len(summary["brain"]["leaked"]) == 2


def test_a_perfect_run_scores_perfectly():
    outcomes = [
        outcome("store", returned=["a"], expect=["a"], rank=1),
        outcome("brain", abstained=True),
        outcome("followup", returned=["b"], expect=["b"], rank=1, rewritten=True),
        outcome("adversarial", returned=["c"], expect=["c"], avoid=["d"], rank=1),
    ]
    summary = summarise(outcomes)

    assert summary["passed"] == 4
    assert summary["store"]["recall@10"] == 1.0
    assert summary["brain"]["abstention_precision"] == 1.0
    assert summary["adversarial"]["decoyed"] == []
    assert summary["followup"]["rewritten"] == 1
    assert summary["mrr_overall"] == 1.0


def test_summarising_nothing_does_not_divide_by_zero():
    summary = summarise([])
    assert summary["questions"] == 0
    assert summary["store"]["recall@5"] == 0.0
    assert summary["brain"]["abstention_precision"] == 1.0
    assert summary["latency_ms"]["total_p50"] == 0.0


def test_latency_is_averaged_per_stage():
    outcomes = [
        outcome("store", timings={"search": 10.0, "rerank": 20.0}),
        outcome("store", timings={"search": 20.0, "rerank": 40.0}),
    ]
    latency = summarise(outcomes)["latency_ms"]

    assert latency["by_stage_mean"]["search"] == pytest.approx(15.0)
    assert latency["by_stage_mean"]["rerank"] == pytest.approx(30.0)
    assert latency["total_mean"] == pytest.approx(45.0)


# -- reporting and comparison ----------------------------------------------


def test_the_report_marks_the_definition_of_done_thresholds():
    good = summarise([outcome("store", returned=["a"], expect=["a"], rank=1, timings={"a": 1.0}),
                      outcome("brain", abstained=True, timings={"a": 1.0})])
    report = format_report(good)

    assert "target >= 0.85" in report
    assert "target >= 0.9" in report
    assert "target <= 400" in report
    assert "[PASS]" in report
    assert "[FAIL]" not in report


def test_the_report_fails_loudly_when_a_threshold_is_missed():
    bad = summarise([outcome("store", returned=["x"], expect=["a"], timings={"a": 1.0}),
                     outcome("brain", returned=["x"], timings={"a": 1.0})])
    report = format_report(bad)

    assert "[FAIL]" in report
    assert "returned chunks for" in report


def test_the_report_flags_wrong_abstentions_and_decoys():
    outcomes = [
        outcome("store", expect=["a"], abstained=True, oid="s1"),
        outcome("adversarial", returned=["a", "bad"], expect=["a"], avoid=["bad"], oid="a1"),
    ]
    report = format_report(summarise(outcomes))

    assert "abstained on 1 answerable" in report
    assert "decoy in the top 3 for: a1" in report


def test_compare_describes_what_moved():
    before = summarise([outcome("store", returned=["x"], expect=["a"], timings={"t": 10.0})])
    after = summarise([outcome("store", returned=["a"], expect=["a"], rank=1,
                               timings={"t": 5.0})])

    lines = compare(before, after)

    assert any("store recall@10" in line and line.startswith("+") for line in lines)
    assert any("p50 latency" in line and line.startswith("+") for line in lines)


def test_compare_on_identical_runs_says_nothing():
    summary = summarise([outcome("store", returned=["a"], expect=["a"], rank=1)])
    assert compare(summary, summary) == []


def test_compare_marks_a_regression_as_a_regression():
    before = summarise([outcome("store", returned=["a"], expect=["a"], rank=1)])
    after = summarise([outcome("store", returned=["x"], expect=["a"])])
    assert any(line.startswith("-") for line in compare(before, after))


# -- results files ---------------------------------------------------------


def test_results_are_saved_with_the_settings_that_produced_them(cfg, tmp_path):
    outcomes = [outcome("store", returned=["a"], expect=["a"], rank=1, oid="s1")]
    path = save_results(summarise(outcomes), outcomes, cfg=cfg, results_dir=tmp_path,
                        label="baseline")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["label"] == "baseline"
    assert payload["config"]["reranker_backend"] == cfg.reranker_backend
    assert payload["config"]["relevance_floor"] == cfg.relevance_floor
    assert payload["questions"][0]["id"] == "s1"
    assert "summary" in payload
    assert "baseline" in path.name


def test_saved_results_can_be_compared_later(cfg, tmp_path):
    first = [outcome("store", returned=["x"], expect=["a"])]
    second = [outcome("store", returned=["a"], expect=["a"], rank=1)]
    path = save_results(summarise(first), first, cfg=cfg, results_dir=tmp_path)

    previous = json.loads(path.read_text(encoding="utf-8"))
    assert compare(previous["summary"], summarise(second))


# -- the real thing --------------------------------------------------------


def test_stem_extracts_the_document_name():
    assert _stem("eval://water-purification.md") == "water-purification"
    assert _stem("eval://parser.py") == "parser"
    assert _stem("") == ""


@needs_vec
def test_the_corpus_ingests_and_the_harness_runs(cfg, store):
    cfg.reranker_backend = "lexical"
    documents = build_corpus_store(cfg, store)

    assert documents >= 15
    stats = store.stats()
    assert stats["chunks"] > documents
    assert stats["pending_chunks"] == 0      # everything got embedded

    sample = [q for q in load_golden() if q["bucket"] in ("store", "brain")][:8]
    outcomes, summary = run(cfg, store, questions=sample)

    assert len(outcomes) == len(sample)
    assert summary["questions"] == len(sample)
    assert all(o.timings_ms for o in outcomes)
    assert format_report(summary)


@needs_vec
def test_a_store_answerable_question_finds_its_document(cfg, store):
    cfg.reranker_backend = "lexical"
    build_corpus_store(cfg, store)

    question = next(q for q in load_golden() if q["id"] == "store-001")
    outcomes, _ = run(cfg, store, questions=[question])

    assert outcomes[0].returned[:1] == ["water-purification"]
    assert outcomes[0].rank == 1
    assert outcomes[0].passed


@needs_vec
def test_a_brain_only_question_abstains(cfg, store):
    cfg.reranker_backend = "lexical"
    build_corpus_store(cfg, store)

    question = next(q for q in load_golden() if q["id"] == "brain-001")
    outcomes, _ = run(cfg, store, questions=[question])

    assert outcomes[0].abstained
    assert outcomes[0].reason == "below_threshold"
    assert outcomes[0].passed
