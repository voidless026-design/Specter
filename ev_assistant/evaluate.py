"""The eval harness - what stops retrieval quality being a matter of opinion.

Four buckets, each aimed at a different way retrieval fails:

- **store**: the answer is in the corpus. Did the right document come back,
  and how high? (recall@5, recall@10, MRR)
- **brain**: the answer is general knowledge the corpus says nothing about.
  Did retrieval *abstain*? This is the bucket that keeps E.V. honest -
  returning the least-bad match here is worse than returning nothing.
- **followup**: a fragment that only makes sense given the previous turn.
  Did rewriting recover the subject?
- **adversarial**: a question whose keywords match a document that does not
  answer it. Did the reranker keep the decoy out of the top three?

Every number is measured at *document* level rather than chunk level: chunk
ids move whenever the chunker changes, and an eval that has to be rewritten
every time you touch chunking gets ignored within a month.

Results are written to a versioned JSON file so two runs can be compared.
Calibrating the relevance floor against this set is the whole point of it
existing - the defaults in rerank.py are placeholders until you do.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_PATH = EVALS_DIR / "golden.json"
RESULTS_DIR = EVALS_DIR / "results"

RECALL_DEPTHS = (5, 10)
ADVERSARIAL_DEPTH = 3     # a decoy in the top 3 is what actually hurts
EVAL_K = 10


@dataclass
class QuestionOutcome:
    """What happened to one question."""

    id: str
    bucket: str
    question: str
    returned: list[str] = field(default_factory=list)   # document stems, in order
    expect: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    rank: int | None = None          # 1-based rank of the first expected doc
    rewritten: bool = False
    timings_ms: dict = field(default_factory=dict)
    best_score: float = 0.0

    @property
    def passed(self) -> bool:
        if self.bucket == "brain":
            return self.abstained
        if self.bucket == "adversarial":
            return self.hit(ADVERSARIAL_DEPTH) and not self.decoyed
        return self.hit(RECALL_DEPTHS[-1])

    @property
    def decoyed(self) -> bool:
        return any(d in self.returned[:ADVERSARIAL_DEPTH] for d in self.avoid)

    def hit(self, depth: int) -> bool:
        return any(d in self.returned[:depth] for d in self.expect)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "bucket": self.bucket, "question": self.question,
            "expect": self.expect, "avoid": self.avoid, "returned": self.returned[:EVAL_K],
            "abstained": self.abstained, "reason": self.reason, "rank": self.rank,
            "rewritten": self.rewritten, "passed": self.passed, "decoyed": self.decoyed,
            "best_score": round(self.best_score, 4),
        }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def summarise(outcomes: list[QuestionOutcome]) -> dict:
    """Turn per-question outcomes into the headline numbers."""
    by_bucket: dict[str, list[QuestionOutcome]] = {}
    for outcome in outcomes:
        by_bucket.setdefault(outcome.bucket, []).append(outcome)

    store = by_bucket.get("store", [])
    brain = by_bucket.get("brain", [])
    followup = by_bucket.get("followup", [])
    adversarial = by_bucket.get("adversarial", [])

    # Abstention is a classification problem: "should have abstained" is the
    # positive class, so precision is "of the times it stayed quiet, how often
    # was that right" and recall is "of the times it should have, how often did
    # it". Precision is the one that matters for trust.
    should_abstain = brain
    did_abstain = [o for o in outcomes if o.abstained]
    correct_abstentions = [o for o in did_abstain if o.bucket == "brain"]
    abstention_precision = (
        len(correct_abstentions) / len(did_abstain) if did_abstain else 1.0
    )
    abstention_recall = (
        len(correct_abstentions) / len(should_abstain) if should_abstain else 1.0
    )

    answerable = store + followup + adversarial
    ranks = [1.0 / o.rank for o in answerable if o.rank]

    stages: dict[str, list[float]] = {}
    for outcome in outcomes:
        for stage, ms in outcome.timings_ms.items():
            stages.setdefault(stage, []).append(ms)
    totals = [sum(o.timings_ms.values()) for o in outcomes if o.timings_ms]

    return {
        "questions": len(outcomes),
        "passed": sum(1 for o in outcomes if o.passed),
        "store": {
            "n": len(store),
            **{f"recall@{d}": _mean([1.0 if o.hit(d) else 0.0 for o in store])
               for d in RECALL_DEPTHS},
            "mrr": _mean([1.0 / o.rank if o.rank else 0.0 for o in store]),
            "abstained_wrongly": sum(1 for o in store if o.abstained),
        },
        "brain": {
            "n": len(brain),
            "abstention_precision": round(abstention_precision, 4),
            "abstention_recall": round(abstention_recall, 4),
            "leaked": [o.id for o in brain if not o.abstained],
        },
        "followup": {
            "n": len(followup),
            **{f"recall@{d}": _mean([1.0 if o.hit(d) else 0.0 for o in followup])
               for d in RECALL_DEPTHS},
            "rewritten": sum(1 for o in followup if o.rewritten),
        },
        "adversarial": {
            "n": len(adversarial),
            f"recall@{ADVERSARIAL_DEPTH}": _mean(
                [1.0 if o.hit(ADVERSARIAL_DEPTH) else 0.0 for o in adversarial]),
            "decoyed": [o.id for o in adversarial if o.decoyed],
        },
        "mrr_overall": _mean(ranks),
        "latency_ms": {
            "total_mean": _mean(totals),
            "total_p50": round(statistics.median(totals), 2) if totals else 0.0,
            "by_stage_mean": {k: _mean(v) for k, v in stages.items()},
        },
    }


def load_golden(path: Path | None = None) -> list[dict]:
    payload = json.loads(Path(path or GOLDEN_PATH).read_text(encoding="utf-8"))
    return payload["questions"]


def build_corpus_store(cfg, store, corpus_dir: Path | None = None) -> int:
    """Ingest the eval corpus into `store`. Returns documents added."""
    from ev_assistant.library import Library

    corpus_dir = Path(corpus_dir or CORPUS_DIR)
    library = Library(cfg, store)
    added = 0
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in (".md", ".txt", ".py"):
            continue
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").title() if path.suffix != ".py" else path.name
        # source_uri carries the stem, which is what the golden set names.
        outcome = library.add(title, text, f"eval://{path.stem}{path.suffix}",
                              source_type="code" if path.suffix == ".py" else "reference")
        if outcome.added:
            added += 1
    library.index()
    return added


def _stem(source_uri: str) -> str:
    """"eval://water-purification.md" -> "water-purification"."""
    tail = (source_uri or "").rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[0] if "." in tail else tail


def run(cfg, store, retriever=None, questions: list[dict] | None = None,
        k: int = EVAL_K, progress=None) -> tuple[list[QuestionOutcome], dict]:
    """Run the golden set. Returns (per-question outcomes, summary)."""
    from ev_assistant.retrieval import Retriever

    retriever = retriever or Retriever(cfg, store)
    questions = questions if questions is not None else load_golden()
    outcomes: list[QuestionOutcome] = []

    for entry in questions:
        started = time.perf_counter()
        result = retriever.retrieve(
            entry["question"], k=k, history=entry.get("history"),
        )
        elapsed = (time.perf_counter() - started) * 1000
        returned: list[str] = []
        for chunk in result.chunks:
            stem = _stem(chunk.document.source_uri if chunk.document else "")
            if stem and stem not in returned:
                returned.append(stem)

        expect = entry.get("expect", [])
        rank = next((i for i, stem in enumerate(returned, start=1) if stem in expect), None)
        timings = dict(result.trace.timings_ms)
        timings.setdefault("wall", round(elapsed, 2))

        outcome = QuestionOutcome(
            id=entry["id"], bucket=entry["bucket"], question=entry["question"],
            returned=returned, expect=expect, avoid=entry.get("avoid", []),
            abstained=not result.chunks, reason=result.reason, rank=rank,
            rewritten=bool(result.plan and result.plan.rewritten),
            timings_ms=timings, best_score=result.trace.best_score,
        )
        outcomes.append(outcome)
        if progress is not None:
            progress(outcome)

    return outcomes, summarise(outcomes)


def save_results(summary: dict, outcomes: list[QuestionOutcome], *, cfg=None,
                 results_dir: Path | None = None, label: str = "") -> Path:
    """Write a versioned result file so regressions show up across changes."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = results_dir / f"{stamp}{'-' + label if label else ''}.json"
    payload = {
        "recorded_at": stamp,
        "label": label,
        "config": _config_snapshot(cfg),
        "summary": summary,
        "questions": [o.as_dict() for o in outcomes],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _config_snapshot(cfg) -> dict:
    """The settings that actually move the numbers, recorded with them."""
    if cfg is None:
        return {}
    keys = ("embedding_backend", "embedding_model", "reranker_backend", "reranker_model",
            "relevance_floor", "rrf_k", "candidates_per_variant", "rerank_top_n",
            "max_chunks_per_document", "neighbor_window", "personal_namespace_boost",
            "other_namespace_weight")
    return {k: getattr(cfg, k, None) for k in keys}


def compare(previous: dict, current: dict) -> list[str]:
    """Human-readable lines describing what moved between two summaries."""
    lines = []

    def delta(path: str, label: str, better_is_higher: bool = True) -> None:
        old, new = previous, current
        for part in path.split("."):
            old = (old or {}).get(part) if isinstance(old, dict) else None
            new = (new or {}).get(part) if isinstance(new, dict) else None
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            return
        change = new - old
        if abs(change) < 1e-9:
            return
        improved = change > 0 if better_is_higher else change < 0
        lines.append(f"{'+' if improved else '-'} {label}: {old:.4g} -> {new:.4g} "
                     f"({change:+.4g})")

    delta("store.recall@5", "store recall@5")
    delta("store.recall@10", "store recall@10")
    delta("store.mrr", "store MRR")
    delta("brain.abstention_precision", "abstention precision")
    delta("brain.abstention_recall", "abstention recall")
    delta("followup.recall@10", "follow-up recall@10")
    delta("adversarial.recall@3", "adversarial recall@3")
    delta("latency_ms.total_p50", "p50 latency (ms)", better_is_higher=False)
    return lines


def format_report(summary: dict, *, targets: dict | None = None) -> str:
    """The text `ev eval` prints."""
    targets = targets or {
        "store.recall@10": 0.85,
        "brain.abstention_precision": 0.9,
        "latency_ms.total_p50": 400.0,
    }

    def get(path: str):
        value = summary
        for part in path.split("."):
            value = (value or {}).get(part) if isinstance(value, dict) else None
        return value

    def mark(path: str, higher_is_better: bool = True) -> str:
        target = targets.get(path)
        value = get(path)
        if target is None or not isinstance(value, (int, float)):
            return " "
        ok = value >= target if higher_is_better else value <= target
        return "PASS" if ok else "FAIL"

    store, brain = summary["store"], summary["brain"]
    follow, adv = summary["followup"], summary["adversarial"]
    latency = summary["latency_ms"]

    lines = [
        f"{summary['questions']} questions, {summary['passed']} passed",
        "",
        f"Store-answerable ({store['n']}):",
        f"  recall@5   {store['recall@5']:.3f}",
        f"  recall@10  {store['recall@10']:.3f}   target >= 0.85  [{mark('store.recall@10')}]",
        f"  MRR        {store['mrr']:.3f}",
    ]
    if store["abstained_wrongly"]:
        lines.append(f"  !! abstained on {store['abstained_wrongly']} answerable question(s)")
    lines += [
        "",
        f"Brain-only ({brain['n']}):",
        f"  abstention precision  {brain['abstention_precision']:.3f}"
        f"   target >= 0.9  [{mark('brain.abstention_precision')}]",
        f"  abstention recall     {brain['abstention_recall']:.3f}",
    ]
    if brain["leaked"]:
        lines.append(f"  !! returned chunks for: {', '.join(brain['leaked'][:6])}")
    lines += [
        "",
        f"Conversational follow-ups ({follow['n']}):",
        f"  recall@10  {follow['recall@10']:.3f}"
        f"   ({follow['rewritten']}/{follow['n']} rewritten by a model)",
        "",
        f"Adversarial ({adv['n']}):",
        f"  recall@3   {adv['recall@3']:.3f}",
    ]
    if adv["decoyed"]:
        lines.append(f"  !! decoy in the top 3 for: {', '.join(adv['decoyed'])}")
    lines += [
        "",
        "Latency:",
        f"  p50 total  {latency['total_p50']:.1f} ms"
        f"   target <= 400  [{mark('latency_ms.total_p50', higher_is_better=False)}]",
        f"  mean total {latency['total_mean']:.1f} ms",
        "  by stage:  " + ", ".join(
            f"{k}={v:.1f}" for k, v in sorted(
                latency["by_stage_mean"].items(), key=lambda kv: -kv[1])),
    ]
    return "\n".join(lines)
