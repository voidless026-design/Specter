"""Hybrid retrieval - the core of the knowledge layer.

One question goes in; the chunks that actually answer it come out, or nothing
does. The pipeline:

1. Plan the query (query.py) - and skip the store entirely if it doesn't
   need one.
2. Generate candidates over every query variant, in parallel:
   dense vector KNN, sparse FTS5/BM25, and a direct lookup in `facts` on
   any entities the question named.
3. Fuse with Reciprocal Rank Fusion. RRF needs no score normalisation, which
   matters because BM25 and cosine distance live on unrelated scales and
   normalising between them is where hybrid search usually goes wrong.
4. Rerank the shortlist with a cross-encoder. Biggest quality win here.
5. Apply the relevance floor. Below it, return nothing - an empty result is
   a correct answer, and it's the only thing that stops E.V. answering a
   question about tungsten out of a document about knots.
6. Cap chunks per document, so one long article can't take the whole budget.
7. Optionally pull neighbouring chunks back in to restore continuity.

Namespace weights deliberately act on *ordering*, never on *admission*: the
boost decides which candidates get reranked and how survivors are sorted, but
the floor is tested against the raw rerank score. Otherwise a 1.25x personal
boost could push an irrelevant chunk past the threshold, which is exactly the
failure the floor exists to prevent.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ev_assistant import query as query_module
from ev_assistant import rerank as rerank_module
from ev_assistant.namespaces import NamespacePlan
from ev_assistant.query import RetrievalPlan
from ev_assistant.store import Chunk, Fact, Store

logger = logging.getLogger(__name__)

RRF_K = 60                  # the standard constant; larger flattens rank influence
CANDIDATES_PER_VARIANT = 50
RERANK_TOP_N = 50
DEFAULT_K = 8
MAX_PER_DOCUMENT = 3
NEIGHBOR_WINDOW = 0
# sqlite-vec computes top-k *before* any join filter, so a filtered search has
# to over-fetch or it comes back nearly empty. Verified, not assumed.
FILTER_OVERFETCH = 8

# Why a result is empty. These reach the brain, so they have to be honest.
OK = "ok"
SKIPPED = "skipped"
EMPTY_STORE = "empty_store"
NO_CANDIDATES = "no_candidates"
BELOW_THRESHOLD = "below_threshold"


@dataclass
class Candidate:
    """One chunk in flight through the pipeline, with its evidence."""

    chunk_id: int
    rrf: float = 0.0
    dense: float = 0.0          # best cosine similarity across variants
    sparse: float = 0.0         # best BM25 (already sign-flipped: higher better)
    rerank: float = 0.0
    score: float = 0.0          # final, namespace-weighted
    namespace: str = ""
    document_id: int = 0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return sorted({key.split(":")[0] for key in self.ranks})


@dataclass
class RetrievalTrace:
    """Per-stage counts and timings, for `ev retrieve --explain` and logs."""

    timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    floor: float = 0.0
    reranker: str = ""
    embedder: str = ""
    dropped_below_floor: int = 0
    best_score: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)

    def stage(self, name: str, started: float) -> None:
        self.timings_ms[name] = round((time.perf_counter() - started) * 1000, 2)

    def describe(self) -> str:
        stages = " ".join(f"{k}={v}ms" for k, v in self.timings_ms.items())
        return f"{stages} total={self.total_ms}ms counts={self.counts}"


@dataclass
class RetrievalResult:
    """What retrieval found, and enough context to explain itself."""

    chunks: list[Chunk] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    plan: RetrievalPlan | None = None
    reason: str = OK
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)

    def __bool__(self) -> bool:
        return bool(self.chunks or self.facts)

    @property
    def empty(self) -> bool:
        return not bool(self)

    def explain(self) -> str:
        if self.reason == SKIPPED:
            return "Retrieval skipped: this question doesn't need the store."
        if self.empty:
            return f"Nothing retrieved ({self.reason}). {self.trace.describe()}"
        return (f"{len(self.chunks)} chunks, {len(self.facts)} facts, "
                f"best={self.trace.best_score:.3f}, floor={self.trace.floor:.3f}. "
                f"{self.trace.describe()}")


def reciprocal_rank_fusion(ranked_lists: dict[str, list[int]], k: int = RRF_K
                           ) -> dict[int, tuple[float, dict[str, int]]]:
    """Fuse ranked id lists into {id: (score, {list_name: rank})}.

    Each list contributes 1/(k + rank). No normalisation, which is the point:
    BM25 scores and cosine distances can't be compared directly, but their
    ranks can.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in ranked_lists.items():
        for position, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            seen = ranks.setdefault(chunk_id, {})
            if name not in seen or position < seen[name]:
                seen[name] = position
    return {cid: (scores[cid], ranks[cid]) for cid in scores}


class Retriever:
    def __init__(self, cfg, store: Store, embedder=None, reranker=None, complete=None):
        self.cfg = cfg
        self.store = store
        self._embedder = embedder
        self._reranker = reranker
        if complete is None:
            from ev_assistant import providers
            complete = providers.complete
        self.complete = complete
        self.per_variant = int(getattr(cfg, "candidates_per_variant", CANDIDATES_PER_VARIANT))
        self.rerank_top_n = int(getattr(cfg, "rerank_top_n", RERANK_TOP_N))
        self.max_per_document = int(getattr(cfg, "max_chunks_per_document", MAX_PER_DOCUMENT))
        self.neighbor_window = int(getattr(cfg, "neighbor_window", NEIGHBOR_WINDOW))
        self.rrf_k = int(getattr(cfg, "rrf_k", RRF_K))

    # -- lazily built collaborators ------------------------------------

    @property
    def embedder(self):
        if self._embedder is None:
            from ev_assistant.embeddings import Embedder
            self._embedder = Embedder(self.cfg, self.store)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = rerank_module.build_reranker(self.cfg)
        return self._reranker

    # -- candidate channels --------------------------------------------

    def _dense(self, blob: bytes, limit: int, plan: NamespacePlan, filters) -> list[tuple[int, float]]:
        """Vector KNN as [(chunk_id, cosine_similarity)], best first."""
        if not self.store.vec_available:
            return []
        clause, params = self._filter_sql(plan, filters)
        # Over-fetch whenever a filter is in play: vec0 picks its top-k first
        # and the join throws rows away afterwards, so asking for exactly k
        # can come back empty.
        want = limit * FILTER_OVERFETCH if clause else limit
        sql = (
            "SELECT v.chunk_id AS chunk_id, v.distance AS distance"
            " FROM chunks_vec v"
            " JOIN chunks c ON c.id = v.chunk_id"
            " JOIN documents d ON d.id = c.document_id"
            " WHERE v.embedding MATCH ? AND k = ?"
            + (f" AND {clause}" if clause else "")
            + " ORDER BY v.distance"
        )
        try:
            with self.store._connect() as conn:
                rows = conn.execute(sql, [blob, want, *params]).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("Vector search unavailable: %s", e)
            return []
        # Vectors are unit length, so L2 and cosine agree: sim = 1 - d^2/2.
        return [(r["chunk_id"], max(0.0, 1.0 - (r["distance"] ** 2) / 2.0))
                for r in rows[:limit]]

    def _sparse(self, match: str, limit: int, plan: NamespacePlan, filters
                ) -> list[tuple[int, float]]:
        """BM25 as [(chunk_id, score)], best first. bm25() is negative-better."""
        if not match:
            return []
        clause, params = self._filter_sql(plan, filters)
        sql = (
            "SELECT f.rowid AS chunk_id, bm25(chunks_fts) AS score"
            " FROM chunks_fts f"
            " JOIN chunks c ON c.id = f.rowid"
            " JOIN documents d ON d.id = c.document_id"
            " WHERE chunks_fts MATCH ?"
            + (f" AND {clause}" if clause else "")
            + " ORDER BY score LIMIT ?"
        )
        try:
            with self.store._connect() as conn:
                rows = conn.execute(sql, [match, *params, limit]).fetchall()
        except sqlite3.OperationalError as e:
            # A malformed MATCH shouldn't take the whole query down.
            logger.warning("Keyword search failed for %r: %s", match[:80], e)
            return []
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    def _facts(self, entities: list[str], limit: int = 12) -> list[Fact]:
        """Facts whose subject or statement names one of the query's entities."""
        if not entities:
            return []
        clauses = " OR ".join(["subject LIKE ? OR statement LIKE ?"] * len(entities))
        params: list[str] = []
        for entity in entities:
            like = f"%{entity}%"
            params.extend([like, like])
        with self.store._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM facts WHERE {clauses} ORDER BY confidence DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [Fact(id=r["id"], document_id=r["document_id"], statement=r["statement"],
                     subject=r["subject"], confidence=r["confidence"],
                     asserted_at=r["asserted_at"]) for r in rows]

    @staticmethod
    def _filter_sql(plan: NamespacePlan, filters) -> tuple[str, list]:
        """WHERE fragment for namespace restriction and published_at bounds."""
        clauses: list[str] = []
        params: list = []
        namespace_clause, namespace_params = plan.sql_filter("d.namespace")
        if namespace_clause:
            clauses.append(namespace_clause)
            params.extend(namespace_params)
        if filters is not None:
            if filters.published_after is not None:
                clauses.append("(d.published_at IS NULL OR d.published_at >= ?)")
                params.append(filters.published_after)
            if filters.published_before is not None:
                clauses.append("(d.published_at IS NULL OR d.published_at <= ?)")
                params.append(filters.published_before)
        return " AND ".join(clauses), params

    # -- the pipeline --------------------------------------------------

    def retrieve(
        self,
        text: str,
        k: int = DEFAULT_K,
        namespaces: list[str] | None = None,
        filters=None,
        history: list[dict] | None = None,
        *,
        restrict_namespaces: bool = False,
        plan: RetrievalPlan | None = None,
    ) -> RetrievalResult:
        trace = RetrievalTrace()
        started = time.perf_counter()

        if plan is None:
            plan = query_module.build_plan(
                text, history, self.cfg, self.complete,
                restrict_namespaces=restrict_namespaces,
            )
        trace.stage("plan", started)

        if not plan.needs_retrieval:
            return RetrievalResult(plan=plan, reason=SKIPPED, trace=trace)

        namespace_plan = plan.namespace_plan
        if namespaces:
            from ev_assistant import namespaces as ns_module
            namespace_plan = ns_module.plan(namespaces, restrict=restrict_namespaces,
                                            cfg=self.cfg)
        active_filters = filters if filters is not None else plan.filters

        if self.store.stats()["chunks"] == 0:
            return RetrievalResult(plan=plan, reason=EMPTY_STORE, trace=trace)

        candidates = self._gather(plan, namespace_plan, active_filters, trace)
        if not candidates:
            return RetrievalResult(plan=plan, reason=NO_CANDIDATES, trace=trace,
                                   facts=self._facts(active_filters.entities))

        survivors, floor = self._rank(plan, candidates, namespace_plan, trace)
        trace.floor = floor
        trace.candidates = survivors[:self.rerank_top_n]

        if not survivors:
            # Deliberate: no chunks, but any fact that matched an entity is
            # still worth handing over - those are exact matches, not guesses.
            logger.debug(
                "Retrieval %r: nothing cleared the floor (best %.3f < %.3f, %d dropped) %s",
                plan.question, trace.best_score, floor, trace.dropped_below_floor,
                trace.describe(),
            )
            return RetrievalResult(plan=plan, reason=BELOW_THRESHOLD, trace=trace,
                                   facts=self._facts(active_filters.entities))

        chunks = self._materialise(survivors, k, trace)
        facts = self._facts(active_filters.entities)
        trace.counts["returned"] = len(chunks)
        trace.counts["facts"] = len(facts)

        result = RetrievalResult(chunks=chunks, facts=facts, plan=plan, reason=OK, trace=trace)
        # Per-stage timings at debug level, so a slow query can be diagnosed
        # from the daemon's log without reproducing it under `ev retrieve`.
        logger.debug("Retrieval %r: %s", plan.question, result.explain())
        return result

    # -- stage 2: candidates -------------------------------------------

    def _gather(self, plan: RetrievalPlan, namespace_plan: NamespacePlan, filters,
                trace: RetrievalTrace) -> dict[int, Candidate]:
        started = time.perf_counter()
        dense_texts = plan.texts("dense")
        sparse_texts = plan.texts("sparse")

        # One batched embed beats N parallel single embeds by a mile.
        blobs: list[bytes] = []
        if dense_texts and self.store.vec_available:
            try:
                blobs = self.embedder.embed_queries(dense_texts)
            except Exception as e:
                logger.warning("Couldn't embed the query, falling back to keyword only: %s", e)
                blobs = []
        trace.stage("embed", started)

        started = time.perf_counter()
        jobs = []
        for i, blob in enumerate(blobs):
            jobs.append((f"dense:{i}", "dense", blob))
        for i, text in enumerate(sparse_texts):
            jobs.append((f"sparse:{i}", "sparse", query_module.fts_match_query(text)))

        ranked: dict[str, list[int]] = {}
        similarity: dict[int, float] = {}
        bm25: dict[int, float] = {}

        def run(job):
            name, kind, payload = job
            if kind == "dense":
                return name, kind, self._dense(payload, self.per_variant, namespace_plan, filters)
            return name, kind, self._sparse(payload, self.per_variant, namespace_plan, filters)

        # WAL means readers don't block each other, so the channels really do
        # run concurrently rather than queueing behind one connection.
        if jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
                for name, kind, hits in pool.map(run, jobs):
                    ranked[name] = [cid for cid, _ in hits]
                    for cid, value in hits:
                        target = similarity if kind == "dense" else bm25
                        target[cid] = max(target.get(cid, 0.0), value)
        trace.stage("search", started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(ranked, self.rrf_k)
        candidates: dict[int, Candidate] = {}
        for chunk_id, (score, ranks) in fused.items():
            candidates[chunk_id] = Candidate(
                chunk_id=chunk_id, rrf=score, ranks=ranks,
                dense=similarity.get(chunk_id, 0.0), sparse=bm25.get(chunk_id, 0.0),
            )
        trace.stage("fuse", started)
        trace.counts["dense_variants"] = len(blobs)
        trace.counts["sparse_variants"] = len(sparse_texts)
        trace.counts["fused"] = len(candidates)
        return candidates

    # -- stages 3-5: rerank, weight, floor -----------------------------

    def _rank(self, plan: RetrievalPlan, candidates: dict[int, Candidate],
              namespace_plan: NamespacePlan, trace: RetrievalTrace
              ) -> tuple[list[Candidate], float]:
        started = time.perf_counter()
        chunks = self.store.get_chunks(list(candidates))
        for chunk_id, candidate in candidates.items():
            chunk = chunks.get(chunk_id)
            if chunk and chunk.document:
                candidate.namespace = chunk.document.namespace
                candidate.document_id = chunk.document.id

        # Namespace weight decides who gets reranked, not who survives.
        shortlist = sorted(
            candidates.values(),
            key=lambda c: c.rrf * namespace_plan.weight(c.namespace),
            reverse=True,
        )[:self.rerank_top_n]
        trace.stage("shortlist", started)
        trace.counts["shortlisted"] = len(shortlist)

        started = time.perf_counter()
        reranker = self.reranker
        passages = [chunks[c.chunk_id].text for c in shortlist if c.chunk_id in chunks]
        shortlist = [c for c in shortlist if c.chunk_id in chunks]
        try:
            scores = reranker.score(plan.question, passages)
        except Exception as e:
            logger.warning("Reranker failed, falling back to word overlap: %s", e)
            reranker = rerank_module.LexicalReranker()
            scores = reranker.score(plan.question, passages)
        for candidate, score in zip(shortlist, scores):
            candidate.rerank = float(score)
            candidate.score = candidate.rerank * namespace_plan.weight(candidate.namespace)
        trace.stage("rerank", started)
        trace.reranker = reranker.describe()

        floor = rerank_module.floor_for(reranker, self.cfg)
        trace.best_score = max((c.rerank for c in shortlist), default=0.0)
        # Tested against the raw rerank score on purpose: a namespace boost
        # must never buy an irrelevant chunk its way past the threshold.
        survivors = [c for c in shortlist if c.rerank >= floor]
        trace.dropped_below_floor = len(shortlist) - len(survivors)
        survivors.sort(key=lambda c: c.score, reverse=True)
        return survivors, floor

    # -- stages 6-7: diversify and expand ------------------------------

    def _materialise(self, survivors: list[Candidate], k: int, trace: RetrievalTrace
                     ) -> list[Chunk]:
        started = time.perf_counter()
        per_document: dict[int, int] = {}
        picked: list[Candidate] = []
        for candidate in survivors:
            count = per_document.get(candidate.document_id, 0)
            if self.max_per_document and count >= self.max_per_document:
                continue
            per_document[candidate.document_id] = count + 1
            picked.append(candidate)
            if len(picked) >= k:
                break
        trace.counts["after_diversify"] = len(picked)

        chunks = self.store.get_chunks([c.chunk_id for c in picked])
        out: list[Chunk] = []
        seen: set[int] = set()
        for candidate in picked:
            chunk = chunks.get(candidate.chunk_id)
            if chunk is None or chunk.id in seen:
                continue
            chunk.score = candidate.score
            seen.add(chunk.id)
            out.append(chunk)
            # Neighbours restore continuity when a fact straddles a boundary.
            # They inherit a fraction of the score so they sort below their
            # parent and never displace a chunk that earned its place.
            if self.neighbor_window:
                for neighbour in self.store.neighbors(chunk.id, self.neighbor_window):
                    if neighbour.id in seen:
                        continue
                    neighbour.score = candidate.score * 0.5
                    neighbour.document = chunk.document
                    seen.add(neighbour.id)
                    out.append(neighbour)
        trace.stage("materialise", started)
        return out


def retrieve(cfg, store: Store, text: str, k: int = DEFAULT_K, **kwargs) -> RetrievalResult:
    """One-shot convenience wrapper. Prefer a long-lived Retriever in the daemon."""
    return Retriever(cfg, store).retrieve(text, k, **kwargs)
