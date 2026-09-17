"""Reranking - the single biggest quality gain in the retrieval pipeline.

Fusion gets a rough shortlist cheaply. A cross-encoder then reads the query
and each candidate *together* and scores how well one answers the other,
which is a fundamentally better judgement than comparing two vectors that
were computed without ever seeing each other.

It is also what makes abstention possible. Fused ranks say which candidate is
least bad; a cross-encoder score says whether any of them is actually good.
Without that, "return nothing" has nothing to key off.

Three backends, same interface:

- ``CrossEncoderReranker`` (default, ``BAAI/bge-reranker-v2-m3``) - local,
  offline once downloaded.
- ``CohereReranker`` - the hosted API, for people who'd rather not run it.
- ``LexicalReranker`` - the fallback when neither is available. It scores
  word overlap between query and passage, which is weak but is a genuine
  relevance signal, so the floor still means something.

Every backend returns scores in 0..1 and declares the floor that suits it,
because a threshold tuned for a cross-encoder is meaningless applied to word
overlap.
"""

from __future__ import annotations

import logging
import math
import re
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH = 16

# LEXICAL_FLOOR is calibrated: sweeping it against the golden set puts the
# best balance at 0.10 (store recall@10 1.000, abstention precision 1.000);
# by 0.20 precision has fallen to 0.867 because answerable questions start
# abstaining too. Re-run `ev eval --floor X` if you change the corpus.
#
# The two model floors are NOT calibrated - bge-reranker-v2-m3 and Cohere
# Rerank could not be reached from the machine this was built on, so these
# are reasonable starting points for a sigmoid-squashed score, nothing more.
# Run `ev eval` on a machine that can load the model and tune them.
CROSS_ENCODER_FLOOR = 0.30
COHERE_FLOOR = 0.30
LEXICAL_FLOOR = 0.10

_WORD_RE = re.compile(r"\w[\w'-]*")
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from get had has have he her
him his how i if in into is it its me my of on or our she should so some that the their
them then there these they this those to us was we were what when where which who why
will with would you your about
""".split())


# Longest first, and deliberately without "ers": stripping it turns
# "widowmakers" into "widowmak" while "widowmaker" stays whole, so the pair
# stops matching. Plain "s" handles it correctly.
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "est", "ed", "es", "ly", "s")


def _stem(word: str) -> str:
    """Crude suffix stripping, to match how the keyword index tokenises.

    FTS5 runs the porter stemmer, so "widowmakers" in a document already
    matches a search for "widowmaker". Without something equivalent here the
    reranker scores that pair at zero and throws away a document the keyword
    index correctly found - which is how "what is a widowmaker" came back
    empty against a corpus that answers it.
    """
    if len(word) < 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            # "ies" -> "y" ("batteries" -> "batery" is wrong, "battery" right)
            return stem + "y" if suffix in ("ies", "ied") else stem
    return word


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)          # avoid overflow on large negative logits
    return e / (1.0 + e)


class BaseReranker:
    name = "base"
    default_floor = 0.0

    def available(self) -> bool:
        return False

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, 0..1, higher is better."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class LexicalReranker(BaseReranker):
    """Word-overlap scoring - the floor when no model is available.

    Coverage of the query's content words by the passage, with a small bonus
    for adjacent pairs appearing in order. It cannot tell that "automobile"
    answers "car", but it reliably says that a passage sharing none of the
    question's words does not answer it - which is the job that matters here.
    """

    name = "lexical"
    default_floor = LEXICAL_FLOOR

    def available(self) -> bool:
        return True

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = [_stem(w.lower()) for w in _WORD_RE.findall(query or "")
                 if w.lower() not in _STOPWORDS]
        if not terms:
            return [0.0] * len(passages)
        wanted = set(terms)
        bigrams = {f"{a} {b}" for a, b in zip(terms, terms[1:])}
        out = []
        for passage in passages:
            words = [_stem(w.lower()) for w in _WORD_RE.findall(passage or "")]
            present = set(words)
            covered = len(wanted & present) / len(wanted)
            if bigrams:
                text = " ".join(words)
                phrase = sum(1 for b in bigrams if b in text) / len(bigrams)
            else:
                phrase = 0.0
            out.append(min(1.0, 0.75 * covered + 0.25 * phrase))
        return out


class CrossEncoderReranker(BaseReranker):
    """A local cross-encoder via sentence-transformers."""

    default_floor = CROSS_ENCODER_FLOOR

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "",
                 batch_size: int = DEFAULT_BATCH):
        self.model_id = model or DEFAULT_MODEL
        self.name = f"cross-encoder:{self.model_id}"
        self.device = device or None
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                self._load_failed = True
                return None
            try:
                logger.info("Loading reranker %s (first run downloads it)", self.model_id)
                self._model = CrossEncoder(self.model_id, device=self.device)
            except Exception as e:
                logger.warning("Couldn't load reranker %s: %s", self.model_id, e)
                self._load_failed = True
            return self._model

    def available(self) -> bool:
        return self._load() is not None

    def score(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        if model is None:
            raise RuntimeError(f"Reranker {self.model_id} isn't available.")
        if not passages:
            return []
        raw = model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # bge-reranker emits logits, so squash them into 0..1 - the floor is
        # configured as a probability and has to mean the same thing here.
        return [_sigmoid(float(v)) for v in raw]


class CohereReranker(BaseReranker):
    """Cohere's hosted Rerank endpoint, behind the same interface."""

    default_floor = COHERE_FLOOR

    def __init__(self, api_key: str, model: str = "rerank-v3.5",
                 base_url: str = "https://api.cohere.com/v2"):
        self.api_key = api_key
        self.model_id = model
        self.base_url = (base_url or "").rstrip("/")
        self.name = f"cohere:{model}"

    def available(self) -> bool:
        return bool(self.api_key and self.model_id and self.base_url)

    def score(self, query: str, passages: list[str]) -> list[float]:
        import httpx

        if not passages:
            return []
        resp = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_id, "query": query, "documents": passages,
                  "top_n": len(passages)},
            timeout=30,
        )
        resp.raise_for_status()
        scores = [0.0] * len(passages)
        for item in resp.json().get("results", []):
            index = int(item.get("index", -1))
            if 0 <= index < len(scores):
                scores[index] = float(item.get("relevance_score", 0.0))
        return scores


def build_reranker(cfg) -> BaseReranker:
    """Pick the reranker from config, falling back down the chain."""
    choice = (getattr(cfg, "reranker_backend", "auto") or "auto").lower()

    def cross() -> CrossEncoderReranker:
        return CrossEncoderReranker(
            model=getattr(cfg, "reranker_model", DEFAULT_MODEL),
            device=getattr(cfg, "embedding_device", ""),
            batch_size=getattr(cfg, "reranker_batch_size", DEFAULT_BATCH),
        )

    def cohere() -> CohereReranker:
        return CohereReranker(
            api_key=getattr(cfg, "cohere_api_key", ""),
            model=getattr(cfg, "cohere_rerank_model", "rerank-v3.5"),
        )

    if choice in ("cross-encoder", "local", "sentence-transformers"):
        return cross()
    if choice == "cohere":
        return cohere()
    if choice in ("lexical", "none", "off"):
        return LexicalReranker()

    from ev_assistant import probe

    for candidate in (cross(), cohere()):
        if probe.recently_failed(cfg, candidate.name):
            logger.debug("Skipping %s - it failed to load recently", candidate.name)
            continue
        if candidate.available():
            return candidate
        probe.remember_failure(cfg, candidate.name)
    logger.warning(
        "No reranker available - falling back to word-overlap scoring. Retrieval will be "
        "noticeably worse. Install sentence-transformers and let %s download.",
        getattr(cfg, "reranker_model", DEFAULT_MODEL),
    )
    return LexicalReranker()


def floor_for(reranker: BaseReranker, cfg=None) -> float:
    """The relevance floor to apply, config overriding the backend's default.

    A threshold tuned for a cross-encoder means nothing applied to word
    overlap, so the default follows whichever backend is actually running.
    """
    configured = getattr(cfg, "relevance_floor", None)
    if configured is not None and float(configured) >= 0:
        return float(configured)
    return reranker.default_floor
