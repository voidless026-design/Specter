"""Chunk and query embeddings - local first, resumable, pinned to one model.

Three backends behind one interface:

- ``SentenceTransformerBackend`` (default, ``BAAI/bge-m3``): a real neural
  embedder running on this machine. Multilingual, 8192-token context, strong
  retrieval scores. Needs the model downloaded once; after that it is fully
  offline. GPU if torch finds one, CPU otherwise.
- ``OpenAIEmbeddingBackend``: any OpenAI-compatible ``/embeddings`` endpoint,
  for people who would rather pay than run a model locally.
- ``HashingBackend``: a dependency-free lexical fallback so vector search
  still functions on a machine that has never had internet. It is *not* a
  semantic model - "car" and "automobile" are unrelated to it - but paired
  with BM25 it is better than having no dense channel at all.

Two invariants the rest of the system leans on:

- The model name and dimension are recorded in the store's ``meta`` table.
  Vectors from two different models must never share an index, so a mismatch
  is refused loudly and `ev reindex` is the way through.
- Indexing is resumable. ``chunks.embedding_status`` is the bookmark, so a
  crash part-way through a 10,000-chunk ingest costs you that batch, not the
  whole run.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
HASHING_DIM = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_QUERY_CACHE = 256

_WORD_RE = re.compile(r"[a-z0-9]+")


def serialize(vector) -> bytes:
    """Pack a vector the way sqlite-vec wants it: little-endian float32."""
    return np.asarray(vector, dtype="<f4").tobytes()


def deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so L2 distance ranks the same way cosine does."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class BaseBackend:
    """One way of turning text into vectors.

    `name` is what gets written to `meta.embedding_model`, so it must change
    whenever the produced vectors would change.
    """

    name = "base"
    dimension = 0

    def available(self) -> bool:
        return False

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} ({self.dimension}d)"


class HashingBackend(BaseBackend):
    """Deterministic lexical embedding - no model, no network, no download.

    Hashes unigrams and bigrams into a fixed number of signed buckets with
    sublinear term frequency, then normalises. Cosine similarity over these
    vectors approximates lexical overlap, not meaning, which is exactly the
    limitation to keep in mind: it is the floor, not the goal.
    """

    def __init__(self, dimension: int = HASHING_DIM):
        self.dimension = int(dimension)
        self.name = f"hashing:{self.dimension}"

    def available(self) -> bool:
        return True

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype="float32")
        words = _WORD_RE.findall((text or "").lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        return vec

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        return _l2_normalize(np.vstack([self._vector(t) for t in texts]))


def _embedding_dimension(model) -> int:
    """How wide this model's vectors are, across sentence-transformers versions.

    v6 renamed `get_sentence_embedding_dimension` to `get_embedding_dimension`
    and the old name now warns on the way to being removed. Ask for the new
    one first so the wrapper works on both.
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            width = getter()
            if width:
                return int(width)
    raise RuntimeError("Couldn't determine the embedding model's dimension.")


class SentenceTransformerBackend(BaseBackend):
    """A local neural embedder via sentence-transformers.

    The model is loaded lazily and only once: loading is slow and the daemon
    starts before anyone asks a question. `available()` does attempt the load,
    because "is sentence-transformers importable" is not the question that
    matters - "can this machine actually produce a vector" is, and a model
    that was never downloaded fails only at that point.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_prefix: str = "",
    ):
        self.model_id = model or DEFAULT_MODEL
        self.name = f"st:{self.model_id}"
        self.device = device or None
        self.batch_size = max(1, int(batch_size))
        # bge-m3 needs no query instruction. bge-v1.5 and friends do - hence
        # the setting rather than a hardcoded prefix.
        self.query_prefix = query_prefix
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()
        self.dimension = 0

    def _load(self):
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.info("sentence-transformers isn't installed")
                self._load_failed = True
                return None
            try:
                logger.info("Loading embedding model %s (first run downloads it)", self.model_id)
                model = SentenceTransformer(self.model_id, device=self.device)
                self.dimension = _embedding_dimension(model)
                self._model = model
            except Exception as e:
                logger.warning("Couldn't load embedding model %s: %s", self.model_id, e)
                self._load_failed = True
            return self._model

    def available(self) -> bool:
        return self._load() is not None

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        model = self._load()
        if model is None:
            raise RuntimeError(f"Embedding model {self.model_id} isn't available.")
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


class OpenAIEmbeddingBackend(BaseBackend):
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, Voyage, local)."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.base_url = (base_url or "").rstrip("/")
        self.model_id = model
        self.api_key = api_key
        self.batch_size = max(1, int(batch_size))
        self.name = f"openai:{model}"
        self.dimension = 0

    def available(self) -> bool:
        if not (self.base_url and self.model_id and self.api_key):
            return False
        try:
            self.encode(["ping"])
            return True
        except Exception as e:
            logger.warning("Embedding endpoint %s isn't reachable: %s", self.base_url, e)
            return False

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        import httpx

        if not texts:
            return np.zeros((0, self.dimension or 1), dtype="float32")
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model_id, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            # The API may return items out of order; `index` is authoritative.
            items = sorted(payload["data"], key=lambda d: d.get("index", 0))
            rows.extend(item["embedding"] for item in items)
        matrix = np.asarray(rows, dtype="float32")
        self.dimension = int(matrix.shape[1])
        return _l2_normalize(matrix)


def build_backend(cfg) -> BaseBackend:
    """Pick the embedding backend from config, falling back down the chain."""
    choice = (getattr(cfg, "embedding_backend", "auto") or "auto").lower()
    batch = getattr(cfg, "embedding_batch_size", DEFAULT_BATCH_SIZE)

    def sentence_transformer() -> SentenceTransformerBackend:
        return SentenceTransformerBackend(
            model=getattr(cfg, "embedding_model", DEFAULT_MODEL),
            device=getattr(cfg, "embedding_device", ""),
            batch_size=batch,
            query_prefix=getattr(cfg, "embedding_query_prefix", ""),
        )

    def openai() -> OpenAIEmbeddingBackend:
        return OpenAIEmbeddingBackend(
            base_url=getattr(cfg, "openai_embedding_base_url", ""),
            model=getattr(cfg, "openai_embedding_model", ""),
            api_key=getattr(cfg, "openai_api_key", ""),
            batch_size=batch,
        )

    if choice in ("sentence-transformers", "st", "local"):
        return sentence_transformer()
    if choice == "openai":
        return openai()
    if choice == "hashing":
        return HashingBackend(getattr(cfg, "hashing_dimension", HASHING_DIM))

    # auto: the best thing this machine can actually do, right now. A model
    # that failed to load recently is skipped rather than re-probed, because
    # each retry costs a network timeout and the CLI is a new process every
    # time (see probe.py).
    from ev_assistant import probe

    for candidate in (sentence_transformer(), openai()):
        if probe.recently_failed(cfg, candidate.name):
            logger.debug("Skipping %s - it failed to load recently", candidate.name)
            continue
        if candidate.available():
            return candidate
        probe.remember_failure(cfg, candidate.name)
    logger.warning(
        "No embedding model available - falling back to lexical hashing vectors. "
        "Install sentence-transformers and let `%s` download for real semantic search.",
        getattr(cfg, "embedding_model", DEFAULT_MODEL),
    )
    return HashingBackend(getattr(cfg, "hashing_dimension", HASHING_DIM))


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------


class QueryCache:
    """Small LRU over query vectors.

    Queries repeat far more than documents do - follow-ups, HyDE variants,
    and the eval harness all re-embed the same strings - and encoding is the
    slowest step in a warm retrieval.
    """

    def __init__(self, maxsize: int = DEFAULT_QUERY_CACHE):
        self.maxsize = max(0, int(maxsize))
        self._entries: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str]) -> bytes | None:
        if self.maxsize == 0:
            return None
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
        self.misses += 1
        return None

    def put(self, key: tuple[str, str], value: bytes) -> None:
        if self.maxsize == 0:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        return {"size": len(self._entries), "maxsize": self.maxsize,
                "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@dataclass
class IndexReport:
    """What one indexing run got through."""

    embedded: int = 0
    failed: int = 0
    batches: int = 0
    stopped_early: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.errors


class ModelMismatch(RuntimeError):
    """Raised rather than mixing vectors from two different models."""


class Embedder:
    """The embedding layer everything else talks to."""

    def __init__(self, cfg, store, backend: BaseBackend | None = None):
        self.cfg = cfg
        self.store = store
        self.backend = backend or build_backend(cfg)
        self.batch_size = max(1, int(getattr(cfg, "embedding_batch_size", DEFAULT_BATCH_SIZE)))
        self.cache = QueryCache(getattr(cfg, "query_cache_size", DEFAULT_QUERY_CACHE))
        self._prepared = False

    # -- setup ---------------------------------------------------------

    def _dimension(self) -> int:
        if not self.backend.dimension:
            # Some backends only learn their width by producing a vector.
            self.backend.encode(["dimension probe"])
        return int(self.backend.dimension)

    def prepare(self) -> bool:
        """Pin the store to this model. False if vectors aren't available.

        Raises ModelMismatch when the store already holds vectors from a
        different model - silently mixing two embedding spaces produces
        retrieval that is wrong in a way nothing downstream can detect.
        """
        if self._prepared:
            return self.store.vec_available
        recorded = self.store.get_meta("embedding_model", "")
        if recorded and recorded != self.backend.name:
            embedded = self.store.stats()["embedded_chunks"]
            if embedded:
                raise ModelMismatch(
                    f"This store was built with '{recorded}' but the configured model is "
                    f"'{self.backend.name}'. Vectors from two models can't share an index. "
                    f"Run `ev reindex` to rebuild, or set the old model back in config."
                )
            self.store.drop_vec_table()
        dim = self._dimension()
        ready = self.store.ensure_vec_table(dim, model=self.backend.name)
        self._prepared = True
        if not ready:
            logger.warning("sqlite-vec unavailable - indexing will be skipped, keyword search "
                           "still works")
        return ready

    # -- encoding ------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        vectors = self.backend.encode(list(texts), is_query=False)
        return [serialize(v) for v in vectors]

    def embed_query(self, text: str) -> bytes:
        """Embed one query, memoised. Keyed by model so a swap can't stale."""
        key = (self.backend.name, text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        blob = serialize(self.backend.encode([text], is_query=True)[0])
        self.cache.put(key, blob)
        return blob

    def embed_queries(self, texts: list[str]) -> list[bytes]:
        """Embed several queries, encoding only the ones not already cached."""
        out: list[bytes | None] = [None] * len(texts)
        todo: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get((self.backend.name, text))
            if cached is None:
                todo.append(i)
            else:
                out[i] = cached
        if todo:
            vectors = self.backend.encode([texts[i] for i in todo], is_query=True)
            for i, vector in zip(todo, vectors):
                blob = serialize(vector)
                self.cache.put((self.backend.name, texts[i]), blob)
                out[i] = blob
        return [blob for blob in out if blob is not None]

    # -- indexing ------------------------------------------------------

    def index_pending(
        self,
        limit: int | None = None,
        stop: threading.Event | None = None,
        progress=None,
    ) -> IndexReport:
        """Embed chunks that don't have vectors yet. Resumable by design.

        Each batch is committed before the next is fetched, so an interrupt -
        Ctrl-C, a crash, the daemon shutting down - costs one batch and the
        next run picks up exactly where this one stopped.
        """
        report = IndexReport()
        try:
            if not self.prepare():
                return report
        except ModelMismatch:
            raise  # needs a person to run `ev reindex`; don't bury it
        except Exception as e:
            # The daemon indexes in a background thread; an unloadable model
            # must not take that thread down with it.
            logger.warning("Embedding backend isn't usable: %s", e)
            report.errors.append(str(e))
            return report
        done = 0
        while True:
            if stop is not None and stop.is_set():
                report.stopped_early = True
                break
            want = self.batch_size
            if limit is not None:
                if done >= limit:
                    break
                want = min(want, limit - done)
            chunks = self.store.pending_chunks(limit=want)
            if not chunks:
                break
            try:
                blobs = self.embed_documents([c.text for c in chunks])
            except Exception as e:
                # Mark the batch failed rather than spinning on it forever -
                # `ev reindex` requeues them once the cause is fixed.
                logger.exception("Embedding batch failed")
                self.store.mark_embedding_failed([c.id for c in chunks])
                report.failed += len(chunks)
                report.errors.append(str(e))
                break
            self.store.store_embeddings(list(zip([c.id for c in chunks], blobs)))
            report.embedded += len(chunks)
            report.batches += 1
            done += len(chunks)
            if progress is not None:
                progress(report.embedded)
        return report

    def describe(self) -> str:
        return self.backend.describe()
