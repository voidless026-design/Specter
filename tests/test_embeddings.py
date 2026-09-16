"""Phase 3 of the retrieval spec: embeddings and resumable indexing.

Runs entirely offline. The neural backend needs a model download, so these
tests use the hashing backend (deterministic, no network) and a real local
HTTP server for the OpenAI-compatible path. What that leaves unverified is
the sentence-transformers adapter itself - see test_sentence_transformers.py.
"""

from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from ev_assistant.chunking import chunk_rows
from ev_assistant.embeddings import (
    DEFAULT_MODEL,
    Embedder,
    HashingBackend,
    ModelMismatch,
    OpenAIEmbeddingBackend,
    QueryCache,
    SentenceTransformerBackend,
    build_backend,
    deserialize,
    serialize,
)
from ev_assistant.store import EMBED_FAILED, vec_supported

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")


@pytest.fixture
def embedder(cfg, store) -> Embedder:
    cfg.embedding_batch_size = 4
    return Embedder(cfg, store, backend=HashingBackend(32))


class CountingBackend(HashingBackend):
    """A hashing backend that remembers how often it was asked to encode."""

    def __init__(self, dimension: int = 32):
        super().__init__(dimension)
        self.calls: list[list[str]] = []

    def encode(self, texts, *, is_query=False):
        self.calls.append(list(texts))
        return super().encode(texts, is_query=is_query)


def add_chunks(store, texts, *, uri="notes://a", namespace="reference") -> int:
    return store.add_document(
        source_uri=uri, source_type="note", title="T", namespace=namespace,
        text=" ".join(texts), chunks=[{"text": t, "ordinal": i} for i, t in enumerate(texts)],
    )


# -- serialization ---------------------------------------------------------


def test_serialize_round_trip():
    blob = serialize([1.0, -0.5, 0.25])
    assert len(blob) == 3 * 4
    assert struct.unpack("<3f", blob) == (1.0, -0.5, 0.25)
    assert np.allclose(deserialize(blob), [1.0, -0.5, 0.25])


def test_serialize_downcasts_to_float32():
    blob = serialize(np.array([1.0, 2.0], dtype="float64"))
    assert len(blob) == 8
    assert deserialize(blob).dtype == np.dtype("<f4")


# -- hashing backend -------------------------------------------------------


def test_hashing_backend_is_deterministic():
    backend = HashingBackend(64)
    first = backend.encode(["boil the water for one minute"])
    second = backend.encode(["boil the water for one minute"])
    assert np.array_equal(first, second)


def test_hashing_vectors_are_unit_length():
    vectors = HashingBackend(64).encode(["alpha beta", "gamma", "a much longer sentence here"])
    assert vectors.shape == (3, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hashing_ranks_lexical_overlap():
    backend = HashingBackend(256)
    [query, near, far] = backend.encode([
        "how do I purify drinking water",
        "purify drinking water by boiling it first",
        "the bowline knot makes a fixed loop",
    ])
    assert float(query @ near) > float(query @ far)


def test_hashing_handles_empty_and_blank_input():
    backend = HashingBackend(16)
    assert backend.encode([]).shape == (0, 16)
    blank = backend.encode(["", "   "])
    assert blank.shape == (2, 16)
    assert not np.isnan(blank).any()


def test_hashing_name_pins_the_dimension():
    assert HashingBackend(128).name == "hashing:128"
    assert HashingBackend(128).name != HashingBackend(64).name


# -- query cache -----------------------------------------------------------


def test_query_cache_hits_and_misses():
    cache = QueryCache(maxsize=2)
    assert cache.get(("m", "a")) is None
    cache.put(("m", "a"), b"A")
    assert cache.get(("m", "a")) == b"A"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_query_cache_evicts_least_recently_used():
    cache = QueryCache(maxsize=2)
    cache.put(("m", "a"), b"A")
    cache.put(("m", "b"), b"B")
    cache.get(("m", "a"))            # "a" is now the most recent
    cache.put(("m", "c"), b"C")      # evicts "b"
    assert cache.get(("m", "a")) == b"A"
    assert cache.get(("m", "b")) is None
    assert cache.get(("m", "c")) == b"C"


def test_query_cache_is_keyed_by_model():
    cache = QueryCache(maxsize=4)
    cache.put(("model-one", "q"), b"1")
    assert cache.get(("model-two", "q")) is None


def test_query_cache_can_be_disabled():
    cache = QueryCache(maxsize=0)
    cache.put(("m", "a"), b"A")
    assert cache.get(("m", "a")) is None
    assert cache.stats()["size"] == 0


def test_query_cache_clear():
    cache = QueryCache(maxsize=4)
    cache.put(("m", "a"), b"A")
    cache.clear()
    assert cache.stats() == {"size": 0, "maxsize": 4, "hits": 0, "misses": 0}


# -- backend selection -----------------------------------------------------


def test_explicit_backend_choices(cfg):
    cfg.embedding_backend = "hashing"
    cfg.hashing_dimension = 64
    assert build_backend(cfg).name == "hashing:64"

    cfg.embedding_backend = "sentence-transformers"
    assert isinstance(build_backend(cfg), SentenceTransformerBackend)

    cfg.embedding_backend = "openai"
    assert isinstance(build_backend(cfg), OpenAIEmbeddingBackend)


def test_auto_falls_back_to_hashing_when_nothing_is_available(cfg, monkeypatch):
    monkeypatch.setattr(SentenceTransformerBackend, "available", lambda self: False)
    monkeypatch.setattr(OpenAIEmbeddingBackend, "available", lambda self: False)
    cfg.embedding_backend = "auto"
    assert build_backend(cfg).name.startswith("hashing:")


def test_auto_prefers_the_local_model_when_it_loads(cfg, monkeypatch):
    monkeypatch.setattr(SentenceTransformerBackend, "available", lambda self: True)
    cfg.embedding_backend = "auto"
    backend = build_backend(cfg)
    assert isinstance(backend, SentenceTransformerBackend)
    assert backend.name == f"st:{DEFAULT_MODEL}"


def test_backend_carries_config_through(cfg):
    cfg.embedding_backend = "sentence-transformers"
    cfg.embedding_model = "some/model"
    cfg.embedding_device = "cpu"
    cfg.embedding_query_prefix = "Query: "
    cfg.embedding_batch_size = 7
    backend = build_backend(cfg)
    assert backend.model_id == "some/model"
    assert backend.device == "cpu"
    assert backend.query_prefix == "Query: "
    assert backend.batch_size == 7


# -- model pinning ---------------------------------------------------------


@needs_vec
def test_prepare_records_the_model_and_dimension(embedder, store):
    assert embedder.prepare() is True
    assert store.get_meta("embedding_model") == "hashing:32"
    assert store.get_meta("embedding_dim") == "32"


@needs_vec
def test_a_different_model_over_existing_vectors_is_refused(cfg, store):
    first = Embedder(cfg, store, backend=HashingBackend(32))
    first.prepare()
    add_chunks(store, ["one", "two"])
    first.index_pending()
    assert store.stats()["embedded_chunks"] == 2

    second = Embedder(cfg, store, backend=HashingBackend(64))
    with pytest.raises(ModelMismatch, match="ev reindex"):
        second.prepare()


@needs_vec
def test_switching_models_on_an_unembedded_store_is_allowed(cfg, store):
    Embedder(cfg, store, backend=HashingBackend(32)).prepare()
    assert store.get_meta("embedding_dim") == "32"

    switched = Embedder(cfg, store, backend=HashingBackend(64))
    assert switched.prepare() is True
    assert store.get_meta("embedding_model") == "hashing:64"
    assert store.get_meta("embedding_dim") == "64"


@needs_vec
def test_reindex_clears_the_way_for_a_new_model(cfg, store):
    first = Embedder(cfg, store, backend=HashingBackend(32))
    first.prepare()
    add_chunks(store, ["one", "two"])
    first.index_pending()

    store.reset_embeddings()
    store.drop_vec_table()

    second = Embedder(cfg, store, backend=HashingBackend(64))
    assert second.prepare() is True
    report = second.index_pending()
    assert report.embedded == 2
    assert store.get_meta("embedding_model") == "hashing:64"


# -- query embedding -------------------------------------------------------


def test_embed_query_is_cached(cfg, store):
    backend = CountingBackend(32)
    embedder = Embedder(cfg, store, backend=backend)

    first = embedder.embed_query("how do I purify water")
    second = embedder.embed_query("how do I purify water")

    assert first == second
    assert len(backend.calls) == 1
    assert embedder.cache.stats()["hits"] == 1


def test_embed_queries_only_encodes_the_uncached(cfg, store):
    backend = CountingBackend(32)
    embedder = Embedder(cfg, store, backend=backend)

    embedder.embed_query("alpha")
    backend.calls.clear()
    out = embedder.embed_queries(["alpha", "beta", "gamma"])

    assert len(out) == 3
    assert backend.calls == [["beta", "gamma"]]


def test_embed_queries_preserves_order(cfg, store):
    embedder = Embedder(cfg, store, backend=HashingBackend(32))
    embedder.embed_query("beta")  # seed the cache out of order
    out = embedder.embed_queries(["alpha", "beta", "gamma"])
    expected = [embedder.embed_query(t) for t in ("alpha", "beta", "gamma")]
    assert out == expected


def test_query_prefix_is_applied_only_to_queries(cfg, store):
    backend = CountingBackend(32)
    backend.query_prefix = "Q: "
    embedder = Embedder(cfg, store, backend=backend)
    embedder.embed_documents(["a document"])
    assert backend.calls[-1] == ["a document"]


def test_embed_documents_on_empty_input(cfg, store):
    assert Embedder(cfg, store, backend=HashingBackend(8)).embed_documents([]) == []


# -- indexing --------------------------------------------------------------


@needs_vec
def test_index_pending_embeds_everything(embedder, store):
    add_chunks(store, [f"chunk number {i}" for i in range(10)])

    report = embedder.index_pending()

    assert report.embedded == 10
    assert report.failed == 0
    assert report.ok
    assert report.batches == 3  # batch_size 4
    assert store.pending_chunks() == []
    assert store.stats()["embedded_chunks"] == 10


@needs_vec
def test_index_pending_is_a_noop_when_nothing_is_waiting(embedder, store):
    add_chunks(store, ["one"])
    embedder.index_pending()
    again = embedder.index_pending()
    assert again.embedded == 0
    assert again.batches == 0


@needs_vec
def test_index_pending_respects_a_limit(embedder, store):
    add_chunks(store, [f"chunk {i}" for i in range(10)])
    report = embedder.index_pending(limit=6)
    assert report.embedded == 6
    assert len(store.pending_chunks(limit=100)) == 4


@needs_vec
def test_indexing_resumes_where_it_stopped(embedder, store):
    add_chunks(store, [f"chunk {i}" for i in range(12)])
    stop = threading.Event()

    # A crash, a Ctrl-C, or the daemon shutting down mid-ingest.
    first = embedder.index_pending(limit=4)
    assert first.embedded == 4
    stop.set()
    interrupted = embedder.index_pending(stop=stop)
    assert interrupted.stopped_early
    assert interrupted.embedded == 0

    resumed = embedder.index_pending()

    assert resumed.embedded == 8
    assert store.stats()["embedded_chunks"] == 12
    assert store.stats()["pending_chunks"] == 0


@needs_vec
def test_a_failing_backend_marks_the_batch_and_reports(cfg, store):
    class Broken(HashingBackend):
        def encode(self, texts, *, is_query=False):
            if len(texts) > 1:
                raise RuntimeError("model went away")
            return super().encode(texts, is_query=is_query)

    embedder = Embedder(cfg, store, backend=Broken(32))
    add_chunks(store, ["one", "two", "three"])

    report = embedder.index_pending()

    assert report.embedded == 0
    assert report.failed == 3
    assert not report.ok
    assert "model went away" in report.errors[0]
    # Failed chunks leave the pending queue so indexing can't spin on them.
    assert store.pending_chunks() == []
    with store._connect() as conn:
        statuses = [r["embedding_status"] for r in
                    conn.execute("SELECT embedding_status FROM chunks").fetchall()]
    assert statuses == [EMBED_FAILED] * 3


@needs_vec
def test_progress_callback_is_called_per_batch(embedder, store):
    add_chunks(store, [f"chunk {i}" for i in range(8)])
    seen: list[int] = []
    embedder.index_pending(progress=seen.append)
    assert seen == [4, 8]


def test_an_unusable_backend_does_not_take_the_indexer_down(cfg, store):
    class Unloadable(HashingBackend):
        def encode(self, texts, *, is_query=False):
            raise RuntimeError("model file is missing")

    embedder = Embedder(cfg, store, backend=Unloadable(0))
    add_chunks(store, ["one"])

    report = embedder.index_pending()   # must not raise: this runs in a thread

    assert report.embedded == 0
    assert "model file is missing" in report.errors[0]


@needs_vec
def test_a_model_mismatch_still_surfaces_from_the_indexer(cfg, store):
    first = Embedder(cfg, store, backend=HashingBackend(32))
    first.prepare()
    add_chunks(store, ["one"])
    first.index_pending()

    second = Embedder(cfg, store, backend=HashingBackend(64))
    with pytest.raises(ModelMismatch):
        second.index_pending()


def test_indexing_without_sqlite_vec_is_skipped_not_fatal(cfg, store, monkeypatch):
    monkeypatch.setattr(store, "vec_available", False)
    monkeypatch.setattr(store, "ensure_vec_table", lambda dim, model="": False)
    embedder = Embedder(cfg, store, backend=HashingBackend(32))
    add_chunks(store, ["one", "two"])

    report = embedder.index_pending()

    assert report.embedded == 0
    assert report.failed == 0
    assert len(store.pending_chunks()) == 2  # still queued for a later reindex


# -- end to end ------------------------------------------------------------


@needs_vec
def test_chunk_store_index_and_search(cfg, store):
    embedder = Embedder(cfg, store, backend=HashingBackend(512))
    documents = {
        "Water": "Boil water for one full minute to purify it before drinking.",
        "Knots": "The bowline knot makes a fixed loop that will not slip under load.",
        "Fire": "Gather dry tinder and kindling before striking a spark for a fire.",
    }
    for title, text in documents.items():
        store.add_document(
            source_uri=f"notes://{title}", source_type="note", title=title,
            text=text, chunks=chunk_rows(text, title=title),
        )

    assert embedder.index_pending().embedded == len(documents)

    query = embedder.embed_query("how do I make water safe to drink")
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = 1"
            " ORDER BY distance", (query,),
        ).fetchall()

    best = store.get_chunk(rows[0]["chunk_id"])
    assert best.document.title == "Water"


# -- OpenAI-compatible backend ---------------------------------------------


class _EmbeddingHandler(BaseHTTPRequestHandler):
    """A real /embeddings endpoint, so the HTTP path is genuinely exercised."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "model": body.get("model"),
            "input": body["input"],
        })
        # Returned deliberately out of order - real APIs don't promise order,
        # which is what the "index" field is for.
        data = [
            {"index": i, "embedding": [float(len(text)), 1.0, 2.0, 3.0]}
            for i, text in enumerate(body["input"])
        ]
        payload = json.dumps({"data": list(reversed(data))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def embedding_server():
    server = HTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_openai_backend_round_trip(embedding_server):
    host, port = embedding_server.server_address
    backend = OpenAIEmbeddingBackend(
        base_url=f"http://{host}:{port}/v1", model="text-embed", api_key="k", batch_size=2,
    )

    vectors = backend.encode(["aa", "bbbb", "cccccc"])

    assert vectors.shape == (3, 4)
    assert backend.dimension == 4
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Order survived the shuffled response: first component encodes length.
    lengths = [float(v[0] / v[1]) for v in vectors]  # v[1] was 1.0 before norming
    assert lengths == pytest.approx([2.0, 4.0, 6.0])


def test_openai_backend_batches_and_authenticates(embedding_server):
    host, port = embedding_server.server_address
    backend = OpenAIEmbeddingBackend(
        base_url=f"http://{host}:{port}/v1", model="text-embed", api_key="secret", batch_size=2,
    )

    backend.encode(["a", "b", "c", "d", "e"])

    assert [len(call["input"]) for call in embedding_server.seen] == [2, 2, 1]
    assert {call["auth"] for call in embedding_server.seen} == {"Bearer secret"}
    assert {call["path"] for call in embedding_server.seen} == {"/v1/embeddings"}
    assert {call["model"] for call in embedding_server.seen} == {"text-embed"}


def test_openai_backend_available_probes_the_endpoint(embedding_server):
    host, port = embedding_server.server_address
    live = OpenAIEmbeddingBackend(f"http://{host}:{port}/v1", "text-embed", "k")
    assert live.available() is True

    dead = OpenAIEmbeddingBackend("http://127.0.0.1:1/v1", "text-embed", "k")
    assert dead.available() is False


def test_openai_backend_needs_configuration():
    assert OpenAIEmbeddingBackend("", "model", "key").available() is False
    assert OpenAIEmbeddingBackend("http://x/v1", "", "key").available() is False
    assert OpenAIEmbeddingBackend("http://x/v1", "model", "").available() is False


@needs_vec
def test_embedder_learns_dimension_from_a_backend_that_only_knows_at_runtime(
    cfg, store, embedding_server
):
    host, port = embedding_server.server_address
    backend = OpenAIEmbeddingBackend(f"http://{host}:{port}/v1", "text-embed", "k")
    assert backend.dimension == 0

    embedder = Embedder(cfg, store, backend=backend)
    assert embedder.prepare() is True

    assert store.get_meta("embedding_dim") == "4"
    assert store.get_meta("embedding_model") == "openai:text-embed"
