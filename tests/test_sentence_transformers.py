"""Exercises the sentence-transformers adapter against the real library.

The default model (BAAI/bge-m3) is a ~2GB download, so these tests build a
SentenceTransformer out of a toy encoder instead: real library, real
`encode()` plumbing, no weights. That verifies everything about the adapter
except the quality of the vectors - batching, normalisation, dtype, device
selection, dimension discovery and the query prefix.

What it does NOT verify is that bge-m3 itself loads and performs well. That
needs the model, and can only be checked on a machine that can reach
huggingface.co.
"""

from __future__ import annotations

import numpy as np
import pytest

from ev_assistant.embeddings import (
    Embedder,
    SentenceTransformerBackend,
    _embedding_dimension,
    deserialize,
)
from ev_assistant.store import vec_supported


def has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not has_sentence_transformers(), reason="sentence-transformers not installed"
)


def toy_model(dim: int = 8, vocab: int = 97):
    """A real SentenceTransformer with made-up weights and no download."""
    import torch
    from torch import nn
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling

    class ToyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(vocab, dim)
            torch.manual_seed(0)
            nn.init.normal_(self.embeddings.weight, 0.0, 1.0)

        def tokenize(self, texts, **kwargs):
            width = max((len(t.split()) for t in texts), default=1) or 1
            ids, masks = [], []
            for text in texts:
                row = [sum(map(ord, word)) % vocab for word in text.split()] or [0]
                masks.append([1] * len(row) + [0] * (width - len(row)))
                ids.append(row + [0] * (width - len(row)))
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

        def forward(self, features):
            features["token_embeddings"] = self.embeddings(features["input_ids"])
            return features

        def get_word_embedding_dimension(self):
            return dim

    encoder = ToyEncoder()
    return SentenceTransformer(modules=[encoder, Pooling(dim, "mean")], device="cpu")


@pytest.fixture
def backend():
    """The real adapter, wired to the toy model instead of a downloaded one."""
    b = SentenceTransformerBackend(model="toy/model", device="cpu", batch_size=2)
    b._model = toy_model()
    b.dimension = _embedding_dimension(b._model)
    return b


def test_dimension_discovery_works_on_this_library_version():
    model = toy_model(dim=12)
    assert _embedding_dimension(model) == 12


def test_dimension_discovery_reports_clearly_when_it_cannot_tell():
    with pytest.raises(RuntimeError, match="dimension"):
        _embedding_dimension(object())


def test_encode_returns_normalized_float32_rows(backend):
    vectors = backend.encode(["boil the water", "tie a bowline knot", "gather dry tinder"])

    assert vectors.shape == (3, 8)
    assert vectors.dtype == np.dtype("float32")
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic_and_batch_size_independent(backend):
    texts = [f"sentence number {i} about water" for i in range(7)]

    backend.batch_size = 2
    small = backend.encode(texts)
    backend.batch_size = 16
    large = backend.encode(texts)

    assert np.allclose(small, large, atol=1e-5)


def test_empty_input_returns_an_empty_matrix(backend):
    assert backend.encode([]).shape == (0, 8)


def test_query_prefix_is_applied_to_queries_only(backend, monkeypatch):
    seen: list[list[str]] = []
    real_encode = backend._model.encode

    def spy(texts, **kwargs):
        seen.append(list(texts))
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(backend._model, "encode", spy)
    backend.query_prefix = "Represent this sentence: "

    backend.encode(["a document"], is_query=False)
    backend.encode(["a question"], is_query=True)

    assert seen[0] == ["a document"]
    assert seen[1] == ["Represent this sentence: a question"]


def test_a_model_that_cannot_load_degrades_instead_of_crashing(monkeypatch):
    # The realistic failure on a fresh install: no network, nothing cached.
    # HF_HUB_OFFLINE keeps the test off the network and fast either way.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    backend = SentenceTransformerBackend(model="definitely/not-a-real-model", device="cpu")
    assert backend.available() is False
    with pytest.raises(RuntimeError, match="isn't available"):
        backend.encode(["anything"])


def test_load_failure_is_not_retried_on_every_call(monkeypatch):
    backend = SentenceTransformerBackend(model="definitely/not-a-real-model", device="cpu")
    attempts = []

    import sentence_transformers

    def counting_init(*args, **kwargs):
        attempts.append(args)
        raise OSError("no such model")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", counting_init)
    backend.available()
    backend.available()
    backend.available()
    assert len(attempts) == 1


def test_backend_name_pins_the_model_id():
    assert SentenceTransformerBackend(model="BAAI/bge-m3").name == "st:BAAI/bge-m3"
    assert SentenceTransformerBackend(model="other/model").name == "st:other/model"


@pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")
def test_end_to_end_through_the_embedder(cfg, store, backend):
    embedder = Embedder(cfg, store, backend=backend)
    assert embedder.prepare() is True
    assert store.get_meta("embedding_model") == "st:toy/model"
    assert store.get_meta("embedding_dim") == "8"

    store.add_document(
        source_uri="notes://a", source_type="note", title="Water",
        text="boil the water", chunks=[{"text": "boil the water", "ordinal": 0}],
    )
    report = embedder.index_pending()

    assert report.embedded == 1
    stored = deserialize(embedder.embed_query("boil the water"))
    assert stored.shape == (8,)
    assert np.isclose(np.linalg.norm(stored), 1.0, atol=1e-5)
