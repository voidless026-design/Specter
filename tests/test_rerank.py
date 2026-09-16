"""Phase 7, part one: the reranker interface and its backends."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ev_assistant.rerank import (
    CROSS_ENCODER_FLOOR,
    LEXICAL_FLOOR,
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    LexicalReranker,
    _sigmoid,
    build_reranker,
    floor_for,
)


# -- sigmoid ---------------------------------------------------------------


def test_sigmoid_maps_logits_into_zero_to_one():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < _sigmoid(-40.0) < 0.01
    assert 0.99 < _sigmoid(40.0) <= 1.0


def test_sigmoid_does_not_overflow_on_extremes():
    # A cross-encoder can emit a large negative logit for an obvious mismatch.
    assert _sigmoid(-10000.0) == pytest.approx(0.0)
    assert _sigmoid(10000.0) == pytest.approx(1.0)


# -- lexical fallback ------------------------------------------------------


def test_lexical_scores_word_overlap():
    reranker = LexicalReranker()
    scores = reranker.score("how do I purify drinking water", [
        "Boil the water to purify it before drinking.",
        "The bowline knot makes a fixed loop.",
    ])
    assert scores[0] > scores[1]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_lexical_gives_an_unrelated_passage_nothing():
    # This is the job that matters: saying "this does not answer the question".
    assert LexicalReranker().score("capital of Mongolia", ["Boil water for a minute."]) == [0.0]


def test_lexical_rewards_a_phrase_appearing_in_order():
    reranker = LexicalReranker()
    in_order = reranker.score("rolling boil", ["hold a rolling boil for a minute"])[0]
    scattered = reranker.score("rolling boil", ["boil it, then let the rolling stop"])[0]
    assert in_order > scattered


def test_lexical_handles_empty_input():
    assert LexicalReranker().score("", ["anything"]) == [0.0]
    assert LexicalReranker().score("a the of", ["anything"]) == [0.0]
    assert LexicalReranker().score("water", []) == []


def test_lexical_is_always_available():
    assert LexicalReranker().available() is True


# -- cross-encoder ---------------------------------------------------------


def test_cross_encoder_degrades_when_the_model_is_missing(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reranker = CrossEncoderReranker(model="definitely/not-a-real-model", device="cpu")
    assert reranker.available() is False
    with pytest.raises(RuntimeError, match="isn't available"):
        reranker.score("q", ["p"])


def test_cross_encoder_squashes_logits(monkeypatch):
    reranker = CrossEncoderReranker(model="fake/model")

    class FakeModel:
        def predict(self, pairs, **kwargs):
            # Logits, the way bge-reranker actually emits them.
            return [8.0, -8.0, 0.0][:len(pairs)]

    reranker._model = FakeModel()
    scores = reranker.score("q", ["good", "bad", "middling"])

    assert scores[0] > 0.99
    assert scores[1] < 0.01
    assert scores[2] == pytest.approx(0.5)


def test_cross_encoder_name_pins_the_model():
    assert CrossEncoderReranker(model="a/b").name == "cross-encoder:a/b"
    assert CrossEncoderReranker(model="a/b").name != CrossEncoderReranker(model="c/d").name


def test_cross_encoder_empty_passages():
    reranker = CrossEncoderReranker(model="fake/model")
    reranker._model = object()
    assert reranker.score("q", []) == []


# -- cohere ----------------------------------------------------------------


class _RerankHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({"auth": self.headers.get("Authorization"), "body": body})
        # Deliberately out of order and partial - real APIs do both.
        results = [{"index": i, "relevance_score": round(1.0 - i * 0.3, 2)}
                   for i in range(len(body["documents"]))]
        payload = json.dumps({"results": list(reversed(results))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def rerank_server():
    server = HTTPServer(("127.0.0.1", 0), _RerankHandler)
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_cohere_reranker_keeps_passage_order(rerank_server):
    host, port = rerank_server.server_address
    reranker = CohereReranker(api_key="k", base_url=f"http://{host}:{port}/v2")

    scores = reranker.score("q", ["first", "second", "third"])

    assert scores == pytest.approx([1.0, 0.7, 0.4])
    assert rerank_server.seen[0]["auth"] == "Bearer k"
    assert rerank_server.seen[0]["body"]["query"] == "q"


def test_cohere_needs_a_key():
    assert CohereReranker(api_key="").available() is False
    assert CohereReranker(api_key="k").available() is True


def test_cohere_empty_passages():
    assert CohereReranker(api_key="k").score("q", []) == []


# -- selection and floors --------------------------------------------------


def test_explicit_backend_choices(cfg):
    cfg.reranker_backend = "lexical"
    assert isinstance(build_reranker(cfg), LexicalReranker)

    cfg.reranker_backend = "cross-encoder"
    assert isinstance(build_reranker(cfg), CrossEncoderReranker)

    cfg.reranker_backend = "cohere"
    assert isinstance(build_reranker(cfg), CohereReranker)


def test_auto_falls_back_to_lexical_when_nothing_is_available(cfg, monkeypatch):
    monkeypatch.setattr(CrossEncoderReranker, "available", lambda self: False)
    monkeypatch.setattr(CohereReranker, "available", lambda self: False)
    cfg.reranker_backend = "auto"
    assert isinstance(build_reranker(cfg), LexicalReranker)


def test_auto_prefers_the_cross_encoder(cfg, monkeypatch):
    monkeypatch.setattr(CrossEncoderReranker, "available", lambda self: True)
    cfg.reranker_backend = "auto"
    assert isinstance(build_reranker(cfg), CrossEncoderReranker)


def test_the_floor_follows_the_backend_in_use(cfg):
    # A threshold tuned for a cross-encoder means nothing for word overlap.
    cfg.relevance_floor = -1
    assert floor_for(LexicalReranker(), cfg) == LEXICAL_FLOOR
    assert floor_for(CrossEncoderReranker(), cfg) == CROSS_ENCODER_FLOOR
    assert floor_for(LexicalReranker(), cfg) != floor_for(CrossEncoderReranker(), cfg)


def test_config_overrides_the_floor(cfg):
    cfg.relevance_floor = 0.8
    assert floor_for(LexicalReranker(), cfg) == 0.8
    assert floor_for(CrossEncoderReranker(), cfg) == 0.8


def test_a_zero_floor_is_respected_not_treated_as_unset(cfg):
    cfg.relevance_floor = 0.0
    assert floor_for(CrossEncoderReranker(), cfg) == 0.0


def test_floor_without_config():
    assert floor_for(LexicalReranker()) == LEXICAL_FLOOR


def test_base_reranker_is_abstract():
    assert BaseReranker().available() is False
    with pytest.raises(NotImplementedError):
        BaseReranker().score("q", ["p"])
