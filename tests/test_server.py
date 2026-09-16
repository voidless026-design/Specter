from __future__ import annotations

import time

from fastapi.testclient import TestClient

from ev_assistant.brain import BrainReply
from ev_assistant.bus import StateBus
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.knowledge import Knowledge
from ev_assistant.server import DaemonStatus, create_app


class FakeBrain:
    def __init__(self, reply=None, sources=None):
        self.calls = []
        self._reply = reply
        self._sources = sources or []

    def respond(self, text: str, confirm=None) -> str:
        return self.respond_detailed(text, confirm).text

    def respond_detailed(self, text: str, confirm=None) -> BrainReply:
        self.calls.append((text, confirm))
        return BrainReply(
            text=self._reply if self._reply is not None else f"echo: {text}",
            provider="fake", sources=self._sources,
            retrieval={"reason": "ok", "chunks": len(self._sources)},
        )


class FakeVoice:
    def __init__(self):
        self.spoken = []

    def say(self, text: str) -> None:
        self.spoken.append(text)


def _make_client(cfg, memory, brain=None, voice=None, shutdown_calls=None):
    brain = brain or FakeBrain()
    voice = voice if voice is not None else FakeVoice()
    shutdown_calls = shutdown_calls if shutdown_calls is not None else []
    feed_loop = DataFeedLoop(cfg, memory)
    knowledge = Knowledge(cfg.knowledge_path)
    status = DaemonStatus()
    status.state = "listening"
    status.wake_word_ready = True

    app = create_app(
        cfg=cfg,
        brain=brain,
        memory=memory,
        knowledge=knowledge,
        feed_loop=feed_loop,
        voice=voice,
        status=status,
        bus=StateBus(),
        request_shutdown=lambda: shutdown_calls.append(True),
    )
    return TestClient(app), brain, voice, shutdown_calls


def _auth(cfg) -> dict:
    return {"Authorization": f"Bearer {cfg.control_token}"}


def test_status_requires_auth(cfg, memory):
    client, *_ = _make_client(cfg, memory)
    assert client.get("/status").status_code == 401


def test_status_rejects_wrong_token(cfg, memory):
    client, *_ = _make_client(cfg, memory)
    resp = client.get("/status", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_status_returns_expected_fields(cfg, memory):
    memory.add_fact("weather", "sunny", external_id="w1")
    client, *_ = _make_client(cfg, memory)

    resp = client.get("/status", headers=_auth(cfg))

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "listening"
    assert body["wake_word_ready"] is True
    assert body["fact_count"] == 1
    assert body["provider"] == cfg.brain_provider
    assert "brain_status" in body


def test_status_with_no_control_token_configured_returns_503(cfg, memory):
    cfg.control_token = ""
    client, *_ = _make_client(cfg, memory)
    resp = client.get("/status", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 503


def test_ask_requires_auth(cfg, memory):
    client, *_ = _make_client(cfg, memory)
    assert client.post("/ask", json={"text": "hi"}).status_code == 401


def test_ask_returns_brain_reply_and_speaks_by_default(cfg, memory):
    client, brain, voice, _ = _make_client(cfg, memory)

    resp = client.post("/ask", json={"text": "hello"}, headers=_auth(cfg))

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "echo: hello"
    assert body["provider"] == "fake"
    assert brain.calls[0][0] == "hello"
    _wait_until(lambda: voice.spoken)
    assert voice.spoken == ["echo: hello"]


def test_ask_returns_citations_and_a_retrieval_trace(cfg, memory):
    sources = [{"tag": "[S1]", "label": "Wikipedia - Tungsten",
                "source_uri": "wikipedia:Tungsten", "score": 0.9, "chunk_id": 1},
               {"tag": "[S2]", "label": "Your notes - Shed", "source_uri": "notes://shed",
                "score": 0.4, "chunk_id": 2}]
    brain = FakeBrain(reply="It melts at 3422 degrees [S1].", sources=sources)
    client, _, _, _ = _make_client(cfg, memory, brain=brain)

    body = client.post("/ask", json={"text": "melting point", "speak": False},
                       headers=_auth(cfg)).json()

    # Only the source actually cited comes back, not everything offered.
    assert [c["tag"] for c in body["citations"]] == ["[S1]"]
    assert body["retrieval_trace"]["reason"] == "ok"
    assert body["retrieval_trace"]["chunks"] == 2


def test_the_retrieval_trace_is_behind_the_same_auth(cfg, memory):
    client, *_ = _make_client(cfg, memory)
    assert client.post("/ask", json={"text": "hi"}).status_code == 401


def test_the_voice_never_says_source_tags(cfg, memory):
    brain = FakeBrain(reply="Boil it for a minute [S1]. Filter it first [S2].")
    client, _, voice, _ = _make_client(cfg, memory, brain=brain)

    body = client.post("/ask", json={"text": "water"}, headers=_auth(cfg)).json()

    assert "[S1]" in body["reply"]          # text keeps them for citing
    assert "[S" not in body["spoken"]       # speech drops them
    _wait_until(lambda: voice.spoken)
    assert "[S" not in voice.spoken[0]


def test_ask_destructive_gate_defaults_to_deny(cfg, memory):
    client, brain, _, _ = _make_client(cfg, memory)
    client.post("/ask", json={"text": "delete stuff"}, headers=_auth(cfg))
    # Without allow_destructive, the confirm callback must deny.
    _, confirm = brain.calls[0]
    assert confirm("run: rm -rf x") is False


def test_ask_allow_destructive_permits(cfg, memory):
    client, brain, _, _ = _make_client(cfg, memory)
    client.post("/ask", json={"text": "delete stuff", "allow_destructive": True}, headers=_auth(cfg))
    _, confirm = brain.calls[0]
    assert confirm("run: rm -rf x") is True


def test_ask_does_not_speak_when_speak_is_false(cfg, memory):
    client, _, voice, _ = _make_client(cfg, memory)

    resp = client.post("/ask", json={"text": "hello", "speak": False}, headers=_auth(cfg))

    assert resp.status_code == 200
    time.sleep(0.1)
    assert voice.spoken == []


def test_stop_requires_auth(cfg, memory):
    client, *_ = _make_client(cfg, memory)
    assert client.post("/stop").status_code == 401


def test_stop_triggers_shutdown_callback(cfg, memory):
    client, _, _, shutdown_calls = _make_client(cfg, memory)

    resp = client.post("/stop", headers=_auth(cfg))

    assert resp.status_code == 200
    _wait_until(lambda: shutdown_calls)
    assert shutdown_calls == [True]


def _wait_until(condition, timeout_s: float = 1.0) -> None:
    """voice.say / request_shutdown run on a background thread from /ask
    and /stop, so give them a moment to complete before asserting."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
