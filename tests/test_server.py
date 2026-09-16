from __future__ import annotations

import time

from fastapi.testclient import TestClient

from ev_assistant.bus import StateBus
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.knowledge import Knowledge
from ev_assistant.server import DaemonStatus, create_app


class FakeBrain:
    def __init__(self):
        self.calls = []

    def respond(self, text: str, confirm=None) -> str:
        self.calls.append((text, confirm))
        return f"echo: {text}"


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
    assert body["model"] == cfg.model


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
    assert resp.json() == {"reply": "echo: hello"}
    assert brain.calls[0][0] == "hello"
    _wait_until(lambda: voice.spoken)
    assert voice.spoken == ["echo: hello"]


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
