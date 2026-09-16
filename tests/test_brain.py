from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from ev_assistant.brain import Brain


def _fake_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def _fake_request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_respond_returns_text_and_records_turns(cfg, memory, monkeypatch):
    brain = Brain(cfg, memory)
    monkeypatch.setattr(brain.client.messages, "create", lambda **kwargs: _fake_response("Sure, here you go."))

    reply = brain.respond("what time is it")

    assert reply == "Sure, here you go."
    turns = memory.recent_turns()
    assert turns[-2] == {"role": "user", "content": "what time is it"}
    assert turns[-1] == {"role": "assistant", "content": "Sure, here you go."}


def test_respond_passes_configured_model_and_effort(cfg, memory, monkeypatch):
    cfg.model = "claude-sonnet-5"
    cfg.effort = "high"
    brain = Brain(cfg, memory)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(brain.client.messages, "create", fake_create)
    brain.respond("hi")

    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_config"] == {"effort": "high"}
    assert captured["max_tokens"] == cfg.max_tokens


def test_respond_includes_memory_context_as_second_system_block(cfg, memory, monkeypatch):
    memory.add_fact("weather", "It's sunny and 25C", external_id="w1")
    brain = Brain(cfg, memory)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(brain.client.messages, "create", fake_create)
    brain.respond("what's the weather")

    system_blocks = captured["system"]
    assert any("sunny" in block["text"].lower() for block in system_blocks)


def test_respond_on_refusal_declines_gracefully(cfg, memory, monkeypatch):
    brain = Brain(cfg, memory)
    monkeypatch.setattr(
        brain.client.messages, "create", lambda **kwargs: _fake_response("", stop_reason="refusal")
    )

    reply = brain.respond("do something bad")

    assert "not going to help" in reply.lower()


def test_respond_falls_back_to_default_text_when_content_is_empty(cfg, memory, monkeypatch):
    brain = Brain(cfg, memory)
    monkeypatch.setattr(brain.client.messages, "create", lambda **kwargs: _fake_response(""))

    reply = brain.respond("hi")

    assert reply  # non-empty fallback line, not a raised exception or blank string


@pytest.mark.parametrize(
    ("exc_factory", "expected_snippet"),
    [
        (lambda: anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=_fake_request()), body=None), "api key"),
        (lambda: anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=_fake_request()), body=None), "rate limited"),
        (lambda: anthropic.APIConnectionError(request=_fake_request()), "reach the network"),
        (lambda: anthropic.APIStatusError("server error", response=httpx2.Response(500, request=_fake_request()), body=None), "went wrong"),
        (lambda: RuntimeError("something unrelated broke"), "broke"),
    ],
)
def test_respond_handles_api_errors_without_raising(cfg, memory, monkeypatch, exc_factory, expected_snippet):
    brain = Brain(cfg, memory)

    def raise_error(**kwargs):
        raise exc_factory()

    monkeypatch.setattr(brain.client.messages, "create", raise_error)

    reply = brain.respond("hi")

    assert expected_snippet in reply.lower()
    # Even on failure, the exchange is still recorded so the daemon's
    # conversation history stays consistent with what was actually said.
    turns = memory.recent_turns()
    assert turns[-2]["content"] == "hi"
