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


def _tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "t1"):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input, id=tool_id)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _fake_request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _brain(cfg, memory, knowledge):
    return Brain(cfg, memory, knowledge)


def test_respond_returns_text_and_records_turns(cfg, memory, knowledge, monkeypatch):
    brain = _brain(cfg, memory, knowledge)
    monkeypatch.setattr(brain.client.messages, "create", lambda **kw: _fake_response("Sure, here you go."))

    reply = brain.respond("what time is it")

    assert reply == "Sure, here you go."
    turns = memory.recent_turns()
    assert turns[-2] == {"role": "user", "content": "what time is it"}
    assert turns[-1] == {"role": "assistant", "content": "Sure, here you go."}


def test_respond_passes_model_effort_and_tools(cfg, memory, knowledge, monkeypatch):
    cfg.model = "claude-sonnet-5"
    cfg.effort = "high"
    brain = _brain(cfg, memory, knowledge)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(brain.client.messages, "create", fake_create)
    brain.respond("hi")

    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_config"] == {"effort": "high"}
    assert captured["max_tokens"] == cfg.max_tokens
    # Tools are offered (standard tier by default includes several).
    assert any(t["name"] == "open_app" for t in captured["tools"])


def test_respond_includes_memory_context_in_system(cfg, memory, knowledge, monkeypatch):
    memory.add_fact("weather", "It's sunny and 25C", external_id="w1")
    brain = _brain(cfg, memory, knowledge)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(brain.client.messages, "create", fake_create)
    brain.respond("what's the weather")

    assert any("sunny" in block["text"].lower() for block in captured["system"])


def test_respond_runs_a_tool_then_summarizes(cfg, memory, knowledge, monkeypatch):
    cfg.permission_tier = "safe"
    brain = _brain(cfg, memory, knowledge)

    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_use_response("open_url", {"url": "example.com"})
        # Second call: the tool result is now in the messages, reply normally.
        assert kwargs["messages"][-1]["role"] == "user"
        assert kwargs["messages"][-1]["content"][0]["type"] == "tool_result"
        return _fake_response("Opened it.")

    monkeypatch.setattr(brain.client.messages, "create", fake_create)

    executed = {}
    from ev_assistant.tools import executor as executor_mod

    def fake_dispatch(self, name, args):
        executed["name"] = name
        executed["args"] = args
        return executor_mod.ToolResult(True, "Opening example.com.")

    monkeypatch.setattr(executor_mod.Executor, "dispatch", fake_dispatch)

    reply = brain.respond("open example.com")

    assert executed["name"] == "open_url"
    assert reply == "Opened it."


def test_respond_on_refusal_declines(cfg, memory, knowledge, monkeypatch):
    brain = _brain(cfg, memory, knowledge)
    monkeypatch.setattr(brain.client.messages, "create", lambda **kw: _fake_response("", stop_reason="refusal"))
    assert "not going to help" in brain.respond("do something bad").lower()


def test_auth_error_is_handled(cfg, memory, knowledge, monkeypatch):
    brain = _brain(cfg, memory, knowledge)

    def raise_auth(**kwargs):
        raise anthropic.AuthenticationError(
            "bad key", response=httpx2.Response(401, request=_fake_request()), body=None
        )

    monkeypatch.setattr(brain.client.messages, "create", raise_auth)
    assert "api key" in brain.respond("hi").lower()
    assert memory.recent_turns()[-2]["content"] == "hi"


def test_connection_error_falls_back_to_offline(cfg, memory, knowledge, monkeypatch):
    knowledge.add_document("Fire", "notes", "To start a fire, gather dry tinder and a spark.")
    brain = _brain(cfg, memory, knowledge)

    def raise_conn(**kwargs):
        raise anthropic.APIConnectionError(request=_fake_request())

    monkeypatch.setattr(brain.client.messages, "create", raise_conn)

    reply = brain.respond("how do I start a fire")
    # The offline brain answers from the knowledge base instead of erroring.
    assert "tinder" in reply.lower()


def test_offline_mode_uses_offline_brain(cfg, memory, knowledge, monkeypatch):
    cfg.offline_mode = "offline"
    knowledge.add_document("Water", "notes", "Boil water for one minute to make it safe to drink.")
    brain = _brain(cfg, memory, knowledge)

    # Even with a client present, offline mode must not call it.
    def boom(**kwargs):
        raise AssertionError("online path should not run in offline mode")

    monkeypatch.setattr(brain.client.messages, "create", boom)

    reply = brain.respond("how do I make water safe")
    assert "boil" in reply.lower()
