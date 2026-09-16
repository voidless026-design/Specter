from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ev_assistant.providers import (
    ClaudeProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_chain,
    _to_openai_tools,
)
from ev_assistant.tools.executor import Executor, ToolResult


def test_to_openai_tools_shape():
    tools = [{"name": "open_app", "description": "open", "input_schema": {"type": "object"}}]
    out = _to_openai_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "open_app"
    assert out[0]["function"]["parameters"] == {"type": "object"}


# ---- chain selection ----

def test_default_chain_is_ollama_only(cfg):
    cfg.brain_provider = "ollama"
    chain = build_chain(cfg)
    assert [p.name for p in chain] == ["ollama"]


def test_claude_chain_has_ollama_fallback(cfg):
    cfg.brain_provider = "claude"
    chain = build_chain(cfg)
    assert chain[0].name == "claude"
    assert chain[-1].name == "ollama"


def test_offline_mode_forces_ollama(cfg):
    cfg.brain_provider = "claude"
    cfg.offline_mode = "offline"
    chain = build_chain(cfg)
    assert [p.name for p in chain] == ["ollama"]


# ---- OpenAI-compatible tool loop (mocked httpx) ----

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _msg(content=None, tool_calls=None):
    return {"choices": [{"message": {"content": content, "tool_calls": tool_calls}}]}


def test_openai_provider_plain_reply(cfg, monkeypatch):
    prov = OpenAICompatibleProvider("http://x/v1", "k", "m", "test")
    monkeypatch.setattr(
        "ev_assistant.providers.httpx.post",
        lambda *a, **k: _Resp(_msg(content="G'day, all sorted.")),
    )
    reply = prov.respond(cfg, "", [], "hi", [], Executor(cfg))
    assert reply == "G'day, all sorted."


def test_openai_provider_runs_a_tool(cfg, monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(_msg(tool_calls=[{
                "id": "t1", "function": {"name": "open_url", "arguments": json_dumps({"url": "x.com"})}}]))
        # second call: tool result should be in the messages
        assert any(m.get("role") == "tool" for m in json["messages"])
        return _Resp(_msg(content="Opened it."))

    monkeypatch.setattr("ev_assistant.providers.httpx.post", fake_post)

    executed = {}

    def fake_dispatch(name, args):
        executed["name"] = name
        return ToolResult(True, "opening")

    ex = Executor(cfg)
    ex.dispatch = fake_dispatch  # type: ignore
    prov = OpenAICompatibleProvider("http://x/v1", "k", "m", "test")
    reply = prov.respond(cfg, "", [], "open x.com", [
        {"name": "open_url", "description": "d", "input_schema": {"type": "object"}}], ex)

    assert executed["name"] == "open_url"
    assert reply == "Opened it."


def test_openai_provider_returns_none_on_connection_error(cfg, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("ev_assistant.providers.httpx.post", boom)
    prov = OpenAICompatibleProvider("http://x/v1", "k", "m", "test")
    assert prov.respond(cfg, "", [], "hi", [], Executor(cfg)) is None


def test_ollama_available_probes_tags(cfg, monkeypatch):
    cfg.ollama_model = "llama3.1"
    prov = OllamaProvider(cfg)
    monkeypatch.setattr("ev_assistant.providers.httpx.get", lambda *a, **k: _Resp({}))
    assert prov.available(cfg) is True

    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("ev_assistant.providers.httpx.get", boom)
    assert prov.available(cfg) is False


def json_dumps(d):
    return json.dumps(d)
