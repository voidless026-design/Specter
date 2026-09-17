from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ev_assistant.providers import (
    BaseProvider,
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


# ---- utility completions (retrieval's classify / rewrite / HyDE calls) ----


def test_utility_completion_is_tool_free_and_persona_free(cfg, monkeypatch):
    sent = {}

    def capture(url, **kwargs):
        sent["url"] = url
        sent["body"] = kwargs["json"]
        return _Resp(_msg(content="SEARCH"))

    monkeypatch.setattr("ev_assistant.providers.httpx.post", capture)
    prov = OpenAICompatibleProvider("http://x/v1", "k", "m", "test")

    assert prov.complete(cfg, "You classify questions.", "how do I purify water", 8) == "SEARCH"
    assert "tools" not in sent["body"]
    assert sent["body"]["max_tokens"] == 8
    assert sent["body"]["messages"][0] == {
        "role": "system", "content": "You classify questions.",
    }
    # E.V.'s persona must not leak in - a sarcastic classifier is a broken one.
    assert "sarcas" not in json.dumps(sent["body"]).lower()


def test_utility_completion_returns_none_on_empty_or_failure(cfg, monkeypatch):
    prov = OpenAICompatibleProvider("http://x/v1", "k", "m", "test")

    monkeypatch.setattr("ev_assistant.providers.httpx.post",
                        lambda *a, **k: _Resp(_msg(content="   ")))
    assert prov.complete(cfg, "s", "u") is None

    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("ev_assistant.providers.httpx.post", boom)
    assert prov.complete(cfg, "s", "u") is None


def test_complete_walks_the_chain_to_the_first_brain_that_answers(cfg, monkeypatch):
    from ev_assistant import providers

    class Dead(BaseProvider):
        name = "dead"

        def available(self, cfg):
            return True

        def complete(self, cfg, system, user, max_tokens=256):
            return None

    class Alive(BaseProvider):
        name = "alive"

        def available(self, cfg):
            return True

        def complete(self, cfg, system, user, max_tokens=256):
            return "SKIP"

    class Unavailable(BaseProvider):
        name = "unavailable"

        def available(self, cfg):
            return False

        def complete(self, cfg, system, user, max_tokens=256):
            raise AssertionError("must not be called when unavailable")

    monkeypatch.setattr(providers, "build_chain",
                        lambda c: [Unavailable(), Dead(), Alive()])
    assert providers.complete(cfg, "s", "u") == "SKIP"

    monkeypatch.setattr(providers, "build_chain", lambda c: [Dead()])
    assert providers.complete(cfg, "s", "u") is None

    monkeypatch.setattr(providers, "build_chain", lambda c: [])
    assert providers.complete(cfg, "s", "u") is None


def test_utility_model_defaults_to_a_small_one(cfg):
    # Classification and rewriting run on every question; they shouldn't cost
    # what the main model costs.
    assert cfg.utility_model == "claude-haiku-4-5"
    assert cfg.utility_model != cfg.model
