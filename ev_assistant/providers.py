"""Pluggable brains for E.V.

Three providers, one interface. Which one is primary is set in config
(`[brain] provider`); if it can't be reached, Brain falls back down the
chain to a local Ollama model and finally to reading offline notes.

- ClaudeProvider           - Anthropic Claude. Best quality, needs a paid key.
- OpenAICompatibleProvider - any OpenAI-style /chat/completions endpoint:
                             Ollama (local, no key), Groq, Google Gemini
                             (compat), OpenRouter, a local vLLM, etc.
- OllamaProvider           - OpenAICompatibleProvider pointed at local Ollama,
                             with a reachability check. No key, works offline.

Each `respond()` returns the spoken reply on success, or None to mean "I
couldn't do it - try the next brain." Both drive their own tool-use loop so
E.V. can act on the machine (open apps, run commands, ...) whichever brain
is active, provided the model supports tool calling.
"""

from __future__ import annotations

import json
import threading
import time
import logging

import httpx

from ev_assistant.config import Config, looks_like_real_key
from ev_assistant.personality import build_persona_text, build_system_blocks
from ev_assistant.tools.executor import Executor

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6


class BaseProvider:
    name = "base"

    def available(self, cfg: Config) -> bool:
        return False

    def respond(
        self,
        cfg: Config,
        context: str,
        history: list[dict],
        user_text: str,
        tools: list[dict],
        executor: Executor,
    ) -> str | None:
        raise NotImplementedError

    def complete(self, cfg: Config, system: str, user: str, max_tokens: int = 256) -> str | None:
        """One short, tool-free, persona-free call.

        Retrieval uses this for classification and query rewriting, where
        E.V.'s personality would actively get in the way - a sarcastic
        classifier is a broken classifier. Returns None if it didn't work,
        so the caller can fall back to rules.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeProvider(BaseProvider):
    name = "claude"

    def available(self, cfg: Config) -> bool:
        return looks_like_real_key(cfg.anthropic_api_key)

    def respond(self, cfg, context, history, user_text, tools, executor) -> str | None:
        import anthropic

        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        system = build_system_blocks(cfg, context)
        messages = list(history) + [{"role": "user", "content": user_text}]
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = client.messages.create(
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                    output_config={"effort": cfg.effort},
                )
                if resp.stop_reason == "refusal":
                    return "I'm not going to help with that one."
                if resp.stop_reason != "tool_use":
                    return _blocks_text(resp.content)
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        r = executor.dispatch(block.name, dict(block.input))
                        results.append({
                            "type": "tool_result", "tool_use_id": block.id,
                            "content": r.message, "is_error": not r.ok,
                        })
                messages.append({"role": "user", "content": results})
            final = client.messages.create(
                model=cfg.model, max_tokens=cfg.max_tokens, system=system,
                messages=messages, output_config={"effort": cfg.effort},
            )
            return _blocks_text(final.content)
        except anthropic.AuthenticationError:
            logger.warning("Claude auth failed; falling back to the next brain")
            return None
        except anthropic.APIConnectionError:
            logger.warning("Claude unreachable; falling back")
            return None
        except anthropic.RateLimitError:
            logger.warning("Claude rate limited; falling back")
            return None
        except Exception:
            logger.exception("Claude error; falling back")
            return None


    def complete(self, cfg, system, user, max_tokens=256) -> str | None:
        import anthropic

        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        try:
            resp = client.messages.create(
                # A small model is the right tool here. Note that `effort` is
                # rejected on Haiku 4.5, so utility calls send no output_config
                # at all - they don't need thinking depth anyway.
                model=cfg.utility_model or cfg.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if resp.stop_reason == "refusal":
                return None
            return _plain_text(resp.content)
        except Exception:
            logger.warning("Claude utility call failed", exc_info=logger.isEnabledFor(logging.DEBUG))
            return None


def _plain_text(content) -> str | None:
    """Text blocks joined, or None. Unlike _blocks_text, no 'Done.' filler."""
    text = " ".join(b.text for b in content if getattr(b, "type", None) == "text").strip()
    return text or None


def _blocks_text(content) -> str:
    return " ".join(b.text for b in content if getattr(b, "type", None) == "text").strip() or "Done."


# ---------------------------------------------------------------------------
# OpenAI-compatible (Ollama / Groq / Gemini / OpenRouter / ...)
# ---------------------------------------------------------------------------


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ]


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str, model: str, name: str = "openai"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name

    def available(self, cfg: Config) -> bool:
        return bool(self.base_url and self.model)

    def _system_text(self, cfg: Config, context: str) -> str:
        persona = build_persona_text(cfg)
        return f"{persona}\n\n{context}" if context else persona

    def respond(self, cfg, context, history, user_text, tools, executor) -> str | None:
        messages = [{"role": "system", "content": self._system_text(cfg, context)}]
        messages += [{"role": t["role"], "content": t["content"]} for t in history]
        messages.append({"role": "user", "content": user_text})
        oa_tools = _to_openai_tools(tools) if tools else None
        headers = {"Authorization": f"Bearer {self.api_key or 'none'}"}

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                body = {"model": self.model, "messages": messages, "max_tokens": cfg.max_tokens, "stream": False}
                if oa_tools:
                    body["tools"] = oa_tools
                resp = httpx.post(
                    f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=120
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]["message"]
                tool_calls = choice.get("tool_calls")
                if not tool_calls:
                    return (choice.get("content") or "").strip() or "Done."
                # Echo the assistant turn (with its tool calls), then results.
                messages.append({
                    "role": "assistant",
                    "content": choice.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = executor.dispatch(fn.get("name", ""), args)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.get("id", ""), "content": result.message,
                    })
            # Ran out of rounds - one plain pass for a summary.
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages, "max_tokens": cfg.max_tokens, "stream": False},
                headers=headers, timeout=120,
            )
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"].get("content") or "").strip() or "Done."
        except httpx.HTTPStatusError as e:
            logger.warning("%s HTTP %s; falling back", self.name, e.response.status_code)
            return None
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
            logger.warning("%s unreachable; falling back", self.name)
            return None
        except Exception:
            logger.exception("%s error; falling back", self.name)
            return None


    def complete(self, cfg, system, user, max_tokens=256) -> str | None:
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {self.api_key or 'none'}"},
                timeout=30,
            )
            resp.raise_for_status()
            text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            return text or None
        except Exception:
            logger.warning("%s utility call failed", self.name,
                           exc_info=logger.isEnabledFor(logging.DEBUG))
            return None


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self, cfg: Config):
        super().__init__(
            base_url=f"{cfg.ollama_host.rstrip('/')}/v1",
            api_key="ollama",
            model=cfg.ollama_model,
            name="ollama",
        )
        self._ollama_host = cfg.ollama_host.rstrip("/")

    def available(self, cfg: Config) -> bool:
        if not cfg.ollama_model:
            return False
        try:
            httpx.get(f"{self._ollama_host}/api/tags", timeout=1.5).raise_for_status()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Chain selection
# ---------------------------------------------------------------------------


def build_chain(cfg: Config) -> list[BaseProvider]:
    """Ordered brains to try. Primary first, then a local Ollama fallback."""
    primary_name = cfg.brain_provider
    ollama = OllamaProvider(cfg)

    def primary() -> BaseProvider:
        if primary_name == "claude":
            return ClaudeProvider()
        if primary_name == "openai":
            return OpenAICompatibleProvider(cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model, "openai")
        return ollama  # default

    # In forced-offline mode, only the local Ollama brain is allowed.
    if cfg.offline_mode == "offline":
        return [ollama]

    chain: list[BaseProvider] = [primary()]
    if primary_name != "ollama":
        chain.append(ollama)  # local, no-key safety net
    return chain


# Utility calls sit on the retrieval hot path, and probing a provider that
# isn't running costs a connection timeout every time. Retrieval has a 400ms
# p50 budget, and three refused connections eat all of it, so the answer is
# remembered briefly. Short enough that starting Ollama is noticed quickly.
_AVAILABILITY_TTL = 30.0
_availability: dict[str, tuple[float, bool]] = {}
_availability_lock = threading.Lock()


def _available_cached(provider: BaseProvider, cfg: Config) -> bool:
    key = f"{provider.name}:{getattr(provider, 'model', '')}"
    now = time.monotonic()
    with _availability_lock:
        cached = _availability.get(key)
        if cached and now - cached[0] < _AVAILABILITY_TTL:
            return cached[1]
    result = provider.available(cfg)
    with _availability_lock:
        _availability[key] = (now, result)
    return result


def forget_availability() -> None:
    """Drop the cache - for tests, and for `ev doctor` wanting live answers."""
    with _availability_lock:
        _availability.clear()


def complete(cfg: Config, system: str, user: str, max_tokens: int = 256) -> str | None:
    """Run a short utility prompt on the first brain that answers.

    Used by retrieval for classification and rewriting. Returns None when no
    brain is reachable, which callers treat as "fall back to rules" rather
    than as an error - retrieval has to work with no model at all.
    """
    for provider in build_chain(cfg):
        if not _available_cached(provider, cfg):
            continue
        reply = provider.complete(cfg, system, user, max_tokens)
        if reply:
            return reply
    return None
