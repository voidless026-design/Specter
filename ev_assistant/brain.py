"""E.V.'s brain: turns a transcribed utterance into a spoken reply, and runs
any system actions the request calls for.

Online, it calls Claude with the system-control tools and drives the
tool-use loop (Claude decides which tools to run; the Executor runs them
under the configured permission tier and confirmation gate). Offline - no
internet, or offline mode forced - it hands off to OfflineBrain for
rule-based commands and local knowledge answering.
"""

from __future__ import annotations

import json
import logging

import anthropic

from ev_assistant.config import Config, looks_like_real_key
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory
from ev_assistant.net import is_online
from ev_assistant.offline import OfflineBrain
from ev_assistant.personality import build_system_blocks
from ev_assistant.tools.executor import ConfirmFn, Executor
from ev_assistant.tools.specs import tools_for_tier

logger = logging.getLogger(__name__)

HISTORY_TURNS = 12
MAX_TOOL_ROUNDS = 6


class Brain:
    def __init__(self, cfg: Config, memory: Memory, knowledge: Knowledge | None = None):
        self.cfg = cfg
        self.memory = memory
        self.knowledge = knowledge or Knowledge(cfg.knowledge_path)
        # Only build a client for a real key - a placeholder/empty key would
        # otherwise produce a confusing 401 instead of a clean offline fallback.
        self.client = (
            anthropic.Anthropic(api_key=cfg.anthropic_api_key)
            if looks_like_real_key(cfg.anthropic_api_key)
            else None
        )

    def respond(self, user_text: str, confirm: ConfirmFn | None = None) -> str:
        """Answer/act on `user_text`, recording the exchange in memory."""
        executor = Executor(self.cfg, confirm=confirm)
        if self._use_online():
            reply = self._respond_online(user_text, executor)
        else:
            reply = OfflineBrain(
                self.cfg, self.knowledge, executor, reason=self._offline_reason()
            ).respond(user_text)

        self.memory.add_turn("user", user_text)
        self.memory.add_turn("assistant", reply)
        return reply

    def _offline_reason(self) -> str:
        # Distinguish "no valid API key" from "genuinely offline" so the
        # spoken guidance points at the right fix.
        if self.client is None and self.cfg.offline_mode != "offline":
            return "no_key"
        return "offline"

    def _use_online(self) -> bool:
        if self.cfg.offline_mode == "offline" or self.client is None:
            return False
        if self.cfg.offline_mode == "online":
            return True
        return is_online()  # auto

    def _respond_online(self, user_text: str, executor: Executor) -> str:
        memory_context = self.memory.format_context(user_text)
        system = build_system_blocks(self.cfg, memory_context)
        tools = tools_for_tier(self.cfg.permission_tier)
        messages: list[dict] = self.memory.recent_turns(limit=HISTORY_TURNS)
        messages.append({"role": "user", "content": user_text})

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self.client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                    output_config={"effort": self.cfg.effort},
                )
                if response.stop_reason == "refusal":
                    return "I'm not going to help with that one."
                if response.stop_reason != "tool_use":
                    return self._final_text(response)

                # Execute every requested tool, feed results back, loop.
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = executor.dispatch(block.name, dict(block.input))
                        logger.info("Tool %s -> %s", block.name, result.ok)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.message,
                                "is_error": not result.ok,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})

            # Ran out of tool rounds - make one last plain request for a summary.
            final = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.cfg.effort},
            )
            return self._final_text(final)
        except anthropic.AuthenticationError:
            logger.exception("Anthropic authentication failed")
            return "I can't reach my brain right now - the Anthropic API key looks wrong or missing."
        except anthropic.RateLimitError:
            return "I'm being rate limited right now. Give it a moment and ask again."
        except anthropic.APIConnectionError:
            # Network dropped mid-request: fall back to the offline brain.
            logger.warning("Network error online; falling back to offline brain")
            return OfflineBrain(self.cfg, self.knowledge, executor).respond(user_text)
        except anthropic.APIStatusError:
            logger.exception("Anthropic API error")
            return "Something went wrong talking to Claude. Worth checking the logs."
        except Exception:
            logger.exception("Unexpected error in online brain")
            return "Something broke on my end and I'm not sure what. Worth checking the logs."

    @staticmethod
    def _final_text(response) -> str:
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        return text or "Done."
