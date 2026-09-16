"""E.V.'s brain: turns a transcribed user utterance into a spoken-style reply.

Calls Claude with a system prompt made of two parts (see personality.py):
a stable, cached persona block, and a per-turn block of relevant facts
pulled from memory.py. Conversation history is stored in and replayed from
the same SQLite-backed memory so E.V. keeps context across restarts.
"""

from __future__ import annotations

import logging

import anthropic

from ev_assistant.config import Config
from ev_assistant.memory import Memory
from ev_assistant.personality import build_system_blocks

logger = logging.getLogger(__name__)

HISTORY_TURNS = 12


class Brain:
    def __init__(self, cfg: Config, memory: Memory):
        self.cfg = cfg
        self.memory = memory
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    def respond(self, user_text: str) -> str:
        """Get E.V.'s reply to `user_text` and record the exchange in memory."""
        memory_context = self.memory.format_context(user_text)
        system = build_system_blocks(self.cfg, memory_context)
        history = self.memory.recent_turns(limit=HISTORY_TURNS)
        messages = history + [{"role": "user", "content": user_text}]

        reply = self._call_claude(system, messages)

        self.memory.add_turn("user", user_text)
        self.memory.add_turn("assistant", reply)
        return reply

    def _call_claude(self, system: list[dict], messages: list[dict]) -> str:
        try:
            response = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.cfg.effort},
            )
        except anthropic.AuthenticationError:
            logger.exception("Anthropic authentication failed")
            return "I can't reach my brain right now - the Anthropic API key looks wrong or missing."
        except anthropic.RateLimitError as e:
            logger.warning("Rate limited: %s", e)
            return "I'm being rate limited right now. Give it a moment and ask again."
        except anthropic.APIConnectionError:
            logger.exception("Network error calling Anthropic")
            return "I can't reach the network right now, so I can't think that one through."
        except anthropic.APIStatusError:
            logger.exception("Anthropic API error")
            return "Something went wrong on my end talking to Claude. Worth checking the logs."
        except Exception:
            logger.exception("Unexpected error calling Claude")
            return "Something broke on my end and I'm not sure what. Worth checking the logs."

        if response.stop_reason == "refusal":
            return "I'm not going to help with that one."

        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or "I didn't come up with anything to say to that."
