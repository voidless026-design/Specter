"""E.V.'s brain: turns a transcribed utterance into a spoken reply and runs
any system actions the request calls for.

It tries the configured brain provider (Ollama by default, or Claude, or an
OpenAI-compatible endpoint), falling back to a local Ollama model and then to
her offline notes if the primary can't be reached. Whichever brain is active
gets the same persona, conversation history, and relevant context (recent
feed facts plus anything taught with `ev learn`), and can drive the
system-control tools under the configured permission tier.
"""

from __future__ import annotations

import logging

from ev_assistant.config import Config
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory
from ev_assistant.offline import OfflineBrain
from ev_assistant.providers import build_chain
from ev_assistant.tools.executor import ConfirmFn, Executor
from ev_assistant.tools.specs import tools_for_tier

logger = logging.getLogger(__name__)

HISTORY_TURNS = 12


class Brain:
    def __init__(self, cfg: Config, memory: Memory, knowledge: Knowledge | None = None):
        self.cfg = cfg
        self.memory = memory
        self.knowledge = knowledge or Knowledge(cfg.knowledge_path)

    def respond(self, user_text: str, confirm: ConfirmFn | None = None) -> str:
        """Answer/act on `user_text`, recording the exchange in memory."""
        executor = Executor(self.cfg, confirm=confirm)
        context = self._context(user_text)
        history = self.memory.recent_turns(limit=HISTORY_TURNS)
        tools = tools_for_tier(self.cfg.permission_tier)

        reply = None
        used_provider = None
        for provider in build_chain(self.cfg):
            if not provider.available(self.cfg):
                continue
            reply = provider.respond(self.cfg, context, history, user_text, tools, executor)
            if reply is not None:
                used_provider = provider.name
                break

        if reply is None:
            # No brain could answer - use the offline rule/knowledge fallback.
            reply = OfflineBrain(
                self.cfg, self.knowledge, executor, reason=self._offline_reason()
            ).respond(user_text)
        else:
            logger.info("Answered via %s", used_provider)

        self.memory.add_turn("user", user_text)
        self.memory.add_turn("assistant", reply)
        return reply

    def _context(self, query: str) -> str:
        """Combine recent feed facts with relevant learned knowledge."""
        parts = []
        facts = self.memory.format_context(query)
        if facts:
            parts.append(facts)
        learned = self.knowledge.format_context(query, limit=4)
        if learned:
            parts.append("Relevant notes E.V. has been taught:\n" + learned)
        return "\n\n".join(parts)

    def _offline_reason(self) -> str:
        # If the primary brain is a cloud one that needs a key we don't have,
        # point the fallback message at fixing that; otherwise it's a real
        # offline/connection situation.
        from ev_assistant.config import looks_like_real_key

        if self.cfg.brain_provider == "claude" and not looks_like_real_key(self.cfg.anthropic_api_key):
            return "no_key"
        if self.cfg.brain_provider == "openai" and not self.cfg.openai_api_key:
            return "no_key"
        return "offline"
