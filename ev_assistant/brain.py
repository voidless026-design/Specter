"""E.V.'s brain: turns a transcribed utterance into a spoken reply and runs
any system actions the request calls for.

It tries the configured brain provider (Ollama by default, or Claude, or an
OpenAI-compatible endpoint), falling back to a local Ollama model and then to
her offline notes if the primary can't be reached. Whichever brain is active
gets the same persona, conversation history, and context.

Context now comes from the retrieval layer: the question is planned, searched
across keyword and vector indexes, reranked, and assembled into a labelled
block the brain can cite from - or, when nothing clears the relevance floor,
an explicit note that the store was searched and had nothing. Recent feed
facts from memory are still blended in alongside.

Replies keep their `[S1]` source tags for text, and `spoken()` strips them
for the voice path - the tags are for citing in writing, not for reading
aloud.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ev_assistant import context as context_module
from ev_assistant.config import Config
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory
from ev_assistant.offline import OfflineBrain
from ev_assistant.providers import build_chain
from ev_assistant.tools.executor import ConfirmFn, Executor
from ev_assistant.tools.specs import tools_for_tier

logger = logging.getLogger(__name__)

HISTORY_TURNS = 12

# "[S1]", "[S2]" - and any run of them, with the spacing they leave behind.
_TAG_RE = re.compile(r"\s*\[S\d+\](?:\s*\[S\d+\])*")


def strip_citations(text: str) -> str:
    """Remove source tags for speech. "...boil it [S1]." -> "...boil it."."""
    cleaned = _TAG_RE.sub("", text or "")
    # Tidy the punctuation the tags were sitting next to.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def cited_tags(text: str) -> list[str]:
    """Which source tags the reply actually used, in order of first use."""
    return list(dict.fromkeys(re.findall(r"\[S\d+\]", text or "")))


@dataclass
class BrainReply:
    """A reply plus everything needed to explain where it came from."""

    text: str
    provider: str = ""
    sources: list[dict] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)

    @property
    def spoken(self) -> str:
        return strip_citations(self.text)

    @property
    def citations(self) -> list[dict]:
        """The sources this reply actually cited, not merely those offered."""
        used = set(cited_tags(self.text))
        return [s for s in self.sources if s["tag"] in used]


class Brain:
    def __init__(self, cfg: Config, memory: Memory, knowledge: Knowledge | None = None,
                 store=None, retriever=None):
        self.cfg = cfg
        self.memory = memory
        self.knowledge = knowledge or Knowledge(cfg.knowledge_path)
        self._store = store
        self._retriever = retriever

    # -- lazily built collaborators ------------------------------------

    @property
    def store(self):
        if self._store is None:
            from ev_assistant.migrate import migrate_if_needed
            from ev_assistant.store import Store

            self._store = Store(self.cfg.store_path)
            # Carry an existing `ev learn` library across the first time, so
            # nobody's knowledge base vanishes on upgrade.
            migrate_if_needed(self.cfg, self._store)
        return self._store

    @property
    def retriever(self):
        if self._retriever is None:
            from ev_assistant.retrieval import Retriever

            self._retriever = Retriever(self.cfg, self.store)
        return self._retriever

    # -- answering -----------------------------------------------------

    def respond(self, user_text: str, confirm: ConfirmFn | None = None) -> str:
        """Answer/act on `user_text`. Returns the text reply, tags included."""
        return self.respond_detailed(user_text, confirm).text

    def respond_detailed(self, user_text: str, confirm: ConfirmFn | None = None
                         ) -> BrainReply:
        """The same, with citations and a retrieval trace attached."""
        executor = Executor(self.cfg, confirm=confirm)
        history = self.memory.recent_turns(limit=HISTORY_TURNS)
        block, trace = self._retrieve(user_text, history)
        context = self._context(user_text, block)
        tools = tools_for_tier(self.cfg.permission_tier)

        reply = None
        used_provider = ""
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
            used_provider = "offline"
        else:
            logger.info("Answered via %s", used_provider)

        self.memory.add_turn("user", user_text)
        self.memory.add_turn("assistant", reply)
        return BrainReply(text=reply, provider=used_provider,
                          sources=block.sources, retrieval=trace)

    # -- context -------------------------------------------------------

    def _retrieve(self, user_text: str, history: list[dict]):
        """Search the store and assemble the context block. Never fatal."""
        try:
            result = self.retriever.retrieve(
                user_text, k=int(getattr(self.cfg, "retrieval_k", 8)), history=history
            )
            block = context_module.build(result, self.cfg)
            return block, self._trace(result, block)
        except Exception:
            # Retrieval failing must not cost the user an answer - she just
            # answers from her own knowledge instead.
            logger.exception("Retrieval failed; answering without the store")
            return context_module.ContextBlock(), {"error": "retrieval failed"}

    @staticmethod
    def _trace(result, block) -> dict:
        plan = result.plan
        return {
            "reason": result.reason,
            "question": plan.question if plan else "",
            "rewritten": bool(plan and plan.rewritten),
            "classified_by": plan.classified_by if plan else "",
            "variants": [{"kind": v.kind, "text": v.text} for v in (plan.variants if plan else [])],
            "namespaces": plan.namespace_plan.describe() if plan else "",
            "chunks": len(result.chunks),
            "facts": len(result.facts),
            "floor": result.trace.floor,
            "best_score": result.trace.best_score,
            "dropped_below_floor": result.trace.dropped_below_floor,
            "reranker": result.trace.reranker,
            "timings_ms": result.trace.timings_ms,
            "total_ms": result.trace.total_ms,
            "context_tokens": block.tokens,
            "sources": block.sources,
        }

    def _context(self, query: str, block=None) -> str:
        """Retrieved material plus recent feed facts, in that order."""
        parts = []
        if block is not None and block.text:
            parts.append(block.text)
        facts = self.memory.format_context(query)
        if facts:
            parts.append(facts)
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
