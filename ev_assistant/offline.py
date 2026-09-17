"""E.V.'s offline brain, used when there's no internet (or by choice).

Two jobs, no cloud:

1. Voice control via a small rule-based intent parser - "open firefox",
   "close spotify", "volume 40", "launch jegeo", "next track" - so the
   things you'd most want hands-free still work with the network down.
2. Question answering from the local knowledge base. If you've configured a
   local Ollama model it's used to phrase a real answer from the retrieved
   passages; otherwise E.V. reads back the most relevant passage directly.

This is deliberately simpler and more literal than the online path - a
small local model can't match Claude, so offline E.V. is a capable
reference tool and command runner rather than a free-flowing conversation.
"""

from __future__ import annotations

import logging
import re

import httpx

from ev_assistant.config import Config
from ev_assistant.knowledge import Knowledge
from ev_assistant.tools.executor import Executor

logger = logging.getLogger(__name__)

# Commands must be imperative: the verb starts the utterance. This keeps
# questions like "how do I START a fire" or "what should I OPEN with" from
# being mistaken for "launch an app".
_APP_OPEN_RE = re.compile(r"^\s*(?:open|launch|start|run)\s+(.+)", re.IGNORECASE)
_APP_CLOSE_RE = re.compile(r"^\s*(?:close|quit|kill|exit)\s+(.+)", re.IGNORECASE)
_VOLUME_RE = re.compile(r"\bvolume\s+(?:to\s+)?(\d{1,3})", re.IGNORECASE)
_WEBSITE_HINT = re.compile(r"\.(com|org|net|io|gov|edu|au|co)\b|\bwebsite\b", re.IGNORECASE)

# Utterances that open with one of these are questions - answer them, don't
# try to run them as a command.
_QUESTION_START = re.compile(
    r"^\s*(?:how|what|why|when|who|where|which|whose|is|are|am|do|does|did|"
    r"can|could|should|would|will|tell|explain|describe|define|give)\b",
    re.IGNORECASE,
)

# If the app/close target starts with one of these it's almost certainly a
# noun phrase in a sentence ("start A fire"), not an application name.
_TARGET_FILLERS = {"a", "an", "the", "my", "some", "this", "that", "it", "up", "of", "your"}

_MEDIA_WORDS = {
    "play": "play",
    "pause": "pause",
    "resume": "play",
    "next": "next",
    "skip": "next",
    "previous": "previous",
    "back": "previous",
}


class OfflineBrain:
    def __init__(
        self,
        cfg: Config,
        knowledge: Knowledge,
        executor: Executor,
        reason: str = "offline",
    ):
        self.cfg = cfg
        self.knowledge = knowledge
        self.executor = executor
        # "offline" (no network) or "no_key" (network fine but no valid API
        # key) - changes the message when she can't answer a question.
        self.reason = reason

    def respond(self, text: str) -> str:
        command_reply = self._try_command(text)
        if command_reply is not None:
            return command_reply
        return self._answer(text)

    # -- rule-based command parsing -----------------------------------

    def _try_command(self, text: str) -> str | None:
        lowered = text.lower().strip()

        # Questions are never commands - let them fall through to answering.
        if _QUESTION_START.match(lowered):
            return None

        if "jegeo" in lowered and re.match(r"^\s*(open|launch|start|run)\b", lowered):
            return self.executor.dispatch("launch_jegeo", {}).message

        vol = _VOLUME_RE.search(lowered)
        if vol:
            return self.executor.dispatch("set_volume", {"percent": int(vol.group(1))}).message
        if re.match(r"^\s*mute\b", lowered):
            return self.executor.dispatch("set_volume", {"percent": 0}).message

        # Single-word media transport ("next", "pause") or "play music".
        for word, action in _MEDIA_WORDS.items():
            if re.search(rf"\b{word}\b", lowered) and (
                "music" in lowered or "song" in lowered or "track" in lowered or lowered == word
            ):
                return self.executor.dispatch("media_control", {"action": action}).message

        open_match = _APP_OPEN_RE.match(text)
        if open_match:
            target = open_match.group(1).strip().strip(".!?")
            first = target.split()[0] if target.split() else ""
            if first.lower() in _TARGET_FILLERS:
                return None  # "start a fire" - a noun phrase, not an app
            if _WEBSITE_HINT.search(target):
                return self.executor.dispatch("open_url", {"url": first}).message
            return self.executor.dispatch("open_app", {"name": first}).message

        close_match = _APP_CLOSE_RE.match(text)
        if close_match:
            target = close_match.group(1).strip().strip(".!?")
            first = target.split()[0] if target.split() else ""
            if first.lower() in _TARGET_FILLERS:
                return None
            return self.executor.dispatch("close_app", {"name": first}).message

        return None

    # -- knowledge answering ------------------------------------------

    def _answer(self, text: str) -> str:
        context = self.knowledge.format_context(text, limit=5)
        if self.cfg.ollama_model:
            answer = self._ask_ollama(text, context)
            if answer:
                return answer
        if not context:
            if self.reason == "no_key":
                return (
                    "I can run commands and read my offline notes, but my brain isn't set up "
                    "to converse yet. Either install Ollama for a free local brain, or add an "
                    "API key. Run ev doctor and I'll tell you exactly what's missing."
                )
            return (
                "I'm offline and I don't have anything on that in my knowledge base yet. "
                "Teach me with 'ev learn' while you're online and I'll remember it for next time."
            )
        # Retrieval-only: read back the most relevant passage, trimmed.
        top = context.split("\n\n")[0]
        top = re.sub(r"^\[.*?\]\s*", "", top)
        if len(top) > 600:
            top = top[:600].rsplit(" ", 1)[0] + "..."
        return f"From what you've taught me: {top}"

    def _ask_ollama(self, question: str, context: str) -> str | None:
        prompt = (
            "You are E.V., a concise offline assistant. Answer the question using the "
            "reference material if it's relevant. If it doesn't cover the question, say so "
            "briefly.\n\n"
            f"Reference material:\n{context or '(none)'}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        try:
            resp = httpx.post(
                f"{self.cfg.ollama_host}/api/generate",
                json={"model": self.cfg.ollama_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return (resp.json().get("response") or "").strip() or None
        except Exception:
            logger.exception("Ollama call failed")
            return None
