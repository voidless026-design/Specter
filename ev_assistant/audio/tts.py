"""Text-to-speech output for E.V., via pyttsx3 (offline; SAPI5 on Windows,
espeak-ng on Fedora/Linux - see README for higher-quality voice options).
"""

from __future__ import annotations

import threading

import pyttsx3


class Voice:
    def __init__(self, rate: int = 175, voice_id: str | None = None):
        self._lock = threading.Lock()
        self._rate = rate
        self._voice_id = voice_id

    def say(self, text: str) -> None:
        """Speak `text` and block until finished. Safe to call from one thread at a time."""
        if not text:
            return
        with self._lock:
            # A fresh engine per call is deliberate: some pyttsx3 drivers
            # (SAPI5, nsss) misbehave when runAndWait() is called more than
            # once on the same engine instance across a long-lived process.
            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", self._rate)
                if self._voice_id:
                    engine.setProperty("voice", self._voice_id)
                engine.say(text)
                engine.runAndWait()
            finally:
                engine.stop()

    @staticmethod
    def list_voices() -> list[dict]:
        engine = pyttsx3.init()
        try:
            return [{"id": v.id, "name": v.name} for v in engine.getProperty("voices")]
        finally:
            engine.stop()
