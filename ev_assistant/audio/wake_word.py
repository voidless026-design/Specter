"""Continuous wake-word spotting for "Hey E.V." (and phonetic variants).

There's no dedicated wake-word engine here - no off-the-shelf custom model
for a phrase like "E.V." exists, and training one (Porcupine, openWakeWord)
needs a synthetic-audio training pipeline that doesn't belong in this repo.
Instead this runs Vosk's small offline speech model continuously and checks
every partial transcript against the configured wake phrases. It costs more
idle CPU than a dedicated keyword spotter, but it works out of the box with
zero training. See the README for the openWakeWord upgrade path if idle CPU
usage matters to you.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model

SAMPLE_RATE = 16000

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


class WakeWordListener:
    def __init__(self, model_dir: Path, wake_phrases: list[str], device: str | int | None = None):
        self.model = Model(str(model_dir))
        self.wake_phrases = [normalize(p) for p in wake_phrases if p.strip()]
        self.device = device

    def listen(self, stop_event: threading.Event) -> bool:
        """Block until a wake phrase is heard or `stop_event` is set.

        Returns True if a wake phrase was heard, False if stopped first.
        """
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        audio_q: "queue.Queue[bytes]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=4000,
            dtype="int16",
            channels=1,
            device=self.device,
            callback=callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(chunk):
                    text = normalize(json.loads(rec.Result()).get("text", ""))
                else:
                    text = normalize(json.loads(rec.PartialResult()).get("partial", ""))

                if text and self._matches(text):
                    return True
        return False

    def _matches(self, text: str) -> bool:
        return any(phrase in text for phrase in self.wake_phrases)
