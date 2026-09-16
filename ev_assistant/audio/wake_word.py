"""Continuous wake-word spotting for E.V.

Runs Vosk's small offline model continuously and, at the end of each spoken
utterance, checks whether E.V.'s name appears in it. This is deliberately
name-anywhere rather than fixed-phrase: "Hey E.V.", "Yo E.V.", "E.V., can
you...", or a bare "E.V." all wake her, because the name is matched as a
token wherever it lands.

If the same utterance also carries a command ("E.V., open Firefox"), that
command is returned inline so the daemon can act on it without a second
recording. Otherwise the command is None and the daemon records a follow-up.

There's no dedicated keyword-spotter here (no off-the-shelf model exists for
a coined name like "E.V.", and training one needs a synthetic-audio pipeline
that doesn't belong in this repo). Running full ASR costs more idle CPU than
a true wake-word engine; the README documents the openWakeWord upgrade path.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

SAMPLE_RATE = 16000

# int16 RMS that maps to "full" on the visualizer. Normal speech sits well
# below the 32767 ceiling, so a few thousand is a sensible full-scale.
_LEVEL_FULL_SCALE = 3000.0

LevelFn = Callable[[float], None]


def chunk_level(chunk: bytes) -> float:
    """Normalized 0-1 loudness of an int16 PCM chunk, for the visualizer."""
    samples = np.frombuffer(chunk, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return min(1.0, rms / _LEVEL_FULL_SCALE)

_WORD_RE = re.compile(r"[a-z0-9']+")

# Filler words stripped from the front of an inline command so
# "E.V., can you open Firefox" yields "open Firefox", not "can you open...".
_LEADING_FILLERS = {"can", "you", "could", "would", "please", "will", "to", "um", "uh"}


def normalize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class WakeMatch:
    """A detected wake event. `command` is the inline command, if any."""

    command: str | None


def _name_tokens(names: list[str]) -> list[list[str]]:
    """Turn each configured name into a token sequence, longest first.

    Longest-first so a two-token name like "e v" is tried before a
    one-token "e" that could otherwise shadow it.
    """
    seqs = [normalize(n) for n in names if n.strip()]
    seqs = [s for s in seqs if s]
    seqs.sort(key=len, reverse=True)
    return seqs


def match_wake(text: str, names: list[str], prefixes: list[str]) -> WakeMatch | None:
    """Return a WakeMatch if E.V.'s name appears in `text`, else None.

    `prefixes` are optional lead-ins ("hey", "yo", ...). They're not required
    - a bare name wakes her - but a name is only accepted mid-sentence if it
    starts the utterance, follows a prefix, or follows punctuation-like
    filler, which keeps stray "ev" sounds inside longer words/phrases from
    triggering. Everything after the matched name becomes the command.
    """
    words = normalize(text)
    if not words:
        return None

    prefix_set = {p.lower() for p in prefixes}
    name_seqs = _name_tokens(names)

    for i in range(len(words)):
        # A name is a valid wake only at the start, right after a prefix word.
        preceding_ok = i == 0 or words[i - 1] in prefix_set
        if not preceding_ok:
            continue
        for seq in name_seqs:
            end = i + len(seq)
            if words[i:end] == seq:
                tail = words[end:]
                while tail and tail[0] in _LEADING_FILLERS:
                    tail = tail[1:]
                command = " ".join(tail).strip() or None
                return WakeMatch(command=command)
    return None


class WakeWordListener:
    def __init__(
        self,
        model_dir: Path,
        names: list[str],
        prefixes: list[str],
        device: str | int | None = None,
    ):
        self.model = Model(str(model_dir))
        self.names = names
        self.prefixes = prefixes
        self.device = device

    def listen(self, stop_event: threading.Event, on_level: LevelFn | None = None) -> WakeMatch | None:
        """Block until E.V.'s name is heard or `stop_event` is set.

        Returns the WakeMatch (with any inline command) on a wake, or None if
        stopped first. `on_level` receives a 0-1 loudness for each audio chunk
        so the GUI visualizer can pulse while she's listening.
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
                    if on_level:
                        on_level(0.0)
                    continue

                if on_level:
                    on_level(chunk_level(chunk))

                # Only act on finalized utterances - partials churn too much
                # to reliably extract a trailing command.
                if rec.AcceptWaveform(chunk):
                    text = json.loads(rec.Result()).get("text", "")
                    match = match_wake(text, self.names, self.prefixes)
                    if match is not None:
                        return match
        return None
