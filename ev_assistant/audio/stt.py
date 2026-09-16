"""Record a spoken command after the wake word and transcribe it.

Endpointing (deciding when the user has stopped talking) uses simple
RMS-based voice activity detection rather than watching for the partial
transcript to stabilize, since transcript updates can lag the actual
audio by hundreds of milliseconds.
"""

from __future__ import annotations

import json
import queue
import time

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from ev_assistant.audio.wake_word import SAMPLE_RATE

_SILENCE_RMS_THRESHOLD = 300.0  # int16 PCM RMS; tune per mic/room if endpointing feels off


def _rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def record_command(
    model: Model,
    silence_timeout_s: float = 1.2,
    max_duration_s: float = 15.0,
    lead_grace_s: float = 4.0,
    device: str | int | None = None,
) -> str:
    """Record until the user stops talking, then return the transcript.

    Waits up to `lead_grace_s` for speech to start (returns "" if it never
    does), then stops `silence_timeout_s` after speech trails off, capped
    at `max_duration_s` either way.
    """
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    heard_speech = False
    silence_started_at: float | None = None
    started_at = time.monotonic()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=4000,
        dtype="int16",
        channels=1,
        device=device,
        callback=callback,
    ):
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                chunk = b""

            now = time.monotonic()
            if chunk:
                rec.AcceptWaveform(chunk)
                if _rms(chunk) >= _SILENCE_RMS_THRESHOLD:
                    heard_speech = True
                    silence_started_at = None
                elif heard_speech and silence_started_at is None:
                    silence_started_at = now

            if not heard_speech and (now - started_at) >= lead_grace_s:
                break
            if heard_speech and silence_started_at is not None and (now - silence_started_at) >= silence_timeout_s:
                break
            if (now - started_at) >= max_duration_s:
                break

    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()
