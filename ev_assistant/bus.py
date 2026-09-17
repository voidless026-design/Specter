"""A tiny thread-safe snapshot of E.V.'s live state.

The voice loop (a worker thread) writes E.V.'s current state and the live
microphone level here; the control API's WebSocket reads it ~20x/second to
drive the GUI's audio visualizer. Plain lock-guarded assignment is plenty -
there's one writer, and readers only ever want the latest value.
"""

from __future__ import annotations

import threading


class StateBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "starting"  # starting|listening|recording|thinking|speaking|stopped
        self._level = 0.0  # 0.0-1.0 recent mic loudness
        self._last_heard = ""
        self._last_reply = ""

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state
            if state not in ("recording", "listening"):
                self._level = 0.0

    def set_level(self, level: float) -> None:
        with self._lock:
            self._level = max(0.0, min(1.0, level))

    def set_transcript(self, heard: str, reply: str) -> None:
        with self._lock:
            self._last_heard = heard
            self._last_reply = reply

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "level": round(self._level, 3),
                "last_heard": self._last_heard,
                "last_reply": self._last_reply,
            }
