"""Localhost-only control API - this is what makes E.V. reachable over SSH.

Bound to 127.0.0.1 by default (see daemon.py / config.py): reaching it
still requires a shell on the machine (SSH counts) or a port-forward you
set up yourself, plus the bearer token in EV_CONTROL_TOKEN. It is not
exposed to the network, and turning that around is a deliberate choice,
not an oversight - see the README's "SSH access, honestly" section.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from ev_assistant.audio.tts import Voice
from ev_assistant.brain import Brain
from ev_assistant.config import Config
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.memory import Memory


class DaemonStatus:
    """Flags the voice loop writes and the control API reads.

    Plain attribute reads/writes are atomic in CPython and no field here
    depends on another, so this deliberately skips a lock.
    """

    def __init__(self) -> None:
        self.state = "starting"  # starting | listening | recording | thinking | speaking | stopped
        self.wake_word_ready = False


class AskRequest(BaseModel):
    text: str
    speak: bool = True


def create_app(
    cfg: Config,
    brain: Brain,
    memory: Memory,
    feed_loop: DataFeedLoop,
    voice: Voice | None,
    status: DaemonStatus,
    request_shutdown: Callable[[], None],
) -> FastAPI:
    app = FastAPI(title="E.V. control API")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if not cfg.control_token:
            raise HTTPException(status_code=503, detail="EV_CONTROL_TOKEN is not configured on the server")
        expected = f"Bearer {cfg.control_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    @app.get("/status")
    def get_status(_: None = Depends(require_token)) -> dict:
        return {
            "state": status.state,
            "wake_word_ready": status.wake_word_ready,
            "model": cfg.model,
            "fact_count": memory.fact_count(),
            "last_feed_run_at": feed_loop.last_run_at,
            "last_feed_error": feed_loop.last_error,
        }

    @app.post("/ask")
    def ask(body: AskRequest, _: None = Depends(require_token)) -> dict:
        reply = brain.respond(body.text)
        if body.speak and voice is not None:
            threading.Thread(target=voice.say, args=(reply,), daemon=True).start()
        return {"reply": reply}

    @app.post("/stop")
    def stop(_: None = Depends(require_token)) -> dict:
        threading.Thread(target=request_shutdown, daemon=True).start()
        return {"ok": True}

    return app
