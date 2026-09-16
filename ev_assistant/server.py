"""Localhost control API + GUI host - E.V.'s text/SSH surface and web UI.

Bound to 127.0.0.1 by default: reaching it needs a shell on the machine
(SSH counts) or a tunnel you set up, plus the bearer token. The GUI is
served from here too, and streams E.V.'s live state over a WebSocket so the
visualizer reacts to what she hears and says.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ev_assistant.audio.tts import Voice
from ev_assistant.brain import Brain
from ev_assistant.bus import StateBus
from ev_assistant.config import Config, config_path
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory
from ev_assistant.settings import apply_updates

GUI_DIR = Path(__file__).parent / "gui"


class DaemonStatus:
    """Flags the voice loop writes and the control API reads. One writer,
    latest-value reads - no lock needed in CPython."""

    def __init__(self) -> None:
        self.state = "starting"
        self.wake_word_ready = False


class AskRequest(BaseModel):
    text: str
    speak: bool = True
    allow_destructive: bool = False


class LearnRequest(BaseModel):
    kind: str  # wikipedia | url | text
    value: str
    title: str = ""


class SettingsRequest(BaseModel):
    updates: dict[str, object]


def create_app(
    cfg: Config,
    brain: Brain,
    memory: Memory,
    knowledge: Knowledge,
    feed_loop: DataFeedLoop,
    voice: Voice | None,
    status: DaemonStatus,
    bus: StateBus,
    request_shutdown: Callable[[], None],
) -> FastAPI:
    app = FastAPI(title="E.V. control API")

    def token_ok(candidate: str | None) -> bool:
        if not cfg.control_token or not candidate:
            return False
        return secrets.compare_digest(candidate, cfg.control_token)

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if not cfg.control_token:
            raise HTTPException(status_code=503, detail="EV_CONTROL_TOKEN is not configured")
        expected = f"Bearer {cfg.control_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    @app.get("/status")
    def get_status(_: None = Depends(require_token)) -> dict:
        return {
            "state": status.state,
            "wake_word_ready": status.wake_word_ready,
            "model": cfg.model,
            "permission_tier": cfg.permission_tier,
            "offline_mode": cfg.offline_mode,
            "fact_count": memory.fact_count(),
            "knowledge_passages": knowledge.passage_count(),
            "last_feed_run_at": feed_loop.last_run_at,
            "last_feed_error": feed_loop.last_error,
        }

    @app.post("/ask")
    def ask(body: AskRequest, _: None = Depends(require_token)) -> dict:
        # A typed request can pre-authorise destructive actions with
        # allow_destructive; otherwise they're refused (no voice to confirm).
        confirm = (lambda _desc: True) if body.allow_destructive else (lambda _desc: False)
        reply = brain.respond(body.text, confirm=confirm)
        if body.speak and voice is not None:
            threading.Thread(target=voice.say, args=(reply,), daemon=True).start()
        return {"reply": reply}

    @app.post("/learn")
    def learn(body: LearnRequest, _: None = Depends(require_token)) -> dict:
        from ev_assistant import ingest

        try:
            if body.kind == "wikipedia":
                title, text = ingest.from_wikipedia(body.value)
            elif body.kind == "url":
                title, text = ingest.from_url(body.value)
            elif body.kind == "text":
                title, text = (body.title or "note"), body.value
            else:
                raise HTTPException(status_code=400, detail=f"Unknown learn kind: {body.kind}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        source = body.value if body.kind != "text" else "typed"
        added = knowledge.add_document(title, source, text)
        return {"title": title, "passages_added": added}

    @app.get("/settings")
    def get_settings(_: None = Depends(require_token)) -> dict:
        return {
            "personality.humor": cfg.humor,
            "personality.honesty": cfg.honesty,
            "personality.verbosity": cfg.verbosity,
            "personality.custom_instructions": cfg.custom_instructions,
            "voice.engine": cfg.voice_engine,
            "voice.edge_voice": cfg.edge_voice,
            "voice.rate": cfg.tts_rate,
            "permissions.tier": cfg.permission_tier,
            "offline.mode": cfg.offline_mode,
        }

    @app.patch("/settings")
    def patch_settings(body: SettingsRequest, _: None = Depends(require_token)) -> dict:
        applied = apply_updates(config_path(), body.updates)
        return {"applied": applied, "note": "Restart the daemon for changes to take effect."}

    @app.post("/stop")
    def stop(_: None = Depends(require_token)) -> dict:
        threading.Thread(target=request_shutdown, daemon=True).start()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        # Browsers can't set Authorization on a WebSocket, so the token comes
        # in as a query param.
        if not token_ok(websocket.query_params.get("token")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(bus.snapshot())
                await asyncio.sleep(0.05)  # ~20 fps
        except WebSocketDisconnect:
            return
        except Exception:
            return

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(GUI_DIR / "index.html")

    if GUI_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=GUI_DIR), name="static")

    return app
