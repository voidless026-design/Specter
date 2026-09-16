"""Orchestrates E.V.: the wake-word -> listen -> think -> speak loop, the
background data-feed learning loop, and the local control API, all in one
long-lived process. This is what `ev daemon` runs, and what the systemd
unit / Windows autostart entry point at.
"""

from __future__ import annotations

import logging
import threading

import uvicorn

from ev_assistant.audio.model_setup import ensure_model
from ev_assistant.audio.stt import record_command
from ev_assistant.audio.tts import Voice
from ev_assistant.audio.wake_word import WakeWordListener
from ev_assistant.brain import Brain
from ev_assistant.config import Config, load_config, validate_for_daemon, write_default_config
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.memory import Memory
from ev_assistant.server import DaemonStatus, create_app

logger = logging.getLogger(__name__)

WAKE_ACK = "Go ahead."
NO_SPEECH_HEARD = "Didn't catch that."


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.memory = Memory(cfg.db_path)
        self.brain = Brain(cfg, self.memory)
        self.voice = Voice(rate=cfg.tts_rate, voice_id=cfg.tts_voice_id)
        self.feed_loop = DataFeedLoop(cfg, self.memory)
        self.status = DaemonStatus()
        self._stop_event = threading.Event()
        self._voice_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def request_shutdown(self) -> None:
        logger.info("Shutdown requested")
        self._stop_event.set()
        if self._server is not None:
            self._server.should_exit = True

    def run(self) -> None:
        problems = validate_for_daemon(self.cfg)
        if problems:
            for p in problems:
                logger.error(p)
            raise SystemExit(1)

        logger.info("Checking for the offline speech model (first run downloads it)...")
        ensure_model(self.cfg.vosk_model_dir, progress=self._log_download_progress)

        self.feed_loop.start()
        logger.info("Data feed loop started (every %s min)", self.cfg.feed_interval_minutes)

        self._voice_thread = threading.Thread(target=self._voice_loop, name="ev-voice-loop", daemon=True)
        self._voice_thread.start()

        app = create_app(
            cfg=self.cfg,
            brain=self.brain,
            memory=self.memory,
            feed_loop=self.feed_loop,
            voice=self.voice,
            status=self.status,
            request_shutdown=self.request_shutdown,
        )
        server_config = uvicorn.Config(
            app, host=self.cfg.control_host, port=self.cfg.control_port, log_level="warning"
        )
        self._server = uvicorn.Server(server_config)

        logger.info(
            "E.V. control API listening on http://%s:%s (use `ev ask` over SSH)",
            self.cfg.control_host,
            self.cfg.control_port,
        )
        # Blocks until self._server.should_exit is set - by an OS signal
        # (uvicorn installs its own SIGINT/SIGTERM handlers) or by /stop.
        self._server.run()

        self._stop_event.set()
        self.feed_loop.stop()
        if self._voice_thread:
            self._voice_thread.join(timeout=2)
        logger.info("E.V. daemon stopped")

    def _log_download_progress(self, downloaded: int, total: int) -> None:
        if total:
            pct = downloaded * 100 // total
            if pct % 20 == 0:
                logger.info("Downloading speech model... %d%%", pct)

    def _voice_loop(self) -> None:
        try:
            listener = WakeWordListener(
                model_dir=self.cfg.vosk_model_dir,
                wake_phrases=self.cfg.wake_phrases,
                device=self.cfg.input_device,
            )
        except Exception:
            logger.exception(
                "Could not start the wake-word listener (no microphone / audio backend?). "
                "The control API still works for `ev ask` over SSH."
            )
            self.status.state = "stopped"
            return

        model = listener.model  # reuse the loaded model for command transcription too
        self.status.wake_word_ready = True
        logger.info("Listening for the wake word (%s)...", ", ".join(self.cfg.wake_phrases))

        while not self._stop_event.is_set():
            self.status.state = "listening"
            heard = listener.listen(self._stop_event)
            if not heard:
                break  # stop_event was set while waiting

            self.status.state = "recording"
            self.voice.say(WAKE_ACK)
            command_text = record_command(
                model,
                silence_timeout_s=self.cfg.silence_timeout_s,
                device=self.cfg.input_device,
            )
            if not command_text:
                self.voice.say(NO_SPEECH_HEARD)
                continue

            self.status.state = "thinking"
            logger.info("Heard: %s", command_text)
            reply = self.brain.respond(command_text)

            self.status.state = "speaking"
            logger.info("Replying: %s", reply)
            self.voice.say(reply)

        self.status.state = "stopped"


def run_daemon() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    write_default_config()
    cfg = load_config()
    Daemon(cfg).run()
