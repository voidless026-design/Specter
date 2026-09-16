"""Orchestrates E.V.: the wake -> listen -> think/act -> speak loop, the
background data-feed loop, and the local control API + GUI, in one
long-lived process. This is what `ev daemon` runs and what the systemd
unit starts.
"""

from __future__ import annotations

import logging
import re
import threading

import uvicorn

from ev_assistant.audio.model_setup import ensure_model
from ev_assistant.audio.stt import record_command
from ev_assistant.audio.tts import Voice
from ev_assistant.audio.wake_word import WakeWordListener
from ev_assistant.brain import Brain
from ev_assistant.bus import StateBus
from ev_assistant.config import Config, load_config, validate_for_daemon, write_default_config
from ev_assistant.data_feeds import DataFeedLoop
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory
from ev_assistant.server import DaemonStatus, create_app

logger = logging.getLogger(__name__)

WAKE_ACK = "Go ahead."
NO_SPEECH_HEARD = "Didn't catch that."

_YES_RE = re.compile(r"\b(yes|yeah|yep|confirm|do it|go ahead|affirmative|proceed)\b", re.IGNORECASE)


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.memory = Memory(cfg.db_path)
        self.knowledge = Knowledge(cfg.knowledge_path)
        self.brain = Brain(cfg, self.memory, self.knowledge)
        self.voice = Voice(cfg)
        self.feed_loop = DataFeedLoop(cfg, self.memory)
        self.status = DaemonStatus()
        self.bus = StateBus()
        self._stop_event = threading.Event()
        self._voice_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._vosk_model = None  # loaded once, shared by wake + command capture

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
            knowledge=self.knowledge,
            feed_loop=self.feed_loop,
            voice=self.voice,
            status=self.status,
            bus=self.bus,
            request_shutdown=self.request_shutdown,
        )
        server_config = uvicorn.Config(
            app, host=self.cfg.control_host, port=self.cfg.control_port, log_level="warning"
        )
        self._server = uvicorn.Server(server_config)

        logger.info(
            "E.V. control API + GUI on http://%s:%s  (open the GUI with `ev gui`)",
            self.cfg.control_host,
            self.cfg.control_port,
        )
        self._server.run()  # blocks until should_exit

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
                names=self.cfg.wake_names,
                prefixes=self.cfg.wake_prefixes,
                device=self.cfg.input_device,
            )
        except Exception:
            logger.exception(
                "Could not start the wake-word listener (no microphone / audio backend?). "
                "The control API and GUI still work for typed questions."
            )
            self._set_state("stopped")
            return

        self._vosk_model = listener.model
        self.status.wake_word_ready = True
        logger.info("Listening for the wake word (names: %s)...", ", ".join(self.cfg.wake_names))

        while not self._stop_event.is_set():
            self._set_state("listening")
            match = listener.listen(self._stop_event, on_level=self.bus.set_level)
            if match is None:
                break  # stopped

            if match.command:
                # "E.V., open Firefox" - act on the inline command directly.
                command_text = match.command
            else:
                # Bare "E.V." - acknowledge and record the follow-up.
                command_text = self._record_after_ack()

            if not command_text:
                self.voice.say(NO_SPEECH_HEARD)
                continue

            self._set_state("thinking")
            logger.info("Heard: %s", command_text)
            reply = self.brain.respond(command_text, confirm=self._voice_confirm)

            self._set_state("speaking")
            logger.info("Replying: %s", reply)
            self.bus.set_transcript(command_text, reply)
            self.voice.say(reply)

        self._set_state("stopped")

    def _record_after_ack(self) -> str:
        self._set_state("recording")
        self.voice.say(WAKE_ACK)
        return record_command(
            self._vosk_model,
            silence_timeout_s=self.cfg.silence_timeout_s,
            device=self.cfg.input_device,
            on_level=self.bus.set_level,
        )

    def _voice_confirm(self, description: str) -> bool:
        """Ask out loud before a destructive action; listen for a yes."""
        self._set_state("speaking")
        self.voice.say(f"You asked me to {description}. Say yes to confirm, or no to cancel.")
        self._set_state("recording")
        answer = record_command(
            self._vosk_model,
            silence_timeout_s=1.0,
            max_duration_s=6.0,
            lead_grace_s=5.0,
            device=self.cfg.input_device,
            on_level=self.bus.set_level,
        )
        confirmed = bool(_YES_RE.search(answer))
        logger.info("Confirmation for %r: heard %r -> %s", description, answer, confirmed)
        return confirmed

    def _set_state(self, state: str) -> None:
        self.status.state = state
        self.bus.set_state(state)


def run_daemon() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    write_default_config()
    cfg = load_config()
    Daemon(cfg).run()
