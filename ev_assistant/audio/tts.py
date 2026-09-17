"""Text-to-speech for E.V., with an Australian voice and offline fallback.

Three engines, tried in order of quality when engine = "auto":

1. edge  - Microsoft Edge neural voices (en-AU-NatashaNeural = Australian
           female). Excellent quality, free, no API key, but needs internet.
2. piper - Rhasspy Piper neural TTS, fully offline. Good quality once a voice
           model is downloaded (`ev voices --install-piper`). No Australian
           model exists upstream, so this is a British/US neural voice.
3. espeak - espeak-ng. Robotic but always works, entirely offline, no setup.

Each engine shells out to a well-defined CLI (installed as a console script
or system package) rather than embedding an async/native library, so a
failure in one is a clean fall-through to the next instead of a crash.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from ev_assistant.config import Config
from ev_assistant.net import is_online

logger = logging.getLogger(__name__)

# Audio players tried in order, first one present wins.
_PLAYERS = ["ffplay", "mpv", "paplay", "aplay", "mpg123"]


def _venv_which(name: str) -> str | None:
    """Find an executable on PATH or in the running interpreter's bin dir.

    Console scripts like `edge-tts` and `piper` live in the venv's bin, which
    is NOT on PATH when the daemon is launched by systemd - so a plain
    shutil.which misses them and the voice silently degrades to espeak.
    """
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else None


def _find_player(for_mp3: bool) -> list[str] | None:
    for name in _PLAYERS:
        path = shutil.which(name)
        if not path:
            continue
        if name == "ffplay":
            return [path, "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if name == "mpv":
            return [path, "--no-video", "--really-quiet"]
        if name == "mpg123":
            if for_mp3:  # mpg123 decodes mp3 only
                return [path, "-q"]
            continue
        if name in ("paplay", "aplay"):
            if for_mp3:  # these play PCM/wav only
                continue
            return [path]
    return None


class Voice:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()

    def say(self, text: str) -> None:
        """Speak `text`, blocking until finished. One utterance at a time."""
        if not text or not text.strip():
            return
        with self._lock:
            for engine in self._engine_chain():
                try:
                    if engine(text):
                        return
                except Exception:
                    logger.exception("TTS engine failed, falling through")
            logger.error("All TTS engines failed - E.V. could not speak")

    def _engine_chain(self):
        """The ordered list of engine callables to attempt for this utterance."""
        choice = self.cfg.voice_engine
        if choice == "edge":
            return [self._say_edge, self._say_espeak]
        if choice == "piper":
            return [self._say_piper, self._say_espeak]
        if choice == "espeak":
            return [self._say_espeak]
        # auto: best available right now
        chain = []
        if is_online():
            chain.append(self._say_edge)
        if self._piper_ready():
            chain.append(self._say_piper)
        chain.append(self._say_espeak)
        return chain

    # -- engines ------------------------------------------------------

    def _say_edge(self, text: str) -> bool:
        if importlib.util.find_spec("edge_tts") is None:
            logger.info("edge_tts not installed; skipping the Australian neural voice")
            return False
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ev.mp3"
            # Run as a module through this interpreter so it works regardless
            # of PATH (the systemd service PATH omits the venv bin).
            cmd = [sys.executable, "-m", "edge_tts", "--voice", self.cfg.edge_voice,
                   "--text", text, "--write-media", str(out)]
            rate_pct = round((self.cfg.tts_rate / 175.0 - 1.0) * 100)
            if rate_pct:
                # Use the --rate=VALUE form: a bare "-10%" would be parsed as
                # a flag by edge-tts's argument parser.
                cmd.append(f"--rate={rate_pct:+d}%")
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            except subprocess.CalledProcessError as e:
                logger.warning("edge-tts failed (%s); falling back", (e.stderr or b"").decode()[:200])
                return False
            if not out.exists() or out.stat().st_size == 0:
                logger.warning("edge-tts produced no audio; falling back")
                return False
            return self._play(out, is_mp3=True)

    def _say_piper(self, text: str) -> bool:
        piper = _venv_which("piper")
        model = self._piper_model_path()
        if not piper or not model:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ev.wav"
            subprocess.run(
                [piper, "--model", str(model), "--output_file", str(out)],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=60,
            )
            if not out.exists() or out.stat().st_size == 0:
                return False
            return self._play(out, is_mp3=False)

    def _say_espeak(self, text: str) -> bool:
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak:
            return False
        subprocess.run(
            [espeak, "-v", self.cfg.espeak_voice, "-s", str(self.cfg.tts_rate), text],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return True

    # -- helpers ------------------------------------------------------

    def _play(self, path: Path, is_mp3: bool) -> bool:
        player = _find_player(for_mp3=is_mp3)
        if not player:
            logger.error(
                "No audio player found. Install one: sudo dnf install ffmpeg-free (or mpv)."
            )
            return False
        subprocess.run(player + [str(path)], check=True, capture_output=True, timeout=120)
        return True

    def _piper_model_path(self) -> Path | None:
        if self.cfg.piper_model:
            explicit = Path(self.cfg.piper_model)
            if explicit.is_file():
                return explicit
            in_dir = self.cfg.piper_dir / self.cfg.piper_model
            if in_dir.is_file():
                return in_dir
        # Otherwise use whatever .onnx voice is sitting in the piper dir.
        if self.cfg.piper_dir.is_dir():
            models = sorted(self.cfg.piper_dir.glob("*.onnx"))
            if models:
                return models[0]
        return None

    def _piper_ready(self) -> bool:
        return _venv_which("piper") is not None and self._piper_model_path() is not None


def list_online_voices(filter_locale: str | None = "en-AU") -> list[str]:
    """Return edge-tts voice short-names, optionally filtered by locale prefix."""
    if importlib.util.find_spec("edge_tts") is None:
        return []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "edge_tts", "--list-voices"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return []
    voices = []
    for line in proc.stdout.splitlines():
        token = line.strip().split()
        if not token:
            continue
        name = token[0]
        if "-" not in name or name.lower() in ("name", "----"):
            continue
        if filter_locale and not name.startswith(filter_locale):
            continue
        voices.append(name)
    return voices
