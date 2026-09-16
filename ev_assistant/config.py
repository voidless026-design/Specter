"""Configuration loading for E.V.

Non-secret settings live in a TOML file under the user's config directory
(``ev init`` writes a commented default). Secrets (API key, control-API
token) are read from the environment only, so they never end up in a
config file that might get committed or copied around.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "ev-assistant"

DEFAULT_WAKE_PHRASES = ["hey e v", "hey eva", "hey e. v.", "hey ev", "hey evie"]

DEFAULT_FEEDS = [
    "https://hnrss.org/frontpage",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

_DEFAULT_TOML = """\
# E.V. configuration. Secrets (ANTHROPIC_API_KEY, EV_CONTROL_TOKEN) are NOT
# stored here - set them as environment variables (see .env.example).

[brain]
# Claude model used for E.V.'s responses. claude-opus-5 is the most capable;
# switch to claude-sonnet-5 if voice round-trip latency feels slow.
model = "claude-opus-5"
# low | medium | high | xhigh | max - lower is faster, better for a live voice loop.
effort = "low"
max_tokens = 1024

[personality]
# 0-100 sliders that shape E.V.'s system prompt. See ev_assistant/personality.py.
humor = 65
honesty = 90
verbosity = "concise"  # concise | normal

[wake_word]
phrases = {wake_phrases}
# Seconds of trailing silence that end a command recording.
silence_timeout_s = 1.2

[audio]
# Input device name or index, from `ev devices`. Leave blank for the system default.
input_device = ""

[voice]
rate = 175
# Leave blank to use the system default TTS voice.
voice_id = ""

[control_api]
host = "127.0.0.1"
port = 8765

[data_feeds]
feeds = {feeds}
interval_minutes = 30
# City name for weather lookups (via wttr.in), e.g. "Miami". Leave blank to disable.
weather_location = ""
"""


@dataclass
class Config:
    anthropic_api_key: str
    control_token: str

    model: str = "claude-opus-5"
    effort: str = "low"
    max_tokens: int = 1024

    humor: int = 65
    honesty: int = 90
    verbosity: str = "concise"

    wake_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_WAKE_PHRASES))
    silence_timeout_s: float = 1.2
    input_device: str | int | None = None

    tts_rate: int = 175
    tts_voice_id: str | None = None

    control_host: str = "127.0.0.1"
    control_port: int = 8765

    feeds: list[str] = field(default_factory=lambda: list(DEFAULT_FEEDS))
    feed_interval_minutes: int = 30
    weather_location: str | None = None

    data_dir: Path = field(default_factory=lambda: Path(user_data_dir(APP_NAME, appauthor=False)))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def vosk_model_dir(self) -> Path:
        return self.data_dir / "vosk-model"


def config_path() -> Path:
    # appauthor=False: skip the vendor subfolder platformdirs otherwise
    # inserts on Windows/macOS, so the path is the same predictable
    # "<base>/ev-assistant" shape on every OS (matches what the Fedora and
    # Windows install scripts assume).
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


def write_default_config(path: Path | None = None) -> Path:
    """Write the commented default config file if one doesn't already exist."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rendered = _DEFAULT_TOML.format(
            wake_phrases=_toml_str_list(DEFAULT_WAKE_PHRASES),
            feeds=_toml_str_list(DEFAULT_FEEDS),
        )
        path.write_text(rendered, encoding="utf-8")
    return path


def _toml_str_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML (if present) layered with defaults, then env secrets."""
    path = path or config_path()
    raw: dict = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

    brain = raw.get("brain", {})
    personality = raw.get("personality", {})
    wake = raw.get("wake_word", {})
    audio = raw.get("audio", {})
    voice = raw.get("voice", {})
    control = raw.get("control_api", {})
    feeds_section = raw.get("data_feeds", {})

    input_device: str | int | None = (audio.get("input_device") or None)
    if isinstance(input_device, str) and input_device.strip().isdigit():
        input_device = int(input_device.strip())

    data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
    data_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    control_token = os.environ.get("EV_CONTROL_TOKEN", "")

    return Config(
        anthropic_api_key=api_key,
        control_token=control_token,
        model=brain.get("model", "claude-opus-5"),
        effort=brain.get("effort", "low"),
        max_tokens=int(brain.get("max_tokens", 1024)),
        humor=int(personality.get("humor", 65)),
        honesty=int(personality.get("honesty", 90)),
        verbosity=personality.get("verbosity", "concise"),
        wake_phrases=wake.get("phrases", list(DEFAULT_WAKE_PHRASES)),
        silence_timeout_s=float(wake.get("silence_timeout_s", 1.2)),
        input_device=input_device,
        tts_rate=int(voice.get("rate", 175)),
        tts_voice_id=(voice.get("voice_id") or None),
        control_host=control.get("host", "127.0.0.1"),
        control_port=int(control.get("port", 8765)),
        feeds=feeds_section.get("feeds", list(DEFAULT_FEEDS)),
        feed_interval_minutes=int(feeds_section.get("interval_minutes", 30)),
        weather_location=(feeds_section.get("weather_location") or None),
        data_dir=data_dir,
    )


def validate_for_daemon(cfg: Config) -> list[str]:
    """Return a list of human-readable problems that block starting the daemon."""
    problems = []
    if not cfg.anthropic_api_key:
        problems.append(
            "ANTHROPIC_API_KEY is not set. Get a key at "
            "https://console.anthropic.com/settings/keys and export it."
        )
    if not cfg.control_token:
        problems.append(
            "EV_CONTROL_TOKEN is not set. Generate one with "
            "`python -c \"import secrets; print(secrets.token_hex(32))\"` and export it."
        )
    return problems
