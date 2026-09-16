"""Configuration loading for E.V.

Non-secret settings live in a TOML file under the user's config directory
(``ev init`` writes a commented default). Secrets (API key, control-API
token) come from the environment, falling back to the same ``env`` file the
systemd unit reads - so ``ev ask`` works from any shell without you having
to export anything by hand.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "ev-assistant"

# Spellings Vosk tends to produce for "E.V." - matched anywhere in an utterance.
DEFAULT_WAKE_NAMES = [
    "e v",
    "ev",
    "eva",
    "evie",
    "evee",
    "e vee",
    "eevee",
    "e b",  # Vosk sometimes hears the "V" as a "B"
]

# Optional lead-ins. Presence isn't required (bare "E.V." wakes her), but when
# one is present it's stripped before the rest is treated as the command.
DEFAULT_WAKE_PREFIXES = ["hey", "yo", "ok", "okay", "hi", "hello", "yes"]

DEFAULT_FEEDS = [
    "https://hnrss.org/frontpage",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

DEFAULT_FORBIDDEN = ["rm -rf /", "mkfs", "dd if=", ":(){", "shutdown -h now"]

_DEFAULT_TOML = """\
# E.V. configuration. Secrets (ANTHROPIC_API_KEY, EV_CONTROL_TOKEN) are NOT
# stored here - they live in the `env` file next to this one.

[brain]
# Claude model used when online. claude-opus-5 is the most capable;
# claude-sonnet-5 is faster and cheaper if the voice round-trip feels slow.
model = "claude-opus-5"
# low | medium | high | xhigh | max - lower is faster, better for live voice.
effort = "low"
max_tokens = 1024

[personality]
# 0-100 sliders that shape E.V.'s system prompt.
humor = 65
honesty = 90
verbosity = "concise"  # concise | normal
# Free-text extra instructions appended to her persona. Say anything you like:
# "call me boss", "never apologise", "be blunter when I'm wrong".
custom_instructions = ""

[wake_word]
# She wakes on any of these names appearing in speech, with or without a
# lead-in word ("Hey E.V.", "Yo E.V.", or just "E.V."). Multiple spellings
# are listed because speech recognition renders "E.V." inconsistently.
names = {wake_names}
prefixes = {wake_prefixes}
# When you say the whole thing at once ("E.V., open Firefox"), she acts on it
# directly instead of waiting for a second utterance.
same_utterance_commands = true
# Seconds of trailing silence that end a command recording.
silence_timeout_s = 1.2

[audio]
# Input device name or index, from `ev devices`. Blank = system default.
input_device = ""

[voice]
# auto   - Australian neural voice online, best available offline
# edge   - always Microsoft Edge neural voices (needs internet)
# piper  - always local Piper neural voice
# espeak - always espeak-ng (robotic, but always works)
engine = "auto"
# Australian female. Run `ev voices --online` to see every option.
edge_voice = "en-AU-NatashaNeural"
# Offline neural voice. `ev voices --install-piper` downloads one.
piper_model = ""
# Final offline fallback. "+f2".."+f4" are female variants.
espeak_voice = "en-gb+f3"
rate = 175

[permissions]
# What E.V. may do on your machine when you ask:
#   safe     - open apps and websites, answer questions. Nothing else.
#   standard - the above + close apps, volume/media, GNOME settings/extensions.
#   full     - the above + run arbitrary shell commands.
# `full` means anything that reaches your microphone can run commands as you.
# Read the Security section of the README before setting it.
tier = "standard"
# Ask out loud before anything destructive (deleting, killing, sudo, rm).
confirm_destructive = true
# Refused outright at every tier, no confirmation offered.
forbidden_patterns = {forbidden}

[offline]
# auto - Claude when the network is up, local knowledge when it isn't.
mode = "auto"
# Local model for offline answers, if you have Ollama installed (e.g. "llama3.2").
# Blank means offline answers come straight from your knowledge base.
ollama_model = ""
ollama_host = "http://127.0.0.1:11434"

[control_api]
host = "127.0.0.1"
port = 8765
# Point these at another machine running `ev daemon` to borrow its brainpower.
# Blank = use this machine. See "Remote brain" in the README.
remote_host = ""
remote_port = 0

[data_feeds]
feeds = {feeds}
interval_minutes = 30
# City for weather lookups, e.g. "Melbourne". Blank disables it.
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
    custom_instructions: str = ""

    wake_names: list[str] = field(default_factory=lambda: list(DEFAULT_WAKE_NAMES))
    wake_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_WAKE_PREFIXES))
    same_utterance_commands: bool = True
    silence_timeout_s: float = 1.2
    input_device: str | int | None = None

    voice_engine: str = "auto"
    edge_voice: str = "en-AU-NatashaNeural"
    piper_model: str = ""
    espeak_voice: str = "en-gb+f3"
    tts_rate: int = 175

    permission_tier: str = "standard"
    confirm_destructive: bool = True
    forbidden_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN))

    offline_mode: str = "auto"
    ollama_model: str = ""
    ollama_host: str = "http://127.0.0.1:11434"

    control_host: str = "127.0.0.1"
    control_port: int = 8765
    remote_host: str = ""
    remote_port: int = 0

    feeds: list[str] = field(default_factory=lambda: list(DEFAULT_FEEDS))
    feed_interval_minutes: int = 30
    weather_location: str | None = None

    data_dir: Path = field(default_factory=lambda: Path(user_data_dir(APP_NAME, appauthor=False)))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def knowledge_path(self) -> Path:
        return self.data_dir / "knowledge.sqlite3"

    @property
    def vosk_model_dir(self) -> Path:
        return self.data_dir / "vosk-model"

    @property
    def piper_dir(self) -> Path:
        return self.data_dir / "piper"

    @property
    def api_host(self) -> str:
        """Where the CLI/GUI talks to - a remote brain if one is configured."""
        return self.remote_host or self.control_host

    @property
    def api_port(self) -> int:
        return self.remote_port or self.control_port


def config_dir() -> Path:
    # appauthor=False keeps the path the same predictable "<base>/ev-assistant"
    # shape on every OS, matching what the install scripts assume.
    return Path(user_config_dir(APP_NAME, appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.toml"


def env_file_path() -> Path:
    """The KEY=VALUE file the systemd unit reads, and our secret fallback."""
    return config_dir() / "env"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse a systemd-style EnvironmentFile into a dict. Never raises."""
    path = path or env_file_path()
    values: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def write_default_config(path: Path | None = None) -> Path:
    """Write the commented default config file if one doesn't already exist."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rendered = _DEFAULT_TOML.format(
            wake_names=_toml_str_list(DEFAULT_WAKE_NAMES),
            wake_prefixes=_toml_str_list(DEFAULT_WAKE_PREFIXES),
            forbidden=_toml_str_list(DEFAULT_FORBIDDEN),
            feeds=_toml_str_list(DEFAULT_FEEDS),
        )
        path.write_text(rendered, encoding="utf-8")
    return path


def _toml_str_list(values: list[str]) -> str:
    return "[" + ", ".join('"' + v.replace('"', '\\"') + '"' for v in values) + "]"


def load_config(path: Path | None = None, env_path: Path | None = None) -> Config:
    """Load TOML settings layered over defaults, plus secrets from env/env-file."""
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
    perms = raw.get("permissions", {})
    offline = raw.get("offline", {})
    control = raw.get("control_api", {})
    feeds_section = raw.get("data_feeds", {})

    input_device: str | int | None = audio.get("input_device") or None
    if isinstance(input_device, str) and input_device.strip().isdigit():
        input_device = int(input_device.strip())

    data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
    data_dir.mkdir(parents=True, exist_ok=True)

    # Shell environment wins; the daemon's env file is the fallback, so that
    # `ev ask` works from a fresh shell or an SSH session with no exports.
    file_env = read_env_file(env_path)

    def secret(name: str) -> str:
        return os.environ.get(name) or file_env.get(name, "")

    return Config(
        anthropic_api_key=secret("ANTHROPIC_API_KEY"),
        control_token=secret("EV_CONTROL_TOKEN"),
        model=brain.get("model", "claude-opus-5"),
        effort=brain.get("effort", "low"),
        max_tokens=int(brain.get("max_tokens", 1024)),
        humor=int(personality.get("humor", 65)),
        honesty=int(personality.get("honesty", 90)),
        verbosity=personality.get("verbosity", "concise"),
        custom_instructions=personality.get("custom_instructions", ""),
        wake_names=wake.get("names", list(DEFAULT_WAKE_NAMES)),
        wake_prefixes=wake.get("prefixes", list(DEFAULT_WAKE_PREFIXES)),
        same_utterance_commands=bool(wake.get("same_utterance_commands", True)),
        silence_timeout_s=float(wake.get("silence_timeout_s", 1.2)),
        input_device=input_device,
        voice_engine=voice.get("engine", "auto"),
        edge_voice=voice.get("edge_voice", "en-AU-NatashaNeural"),
        piper_model=voice.get("piper_model", ""),
        espeak_voice=voice.get("espeak_voice", "en-gb+f3"),
        tts_rate=int(voice.get("rate", 175)),
        permission_tier=perms.get("tier", "standard"),
        confirm_destructive=bool(perms.get("confirm_destructive", True)),
        forbidden_patterns=perms.get("forbidden_patterns", list(DEFAULT_FORBIDDEN)),
        offline_mode=offline.get("mode", "auto"),
        ollama_model=offline.get("ollama_model", ""),
        ollama_host=offline.get("ollama_host", "http://127.0.0.1:11434"),
        control_host=control.get("host", "127.0.0.1"),
        control_port=int(control.get("port", 8765)),
        remote_host=control.get("remote_host", ""),
        remote_port=int(control.get("remote_port", 0)),
        feeds=feeds_section.get("feeds", list(DEFAULT_FEEDS)),
        feed_interval_minutes=int(feeds_section.get("interval_minutes", 30)),
        weather_location=(feeds_section.get("weather_location") or None),
        data_dir=data_dir,
    )


def validate_for_daemon(cfg: Config) -> list[str]:
    """Human-readable problems that block starting the daemon."""
    problems = []
    if not cfg.anthropic_api_key and cfg.offline_mode != "offline":
        problems.append(
            f"ANTHROPIC_API_KEY is not set. Put it in {env_file_path()} "
            "(get a key at https://console.anthropic.com/settings/keys), or set "
            'offline.mode = "offline" in config.toml to run without Claude.'
        )
    if not cfg.control_token:
        problems.append(
            f"EV_CONTROL_TOKEN is not set. Run `ev init` to generate one into {env_file_path()}."
        )
    return problems
