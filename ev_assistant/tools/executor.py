"""Executes the actions E.V. is allowed to take on your machine.

Everything routes through one Executor so permission tiers, the
confirm-before-destructive gate, and the always-on forbidden-pattern
blocklist are enforced in a single place - the model can only ever reach
these methods, never a raw shell, and each method decides for itself what
it will and won't do.

Permission tiers (config: permissions.tier):
    safe     - open apps, open websites, answer questions
    standard - + close apps, volume/media, GNOME settings & extensions
    full     - + arbitrary shell commands

Tiers are enforced twice: the tool list handed to Claude is filtered by
tier (so the model isn't even offered tools it can't use), and dispatch()
re-checks on the way in (so a hallucinated or replayed tool call can't slip
past). Destructive shell commands additionally require the injected
`confirm` callback to return True.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ev_assistant.config import Config

logger = logging.getLogger(__name__)

# Tier ordering for "at least this tier" checks.
_TIER_RANK = {"safe": 0, "standard": 1, "full": 2}

# Substrings that mark a shell command as destructive (needs confirmation).
_DESTRUCTIVE_MARKERS = [
    "rm ", "rmdir", "mkfs", "dd ", "shutdown", "reboot", "poweroff", "kill",
    "pkill", "sudo", "chmod -r", "chown -r", "> /dev", "mv /", "truncate",
    "userdel", "passwd", "fdisk", "parted", "wipefs", "systemctl stop",
    "systemctl disable", "git reset --hard", "git clean", "> /etc",
]

ConfirmFn = Callable[[str], bool]


@dataclass
class ToolResult:
    ok: bool
    message: str


def _run(cmd: list[str], timeout: int = 30, text_input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, input=text_input
    )


def _spawn_detached(cmd: list[str]) -> None:
    """Launch a GUI app/process without blocking or tying it to our lifetime."""
    subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


class Executor:
    def __init__(self, cfg: Config, confirm: ConfirmFn | None = None):
        self.cfg = cfg
        # Default-deny: if no confirmer is wired up, destructive actions are
        # refused rather than silently run.
        self.confirm = confirm or (lambda _desc: False)

    def tier_ok(self, required: str) -> bool:
        return _TIER_RANK.get(self.cfg.permission_tier, 0) >= _TIER_RANK[required]

    # -- dispatch -----------------------------------------------------

    def dispatch(self, name: str, args: dict) -> ToolResult:
        handler = _HANDLERS.get(name)
        if handler is None:
            return ToolResult(False, f"Unknown tool: {name}")
        required = _TOOL_TIER[name]
        if not self.tier_ok(required):
            return ToolResult(
                False,
                f"That needs the '{required}' permission tier, but E.V. is set to "
                f"'{self.cfg.permission_tier}'. Change permissions.tier in the config to allow it.",
            )
        try:
            return handler(self, args)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "That command took too long and was stopped.")
        except FileNotFoundError as e:
            return ToolResult(False, f"A required program isn't installed: {e}")
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return ToolResult(False, f"That failed: {e}")

    # -- individual tools ---------------------------------------------

    def open_app(self, args: dict) -> ToolResult:
        name = (args.get("name") or "").strip()
        if not name:
            return ToolResult(False, "No application name given.")
        binary = shutil.which(name)
        if binary:
            _spawn_detached([binary])
            return ToolResult(True, f"Opened {name}.")
        if shutil.which("gtk-launch"):
            desktop = name if name.endswith(".desktop") else f"{name}.desktop"
            proc = _run(["gtk-launch", desktop.removesuffix(".desktop")])
            if proc.returncode == 0:
                return ToolResult(True, f"Launched {name}.")
        return ToolResult(False, f"Couldn't find an app called '{name}'.")

    def close_app(self, args: dict) -> ToolResult:
        name = (args.get("name") or "").strip()
        if not name:
            return ToolResult(False, "No application name given.")
        if not shutil.which("pkill"):
            return ToolResult(False, "pkill isn't available to close apps.")
        # Graceful terminate (SIGTERM), case-insensitive, matched on the name.
        proc = _run(["pkill", "-TERM", "-i", name])
        if proc.returncode == 0:
            return ToolResult(True, f"Closed {name}.")
        return ToolResult(False, f"Didn't find a running app matching '{name}'.")

    def open_url(self, args: dict) -> ToolResult:
        url = (args.get("url") or "").strip()
        if not url:
            return ToolResult(False, "No URL given.")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url
        if not shutil.which("xdg-open"):
            return ToolResult(False, "xdg-open isn't available to open URLs.")
        _spawn_detached(["xdg-open", url])
        return ToolResult(True, f"Opening {url}.")

    def set_volume(self, args: dict) -> ToolResult:
        percent = args.get("percent")
        if percent is None:
            return ToolResult(False, "No volume level given.")
        percent = max(0, min(150, int(percent)))
        if shutil.which("wpctl"):
            _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent / 100:.2f}"])
            return ToolResult(True, f"Volume set to {percent}%.")
        if shutil.which("pactl"):
            _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
            return ToolResult(True, f"Volume set to {percent}%.")
        return ToolResult(False, "No PipeWire/PulseAudio volume control found.")

    def media_control(self, args: dict) -> ToolResult:
        action = (args.get("action") or "").strip().lower()
        valid = {"play", "pause", "play-pause", "next", "previous", "stop"}
        if action not in valid:
            return ToolResult(False, f"Unknown media action '{action}'.")
        if not shutil.which("playerctl"):
            return ToolResult(False, "playerctl isn't installed (sudo dnf install playerctl).")
        _run(["playerctl", action])
        return ToolResult(True, f"Media: {action}.")

    def gnome_extension(self, args: dict) -> ToolResult:
        action = (args.get("action") or "list").strip().lower()
        name = (args.get("name") or "").strip()
        if not shutil.which("gnome-extensions"):
            return ToolResult(False, "gnome-extensions isn't available (are you on GNOME?).")
        if action == "list":
            proc = _run(["gnome-extensions", "list"])
            listed = proc.stdout.strip() or "(none)"
            return ToolResult(True, f"Installed extensions:\n{listed}")
        if action in ("enable", "disable", "info") and name:
            proc = _run(["gnome-extensions", action, name])
            if proc.returncode == 0:
                return ToolResult(True, f"{action.title()}d extension {name}." if action != "info"
                                  else proc.stdout.strip())
            return ToolResult(False, proc.stderr.strip() or f"Couldn't {action} {name}.")
        return ToolResult(False, "Give an action (list/enable/disable/info) and an extension name.")

    def gnome_setting(self, args: dict) -> ToolResult:
        op = (args.get("op") or "").strip().lower()
        schema = (args.get("schema") or "").strip()
        key = (args.get("key") or "").strip()
        value = args.get("value")
        if not shutil.which("gsettings"):
            return ToolResult(False, "gsettings isn't available.")
        if op == "get" and schema and key:
            proc = _run(["gsettings", "get", schema, key])
            return ToolResult(proc.returncode == 0, proc.stdout.strip() or proc.stderr.strip())
        if op == "set" and schema and key and value is not None:
            proc = _run(["gsettings", "set", schema, key, str(value)])
            if proc.returncode == 0:
                return ToolResult(True, f"Set {schema} {key} to {value}.")
            return ToolResult(False, proc.stderr.strip() or "gsettings set failed.")
        return ToolResult(False, "Give op (get/set), schema, key, and (for set) a value.")

    def launch_jegeo(self, args: dict) -> ToolResult:
        module = (args.get("module") or "").strip()
        base = None
        if shutil.which("jegeo"):
            base = ["jegeo"]
        elif shutil.which("python3"):
            base = ["python3", "-m", "jegeo"]
        if not base:
            return ToolResult(
                False,
                "Jegeo isn't installed. Install it from "
                "https://github.com/voidless026-design/dig_atk first.",
            )
        cmd = base + ([module] if module else [])
        _spawn_detached(cmd)
        target = f" (module: {module})" if module else ""
        return ToolResult(True, f"Launched Jegeo{target}.")

    def run_shell(self, args: dict) -> ToolResult:
        command = (args.get("command") or "").strip()
        if not command:
            return ToolResult(False, "No command given.")

        lowered = command.lower()
        for pattern in self.cfg.forbidden_patterns:
            if pattern.lower() in lowered:
                return ToolResult(
                    False, f"Refused: that command matches a forbidden pattern ('{pattern}')."
                )

        is_destructive = any(marker in lowered for marker in _DESTRUCTIVE_MARKERS)
        if is_destructive and self.cfg.confirm_destructive:
            if not self.confirm(f"run the command: {command}"):
                return ToolResult(False, "Cancelled - you didn't confirm.")

        proc = _run(["bash", "-lc", command], timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        output = output.strip()
        if len(output) > 1500:
            output = output[:1500] + "\n...(truncated)"
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return ToolResult(
            proc.returncode == 0,
            f"[{status}] {output}" if output else f"Command finished ({status}).",
        )


# name -> (bound method name, minimum tier)
_HANDLERS: dict[str, Callable[[Executor, dict], ToolResult]] = {
    "open_app": Executor.open_app,
    "close_app": Executor.close_app,
    "open_url": Executor.open_url,
    "set_volume": Executor.set_volume,
    "media_control": Executor.media_control,
    "gnome_extension": Executor.gnome_extension,
    "gnome_setting": Executor.gnome_setting,
    "launch_jegeo": Executor.launch_jegeo,
    "run_shell": Executor.run_shell,
}

_TOOL_TIER = {
    "open_app": "safe",
    "open_url": "safe",
    "launch_jegeo": "safe",
    "close_app": "standard",
    "set_volume": "standard",
    "media_control": "standard",
    "gnome_extension": "standard",
    "gnome_setting": "standard",
    "run_shell": "full",
}
