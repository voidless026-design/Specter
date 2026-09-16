"""Anthropic tool definitions for E.V.'s system-control tools.

The tool list handed to Claude is filtered by the active permission tier
(see tools_for_tier), so the model is never even offered a capability the
user hasn't granted. Executor.dispatch re-checks the tier on the way in as
a second line of defence.
"""

from __future__ import annotations

from ev_assistant.tools.executor import _TOOL_TIER, _TIER_RANK

_ALL_SPECS = [
    {
        "name": "open_app",
        "description": "Open/launch an application on the user's computer by name (e.g. 'firefox', 'code', 'spotify').",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Application or command name."}},
            "required": ["name"],
        },
    },
    {
        "name": "open_url",
        "description": "Open a website in the user's default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL or domain to open."}},
            "required": ["url"],
        },
    },
    {
        "name": "launch_jegeo",
        "description": "Launch the user's Jegeo security-operator console (the dig_atk app). Optionally name a module to open.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Optional module name to open, e.g. 'network scanner', 'osint'.",
                }
            },
        },
    },
    {
        "name": "close_app",
        "description": "Close/quit a running application by name (graceful terminate).",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Application/process name."}},
            "required": ["name"],
        },
    },
    {
        "name": "set_volume",
        "description": "Set the system output volume to a percentage (0-150).",
        "input_schema": {
            "type": "object",
            "properties": {"percent": {"type": "integer", "description": "Volume 0-150."}},
            "required": ["percent"],
        },
    },
    {
        "name": "media_control",
        "description": "Control media playback (Spotify, browser video, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "play-pause", "next", "previous", "stop"],
                }
            },
            "required": ["action"],
        },
    },
    {
        "name": "gnome_extension",
        "description": "List, enable, disable, or inspect GNOME Shell extensions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "enable", "disable", "info"]},
                "name": {"type": "string", "description": "Extension UUID (needed for enable/disable/info)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "gnome_setting",
        "description": "Read or change a GNOME desktop setting via gsettings (theme, wallpaper, behaviour, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["get", "set"]},
                "schema": {"type": "string", "description": "gsettings schema, e.g. org.gnome.desktop.interface."},
                "key": {"type": "string", "description": "Setting key, e.g. color-scheme."},
                "value": {"type": "string", "description": "New value (for set)."},
            },
            "required": ["op", "schema", "key"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a shell command on the user's Fedora machine and return its output. "
            "Use this only when no more specific tool fits. Destructive commands require "
            "the user to confirm out loud first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run."}},
            "required": ["command"],
        },
    },
]


def tools_for_tier(tier: str) -> list[dict]:
    """Return the tool definitions available at the given permission tier."""
    rank = _TIER_RANK.get(tier, 0)
    return [spec for spec in _ALL_SPECS if _TIER_RANK[_TOOL_TIER[spec["name"]]] <= rank]
