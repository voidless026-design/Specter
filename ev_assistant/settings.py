"""In-place editing of config.toml scalar values for the GUI settings panel.

Rewrites individual `key = value` lines within their section, leaving the
file's comments and layout intact - which a read-modify-reserialize round
trip through a TOML writer would destroy. Only the small set of
GUI-editable scalars is exposed; everything else stays hand-edited.
"""

from __future__ import annotations

import re
from pathlib import Path

# (section, key) -> value type, for the fields the GUI may change.
EDITABLE: dict[tuple[str, str], str] = {
    ("brain", "provider"): "str",
    ("personality", "humor"): "int",
    ("personality", "honesty"): "int",
    ("personality", "sarcasm"): "int",
    ("personality", "warmth"): "int",
    ("personality", "formality"): "int",
    ("personality", "verbosity"): "str",
    ("personality", "address_as"): "str",
    ("personality", "custom_instructions"): "str",
    ("voice", "engine"): "str",
    ("voice", "edge_voice"): "str",
    ("voice", "rate"): "int",
    ("permissions", "tier"): "str",
    ("offline", "mode"): "str",
    ("ui", "theme"): "str",
    ("ui", "accent"): "str",
}


def _format_value(value, value_type: str) -> str:
    if value_type == "int":
        return str(int(value))
    if value_type == "bool":
        return "true" if value else "false"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def set_scalar(text: str, section: str, key: str, value, value_type: str) -> str:
    """Return `text` with [section] key set to value. Adds the line if absent."""
    lines = text.splitlines()
    section_header = re.compile(r"^\s*\[(.+?)\]\s*$")
    key_line = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*(?:#.*)?)$")

    formatted = _format_value(value, value_type)
    in_section = False
    section_start = None
    for i, line in enumerate(lines):
        header = section_header.match(line)
        if header:
            in_section = header.group(1) == section
            if in_section:
                section_start = i
            continue
        if in_section:
            m = key_line.match(line)
            if m:
                lines[i] = f"{m.group(1)}{formatted}{m.group(3)}"
                return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    # Key not found in the section: insert it just after the header, or add
    # the whole section at the end if it doesn't exist.
    if section_start is not None:
        lines.insert(section_start + 1, f"{key} = {formatted}")
    else:
        lines.append(f"[{section}]")
        lines.append(f"{key} = {formatted}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def apply_updates(path: Path, updates: dict[str, object]) -> list[str]:
    """Apply {"'section.key'": value} updates to config.toml. Returns applied keys.

    Unknown or non-editable keys are ignored (defence against a tampered GUI
    request writing arbitrary config).
    """
    text = path.read_text(encoding="utf-8")
    applied = []
    for dotted, value in updates.items():
        if "." not in dotted:
            continue
        section, _, key = dotted.partition(".")
        spec = EDITABLE.get((section, key))
        if spec is None:
            continue
        text = set_scalar(text, section, key, value, spec)
        applied.append(dotted)
    path.write_text(text, encoding="utf-8")
    return applied
