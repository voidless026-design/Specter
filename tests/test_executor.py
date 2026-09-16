from __future__ import annotations

import pytest

from ev_assistant.tools.executor import Executor
from ev_assistant.tools.specs import tools_for_tier


def test_tier_ordering(cfg):
    cfg.permission_tier = "standard"
    ex = Executor(cfg)
    assert ex.tier_ok("safe")
    assert ex.tier_ok("standard")
    assert not ex.tier_ok("full")


def test_tools_for_tier_filters():
    safe = {t["name"] for t in tools_for_tier("safe")}
    standard = {t["name"] for t in tools_for_tier("standard")}
    full = {t["name"] for t in tools_for_tier("full")}

    assert "open_app" in safe
    assert "run_shell" not in safe
    assert "close_app" not in safe

    assert "close_app" in standard
    assert "gnome_setting" in standard
    assert "run_shell" not in standard

    assert "run_shell" in full


def test_dispatch_denies_below_tier(cfg):
    cfg.permission_tier = "safe"
    ex = Executor(cfg)
    result = ex.dispatch("run_shell", {"command": "echo hi"})
    assert not result.ok
    assert "permission tier" in result.message


def test_unknown_tool(cfg):
    result = Executor(cfg).dispatch("does_not_exist", {})
    assert not result.ok


def test_forbidden_pattern_is_blocked(cfg):
    cfg.permission_tier = "full"
    cfg.forbidden_patterns = ["rm -rf /"]
    ex = Executor(cfg, confirm=lambda _d: True)
    result = ex.dispatch("run_shell", {"command": "sudo rm -rf / --no-preserve-root"})
    assert not result.ok
    assert "forbidden" in result.message.lower()


def test_destructive_requires_confirmation(cfg):
    cfg.permission_tier = "full"
    cfg.confirm_destructive = True
    denied = Executor(cfg, confirm=lambda _d: False)
    result = denied.dispatch("run_shell", {"command": "rm important.txt"})
    assert not result.ok
    assert "confirm" in result.message.lower()


def test_non_destructive_shell_runs(cfg):
    cfg.permission_tier = "full"
    ex = Executor(cfg, confirm=lambda _d: False)
    result = ex.dispatch("run_shell", {"command": "echo hello-from-ev"})
    assert result.ok
    assert "hello-from-ev" in result.message


def test_confirmed_destructive_runs(cfg, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("bye")
    cfg.permission_tier = "full"
    ex = Executor(cfg, confirm=lambda _d: True)
    result = ex.dispatch("run_shell", {"command": f"rm {victim}"})
    assert result.ok
    assert not victim.exists()
