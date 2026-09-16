from __future__ import annotations

from ev_assistant.config import (
    API_KEY_PLACEHOLDER,
    looks_like_real_key,
    read_env_file,
    write_env_var,
)


def test_looks_like_real_key():
    assert looks_like_real_key("sk-ant-abc123")
    assert not looks_like_real_key("")
    assert not looks_like_real_key(API_KEY_PLACEHOLDER)
    assert not looks_like_real_key("random-string")
    assert not looks_like_real_key("sk-openai-xyz")


def test_write_env_var_preserves_other_keys(tmp_path):
    env = tmp_path / "env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-old\nEV_CONTROL_TOKEN=tok\n", encoding="utf-8")

    write_env_var("ANTHROPIC_API_KEY", "sk-ant-new", path=env)

    parsed = read_env_file(env)
    assert parsed["ANTHROPIC_API_KEY"] == "sk-ant-new"
    assert parsed["EV_CONTROL_TOKEN"] == "tok"  # untouched


def test_write_env_var_creates_file(tmp_path):
    env = tmp_path / "sub" / "env"
    write_env_var("EV_CONTROL_TOKEN", "generated", path=env)
    assert read_env_file(env)["EV_CONTROL_TOKEN"] == "generated"


def test_claude_provider_unavailable_without_key(cfg):
    from ev_assistant.providers import ClaudeProvider

    cfg.anthropic_api_key = "sk-ant-your-key-here"  # placeholder
    assert ClaudeProvider().available(cfg) is False
    cfg.anthropic_api_key = "sk-ant-realish"
    assert ClaudeProvider().available(cfg) is True


def test_offline_no_key_still_runs_commands(cfg, memory, knowledge, monkeypatch):
    from ev_assistant.brain import Brain

    # No brain reachable, but a command should still be executed by the
    # offline rule parser rather than returning "need a key".
    cfg.brain_provider = "claude"
    cfg.anthropic_api_key = ""
    cfg.permission_tier = "safe"
    monkeypatch.setattr("ev_assistant.brain.build_chain", lambda c: [])
    brain = Brain(cfg, memory, knowledge)
    reply = brain.respond("open firefox")
    assert "brain" not in reply.lower() or "firefox" in reply.lower()
