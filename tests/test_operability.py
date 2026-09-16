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


def test_brain_status_no_key(cfg, memory, knowledge):
    from ev_assistant.brain import Brain

    cfg.anthropic_api_key = "sk-ant-your-key-here"  # placeholder
    cfg.offline_mode = "auto"
    brain = Brain(cfg, memory, knowledge)
    # Placeholder key => no client => offline path with a key-specific message.
    assert brain.client is None
    reply = brain.respond("what is the capital of France")
    assert "api key" in reply.lower() or "set-key" in reply.lower()


def test_offline_no_key_still_runs_commands(cfg, memory, knowledge):
    from ev_assistant.brain import Brain

    cfg.anthropic_api_key = ""
    cfg.offline_mode = "auto"
    cfg.permission_tier = "safe"
    brain = Brain(cfg, memory, knowledge)
    reply = brain.respond("open firefox")
    # A command still works offline even without a key (executor tries to run
    # it); the reply is the executor's message, not the "need a key" text.
    assert "api key" not in reply.lower()
