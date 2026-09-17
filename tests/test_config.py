from __future__ import annotations

from ev_assistant.config import load_config, read_env_file, validate_for_daemon, write_default_config


def test_write_default_config_is_idempotent(tmp_path):
    path = tmp_path / "config.toml"
    written = write_default_config(path)
    original = written.read_text(encoding="utf-8")
    write_default_config(path)  # should not overwrite
    assert written.read_text(encoding="utf-8") == original


def test_load_config_reads_written_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    monkeypatch.setenv("EV_CONTROL_TOKEN", "shh")

    path = tmp_path / "config.toml"
    write_default_config(path)
    cfg = load_config(path, env_path=tmp_path / "no-env")

    assert cfg.anthropic_api_key == "sk-ant-abc"
    assert cfg.control_token == "shh"
    assert cfg.model == "claude-opus-5"
    assert "e v" in cfg.wake_names
    assert "hey" in cfg.wake_prefixes
    assert cfg.permission_tier == "standard"
    assert cfg.voice_engine == "auto"
    assert cfg.edge_voice == "en-AU-NatashaNeural"


def test_secrets_fall_back_to_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EV_CONTROL_TOKEN", raising=False)
    env = tmp_path / "env"
    env.write_text('ANTHROPIC_API_KEY="sk-ant-fromfile"\nEV_CONTROL_TOKEN=tok123\n', encoding="utf-8")

    cfg = load_config(tmp_path / "config.toml", env_path=env)

    assert cfg.anthropic_api_key == "sk-ant-fromfile"
    assert cfg.control_token == "tok123"


def test_shell_env_wins_over_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_CONTROL_TOKEN", "from-shell")
    env = tmp_path / "env"
    env.write_text("EV_CONTROL_TOKEN=from-file\n", encoding="utf-8")

    cfg = load_config(tmp_path / "config.toml", env_path=env)

    assert cfg.control_token == "from-shell"


def test_read_env_file_ignores_comments_and_blanks(tmp_path):
    env = tmp_path / "env"
    env.write_text("# a comment\n\nKEY=value\nQUOTED=\"spaced value\"\n", encoding="utf-8")
    parsed = read_env_file(env)
    assert parsed == {"KEY": "value", "QUOTED": "spaced value"}


def test_secrets_are_never_read_from_the_toml_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('[brain]\nmodel = "claude-opus-5"\n', encoding="utf-8")
    cfg = load_config(path, env_path=tmp_path / "no-env")
    assert cfg.anthropic_api_key == ""


def test_input_device_numeric_string_is_coerced_to_int(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\ninput_device = "2"\n', encoding="utf-8")
    assert load_config(path, env_path=tmp_path / "no-env").input_device == 2


def test_input_device_name_stays_a_string(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\ninput_device = "USB Microphone"\n', encoding="utf-8")
    assert load_config(path, env_path=tmp_path / "no-env").input_device == "USB Microphone"


def test_validate_only_requires_control_token(cfg):
    # The brain is optional (Ollama needs no key; a missing brain isn't fatal),
    # so only the control token blocks startup now.
    cfg.anthropic_api_key = ""
    cfg.control_token = ""
    problems = validate_for_daemon(cfg)
    assert len(problems) == 1
    assert "EV_CONTROL_TOKEN" in problems[0]


def test_validate_passes_with_token_and_no_key(cfg):
    cfg.anthropic_api_key = ""
    assert validate_for_daemon(cfg) == []


def test_default_provider_is_ollama(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    write_default_config(path)
    cfg = load_config(path, env_path=tmp_path / "no-env")
    assert cfg.brain_provider == "ollama"
    assert cfg.ollama_model == "llama3.1"
