from __future__ import annotations

from ev_assistant.config import load_config, validate_for_daemon, write_default_config


def test_write_default_config_is_idempotent(tmp_path):
    path = tmp_path / "config.toml"
    written = write_default_config(path)
    original_contents = written.read_text(encoding="utf-8")

    write_default_config(path)  # should not overwrite

    assert written.read_text(encoding="utf-8") == original_contents


def test_load_config_reads_written_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    monkeypatch.setenv("EV_CONTROL_TOKEN", "shh")

    path = tmp_path / "config.toml"
    write_default_config(path)
    cfg = load_config(path)

    assert cfg.anthropic_api_key == "sk-ant-abc"
    assert cfg.control_token == "shh"
    assert cfg.model == "claude-opus-5"
    assert cfg.effort == "low"
    assert "hey e v" in cfg.wake_phrases
    assert cfg.feed_interval_minutes == 30


def test_load_config_missing_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EV_CONTROL_TOKEN", raising=False)

    cfg = load_config(tmp_path / "does-not-exist.toml")

    assert cfg.anthropic_api_key == ""
    assert cfg.control_token == ""
    assert cfg.model == "claude-opus-5"


def test_secrets_are_never_read_from_the_toml_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('[brain]\nmodel = "claude-opus-5"\n', encoding="utf-8")

    cfg = load_config(path)

    assert cfg.anthropic_api_key == ""


def test_input_device_numeric_string_is_coerced_to_int(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\ninput_device = "2"\n', encoding="utf-8")

    cfg = load_config(path)

    assert cfg.input_device == 2


def test_input_device_name_stays_a_string(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\ninput_device = "USB Microphone"\n', encoding="utf-8")

    cfg = load_config(path)

    assert cfg.input_device == "USB Microphone"


def test_validate_for_daemon_flags_missing_secrets(cfg):
    cfg.anthropic_api_key = ""
    cfg.control_token = ""

    problems = validate_for_daemon(cfg)

    assert len(problems) == 2


def test_validate_for_daemon_passes_when_secrets_set(cfg):
    assert validate_for_daemon(cfg) == []
