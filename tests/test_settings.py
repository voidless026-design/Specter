from __future__ import annotations

from ev_assistant.config import load_config, write_default_config
from ev_assistant.settings import apply_updates, set_scalar


def test_set_scalar_replaces_value_and_keeps_comments():
    text = "[personality]\n# a comment\nhumor = 65\nhonesty = 90\n"
    out = set_scalar(text, "personality", "humor", 20, "int")
    assert "humor = 20" in out
    assert "# a comment" in out  # comment preserved
    assert "honesty = 90" in out  # other keys untouched


def test_set_scalar_quotes_strings():
    text = '[voice]\nengine = "auto"\n'
    out = set_scalar(text, "voice", "engine", "espeak", "str")
    assert 'engine = "espeak"' in out


def test_set_scalar_only_touches_target_section():
    text = "[a]\nx = 1\n\n[b]\nx = 2\n"
    out = set_scalar(text, "b", "x", 9, "int")
    assert "[a]\nx = 1" in out
    assert "x = 9" in out


def test_apply_updates_roundtrips_through_config(tmp_path):
    path = tmp_path / "config.toml"
    write_default_config(path)

    applied = apply_updates(
        path,
        {
            "personality.humor": 10,
            "voice.engine": "espeak",
            "permissions.tier": "full",
            "bogus.key": "ignored",  # not in EDITABLE - must be skipped
        },
    )

    assert set(applied) == {"personality.humor", "voice.engine", "permissions.tier"}
    cfg = load_config(path, env_path=tmp_path / "no-env")
    assert cfg.humor == 10
    assert cfg.voice_engine == "espeak"
    assert cfg.permission_tier == "full"


def test_apply_updates_ignores_non_editable_keys(tmp_path):
    path = tmp_path / "config.toml"
    write_default_config(path)
    applied = apply_updates(path, {"brain.model": "evil-model"})
    assert applied == []
    cfg = load_config(path, env_path=tmp_path / "no-env")
    assert cfg.model == "claude-opus-5"  # unchanged
