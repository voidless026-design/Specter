from __future__ import annotations

from ev_assistant.personality import build_persona_text, build_system_blocks


def test_persona_text_never_includes_markdown_instruction_is_present(cfg):
    text = build_persona_text(cfg)
    assert "markdown" in text.lower()


def test_high_honesty_is_blunt(cfg):
    cfg.honesty = 95
    text = build_persona_text(cfg)
    assert "bluntly candid" in text.lower()


def test_low_honesty_is_diplomatic(cfg):
    cfg.honesty = 10
    text = build_persona_text(cfg)
    assert "soften" in text.lower()


def test_high_humor_encourages_wit(cfg):
    cfg.humor = 90
    text = build_persona_text(cfg)
    assert "dry wit" in text.lower()


def test_low_humor_stays_plain(cfg):
    cfg.humor = 5
    text = build_persona_text(cfg)
    assert "matter-of-fact" in text.lower()


def test_system_blocks_cache_only_the_persona_block(cfg):
    blocks = build_system_blocks(cfg, memory_context="Things E.V. currently knows:\n- it's sunny")

    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
    assert blocks[1]["text"].startswith("Things E.V. currently knows")


def test_system_blocks_omit_second_block_when_no_memory(cfg):
    blocks = build_system_blocks(cfg, memory_context="")
    assert len(blocks) == 1
