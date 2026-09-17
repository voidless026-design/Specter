from __future__ import annotations

from ev_assistant.config import DEFAULT_WAKE_NAMES, DEFAULT_WAKE_PREFIXES
from ev_assistant.audio.wake_word import match_wake

NAMES = DEFAULT_WAKE_NAMES
PREFIXES = DEFAULT_WAKE_PREFIXES


def test_bare_name_wakes():
    m = match_wake("e v", NAMES, PREFIXES)
    assert m is not None
    assert m.command is None


def test_prefix_plus_name_wakes():
    assert match_wake("hey e v", NAMES, PREFIXES) is not None
    assert match_wake("yo ev", NAMES, PREFIXES) is not None


def test_inline_command_is_extracted():
    m = match_wake("e v open firefox", NAMES, PREFIXES)
    assert m is not None
    assert m.command == "open firefox"


def test_filler_after_name_is_stripped():
    m = match_wake("hey e v can you open firefox", NAMES, PREFIXES)
    assert m is not None
    assert m.command == "open firefox"


def test_name_mid_sentence_without_prefix_does_not_wake():
    # "seven" won't match, and a stray "ev" inside a sentence shouldn't wake
    # unless it starts the utterance or follows a prefix word.
    assert match_wake("i went to the shop today", NAMES, PREFIXES) is None


def test_unrelated_speech_does_not_wake():
    assert match_wake("the weather is nice today", NAMES, PREFIXES) is None
    assert match_wake("", NAMES, PREFIXES) is None


def test_b_mishearing_variant_wakes():
    # Vosk sometimes transcribes the V as a B.
    assert match_wake("hey e b", NAMES, PREFIXES) is not None


def test_command_after_prefixed_name():
    m = match_wake("okay evie what's the weather", NAMES, PREFIXES)
    assert m is not None
    assert m.command == "what's the weather"
