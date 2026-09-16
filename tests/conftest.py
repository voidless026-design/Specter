from __future__ import annotations

import pytest

from ev_assistant.config import Config
from ev_assistant.knowledge import Knowledge
from ev_assistant.memory import Memory


@pytest.fixture
def memory(tmp_path) -> Memory:
    return Memory(tmp_path / "memory.sqlite3")


@pytest.fixture
def knowledge(tmp_path) -> Knowledge:
    return Knowledge(tmp_path / "knowledge.sqlite3")


@pytest.fixture
def cfg(tmp_path) -> Config:
    # offline_mode="online" forces the online path in brain tests without a
    # real network check; individual tests override fields as needed.
    return Config(
        anthropic_api_key="sk-ant-test",
        control_token="test-token",
        offline_mode="online",
        data_dir=tmp_path,
    )
