from __future__ import annotations

import pytest

from ev_assistant.config import Config
from ev_assistant.memory import Memory


@pytest.fixture
def memory(tmp_path) -> Memory:
    return Memory(tmp_path / "memory.sqlite3")


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        anthropic_api_key="sk-ant-test",
        control_token="test-token",
        data_dir=tmp_path,
    )
