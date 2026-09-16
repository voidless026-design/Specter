from __future__ import annotations

from ev_assistant.offline import OfflineBrain
from ev_assistant.tools.executor import Executor, ToolResult


class RecordingExecutor(Executor):
    """Executor that records dispatch calls instead of touching the system."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return ToolResult(True, f"did {name}")


def _brain(cfg, knowledge):
    ex = RecordingExecutor(cfg)
    return OfflineBrain(cfg, knowledge, ex), ex


def test_open_app_command(cfg, knowledge):
    brain, ex = _brain(cfg, knowledge)
    brain.respond("open firefox")
    assert ("open_app", {"name": "firefox"}) in ex.calls


def test_open_website_command(cfg, knowledge):
    brain, ex = _brain(cfg, knowledge)
    brain.respond("open github.com")
    assert ex.calls[0][0] == "open_url"


def test_close_app_command(cfg, knowledge):
    brain, ex = _brain(cfg, knowledge)
    brain.respond("close spotify")
    assert ("close_app", {"name": "spotify"}) in ex.calls


def test_volume_command(cfg, knowledge):
    brain, ex = _brain(cfg, knowledge)
    brain.respond("set volume to 40")
    assert ("set_volume", {"percent": 40}) in ex.calls


def test_launch_jegeo_command(cfg, knowledge):
    brain, ex = _brain(cfg, knowledge)
    brain.respond("launch jegeo")
    assert ex.calls[0][0] == "launch_jegeo"


def test_question_falls_through_to_knowledge(cfg, knowledge):
    knowledge.add_document("Snakebite", "notes", "For a snakebite, keep the limb still and below the heart.")
    brain, ex = _brain(cfg, knowledge)
    reply = brain.respond("what do I do about a snakebite")
    assert ex.calls == []  # not treated as a command
    assert "limb" in reply.lower()


def test_unknown_question_with_empty_kb(cfg, knowledge):
    brain, _ = _brain(cfg, knowledge)
    reply = brain.respond("what is the airspeed of a swallow")
    assert "knowledge base" in reply.lower() or "offline" in reply.lower()
