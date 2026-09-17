from __future__ import annotations

from ev_assistant.brain import Brain


class FakeProvider:
    def __init__(self, name="fake", reply="hello there", avail=True):
        self.name = name
        self._reply = reply
        self._avail = avail
        self.seen = {}

    def available(self, cfg):
        return self._avail

    def respond(self, cfg, context, history, user_text, tools, executor):
        self.seen = {"context": context, "history": history, "user_text": user_text, "tools": tools}
        return self._reply


def _use_chain(monkeypatch, providers):
    monkeypatch.setattr("ev_assistant.brain.build_chain", lambda cfg: providers)


def test_respond_uses_first_available_provider(cfg, memory, knowledge, monkeypatch):
    fp = FakeProvider(reply="Sure thing.")
    _use_chain(monkeypatch, [fp])
    brain = Brain(cfg, memory, knowledge)

    reply = brain.respond("what time is it")

    assert reply == "Sure thing."
    assert fp.seen["user_text"] == "what time is it"
    turns = memory.recent_turns()
    assert turns[-2] == {"role": "user", "content": "what time is it"}
    assert turns[-1] == {"role": "assistant", "content": "Sure thing."}


def test_respond_falls_through_to_next_provider(cfg, memory, knowledge, monkeypatch):
    dead = FakeProvider(name="dead", reply=None)  # returns None -> try next
    live = FakeProvider(name="live", reply="I got it.")
    _use_chain(monkeypatch, [dead, live])
    brain = Brain(cfg, memory, knowledge)

    assert brain.respond("hi") == "I got it."


def test_respond_skips_unavailable_providers(cfg, memory, knowledge, monkeypatch):
    off = FakeProvider(name="off", avail=False)
    live = FakeProvider(name="live", reply="online")
    _use_chain(monkeypatch, [off, live])
    brain = Brain(cfg, memory, knowledge)

    assert brain.respond("hi") == "online"
    assert off.seen == {}  # never called


def test_context_includes_feed_facts_and_learned_notes(cfg, memory, knowledge, monkeypatch):
    memory.add_fact("weather", "It's sunny and 25C", external_id="w1")
    knowledge.add_document("Water", "notes", "Boil water for a minute to purify it.")
    fp = FakeProvider()
    _use_chain(monkeypatch, [fp])
    brain = Brain(cfg, memory, knowledge)

    brain.respond("how do I purify water and what's the weather")

    ctx = fp.seen["context"].lower()
    assert "sunny" in ctx
    assert "boil water" in ctx


def test_falls_back_to_offline_when_no_provider_answers(cfg, memory, knowledge, monkeypatch):
    knowledge.add_document("Fire", "notes", "Gather dry tinder and strike a spark.")
    _use_chain(monkeypatch, [FakeProvider(reply=None)])
    brain = Brain(cfg, memory, knowledge)

    reply = brain.respond("how do I start a fire")
    assert "tinder" in reply.lower()


def test_offline_reason_no_key_for_claude_without_key(cfg, memory, knowledge, monkeypatch):
    cfg.brain_provider = "claude"
    cfg.anthropic_api_key = ""
    _use_chain(monkeypatch, [FakeProvider(reply=None)])
    brain = Brain(cfg, memory, knowledge)

    # Empty knowledge + no key => the "brain isn't set up" guidance.
    reply = brain.respond("tell me a joke")
    assert "brain" in reply.lower() or "doctor" in reply.lower()
