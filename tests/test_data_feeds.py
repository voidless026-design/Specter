from __future__ import annotations

from types import SimpleNamespace

from ev_assistant.data_feeds import DataFeedLoop, _clean

_SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item>
  <title>First &amp; Best Headline</title>
  <link>https://example.com/1</link>
  <guid>https://example.com/1</guid>
  <description><![CDATA[Some <b>bold</b> summary text.]]></description>
</item>
<item>
  <title>Second Headline</title>
  <link>https://example.com/2</link>
  <guid>https://example.com/2</guid>
  <description>Plain summary.</description>
</item>
</channel>
</rss>
"""


def test_clean_strips_html_and_collapses_whitespace():
    assert _clean("Some <b>bold</b>   text\nwith   newlines") == "Some bold text with newlines"


def test_clean_truncates_long_text():
    assert len(_clean("x" * 500)) == 220


def test_pull_feed_adds_facts_and_dedups(cfg, memory):
    loop = DataFeedLoop(cfg, memory)

    added_first = loop._pull_feed(_SAMPLE_RSS)
    added_second = loop._pull_feed(_SAMPLE_RSS)

    assert added_first == 2
    assert added_second == 0  # same guids - already known
    assert memory.fact_count() == 2

    facts = memory.recent_facts()
    assert any("First & Best Headline" in f.text for f in facts)
    assert any("bold summary text" in f.text for f in facts)


def test_pull_weather_stores_a_fact(cfg, memory, monkeypatch):
    loop = DataFeedLoop(cfg, memory)

    def fake_get(url, params=None, timeout=None, headers=None):
        assert "wttr.in" in url
        return SimpleNamespace(
            text="Miami: ⛅ +30°C",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("ev_assistant.data_feeds.httpx.get", fake_get)

    added = loop._pull_weather("Miami")

    assert added == 1
    facts = memory.recent_facts()
    assert any("30" in f.text for f in facts)


def test_pull_weather_failure_does_not_raise(cfg, memory, monkeypatch):
    loop = DataFeedLoop(cfg, memory)

    def fake_get(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("ev_assistant.data_feeds.httpx.get", fake_get)

    added = loop._pull_weather("Nowhere")

    assert added == 0


def test_run_once_pulls_configured_feeds_and_weather(cfg, memory, monkeypatch):
    cfg.feeds = [_SAMPLE_RSS]
    cfg.weather_location = "Miami"
    loop = DataFeedLoop(cfg, memory)
    monkeypatch.setattr(loop, "_pull_weather", lambda location: 1)

    new_facts = loop.run_once()

    assert new_facts == 3  # 2 headlines + 1 weather reading
    assert loop.last_run_at is not None
    assert loop.last_error is None
