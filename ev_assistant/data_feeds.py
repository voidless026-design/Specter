"""Background loop that keeps pulling fresh data into E.V.'s memory.

This is what "getting smarter over time" actually means in this project: a
growing, timestamped fact store E.V. can draw on - not retrained model
weights. See the README for why that's the honest version of the claim.

Security note: feed content is passed to Claude as labeled data ("Things
E.V. currently knows"), never as instructions, and E.V. has no tool-use /
action capability yet - but a feed you don't control could still attempt a
prompt-injection-style headline. Only point `feeds` at sources you trust.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import feedparser
import httpx

from ev_assistant.config import Config
from ev_assistant.memory import Memory

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SNIPPET_MAX = 220


def _clean(text: str) -> str:
    text = _TAG_RE.sub(" ", text or "")
    text = _WS_RE.sub(" ", text).strip()
    return text[:_SNIPPET_MAX]


class DataFeedLoop:
    def __init__(self, cfg: Config, memory: Memory):
        self.cfg = cfg
        self.memory = memory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.last_run_at: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="ev-data-feeds", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:  # a bad fetch cycle shouldn't kill the loop
                logger.exception("Data feed cycle failed")
                self.last_error = str(e)
            self._stop_event.wait(max(60, self.cfg.feed_interval_minutes * 60))

    def run_once(self) -> int:
        """Run one fetch cycle now. Returns how many new facts were learned."""
        new_facts = 0
        for url in self.cfg.feeds:
            new_facts += self._pull_feed(url)
        if self.cfg.weather_location:
            new_facts += self._pull_weather(self.cfg.weather_location)
        self.last_run_at = time.time()
        self.last_error = None
        return new_facts

    def _pull_feed(self, url: str) -> int:
        try:
            parsed = feedparser.parse(url)
        except Exception:
            logger.exception("Failed to fetch feed %s", url)
            return 0

        source_title = _clean(getattr(parsed.feed, "title", "")) or url
        added = 0
        for entry in parsed.entries[:15]:
            external_id = entry.get("id") or entry.get("link")
            title = _clean(entry.get("title", ""))
            if not external_id or not title:
                continue
            summary = _clean(entry.get("summary", ""))
            text = f"{source_title}: {title}"
            if summary and summary != title:
                text += f" - {summary}"
            inserted = self.memory.add_fact(
                source=f"feed:{source_title}", text=text, tags="news", external_id=external_id
            )
            if inserted:
                added += 1
        return added

    def _pull_weather(self, location: str) -> int:
        try:
            response = httpx.get(
                f"https://wttr.in/{location}",
                params={"format": "3"},
                timeout=10.0,
                headers={"User-Agent": "curl/8.0"},  # wttr.in serves plain text for curl-like UAs
            )
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to fetch weather for %s", location)
            return 0

        text = _clean(response.text)
        if not text:
            return 0
        self.memory.add_fact(source="weather", text=f"Current weather - {text}", tags="weather")
        return 1
