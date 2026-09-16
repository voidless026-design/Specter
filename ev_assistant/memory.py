"""Persistent memory for E.V.: conversation history + a growing fact store.

This is intentionally simple (SQLite + keyword overlap scoring) rather than
an embeddings/vector-DB setup - it needs no extra ML dependencies, stays
fast on CPU, and is good enough for a personal assistant's memory size.
Facts come from two places: background data feeds (data_feeds.py) and
anything explicitly told to E.V. in conversation could be added the same
way by a future version.

Every connection is opened per-call rather than held open, so the daemon
thread, the data-feed thread, and the control-API thread(s) can all touch
the store without coordinating locks themselves; WAL mode keeps concurrent
readers/writers from blocking each other.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them",
    "my", "your", "his", "its", "our", "their", "of", "to", "in", "on",
    "for", "and", "or", "but", "with", "at", "by", "from", "about", "as",
    "what", "whats", "how", "do", "does", "did", "can", "could", "will",
    "would", "should", "hey", "please", "tell", "know",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Fact:
    id: int
    source: str
    text: str
    tags: str
    created_at: float


def _tokenize(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


class Memory:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    external_id TEXT UNIQUE,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        try:
            yield conn
        finally:
            conn.close()

    # -- facts --------------------------------------------------------

    def add_fact(self, source: str, text: str, tags: str = "", external_id: str | None = None) -> bool:
        """Insert a fact. Returns False if it was already known (dedup by external_id)."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO facts (source, external_id, text, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, external_id, text, tags, time.time()),
            )
            conn.commit()
            return cur.rowcount > 0

    def fact_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def recent_facts(self, limit: int = 20) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source, text, tags, created_at FROM facts "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Fact(id=r[0], source=r[1], text=r[2], tags=r[3], created_at=r[4]) for r in rows]

    def search_facts(self, query: str, limit: int = 6, recent_pool: int = 200) -> list[Fact]:
        """Keyword-overlap + recency ranking over the most recent `recent_pool` facts."""
        candidates = self.recent_facts(limit=recent_pool)
        if not candidates:
            return []

        query_words = _tokenize(query)
        now = time.time()
        scored: list[tuple[float, Fact]] = []
        for fact in candidates:
            fact_words = _tokenize(fact.text) | _tokenize(fact.tags)
            overlap = len(query_words & fact_words)
            age_days = max(0.0, (now - fact.created_at) / 86400)
            recency_bonus = max(0.0, 1.0 - age_days / 30.0) * 0.5
            scored.append((overlap * 2.0 + recency_bonus, fact))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        matched = [fact for score, fact in scored if score > 0][:limit]
        if len(matched) < limit:
            matched_ids = {f.id for f in matched}
            for fact in candidates:
                if len(matched) >= limit:
                    break
                if fact.id not in matched_ids:
                    matched.append(fact)
                    matched_ids.add(fact.id)
        return matched[:limit]

    def format_context(self, query: str, limit: int = 6) -> str:
        facts = self.search_facts(query, limit=limit)
        if not facts:
            return ""
        lines = [f"- {f.text}" for f in facts]
        return "Things E.V. currently knows (from memory and background feeds):\n" + "\n".join(lines)

    # -- conversation history ------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )
            conn.commit()

    def recent_turns(self, limit: int = 12) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
