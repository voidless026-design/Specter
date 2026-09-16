"""Offline knowledge base for E.V. - the survivalist/reference brain.

Documents you feed in with `ev learn` (Wikipedia articles, text/markdown
files, PDFs, web pages) are chunked and stored in a full-text SQLite index
so E.V. can answer from them with no internet. Uses SQLite's FTS5 when the
Python build has it (it usually does), and transparently falls back to a
LIKE-based search when it doesn't, so ingestion and lookup never hard-fail.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_WS_RE = re.compile(r"\s+")


@dataclass
class Passage:
    title: str
    source: str
    text: str
    score: float = 0.0


def _fts5_available() -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        return True
    except sqlite3.OperationalError:
        return False


# Rough conversion for sizing chunks in tokens without shipping a tokenizer.
# English averages ~4 characters per token; close enough for chunk budgets.
CHARS_PER_TOKEN = 4


def chunk_text(text: str, target_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
    """Split text into ~`target_tokens` chunks with `overlap_tokens` of carry-over.

    The overlap matters for retrieval: without it, a fact that straddles a
    chunk boundary is split in half and neither half answers the question.
    """
    target_chars = max(200, target_tokens * CHARS_PER_TOKEN)
    overlap_chars = max(0, min(overlap_tokens * CHARS_PER_TOKEN, target_chars // 2))

    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        para = _WS_RE.sub(" ", para)
        if len(buf) + len(para) + 1 <= target_chars:
            buf = f"{buf} {para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= target_chars:
                buf = para
            else:
                for i in range(0, len(para), target_chars):
                    chunks.append(para[i : i + target_chars])
                buf = ""
    if buf:
        chunks.append(buf)

    if overlap_chars and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            overlapped.append((prev[-overlap_chars:] + " " + cur).strip())
        chunks = overlapped
    return chunks


class Knowledge:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_fts = _fts5_available()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            if self.use_fts:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS passages "
                    "USING fts5(title, source, text, created_at UNINDEXED)"
                )
            else:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS passages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT, source TEXT, text TEXT, created_at REAL
                    )
                    """
                )
            # Document-level ledger, kept separate from the FTS table so it can
            # be added to an existing database without rebuilding the index.
            # The content hash is what makes re-ingesting the same page a no-op.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    content_hash TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    passages INTEGER NOT NULL,
                    fetched_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            yield conn
        finally:
            conn.close()

    def add_document(self, title: str, source: str, text: str, force: bool = False) -> int:
        """Chunk and store a document. Returns passages added (0 if already known).

        Identical content is skipped by SHA-256 so re-running `ev learn` on the
        same page - or crawling into a page twice - doesn't duplicate passages.
        Pass force=True to replace an existing copy (e.g. the page changed).
        """
        text = (text or "").strip()
        if not text:
            return 0
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT passages FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if existing and not force:
                return 0
            if existing and force:
                conn.execute("DELETE FROM passages WHERE title = ? AND source = ?", (title, source))
                conn.execute("DELETE FROM documents WHERE content_hash = ?", (content_hash,))

            chunks = chunk_text(text)
            now = time.time()
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO passages (title, source, text, created_at) VALUES (?, ?, ?, ?)",
                    (title, source, chunk, now),
                )
            conn.execute(
                "INSERT OR REPLACE INTO documents (content_hash, title, source, passages, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (content_hash, title, source, len(chunks), now),
            )
            conn.commit()
        return len(chunks)

    def knows_source(self, source: str) -> bool:
        """True if a document from this exact source has been ingested."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,)
            ).fetchone() is not None

    def document_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def passage_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def sources(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT title FROM passages ORDER BY title"
            ).fetchall()
        return [r[0] for r in rows]

    def search(self, query: str, limit: int = 5) -> list[Passage]:
        terms = re.findall(r"[a-zA-Z0-9]+", query)
        if not terms:
            return []
        if self.use_fts:
            return self._search_fts(terms, limit)
        return self._search_like(terms, limit)

    def _search_fts(self, terms: list[str], limit: int) -> list[Passage]:
        # OR the terms together and rank by bm25 (lower = better match).
        match_expr = " OR ".join(terms)
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT title, source, text, bm25(passages) AS rank "
                    "FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
                    (match_expr, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Passage(title=r[0], source=r[1], text=r[2], score=-float(r[3])) for r in rows]

    def _search_like(self, terms: list[str], limit: int) -> list[Passage]:
        where = " OR ".join(["text LIKE ?"] * len(terms))
        params = [f"%{t}%" for t in terms]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT title, source, text FROM passages WHERE {where} LIMIT ?",
                (*params, limit * 4),
            ).fetchall()
        scored = []
        for title, source, text in rows:
            low = text.lower()
            score = sum(low.count(t.lower()) for t in terms)
            scored.append(Passage(title=title, source=source, text=text, score=float(score)))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:limit]

    def format_context(self, query: str, limit: int = 5) -> str:
        passages = self.search(query, limit=limit)
        if not passages:
            return ""
        blocks = [f"[{p.title}] {p.text}" for p in passages]
        return "\n\n".join(blocks)
