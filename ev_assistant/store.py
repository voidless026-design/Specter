"""Retrieval store for E.V. - documents, chunks, facts, and their indexes.

One SQLite file holds everything: the prose (``chunks``), a keyword index
(``chunks_fts``, FTS5) and a vector index (``chunks_vec``, sqlite-vec). No
external service, so the whole knowledge base stays a single file you can
copy or back up - which is why `ev export` still works.

A few structural notes:

- The vector table's dimension is fixed at creation, so it's built lazily
  once the embedding model's dimension is known (see ``ensure_vec_table``),
  and the model name/dimension are recorded in ``meta`` so a model swap can
  be refused rather than silently mixing incompatible vectors.
- FTS rows are kept in sync by triggers. Vector rows are *deleted* by
  trigger but inserted by the embedding worker, because a chunk's vector
  doesn't exist at insert time - embedding happens afterwards, resumably.
- ``chunks.embedding_status`` is what makes indexing resumable: a crash
  mid-ingest leaves pending rows, and the worker picks up where it left off.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

EMBED_PENDING = 0
EMBED_DONE = 1
EMBED_FAILED = 2

# Same three states for ingest-time enrichment (summary + facts), so a
# document that has been looked at once isn't looked at again forever.
ENRICH_PENDING = 0
ENRICH_DONE = 1
ENRICH_FAILED = 2


@dataclass
class Document:
    id: int
    source_uri: str
    source_type: str
    namespace: str
    title: str
    content_hash: str
    fetched_at: float
    published_at: float | None = None
    license: str = ""
    summary: str = ""
    token_count: int = 0
    enrichment_status: int = ENRICH_PENDING


@dataclass
class Chunk:
    id: int
    document_id: int
    ordinal: int
    text: str
    heading_path: str = ""
    token_count: int = 0
    # Populated when a chunk is read back as part of a search result.
    document: Document | None = None
    score: float = 0.0


@dataclass
class Fact:
    id: int
    document_id: int
    statement: str
    subject: str
    confidence: float
    asserted_at: float


def normalize_text(text: str) -> str:
    """Canonical form used for the dedupe hash (whitespace-insensitive)."""
    return " ".join((text or "").split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _load_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into this connection.

    Extensions are per-connection, so every connection we hand out must do
    this. Returns False when sqlite-vec isn't installed, which downgrades
    the store to keyword-only search rather than failing outright.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError):
        return False  # Python built without extension loading support.
    try:
        sqlite_vec.load(conn)
        return True
    except sqlite3.OperationalError:
        logger.exception("sqlite-vec is installed but wouldn't load")
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError):
            pass


@lru_cache(maxsize=1)
def vec_supported() -> bool:
    """Whether sqlite-vec can be loaded at all on this install.

    Probed once against an in-memory database so the answer doesn't depend
    on which connection happens to ask.
    """
    probe = sqlite3.connect(":memory:")
    try:
        return _load_vec(probe)
    finally:
        probe.close()


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vec_available = vec_supported()
        with self._connect() as conn:
            self._migrate(conn)
        if not self.vec_available:
            logger.warning(
                "sqlite-vec unavailable - semantic search is disabled, keyword search still works. "
                "Install it with: pip install sqlite-vec"
            )

    # -- connections ---------------------------------------------------

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _load_vec(conn)
            # Cascades must reach the FTS/vec triggers on `chunks`, so both
            # pragmas are set on every connection (they are per-connection).
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA recursive_triggers=ON;")
            yield conn
        finally:
            conn.close()

    # -- schema --------------------------------------------------------

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Create or upgrade the schema. Safe to run on every startup."""
        # Tuned for a store that may reach tens of GB: WAL for concurrent
        # readers during ingest, a large page cache, and memory temp tables.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")  # ~64MB
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source_uri TEXT NOT NULL,
                source_type TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'reference',
                title TEXT NOT NULL DEFAULT '',
                content_hash TEXT UNIQUE NOT NULL,
                fetched_at REAL NOT NULL,
                published_at REAL,
                license TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                enrichment_status INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_uri);
            CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);
            CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at);

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                heading_path TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                embedding_status INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
            CREATE INDEX IF NOT EXISTS idx_chunks_embed ON chunks(embedding_status);

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                statement TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                asserted_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            -- Keep the keyword index in lockstep with chunks.
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        # Columns added after v1. CREATE TABLE above covers a fresh store; this
        # covers one that already exists. Additive only - a column is never
        # dropped or retyped, so an older E.V. can still open a newer file.
        self._add_column(conn, "documents", "enrichment_status",
                         "INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_enrich ON documents(enrichment_status)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

        # The vector table needs a dimension, so it's only created once an
        # embedding model has registered one.
        dim = self.get_meta("embedding_dim", conn=conn)
        if self.vec_available and dim:
            self._create_vec_table(conn, int(dim))

    @staticmethod
    def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        """ALTER TABLE ... ADD COLUMN, but only when it isn't already there."""
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            logger.info("Migrating %s: adding %s", table, column)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _create_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )
        # A vec0 table can't be the target of a plain FK, so deletions are
        # mirrored by trigger to guarantee no orphaned vectors.
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_vec WHERE chunk_id = old.id;
            END;
            """
        )
        conn.commit()

    def ensure_vec_table(self, dim: int, model: str = "") -> bool:
        """Create the vector index for `dim`. False if vectors are unavailable.

        The dimension is baked into the vec0 table at creation, so switching
        embedding models is refused here rather than silently mixing vectors
        that don't share a space. `ev reindex` is the way through.
        """
        if not self.vec_available:
            return False
        recorded = self.get_meta("embedding_dim")
        if recorded and int(recorded) != int(dim):
            raise ValueError(
                f"This store was built with {recorded}-dimension embeddings; the current model "
                f"produces {dim}. Run `ev reindex` to rebuild it, or switch back."
            )
        with self._connect() as conn:
            self._create_vec_table(conn, dim)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_dim', ?)", (str(dim),)
            )
            if model:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_model', ?)",
                    (model,),
                )
            conn.commit()
        return True

    def drop_vec_table(self) -> None:
        """Tear the vector index down so a different model can rebuild it."""
        with self._connect() as conn:
            conn.execute("DROP TRIGGER IF EXISTS chunks_vec_ad")
            if self._has_vec_table(conn):
                conn.execute("DROP TABLE chunks_vec")
            conn.execute("DELETE FROM meta WHERE key IN ('embedding_dim', 'embedding_model')")
            conn.execute(f"UPDATE chunks SET embedding_status = {EMBED_PENDING}")
            conn.commit()

    # -- meta ----------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None, conn=None) -> str | None:
        def _get(c):
            row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

        if conn is not None:
            return _get(conn)
        with self._connect() as c:
            return _get(c)

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    # -- writes --------------------------------------------------------

    def add_document(
        self,
        *,
        source_uri: str,
        source_type: str,
        title: str,
        text: str,
        chunks: list[dict],
        namespace: str = "reference",
        published_at: float | None = None,
        license: str = "",
        token_count: int = 0,
        force: bool = False,
    ) -> int | None:
        """Insert a document and its chunks. Returns the document id, or None
        if identical content is already stored (and `force` is False).

        `chunks` are dicts of {text, heading_path, token_count, ordinal?} as
        produced by chunking.py.
        """
        chash = content_hash(text)
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?", (chash,)
            ).fetchone()
            if existing and not force:
                return None
            if existing:
                # Re-ingesting identical text: drop the old copy so the new
                # one doesn't collide on content_hash.
                self._purge_document(conn, int(existing["id"]))

            cur = conn.execute(
                "INSERT INTO documents (source_uri, source_type, namespace, title, content_hash,"
                " fetched_at, published_at, license, summary, token_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?)",
                (source_uri, source_type, namespace, title, chash, now,
                 published_at, license, token_count),
            )
            doc_id = int(cur.lastrowid)
            for i, ch in enumerate(chunks):
                conn.execute(
                    "INSERT INTO chunks (document_id, ordinal, text, heading_path, token_count)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (doc_id, ch.get("ordinal", i), ch["text"],
                     ch.get("heading_path", ""), ch.get("token_count", 0)),
                )
            conn.commit()
        return doc_id

    def set_summary(self, document_id: int, summary: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE documents SET summary = ? WHERE id = ?", (summary, document_id))
            conn.commit()

    def replace_facts(self, document_id: int, facts: list[dict]) -> int:
        """Set a document's facts, dropping whatever was there before.

        Re-enrichment must not double a document's facts, and the extractor
        is not deterministic enough to dedupe by statement text.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
            conn.commit()
        return self.add_facts(document_id, facts)

    def add_facts(self, document_id: int, facts: list[dict]) -> int:
        now = time.time()
        with self._connect() as conn:
            for f in facts:
                conn.execute(
                    "INSERT INTO facts (document_id, statement, subject, confidence, asserted_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (document_id, f["statement"], f.get("subject", ""),
                     float(f.get("confidence", 0.0)), now),
                )
            conn.commit()
        return len(facts)

    # -- enrichment bookkeeping ----------------------------------------

    def pending_enrichment(self, limit: int = 32) -> list[Document]:
        """Documents still awaiting a summary and facts, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE enrichment_status = ? ORDER BY id LIMIT ?",
                (ENRICH_PENDING, limit),
            ).fetchall()
        return [_document_from_row(r) for r in rows]

    def mark_enriched(self, document_id: int, status: int = ENRICH_DONE) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE documents SET enrichment_status = ? WHERE id = ?",
                         (status, document_id))
            conn.commit()

    def reset_enrichment(self, namespace: str | None = None) -> int:
        """Requeue documents for enrichment (for `ev reindex --enrich`)."""
        with self._connect() as conn:
            if namespace:
                cur = conn.execute(
                    "UPDATE documents SET enrichment_status = ? WHERE namespace = ?",
                    (ENRICH_PENDING, namespace),
                )
            else:
                cur = conn.execute("UPDATE documents SET enrichment_status = ?",
                                   (ENRICH_PENDING,))
            conn.commit()
            return cur.rowcount

    def document_text(self, document_id: int, max_chars: int = 0) -> str:
        """A document's prose, reassembled from its chunks in order.

        The full text isn't stored - chunks are the source of truth - so this
        is how enrichment gets something to summarise.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        text = "\n\n".join(r["text"] for r in rows)
        if max_chars and len(text) > max_chars:
            # Prefer a paragraph break, so the model never sees half a
            # sentence. When the first paragraph alone is over budget there
            # isn't one to use, so fall back to a word boundary rather than
            # handing over a truncated word.
            cut = text.rfind("\n\n", 0, max_chars)
            if cut <= max_chars // 2:
                cut = text.rfind(" ", 0, max_chars)
            text = text[:cut] if cut > 0 else text[:max_chars]
        return text.rstrip()

    def facts_for(self, document_id: int) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE document_id = ? ORDER BY id", (document_id,)
            ).fetchall()
        return [Fact(id=r["id"], document_id=r["document_id"], statement=r["statement"],
                     subject=r["subject"], confidence=r["confidence"],
                     asserted_at=r["asserted_at"]) for r in rows]

    # -- embedding bookkeeping -----------------------------------------

    def pending_chunks(self, limit: int = 256) -> list[Chunk]:
        """Chunks still awaiting a vector, oldest first (resumable indexing)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, heading_path, token_count"
                " FROM chunks WHERE embedding_status = ? ORDER BY id LIMIT ?",
                (EMBED_PENDING, limit),
            ).fetchall()
        return [_chunk_from_row(r) for r in rows]

    def store_embeddings(self, pairs: list[tuple[int, bytes]]) -> None:
        """Write vectors for chunk ids and mark them done, atomically."""
        if not pairs:
            return
        with self._connect() as conn:
            if not self._has_vec_table(conn):
                raise RuntimeError(
                    "No vector table yet - call Store.ensure_vec_table(dim) before storing "
                    "embeddings so the dimension is fixed."
                )
            for chunk_id, blob in pairs:
                conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)", (chunk_id, blob)
                )
                conn.execute(
                    "UPDATE chunks SET embedding_status = ? WHERE id = ?", (EMBED_DONE, chunk_id)
                )
            conn.commit()

    def mark_embedding_failed(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE chunks SET embedding_status = ? WHERE id = ?",
                [(EMBED_FAILED, cid) for cid in chunk_ids],
            )
            conn.commit()

    def reset_embeddings(self, namespace: str | None = None) -> int:
        """Mark chunks unembedded and drop their vectors (for `ev reindex`)."""
        with self._connect() as conn:
            if namespace:
                ids = [r["id"] for r in conn.execute(
                    "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
                    " WHERE d.namespace = ?", (namespace,)).fetchall()]
            else:
                ids = [r["id"] for r in conn.execute("SELECT id FROM chunks").fetchall()]
            has_vec = self._has_vec_table(conn)
            for batch in _batched(ids, 500):
                placeholders = ",".join("?" * len(batch))
                if has_vec:
                    conn.execute(
                        f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})", batch
                    )
                conn.execute(
                    f"UPDATE chunks SET embedding_status = {EMBED_PENDING}"
                    f" WHERE id IN ({placeholders})",
                    batch,
                )
            conn.commit()
        return len(ids)

    # -- deletes -------------------------------------------------------

    def _purge_document(self, conn: sqlite3.Connection, document_id: int) -> bool:
        """Remove a document, its chunks, facts and index rows. No commit.

        Chunks are deleted explicitly rather than left to the foreign-key
        cascade so the FTS and vector triggers fire on every SQLite build,
        regardless of how it was compiled.
        """
        if self._has_vec_table(conn):
            conn.execute(
                "DELETE FROM chunks_vec WHERE chunk_id IN"
                " (SELECT id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cur.rowcount > 0

    def delete_document(self, document_id: int) -> bool:
        """Delete a document and everything hanging off it."""
        with self._connect() as conn:
            deleted = self._purge_document(conn, document_id)
            conn.commit()
        return deleted

    def delete_by_source(self, source_uri: str) -> int:
        """Delete every document ingested from `source_uri`. Returns the count."""
        with self._connect() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM documents WHERE source_uri = ?", (source_uri,)).fetchall()]
            for doc_id in ids:
                self._purge_document(conn, int(doc_id))
            conn.commit()
        return len(ids)

    def _has_vec_table(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        return row is not None

    # -- reads ---------------------------------------------------------

    def get_document(self, document_id: int) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _document_from_row(row) if row else None

    def get_chunk(self, chunk_id: int, with_document: bool = True) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, document_id, ordinal, text, heading_path, token_count"
                " FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if not row:
                return None
            chunk = _chunk_from_row(row)
            if with_document:
                drow = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (chunk.document_id,)
                ).fetchone()
                if drow:
                    chunk.document = _document_from_row(drow)
        return chunk

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, Chunk]:
        """Batch chunk fetch with documents attached, keyed by chunk id."""
        if not chunk_ids:
            return {}
        out: dict[int, Chunk] = {}
        rows: list = []
        with self._connect() as conn:
            for batch in _batched(list(chunk_ids), 500):
                placeholders = ",".join("?" * len(batch))
                rows.extend(conn.execute(
                    f"SELECT c.id, c.document_id, c.ordinal, c.text, c.heading_path, c.token_count,"
                    f" d.id AS d_id, d.source_uri, d.source_type, d.namespace, d.title,"
                    f" d.content_hash, d.fetched_at, d.published_at, d.license, d.summary,"
                    f" d.token_count AS d_token_count"
                    f" FROM chunks c JOIN documents d ON d.id = c.document_id"
                    f" WHERE c.id IN ({placeholders})",
                    batch,
                ).fetchall())
        for r in rows:
            chunk = _chunk_from_row(r)
            chunk.document = Document(
                id=r["d_id"], source_uri=r["source_uri"], source_type=r["source_type"],
                namespace=r["namespace"], title=r["title"], content_hash=r["content_hash"],
                fetched_at=r["fetched_at"], published_at=r["published_at"],
                license=r["license"], summary=r["summary"], token_count=r["d_token_count"],
            )
            out[chunk.id] = chunk
        return out

    def neighbors(self, chunk_id: int, window: int = 1) -> list[Chunk]:
        """Adjacent chunks by ordinal, for restoring continuity."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document_id, ordinal FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if not row:
                return []
            rows = conn.execute(
                "SELECT id, document_id, ordinal, text, heading_path, token_count FROM chunks"
                " WHERE document_id = ? AND ordinal BETWEEN ? AND ? AND id != ? ORDER BY ordinal",
                (row["document_id"], row["ordinal"] - window, row["ordinal"] + window, chunk_id),
            ).fetchall()
        return [_chunk_from_row(r) for r in rows]

    def knows_hash(self, text: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1", (content_hash(text),)
            ).fetchone() is not None

    # -- stats ---------------------------------------------------------

    def stats(self) -> dict:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            facts = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
            embedded = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE embedding_status = ?", (EMBED_DONE,)
            ).fetchone()["n"]
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE embedding_status = ?", (EMBED_PENDING,)
            ).fetchone()["n"]
            by_ns = {r["namespace"]: r["n"] for r in conn.execute(
                "SELECT namespace, COUNT(*) AS n FROM documents GROUP BY namespace").fetchall()}
            by_type = {r["source_type"]: r["n"] for r in conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type").fetchall()}
            orphans = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks c LEFT JOIN documents d ON d.id = c.document_id"
                " WHERE d.id IS NULL"
            ).fetchone()["n"]
            enriched = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE enrichment_status = ?",
                (ENRICH_DONE,),
            ).fetchone()["n"]
            unenriched = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE enrichment_status = ?",
                (ENRICH_PENDING,),
            ).fetchone()["n"]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "documents": docs, "chunks": chunks, "facts": facts,
            "embedded_chunks": embedded, "pending_chunks": pending,
            "orphan_chunks": orphans, "by_namespace": by_ns, "by_source_type": by_type,
            "enriched_documents": enriched, "pending_enrichment": unenriched,
            "size_bytes": size,
            "embedding_model": self.get_meta("embedding_model", ""),
            "embedding_dim": self.get_meta("embedding_dim", ""),
            "vec_available": self.vec_available,
        }


def _batched(items: list, size: int):
    """Yield fixed-size slices - SQLite caps how many `?` one query may bind."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _chunk_from_row(row) -> Chunk:
    return Chunk(
        id=row["id"], document_id=row["document_id"], ordinal=row["ordinal"],
        text=row["text"], heading_path=row["heading_path"], token_count=row["token_count"],
    )


def _document_from_row(row) -> Document:
    return Document(
        id=row["id"], source_uri=row["source_uri"], source_type=row["source_type"],
        namespace=row["namespace"], title=row["title"], content_hash=row["content_hash"],
        fetched_at=row["fetched_at"], published_at=row["published_at"],
        license=row["license"], summary=row["summary"], token_count=row["token_count"],
        enrichment_status=row["enrichment_status"] if "enrichment_status" in row.keys()
        else ENRICH_PENDING,
    )
