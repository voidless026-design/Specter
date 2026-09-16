"""Phase 1 of the retrieval spec: schema, index sync, and clean deletes.

Everything here runs offline with no microphone and no model download. The
vector-specific tests skip themselves when sqlite-vec isn't installed, so a
keyword-only install still gets a green suite.
"""

from __future__ import annotations

import sqlite3
import struct

import pytest

from ev_assistant.store import (
    EMBED_DONE,
    EMBED_FAILED,
    EMBED_PENDING,
    SCHEMA_VERSION,
    Store,
    content_hash,
    normalize_text,
    vec_supported,
)

needs_vec = pytest.mark.skipif(not vec_supported(), reason="sqlite-vec not installed")


def make_chunks(*texts: str) -> list[dict]:
    return [
        {"text": t, "heading_path": "Top > Section", "token_count": len(t) // 4}
        for t in texts
    ]


def add(store: Store, *, uri="notes://a", title="A", text="hello world", chunks=None, **kw) -> int:
    doc_id = store.add_document(
        source_uri=uri,
        source_type=kw.pop("source_type", "note"),
        title=title,
        text=text,
        chunks=chunks if chunks is not None else make_chunks(text),
        **kw,
    )
    return doc_id


def fts_rowids(store: Store) -> set[int]:
    with store._connect() as conn:
        return {r["rowid"] for r in conn.execute("SELECT rowid FROM chunks_fts").fetchall()}


def vec_ids(store: Store) -> set[int]:
    with store._connect() as conn:
        if not store._has_vec_table(conn):
            return set()
        return {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks_vec").fetchall()}


def blob(dim: int, value: float = 0.1) -> bytes:
    return struct.pack(f"{dim}f", *([value] * dim))


# -- hashing ---------------------------------------------------------------


def test_normalize_collapses_whitespace():
    assert normalize_text("  a\n\n b\tc  ") == "a b c"


def test_content_hash_ignores_formatting_only_changes():
    assert content_hash("one   two") == content_hash("one\ntwo")
    assert content_hash("one two") != content_hash("one three")


# -- schema ----------------------------------------------------------------


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "store.sqlite3"
    first = Store(path)
    doc_id = add(first, text="persisted across reopens")

    # Re-opening runs _migrate again; nothing may be lost or duplicated.
    second = Store(path)
    assert second.get_document(doc_id) is not None
    assert second.stats()["documents"] == 1
    assert second.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_wal_mode_and_foreign_keys_are_on(store):
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_foreign_key_rejects_orphan_chunk(store):
    with store._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunks (document_id, ordinal, text) VALUES (9999, 0, 'orphan')"
            )


# -- documents and chunks --------------------------------------------------


def test_add_document_stores_chunks_in_order(store):
    doc_id = add(store, text="alpha beta gamma", chunks=make_chunks("alpha", "beta", "gamma"))
    assert doc_id is not None
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT ordinal, text, heading_path, embedding_status FROM chunks"
            " WHERE document_id = ? ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
    assert [r["text"] for r in rows] == ["alpha", "beta", "gamma"]
    assert [r["ordinal"] for r in rows] == [0, 1, 2]
    assert all(r["heading_path"] == "Top > Section" for r in rows)
    # Nothing is embedded at insert time - that's the worker's job.
    assert all(r["embedding_status"] == EMBED_PENDING for r in rows)


def test_duplicate_content_is_skipped(store):
    first = add(store, uri="notes://a", text="Boil water for one minute.")
    again = add(store, uri="notes://b", text="Boil   water for one\nminute.")
    assert first is not None
    assert again is None
    assert store.stats()["documents"] == 1


def test_force_replaces_the_existing_copy(store):
    add(store, title="Old", text="same text", chunks=make_chunks("same text"))
    replaced = add(store, title="New", text="same text", chunks=make_chunks("same text"),
                   force=True)

    assert replaced is not None
    assert store.get_document(replaced).title == "New"
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["orphan_chunks"] == 0
    assert len(fts_rowids(store)) == 1


def test_knows_hash_matches_normalized_text(store):
    add(store, text="a stored paragraph")
    assert store.knows_hash("a   stored\nparagraph")
    assert not store.knows_hash("something else")


def test_metadata_round_trips(store):
    doc_id = add(
        store,
        uri="https://example.org/x",
        title="Title",
        text="body text",
        namespace="personal",
        published_at=1700000000.0,
        license="CC-BY-SA",
        token_count=42,
        source_type="web",
    )
    doc = store.get_document(doc_id)
    assert doc.source_uri == "https://example.org/x"
    assert doc.source_type == "web"
    assert doc.namespace == "personal"
    assert doc.published_at == 1700000000.0
    assert doc.license == "CC-BY-SA"
    assert doc.token_count == 42
    assert doc.summary == ""

    store.set_summary(doc_id, "A short summary.")
    assert store.get_document(doc_id).summary == "A short summary."


def test_default_namespace_is_reference(store):
    doc_id = add(store, text="unlabelled")
    assert store.get_document(doc_id).namespace == "reference"


# -- FTS sync --------------------------------------------------------------


def test_fts_indexes_chunks_on_insert(store):
    add(store, text="the bowline makes a fixed loop", chunks=make_chunks("the bowline makes a fixed loop"))
    with store._connect() as conn:
        hits = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'bowline'").fetchall()
    assert len(hits) == 1


def test_fts_follows_chunk_updates(store):
    doc_id = add(store, text="original wording", chunks=make_chunks("original wording"))
    with store._connect() as conn:
        chunk_id = conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchone()["id"]
        conn.execute("UPDATE chunks SET text = 'replacement wording' WHERE id = ?", (chunk_id,))
        conn.commit()
        assert conn.execute(
            "SELECT count(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'original'"
        ).fetchone()["n"] == 0
        assert conn.execute(
            "SELECT count(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'replacement'"
        ).fetchone()["n"] == 1


def test_fts_stays_consistent_after_deletes(store):
    doc_id = add(store, text="purify water by boiling", chunks=make_chunks("purify water by boiling"))
    assert fts_rowids(store)
    store.delete_document(doc_id)
    assert fts_rowids(store) == set()
    # FTS5's own audit: catches a desynced external-content index.
    with store._connect() as conn:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")


# -- deletes ---------------------------------------------------------------


def test_delete_document_leaves_no_orphans(store):
    doc_id = add(store, text="doomed", chunks=make_chunks("one", "two", "three"))
    store.add_facts(doc_id, [{"statement": "a fact", "subject": "s", "confidence": 0.9}])
    keeper = add(store, uri="notes://keep", text="kept", chunks=make_chunks("kept"))

    assert store.delete_document(doc_id) is True

    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["facts"] == 0
    assert stats["orphan_chunks"] == 0
    assert len(fts_rowids(store)) == 1  # only the keeper's chunk is still indexed
    assert store.get_document(keeper) is not None


def test_delete_document_returns_false_when_missing(store):
    assert store.delete_document(4242) is False


def test_delete_by_source_removes_every_copy(store):
    add(store, uri="feed://news", text="story one", chunks=make_chunks("story one"))
    add(store, uri="feed://news", text="story two", chunks=make_chunks("story two"))
    add(store, uri="feed://other", text="story three", chunks=make_chunks("story three"))

    assert store.delete_by_source("feed://news") == 2

    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["orphan_chunks"] == 0
    assert len(fts_rowids(store)) == 1


def test_delete_by_source_unknown_uri_is_a_noop(store):
    add(store, text="kept")
    assert store.delete_by_source("feed://nothing") == 0
    assert store.stats()["documents"] == 1


# -- facts -----------------------------------------------------------------


def test_facts_are_stored_and_cascade(store):
    doc_id = add(store, text="source doc")
    n = store.add_facts(doc_id, [
        {"statement": "Water boils at 100C at sea level.", "subject": "water", "confidence": 0.95},
        {"statement": "Boiling kills most pathogens.", "subject": "water"},
    ])
    assert n == 2
    assert store.stats()["facts"] == 2
    store.delete_document(doc_id)
    assert store.stats()["facts"] == 0


# -- embedding bookkeeping -------------------------------------------------


def test_pending_chunks_are_returned_oldest_first(store):
    add(store, text="a b c", chunks=make_chunks("a", "b", "c"))
    pending = store.pending_chunks()
    assert [c.text for c in pending] == ["a", "b", "c"]
    assert [c.ordinal for c in pending] == [0, 1, 2]


def test_pending_chunks_respects_limit(store):
    add(store, text="a b c", chunks=make_chunks("a", "b", "c"))
    assert len(store.pending_chunks(limit=2)) == 2


def test_mark_embedding_failed_drops_chunks_from_the_queue(store):
    add(store, text="a b", chunks=make_chunks("a", "b"))
    ids = [c.id for c in store.pending_chunks()]
    store.mark_embedding_failed(ids[:1])
    assert [c.id for c in store.pending_chunks()] == ids[1:]
    with store._connect() as conn:
        status = conn.execute(
            "SELECT embedding_status FROM chunks WHERE id = ?", (ids[0],)
        ).fetchone()["embedding_status"]
    assert status == EMBED_FAILED


def test_store_embeddings_without_a_vec_table_is_refused(store):
    add(store, text="a")
    chunk_id = store.pending_chunks()[0].id
    with pytest.raises(RuntimeError, match="ensure_vec_table"):
        store.store_embeddings([(chunk_id, blob(4))])


# -- vectors ---------------------------------------------------------------


@needs_vec
def test_embedding_round_trip_and_knn(store):
    dim = 4
    assert store.ensure_vec_table(dim, model="test-model") is True
    add(store, text="near far", chunks=make_chunks("near", "far"))
    chunks = store.pending_chunks()
    near, far = chunks[0], chunks[1]
    store.store_embeddings([
        (near.id, struct.pack("4f", 1.0, 0.0, 0.0, 0.0)),
        (far.id, struct.pack("4f", 0.0, 1.0, 0.0, 0.0)),
    ])

    assert store.pending_chunks() == []
    stats = store.stats()
    assert stats["embedded_chunks"] == 2
    assert stats["pending_chunks"] == 0
    assert stats["embedding_model"] == "test-model"
    assert stats["embedding_dim"] == str(dim)

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = 1"
            " ORDER BY distance",
            (struct.pack("4f", 0.9, 0.1, 0.0, 0.0),),
        ).fetchall()
    assert [r["chunk_id"] for r in rows] == [near.id]


@needs_vec
def test_storing_an_embedding_twice_replaces_it(store):
    store.ensure_vec_table(4)
    add(store, text="a")
    chunk_id = store.pending_chunks()[0].id
    store.store_embeddings([(chunk_id, blob(4, 0.1))])
    store.store_embeddings([(chunk_id, blob(4, 0.9))])
    assert vec_ids(store) == {chunk_id}


@needs_vec
def test_deleting_a_document_removes_its_vectors(store):
    store.ensure_vec_table(4)
    doc_id = add(store, text="a b", chunks=make_chunks("a", "b"))
    store.store_embeddings([(c.id, blob(4)) for c in store.pending_chunks()])
    assert len(vec_ids(store)) == 2

    store.delete_document(doc_id)
    assert vec_ids(store) == set()
    assert store.stats()["chunks"] == 0


@needs_vec
def test_vec_table_survives_reopen(tmp_path):
    path = tmp_path / "store.sqlite3"
    first = Store(path)
    first.ensure_vec_table(4, model="test-model")
    doc_id = add(first, text="a")
    first.store_embeddings([(c.id, blob(4)) for c in first.pending_chunks()])

    second = Store(path)
    assert second.get_meta("embedding_dim") == "4"
    assert len(vec_ids(second)) == 1
    # And the trigger is still wired up after the reopen.
    second.delete_document(doc_id)
    assert vec_ids(second) == set()


@needs_vec
def test_changing_embedding_dimension_is_refused(store):
    store.ensure_vec_table(4, model="small")
    with pytest.raises(ValueError, match="reindex"):
        store.ensure_vec_table(8, model="big")


@needs_vec
def test_drop_vec_table_clears_vectors_and_requeues(store):
    store.ensure_vec_table(4)
    add(store, text="a b", chunks=make_chunks("a", "b"))
    store.store_embeddings([(c.id, blob(4)) for c in store.pending_chunks()])

    store.drop_vec_table()

    assert store.get_meta("embedding_dim") is None
    assert len(store.pending_chunks()) == 2
    assert store.ensure_vec_table(8, model="big") is True


@needs_vec
def test_reset_embeddings_requeues_everything(store):
    store.ensure_vec_table(4)
    add(store, uri="notes://ref", text="a b", chunks=make_chunks("a", "b"))
    store.store_embeddings([(c.id, blob(4)) for c in store.pending_chunks()])
    assert store.stats()["embedded_chunks"] == 2

    assert store.reset_embeddings() == 2
    assert vec_ids(store) == set()
    assert len(store.pending_chunks()) == 2


@needs_vec
def test_reset_embeddings_can_target_one_namespace(store):
    store.ensure_vec_table(4)
    add(store, uri="notes://ref", text="a", namespace="reference")
    add(store, uri="notes://me", text="b", namespace="personal")
    store.store_embeddings([(c.id, blob(4)) for c in store.pending_chunks()])

    assert store.reset_embeddings(namespace="personal") == 1
    pending = store.pending_chunks()
    assert len(pending) == 1
    assert pending[0].text == "b"
    assert store.stats()["embedded_chunks"] == 1


# -- reads -----------------------------------------------------------------


def test_get_chunk_attaches_its_document(store):
    doc_id = add(store, title="Knots", text="bowline", chunks=make_chunks("bowline"))
    chunk_id = store.pending_chunks()[0].id

    chunk = store.get_chunk(chunk_id)
    assert chunk.text == "bowline"
    assert chunk.document is not None
    assert chunk.document.id == doc_id
    assert chunk.document.title == "Knots"

    bare = store.get_chunk(chunk_id, with_document=False)
    assert bare.document is None


def test_get_chunk_missing_returns_none(store):
    assert store.get_chunk(999) is None


def test_get_chunks_batches_with_documents(store):
    add(store, title="Doc", text="a b c", chunks=make_chunks("a", "b", "c"))
    ids = [c.id for c in store.pending_chunks()]

    got = store.get_chunks(ids)
    assert set(got) == set(ids)
    assert all(c.document.title == "Doc" for c in got.values())
    assert store.get_chunks([]) == {}


def test_neighbors_returns_surrounding_chunks(store):
    add(store, text="a b c d e", chunks=make_chunks("a", "b", "c", "d", "e"))
    ids = [c.id for c in store.pending_chunks()]

    around_c = store.neighbors(ids[2], window=1)
    assert [c.text for c in around_c] == ["b", "d"]

    around_a = store.neighbors(ids[0], window=1)
    assert [c.text for c in around_a] == ["b"]

    assert store.neighbors(9999) == []


def test_neighbors_stay_inside_their_document(store):
    add(store, uri="notes://one", text="a b", chunks=make_chunks("a", "b"))
    add(store, uri="notes://two", text="c d", chunks=make_chunks("c", "d"))
    second_doc_first_chunk = store.pending_chunks()[2]

    assert [c.text for c in store.neighbors(second_doc_first_chunk.id, window=5)] == ["d"]


# -- stats -----------------------------------------------------------------


def test_stats_groups_by_namespace_and_type(store):
    add(store, uri="a", text="one", namespace="reference", source_type="wikipedia")
    add(store, uri="b", text="two", namespace="personal", source_type="note")
    add(store, uri="c", text="three", namespace="personal", source_type="note")

    stats = store.stats()
    assert stats["by_namespace"] == {"reference": 1, "personal": 2}
    assert stats["by_source_type"] == {"wikipedia": 1, "note": 2}
    assert stats["size_bytes"] > 0
    assert stats["vec_available"] is vec_supported()


def test_stats_on_an_empty_store(store):
    stats = store.stats()
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["orphan_chunks"] == 0
    assert stats["by_namespace"] == {}


def test_embedding_status_constants_are_distinct():
    assert len({EMBED_PENDING, EMBED_DONE, EMBED_FAILED}) == 3
