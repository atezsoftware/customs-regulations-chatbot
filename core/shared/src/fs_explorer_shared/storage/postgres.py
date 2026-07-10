"""
Postgres (+ pgvector) storage backend for index persistence.

Schema lives in `db/migrations/` (the `core_*` tables), not here — this class
assumes the schema already exists and only does data access. This mirrors
`DuckDBStorage`'s method-for-method behavior exactly; the only intentional
difference is that table creation is migration-owned, not runtime-owned.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from psycopg_pool import ConnectionPool

from .base import (
    ChunkRecord,
    DocumentRecord,
    SchemaRecord,
    make_amendment_chunk_id,
    make_chunk_id,
    make_document_id,
    stable_id,
)

__all__ = ["PostgresStorage"]


def _query_terms(query: str, max_terms: int = 8) -> list[str]:
    # \w is Unicode-aware for `str` patterns in Python 3, so this correctly
    # tokenizes Turkish ı/ğ/ü/ş/ö/ç — a plain [a-zA-Z0-9_] class would split
    # words on those characters.
    terms = re.findall(r"\w{3,}", query.lower())
    unique_terms: list[str] = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
        if len(unique_terms) >= max_terms:
            break
    if unique_terms:
        return unique_terms
    fallback = query.strip().lower()
    return [fallback] if fallback else []


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


class PostgresStorage:
    """Postgres-backed persistence for corpora, documents, chunks, and schemas."""

    def __init__(
        self,
        database_url: str,
        *,
        read_only: bool = False,
        initialize: bool = True,
        embedding_dim: int = 768,
        min_size: int = 1,
        max_size: int = 8,
    ) -> None:
        self.database_url = database_url
        self.read_only = read_only
        self.embedding_dim = embedding_dim
        self._pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            open=True,
            configure=self._configure_connection if read_only else None,
        )
        if initialize:
            self.initialize()

    def _configure_connection(self, conn: Any) -> None:
        if self.read_only:
            with conn.cursor() as cur:
                cur.execute("SET default_transaction_read_only = on")
            conn.commit()

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._pool.close()

    def initialize(self) -> None:
        """No-op: schema is owned by `db/migrations/`, not runtime code."""
        return None

    def get_or_create_corpus(self, root_path: str) -> str:
        normalized = str(Path(root_path).resolve())
        corpus_id = stable_id("corpus", normalized)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO core_corpora (id, root_path)
                    VALUES (%s, %s)
                    ON CONFLICT (root_path) DO NOTHING
                    """,
                    (corpus_id, normalized),
                )
                cur.execute(
                    "SELECT id FROM core_corpora WHERE root_path = %s", (normalized,)
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError(f"Failed to create corpus for path: {normalized}")
        return str(row[0])

    def get_corpus_id(self, root_path: str) -> str | None:
        normalized = str(Path(root_path).resolve())
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM core_corpora WHERE root_path = %s", (normalized,)
                )
                row = cur.fetchone()
        return str(row[0]) if row else None

    def upsert_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM core_chunk_embeddings
                    WHERE chunk_id IN (SELECT id FROM core_chunks WHERE document_id = %s)
                    """,
                    (document.id,),
                )
                cur.execute(
                    "DELETE FROM core_chunks WHERE document_id = %s", (document.id,)
                )
                cur.execute(
                    """
                    INSERT INTO core_documents (
                        id, corpus_id, relative_path, absolute_path, content,
                        metadata_json, file_mtime, file_size, content_sha256, is_deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, false)
                    ON CONFLICT (id) DO UPDATE SET
                        corpus_id = excluded.corpus_id,
                        relative_path = excluded.relative_path,
                        absolute_path = excluded.absolute_path,
                        content = excluded.content,
                        metadata_json = excluded.metadata_json,
                        file_mtime = excluded.file_mtime,
                        file_size = excluded.file_size,
                        content_sha256 = excluded.content_sha256,
                        last_indexed_at = now(),
                        is_deleted = false
                    """,
                    (
                        document.id,
                        document.corpus_id,
                        document.relative_path,
                        document.absolute_path,
                        document.content,
                        document.metadata_json,
                        document.file_mtime,
                        document.file_size,
                        document.content_sha256,
                    ),
                )
                if chunks:
                    cur.executemany(
                        """
                        INSERT INTO core_chunks (
                            id, document_id, text, position, start_char, end_char,
                            chunk_type, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        [
                            (
                                chunk.id,
                                chunk.doc_id,
                                chunk.text,
                                chunk.position,
                                chunk.start_char,
                                chunk.end_char,
                                chunk.chunk_type,
                                json.dumps(chunk.metadata or {}),
                            )
                            for chunk in chunks
                        ],
                    )
            conn.commit()

    def insert_chunk(self, *, chunk: ChunkRecord) -> None:
        """Insert a single chunk without touching any other chunk of its
        document. Idempotent (ON CONFLICT DO NOTHING) — unlike
        `upsert_document`, this never deletes existing chunks; it's the
        primitive amendment approval uses to add exactly one new chunk."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO core_chunks (
                        id, document_id, text, position, start_char, end_char,
                        chunk_type, metadata, source, status,
                        effective_start_date, effective_end_date,
                        supersedes_chunk_id, superseded_by_chunk_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.position,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.chunk_type,
                        json.dumps(chunk.metadata or {}),
                        chunk.source,
                        chunk.status,
                        chunk.effective_start_date,
                        chunk.effective_end_date,
                        chunk.supersedes_chunk_id,
                        chunk.superseded_by_chunk_id,
                    ),
                )
            conn.commit()

    def get_chunk(self, *, chunk_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, document_id, text, position, start_char, end_char,
                        chunk_type, metadata::text, source, status,
                        effective_start_date::text, effective_end_date::text,
                        supersedes_chunk_id, superseded_by_chunk_id
                    FROM core_chunks
                    WHERE id = %s
                    """,
                    (chunk_id,),
                )
                row = cur.fetchone()
        return self._row_to_chunk_dict(row) if row else None

    def supersede_chunk(
        self, *, old_chunk_id: str, new_chunk_id: str, effective_end_date: str | None
    ) -> bool:
        """Mark a chunk superseded by a specific successor. Returns False
        (no-op) if the chunk was not 'active' — callers should treat that as
        a race/conflict, not a silent success."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE core_chunks
                    SET status = 'superseded',
                        superseded_by_chunk_id = %s,
                        effective_end_date = COALESCE(%s, effective_end_date, CURRENT_DATE)
                    WHERE id = %s AND status = 'active'
                    """,
                    (new_chunk_id, effective_end_date, old_chunk_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def expire_chunk(self, *, chunk_id: str, effective_end_date: str) -> None:
        """Mark a chunk expired: its validity window closed without a direct
        successor (e.g. a temporary provision that lapsed)."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE core_chunks
                    SET status = 'expired', effective_end_date = %s
                    WHERE id = %s AND status = 'active'
                    """,
                    (effective_end_date, chunk_id),
                )
            conn.commit()

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if not active_relative_paths:
                    cur.execute(
                        """
                        UPDATE core_documents
                        SET is_deleted = true
                        WHERE corpus_id = %s AND is_deleted = false
                        """,
                        (corpus_id,),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE core_documents
                        SET is_deleted = true
                        WHERE corpus_id = %s
                          AND is_deleted = false
                          AND relative_path <> ALL(%s)
                        """,
                        (corpus_id, sorted(active_relative_paths)),
                    )
                cur.execute(
                    """
                    SELECT COUNT(*) FROM core_documents
                    WHERE corpus_id = %s AND is_deleted = true
                    """,
                    (corpus_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    def list_documents(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id, relative_path, absolute_path, file_size, file_mtime, is_deleted
            FROM core_documents
            WHERE corpus_id = %s
        """
        params: list[Any] = [corpus_id]
        if not include_deleted:
            sql += " AND is_deleted = false"
        sql += " ORDER BY relative_path"

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            {
                "id": str(row[0]),
                "relative_path": str(row[1]),
                "absolute_path": str(row[2]),
                "file_size": int(row[3]),
                "file_mtime": float(row[4]),
                "is_deleted": bool(row[5]),
            }
            for row in rows
        ]

    def count_chunks(self, *, corpus_id: str) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM core_chunks c
                    JOIN core_documents d ON d.id = c.document_id
                    WHERE d.corpus_id = %s AND d.is_deleted = false
                    """,
                    (corpus_id,),
                )
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def search_chunks(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        terms = _query_terms(query)
        if not terms:
            return []

        score_expr = " + ".join(
            ["CASE WHEN lower(c.text) LIKE '%%' || %s || '%%' THEN 1 ELSE 0 END"]
            * len(terms)
        )
        sql = f"""
            SELECT * FROM (
                SELECT
                    d.id AS doc_id,
                    d.relative_path,
                    d.absolute_path,
                    c.id AS chunk_id,
                    c.position,
                    c.text,
                    c.chunk_type,
                    c.metadata::text,
                    ({score_expr}) AS score
                FROM core_chunks c
                JOIN core_documents d ON d.id = c.document_id
                WHERE d.corpus_id = %s
                  AND d.is_deleted = false
                  AND c.status = 'active'
            ) ranked
            WHERE score > 0
            ORDER BY score DESC, relative_path ASC, position ASC
            LIMIT %s
        """
        params: list[Any] = [*terms, corpus_id, limit]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            {
                "doc_id": str(row[0]),
                "relative_path": str(row[1]),
                "absolute_path": str(row[2]),
                "chunk_id": str(row[3]),
                "position": int(row[4]),
                "text": str(row[5]),
                "chunk_type": str(row[6]) if row[6] is not None else None,
                "metadata": json.loads(str(row[7])) if row[7] is not None else {},
                "score": int(row[8]),
            }
            for row in rows
        ]

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not filters:
            return []

        sql = """
            SELECT
                d.id,
                d.relative_path,
                d.absolute_path,
                substring(d.content, 1, 320) AS preview_text
            FROM core_documents d
            WHERE d.corpus_id = %s
              AND d.is_deleted = false
        """
        params: list[Any] = [corpus_id]

        for flt in filters:
            clause, clause_params = self._metadata_clause(
                field=str(flt["field"]),
                operator=str(flt["operator"]),
                value=flt["value"],
            )
            sql += f"\n  AND {clause}"
            params.extend(clause_params)

        sql += "\nORDER BY d.relative_path ASC\nLIMIT %s"
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)  # ty: ignore[invalid-argument-type]
                rows = cur.fetchall()
        metadata_score = len(filters)
        return [
            {
                "doc_id": str(row[0]),
                "relative_path": str(row[1]),
                "absolute_path": str(row[2]),
                "preview_text": str(row[3]),
                "metadata_score": metadata_score,
            }
            for row in rows
        ]

    def get_document(self, *, doc_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, corpus_id, relative_path, absolute_path, content,
                        metadata_json::text, is_deleted
                    FROM core_documents
                    WHERE id = %s
                    LIMIT 1
                    """,
                    (doc_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "corpus_id": str(row[1]),
            "relative_path": str(row[2]),
            "absolute_path": str(row[3]),
            "content": str(row[4]),
            "metadata_json": str(row[5]),
            "is_deleted": bool(row[6]),
        }

    def list_document_chunks(self, *, doc_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, document_id, text, position, start_char, end_char,
                        chunk_type, metadata::text
                    FROM core_chunks
                    WHERE document_id = %s
                    ORDER BY position ASC
                    """,
                    (doc_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(row[0]),
                "doc_id": str(row[1]),
                "text": str(row[2]),
                "position": int(row[3]),
                "start_char": int(row[4]) if row[4] is not None else None,
                "end_char": int(row[5]) if row[5] is not None else None,
                "chunk_type": str(row[6]) if row[6] is not None else None,
                "metadata": json.loads(str(row[7])) if row[7] is not None else {},
            }
            for row in rows
        ]

    def get_document_chunks_by_prefix(
        self, *, corpus_root: str, relative_path_prefix: str
    ) -> dict[str, Any] | None:
        """Find the (single) active document under `corpus_root` whose
        `relative_path` starts with `relative_path_prefix`, plus all of its
        chunks and per-chunk embedding status.

        Mirrors the lookup a caller used to run directly against
        `core_corpora`/`core_documents`/`core_chunks`/`core_chunk_embeddings` —
        kept here so this schema has exactly one reader and one writer.
        """
        normalized = str(Path(corpus_root).resolve())
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        d.relative_path,
                        d.absolute_path,
                        c.text,
                        c.position,
                        c.start_char,
                        c.end_char,
                        c.chunk_type,
                        c.metadata::text,
                        EXISTS (
                            SELECT 1
                            FROM core_chunk_embeddings e
                            WHERE e.chunk_id = c.id
                        ) AS has_embedding
                    FROM core_corpora corpus
                    JOIN core_documents d ON d.corpus_id = corpus.id
                    JOIN core_chunks c ON c.document_id = d.id
                    WHERE corpus.root_path = %s
                      AND d.relative_path LIKE %s
                      AND d.is_deleted = false
                    ORDER BY c.position ASC
                    """,
                    (normalized, f"{relative_path_prefix}%"),
                )
                rows = cur.fetchall()

        if not rows:
            return None

        return {
            "document": {
                "id": str(rows[0][1]),
                "relative_path": str(rows[0][2]),
                "absolute_path": str(rows[0][3]),
            },
            "chunks": [
                {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "relative_path": str(row[2]),
                    "absolute_path": str(row[3]),
                    "text": str(row[4]),
                    "position": int(row[5]),
                    "start_char": int(row[6]) if row[6] is not None else None,
                    "end_char": int(row[7]) if row[7] is not None else None,
                    "chunk_type": str(row[8]) if row[8] is not None else None,
                    "metadata": json.loads(str(row[9])) if row[9] is not None else {},
                    "has_embedding": bool(row[10]),
                }
                for row in rows
            ],
        }

    def save_schema(
        self,
        *,
        corpus_id: str,
        name: str,
        schema_def: dict[str, Any],
        is_active: bool = True,
    ) -> str:
        schema_id = stable_id("schema", f"{corpus_id}:{name}")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if is_active:
                    cur.execute(
                        "UPDATE core_schemas SET is_active = false WHERE corpus_id = %s",
                        (corpus_id,),
                    )
                cur.execute(
                    """
                    INSERT INTO core_schemas (id, corpus_id, name, schema_def, is_active)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (corpus_id, name) DO UPDATE SET
                        schema_def = excluded.schema_def,
                        is_active = excluded.is_active
                    """,
                    (
                        schema_id,
                        corpus_id,
                        name,
                        json.dumps(schema_def, sort_keys=True),
                        is_active,
                    ),
                )
            conn.commit()
        return schema_id

    def list_schemas(self, *, corpus_id: str) -> list[SchemaRecord]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, corpus_id, name, schema_def::text, is_active, created_at
                    FROM core_schemas
                    WHERE corpus_id = %s
                    ORDER BY created_at DESC, name ASC
                    """,
                    (corpus_id,),
                )
                rows = cur.fetchall()
        return [self._row_to_schema_record(row) for row in rows]

    def get_schema_by_name(self, *, corpus_id: str, name: str) -> SchemaRecord | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, corpus_id, name, schema_def::text, is_active, created_at
                    FROM core_schemas
                    WHERE corpus_id = %s AND name = %s
                    LIMIT 1
                    """,
                    (corpus_id, name),
                )
                row = cur.fetchone()
        return self._row_to_schema_record(row) if row else None

    def get_active_schema(self, *, corpus_id: str) -> SchemaRecord | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, corpus_id, name, schema_def::text, is_active, created_at
                    FROM core_schemas
                    WHERE corpus_id = %s AND is_active = true
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (corpus_id,),
                )
                row = cur.fetchone()
        return self._row_to_schema_record(row) if row else None

    make_document_id = staticmethod(make_document_id)
    make_chunk_id = staticmethod(make_chunk_id)

    @staticmethod
    def _row_to_schema_record(row: tuple[Any, ...]) -> SchemaRecord:
        return SchemaRecord(
            id=str(row[0]),
            corpus_id=str(row[1]),
            name=str(row[2]),
            schema_def=json.loads(str(row[3])),
            is_active=bool(row[4]),
            created_at=str(row[5]),
        )

    @staticmethod
    def _row_to_chunk_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "document_id": str(row[1]),
            "text": str(row[2]),
            "position": int(row[3]),
            "start_char": int(row[4]) if row[4] is not None else None,
            "end_char": int(row[5]) if row[5] is not None else None,
            "chunk_type": str(row[6]) if row[6] is not None else None,
            "metadata": json.loads(str(row[7])) if row[7] is not None else {},
            "source": str(row[8]),
            "status": str(row[9]),
            "effective_start_date": str(row[10]) if row[10] is not None else None,
            "effective_end_date": str(row[11]) if row[11] is not None else None,
            "supersedes_chunk_id": str(row[12]) if row[12] is not None else None,
            "superseded_by_chunk_id": str(row[13]) if row[13] is not None else None,
        }

    @staticmethod
    def _row_to_search_hit(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "doc_id": str(row[0]),
            "relative_path": str(row[1]),
            "absolute_path": str(row[2]),
            "chunk_id": str(row[3]),
            "position": int(row[4]),
            "text": str(row[5]),
            "chunk_type": str(row[6]) if row[6] is not None else None,
            "metadata": json.loads(str(row[7])) if row[7] is not None else {},
            "score": float(row[8]),
        }

    @staticmethod
    def _row_to_proposal_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "batch_id": str(row[1]),
            "instruction_index": int(row[2]),
            "instruction_text": str(row[3]),
            "old_chunk_id": str(row[4]) if row[4] is not None else None,
            "old_chunk_snapshot": json.loads(str(row[5])),
            "new_chunk_draft": json.loads(str(row[6])),
            "match_confidence": float(row[7]) if row[7] is not None else None,
            "match_rationale": str(row[8]) if row[8] is not None else None,
            "date_rationale": str(row[9]) if row[9] is not None else None,
            "status": str(row[10]),
            "applied_new_chunk_id": str(row[11]) if row[11] is not None else None,
            "decided_by": str(row[12]) if row[12] is not None else None,
            "decided_at": str(row[13]) if row[13] is not None else None,
            "created_at": str(row[14]),
            "updated_at": str(row[15]),
        }

    def store_chunk_embeddings(
        self,
        *,
        corpus_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
    ) -> int:
        if not chunk_embeddings:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO core_chunk_embeddings (chunk_id, corpus_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        corpus_id = excluded.corpus_id,
                        embedding = excluded.embedding
                    """,
                    [
                        (chunk_id, corpus_id, _vector_literal(embedding))
                        for chunk_id, embedding in chunk_embeddings
                    ],
                )
            conn.commit()
        return len(chunk_embeddings)

    def search_chunks_semantic(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.id AS chunk_id,
                c.position,
                c.text,
                c.chunk_type,
                c.metadata::text,
                1 - (ce.embedding <=> %s::vector) AS score
            FROM core_chunk_embeddings ce
            JOIN core_chunks c ON c.id = ce.chunk_id
            JOIN core_documents d ON d.id = c.document_id
            WHERE ce.corpus_id = %s
              AND d.is_deleted = false
              AND c.status = 'active'
            ORDER BY ce.embedding <=> %s::vector ASC
            LIMIT %s
        """
        vector_literal = _vector_literal(query_embedding)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_literal, corpus_id, vector_literal, limit))
                rows = cur.fetchall()
        return [self._row_to_search_hit(row) for row in rows]

    def search_chunks_trigram(
        self,
        *,
        corpus_id: str,
        query_text: str,
        limit: int = 10,
        active_only: bool = True,
        similarity_threshold: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Fuzzy (pg_trgm) search over chunk text.

        Trigram similarity is character-n-gram based, so it works on Turkish
        text (ı/ğ/ü/ş/ö/ç) without a language-specific tokenizer — unlike
        `search_chunks`'s keyword matching. Used by the amendment candidate
        finder to locate chunks whose wording resembles an amendment
        instruction even when article numbering/headings don't line up.

        Uses `set_limit()` (not the `pg_trgm.similarity_threshold` session
        GUC's default) so the threshold is explicit and still index-backed
        via the `%` operator.
        """
        status_clause = "AND c.status = 'active'" if active_only else ""
        sql = f"""
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.id AS chunk_id,
                c.position,
                c.text,
                c.chunk_type,
                c.metadata::text,
                similarity(c.text, %s) AS score
            FROM core_chunks c
            JOIN core_documents d ON d.id = c.document_id
            WHERE d.corpus_id = %s
              AND d.is_deleted = false
              {status_clause}
              AND c.text %% %s
            ORDER BY score DESC
            LIMIT %s
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_limit(%s::real)", (similarity_threshold,))
                cur.execute(sql, (query_text, corpus_id, query_text, limit))
                rows = cur.fetchall()
        return [self._row_to_search_hit(row) for row in rows]

    def search_chunks_by_heading_trigram(
        self,
        *,
        corpus_id: str,
        heading_query: str,
        limit: int = 10,
        active_only: bool = True,
        similarity_threshold: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Fuzzy (pg_trgm) search over each chunk's heading_path, joined into
        a single string via `core_chunk_heading_path_text`. Complements —
        does not replace — exact heading_path matching, which is unreliable
        for inconsistently-formatted or OCR'd source documents."""
        status_clause = "AND c.status = 'active'" if active_only else ""
        sql = f"""
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.id AS chunk_id,
                c.position,
                c.text,
                c.chunk_type,
                c.metadata::text,
                similarity(core_chunk_heading_path_text(c.metadata), %s) AS score
            FROM core_chunks c
            JOIN core_documents d ON d.id = c.document_id
            WHERE d.corpus_id = %s
              AND d.is_deleted = false
              {status_clause}
              AND core_chunk_heading_path_text(c.metadata) %% %s
            ORDER BY score DESC
            LIMIT %s
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_limit(%s::real)", (similarity_threshold,))
                cur.execute(sql, (heading_query, corpus_id, heading_query, limit))
                rows = cur.fetchall()
        return [self._row_to_search_hit(row) for row in rows]

    def search_chunks_by_structured_metadata(
        self,
        *,
        corpus_id: str,
        article_no: str | None,
        document_number: str | None,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Exact-match candidates on short structured locators (article
        number / document number) — as opposed to the fuzzy trigram searches
        above, which exist for noisy/OCR'd text instead. Returns nothing if
        neither argument is given."""
        if not article_no and not document_number:
            return []
        status_clause = "AND c.status = 'active'" if active_only else ""
        conditions: list[str] = []
        params: list[Any] = [corpus_id]
        if article_no:
            conditions.append("c.metadata->>'article_no' = %s")
            params.append(article_no)
        if document_number:
            conditions.append("c.metadata->>'document_number' = %s")
            params.append(document_number)
        match_clause = " OR ".join(conditions)
        sql = f"""
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.id AS chunk_id,
                c.position,
                c.text,
                c.chunk_type,
                c.metadata::text,
                1.0 AS score
            FROM core_chunks c
            JOIN core_documents d ON d.id = c.document_id
            WHERE d.corpus_id = %s
              AND d.is_deleted = false
              {status_clause}
              AND ({match_clause})
            ORDER BY c.position ASC
            LIMIT %s
        """
        params.append(limit)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)  # ty: ignore[invalid-argument-type]
                rows = cur.fetchall()
        return [self._row_to_search_hit(row) for row in rows]

    def get_metadata_field_values(
        self,
        *,
        corpus_id: str,
        field_names: list[str],
        max_distinct: int = 10,
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for field in field_names:
                    cur.execute(
                        """
                        SELECT DISTINCT d.metadata_json->>%s AS val
                        FROM core_documents d
                        WHERE d.corpus_id = %s
                          AND d.is_deleted = false
                          AND d.metadata_json->>%s IS NOT NULL
                          AND d.metadata_json->>%s <> ''
                        LIMIT %s
                        """,
                        (field, corpus_id, field, field, max_distinct),
                    )
                    rows = cur.fetchall()
                    result[field] = [str(row[0]) for row in rows]
        return result

    def has_embeddings(self, *, corpus_id: str) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM core_chunk_embeddings WHERE corpus_id = %s",
                    (corpus_id,),
                )
                row = cur.fetchone()
        return bool(row and int(row[0]) > 0)

    def list_chunks_missing_embeddings(self, *, corpus_id: str) -> list[dict[str, Any]]:
        sql = """
            SELECT c.id, c.text
            FROM core_chunks c
            JOIN core_documents d ON d.id = c.document_id
            LEFT JOIN core_chunk_embeddings ce ON ce.chunk_id = c.id
            WHERE d.corpus_id = %s
              AND d.is_deleted = false
              AND ce.chunk_id IS NULL
            ORDER BY d.relative_path ASC, c.position ASC
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (corpus_id,))
                rows = cur.fetchall()
        return [{"id": str(row[0]), "text": str(row[1])} for row in rows]

    def create_amendment_batch(
        self, *, corpus_id: str, raw_text: str, created_by: str | None
    ) -> str:
        batch_id = f"batch_{uuid.uuid4().hex}"
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO core_amendment_batches (id, corpus_id, raw_text, created_by)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (batch_id, corpus_id, raw_text, created_by),
                )
            conn.commit()
        return batch_id

    def update_amendment_batch(
        self,
        *,
        batch_id: str,
        status: str,
        reference_date: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE core_amendment_batches
                    SET status = %s, reference_date = %s, error_message = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    (status, reference_date, error_message, batch_id),
                )
            conn.commit()

    def get_amendment_batch(self, *, batch_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, corpus_id, raw_text, reference_date::text, status,
                        error_message, created_by, created_at::text, updated_at::text
                    FROM core_amendment_batches
                    WHERE id = %s
                    """,
                    (batch_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "corpus_id": str(row[1]),
            "raw_text": str(row[2]),
            "reference_date": str(row[3]) if row[3] is not None else None,
            "status": str(row[4]),
            "error_message": str(row[5]) if row[5] is not None else None,
            "created_by": str(row[6]) if row[6] is not None else None,
            "created_at": str(row[7]),
            "updated_at": str(row[8]),
        }

    def create_amendment_proposals(
        self, *, batch_id: str, proposals: list[dict[str, Any]]
    ) -> list[str]:
        """Bulk-insert proposals for a batch. Each dict must have:
        instruction_index, instruction_text, old_chunk_id (optional),
        old_chunk_snapshot, new_chunk_draft, match_confidence (optional),
        match_rationale (optional), date_rationale (optional). Returns the
        generated proposal ids in the same order as `proposals`."""
        if not proposals:
            return []
        proposal_ids = [f"proposal_{uuid.uuid4().hex}" for _ in proposals]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO core_amendment_proposals (
                        id, batch_id, instruction_index, instruction_text,
                        old_chunk_id, old_chunk_snapshot, new_chunk_draft,
                        match_confidence, match_rationale, date_rationale
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                    """,
                    [
                        (
                            proposal_id,
                            batch_id,
                            proposal["instruction_index"],
                            proposal["instruction_text"],
                            proposal.get("old_chunk_id"),
                            json.dumps(proposal["old_chunk_snapshot"]),
                            json.dumps(proposal["new_chunk_draft"]),
                            proposal.get("match_confidence"),
                            proposal.get("match_rationale"),
                            proposal.get("date_rationale"),
                        )
                        for proposal_id, proposal in zip(proposal_ids, proposals)
                    ],
                )
            conn.commit()
        return proposal_ids

    _PROPOSAL_COLUMNS = """
        p.id, p.batch_id, p.instruction_index, p.instruction_text, p.old_chunk_id,
        p.old_chunk_snapshot::text, p.new_chunk_draft::text, p.match_confidence,
        p.match_rationale, p.date_rationale, p.status, p.applied_new_chunk_id,
        p.decided_by, p.decided_at::text, p.created_at::text, p.updated_at::text
    """

    def list_amendment_proposals(
        self,
        *,
        status: str | None = None,
        batch_id: str | None = None,
        corpus_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("p.status = %s")
            params.append(status)
        if batch_id:
            conditions.append("p.batch_id = %s")
            params.append(batch_id)
        if corpus_id:
            conditions.append("b.corpus_id = %s")
            params.append(corpus_id)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT {self._PROPOSAL_COLUMNS}
            FROM core_amendment_proposals p
            JOIN core_amendment_batches b ON b.id = p.batch_id
            {where_clause}
            ORDER BY p.created_at DESC
            LIMIT %s
        """
        params.append(limit)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)  # ty: ignore[invalid-argument-type]
                rows = cur.fetchall()
        return [self._row_to_proposal_dict(row) for row in rows]

    def get_amendment_proposal(self, *, proposal_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {self._PROPOSAL_COLUMNS}
                    FROM core_amendment_proposals p
                    WHERE p.id = %s
                    """,  # ty: ignore[invalid-argument-type]
                    (proposal_id,),
                )
                row = cur.fetchone()
        return self._row_to_proposal_dict(row) if row else None

    def approve_amendment_proposal(
        self, *, proposal_id: str, decided_by: str | None
    ) -> dict[str, Any]:
        """Apply one pending proposal in a single transaction:

        1. Lock + verify the proposal is still 'pending' (idempotency fast
           path — a retried/duplicate approval call is a no-op past here).
        2. Insert the new chunk (source='amendment'), id derived from the
           proposal id (`make_amendment_chunk_id`) so a retried insert after
           a partial failure is also idempotent at the DB level.
        3. If old_chunk_id is set, mark it superseded — guarded by
           `status = 'active'` so two proposals racing to supersede the same
           chunk can't both succeed silently. The new chunk's
           `effective_start_date` doubles as the old chunk's supersession
           boundary (when the new text starts applying is when the old text
           stops).
        4. Mark the proposal approved and record which chunk it produced.

        Raises ValueError on any conflict (not found, already decided, old
        chunk already superseded) so callers can surface a clear per-proposal
        failure reason instead of a generic error.
        """
        new_chunk_id = make_amendment_chunk_id(proposal_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, old_chunk_id, new_chunk_draft::text
                    FROM core_amendment_proposals WHERE id = %s FOR UPDATE
                    """,
                    (proposal_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Amendment proposal not found: {proposal_id}")
                current_status, old_chunk_id, draft_text = row
                if current_status != "pending":
                    raise ValueError(
                        f"Amendment proposal {proposal_id} is already {current_status}."
                    )

                draft = json.loads(str(draft_text))
                cur.execute(
                    """
                    INSERT INTO core_chunks (
                        id, document_id, text, position, start_char, end_char,
                        chunk_type, metadata, source, status,
                        effective_start_date, effective_end_date, supersedes_chunk_id
                    )
                    VALUES (
                        %s, %s, %s, %s, NULL, NULL, %s, %s::jsonb,
                        'amendment', 'active', %s, %s, %s
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        new_chunk_id,
                        draft["document_id"],
                        draft["text"],
                        draft.get("position") or 0,
                        draft.get("chunk_type"),
                        json.dumps(draft.get("metadata") or {}),
                        draft.get("effective_start_date"),
                        draft.get("effective_end_date"),
                        old_chunk_id,
                    ),
                )

                if old_chunk_id:
                    cur.execute(
                        """
                        UPDATE core_chunks
                        SET status = 'superseded',
                            superseded_by_chunk_id = %s,
                            effective_end_date = COALESCE(%s, effective_end_date, CURRENT_DATE)
                        WHERE id = %s AND status = 'active'
                        """,
                        (new_chunk_id, draft.get("effective_start_date"), old_chunk_id),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Old chunk {old_chunk_id} is no longer active "
                            "(already superseded/expired by another approval)."
                        )

                cur.execute(
                    """
                    UPDATE core_amendment_proposals
                    SET status = 'approved', applied_new_chunk_id = %s,
                        decided_by = %s, decided_at = now(), updated_at = now()
                    WHERE id = %s
                    """,
                    (new_chunk_id, decided_by, proposal_id),
                )

                cur.execute(
                    """
                    SELECT
                        id, document_id, text, position, start_char, end_char,
                        chunk_type, metadata::text, source, status,
                        effective_start_date::text, effective_end_date::text,
                        supersedes_chunk_id, superseded_by_chunk_id
                    FROM core_chunks WHERE id = %s
                    """,
                    (new_chunk_id,),
                )
                applied_row = cur.fetchone()
            conn.commit()
        if applied_row is None:
            raise ValueError(
                f"Failed to load applied chunk {new_chunk_id} after insert."
            )
        return self._row_to_chunk_dict(applied_row)

    def reject_amendment_proposal(
        self, *, proposal_id: str, decided_by: str | None
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE core_amendment_proposals
                    SET status = 'rejected', decided_by = %s, decided_at = now(), updated_at = now()
                    WHERE id = %s AND status = 'pending'
                    """,
                    (decided_by, proposal_id),
                )
                if cur.rowcount == 0:
                    raise ValueError(
                        f"Amendment proposal {proposal_id} is not pending."
                    )
            conn.commit()

    def delete_amendment_proposal(self, *, proposal_id: str) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM core_amendment_proposals WHERE id = %s AND status <> 'approved'",
                    (proposal_id,),
                )
                if cur.rowcount == 0:
                    raise ValueError(
                        f"Amendment proposal {proposal_id} not found or already approved "
                        "(approved proposals cannot be deleted, only their audit trail persists)."
                    )
            conn.commit()

    @staticmethod
    def _metadata_clause(
        *,
        field: str,
        operator: str,
        value: Any,
    ) -> tuple[str, list[Any]]:
        json_expr = "(d.metadata_json->>%s)"
        cast_expr = f"try_cast_double({json_expr})"

        if operator in {"eq", "ne"}:
            comparator = "=" if operator == "eq" else "<>"
            if isinstance(value, bool):
                return (
                    f"lower(coalesce({json_expr}, '')) {comparator} %s",
                    [field, "true" if value else "false"],
                )
            if isinstance(value, (int, float)):
                return (
                    f"{cast_expr} {comparator} %s",
                    [field, float(value)],
                )
            return (
                f"lower(coalesce({json_expr}, '')) {comparator} lower(%s)",
                [field, str(value)],
            )

        if operator in {"gt", "gte", "lt", "lte"}:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Metadata operator {operator!r} requires numeric value for field {field!r}."
                )
            comparator_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            return (
                f"{cast_expr} {comparator_map[operator]} %s",
                [field, float(value)],
            )

        if operator == "contains":
            return (
                f"lower(coalesce({json_expr}, '')) LIKE '%%' || lower(%s) || '%%'",
                [field, str(value)],
            )

        if operator == "in":
            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"Metadata `in` filter for field {field!r} has no values."
                )

            if all(isinstance(item, bool) for item in value):
                placeholders = ", ".join(["%s"] * len(value))
                return (
                    f"lower(coalesce({json_expr}, '')) IN ({placeholders})",
                    [field, *["true" if bool(item) else "false" for item in value]],
                )

            if all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                placeholders = ", ".join(["%s"] * len(value))
                return (
                    f"{cast_expr} IN ({placeholders})",
                    [field, *[float(item) for item in value]],
                )

            placeholders = ", ".join(["%s"] * len(value))
            return (
                f"lower(coalesce({json_expr}, '')) IN ({placeholders})",
                [field, *[str(item).lower() for item in value]],
            )

        raise ValueError(f"Unsupported metadata operator: {operator!r}")
