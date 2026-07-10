"""
Storage interfaces and data models for index persistence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


def stable_id(prefix: str, value: str) -> str:
    """Deterministic content-hash id, shared by every storage backend.

    Indexing is idempotent (re-indexing the same file must upsert the same
    logical row, not create a new one), which depends on doc/chunk ids being
    a pure function of (corpus, path) / (doc, position, offsets) rather than
    backend-assigned. Keep this scheme regardless of which backend stores it.
    """
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def make_document_id(corpus_id: str, relative_path: str) -> str:
    return stable_id("doc", f"{corpus_id}:{relative_path}")


def make_chunk_id(doc_id: str, position: int, start_char: int, end_char: int) -> str:
    return stable_id("chunk", f"{doc_id}:{position}:{start_char}:{end_char}")


def make_amendment_chunk_id(proposal_id: str) -> str:
    """Deterministic id for a chunk created by approving an amendment proposal.

    Unlike `make_chunk_id`, this is keyed by proposal id, not document
    position/offsets — amendment-created chunks have no real source
    offsets. It only needs to be stable across retries of approving the
    *same* proposal (so `insert_chunk`/`ON CONFLICT DO NOTHING` is
    idempotent), not across re-indexing.
    """
    return stable_id("chunk", f"amend:{proposal_id}")


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk stored for a document.

    `start_char`/`end_char` are `None` for amendment-created chunks (no real
    parse offsets exist), and required for indexer-produced chunks — enforced
    by `core_chunks_indexed_offsets_check` at the DB layer, not here.
    """

    id: str
    doc_id: str
    text: str
    position: int
    start_char: int | None
    end_char: int | None
    embedding: list[float] | None = None
    chunk_type: str | None = None
    metadata: dict[str, Any] | None = None
    source: str = "indexed"
    status: str = "active"
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    supersedes_chunk_id: str | None = None
    superseded_by_chunk_id: str | None = None


@dataclass(frozen=True)
class DocumentRecord:
    """A normalized document record for indexing."""

    id: str
    corpus_id: str
    relative_path: str
    absolute_path: str
    content: str
    metadata_json: str
    file_mtime: float
    file_size: int
    content_sha256: str


@dataclass(frozen=True)
class SchemaRecord:
    """A stored schema entry."""

    id: str
    corpus_id: str
    name: str
    schema_def: dict[str, Any]
    is_active: bool
    created_at: str


class StorageBackend(Protocol):
    """Protocol for persistence operations used by indexing and schema workflows."""

    def initialize(self) -> None:
        """Initialize required tables/indexes."""

    def get_or_create_corpus(self, root_path: str) -> str:
        """Return corpus id for a root path, creating if needed."""

    def get_corpus_id(self, root_path: str) -> str | None:
        """Return corpus id for a root path if present."""

    def upsert_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        """Insert or update a document and replace its chunks."""

    def insert_chunk(self, *, chunk: ChunkRecord) -> None:
        """Insert a single chunk without touching any other chunk of its
        document. Idempotent (ON CONFLICT DO NOTHING)."""

    def get_chunk(self, *, chunk_id: str) -> dict[str, Any] | None:
        """Fetch one chunk row (all columns) by id."""

    def supersede_chunk(
        self, *, old_chunk_id: str, new_chunk_id: str, effective_end_date: str | None
    ) -> bool:
        """Mark a chunk superseded by a specific successor. Returns False if
        the chunk was not 'active' (race/conflict, not a silent success)."""

    def expire_chunk(self, *, chunk_id: str, effective_end_date: str) -> None:
        """Mark a chunk expired (validity window closed with no direct
        successor)."""

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
        """Mark documents deleted when not present in the latest index run."""

    def list_documents(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List documents for a corpus."""

    def count_chunks(self, *, corpus_id: str) -> int:
        """Count chunks for active documents in a corpus."""

    def search_chunks(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks and return ranked matches."""

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search indexed documents by metadata filters."""

    def get_document(self, *, doc_id: str) -> dict[str, Any] | None:
        """Get a document by id."""

    def list_document_chunks(self, *, doc_id: str) -> list[dict[str, Any]]:
        """List chunks for a document ordered by their original position."""

    def get_document_chunks_by_prefix(
        self, *, corpus_root: str, relative_path_prefix: str
    ) -> dict[str, Any] | None:
        """Find the active document under `corpus_root` whose `relative_path`
        starts with `relative_path_prefix`, plus its chunks and embedding status."""

    def save_schema(
        self,
        *,
        corpus_id: str,
        name: str,
        schema_def: dict[str, Any],
        is_active: bool = True,
    ) -> str:
        """Create or update a schema entry."""

    def list_schemas(self, *, corpus_id: str) -> list[SchemaRecord]:
        """List all schemas for a corpus."""

    def get_schema_by_name(self, *, corpus_id: str, name: str) -> SchemaRecord | None:
        """Fetch a schema by name."""

    def get_active_schema(self, *, corpus_id: str) -> SchemaRecord | None:
        """Fetch active schema for a corpus if present."""

    def store_chunk_embeddings(
        self,
        *,
        corpus_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
    ) -> int:
        """Bulk-store (chunk_id, embedding) pairs. Return count written."""

    def search_chunks_semantic(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search chunks by cosine similarity against a query embedding."""

    def search_chunks_trigram(
        self,
        *,
        corpus_id: str,
        query_text: str,
        limit: int = 10,
        active_only: bool = True,
        similarity_threshold: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Fuzzy (pg_trgm) search over chunk text — Turkish-aware since
        trigram similarity is character-n-gram based, not tokenizer-based."""

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
        a single string. Complements (not replaces) exact heading_path
        matching, which is unreliable for inconsistently-formatted sources."""

    def search_chunks_by_structured_metadata(
        self,
        *,
        corpus_id: str,
        article_no: str | None,
        document_number: str | None,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Exact-match candidates on short structured locators. Empty if
        neither argument is given."""

    def create_amendment_batch(
        self, *, corpus_id: str, raw_text: str, created_by: str | None
    ) -> str:
        """Create an amendment analysis batch (the audit record for one
        pasted gazette text) and return its id."""

    def update_amendment_batch(
        self,
        *,
        batch_id: str,
        status: str,
        reference_date: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a batch's analysis status/reference date/error."""

    def get_amendment_batch(self, *, batch_id: str) -> dict[str, Any] | None:
        """Fetch one amendment batch by id."""

    def create_amendment_proposals(
        self, *, batch_id: str, proposals: list[dict[str, Any]]
    ) -> list[str]:
        """Bulk-insert proposals for a batch; returns generated proposal ids
        in the same order as the input."""

    def list_amendment_proposals(
        self,
        *,
        status: str | None = None,
        batch_id: str | None = None,
        corpus_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List proposals, optionally filtered by status, batch, and/or the
        corpus of the batch that produced them."""

    def get_amendment_proposal(self, *, proposal_id: str) -> dict[str, Any] | None:
        """Fetch one proposal by id."""

    def approve_amendment_proposal(
        self, *, proposal_id: str, decided_by: str | None
    ) -> dict[str, Any]:
        """Apply a pending proposal in one transaction: insert the new
        chunk, supersede the old one if any, mark the proposal approved.
        Returns the applied chunk. Raises ValueError on conflict (already
        decided, or the old chunk was already superseded by another
        approval)."""

    def reject_amendment_proposal(
        self, *, proposal_id: str, decided_by: str | None
    ) -> None:
        """Mark a pending proposal rejected. Raises ValueError if it isn't
        pending."""

    def delete_amendment_proposal(self, *, proposal_id: str) -> None:
        """Delete a proposal. Raises ValueError if it doesn't exist or is
        already approved (an applied chunk's audit trail must not be
        orphaned by deleting it)."""

    def get_metadata_field_values(
        self,
        *,
        corpus_id: str,
        field_names: list[str],
        max_distinct: int = 10,
    ) -> dict[str, list[str]]:
        """Return up to *max_distinct* distinct non-empty values per metadata field."""

    def has_embeddings(self, *, corpus_id: str) -> bool:
        """Return True if the corpus has stored embeddings."""

    def list_chunks_missing_embeddings(self, *, corpus_id: str) -> list[dict[str, Any]]:
        """List (id, text) for active chunks in a corpus that have no embedding yet."""
