"""
Indexed query helpers for agent tools.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import os

from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.storage import StorageBackend
from ..rerank import get_reranker
from .filters import MetadataFilter, parse_metadata_filters
from .ranker import RankedDocument, rank_documents

# Without these, a handful of near-duplicate/overlapping chunks from the
# same source document can crowd out genuinely different sources in the
# final top-K — reducing `limit` alone doesn't help if most of what
# survives is redundant. `_RERANK_MIN_RELEVANCE` only applies to real
# cross-encoder scores (0-1 range); the heuristic combined_score is a
# different scale entirely, so it's skipped there.
_MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("FS_EXPLORER_MAX_CHUNKS_PER_DOCUMENT", "2"))
_RERANK_MIN_RELEVANCE = float(os.getenv("FS_EXPLORER_RERANK_MIN_RELEVANCE", "0.05"))


@dataclass(frozen=True)
class SearchHit:
    """Ranked document hit from indexed retrieval."""

    doc_id: str
    relative_path: str
    absolute_path: str
    position: int | None
    text: str
    semantic_score: float
    metadata_score: int
    score: float
    matched_by: str
    chunk_id: str | None = None
    chunk_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IndexedQueryEngine:
    """Parallel retrieval engine for semantic + metadata query paths."""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.storage = storage
        self.embedding_provider = embedding_provider

    def search(
        self,
        *,
        corpus_id: str,
        query: str,
        filters: str | None = None,
        limit: int = 5,
        enable_semantic: bool = True,
        enable_metadata: bool = True,
        as_of_date: str | None = None,
        overfetch_multiplier: int = 4,
    ) -> list[SearchHit]:
        """`as_of_date` (YYYY-MM-DD) restricts chunk results to those whose
        validity interval covers that date — omit it to mean "today".

        `overfetch_multiplier` controls how many candidates are pulled from
        each retrieval path before ranking/reranking trims back down to
        `limit` — a larger pool gives the cross-encoder reranker (see
        `_rank`) more to work with, at the cost of a bigger rerank call."""
        normalized_limit = max(limit, 1)
        parsed_filters = self._parse_filters(corpus_id=corpus_id, filters=filters)
        semantic_limit = max(normalized_limit * overfetch_multiplier, normalized_limit)
        metadata_limit = max(normalized_limit * overfetch_multiplier, normalized_limit)

        run_semantic = enable_semantic
        run_metadata = enable_metadata and bool(parsed_filters)

        semantic_rows: list[dict[str, Any]]
        metadata_rows: list[dict[str, Any]]
        if run_semantic and run_metadata:
            semantic_rows, metadata_rows = self._search_parallel(
                corpus_id=corpus_id,
                query=query,
                metadata_filters=parsed_filters,
                semantic_limit=semantic_limit,
                metadata_limit=metadata_limit,
                as_of_date=as_of_date,
            )
        elif run_semantic:
            semantic_rows = self._semantic_query(
                corpus_id=corpus_id,
                query=query,
                limit=semantic_limit,
                as_of_date=as_of_date,
            )
            metadata_rows = []
        elif run_metadata:
            semantic_rows = []
            metadata_rows = self._metadata_query(
                corpus_id=corpus_id,
                metadata_filters=parsed_filters,
                limit=metadata_limit,
            )
        else:
            semantic_rows, metadata_rows = [], []

        merged_documents = self._merge(
            semantic_rows=semantic_rows, metadata_rows=metadata_rows
        )
        ranked = self.rank_candidates(
            query=query, documents=merged_documents, limit=normalized_limit
        )
        return [
            SearchHit(
                doc_id=doc.doc_id,
                relative_path=doc.relative_path,
                absolute_path=doc.absolute_path,
                position=doc.position,
                text=doc.text,
                semantic_score=doc.semantic_score,
                metadata_score=doc.metadata_score,
                score=score,
                matched_by=doc.matched_by,
                chunk_id=doc.chunk_id,
                chunk_type=doc.chunk_type,
                metadata=doc.metadata,
            )
            for doc, score in ranked
        ]

    def _parse_filters(
        self, *, corpus_id: str, filters: str | None
    ) -> list[MetadataFilter]:
        if filters is None or not filters.strip():
            return []
        allowed_fields = self._allowed_filter_fields(corpus_id=corpus_id)
        return parse_metadata_filters(filters, allowed_fields=allowed_fields)

    def _allowed_filter_fields(self, *, corpus_id: str) -> set[str] | None:
        active_schema = self.storage.get_active_schema(corpus_id=corpus_id)
        if active_schema is None:
            return None
        fields = active_schema.schema_def.get("fields")
        if not isinstance(fields, list):
            return None
        allowed: set[str] = set()
        for schema_field in fields:
            if isinstance(schema_field, dict):
                name = schema_field.get("name")
                if isinstance(name, str):
                    allowed.add(name)
        return allowed if allowed else None

    def _search_parallel(
        self,
        *,
        corpus_id: str,
        query: str,
        metadata_filters: list[MetadataFilter],
        semantic_limit: int,
        metadata_limit: int,
        as_of_date: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = executor.submit(
                self._semantic_query,
                corpus_id=corpus_id,
                query=query,
                limit=semantic_limit,
                as_of_date=as_of_date,
            )
            metadata_future = executor.submit(
                self._metadata_query,
                corpus_id=corpus_id,
                metadata_filters=metadata_filters,
                limit=metadata_limit,
            )
            semantic_rows = semantic_future.result()
            metadata_rows = metadata_future.result()
        return semantic_rows, metadata_rows

    def _semantic_query(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int,
        as_of_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.embedding_provider is not None and self.storage.has_embeddings(
            corpus_id=corpus_id
        ):
            query_embedding = self.embedding_provider.embed_query(query)
            return self.storage.search_chunks_semantic(
                corpus_id=corpus_id,
                query_embedding=query_embedding,
                limit=limit,
                as_of_date=as_of_date,
            )
        return self.storage.search_chunks(
            corpus_id=corpus_id, query=query, limit=limit, as_of_date=as_of_date
        )

    def _metadata_query(
        self,
        *,
        corpus_id: str,
        metadata_filters: list[MetadataFilter],
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.storage.search_documents_by_metadata(
            corpus_id=corpus_id,
            filters=[flt.to_storage_dict() for flt in metadata_filters],
            limit=limit,
        )

    def _merge(
        self,
        *,
        semantic_rows: list[dict[str, Any]],
        metadata_rows: list[dict[str, Any]],
    ) -> list[RankedDocument]:
        merged: dict[str, dict[str, Any]] = {}

        for row in semantic_rows:
            doc_id = str(row["doc_id"])
            chunk_id = str(row["chunk_id"]) if row.get("chunk_id") else None
            merge_key = f"chunk:{chunk_id}" if chunk_id else f"doc:{doc_id}"
            score = float(row["score"])
            position = int(row["position"])
            entry = merged.setdefault(
                merge_key,
                {
                    "doc_id": doc_id,
                    "relative_path": str(row["relative_path"]),
                    "absolute_path": str(row["absolute_path"]),
                    "position": position,
                    "text": str(row["text"]),
                    "semantic_score": 0.0,
                    "metadata_score": 0,
                    "chunk_id": chunk_id,
                    "chunk_type": row.get("chunk_type"),
                    "metadata": row.get("metadata") or {},
                },
            )
            if score > float(entry["semantic_score"]):
                entry["semantic_score"] = score
                entry["position"] = position
                entry["text"] = str(row["text"])
                entry["chunk_id"] = chunk_id
                entry["chunk_type"] = row.get("chunk_type")
                entry["metadata"] = row.get("metadata") or {}

        metadata_by_doc = {str(row["doc_id"]): row for row in metadata_rows}
        semantic_doc_ids = {
            str(entry["doc_id"])
            for entry in merged.values()
            if entry.get("chunk_id") is not None
        }

        # A metadata match is document-level, while the model needs actual
        # chunk evidence. Boost already-retrieved chunks from matching
        # documents, then hydrate metadata-only documents into full chunk
        # candidates before the cross-encoder sees them. Do not pre-rank or
        # truncate this candidate set: every retrieved chunk must be scored
        # by the cross-encoder.
        for entry in merged.values():
            metadata_row = metadata_by_doc.get(str(entry["doc_id"]))
            if metadata_row is not None:
                entry["metadata_score"] = max(
                    int(entry["metadata_score"]),
                    int(metadata_row.get("metadata_score", 1)),
                )

        for doc_id, row in metadata_by_doc.items():
            if doc_id in semantic_doc_ids:
                continue
            chunks = self.storage.list_document_chunks(doc_id=doc_id)
            for chunk in chunks:
                chunk_id = str(chunk["id"])
                merged[f"chunk:{chunk_id}"] = {
                    "doc_id": doc_id,
                    "relative_path": str(row["relative_path"]),
                    "absolute_path": str(row["absolute_path"]),
                    "position": int(chunk["position"]),
                    "text": str(chunk["text"]),
                    "semantic_score": 0.0,
                    "metadata_score": int(row.get("metadata_score", 1)),
                    "chunk_id": chunk_id,
                    "chunk_type": chunk.get("chunk_type"),
                    "metadata": chunk.get("metadata") or {},
                }

        documents = [
            RankedDocument(
                doc_id=str(entry["doc_id"]),
                relative_path=str(entry["relative_path"]),
                absolute_path=str(entry["absolute_path"]),
                position=int(entry["position"])
                if entry["position"] is not None
                else None,
                text=str(entry["text"]),
                semantic_score=float(entry["semantic_score"]),
                metadata_score=int(entry["metadata_score"]),
                chunk_id=str(entry["chunk_id"]) if entry.get("chunk_id") else None,
                chunk_type=str(entry["chunk_type"])
                if entry.get("chunk_type")
                else None,
                metadata=entry["metadata"]
                if isinstance(entry.get("metadata"), dict)
                else {},
            )
            for entry in merged.values()
        ]
        return documents

    @staticmethod
    def rank_candidates(
        *, query: str, documents: list[RankedDocument], limit: int
    ) -> list[tuple[RankedDocument, float]]:
        """Cross-encoder rerank when available, falling back to the linear
        semantic/metadata heuristic in `rank_documents` on any failure.

        Returns (document, score) pairs rather than bare documents so a
        caller merging several independently-ranked result sets (e.g.
        `semantic_search` fanning out across multiple linked corpora) can
        re-sort the union by a score that's consistent with whichever path
        produced each entry, instead of resorting by the heuristic score
        alone and silently undoing the rerank ordering."""
        reranker = get_reranker()
        if reranker is not None and documents:
            # Ask for the full ranked order (not just `limit`) so
            # `_diversify` has enough candidates to backfill from after
            # dropping near-duplicates/low-relevance ones.
            pairs = reranker.rerank(
                query, [doc.text for doc in documents], top_n=len(documents)
            )
            if pairs is not None:
                ranked = [(documents[i], score) for i, score in pairs]
                diversified = IndexedQueryEngine._diversify(
                    ranked, limit=limit, min_relevance=_RERANK_MIN_RELEVANCE
                )
                if diversified:
                    return diversified
        heuristic_ranked = [
            (doc, doc.combined_score)
            for doc in rank_documents(documents, limit=len(documents))
        ]
        return IndexedQueryEngine._diversify(
            heuristic_ranked, limit=limit, min_relevance=None
        )

    @staticmethod
    def _diversify(
        ranked: list[tuple[RankedDocument, float]],
        *,
        limit: int,
        min_relevance: float | None,
    ) -> list[tuple[RankedDocument, float]]:
        """Trim a fully-ranked candidate list down to `limit`, capping how
        many chunks from the same source document can appear (so a handful
        of overlapping/near-duplicate chunks from one document can't crowd
        out other sources) and dropping anything below `min_relevance` when
        set. Walks the full ranked order so a document hitting its cap or a
        below-threshold chunk is simply skipped in favor of the next
        distinct, qualifying one — not just truncated."""
        per_doc_count: dict[str, int] = {}
        result: list[tuple[RankedDocument, float]] = []
        for doc, score in ranked:
            if min_relevance is not None and score < min_relevance:
                continue
            if per_doc_count.get(doc.doc_id, 0) >= _MAX_CHUNKS_PER_DOCUMENT:
                continue
            per_doc_count[doc.doc_id] = per_doc_count.get(doc.doc_id, 0) + 1
            result.append((doc, score))
            if len(result) >= limit:
                break
        return result
