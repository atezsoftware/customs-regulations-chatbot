"""Hybrid candidate-chunk lookup for the amendment pipeline.

Combines semantic (pgvector), fuzzy text/heading (pg_trgm), and structured
(article number) search, since heading_path alone is unreliable — it's a
best-effort reconstruction from document formatting (see
`RegulatoryChunker`) — and exact wording rarely survives a paraphrase in a
gazette amendment instruction.
"""

from __future__ import annotations

import re
from typing import Any
from dataclasses import replace

from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.storage import StorageBackend

from ..rerank import get_reranker
from .models import AmendmentInstruction
from .ranker import CandidateChunk, rank_candidates

_ARTICLE_NUMBER_RE = re.compile(r"(\d+)")


def _article_no_from_reference(article_reference: str | None) -> str | None:
    if not article_reference:
        return None
    match = _ARTICLE_NUMBER_RE.search(article_reference)
    return match.group(1) if match else None


def find_candidates(
    storage: StorageBackend,
    embedding_provider: EmbeddingProvider | None,
    *,
    corpus_id: str,
    instruction: AmendmentInstruction,
    limit: int = 5,
) -> list[CandidateChunk]:
    """Merge semantic + trigram(text/heading) + structured search results
    for one amendment instruction into a ranked candidate list."""
    fetch_limit = limit * 3
    merged: dict[str, CandidateChunk] = {}

    def _get_or_create(row: dict[str, Any]) -> CandidateChunk:
        chunk_id = str(row["chunk_id"])
        existing = merged.get(chunk_id)
        if existing is not None:
            return existing
        created = CandidateChunk(
            chunk_id=chunk_id,
            doc_id=str(row["doc_id"]),
            relative_path=str(row["relative_path"]),
            text=str(row["text"]),
            metadata=row.get("metadata") or {},
        )
        merged[chunk_id] = created
        return created

    def _upsert_score(row: dict[str, Any], field_name: str) -> None:
        current = _get_or_create(row)
        new_score = max(getattr(current, field_name), float(row["score"]))
        merged[current.chunk_id] = replace(current, **{field_name: new_score})

    def _upsert_structured(row: dict[str, Any]) -> None:
        current = _get_or_create(row)
        merged[current.chunk_id] = replace(current, structured_match=True)

    query_text = instruction.instruction_text

    if embedding_provider is not None and storage.has_embeddings(corpus_id=corpus_id):
        query_embedding = embedding_provider.embed_query(query_text)
        for row in storage.search_chunks_semantic(
            corpus_id=corpus_id, query_embedding=query_embedding, limit=fetch_limit
        ):
            _upsert_score(row, "semantic_score")

    for row in storage.search_chunks_trigram(
        corpus_id=corpus_id, query_text=query_text, limit=fetch_limit
    ):
        _upsert_score(row, "text_trgm_score")

    if instruction.article_reference:
        for row in storage.search_chunks_by_heading_trigram(
            corpus_id=corpus_id,
            heading_query=instruction.article_reference,
            limit=fetch_limit,
        ):
            _upsert_score(row, "heading_trgm_score")

        article_no = _article_no_from_reference(instruction.article_reference)
        if article_no:
            for row in storage.search_chunks_by_structured_metadata(
                corpus_id=corpus_id,
                article_no=article_no,
                document_number=None,
                limit=fetch_limit,
            ):
                _upsert_structured(row)

    return _rank(query_text, list(merged.values()), limit=limit)


def _rank(
    query_text: str, candidates: list[CandidateChunk], *, limit: int
) -> list[CandidateChunk]:
    """Cross-encoder rerank when available, falling back to the linear
    hybrid-score heuristic in `rank_candidates` on any failure.

    This is the highest-value rerank target in the codebase: `matcher.py`'s
    `confirm_match` sends every returned candidate's *full, untruncated*
    chunk text to the LLM, so narrowing `limit` candidates down from a
    higher-precision cross-encoder pass (instead of the linear heuristic
    alone) directly cuts full-chunk tokens sent per amendment instruction."""
    reranker = get_reranker()
    if reranker is not None and len(candidates) > 1:
        pairs = reranker.rerank(
            query_text, [candidate.text for candidate in candidates], top_n=limit
        )
        if pairs is not None:
            return [candidates[i] for i, _score in pairs][:limit]
    return rank_candidates(candidates, limit=limit)
