"""Hybrid candidate-chunk lookup for the amendment pipeline.

Combines fuzzy (pg_trgm) text/heading search with structured (article
number) search, since heading_path alone is unreliable — it's a best-effort
reconstruction from document formatting (see RegulatoryChunker) — and exact
wording rarely survives a paraphrase in an amendment instruction.

Deliberately scoped to a fixed set of `user_file_id`s (the directory being
amended) rather than running unscoped: an amendment must never touch a chunk
outside the directory the admin pasted the text against.
"""

import json
import re
from dataclasses import replace
from typing import Any, cast
from uuid import UUID

from sqlalchemy import bindparam
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import (
    CandidateChunk,
    rank_candidates,
    with_score,
)

_ARTICLE_NUMBER_RE = re.compile(r"(\d+)")

_TEXT_TRGM_SQL = sa_text(
    """
    SELECT rc.id, rc.user_file_id, rc.text, rc.chunk_metadata,
           uf.name AS document_name,
           word_similarity(uf.name, :query_text) AS document_name_score,
           similarity(rc.text, :query_text) AS score
    FROM regulatory_chunk AS rc
    JOIN user_file AS uf ON uf.id = rc.user_file_id
    WHERE rc.status = 'active'
      AND rc.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND rc.user_file_id IN :user_file_ids
      AND rc.text % :query_text
    ORDER BY score DESC
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_HEADING_TRGM_SQL = sa_text(
    """
    SELECT rc.id, rc.user_file_id, rc.text, rc.chunk_metadata,
           uf.name AS document_name,
           word_similarity(uf.name, :document_query_text) AS document_name_score,
           similarity(regulatory_chunk_heading_path_text(rc.heading_path), :query_text) AS score
    FROM regulatory_chunk AS rc
    JOIN user_file AS uf ON uf.id = rc.user_file_id
    WHERE rc.status = 'active'
      AND rc.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND rc.user_file_id IN :user_file_ids
      AND regulatory_chunk_heading_path_text(rc.heading_path) % :query_text
    ORDER BY score DESC
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_STRUCTURED_SQL = sa_text(
    """
    SELECT rc.id, rc.user_file_id, rc.text, rc.chunk_metadata,
           uf.name AS document_name,
           word_similarity(uf.name, :query_text) AS document_name_score,
           similarity(rc.text, :query_text) AS text_score
    FROM regulatory_chunk AS rc
    JOIN user_file AS uf ON uf.id = rc.user_file_id
    WHERE rc.status = 'active'
      AND rc.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND rc.user_file_id IN :user_file_ids
      AND rc.chunk_metadata->>'article_no' = :article_no
    ORDER BY document_name_score DESC, text_score DESC, rc.id
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))


def _article_no_from_reference(article_reference: str | None) -> str | None:
    if not article_reference:
        return None
    match = _ARTICLE_NUMBER_RE.search(article_reference)
    return match.group(1) if match else None


def find_candidates(
    db_session: Session,
    *,
    user_file_ids: list[UUID],
    instruction: AmendmentInstruction,
    limit: int = 5,
    similarity_threshold: float = 0.15,
) -> list[CandidateChunk]:
    """Merge trigram(text/heading) + structured search results for one
    amendment instruction into a ranked candidate list."""
    if not user_file_ids:
        return []

    fetch_limit = limit * 3
    query_text = instruction.instruction_text
    merged: dict[str, CandidateChunk] = {}

    def _get_or_create(row: object) -> CandidateChunk:
        typed_row = cast(Any, row)
        chunk_id = str(typed_row.id)
        existing = merged.get(chunk_id)
        if existing is not None:
            return existing
        metadata = typed_row.chunk_metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        created = CandidateChunk(
            chunk_id=chunk_id,
            user_file_id=str(typed_row.user_file_id),
            text=str(typed_row.text),
            metadata=metadata or {},
            document_name=(
                str(typed_row.document_name)
                if getattr(typed_row, "document_name", None)
                else None
            ),
            document_name_score=float(
                getattr(typed_row, "document_name_score", 0.0) or 0.0
            ),
        )
        merged[chunk_id] = created
        return created

    db_session.execute(sa_text("SELECT set_limit(:t)"), {"t": similarity_threshold})

    for row in db_session.execute(
        _TEXT_TRGM_SQL,
        {
            "query_text": query_text,
            "user_file_ids": user_file_ids,
            "limit": fetch_limit,
        },
    ).all():
        candidate = _get_or_create(row)
        merged[candidate.chunk_id] = with_score(
            candidate, "text_trgm_score", float(row.score)
        )

    if instruction.article_reference:
        for row in db_session.execute(
            _HEADING_TRGM_SQL,
            {
                "query_text": instruction.article_reference,
                "document_query_text": query_text,
                "user_file_ids": user_file_ids,
                "limit": fetch_limit,
            },
        ).all():
            candidate = _get_or_create(row)
            merged[candidate.chunk_id] = with_score(
                candidate, "heading_trgm_score", float(row.score)
            )

        article_no = _article_no_from_reference(instruction.article_reference)
        if article_no:
            for row in db_session.execute(
                _STRUCTURED_SQL,
                {
                    "article_no": article_no,
                    "query_text": query_text,
                    "user_file_ids": user_file_ids,
                    "limit": fetch_limit,
                },
            ).all():
                candidate = _get_or_create(row)
                candidate = with_score(
                    candidate,
                    "document_name_score",
                    float(getattr(row, "document_name_score", 0.0) or 0.0),
                )
                candidate = with_score(
                    candidate,
                    "text_trgm_score",
                    float(getattr(row, "text_score", 0.0) or 0.0),
                )
                merged[candidate.chunk_id] = replace(candidate, structured_match=True)

    return rank_candidates(list(merged.values()), limit=limit)
