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
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast
from uuid import UUID

from sqlalchemy import bindparam
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import (
    CandidateChunk,
    with_score,
)
from onyx.tools.tool_implementations.search.search_utils import (
    weighted_reciprocal_rank_fusion,
)

_ARTICLE_NUMBER_RE = re.compile(r"(\d+)")

_TEXT_TRGM_SQL = sa_text(
    """
    SELECT regulatory_chunk.id, regulatory_chunk.user_file_id,
           regulatory_chunk.text, regulatory_chunk.chunk_metadata,
           user_file.name AS source_name,
           similarity(regulatory_chunk.text, :query_text) AS score,
           similarity(
               lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
               :source_query
           ) AS source_score
    FROM regulatory_chunk
    JOIN user_file ON user_file.id = regulatory_chunk.user_file_id
    WHERE regulatory_chunk.status = 'active'
      AND regulatory_chunk.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND regulatory_chunk.user_file_id IN :user_file_ids
      AND regulatory_chunk.text % :query_text
    ORDER BY score DESC
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_HEADING_TRGM_SQL = sa_text(
    """
    SELECT regulatory_chunk.id, regulatory_chunk.user_file_id,
           regulatory_chunk.text, regulatory_chunk.chunk_metadata,
           user_file.name AS source_name,
           similarity(regulatory_chunk_heading_path_text(heading_path), :query_text) AS score,
           similarity(
               lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
               :source_query
           ) AS source_score
    FROM regulatory_chunk
    JOIN user_file ON user_file.id = regulatory_chunk.user_file_id
    WHERE regulatory_chunk.status = 'active'
      AND regulatory_chunk.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND regulatory_chunk.user_file_id IN :user_file_ids
      AND regulatory_chunk_heading_path_text(heading_path) % :query_text
    ORDER BY score DESC
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_STRUCTURED_SQL = sa_text(
    """
    SELECT regulatory_chunk.id, regulatory_chunk.user_file_id,
           regulatory_chunk.text, regulatory_chunk.chunk_metadata,
           user_file.name AS source_name,
           similarity(regulatory_chunk.text, :query_text) AS score,
           similarity(
               lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
               :source_query
           ) AS source_score
    FROM regulatory_chunk
    JOIN user_file ON user_file.id = regulatory_chunk.user_file_id
    WHERE regulatory_chunk.status = 'active'
      AND regulatory_chunk.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND regulatory_chunk.user_file_id IN :user_file_ids
      AND regulatory_chunk.chunk_metadata->>'article_no' = :article_no
    ORDER BY source_score DESC, score DESC, regulatory_chunk.position
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_SOURCE_FILES_SQL = sa_text(
    """
    SELECT user_file.id,
           similarity(
               lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
               :source_query
           ) AS source_score
    FROM user_file
    WHERE user_file.id IN :user_file_ids
      AND similarity(
          lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
          :source_query
      ) >= :minimum_source_score
    ORDER BY source_score DESC, user_file.id
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_SOURCE_SQL = sa_text(
    """
    SELECT regulatory_chunk.id, regulatory_chunk.user_file_id,
           regulatory_chunk.text, regulatory_chunk.chunk_metadata,
           user_file.name AS source_name,
           0.0 AS score,
           similarity(
               lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
               :source_query
           ) AS source_score
    FROM regulatory_chunk
    JOIN user_file ON user_file.id = regulatory_chunk.user_file_id
    WHERE regulatory_chunk.status = 'active'
      AND regulatory_chunk.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND regulatory_chunk.user_file_id IN :user_file_ids
      AND similarity(
          lower(regexp_replace(replace(replace(user_file.name, '_x1', ' '), 'x2', ' '), '[_/.-]+', ' ', 'g')),
          :source_query
      ) > 0.1
    ORDER BY source_score DESC, regulatory_chunk.position
    LIMIT :limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))

_QUOTED_OLD_VALUE_RE = re.compile(
    r"[\"“]([^\"”]{3,120})[\"”]\s+(?:ibaresi|ifadesi|kelimesi)", re.IGNORECASE
)


def _normalize_source_query(source: str | None) -> str:
    if not source:
        return ""
    return " ".join(re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", " ", source).lower().split())


def _lexical_query_lanes(instruction_text: str) -> list[str]:
    anchors = [
        match.group(1) for match in _QUOTED_OLD_VALUE_RE.finditer(instruction_text)
    ]
    return list(dict.fromkeys([instruction_text, *anchors]))[:3]


def fuse_candidate_lanes(
    lanes: list[tuple[list[CandidateChunk], float]], *, limit: int
) -> list[CandidateChunk]:
    """Apply the same bounded weighted-RRF strategy used by Atez Search V2."""

    non_empty = [(candidates, weight) for candidates, weight in lanes if candidates]
    if not non_empty:
        return []
    fused = weighted_reciprocal_rank_fusion(
        [candidates for candidates, _weight in non_empty],
        [weight for _candidates, weight in non_empty],
        lambda candidate: candidate.chunk_id,
    )
    return fused[: max(limit, 1)]


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
    source_scope_cache: dict[str, list[UUID]] | None = None,
) -> list[CandidateChunk]:
    """Merge trigram(text/heading) + structured search results for one
    amendment instruction into a ranked candidate list."""
    if not user_file_ids:
        return []

    fetch_limit = limit * 3
    query_text = instruction.instruction_text
    source_query = _normalize_source_query(instruction.target_source)
    merged: dict[str, CandidateChunk] = {}
    ranked_lane_ids: list[tuple[list[str], float]] = []

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
            source_name=str(getattr(typed_row, "source_name", "")),
            metadata=metadata or {},
        )
        merged[chunk_id] = created
        return created

    db_session.execute(sa_text("SELECT set_limit(:t)"), {"t": similarity_threshold})

    candidate_scope_ids = user_file_ids
    if source_query:
        source_file_ids = (
            source_scope_cache.get(source_query) if source_scope_cache else None
        )
        if source_file_ids is None:
            source_file_rows = db_session.execute(
                _SOURCE_FILES_SQL,
                {
                    "source_query": source_query,
                    "user_file_ids": user_file_ids,
                    "minimum_source_score": 0.15,
                    "limit": 12,
                },
            ).all()
            source_file_ids = [row.id for row in source_file_rows]
            if source_scope_cache is not None:
                source_scope_cache[source_query] = source_file_ids
        if source_file_ids:
            candidate_scope_ids = source_file_ids

    def _record_rows(
        rows: Sequence[object],
        *,
        weight: float,
        score_field: str | None = None,
        structured: bool = False,
    ) -> None:
        lane_ids: list[str] = []
        for row in rows:
            candidate = _get_or_create(row)
            if score_field is not None:
                candidate = with_score(
                    candidate, score_field, float(cast(Any, row).score)
                )
            source_score = float(getattr(cast(Any, row), "source_score", 0.0))
            candidate = with_score(candidate, "source_score", source_score)
            if structured:
                candidate = replace(candidate, structured_match=True)
            merged[candidate.chunk_id] = candidate
            lane_ids.append(candidate.chunk_id)
        if lane_ids:
            ranked_lane_ids.append((lane_ids, weight))

    for lexical_index, lexical_query in enumerate(_lexical_query_lanes(query_text)):
        rows = db_session.execute(
            _TEXT_TRGM_SQL,
            {
                "query_text": lexical_query,
                "source_query": source_query,
                "user_file_ids": candidate_scope_ids,
                "limit": fetch_limit,
            },
        ).all()
        _record_rows(
            rows,
            weight=0.7 if lexical_index == 0 else 1.2,
            score_field="text_trgm_score",
        )

    if instruction.article_reference:
        rows = db_session.execute(
            _HEADING_TRGM_SQL,
            {
                "query_text": instruction.article_reference,
                "source_query": source_query,
                "user_file_ids": candidate_scope_ids,
                "limit": fetch_limit,
            },
        ).all()
        _record_rows(rows, weight=0.6, score_field="heading_trgm_score")

        article_no = _article_no_from_reference(instruction.article_reference)
        if article_no:
            rows = db_session.execute(
                _STRUCTURED_SQL,
                {
                    "article_no": article_no,
                    "query_text": query_text,
                    "source_query": source_query,
                    "user_file_ids": candidate_scope_ids,
                    "limit": fetch_limit,
                },
            ).all()
            _record_rows(rows, weight=1.5, structured=True)
    if source_query:
        rows = db_session.execute(
            _SOURCE_SQL,
            {
                "source_query": source_query,
                "user_file_ids": candidate_scope_ids,
                "limit": fetch_limit,
            },
        ).all()
        _record_rows(rows, weight=1.2)

    lanes = [
        ([merged[chunk_id] for chunk_id in chunk_ids], weight)
        for chunk_ids, weight in ranked_lane_ids
    ]
    return fuse_candidate_lanes(lanes, limit=limit)
