"""Preserve independent retrieval leads before bounded relevance assessment."""

from collections.abc import Sequence
from typing import TypedDict

from onyx.context.search.models import InferenceChunk


def select_lane_candidates(
    fused: Sequence[InferenceChunk],
    lanes: Sequence[Sequence[InferenceChunk]],
    *,
    limit: int,
    heads_per_lane: int = 5,
    diversity_candidates: Sequence[InferenceChunk] | None = None,
) -> list[InferenceChunk]:
    """Retain strong lane leads and optionally reserve room for distinct sources."""
    selected: list[InferenceChunk] = []
    seen: set[str] = set()
    selection_limit = limit if diversity_candidates is None else (limit + 1) // 2

    def add(chunk: InferenceChunk) -> None:
        if len(selected) < selection_limit and chunk.unique_id not in seen:
            selected.append(chunk)
            seen.add(chunk.unique_id)

    if fused:
        add(fused[0])
    for rank in range(heads_per_lane):
        for lane in lanes:
            if rank < len(lane):
                add(lane[rank])
    for chunk in fused:
        add(chunk)
    if diversity_candidates is not None:
        selection_limit = limit
        represented_sources = {chunk.document_id for chunk in selected}
        for chunk in diversity_candidates:
            if chunk.document_id not in represented_sources:
                add(chunk)
                represented_sources.add(chunk.document_id)
        for chunk in fused:
            add(chunk)
        for chunk in diversity_candidates:
            add(chunk)
    return selected


class CandidateLineage(TypedDict):
    candidate_id: str
    regulatory_chunk_id: str | None
    document_id: str
    lane_ranks: dict[str, int]
    fused_rank: int | None
    submitted_rank: int | None
    disposition: str


def candidate_lineage(
    fused: Sequence[InferenceChunk],
    lanes: Sequence[Sequence[InferenceChunk]],
    submitted: Sequence[InferenceChunk],
) -> list[CandidateLineage]:
    """Record ranks and selection without duplicating original source text."""
    rows: dict[str, CandidateLineage] = {}
    for lane_index, lane in enumerate(lanes):
        for rank, chunk in enumerate(lane, 1):
            row = rows.setdefault(
                chunk.unique_id,
                {
                    "candidate_id": chunk.unique_id,
                    "regulatory_chunk_id": chunk.regulatory_chunk_id,
                    "document_id": chunk.document_id,
                    "lane_ranks": {},
                    "fused_rank": None,
                    "submitted_rank": None,
                    "disposition": "pre_rerank_limit",
                },
            )
            row["lane_ranks"][str(lane_index)] = rank
    for rank, chunk in enumerate(fused, 1):
        row = rows.setdefault(
            chunk.unique_id,
            {
                "candidate_id": chunk.unique_id,
                "regulatory_chunk_id": chunk.regulatory_chunk_id,
                "document_id": chunk.document_id,
                "lane_ranks": {},
                "fused_rank": None,
                "submitted_rank": None,
                "disposition": "pre_rerank_limit",
            },
        )
        row["fused_rank"] = rank
    for rank, chunk in enumerate(submitted, 1):
        row = rows.setdefault(
            chunk.unique_id,
            {
                "candidate_id": chunk.unique_id,
                "regulatory_chunk_id": chunk.regulatory_chunk_id,
                "document_id": chunk.document_id,
                "lane_ranks": {},
                "fused_rank": None,
                "submitted_rank": None,
                "disposition": "submitted",
            },
        )
        row["submitted_rank"] = rank
        row["disposition"] = "submitted"
    return list(rows.values())
