from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

from onyx.context.search.models import InferenceChunk
from onyx.reranking.payload import canonical_chunk_source

RELEVANCE_COMPETITION_BAND = 0.05
NEAR_DUPLICATE_PENALTY = 0.50
UNSEEN_SOURCE_BONUS = 0.01
MISSING_SCORE_STEP = 0.000001
NEAR_DUPLICATE_SHINGLE_SIMILARITY = 0.82


def _normalize_body(chunk: InferenceChunk) -> str:
    normalized = unicodedata.normalize("NFKD", chunk.content or chunk.blurb).casefold()
    characters = (
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
        if not unicodedata.category(character).startswith("M")
    )
    return " ".join("".join(characters).split())


def _word_shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    words = text.split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


def _is_near_duplicate(first: str, second: str) -> bool:
    if not first or not second:
        return False
    if first == second:
        return True
    first_shingles = _word_shingles(first)
    second_shingles = _word_shingles(second)
    if not first_shingles or not second_shingles:
        return False
    intersection_size = len(first_shingles & second_shingles)
    union_size = len(first_shingles | second_shingles)
    return intersection_size / union_size >= NEAR_DUPLICATE_SHINGLE_SIMILARITY


def _raw_relevance_scores(
    chunks: Sequence[InferenceChunk],
    scores: Mapping[tuple[str, int], float],
) -> list[float]:
    if scores:
        missing_score_start = min(scores.values())
    else:
        missing_score_start = 0.0
    return [
        scores.get(
            (chunk.document_id, chunk.chunk_id),
            missing_score_start - (rank * MISSING_SCORE_STEP),
        )
        for rank, chunk in enumerate(chunks)
    ]


def apply_soft_diversity(
    *,
    chunks: Sequence[InferenceChunk],
    scores: Mapping[tuple[str, int], float],
    limit: int,
) -> list[InferenceChunk]:
    """Keep relevance primary while softly suppressing textual repetition."""

    if limit <= 0 or not chunks:
        return []

    normalized_bodies = [_normalize_body(chunk) for chunk in chunks]
    canonical_sources = [canonical_chunk_source(chunk) for chunk in chunks]
    raw_relevance = _raw_relevance_scores(chunks, scores)
    remaining_indices = list(range(len(chunks)))
    selected_indices: list[int] = []
    selected_source_counts: Counter[str] = Counter()

    while remaining_indices and len(selected_indices) < min(limit, len(chunks)):
        best_remaining_relevance = max(
            raw_relevance[index] for index in remaining_indices
        )
        competitive_indices = [
            index
            for index in remaining_indices
            if raw_relevance[index]
            >= best_remaining_relevance - RELEVANCE_COMPETITION_BAND
        ]

        def adjusted_rank(index: int) -> tuple[float, int]:
            duplicate_penalty = (
                NEAR_DUPLICATE_PENALTY
                if any(
                    _is_near_duplicate(
                        normalized_bodies[index], normalized_bodies[selected_index]
                    )
                    for selected_index in selected_indices
                )
                else 0.0
            )
            source_bonus = (
                UNSEEN_SOURCE_BONUS / (1 + len(selected_source_counts))
                if scores and canonical_sources[index] not in selected_source_counts
                else 0.0
            )
            return (
                raw_relevance[index] - duplicate_penalty + source_bonus,
                -index,
            )

        selected_index = max(competitive_indices, key=adjusted_rank)
        selected_indices.append(selected_index)
        selected_source_counts[canonical_sources[selected_index]] += 1
        remaining_indices.remove(selected_index)

    return [chunks[index] for index in selected_indices]
