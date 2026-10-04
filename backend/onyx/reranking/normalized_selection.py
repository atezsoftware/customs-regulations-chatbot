"""Preserve baseline coverage and the complete relative high-score band."""

import math
from collections.abc import Mapping, Sequence

from onyx.context.search.models import InferenceChunk
from onyx.reranking.models import RerankSelection


def normalize_rerank_scores(
    scores: Mapping[tuple[str, int], float],
) -> dict[tuple[str, int], float]:
    """Min-max normalize one search pool, without treating scores as probabilities."""
    valid = [score for score in scores.values() if math.isfinite(score)]
    minimum = min(valid, default=0.0)
    maximum = max(valid, default=0.0)
    span = maximum - minimum
    scale = max(abs(minimum), abs(maximum)) if math.isinf(span) else 1.0
    scaled_minimum = minimum / scale
    span = maximum / scale - scaled_minimum
    return {
        key: (
            (score / scale - scaled_minimum) / span if span > 0 else float(maximum > 0)
        )
        if math.isfinite(score)
        else 0.0
        for key, score in scores.items()
    }


def select_normalized_rerank_candidates(
    *,
    chunks: Sequence[InferenceChunk],
    scores: Mapping[tuple[str, int], float],
    baseline_limit: int,
    threshold: float = 0.90,
) -> RerankSelection:
    if baseline_limit < 0 or not 0 < threshold <= 1:
        raise ValueError("Invalid rerank selection limits")
    candidate_ids = {(chunk.document_id, chunk.chunk_id) for chunk in chunks}
    normalized = normalize_rerank_scores(
        {key: score for key, score in scores.items() if key in candidate_ids}
    )
    baseline_ids = {
        (chunk.document_id, chunk.chunk_id) for chunk in chunks[:baseline_limit]
    }
    selected: list[InferenceChunk] = []
    qualified: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = chunk.document_id, chunk.chunk_id
        if key in seen:
            continue
        seen.add(key)
        score = normalized.get(key, 0.0)
        above_threshold = score >= threshold or math.isclose(
            score, threshold, rel_tol=0.0, abs_tol=1e-12
        )
        if above_threshold:
            qualified.append(key)
        if key in baseline_ids or above_threshold:
            selected.append(chunk)
    return RerankSelection(
        ordered_chunks=selected,
        normalized_scores_by_chunk=normalized,
        qualified_chunk_ids=qualified,
        threshold=threshold,
    )
