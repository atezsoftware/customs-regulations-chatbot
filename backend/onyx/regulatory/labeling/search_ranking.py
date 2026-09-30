"""Bounded label ranking that preserves an independent text retrieval path."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import cast

from onyx.context.search.models import InferenceChunk
from onyx.regulatory.labeling.defaults import load_default_taxonomy
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from onyx.regulatory.labeling.search_models import (
    LabelFacet,
    LabelFusionScore,
    LabelSearchHint,
    LabelSearchSnapshot,
    SearchLabelEvidence,
)

MAX_LABEL_PROMOTION = 2
LABEL_FUSION_RANK_CONSTANT = 60


def resolve_label_facets(taxonomy: TaxonomyDefinition) -> dict[str, LabelFacet]:
    approved = {label.id: label for label in load_default_taxonomy().labels}
    families: dict[str, LabelFacet] = {
        "SUB": "subject",
        "EFF": "effect",
        "ANX": "annex",
        "SEC": "sector",
    }
    return {
        label.id: (
            families[label.id.split(".")[0]]
            if approved.get(label.id) == label
            else "untyped"
        )
        for label in taxonomy.labels
    }


def validate_search_hint(
    value: object, snapshot: LabelSearchSnapshot
) -> LabelSearchHint:
    if not isinstance(value, dict):
        return LabelSearchHint()
    raw = cast(dict[str, object], value).get("label_ids")
    if not isinstance(raw, (list, tuple)) or len(raw) > 12:
        return LabelSearchHint()
    allowed = {label.id for label in snapshot.taxonomy.labels}
    return LabelSearchHint(
        label_ids=tuple(
            dict.fromkeys(
                label for label in raw if isinstance(label, str) and label in allowed
            )
        )
    )


def merge_label_candidates(
    baseline: Sequence[InferenceChunk],
    extra: Sequence[InferenceChunk],
    *,
    limit: int,
) -> list[InferenceChunk]:
    if limit < 1:
        return []
    if not extra:
        return list(baseline[:limit])
    seen = {chunk.regulatory_chunk_id or chunk.unique_id for chunk in baseline}
    additions: list[InferenceChunk] = []
    for chunk in extra:
        key = chunk.regulatory_chunk_id or chunk.unique_id
        if key not in seen:
            additions.append(chunk)
            seen.add(key)
    additions = additions[: max(1, limit // 4)]
    keep = max(0, limit - len(additions))
    return [*baseline[:keep], *additions][:limit]


def fuse_label_candidate_scores(
    baseline: Sequence[InferenceChunk],
    extra: Sequence[InferenceChunk],
    *,
    label_ranked_ids: Sequence[str],
    limit: int,
) -> tuple[list[InferenceChunk], tuple[LabelFusionScore, ...]]:
    """Give a verified label lane the same RRF weight as the baseline retrieval lane."""
    candidates = merge_label_candidates(baseline, extra, limit=limit)
    if not candidates or not label_ranked_ids:
        return candidates, ()

    def identifier(chunk: InferenceChunk) -> str:
        return chunk.regulatory_chunk_id or chunk.unique_id

    baseline_ranks = {
        identifier(chunk): rank for rank, chunk in enumerate(baseline, start=1)
    }
    label_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(dict.fromkeys(label_ranked_ids), start=1)
    }
    scored: list[tuple[InferenceChunk, LabelFusionScore]] = []
    for chunk in candidates:
        chunk_id = identifier(chunk)
        baseline_rank = baseline_ranks.get(chunk_id)
        label_rank = label_ranks.get(chunk_id)
        baseline_score = (
            1.0 / (LABEL_FUSION_RANK_CONSTANT + baseline_rank)
            if baseline_rank is not None
            else 0.0
        )
        label_score = (
            1.0 / (LABEL_FUSION_RANK_CONSTANT + label_rank)
            if label_rank is not None
            else 0.0
        )
        scored.append(
            (
                chunk,
                LabelFusionScore(
                    candidate_id=chunk_id,
                    baseline_rank=baseline_rank,
                    label_rank=label_rank,
                    baseline_score=baseline_score,
                    label_score=label_score,
                    combined_score=baseline_score + label_score,
                ),
            )
        )
    scored.sort(
        key=lambda entry: (
            -entry[1].combined_score,
            entry[1].baseline_rank or limit + 1,
            entry[1].label_rank or limit + 1,
            entry[1].candidate_id,
        )
    )
    return [chunk for chunk, _ in scored], tuple(score for _, score in scored)


def rank_near_tied_label_candidates(
    candidates: Sequence[InferenceChunk],
    *,
    scores: Mapping[tuple[str, int], float],
    evidence: Mapping[str, tuple[SearchLabelEvidence, ...]],
    hint: LabelSearchHint,
) -> list[InferenceChunk]:
    """Use verified labels to break near ties without overriding semantic relevance."""
    ranked = list(candidates)
    requested = set(hint.label_ids)

    def matches(chunk: InferenceChunk) -> bool:
        return any(
            entry.label_id in requested
            for entry in evidence.get(chunk.regulatory_chunk_id or "", ())
        )

    for original_index, chunk in enumerate(candidates):
        if not matches(chunk):
            continue
        position = next(i for i, candidate in enumerate(ranked) if candidate is chunk)
        score = scores.get((chunk.document_id, chunk.chunk_id))
        if score is None or not isfinite(score):
            continue
        while position > max(0, original_index - MAX_LABEL_PROMOTION):
            previous = ranked[position - 1]
            previous_score = scores.get((previous.document_id, previous.chunk_id))
            if (
                matches(previous)
                or previous_score is None
                or not isfinite(previous_score)
                or abs(score - previous_score)
                > 0.05 * max(abs(score), abs(previous_score), 1e-8)
            ):
                break
            ranked[position - 1], ranked[position] = chunk, previous
            position -= 1
    return ranked
