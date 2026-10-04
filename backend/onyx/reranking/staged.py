"""Payload-bounded reranking for regulatory candidate pools."""

import math
from collections.abc import Callable, Sequence

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.reranking.payload import (
    payload_limits_for_reranker,
    serialize_rerank_candidates,
)
from onyx.tracing.answer_graph import graph_step

RerankCall = Callable[..., RerankResult]
STAGED_BATCH_SIZE = 32
STAGED_FINAL_LIMIT = 48


def rerank_regulatory_candidates_in_batches(
    *,
    query: str,
    chunks: Sequence[InferenceChunk],
    config: RerankerRuntimeConfig,
    rerank: RerankCall,
) -> RerankResult:
    """Merge all pointwise scores without a per-batch finalist quota."""
    original = list(chunks)
    if not original or not config.enabled:
        return rerank(query=query, chunks=original, config=config)
    limits = payload_limits_for_reranker(config.model_name)
    scores: dict[tuple[str, int], float] = {}
    submitted_count = 0
    result_count = 0
    batch_outcomes: list[str] = []
    start = 0
    with graph_step(
        "rerank.batched",
        {"candidate_count": len(original), "batch_size": STAGED_BATCH_SIZE},
    ) as batch_step:
        while start < len(original):
            payload = serialize_rerank_candidates(
                original[start : start + STAGED_BATCH_SIZE], limits=limits
            )
            batch = payload.submitted_chunks
            if not batch:
                result = RerankResult(
                    ordered_chunks=original,
                    scores_by_chunk={},
                    submitted_count=submitted_count,
                    result_count=result_count,
                    outcome=RerankOutcome.INVALID_RESPONSE,
                    fallback_used=True,
                )
            else:
                result = rerank(query=query, chunks=batch, config=config)
                submitted_count += result.submitted_count
                result_count += result.result_count
            batch_outcomes.append(result.outcome.value)
            batch_ids = {(chunk.document_id, chunk.chunk_id) for chunk in batch}
            complete = (
                bool(batch)
                and result.used_external
                and batch_ids == result.scores_by_chunk.keys()
                and all(
                    math.isfinite(score) for score in result.scores_by_chunk.values()
                )
            )
            if not complete:
                batch_step.summary = "incomplete scoring; complete baseline retained"
                batch_step.output_value = {
                    "batch_outcomes": batch_outcomes,
                    "scored_count": len(scores),
                    "fallback_used": True,
                }
                return result.model_copy(
                    update={
                        "ordered_chunks": original,
                        "scores_by_chunk": {},
                        "submitted_count": submitted_count,
                        "result_count": result_count,
                        "outcome": (
                            result.outcome
                            if not result.used_external
                            else RerankOutcome.INVALID_RESPONSE
                        ),
                        "fallback_used": True,
                    }
                )
            scores.update(result.scores_by_chunk)
            start += len(batch)
        ordered = sorted(
            original,
            key=lambda chunk: scores[(chunk.document_id, chunk.chunk_id)],
            reverse=True,
        )
        batch_step.summary = (
            f"{len(original)} candidates scored in {len(batch_outcomes)} batches"
        )
        batch_step.output_value = {
            "batch_outcomes": batch_outcomes,
            "scored_count": len(scores),
            "fallback_used": False,
        }
        return RerankResult(
            ordered_chunks=ordered,
            scores_by_chunk=scores,
            submitted_count=submitted_count,
            result_count=result_count,
            outcome=RerankOutcome.SUCCESS,
            fallback_used=False,
        )


def rerank_regulatory_candidates_in_stages(
    *,
    query: str,
    chunks: Sequence[InferenceChunk],
    config: RerankerRuntimeConfig,
    rerank: RerankCall,
    final_limit: int = STAGED_FINAL_LIMIT,
) -> RerankResult:
    """Score all candidates in bounded batches, then compare their shortlists.

    A failed provider call returns the complete baseline order. Every source
    remains available to downstream deterministic selection and validation.
    """

    original = list(chunks)
    if final_limit < 1:
        raise ValueError("final_limit must be positive")
    if len(original) <= final_limit:
        return rerank(query=query, chunks=original, config=config)

    batch_count = (len(original) + STAGED_BATCH_SIZE - 1) // STAGED_BATCH_SIZE
    base_slots, extra_slots = divmod(final_limit, batch_count)
    batch_shortlists = [
        base_slots + int(index < extra_slots) for index in range(batch_count)
    ]
    with graph_step(
        "rerank.staged",
        {
            "candidate_count": len(original),
            "batch_size": STAGED_BATCH_SIZE,
            "batch_shortlists": batch_shortlists,
            "final_limit": final_limit,
        },
    ) as staged_step:
        shortlist: list[InferenceChunk] = []
        batch_outcomes: list[str] = []
        for index, start in enumerate(range(0, len(original), STAGED_BATCH_SIZE)):
            batch = original[start : start + STAGED_BATCH_SIZE]
            result = rerank(query=query, chunks=batch, config=config)
            batch_outcomes.append(result.outcome.value)
            if not result.used_external:
                staged_step.summary = "provider fallback; complete baseline retained"
                staged_step.output_value = {
                    "batch_outcomes": batch_outcomes,
                    "finalist_count": 0,
                    "fallback_used": True,
                }
                return result.model_copy(
                    update={"ordered_chunks": original, "fallback_used": True}
                )
            shortlist.extend(result.ordered_chunks[: batch_shortlists[index]])

        finalists = shortlist
        final_result = rerank(query=query, chunks=finalists, config=config)
        if not final_result.used_external:
            staged_step.summary = "final provider fallback; complete baseline retained"
            staged_step.output_value = {
                "batch_outcomes": batch_outcomes,
                "final_outcome": final_result.outcome.value,
                "finalist_count": len(finalists),
                "fallback_used": True,
            }
            return final_result.model_copy(
                update={"ordered_chunks": original, "fallback_used": True}
            )

        finalist_ids = {
            (chunk.document_id, chunk.chunk_id) for chunk in final_result.ordered_chunks
        }
        ordered = [
            *final_result.ordered_chunks,
            *(
                chunk
                for chunk in original
                if (chunk.document_id, chunk.chunk_id) not in finalist_ids
            ),
        ]
        staged_step.summary = (
            f"{len(original)} candidates in {len(batch_outcomes)} batches; "
            f"{len(finalists)} finalists"
        )
        staged_step.output_value = {
            "batch_outcomes": batch_outcomes,
            "final_outcome": final_result.outcome.value,
            "finalist_count": len(finalists),
            "fallback_used": False,
        }
        return final_result.model_copy(update={"ordered_chunks": ordered})
