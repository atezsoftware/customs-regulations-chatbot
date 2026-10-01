"""Bounded two-stage reranking for large regulatory candidate pools."""

from collections.abc import Callable, Sequence

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankResult
from onyx.tracing.answer_graph import graph_step

RerankCall = Callable[..., RerankResult]
STAGED_BATCH_SIZE = 32
STAGED_FINAL_LIMIT = 48


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
