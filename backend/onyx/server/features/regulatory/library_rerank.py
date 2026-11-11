"""Provider-bounded scoring and one cooldown-aware retry for library searches."""

import time
from collections.abc import Sequence

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.reranking.payload import payload_limits_for_reranker
from onyx.reranking.staged import RerankCall, rerank_regulatory_candidates_in_batches

MAX_LIBRARY_COOLDOWN_SECONDS = 60
LIBRARY_RETRY_BUDGET_SECONDS = 65


def rerank_library_candidates(
    *,
    query: str,
    chunks: Sequence[InferenceChunk],
    config: RerankerRuntimeConfig,
    rerank: RerankCall,
) -> RerankResult:
    started = time.monotonic()
    retried = False

    def score_batch(**kwargs: object) -> RerankResult:
        nonlocal retried
        result = rerank(**kwargs)
        delay = result.retry_after_seconds
        if (
            retried
            or result.outcome
            not in {RerankOutcome.RATE_LIMITED, RerankOutcome.CIRCUIT_OPEN}
            or delay is None
            or delay > MAX_LIBRARY_COOLDOWN_SECONDS
            or time.monotonic() - started + delay > LIBRARY_RETRY_BUDGET_SECONDS
        ):
            return result
        retried = True
        time.sleep(max(0.0, delay) + 0.05)
        return rerank(**kwargs)

    return rerank_regulatory_candidates_in_batches(
        query=query,
        chunks=chunks,
        config=config,
        rerank=score_batch,
        batch_size=payload_limits_for_reranker(config.model_name).max_candidates,
    )
