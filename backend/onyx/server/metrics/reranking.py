from prometheus_client import Counter, Histogram

from onyx.reranking.models import RerankOutcome
from onyx.utils.logger import setup_logger

logger = setup_logger()

RERANK_REQUESTS = Counter(
    "onyx_rerank_requests_total",
    "Reranking service outcomes",
    ["outcome", "fallback_used"],
)
RERANK_LATENCY = Histogram(
    "onyx_rerank_latency_seconds",
    "Reranking service latency",
    ["outcome"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
RERANK_SUBMITTED_CANDIDATES = Counter(
    "onyx_rerank_submitted_candidates_total",
    "Candidates submitted to the external reranker",
    ["outcome"],
)
RERANK_RESULTS = Counter(
    "onyx_rerank_results_total",
    "Validated results returned by the external reranker",
    ["outcome"],
)


def observe_rerank(
    *,
    outcome: RerankOutcome,
    fallback_used: bool,
    latency_seconds: float,
    submitted_count: int,
    result_count: int,
) -> None:
    try:
        outcome_label = outcome.value
        RERANK_REQUESTS.labels(
            outcome=outcome_label, fallback_used=str(fallback_used).lower()
        ).inc()
        RERANK_LATENCY.labels(outcome=outcome_label).observe(latency_seconds)
        RERANK_SUBMITTED_CANDIDATES.labels(outcome=outcome_label).inc(submitted_count)
        RERANK_RESULTS.labels(outcome=outcome_label).inc(result_count)
    except Exception:
        logger.debug("Failed to record reranking metrics", exc_info=True)
