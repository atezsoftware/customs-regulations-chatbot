import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.circuit_breaker import (
    RerankCircuitBreaker,
    reranker_configuration_fingerprint,
)
from onyx.reranking.constants import MAX_RERANK_CANDIDATES
from onyx.reranking.models import (
    InvalidRerankResponse,
    RerankCircuitKey,
    RerankOutcome,
    RerankPayloadLimits,
    RerankProviderError,
    RerankRateLimited,
    RerankResult,
    RerankTimeout,
)
from onyx.reranking.openrouter import OpenRouterRerankClient
from onyx.reranking.payload import serialize_rerank_candidates
from onyx.server.metrics.reranking import observe_rerank
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id
from shared_configs.enums import RerankerProvider

logger = setup_logger()

TraceCall = Callable[..., AbstractContextManager[Any]]
_IMMEDIATE_PROVIDER_FAILURES = frozenset({400, 401, 402, 403, 404, 413, 422})


class RerankingService:
    def __init__(
        self,
        *,
        client: OpenRouterRerankClient,
        circuit_breaker: RerankCircuitBreaker,
        trace_call: TraceCall = traced_llm_call,
    ) -> None:
        self._client = client
        self._circuit_breaker = circuit_breaker
        self._trace_call = trace_call

    def _result(
        self,
        *,
        started_at: float,
        model: str,
        chunks: Sequence[InferenceChunk],
        ordered_chunks: Sequence[InferenceChunk] | None = None,
        scores_by_chunk: dict[tuple[str, int], float] | None = None,
        submitted_count: int = 0,
        result_count: int = 0,
        outcome: RerankOutcome,
        fallback_used: bool,
    ) -> RerankResult:
        latency_seconds = time.monotonic() - started_at
        observe_rerank(
            outcome=outcome,
            fallback_used=fallback_used,
            latency_seconds=latency_seconds,
            submitted_count=submitted_count,
            result_count=result_count,
        )
        logger.info(
            "Rerank completed model=%s submitted_count=%d result_count=%d outcome=%s fallback_used=%s latency_seconds=%.3f",
            model,
            submitted_count,
            result_count,
            outcome.value,
            fallback_used,
            latency_seconds,
        )
        return RerankResult(
            ordered_chunks=list(
                ordered_chunks if ordered_chunks is not None else chunks
            ),
            scores_by_chunk=scores_by_chunk or {},
            submitted_count=submitted_count,
            result_count=result_count,
            outcome=outcome,
            fallback_used=fallback_used,
        )

    def rerank_chunks(
        self,
        *,
        query: str,
        chunks: Sequence[InferenceChunk],
        config: RerankerRuntimeConfig,
    ) -> RerankResult:
        started_at = time.monotonic()
        model = config.model_name or "unconfigured"
        if not config.enabled:
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                outcome=RerankOutcome.DISABLED,
                fallback_used=True,
            )
        if (
            config.provider_type is not RerankerProvider.OPENROUTER
            or config.model_name is None
            or config.api_key is None
        ):
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                outcome=RerankOutcome.PROVIDER_ERROR,
                fallback_used=True,
            )
        if not chunks:
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                outcome=RerankOutcome.SUCCESS,
                fallback_used=False,
            )

        api_key = config.api_key.get_value(apply_mask=False)
        circuit_key = RerankCircuitKey(
            tenant_id=get_current_tenant_id(),
            config_fingerprint=reranker_configuration_fingerprint(
                model=config.model_name, api_key=api_key
            ),
        )
        if self._circuit_breaker.is_open(circuit_key):
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                outcome=RerankOutcome.CIRCUIT_OPEN,
                fallback_used=True,
            )

        payload = serialize_rerank_candidates(
            chunks,
            limits=RerankPayloadLimits(max_candidates=MAX_RERANK_CANDIDATES),
        )
        if not payload.documents:
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                outcome=RerankOutcome.PROVIDER_ERROR,
                fallback_used=True,
            )

        try:
            with self._trace_call(
                flow=LLMFlow.RERANK,
                model=config.model_name,
                provider=RerankerProvider.OPENROUTER.value,
            ):
                scores = self._client.rerank(
                    api_key=api_key,
                    model=config.model_name,
                    query=query,
                    documents=payload.documents,
                    top_n=len(payload.documents),
                )
        except RerankTimeout:
            self._circuit_breaker.record_failure(circuit_key)
            outcome = RerankOutcome.TIMEOUT
        except RerankRateLimited as error:
            self._circuit_breaker.record_failure(
                circuit_key,
                retry_after_seconds=error.retry_after_seconds,
                immediate=True,
            )
            outcome = RerankOutcome.RATE_LIMITED
        except RerankProviderError as error:
            self._circuit_breaker.record_failure(
                circuit_key,
                retry_after_seconds=error.retry_after_seconds,
                immediate=error.status_code in _IMMEDIATE_PROVIDER_FAILURES,
            )
            outcome = RerankOutcome.PROVIDER_ERROR
        except InvalidRerankResponse:
            self._circuit_breaker.record_failure(circuit_key)
            outcome = RerankOutcome.INVALID_RESPONSE
        else:
            self._circuit_breaker.record_success(circuit_key)
            returned_indices = {score.index for score in scores}
            ordered_chunks = [payload.submitted_chunks[score.index] for score in scores]
            ordered_chunks.extend(
                chunk
                for index, chunk in enumerate(payload.submitted_chunks)
                if index not in returned_indices
            )
            ordered_chunks.extend(payload.unsent_chunks)
            scores_by_chunk = {
                (
                    payload.submitted_chunks[score.index].document_id,
                    payload.submitted_chunks[score.index].chunk_id,
                ): score.relevance_score
                for score in scores
            }
            return self._result(
                started_at=started_at,
                model=model,
                chunks=chunks,
                ordered_chunks=ordered_chunks,
                scores_by_chunk=scores_by_chunk,
                submitted_count=len(payload.documents),
                result_count=len(scores),
                outcome=RerankOutcome.SUCCESS,
                fallback_used=False,
            )

        return self._result(
            started_at=started_at,
            model=model,
            chunks=chunks,
            submitted_count=len(payload.documents),
            outcome=outcome,
            fallback_used=True,
        )


_DEFAULT_SERVICE: RerankingService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def _get_default_service() -> RerankingService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = RerankingService(
                    client=OpenRouterRerankClient(),
                    circuit_breaker=RerankCircuitBreaker(),
                )
    return _DEFAULT_SERVICE


def rerank_chunks(
    *,
    query: str,
    chunks: Sequence[InferenceChunk],
    config: RerankerRuntimeConfig,
) -> RerankResult:
    return _get_default_service().rerank_chunks(
        query=query, chunks=chunks, config=config
    )
