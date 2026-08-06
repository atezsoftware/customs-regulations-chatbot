import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.circuit_breaker import RerankCircuitBreaker
from onyx.reranking.models import (
    InvalidRerankResponse,
    RerankOutcome,
    RerankProviderError,
    RerankRateLimited,
    RerankScore,
    RerankTimeout,
)
from onyx.reranking.openrouter import OpenRouterRerankClient
from onyx.reranking.service import RerankingService
from onyx.tracing.flows import LLMFlow
from onyx.utils.sensitive import make_mock_sensitive_value
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from shared_configs.enums import RerankerProvider


def _chunk(document_id: str, *, body: str = "gövde") -> InferenceChunk:
    return InferenceChunk(
        chunk_id=0,
        blurb=body,
        content=body,
        source_links=None,
        image_file_id=None,
        section_continuation=False,
        document_id=document_id,
        source_type=DocumentSource.FILE,
        semantic_identifier=document_id,
        title=document_id,
        boost=0,
        score=1.0,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )


def _config(*, enabled: bool = True) -> RerankerRuntimeConfig:
    return RerankerRuntimeConfig(
        enabled=enabled,
        provider_type=RerankerProvider.OPENROUTER if enabled else None,
        model_name="cohere/rerank-v3.5" if enabled else None,
        api_key=make_mock_sensitive_value("top-secret") if enabled else None,
    )


@contextmanager
def _span(**_kwargs: object) -> Iterator[MagicMock]:
    yield MagicMock()


@pytest.fixture
def client() -> MagicMock:
    return MagicMock(spec=OpenRouterRerankClient)


@pytest.fixture
def service(client: MagicMock) -> RerankingService:
    return RerankingService(
        client=client,
        circuit_breaker=RerankCircuitBreaker(failure_threshold=1),
        trace_call=_span,
    )


def test_disabled_configuration_preserves_order_without_provider_call(
    service: RerankingService, client: MagicMock
) -> None:
    chunks = [_chunk("d1"), _chunk("d2")]

    result = service.rerank_chunks(
        query="soru", chunks=chunks, config=_config(enabled=False)
    )

    assert result.ordered_chunks == chunks
    assert result.outcome is RerankOutcome.DISABLED
    assert result.fallback_used is True
    assert result.submitted_count == 0
    assert result.result_count == 0
    client.rerank.assert_not_called()


def test_partial_response_appends_omitted_and_unsent_tail(
    service: RerankingService, client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("d1"), _chunk("d2"), _chunk("d3")]
    client.rerank.return_value = [RerankScore(index=1, relevance_score=0.9)]
    monkeypatch.setattr("onyx.reranking.service.MAX_RERANK_CANDIDATES", 2)

    result = service.rerank_chunks(query="q", chunks=chunks, config=_config())

    assert [chunk.unique_id for chunk in result.ordered_chunks] == [
        "d2__0",
        "d1__0",
        "d3__0",
    ]
    assert result.scores_by_chunk == {("d2", 0): 0.9}
    assert result.submitted_count == 2
    assert result.result_count == 1
    assert result.outcome is RerankOutcome.SUCCESS
    assert result.fallback_used is False
    assert client.rerank.call_args.kwargs["top_n"] == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RerankTimeout(), RerankOutcome.TIMEOUT),
        (RerankRateLimited(retry_after_seconds=30), RerankOutcome.RATE_LIMITED),
        (RerankProviderError(status_code=503), RerankOutcome.PROVIDER_ERROR),
        (InvalidRerankResponse(), RerankOutcome.INVALID_RESPONSE),
    ],
)
def test_provider_failures_preserve_all_chunks(
    service: RerankingService,
    client: MagicMock,
    error: Exception,
    expected: RerankOutcome,
) -> None:
    chunks = [_chunk("d1"), _chunk("d2")]
    client.rerank.side_effect = error

    result = service.rerank_chunks(
        query="secret-query", chunks=chunks, config=_config()
    )

    assert result.ordered_chunks == chunks
    assert result.outcome is expected
    assert result.fallback_used is True
    assert result.result_count == 0


def test_immediate_provider_error_opens_only_current_tenant_and_fingerprint(
    client: MagicMock,
) -> None:
    circuit = RerankCircuitBreaker(failure_threshold=3)
    service = RerankingService(client=client, circuit_breaker=circuit, trace_call=_span)
    client.rerank.side_effect = RerankProviderError(status_code=401)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("tenant-a")
    try:
        first = service.rerank_chunks(
            query="q", chunks=[_chunk("d1")], config=_config()
        )
        second = service.rerank_chunks(
            query="q", chunks=[_chunk("d1")], config=_config()
        )
        CURRENT_TENANT_ID_CONTEXTVAR.set("tenant-b")
        third = service.rerank_chunks(
            query="q", chunks=[_chunk("d1")], config=_config()
        )
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)

    assert first.outcome is RerankOutcome.PROVIDER_ERROR
    assert second.outcome is RerankOutcome.CIRCUIT_OPEN
    assert third.outcome is RerankOutcome.PROVIDER_ERROR
    assert client.rerank.call_count == 2


def test_trace_uses_rerank_flow_without_recording_request_content(
    client: MagicMock,
) -> None:
    trace_call = MagicMock(side_effect=_span)
    service = RerankingService(
        client=client,
        circuit_breaker=RerankCircuitBreaker(),
        trace_call=trace_call,
    )
    client.rerank.return_value = [RerankScore(index=0, relevance_score=0.5)]

    service.rerank_chunks(
        query="sensitive query", chunks=[_chunk("d1")], config=_config()
    )

    assert trace_call.call_args.kwargs == {
        "flow": LLMFlow.RERANK,
        "model": "cohere/rerank-v3.5",
        "provider": "openrouter",
    }


def test_logs_are_secret_free(
    service: RerankingService,
    client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    query = "QUERY-MUST-NOT-LEAK"
    body = "BODY-MUST-NOT-LEAK"
    api_key = "top-secret"
    client.rerank.side_effect = RerankProviderError(status_code=503)

    with caplog.at_level(logging.INFO, logger="onyx"):
        service.rerank_chunks(
            query=query, chunks=[_chunk("d1", body=body)], config=_config()
        )

    combined = " ".join(record.getMessage() for record in caplog.records)
    assert query not in combined
    assert body not in combined
    assert api_key not in combined
    assert "cohere/rerank-v3.5" in combined
    assert "provider_error" in combined
