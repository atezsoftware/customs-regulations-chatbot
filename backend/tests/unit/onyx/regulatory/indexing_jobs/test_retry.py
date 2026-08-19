from uuid import UUID

import httpx
import pytest

from onyx.db.enums import RegulatoryIndexingStage
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayTimeoutError,
    RetryDisposition,
    RetryReason,
)
from onyx.regulatory.indexing_jobs.retry import (
    classify_indexing_error,
    retry_delay_seconds,
)


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/jobs")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        "provider request failed",
        request=request,
        response=response,
    )


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503, 599])
def test_retryable_http_boundaries(status_code: int) -> None:
    decision = classify_indexing_error(_http_status_error(status_code))

    assert decision.disposition is RetryDisposition.RETRYABLE
    assert decision.reason is RetryReason.HTTP_STATUS
    assert decision.status_code == status_code


@pytest.mark.parametrize("status_code", [400, 401, 403, 600])
def test_terminal_http_boundaries(status_code: int) -> None:
    decision = classify_indexing_error(_http_status_error(status_code))

    assert decision.disposition is RetryDisposition.TERMINAL
    assert decision.reason is RetryReason.HTTP_STATUS
    assert decision.status_code == status_code


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TimeoutError("timed out"), RetryReason.TIMEOUT),
        (httpx.ReadTimeout("timed out"), RetryReason.TIMEOUT),
        (ConnectionResetError("connection reset"), RetryReason.NETWORK),
        (httpx.ConnectError("connection failed"), RetryReason.NETWORK),
    ],
)
def test_transient_transport_errors_are_retryable(
    error: Exception, reason: RetryReason
) -> None:
    decision = classify_indexing_error(error)

    assert decision.disposition is RetryDisposition.RETRYABLE
    assert decision.reason is reason
    assert decision.status_code is None


def test_unknown_errors_fail_closed() -> None:
    decision = classify_indexing_error(ValueError("malformed provider output"))

    assert decision.disposition is RetryDisposition.TERMINAL
    assert decision.reason is RetryReason.UNKNOWN


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (IndexingGatewayTimeoutError(), RetryReason.TIMEOUT),
        (IndexingGatewayConnectionError(), RetryReason.NETWORK),
    ],
)
def test_normalized_gateway_transport_errors_are_retryable(
    error: Exception,
    reason: RetryReason,
) -> None:
    decision = classify_indexing_error(error)

    assert decision.disposition is RetryDisposition.RETRYABLE
    assert decision.reason is reason
    assert decision.status_code is None


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503, 599])
def test_normalized_gateway_retryable_http_boundaries(status_code: int) -> None:
    decision = classify_indexing_error(IndexingGatewayHTTPError(status_code))

    assert decision.disposition is RetryDisposition.RETRYABLE
    assert decision.reason is RetryReason.HTTP_STATUS
    assert decision.status_code == status_code


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_normalized_gateway_terminal_http_boundaries(status_code: int) -> None:
    decision = classify_indexing_error(IndexingGatewayHTTPError(status_code))

    assert decision.disposition is RetryDisposition.TERMINAL
    assert decision.reason is RetryReason.HTTP_STATUS
    assert decision.status_code == status_code


def test_retry_delay_is_deterministic_full_jitter() -> None:
    job_id = UUID("00000000-0000-0000-0000-000000000001")

    delay = retry_delay_seconds(
        job_id=job_id,
        stage=RegulatoryIndexingStage.EMBEDDING,
        attempt=3,
        base_seconds=15,
        max_seconds=900,
    )

    assert delay == pytest.approx(3.1943631762294)
    assert retry_delay_seconds(
        job_id=job_id,
        stage=RegulatoryIndexingStage.EMBEDDING,
        attempt=3,
        base_seconds=15,
        max_seconds=900,
    ) == pytest.approx(delay)
    assert 0 <= delay < 60


def test_retry_delay_never_exceeds_maximum() -> None:
    delay = retry_delay_seconds(
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        stage=RegulatoryIndexingStage.EMBEDDING,
        attempt=20,
        base_seconds=15,
        max_seconds=900,
    )

    assert delay == pytest.approx(727.4887049369461)
    assert 0 <= delay < 900


@pytest.mark.parametrize(
    ("attempt", "base_seconds", "max_seconds"),
    [(0, 15, 900), (1, 0, 900), (1, 15, 0), (1, 30, 15)],
)
def test_retry_delay_rejects_invalid_policy(
    attempt: int, base_seconds: float, max_seconds: float
) -> None:
    with pytest.raises(ValueError):
        retry_delay_seconds(
            job_id=UUID("00000000-0000-0000-0000-000000000001"),
            stage=RegulatoryIndexingStage.EMBEDDING,
            attempt=attempt,
            base_seconds=base_seconds,
            max_seconds=max_seconds,
        )
