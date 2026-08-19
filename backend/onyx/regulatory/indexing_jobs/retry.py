import math
import socket
from hashlib import sha256
from uuid import UUID

import httpx

from onyx.db.enums import RegulatoryIndexingStage
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayTimeoutError,
    RetryDecision,
    RetryDisposition,
    RetryReason,
)

_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429})


def _http_retry_decision(status_code: int) -> RetryDecision:
    disposition = (
        RetryDisposition.RETRYABLE
        if status_code in _RETRYABLE_HTTP_STATUS_CODES or 500 <= status_code < 600
        else RetryDisposition.TERMINAL
    )
    return RetryDecision(
        disposition=disposition,
        reason=RetryReason.HTTP_STATUS,
        status_code=status_code,
    )


def classify_indexing_error(error: Exception) -> RetryDecision:
    """Classify known transport failures without inspecting exception messages."""

    if isinstance(
        error,
        (IndexingGatewayTimeoutError, TimeoutError, httpx.TimeoutException),
    ):
        return RetryDecision(
            disposition=RetryDisposition.RETRYABLE,
            reason=RetryReason.TIMEOUT,
        )
    if isinstance(
        error,
        (
            IndexingGatewayConnectionError,
            ConnectionError,
            socket.gaierror,
            httpx.NetworkError,
        ),
    ):
        return RetryDecision(
            disposition=RetryDisposition.RETRYABLE,
            reason=RetryReason.NETWORK,
        )
    if isinstance(error, httpx.HTTPStatusError):
        return _http_retry_decision(error.response.status_code)
    if isinstance(error, IndexingGatewayHTTPError):
        return _http_retry_decision(error.status_code)
    return RetryDecision(
        disposition=RetryDisposition.TERMINAL,
        reason=RetryReason.UNKNOWN,
    )


def retry_delay_seconds(
    job_id: UUID,
    stage: RegulatoryIndexingStage,
    attempt: int,
    base_seconds: float,
    max_seconds: float,
) -> float:
    """Return deterministic exponential full jitter for a persisted retry."""

    if attempt < 1:
        raise ValueError("attempt must be at least one")
    if not math.isfinite(base_seconds) or base_seconds <= 0:
        raise ValueError("base_seconds must be positive and finite")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if max_seconds < base_seconds:
        raise ValueError("max_seconds must be at least base_seconds")

    exponent = attempt - 1
    capped_exponent = math.ceil(math.log2(max_seconds / base_seconds))
    delay_ceiling = (
        max_seconds if exponent >= capped_exponent else base_seconds * (2**exponent)
    )
    seed = f"{job_id}:{stage.value}:{attempt}".encode()
    jitter_fraction = int.from_bytes(sha256(seed).digest()[:8], "big") / 2**64
    return jitter_fraction * delay_ceiling
