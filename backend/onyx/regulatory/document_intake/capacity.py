"""Lease-aware, bounded retries for explicit metadata provider capacity rejection."""

import hashlib
import re
from collections.abc import Callable
from threading import Event

from onyx.regulatory.indexing_jobs.models import IndexingGatewayHTTPError

_MAX_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 30.0
_MAX_DELAY_SECONDS = 120.0
_WAIT_TICK_SECONDS = 15.0


class MetadataCapacityStopped(RuntimeError):
    """Another window stopped this metadata operation during its capacity wait."""


def capacity_retry_delay_seconds(request_hash: str, retry_number: int) -> float:
    """Deterministic half-to-full exponential jitter keeps concurrent retries apart."""
    if re.fullmatch(r"[0-9a-f]{64}", request_hash) is None:
        raise ValueError("Metadata capacity retry requires its exact request hash")
    if type(retry_number) is not int or not 1 <= retry_number < _MAX_ATTEMPTS:
        raise ValueError("Metadata capacity retry number must be between one and three")
    ceiling = min(_MAX_DELAY_SECONDS, _BASE_DELAY_SECONDS * 2 ** (retry_number - 1))
    seed = f"metadata-capacity:{request_hash}:{retry_number}".encode()
    fraction = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") / 2**64
    return ceiling * (0.5 + 0.5 * fraction)


def run_with_metadata_capacity_retry[T](
    operation: Callable[[], T],
    *,
    request_hash: str,
    heartbeat: Callable[[], None],
    stopped: Event,
) -> T:
    """Repeat only explicit HTTP 429 rejections using the unchanged operation."""
    capacity_retry_delay_seconds(request_hash, 1)

    def check_stopped() -> None:
        if stopped.is_set():
            raise MetadataCapacityStopped("Metadata capacity wait was stopped")

    for attempt in range(_MAX_ATTEMPTS):
        check_stopped()
        heartbeat()
        check_stopped()
        try:
            return operation()
        except IndexingGatewayHTTPError as error:
            if error.status_code != 429 or attempt == _MAX_ATTEMPTS - 1:
                raise
            remaining = capacity_retry_delay_seconds(request_hash, attempt + 1)
            while remaining > 0:
                check_stopped()
                heartbeat()
                check_stopped()
                tick = min(_WAIT_TICK_SECONDS, remaining)
                if stopped.wait(tick):
                    raise MetadataCapacityStopped("Metadata capacity wait was stopped")
                check_stopped()
                remaining -= tick
    raise AssertionError("Metadata capacity attempts exhausted without an outcome")
