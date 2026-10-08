"""Request-local cancellation observations shared by parallel source workers."""

import math
import threading
import time
from collections.abc import Callable

from pydantic import JsonValue


class PollingCancellation:
    """Rate-limit remote cancellation checks while keeping cancellation sticky."""

    def __init__(
        self,
        callback: Callable[[], bool],
        *,
        interval_seconds: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError(
                "Cancellation polling interval must be finite and positive"
            )
        self._callback = callback
        self._interval = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._next_poll_at = -math.inf
        self._cancelled = False
        self._error: Exception | None = None
        self._checks = 0
        self._remote_observations = 0
        self._remote_total_seconds = 0.0
        self._remote_max_seconds = 0.0
        self._error_count = 0

    def __call__(self) -> bool:
        with self._lock:
            self._checks += 1
            if self._cancelled:
                return True
            started = self._clock()
            if started < self._next_poll_at:
                if self._error is not None:
                    raise self._error
                return False
            self._remote_observations += 1
            try:
                cancelled = self._callback()
                if not isinstance(cancelled, bool):
                    raise TypeError("Cancellation observer must return a boolean")
            except Exception as error:
                self._error = error
                self._error_count += 1
                self._record_observation(started)
                raise
            self._cancelled = cancelled
            self._error = None
            self._record_observation(started)
            return cancelled

    def _record_observation(self, started: float) -> None:
        completed = self._clock()
        elapsed = max(0.0, completed - started)
        self._remote_total_seconds += elapsed
        self._remote_max_seconds = max(self._remote_max_seconds, elapsed)
        self._next_poll_at = completed + self._interval

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "checks": self._checks,
                "remote_observations": self._remote_observations,
                "remote_total_seconds": self._remote_total_seconds,
                "remote_max_seconds": self._remote_max_seconds,
                "cancelled": self._cancelled,
                "error_count": self._error_count,
            }
