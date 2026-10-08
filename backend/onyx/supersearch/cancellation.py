"""Share short stop-signal polls across one run's canonical readers."""

from collections.abc import Callable
from threading import Lock
from time import monotonic


class CancellationProbe:
    def __init__(
        self,
        cancelled: Callable[[], bool],
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._cancelled = cancelled
        self._clock = clock
        self._checked_at = float("-inf")
        self._stopped = False
        self._lock = Lock()

    def __call__(self, *, force: bool = False) -> bool:
        # Only stop-signal I/O is shared; authorization and publication stay fresh.
        with self._lock:
            if self._stopped:
                return True
            if force or self._clock() - self._checked_at >= 0.1:
                stopped = self._cancelled()
                self._stopped = stopped
                self._checked_at = self._clock()
            return self._stopped
