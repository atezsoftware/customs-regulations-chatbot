import hashlib
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from onyx.reranking.constants import (
    RERANK_CIRCUIT_COOLDOWN_SECONDS,
    RERANK_CIRCUIT_FAILURE_THRESHOLD,
    RERANK_CIRCUIT_MAX_COOLDOWN_SECONDS,
)
from onyx.reranking.models import RerankCircuitKey


class _CircuitState(NamedTuple):
    failures: int
    open_until: float | None


def reranker_configuration_fingerprint(*, model: str, api_key: str) -> str:
    material = f"{model}\0{api_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class RerankCircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = RERANK_CIRCUIT_FAILURE_THRESHOLD,
        default_cooldown_seconds: float = RERANK_CIRCUIT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = RERANK_CIRCUIT_MAX_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self._failure_threshold = failure_threshold
        self._default_cooldown_seconds = default_cooldown_seconds
        self._max_cooldown_seconds = max_cooldown_seconds
        self._clock = clock
        self._states: dict[RerankCircuitKey, _CircuitState] = {}
        self._lock = threading.Lock()

    def is_open(self, key: RerankCircuitKey) -> bool:
        with self._lock:
            state = self._states.get(key)
            if state is None or state.open_until is None:
                return False
            if self._clock() < state.open_until:
                return True
            self._states.pop(key, None)
            return False

    def record_success(self, key: RerankCircuitKey) -> None:
        with self._lock:
            self._states.pop(key, None)

    def record_failure(
        self,
        key: RerankCircuitKey,
        *,
        retry_after_seconds: float | None = None,
        immediate: bool = False,
    ) -> None:
        with self._lock:
            current = self._states.get(key, _CircuitState(0, None))
            failures = current.failures + 1
            if not immediate and failures < self._failure_threshold:
                self._states[key] = _CircuitState(failures, None)
                return
            requested = (
                retry_after_seconds
                if retry_after_seconds is not None
                else self._default_cooldown_seconds
            )
            cooldown = min(max(requested, 0.0), self._max_cooldown_seconds)
            self._states[key] = _CircuitState(failures, self._clock() + cooldown)

    def invalidate(
        self, *, tenant_id: str, config_fingerprint: str | None = None
    ) -> None:
        with self._lock:
            keys = [
                key
                for key in self._states
                if key.tenant_id == tenant_id
                and (
                    config_fingerprint is None
                    or key.config_fingerprint == config_fingerprint
                )
            ]
            for key in keys:
                self._states.pop(key, None)
