"""Fast-only host timing without changing research, cancellation or checkpoint data."""

from collections.abc import Callable
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import Concatenate, ParamSpec, Protocol, TypeVar

from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.tracing.answer_graph import graph_step


class ContextOwner(Protocol):
    @property
    def context(self) -> RunContext: ...


Owner = TypeVar("Owner", bound=ContextOwner)
P = ParamSpec("P")
R = TypeVar("R")


def profile_fast_step(
    name: str,
) -> Callable[[Callable[Concatenate[Owner, P], R]], Callable[Concatenate[Owner, P], R]]:
    def decorate(
        function: Callable[Concatenate[Owner, P], R],
    ) -> Callable[Concatenate[Owner, P], R]:
        @wraps(function)
        def measured(owner: Owner, *args: P.args, **kwargs: P.kwargs) -> R:
            if owner.context.services.get("asv3_fast") is not True:
                return function(owner, *args, **kwargs)
            with graph_step(name, {}):
                return function(owner, *args, **kwargs)

        return measured

    return decorate


class FastCancellationProbe:
    """Count actual cache checks; never reuse, delay or suppress their result."""

    def __init__(self, check: Callable[[], bool]) -> None:
        self._check = check
        self._lock = Lock()
        self._count = 0
        self._seconds = 0.0
        self._maximum = 0.0

    def __call__(self) -> bool:
        started = perf_counter()
        try:
            return self._check()
        finally:
            duration = perf_counter() - started
            with self._lock:
                self._count += 1
                self._seconds += duration
                self._maximum = max(self._maximum, duration)

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "count": self._count,
                "cumulative_seconds": self._seconds,
                "maximum_seconds": self._maximum,
                "elapsed_is_cumulative_not_wall": True,
                "cached_results": 0,
            }
