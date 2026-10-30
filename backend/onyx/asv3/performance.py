"""Fast-only host timing without changing research, cancellation or checkpoint data."""

import sys
from collections.abc import Callable
from functools import wraps
from threading import Lock, get_ident
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
        self._root_thread = get_ident()
        self._locations: dict[tuple[str, str, bool], tuple[int, float, float]] = {}

    def _location(self) -> tuple[str, str, bool]:
        frame = sys._getframe(1)
        try:
            while frame is not None:
                module = frame.f_globals.get("__name__")
                if isinstance(module, str) and module not in {
                    "onyx.asv3.performance",
                    "onyx.asv3.models",
                }:
                    if module.startswith("onyx."):
                        return (
                            module,
                            frame.f_code.co_qualname,
                            get_ident() == self._root_thread,
                        )
                    break
                frame = frame.f_back
            return ("external", "external", get_ident() == self._root_thread)
        finally:
            del frame

    def __call__(self) -> bool:
        location = self._location()
        started = perf_counter()
        try:
            return self._check()
        finally:
            duration = perf_counter() - started
            with self._lock:
                self._count += 1
                self._seconds += duration
                self._maximum = max(self._maximum, duration)
                if location not in self._locations and len(self._locations) >= 64:
                    location = ("other", "other", location[2])
                count, total, maximum = self._locations.get(location, (0, 0.0, 0.0))
                self._locations[location] = (
                    count + 1,
                    total + duration,
                    max(maximum, duration),
                )

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "count": self._count,
                "cumulative_seconds": self._seconds,
                "maximum_seconds": self._maximum,
                "elapsed_is_cumulative_not_wall": True,
                "cached_results": 0,
                "locations": [
                    {
                        "module": module,
                        "function": function,
                        "root_thread": root,
                        "count": values[0],
                        "cumulative_seconds": values[1],
                        "maximum_seconds": values[2],
                    }
                    for (module, function, root), values in sorted(
                        self._locations.items()
                    )
                ],
            }
