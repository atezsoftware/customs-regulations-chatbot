"""Bounded Fast host timings; operation capture never writes graph nodes."""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock, get_ident
from time import perf_counter
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.tracing.answer_graph import graph_step

FAST_TIMING_SERVICE = "asv3_fast_timing"
MAX_FAST_TIMING_INTERVALS = 2048
_OPERATIONS = frozenset(
    {
        "asv3.checkpoint_save",
        "asv3.checkpoint_preparation",
        "asv3.checkpoint_capture",
        "asv3.receipt_commit",
        "asv3.source_notice_lookup",
        "asv3.context_preparation",
        "asv3.model_request",
        "asv3.model_decision",
        "asv3.other",
    }
)
_METRICS = frozenset(
    {
        "query_count",
        "notice_count",
        "sequence",
        "revision",
        "root_snapshot_seconds",
        "capture_lock_wait_seconds",
        "augmentation_seconds",
        "submit_seconds",
        "flush_seconds",
        "durable_barrier",
        "snapshot_seconds",
        "write_seconds",
    }
)
_EXCEPTIONS = frozenset(
    {
        "RunStopped",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "PermissionError",
        "TimeoutError",
        "ConnectionError",
        "KeyError",
        "AssertionError",
        "KeyboardInterrupt",
        "SystemExit",
        "CancelledError",
    }
)
_ACTIVE_TIMING: ContextVar[FastTimingCollector | None] = ContextVar(
    "asv3_fast_timing", default=None
)
_INSTALL_LOCK = Lock()
logger = logging.getLogger(__name__)


def _exception_name(error: BaseException | None) -> str | None:
    if error is None:
        return None
    name = type(error).__name__
    return name if name in _EXCEPTIONS else "other"


@dataclass
class _Totals:
    count: int = 0
    seconds: float = 0.0
    maximum: float = 0.0
    errors: int = 0


class FastTimingCollector:
    """One run's bounded numeric diagnostics, shared by its execution contexts."""

    def __init__(self, *, max_intervals: int = MAX_FAST_TIMING_INTERVALS) -> None:
        if (
            type(max_intervals) is not int
            or not 0 <= max_intervals <= MAX_FAST_TIMING_INTERVALS
        ):
            raise ValueError("Invalid Fast timing interval limit")
        self._limit = max_intervals
        self._lock = Lock()
        self._started = perf_counter()
        self._root_thread = get_ident()
        self._run_id: str | None = None
        self._intervals: list[JsonValue] = []
        self._totals: dict[str, _Totals] = {}
        self._active = 0
        self._dropped = 0
        self._rejected_metadata = 0
        self._closed = False
        self._probe: FastCancellationProbe | None = None

    def bind(
        self, context: RunContext, probe: FastCancellationProbe | None = None
    ) -> bool:
        with self._lock:
            if self._run_id not in (None, context.run_id):
                return False
            self._run_id = context.run_id
            if probe is not None:
                self._probe = probe
            return True

    def metrics(self, values: Mapping[str, object] | None) -> dict[str, JsonValue]:
        clean: dict[str, JsonValue] = {}
        rejected = 0
        for key, value in (values or {}).items():
            if key in _METRICS and (
                value is None
                or type(value) is bool
                or (type(value) is int and abs(value) <= (1 << 63) - 1)
                or (type(value) is float and math.isfinite(value))
            ):
                clean[key] = cast(JsonValue, value)
            else:
                rejected += 1
        with self._lock:
            if not self._closed:
                self._rejected_metadata += rejected
        return clean

    def begin(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._active += 1
            return True

    def record(
        self,
        name: str,
        started: float,
        error: BaseException | None,
        metrics: dict[str, JsonValue],
    ) -> None:
        metrics = self.metrics(metrics)
        finished = perf_counter()
        duration = finished - started
        operation = name if name in _OPERATIONS else "asv3.other"
        with self._lock:
            if self._closed:
                return
            self._active -= 1
            totals = self._totals.setdefault(operation, _Totals())
            totals.count += 1
            totals.seconds += duration
            totals.maximum = max(totals.maximum, duration)
            totals.errors += error is not None
            if len(self._intervals) >= self._limit:
                self._dropped += 1
                return
            self._intervals.append(
                {
                    "operation": operation,
                    "start_seconds": started - self._started,
                    "duration_seconds": duration,
                    "root_thread": get_ident() == self._root_thread,
                    "exception_type": _exception_name(error),
                    "metrics": metrics,
                }
            )

    def flush(self, error: BaseException | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            probe, self._probe = self._probe, None
            payload: dict[str, JsonValue] = {
                "schema_version": 1,
                "duration_seconds": perf_counter() - self._started,
                "exception_type": _exception_name(error),
                "interval_limit": self._limit,
                "dropped_intervals": self._dropped,
                "unfinished_intervals": self._active,
                "truncated": bool(self._dropped or self._active),
                "rejected_metadata_fields": self._rejected_metadata,
                "durations_are_inclusive_not_wall": True,
                "intervals": self._intervals,
                "operations": [
                    {
                        "operation": name,
                        "count": total.count,
                        "cumulative_seconds": total.seconds,
                        "maximum_seconds": total.maximum,
                        "errors": total.errors,
                    }
                    for name, total in sorted(self._totals.items())
                ],
            }
            self._intervals = []
            self._totals = {}
        try:
            if probe is not None:
                payload["cancellation_checks"] = probe.snapshot()
            with graph_step("asv3.fast_host_timing", {}) as capture:
                capture.output_value = payload
        except Exception:
            # Diagnostics must not replace an operation's result or exception.
            logger.warning("Fast host timing capture failed")


class FastTimingStep:
    def __init__(
        self,
        collector: FastTimingCollector,
        metrics: Mapping[str, object] | None,
    ) -> None:
        self._collector = collector
        self._metrics = collector.metrics(metrics)
        self._output: dict[str, JsonValue] = {}

    @property
    def output_value(self) -> dict[str, JsonValue]:
        return self._output

    @output_value.setter
    def output_value(self, value: Mapping[str, object]) -> None:
        self._output = self._collector.metrics(value)


def install_fast_timing(
    context: RunContext,
    *,
    cancellation_probe: FastCancellationProbe | None = None,
) -> FastTimingCollector | None:
    if context.services.get("asv3_fast") is not True:
        return None
    with _INSTALL_LOCK:
        existing = context.services.get(FAST_TIMING_SERVICE)
        collector = (
            existing
            if isinstance(existing, FastTimingCollector)
            else _ACTIVE_TIMING.get()
        )
        if collector is None:
            collector = FastTimingCollector()
        if not collector.bind(context, cancellation_probe):
            return None
        context.services[FAST_TIMING_SERVICE] = collector
        return collector


@contextmanager
def fast_timing_step(
    context: RunContext,
    name: str,
    metrics: Mapping[str, object] | None = None,
) -> Iterator[FastTimingStep | None]:
    collector = context.services.get(FAST_TIMING_SERVICE)
    if not isinstance(collector, FastTimingCollector):
        collector = install_fast_timing(context)
    if collector is None or not collector.bind(context) or not collector.begin():
        yield None
        return
    started = perf_counter()
    step = FastTimingStep(collector, metrics)
    error: BaseException | None = None
    try:
        yield step
    except BaseException as caught:
        error = caught
        raise
    finally:
        collector.record(name, started, error, {**step._metrics, **step.output_value})


@contextmanager
def fast_timing_run(enabled: bool) -> Iterator[None]:
    if enabled is not True:
        yield
        return
    collector = FastTimingCollector()
    token = _ACTIVE_TIMING.set(collector)
    error: BaseException | None = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        try:
            collector.flush(error)
        finally:
            _ACTIVE_TIMING.reset(token)


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
            if owner.context.services.get("asv3_fast") is not True and not isinstance(
                owner.context.services.get(FAST_TIMING_SERVICE), FastTimingCollector
            ):
                return function(owner, *args, **kwargs)
            with fast_timing_step(owner.context, name):
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
