"""Conservative admission for independent matching inside a supervised process."""

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TypeVar, cast

MIB = 1024 * 1024
MAX_ANALYSIS_PARALLELISM = 10
T = TypeVar("T")
R = TypeVar("R")


class ResourcePressure(RuntimeError):
    """Transient capacity failure, never an unresolved legal instruction."""

    def __init__(self, reason: str, *, started: bool = False) -> None:
        super().__init__(reason)
        self.started = started


@dataclass(frozen=True)
class MemorySample:
    current: int
    limit: int


def read_memory(root: Path = Path("/sys/fs/cgroup")) -> MemorySample | None:
    """Read the container cgroup, rejecting missing/unlimited/invalid limits."""
    try:
        current = int((root / "memory.current").read_text().strip())
        limit = int((root / "memory.max").read_text().strip())
        if current < 0 or limit <= 0:
            return None
        return MemorySample(current, limit)
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class MemoryPolicy:
    reserve_bytes: int = 500 * MIB
    item_bytes: int = 512 * MIB

    def check(self, reading: MemorySample | None) -> MemorySample:
        if reading is None:
            raise ResourcePressure("memory_telemetry_unavailable")
        ceiling = reading.limit - self.reserve_bytes
        if reading.current >= ceiling:
            raise ResourcePressure("memory_pressure")
        return reading

    def admit(self, reading: MemorySample | None, *, active: int) -> bool:
        value = self.check(reading)
        # Account for growth of every in-flight call, not just the next one.
        ceiling = value.limit - self.reserve_bytes
        return value.current + (active + 1) * self.item_bytes <= ceiling


@dataclass(frozen=True)
class _InFlight:
    key: Hashable | None
    work_class: Hashable
    baseline: int


def bounded_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    sample: Callable[[], MemorySample | None] = read_memory,
    policy: MemoryPolicy = MemoryPolicy(),
    max_parallel: int = 2,
    report: Callable[[dict[str, int]], None] | None = None,
    concurrency_key: Callable[[T], Hashable] | None = None,
    work_class: Callable[[T], Hashable] | None = None,
) -> Iterator[R]:
    """Warm up serially, then admit bounded calls using fresh container samples."""
    if (
        type(max_parallel) is not int
        or not 1 <= max_parallel <= MAX_ANALYSIS_PARALLELISM
    ):
        raise ValueError("Matching concurrency must be between one and ten")
    iterator = iter(items)
    pending: dict[Future[R], _InFlight] = {}
    queued: deque[T] = deque()
    warmed_up = False
    exhausted = False
    measured: set[Hashable] = set()
    observed_costs: dict[Hashable, int] = {}
    floor = min(policy.item_bytes, 256 * MIB)
    peak_bytes = peak_active = 0
    next_report = 0.0
    last_state: tuple[int, bool, bool, bool] | None = None
    admission_limited = False
    dependency_limited = False
    pool = ThreadPoolExecutor(
        max_workers=max_parallel, thread_name_prefix="amendment-match"
    )

    def estimate(kind: Hashable) -> int:
        return max(
            observed_costs.get(kind, floor),
            floor if kind in measured else policy.item_bytes,
        )

    def measure(reading: MemorySample) -> None:
        # Concurrent growth cannot be attributed to one Python thread. Charge
        # its full upper bound to each participating class rather than undercount.
        for item in pending.values():
            growth = max(0, reading.current - item.baseline)
            observed_costs[item.work_class] = max(
                observed_costs.get(item.work_class, floor), growth * 3 // 2
            )

    def next_eligible() -> tuple[T, Hashable | None] | None:
        nonlocal exhausted
        occupied = (
            {item.key for item in pending.values()}
            if concurrency_key is not None
            else set()
        )
        for index, item in enumerate(queued):
            key = concurrency_key(item) if concurrency_key is not None else None
            if key not in occupied:
                queued.rotate(-index)
                queued.popleft()
                queued.rotate(index)
                return item, key
        while not exhausted:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            key = concurrency_key(item) if concurrency_key is not None else None
            if key not in occupied:
                return item, key
            queued.append(item)
        return None

    def observe(reading: MemorySample) -> None:
        nonlocal peak_bytes, peak_active, next_report, last_state
        peak_bytes = max(peak_bytes, reading.current)
        measure(reading)
        peak_active = max(peak_active, len(pending))
        now = monotonic()
        state = (len(pending), warmed_up, admission_limited, dependency_limited)
        if report is not None and (now >= next_report or state != last_state):
            report(
                {
                    "current_bytes": reading.current,
                    "limit_bytes": reading.limit,
                    "peak_bytes": peak_bytes,
                    "active": len(pending),
                    "peak_active": peak_active,
                    "max_parallel": max_parallel,
                    "item_budget_bytes": max(
                        (estimate(kind) for kind in observed_costs),
                        default=policy.item_bytes,
                    ),
                    "reserve_bytes": policy.reserve_bytes,
                    "calibrating": int(not warmed_up),
                    "admission_limited": int(admission_limited),
                    "dependency_limited": int(dependency_limited),
                }
            )
            next_report = now + 5
            last_state = state

    try:
        while not exhausted or queued or pending:
            reading = policy.check(sample())
            measure(reading)
            slots = max_parallel if warmed_up else 1
            admission_limited = dependency_limited = False
            while len(pending) < slots:
                eligible = next_eligible()
                if eligible is None:
                    dependency_limited = bool(queued)
                    break
                item, key = eligible
                kind = work_class(item) if work_class is not None else "default"
                admission = policy.check(sample())
                measure(admission)
                # Estimates limit concurrency, not the first serial operation.
                # The enclosing process watchdog still enforces the reserve.
                required = estimate(kind) + sum(
                    estimate(item.work_class) for item in pending.values()
                )
                if (
                    pending
                    and admission.current + required
                    > admission.limit - policy.reserve_bytes
                ):
                    admission_limited = True
                    queued.appendleft(item)
                    break
                future = cast(
                    Future[R], pool.submit(copy_context().run, function, item)
                )
                pending[future] = _InFlight(key, kind, admission.current)
            observe(policy.check(sample()))
            if not pending:
                break
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            measure(policy.check(sample()))
            for future in done:
                finished = pending.pop(future)
                measured.add(finished.work_class)
                yield future.result()
                warmed_up = True
        observe(policy.check(sample()))
    finally:
        for future in pending:
            future.cancel()
        # Only the enclosing supervised process may forcibly interrupt a call.
        pool.shutdown(wait=True, cancel_futures=True)
