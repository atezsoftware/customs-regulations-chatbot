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
INITIAL_ANALYSIS_PARALLELISM = 4
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


@dataclass
class _InFlight:
    key: Hashable | None
    work_class: Hashable
    baseline: int
    isolated: bool


def bounded_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    sample: Callable[[], MemorySample | None] = read_memory,
    policy: MemoryPolicy = MemoryPolicy(),
    max_parallel: int = 2,
    initial_parallel: int = INITIAL_ANALYSIS_PARALLELISM,
    report: Callable[[dict[str, int]], None] | None = None,
    concurrency_key: Callable[[T], Hashable] | None = None,
    work_class: Callable[[T], Hashable] | None = None,
) -> Iterator[R]:
    """Start a bounded cohort, then refill using fresh container samples."""
    if (
        type(max_parallel) is not int
        or not 1 <= max_parallel <= MAX_ANALYSIS_PARALLELISM
    ):
        raise ValueError("Matching concurrency must be between one and ten")
    if (
        type(initial_parallel) is not int
        or not 1 <= initial_parallel <= MAX_ANALYSIS_PARALLELISM
    ):
        raise ValueError("Initial concurrency must be between one and ten")
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
    cohort_baseline: int | None = None
    shared_growth_budget = 0
    pool = ThreadPoolExecutor(
        max_workers=max_parallel, thread_name_prefix="amendment-match"
    )

    def estimate(kind: Hashable) -> int:
        return max(
            observed_costs.get(kind, floor),
            floor if kind in measured else policy.item_bytes,
        )

    def measure(reading: MemorySample) -> None:
        nonlocal shared_growth_budget
        item = next(iter(pending.values())) if len(pending) == 1 else None
        if item is not None and item.isolated:
            growth = max(0, reading.current - item.baseline)
            observed_costs[item.work_class] = max(
                observed_costs.get(item.work_class, floor), growth * 3 // 2
            )
        if cohort_baseline is not None:
            growth = max(0, reading.current - cohort_baseline)
            # Unattributed growth belongs to the group, not to every task.
            shared_growth_budget = max(
                shared_growth_budget,
                growth * 3 // 2
                - sum(estimate(item.work_class) for item in pending.values()),
            )

    def remaining_growth(reading: MemorySample) -> int:
        reserved = sum(estimate(item.work_class) for item in pending.values())
        materialized = (
            max(0, reading.current - cohort_baseline)
            if cohort_baseline is not None
            else 0
        )
        return max(0, reserved + shared_growth_budget - materialized)

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
                    "remaining_growth_bytes": remaining_growth(reading),
                    "shared_growth_budget_bytes": shared_growth_budget,
                }
            )
            next_report = now + 5
            last_state = state

    try:
        while not exhausted or queued or pending:
            reading = policy.check(sample())
            measure(reading)
            slots = max_parallel if warmed_up else min(initial_parallel, max_parallel)
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
                required = estimate(kind) + remaining_growth(admission)
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
                isolated = not pending
                if isolated:
                    cohort_baseline = admission.current
                    shared_growth_budget = 0
                else:
                    for running in pending.values():
                        running.isolated = False
                pending[future] = _InFlight(key, kind, admission.current, isolated)
            observe(policy.check(sample()))
            if not pending:
                break
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            reading = policy.check(sample())
            measure(reading)
            for future in done:
                finished = pending.pop(future)
                if finished.isolated:
                    measured.add(finished.work_class)
                # A completed task may retain memory. Discard its group's
                # growth credit rather than crediting it to surviving tasks.
                cohort_baseline = reading.current if pending else None
                if not pending:
                    shared_growth_budget = 0
                yield future.result()
                warmed_up = True
        observe(policy.check(sample()))
    finally:
        admission_limited = dependency_limited = False
        warmed_up = True
        for future in pending:
            future.cancel()
        # A failed result can leave peers finishing their saved checkpoints.
        # Keep activity current without replacing the original exception.
        while pending:
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
            try:
                reading = sample()
                if reading is not None:
                    observe(reading)
            except Exception:
                pass
        try:
            reading = sample()
            if reading is not None:
                observe(reading)
        except Exception:
            pass
        # Only the enclosing supervised process may forcibly interrupt a call.
        pool.shutdown(wait=True, cancel_futures=True)
