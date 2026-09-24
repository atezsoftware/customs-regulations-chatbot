"""Conservative admission for independent matching inside a supervised process."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import TypeVar, cast

MIB = 1024 * 1024
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


def bounded_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    sample: Callable[[], MemorySample | None] = read_memory,
    policy: MemoryPolicy = MemoryPolicy(),
    max_parallel: int = 2,
    report: Callable[[dict[str, int]], None] | None = None,
) -> Iterator[R]:
    """Warm up serially, then admit bounded calls using fresh container samples."""
    if max_parallel not in (1, 2, 3, 4):
        raise ValueError("Matching concurrency must be between one and four")
    iterator = iter(items)
    pending: dict[Future[R], None] = {}
    queued: deque[T] = deque()
    warmed_up = False
    exhausted = False
    warmup_baseline: int | None = None
    warmup_peak = 0
    peak_bytes = peak_active = 0
    next_report = 0.0
    last_active = -1
    pool = ThreadPoolExecutor(
        max_workers=max_parallel, thread_name_prefix="amendment-match"
    )

    def observe(reading: MemorySample) -> None:
        nonlocal peak_bytes, peak_active, next_report, last_active
        peak_bytes = max(peak_bytes, reading.current)
        peak_active = max(peak_active, len(pending))
        now = monotonic()
        if report is not None and (now >= next_report or len(pending) != last_active):
            report(
                {
                    "current_bytes": reading.current,
                    "limit_bytes": reading.limit,
                    "peak_bytes": peak_bytes,
                    "active": len(pending),
                    "peak_active": peak_active,
                    "max_parallel": max_parallel,
                    "item_budget_bytes": policy.item_bytes,
                    "reserve_bytes": policy.reserve_bytes,
                }
            )
            next_report = now + 5
            last_active = len(pending)

    try:
        while not exhausted or pending:
            reading = policy.check(sample())
            if not warmed_up:
                warmup_peak = max(warmup_peak, reading.current)
            slots = max_parallel if warmed_up else 1
            while not exhausted and len(pending) < slots:
                if not queued:
                    try:
                        queued.append(next(iterator))
                    except StopIteration:
                        exhausted = True
                        break
                if not policy.admit(sample(), active=len(pending)):
                    if not pending:
                        raise ResourcePressure("insufficient_instruction_headroom")
                    break
                item = queued.popleft()
                if warmup_baseline is None:
                    warmup_baseline = reading.current
                future = cast(
                    Future[R], pool.submit(copy_context().run, function, item)
                )
                pending[future] = None
            observe(policy.check(sample()))
            if not pending:
                break
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                del pending[future]
                yield future.result()
                if not warmed_up and warmup_baseline is not None:
                    growth = max(0, warmup_peak - warmup_baseline)
                    policy = replace(
                        policy,
                        item_bytes=max(
                            min(policy.item_bytes, 256 * MIB), growth * 3 // 2
                        ),
                    )
                warmed_up = True
        observe(policy.check(sample()))
    finally:
        for future in pending:
            future.cancel()
        # Only the enclosing supervised process may forcibly interrupt a call.
        pool.shutdown(wait=True, cancel_futures=True)
