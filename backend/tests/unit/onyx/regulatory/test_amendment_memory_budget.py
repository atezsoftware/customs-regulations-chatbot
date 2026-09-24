from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from onyx.regulatory.amendments.memory_budget import (
    MemoryPolicy,
    MemorySample,
    ResourcePressure,
    bounded_map,
    read_memory,
)


def test_budget_reserves_future_growth_and_stops_before_limit() -> None:
    policy = MemoryPolicy(reserve_bytes=100, item_bytes=150)
    assert policy.admit(MemorySample(200, 1000), active=0)
    assert not policy.admit(MemorySample(450, 1000), active=1)
    with pytest.raises(ResourcePressure):
        policy.check(MemorySample(810, 1000))
    with pytest.raises(ResourcePressure):
        policy.check(None)


def test_cgroup_reader_rejects_unbounded_or_inconsistent_measurements(
    tmp_path: Path,
) -> None:
    (tmp_path / "memory.current").write_text("123")
    (tmp_path / "memory.max").write_text("1000")
    assert read_memory(tmp_path) == MemorySample(123, 1000)
    (tmp_path / "memory.max").write_text("max")
    assert read_memory(tmp_path) is None
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("-1")
    assert read_memory(tmp_path) is None


def test_bounded_parallelism_preserves_context_and_runs_every_item_once() -> None:
    tenant = ContextVar("test_tenant", default="wrong")
    tenant.set("tenant_a")
    running = peak = 0
    completed: list[int] = []
    lock = Lock()

    def work(value: int) -> int:
        nonlocal running, peak
        assert tenant.get() == "tenant_a"
        with lock:
            running += 1
            peak = max(peak, running)
        sleep(0.015)
        with lock:
            running -= 1
            completed.append(value)
        return value

    results = list(
        bounded_map(
            work,
            list(range(7)),
            sample=lambda: MemorySample(100, 1000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    )
    assert sorted(results) == list(range(7))
    assert sorted(completed) == list(range(7))
    assert peak == 2
    assert completed[0] == 0  # warm up with one task


def test_unknown_memory_never_starts_work() -> None:
    started: list[int] = []
    with pytest.raises(ResourcePressure):
        list(bounded_map(lambda n: started.append(n), [1, 2], sample=lambda: None))
    assert started == []


def test_small_budget_stays_serial_without_dropping_work() -> None:
    seen: list[int] = []
    assert list(
        bounded_map(
            lambda n: seen.append(n) or n,
            [1, 2, 3],
            sample=lambda: MemorySample(450, 1000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=150),
        )
    ) == [1, 2, 3]
    assert seen == [1, 2, 3]


def test_completed_last_item_does_not_reserve_memory_for_nonexistent_work() -> None:
    current = 100

    def work(value: int) -> int:
        nonlocal current
        current = 650
        return value

    assert list(
        bounded_map(
            work,
            [1],
            sample=lambda: MemorySample(current, 1000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    ) == [1]


def test_warmup_peak_can_keep_later_instructions_serial() -> None:
    current = 100
    running = peak = 0
    lock = Lock()

    def work(value: int) -> int:
        nonlocal current, running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        if value == 0:
            current = 500
            sleep(0.25)
            current = 100
        else:
            sleep(0.03)
        with lock:
            running -= 1
        return value

    assert sorted(
        bounded_map(
            work,
            range(4),
            sample=lambda: MemorySample(current, 1000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    ) == [0, 1, 2, 3]
    assert peak == 1
