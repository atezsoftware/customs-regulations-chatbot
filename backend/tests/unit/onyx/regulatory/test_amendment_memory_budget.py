from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from onyx.regulatory.amendments.memory_budget import (
    MIB,
    MemoryPolicy,
    MemorySample,
    ResourcePressure,
    bounded_map,
    read_memory,
)


def test_budget_reserves_future_growth_and_stops_before_limit() -> None:
    policy = MemoryPolicy(reserve_bytes=100, item_bytes=150)
    assert policy.admit(MemorySample(200, 1000), active=0)
    assert policy.admit(MemorySample(700, 1000), active=0)
    assert not policy.admit(MemorySample(700, 1000), active=1)
    assert policy.check(MemorySample(899, 1000))
    with pytest.raises(ResourcePressure):
        policy.check(MemorySample(900, 1000))
    with pytest.raises(ResourcePressure):
        policy.check(None)


def test_dev_budget_uses_available_memory_but_retains_500_mib() -> None:
    policy = MemoryPolicy()
    # Observed DEV cgroup usage before the analysis process was launched.
    assert policy.admit(MemorySample(2087477248, 3584 * MIB), active=1)
    # After process setup, one instruction fits but two do not.
    assert policy.admit(MemorySample(2500 * MIB, 3584 * MIB), active=0)
    assert not policy.admit(MemorySample(2500 * MIB, 3584 * MIB), active=1)
    assert policy.check(MemorySample(3084 * MIB - 1, 3584 * MIB))
    with pytest.raises(ResourcePressure, match="memory_pressure"):
        policy.check(MemorySample(3084 * MIB, 3584 * MIB))


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


def test_unknown_memory_never_starts_work() -> None:
    started: list[int] = []
    with pytest.raises(ResourcePressure):
        list(bounded_map(lambda n: started.append(n), [1, 2], sample=lambda: None))
    assert started == []


def test_four_tasks_start_together_and_refill_before_slow_peers_finish() -> None:
    from threading import Event

    four_started = Event()
    replacement_started = Event()
    lock = Lock()
    started: set[int] = set()

    def work(index: int) -> int:
        with lock:
            started.add(index)
            if len(started) == 4:
                four_started.set()
        if index == 0:
            assert four_started.wait(2), "Startup waited for a serial warmup"
        elif index < 4:
            assert replacement_started.wait(3), "Freed slot did not refill"
        else:
            replacement_started.set()
        return index

    assert sorted(
        bounded_map(
            work,
            range(5),
            max_parallel=10,
            sample=lambda: MemorySample(100, 600),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    ) == list(range(5))


def test_failure_reports_surviving_tasks_until_the_pool_drains() -> None:
    from threading import Event

    peer_started = Event()
    draining_reported = Event()
    peer_observed_update: list[bool] = []
    reports: list[dict[str, int]] = []

    def work(index: int) -> int:
        if index == 1:
            assert peer_started.wait(2)
            raise ValueError("unexpected draft failure")
        if index == 2:
            peer_started.set()
            peer_observed_update.append(draining_reported.wait(2))
        return index

    def report(value: dict[str, int]) -> None:
        reports.append(value)
        if peer_started.is_set() and value["active"] == 1:
            draining_reported.set()

    with pytest.raises(ValueError, match="unexpected draft failure"):
        list(
            bounded_map(
                work,
                range(3),
                max_parallel=2,
                sample=lambda: MemorySample(100, 1000),
                policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
                report=report,
            )
        )
    assert peer_observed_update == [True]
    assert reports[-1]["active"] == 0


def test_serial_progress_with_headroom_below_estimated_item_cost() -> None:
    running = peak = 0
    lock = Lock()

    def work(value: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        sleep(0.02)
        with lock:
            running -= 1
        return value

    assert list(
        bounded_map(
            work,
            range(4),
            sample=lambda: MemorySample(2900 * MIB, 3584 * MIB),
            max_parallel=4,
        )
    ) == [0, 1, 2, 3]
    assert peak == 1


def test_matching_at_safety_reserve_never_starts_work() -> None:
    started: list[int] = []
    with pytest.raises(ResourcePressure, match="memory_pressure"):
        list(
            bounded_map(
                lambda n: started.append(n),
                [1, 2],
                sample=lambda: MemorySample(3084 * MIB, 3584 * MIB),
            )
        )
    assert started == []


def test_small_budget_stays_serial_without_dropping_work() -> None:
    seen: list[int] = []
    assert list(
        bounded_map(
            lambda n: seen.append(n) or n,
            [1, 2, 3],
            sample=lambda: MemorySample(700, 1000),
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


def test_parallelism_recovers_above_half_container_usage() -> None:
    current = 750
    running = peak = 0
    lock = Lock()

    def work(value: int) -> int:
        nonlocal current, running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        sleep(0.03)
        if value == 1:
            current = 550  # two calls now fit, without dropping below 50%
        with lock:
            running -= 1
        return value

    assert sorted(
        bounded_map(
            work,
            range(8),
            sample=lambda: MemorySample(current, 1000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    ) == list(range(8))
    assert peak == 2


def test_adaptive_scheduler_can_use_four_slots_with_cold_budgets() -> None:
    running = peak = 0
    lock = Lock()
    readings: list[dict[str, int]] = []

    def work(value: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        sleep(0.03)
        with lock:
            running -= 1
        return value

    assert sorted(
        bounded_map(
            work,
            range(9),
            sample=lambda: MemorySample(1800 * MIB, 5120 * MIB),
            max_parallel=4,
            report=readings.append,
        )
    ) == list(range(9))
    assert peak == 4
    assert readings[-1]["active"] == 0
    assert max(row["peak_active"] for row in readings) == 4
    assert all(row["current_bytes"] == 1800 * MIB for row in readings)


def test_ten_slots_refill_without_waiting_for_slow_peers() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    initial_full = Event()
    refilled = Event()
    release = Event()
    lock = Lock()
    active = peak = 0
    started: set[int] = set()

    def work(value: int) -> int:
        nonlocal active, peak
        if value == 0:
            return value
        with lock:
            active += 1
            peak = max(peak, active)
            started.add(value)
            if len(started) == 10:
                initial_full.set()
            if value == 11:
                refilled.set()
        try:
            if value == 1:
                assert initial_full.wait(3)
            else:
                assert release.wait(5)
            return value
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=1) as consumer:
        result = consumer.submit(
            lambda: list(
                bounded_map(
                    work,
                    range(12),
                    max_parallel=10,
                    sample=lambda: MemorySample(100, 5000),
                    policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
                )
            )
        )
        try:
            assert refilled.wait(4), (
                "freed slot must refill while nine peers remain blocked"
            )
            assert not release.is_set()
            assert peak == 10
        finally:
            release.set()
        assert sorted(result.result(timeout=5)) == list(range(12))


def test_same_parent_queue_is_not_reported_as_memory_pressure() -> None:
    reports: list[dict[str, int]] = []
    assert list(
        bounded_map(
            lambda value: value,
            range(4),
            max_parallel=10,
            concurrency_key=lambda _: "one-parent",
            report=reports.append,
            sample=lambda: MemorySample(100, 5000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    ) == list(range(4))
    assert all(item["admission_limited"] == 0 for item in reports)
    assert any(item["dependency_limited"] == 1 for item in reports)


def test_later_large_work_updates_the_estimate_after_light_warmup() -> None:
    current = 100
    reports: list[dict[str, int]] = []

    def work(value: int) -> int:
        nonlocal current
        if value == 1:
            current = 500
            sleep(0.25)
            current = 100
        return value

    assert list(
        bounded_map(
            work,
            range(3),
            max_parallel=1,
            sample=lambda: MemorySample(current, 2000),
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
            report=reports.append,
        )
    ) == [0, 1, 2]
    assert reports[-1]["item_budget_bytes"] >= 600


def test_new_work_class_keeps_its_cold_budget_until_measured() -> None:
    from threading import Event

    first_large_finished = Event()
    running = peak = 0
    lock = Lock()

    def work(value: int) -> int:
        nonlocal running, peak
        if value == 0:
            return value
        if value == 2:
            assert first_large_finished.is_set(), (
                "Light work calibrated an unseen large class"
            )
        with lock:
            running += 1
            peak = max(peak, running)
        sleep(0.04)
        with lock:
            running -= 1
        if value == 1:
            first_large_finished.set()
        return value

    assert sorted(
        bounded_map(
            work,
            range(4),
            max_parallel=2,
            work_class=lambda value: "provision" if value == 0 else "annex",
            sample=lambda: MemorySample(1000 * MIB, 2100 * MIB),
        )
    ) == list(range(4))
    assert peak == 2


def test_concurrent_growth_is_reserved_once_and_reaches_ten_safe_slots() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    release = Event()
    all_started = Event()
    lock = Lock()
    running = peak = 0
    reports: list[dict[str, int]] = []

    def sample() -> MemorySample:
        with lock:
            return MemorySample((1800 + 120 * running) * MIB, 5120 * MIB)

    def work(value: int) -> int:
        nonlocal running, peak
        if value == 0:
            return value
        with lock:
            running += 1
            peak = max(peak, running)
            if running == 10:
                all_started.set()
        try:
            assert release.wait(5)
            return value
        finally:
            with lock:
                running -= 1

    with ThreadPoolExecutor(max_workers=1) as consumer:
        result = consumer.submit(
            lambda: list(
                bounded_map(
                    work,
                    range(11),
                    sample=sample,
                    max_parallel=10,
                    initial_parallel=1,
                    report=reports.append,
                )
            )
        )
        try:
            assert all_started.wait(3), (
                "Ten 120 MiB tasks fit below the 500 MiB reserve; shared "
                "growth must not inflate every task's estimated cost"
            )
        finally:
            release.set()
        assert sorted(result.result(timeout=5)) == list(range(11))
    assert peak == 10
    assert max(row["item_budget_bytes"] for row in reports) <= 512 * MIB


@pytest.mark.parametrize("retained", [True, False])
def test_finished_task_memory_is_not_credited_to_surviving_tasks(
    retained: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    pair_started = Event()
    growth_observed = Event()
    waiting = Event()
    third_started = Event()
    release = Event()
    current = 100
    reports: list[dict[str, int]] = []

    def sample() -> MemorySample:
        value = current
        if value == 300:
            growth_observed.set()
        return MemorySample(value, 550)

    def report(value: dict[str, int]) -> None:
        reports.append(value)
        if value["active"] == 1 and value["admission_limited"]:
            waiting.set()

    def work(value: int) -> int:
        nonlocal current
        if value == 1:
            assert pair_started.wait(3)
            current = 300
            assert growth_observed.wait(3)
            if not retained:
                current = 100
        elif value == 2:
            pair_started.set()
            assert release.wait(5)
        elif value == 3:
            third_started.set()
        return value

    with ThreadPoolExecutor(max_workers=1) as consumer:
        result = consumer.submit(
            lambda: list(
                bounded_map(
                    work,
                    range(4),
                    sample=sample,
                    policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
                    max_parallel=2,
                    report=report,
                )
            )
        )
        try:
            if retained:
                assert waiting.wait(3)
                assert not third_started.is_set()
            else:
                assert third_started.wait(3), (
                    "Reclaimed memory should admit work before the slow peer ends"
                )
        finally:
            release.set()
        assert sorted(result.result(timeout=5)) == list(range(4))
    assert max(row["item_budget_bytes"] for row in reports) == 100
    if retained:
        blocked = [row for row in reports if row["admission_limited"]]
        assert blocked[-1]["shared_growth_budget_bytes"] == 100
        assert blocked[-1]["remaining_growth_bytes"] == 200
