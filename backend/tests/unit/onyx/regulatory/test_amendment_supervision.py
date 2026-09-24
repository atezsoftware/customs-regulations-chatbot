import sys
from pathlib import Path

import pytest

from onyx.regulatory.amendments.memory_budget import (
    MemoryPolicy,
    MemorySample,
    ResourcePressure,
)
from onyx.regulatory.amendments.supervision import supervise


def test_missing_telemetry_does_not_launch_a_child(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    with pytest.raises(ResourcePressure):
        supervise(
            [sys.executable, "-c", f'open({str(marker)!r}, "w").close()'],
            sample=lambda: None,
            lock_path=tmp_path / "lock",
        )
    assert not marker.exists()


def test_success_releases_container_slot(tmp_path: Path) -> None:
    for _ in range(2):
        supervise(
            [sys.executable, "-c", "pass"],
            sample=lambda: MemorySample(10, 10000),
            lock_path=tmp_path / "lock",
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )


def test_dev_baseline_can_launch_analysis_with_reserved_headroom(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "started"
    supervise(
        [sys.executable, "-c", f'open({str(marker)!r}, "w").close()'],
        sample=lambda: MemorySample(2087477248, 3758096384),
        lock_path=tmp_path / "lock",
    )
    assert marker.exists()


def test_memory_spike_terminates_only_the_owned_child(tmp_path: Path) -> None:
    calls = 0

    def sample() -> MemorySample:
        nonlocal calls
        calls += 1
        return MemorySample(10 if calls < 3 else 9900, 10000)

    with pytest.raises(ResourcePressure) as error:
        supervise(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            sample=sample,
            lock_path=tmp_path / "lock",
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    assert error.value.started
    # The slot must also be released following interruption.
    supervise(
        [sys.executable, "-c", "pass"],
        sample=lambda: MemorySample(10, 10000),
        lock_path=tmp_path / "lock",
        policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
    )


def test_ownership_loss_stops_child_without_restarting(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ownership"):
        supervise(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            sample=lambda: MemorySample(10, 10000),
            owned=lambda: False,
            lock_path=tmp_path / "lock",
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )


def test_existing_container_slot_cannot_be_claimed_twice(tmp_path: Path) -> None:
    import fcntl

    with (tmp_path / "lock").open("w") as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ResourcePressure, match="analysis_slot_busy"):
            supervise(
                [sys.executable, "-c", "pass"],
                sample=lambda: MemorySample(10, 10000),
                lock_path=tmp_path / "lock",
                policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
            )


def test_slow_ownership_query_cannot_block_memory_watchdog(tmp_path: Path) -> None:
    from threading import Event
    from time import sleep

    probing = Event()
    calls = 0
    marker = tmp_path / "unsafe_continuation"

    def owned() -> bool:
        nonlocal calls
        calls += 1
        if calls > 1:
            probing.set()
            sleep(0.8)
        return True

    def sample() -> MemorySample:
        return MemorySample(9900 if probing.is_set() else 10, 10000)

    with pytest.raises(ResourcePressure):
        supervise(
            [
                sys.executable,
                "-c",
                f"import time; time.sleep(.5); open({str(marker)!r}, 'w').close(); time.sleep(2)",
            ],
            owned=owned,
            sample=sample,
            lock_path=tmp_path / "lock",
            policy=MemoryPolicy(reserve_bytes=100, item_bytes=100),
        )
    assert not marker.exists()
