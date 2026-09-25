import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from onyx.regulatory.amendments.memory_budget import (
    MemoryPolicy,
    MemorySample,
    ResourcePressure,
)
from onyx.regulatory.amendments.supervision import supervise


def test_child_pool_supports_nested_parallel_reads_and_releases_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from contextvars import copy_context
    from functools import partial
    from threading import Event, Lock

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import TimeoutError as PoolTimeout
    from sqlalchemy.pool import QueuePool

    from onyx.context.search.retrieval.concurrency import search_slot
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.regulatory.amendments import job, supervision
    from onyx.utils import variable_functionality as versioning

    engine: Engine | None = None

    def initialize(*, pool_size: int, max_overflow: int, **_kwargs: object) -> None:
        nonlocal engine
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=0.1,
        )

    def run_batch(**kwargs: object) -> None:
        assert engine is not None
        runner = kwargs["instruction_runner"]
        assert isinstance(runner, partial)
        assert runner.keywords["max_parallel"] == 10
        analysis_engine = engine
        full = Event()
        release = Event()
        lock = Lock()
        active = peak = 0

        def read(_index: int) -> int:
            nonlocal active, peak
            with search_slot():
                with (
                    analysis_engine.connect(),
                    analysis_engine.connect() as publication,
                ):
                    with lock:
                        active += 1
                        peak = max(peak, active)
                        if active == 12:
                            full.set()
                    try:
                        assert release.wait(5)
                        return publication.execute(text("SELECT 1")).scalar_one()
                    finally:
                        with lock:
                            active -= 1

        with ExitStack() as outer, ThreadPoolExecutor(max_workers=50) as workers:
            for _ in range(10):
                outer.enter_context(analysis_engine.connect())
            futures = [
                workers.submit(copy_context().run, read, index) for index in range(50)
            ]
            try:
                assert full.wait(3)
                assert isinstance(analysis_engine.pool, QueuePool)
                assert analysis_engine.pool.checkedout() == 34
            finally:
                release.set()
            assert [future.result(timeout=5) for future in futures] == [1] * 50
        assert peak == 12
        pool = analysis_engine.pool
        assert isinstance(pool, QueuePool)
        assert pool.checkedout() == 0
        assert pool.overflow() == 0
        assert pool.checkedin() < 40  # Burst connections are closed on return.
        with ExitStack() as connections:
            for _ in range(48):
                connections.enter_context(analysis_engine.connect())
            with pytest.raises(PoolTimeout):
                analysis_engine.connect()
        assert pool.checkedout() == 0

    monkeypatch.setattr(
        supervision, "read_memory", lambda: MemorySample(1024**3, 5 * 1024**3)
    )
    monkeypatch.setattr(versioning, "set_is_ee_based_on_env_variable", lambda: None)
    monkeypatch.setattr(supervision, "protect_parent_lifetime", lambda: None)
    monkeypatch.setattr(SqlEngine, "reset_engine", lambda: None)
    monkeypatch.setattr(SqlEngine, "set_app_name", lambda _name: None)
    monkeypatch.setattr(SqlEngine, "init_engine", initialize)
    monkeypatch.setattr(job, "run_amendment_batch", run_batch)
    monkeypatch.setattr(sys, "argv", ["supervision", "189", "1", "public", "parallel"])
    try:
        supervision._child_main()
    finally:
        if engine is not None:
            engine.dispose()


@pytest.mark.parametrize("current_mib", [2800, 3084])
def test_serial_child_phases_use_actual_reserve(
    monkeypatch: pytest.MonkeyPatch, current_mib: int
) -> None:
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.regulatory.amendments import job, supervision
    from onyx.utils import variable_functionality as versioning

    completed: list[str] = []

    def run_batch(*, before_work: Callable[[], None], **_kwargs: object) -> None:
        for phase in ("segmentation", "drafting"):
            before_work()
            completed.append(phase)

    monkeypatch.setattr(versioning, "set_is_ee_based_on_env_variable", lambda: None)
    monkeypatch.setattr(supervision, "protect_parent_lifetime", lambda: None)
    monkeypatch.setattr(
        supervision,
        "read_memory",
        lambda: MemorySample(current_mib * 1024**2, 3584 * 1024**2),
    )
    monkeypatch.setattr(SqlEngine, "reset_engine", lambda: None)
    monkeypatch.setattr(SqlEngine, "set_app_name", lambda _name: None)
    monkeypatch.setattr(SqlEngine, "init_engine", lambda **_kwargs: None)
    monkeypatch.setattr(job, "run_amendment_batch", run_batch)
    monkeypatch.setattr(sys, "argv", ["supervision", "188", "1", "public", "parallel"])
    if current_mib == 3084:
        with pytest.raises(SystemExit) as error:
            supervision._child_main()
        assert error.value.code == 75
        assert completed == []
    else:
        supervision._child_main()
        assert completed == ["segmentation", "drafting"]


@pytest.mark.parametrize(
    "paid,licensed", [(True, False), (False, True), (False, False)]
)
def test_child_initializes_configured_secret_decryption(
    monkeypatch: pytest.MonkeyPatch, paid: bool, licensed: bool
) -> None:
    from ee.onyx.utils import encryption as encrypted_storage
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.regulatory.amendments import job, supervision
    from onyx.utils import variable_functionality as versioning
    from onyx.utils.encryption import decrypt_bytes_to_string

    monkeypatch.setattr(versioning, "global_version", versioning.OnyxVersion())
    monkeypatch.setattr(versioning, "ENTERPRISE_EDITION_ENABLED", paid)
    monkeypatch.setattr(versioning, "_LICENSE_ENFORCEMENT_ENABLED", licensed)
    versioning.fetch_versioned_implementation.cache_clear()
    key = "synthetic-test-key-for-child-only"
    value = "synthetic-embedding-credential"
    # Make raw UTF-8 decoding fail deterministically, as for encrypted DEV data.
    monkeypatch.setattr(encrypted_storage, "urandom", lambda size: b"\xde" * size)
    stored = (
        encrypted_storage._encrypt_string(value, key=key)
        if paid or licensed
        else value.encode()
    )
    observed: list[str] = []

    def run_batch(**_kwargs: object) -> None:
        observed.append(decrypt_bytes_to_string(stored, key=key))

    monkeypatch.setattr(supervision, "protect_parent_lifetime", lambda: None)
    monkeypatch.setattr(SqlEngine, "reset_engine", lambda: None)
    monkeypatch.setattr(SqlEngine, "set_app_name", lambda _name: None)
    monkeypatch.setattr(SqlEngine, "init_engine", lambda **_kwargs: None)
    monkeypatch.setattr(job, "run_amendment_batch", run_batch)
    monkeypatch.setattr(sys, "argv", ["supervision", "161", "25", "public", "parallel"])
    try:
        supervision._child_main()
        assert observed == [value]
    finally:
        versioning.fetch_versioned_implementation.cache_clear()


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


def test_serial_start_does_not_require_speculative_parallel_headroom(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "started"
    supervise(
        [sys.executable, "-c", f'open({str(marker)!r}, "w").close()'],
        sample=lambda: MemorySample(2800 * 1024**2, 3584 * 1024**2),
        lock_path=tmp_path / "lock",
    )
    assert marker.exists()


def test_start_at_safety_reserve_never_launches_child(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    with pytest.raises(ResourcePressure, match="memory_pressure"):
        supervise(
            [sys.executable, "-c", f'open({str(marker)!r}, "w").close()'],
            sample=lambda: MemorySample(3084 * 1024**2, 3584 * 1024**2),
            lock_path=tmp_path / "lock",
        )
    assert not marker.exists()


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
