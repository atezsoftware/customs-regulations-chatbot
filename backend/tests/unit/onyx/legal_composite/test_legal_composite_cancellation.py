import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_composite.cancellation import PollingCancellation


def test_24_workers_share_one_remote_observation_for_24000_checks() -> None:
    clock = MagicMock(return_value=100.0)
    callback = MagicMock(return_value=False)
    probe = PollingCancellation(callback, clock=clock)
    start = threading.Barrier(24)

    def check_worker() -> None:
        start.wait(timeout=5)
        for _ in range(1000):
            assert not probe()

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(check_worker) for _ in range(24)]
        for future in futures:
            future.result(timeout=5)
    callback.assert_called_once_with()
    assert probe.snapshot() == {
        "checks": 24_000,
        "remote_observations": 1,
        "remote_total_seconds": 0.0,
        "remote_max_seconds": 0.0,
        "cancelled": False,
        "error_count": 0,
    }


def test_interval_bounds_remote_checks_and_observed_cancellation_stays_true() -> None:
    clock = MagicMock(return_value=0.0)
    callback = MagicMock(side_effect=[False, False, False, True])
    probe = PollingCancellation(callback, clock=clock)
    for instant in (0.0, 0.249, 0.25, 0.499, 0.5, 0.749):
        clock.return_value = instant
        assert not probe()
    clock.return_value = 0.75
    assert probe()
    assert callback.call_count == 4
    for instant in (0.8, 2.0, 100.0):
        clock.return_value = instant
        assert probe()
    assert callback.call_count == 4


def test_concurrent_due_probe_calls_the_remote_observer_once() -> None:
    clock = MagicMock(return_value=0.0)
    entered, release, second_started = (threading.Event() for _ in range(3))
    calls = 0

    def observe() -> bool:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return True

    probe = PollingCancellation(observe, clock=clock)

    def follower() -> bool:
        second_started.set()
        return probe()

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(probe)
        try:
            assert entered.wait(timeout=5)
            waiting = executor.submit(follower)
            assert second_started.wait(timeout=5)
            assert calls == 1
        finally:
            release.set()
        assert owner.result(timeout=5)
        assert waiting.result(timeout=5)
    assert calls == 1


def test_observer_failure_is_raised_until_next_retry() -> None:
    clock = MagicMock(return_value=0.0)
    failure = ConnectionError("Cancellation observer unavailable")
    callback = MagicMock(side_effect=[False, failure, True])
    probe = PollingCancellation(callback, clock=clock)
    assert not probe()
    clock.return_value = 0.25
    with pytest.raises(ConnectionError, match="observer unavailable"):
        probe()
    clock.return_value = 0.49
    with pytest.raises(ConnectionError, match="observer unavailable"):
        probe()
    assert callback.call_count == 2
    clock.return_value = 0.5
    assert probe()
    assert callback.call_count == 3


def test_slow_observer_retains_a_full_interval_after_completion() -> None:
    clock = MagicMock(return_value=0.0)

    def observe() -> bool:
        clock.return_value += 1.0
        return False

    callback = MagicMock(side_effect=observe)
    probe = PollingCancellation(callback, clock=clock)
    assert not probe()
    clock.return_value = 1.249
    assert not probe()
    callback.assert_called_once_with()
    clock.return_value = 1.25
    assert not probe()
    assert callback.call_count == 2


def test_local_context_deadlines_are_checked_without_another_remote_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MagicMock(return_value=0.0)
    callback = MagicMock(return_value=False)
    probe = PollingCancellation(callback, clock=clock)
    context = RunContext(deadline=10.0, research_deadline=9.0, cancelled=probe)
    monotonic = MagicMock(return_value=1.0)
    monkeypatch.setattr(time, "monotonic", monotonic)
    context.check_research_active()
    monotonic.return_value = 9.0
    with pytest.raises(RunStopped, match="finalization time retained"):
        context.check_research_active()
    monotonic.return_value = 10.0
    with pytest.raises(RunStopped, match="Research deadline exceeded"):
        context.check_active()
    callback.assert_called_once_with()


def test_polling_state_does_not_cross_requests() -> None:
    callback = MagicMock(side_effect=[True, False])
    assert PollingCancellation(callback)()
    assert not PollingCancellation(callback)()
    assert callback.call_count == 2


def test_snapshot_measures_remote_io_but_excludes_cached_checks() -> None:
    clock = MagicMock(return_value=10.0)
    values = iter([(0.125, False), (0.5, True)])

    def observe() -> bool:
        elapsed, cancelled = next(values)
        clock.return_value += elapsed
        return cancelled

    probe = PollingCancellation(observe, clock=clock)
    assert not probe()
    clock.return_value = 10.2
    assert not probe()
    clock.return_value = 10.375
    assert probe()
    clock.return_value = 100.0
    assert probe()
    assert probe.snapshot() == {
        "checks": 4,
        "remote_observations": 2,
        "remote_total_seconds": 0.625,
        "remote_max_seconds": 0.5,
        "cancelled": True,
        "error_count": 0,
    }


def test_snapshot_counts_failed_remote_observations_without_error_text() -> None:
    clock = MagicMock(return_value=0.0)
    callback = MagicMock(side_effect=[ConnectionError("private endpoint"), False])
    probe = PollingCancellation(callback, clock=clock)
    with pytest.raises(ConnectionError):
        probe()
    with pytest.raises(ConnectionError):
        probe()
    assert probe.snapshot() == {
        "checks": 2,
        "remote_observations": 1,
        "remote_total_seconds": 0.0,
        "remote_max_seconds": 0.0,
        "cancelled": False,
        "error_count": 1,
    }
    clock.return_value = 0.25
    assert not probe()
    assert probe.snapshot()["remote_observations"] == 2
    assert probe.snapshot()["error_count"] == 1
    assert "private endpoint" not in str(probe.snapshot())


@pytest.mark.parametrize("interval", [0.0, -0.25, float("inf"), float("nan"), True])
def test_invalid_polling_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PollingCancellation(lambda: False, interval_seconds=interval)
