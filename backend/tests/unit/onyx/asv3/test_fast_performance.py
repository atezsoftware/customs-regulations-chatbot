"""Host timing must preserve the exact operation and cancellation semantics."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import performance
from onyx.asv3.models import RunContext
from onyx.asv3.performance import FastCancellationProbe, profile_fast_step


@dataclass
class Operation:
    context: RunContext

    @profile_fast_step("asv3.host_test")
    def calculate(self, number: int, *, fail: bool = False) -> int:
        if fail:
            raise ValueError("operation failed")
        return number * 2


@pytest.mark.parametrize("flag", [None, False, True, 1, "true"])
def test_host_timing_requires_strict_optin_and_preserves_result_and_error(
    monkeypatch: pytest.MonkeyPatch, flag: object
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    operation = Operation(RunContext(services={"asv3_fast": flag}))
    assert operation.calculate(21) == 42
    with pytest.raises(ValueError, match="operation failed"):
        operation.calculate(21, fail=True)
    assert capture.call_count == (2 if flag is True else 0)
    if flag is True:
        capture.assert_called_with("asv3.host_test", {})
        assert capture.return_value.__exit__.call_args.args[0] is ValueError


def test_cancellation_probe_checks_every_time_including_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.75])
    monkeypatch.setattr(performance, "perf_counter", lambda: next(clock))
    check = MagicMock(side_effect=[False, True, RuntimeError("cache unavailable")])
    probe = FastCancellationProbe(check)
    assert probe() is False
    assert probe() is True
    with pytest.raises(RuntimeError, match="cache unavailable"):
        probe()
    assert check.call_count == 3
    assert probe.snapshot() == {
        "count": 3,
        "cumulative_seconds": 1.5,
        "maximum_seconds": 0.75,
        "elapsed_is_cumulative_not_wall": True,
        "cached_results": 0,
    }


def test_parallel_cancellation_checks_are_never_coalesced() -> None:
    check = MagicMock(return_value=False)
    probe = FastCancellationProbe(check)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda _: probe(), range(100))) == [False] * 100
    assert check.call_count == probe.snapshot()["count"] == 100
