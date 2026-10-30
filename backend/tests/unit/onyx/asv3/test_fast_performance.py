"""Host timing must preserve the exact operation and cancellation semantics."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import FunctionType
from typing import Callable, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import performance
from onyx.asv3.models import RunContext
from onyx.asv3.performance import (
    FAST_TIMING_SERVICE,
    FastCancellationProbe,
    FastTimingCollector,
    fast_timing_run,
    fast_timing_step,
    install_fast_timing,
    profile_fast_step,
)


@dataclass
class Operation:
    context: RunContext

    @profile_fast_step("asv3.model_request")
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
    with fast_timing_run(flag is True):
        assert operation.calculate(21) == 42
        with pytest.raises(ValueError, match="operation failed"):
            operation.calculate(21, fail=True)
        assert capture.call_count == 0
    assert capture.call_count == (1 if flag is True else 0)
    if flag is True:
        capture.assert_called_with("asv3.fast_host_timing", {})
        payload = capture.return_value.__enter__.return_value.output_value
        assert payload["operations"][0]["count"] == 2
        assert payload["operations"][0]["errors"] == 1
        assert payload["intervals"][1]["exception_type"] == "ValueError"
    else:
        assert FAST_TIMING_SERVICE not in operation.context.services


def test_collector_bounds_parallel_intervals_but_keeps_all_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    collector = FastTimingCollector(max_intervals=3)
    context = RunContext(services={"asv3_fast": True, FAST_TIMING_SERVICE: collector})

    def call(number: int) -> None:
        with fast_timing_step(context, "asv3.context_preparation") as step:
            assert step is not None
            step.output_value = {"sequence": number}

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(call, range(100)))
    assert capture.call_count == 0
    collector.flush()
    collector.flush()
    assert capture.call_count == 1
    payload = capture.return_value.__enter__.return_value.output_value
    assert len(payload["intervals"]) == 3
    assert payload["dropped_intervals"] == 97
    assert payload["truncated"] is True
    assert payload["unfinished_intervals"] == 0
    assert payload["operations"][0]["count"] == 100
    assert all(not row["root_thread"] for row in payload["intervals"])
    assert collector._intervals == [] and collector._totals == {}


def test_timing_metadata_never_retains_payloads_names_or_error_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrivateFailure(Exception):
        pass

    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    context = RunContext(services={"asv3_fast": True})
    secret = "PRIVATE_SOURCE_REASONING_AND_ERROR"
    with fast_timing_run(True):
        with pytest.raises(PrivateFailure, match=secret):
            with fast_timing_step(
                context, secret, {"query_count": 2, "payload": secret}
            ) as step:
                assert step is not None
                step.output_value = {
                    "sequence": 7,
                    "write_seconds": float("nan"),
                    "source": secret,
                }
                step.output_value["reasoning"] = secret
                raise PrivateFailure(secret)
    payload = capture.return_value.__enter__.return_value.output_value
    assert secret not in json.dumps(payload)
    interval = payload["intervals"][0]
    assert interval["operation"] == "asv3.other"
    assert interval["exception_type"] == "other"
    assert interval["metrics"] == {"query_count": 2, "sequence": 7}
    assert payload["rejected_metadata_fields"] == 4


def test_run_setup_failure_flushes_once_and_capture_failure_never_masks_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock(side_effect=RuntimeError("capture failed"))
    monkeypatch.setattr(performance, "graph_step", capture)
    with pytest.raises(ValueError, match="original failure"):
        with fast_timing_run(True):
            raise ValueError("original failure")
    assert capture.call_count == 1
    assert performance._ACTIVE_TIMING.get() is None


def test_run_failure_records_only_exception_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    with pytest.raises(ValueError, match="private failure"):
        with fast_timing_run(True):
            raise ValueError("private failure")
    payload = capture.return_value.__enter__.return_value.output_value
    assert payload["exception_type"] == "ValueError"
    assert "private failure" not in json.dumps(payload)
    assert payload["intervals"] == []


def test_closed_collector_does_not_reopen_or_retain_late_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    context = RunContext(services={"asv3_fast": True})
    collector = install_fast_timing(context)
    assert collector is not None
    collector.flush()
    assert Operation(context).calculate(9) == 18
    assert context.services[FAST_TIMING_SERVICE] is collector
    assert collector._intervals == [] and collector._totals == {}
    assert capture.call_count == 1


def test_flush_marks_unfinished_interval_and_discards_its_late_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    context = RunContext(services={"asv3_fast": True})
    collector = install_fast_timing(context)
    assert collector is not None
    with fast_timing_step(context, "asv3.model_request"):
        collector.flush()
    payload = capture.return_value.__enter__.return_value.output_value
    assert payload["unfinished_intervals"] == 1
    assert payload["truncated"] is True
    assert payload["intervals"] == []
    assert collector._intervals == [] and collector._totals == {}
    collector.flush()
    assert capture.call_count == 1


def test_collector_cannot_capture_another_run() -> None:
    owner = RunContext(services={"asv3_fast": True})
    collector = install_fast_timing(owner)
    assert collector is not None
    other = RunContext(services={FAST_TIMING_SERVICE: collector})
    assert Operation(other).calculate(4) == 8
    assert collector._totals == {}


def test_run_flush_includes_fresh_cancellation_snapshot_without_rechecking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    monkeypatch.setattr(performance, "graph_step", capture)
    check = MagicMock(return_value=False)
    probe = FastCancellationProbe(check)
    context = RunContext(services={"asv3_fast": True})
    with fast_timing_run(True):
        install_fast_timing(context, cancellation_probe=probe)
        assert probe() is False
    assert check.call_count == 1
    payload = capture.return_value.__enter__.return_value.output_value
    assert payload["cancellation_checks"]["count"] == 1


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
        "locations": [
            {
                "module": "external",
                "function": "external",
                "root_thread": True,
                "count": 3,
                "cumulative_seconds": 1.5,
                "maximum_seconds": 0.75,
            }
        ],
    }


def test_parallel_cancellation_checks_are_never_coalesced() -> None:
    check = MagicMock(return_value=False)
    probe = FastCancellationProbe(check)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda _: probe(), range(100))) == [False] * 100
    assert check.call_count == probe.snapshot()["count"] == 100
    assert probe.snapshot()["locations"] == [
        {
            "module": "external",
            "function": "external",
            "root_thread": False,
            "count": 100,
            "cumulative_seconds": probe.snapshot()["cumulative_seconds"],
            "maximum_seconds": probe.snapshot()["maximum_seconds"],
        }
    ]


def test_cancellation_location_skips_context_wrappers_without_inspecting_values() -> (
    None
):
    def check_context(context: RunContext) -> None:
        context.check_research_active()

    call = cast(
        Callable[[RunContext], None],
        FunctionType(check_context.__code__, {"__name__": "onyx.asv3.example"}),
    )
    probe = FastCancellationProbe(lambda: False)
    call(RunContext(cancelled=probe))
    locations = probe.snapshot()["locations"]
    assert isinstance(locations, list) and len(locations) == 1
    location = locations[0]
    assert isinstance(location, dict)
    assert location["module"] == "onyx.asv3.example"
    assert location["function"] == check_context.__qualname__
    assert location["root_thread"] is True
    assert location["count"] == 1
    assert set(location) == {
        "module",
        "function",
        "root_thread",
        "count",
        "cumulative_seconds",
        "maximum_seconds",
    }
