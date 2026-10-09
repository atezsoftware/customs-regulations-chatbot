"""Worker timing is observational and separates originals from incomplete outcomes."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, local
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
)
from onyx.legal_composite import acquisition
from onyx.legal_composite.acquisition import CanonicalAcquirer, CanonicalEvidenceStage
from onyx.legal_composite.models import SourceAction, WorkflowPolicy
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    original,
    plan,
    read_action,
    registry,
    search_action,
)


class WorkerClock:
    def __init__(self) -> None:
        self._local = local()

    def monotonic(self) -> float:
        return getattr(self._local, "seconds", 1_000.0)

    def advance(self, seconds: float) -> None:
        self._local.seconds = self.monotonic() + seconds


class RecordedStep:
    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.summary: str | None = None
        self.output_value: object = None


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> list[RecordedStep]:
    recorded: list[RecordedStep] = []

    @contextmanager
    def graph_step(
        operation: str, _input_value: object, *, summary: str | None = None
    ) -> Iterator[RecordedStep]:
        step = RecordedStep(operation)
        step.summary = summary
        recorded.append(step)
        yield step

    monkeypatch.setattr(acquisition, "graph_step", graph_step)
    return recorded


def fixture(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
    *,
    capture: bool = True,
) -> CanonicalAcquirer:
    return CanonicalAcquirer(
        registry(handler, handler),
        RunContext(timeout_seconds=30),
        EvidenceLedger(),
        WorkflowPolicy(max_parallel_tools=2),
        capture_task_timings=capture,
    )


def dispatch(
    acquirer: CanonicalAcquirer, action: SourceAction | None = None
) -> ToolOutcome:
    action = action or read_action()
    child = acquirer.context.child()
    child.services["legal_composite_original_stage"] = CanonicalEvidenceStage(
        acquirer.ledger, child, action.need_ids
    )
    return acquirer._dispatch_action(
        action,
        CapabilityCall(name=action.tool, arguments=dict(action.arguments)),
        child,
    )


def found(text: str = "Fictional whole canonical condition.") -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.FOUND,
        summary="Original retrieved.",
        evidence=[original(str(uuid4()), text)],
    )


@pytest.mark.parametrize("invalid", [1, 0, "true", "false", None, [], {}])
def test_capture_task_timings_requires_an_explicit_boolean(invalid: object) -> None:
    with pytest.raises(ValueError, match="capture_task_timings must be a boolean"):
        fixture(lambda _arguments, _child: found(), capture=cast(bool, invalid))


def test_mixed_parallel_batch_records_each_worker_and_includes_stage_closure(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    both_started = Barrier(2)
    closed = CanonicalEvidenceStage.close
    arguments_seen: list[dict[str, JsonValue]] = []

    def close(stage: CanonicalEvidenceStage) -> None:
        clock.advance(2)
        closed(stage)

    monkeypatch.setattr(CanonicalEvidenceStage, "close", close)

    def handler(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        arguments_seen.append(dict(arguments))
        both_started.wait(timeout=3)
        clock.advance(90 if "query" in arguments else 4)
        return found()

    acquirer = fixture(handler)
    actions = [search_action(), read_action()]
    before = [action.model_dump(mode="json") for action in actions]
    receipts = acquirer.acquire(actions, plan())
    snapshot = acquirer.task_timing_snapshot()

    assert snapshot.completed["search_corpus"].max_seconds == 92
    assert snapshot.completed["read_provision"].max_seconds == 6
    assert snapshot.completed["read_provision"].count == 1
    assert snapshot.diagnostics == {}
    assert len(receipts) == 2 and acquirer.ledger.citation_numbers() == (1, 2)
    assert [action.model_dump(mode="json") for action in actions] == before
    assert all("elapsed_seconds" not in row for row in receipts)
    assert {row["tool"] for row in receipts} == {"search_corpus", "read_provision"}
    assert arguments_seen == [
        {"query": actions[0].arguments["query"], "expand_query": False},
        dict(actions[1].arguments),
    ] or arguments_seen == [
        dict(actions[1].arguments),
        {"query": actions[0].arguments["query"], "expand_query": False},
    ]
    assert any("tool=read_provision" in (step.summary or "") for step in steps)
    assert all(len(step.summary or "") <= 160 for step in steps)


def test_parallel_counts_maxima_and_returned_snapshots_are_independent(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)

    def handler(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        article = arguments["article"]
        assert isinstance(article, str)
        clock.advance(float(article))
        return found()

    acquirer = fixture(handler)
    actions = [
        read_action().model_copy(
            update={"arguments": {**read_action().arguments, "article": str(index)}}
        )
        for index in range(1, 25)
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(
            executor.map(lambda action: dispatch(acquirer, action), actions)
        )
    assert all(outcome.status == OutcomeStatus.FOUND for outcome in outcomes)
    snapshot = acquirer.task_timing_snapshot()
    stats = snapshot.completed["read_provision"]
    assert (stats.count, stats.total_seconds, stats.max_seconds) == (24, 300, 24)
    with pytest.raises(ValidationError):
        stats.count = 999
    snapshot.completed.clear()
    snapshot.diagnostics["private-source"] = {}
    current = acquirer.task_timing_snapshot()
    assert current.completed["read_provision"] == stats
    assert current.diagnostics == {}
    assert len(steps) == 24


@pytest.mark.parametrize(
    "status", [status for status in OutcomeStatus if status != OutcomeStatus.FOUND]
)
def test_incomplete_outcomes_never_enter_successful_original_timings(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep], status: OutcomeStatus
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found().model_copy(update={"status": status})

    def handler(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        clock.advance(0.25)
        return expected

    acquirer = fixture(handler)
    assert dispatch(acquirer) is expected
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert snapshot.diagnostics["read_provision"][status.value].count == 1
    assert snapshot.diagnostics["read_provision"][status.value].max_seconds == 0.25
    assert f"status={status.value}" in (steps[0].summary or "")


@pytest.mark.parametrize("status", ["empty", "noncanonical"])
def test_found_without_whole_citable_original_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep], status: str
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found()
    expected.evidence = (
        []
        if status == "empty"
        else [EvidenceItem(source_id="private-source", text="Navigation snippet.")]
    )
    acquirer = fixture(lambda _arguments, _child: expected)
    assert dispatch(acquirer) is expected
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert snapshot.diagnostics["read_provision"][status].count == 1
    assert "canonical_complete=0" in (steps[0].summary or "")


@pytest.mark.parametrize("layer", ["raw", "canonical", "citation"])
@pytest.mark.parametrize("flag", ["external", "derived", "untrusted", "truncated"])
@pytest.mark.parametrize("flag_value", [True, 1, "true", "false"])
def test_unsafe_original_layers_are_not_forecast_samples(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[RecordedStep],
    layer: str,
    flag: str,
    flag_value: JsonValue,
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found()
    item = expected.evidence[0]
    if layer == "raw":
        item.metadata[flag] = flag_value
    elif layer == "canonical":
        item.metadata["canonical_metadata"] = {flag: flag_value}
    acquirer = fixture(lambda _arguments, _child: expected)
    if layer == "citation":
        recorded = acquirer.ledger.get

        def unsafe_citation(number: int) -> EvidenceItem | None:
            observed = recorded(number)
            assert observed is not None and observed.search_doc is not None
            cast(dict[str, JsonValue], observed.search_doc.metadata)[flag] = flag_value
            return observed

        monkeypatch.setattr(acquirer.ledger, "get", unsafe_citation)
    assert dispatch(acquirer) is expected
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert snapshot.diagnostics["read_provision"]["noncanonical"].count == 1
    assert "canonical_complete=0" in (steps[0].summary or "")


@pytest.mark.parametrize("cancelled", [False, True])
def test_original_dispatch_exception_is_preserved_and_never_a_success_sample(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep], cancelled: bool
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    error = (
        RunStopped("private cancellation details")
        if cancelled
        else RuntimeError("private transport details")
    )
    acquirer = fixture(lambda _arguments, _child: found())

    def fail(_call: CapabilityCall, _child: RunContext) -> ToolOutcome:
        clock.advance(5)
        raise error

    monkeypatch.setattr(acquirer.registry, "dispatch", fail)
    with pytest.raises(type(error)) as raised:
        dispatch(acquirer)
    assert raised.value is error
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert (
        snapshot.diagnostics["read_provision"][
            "cancelled" if cancelled else "error"
        ].max_seconds
        == 5
    )
    assert "private" not in (steps[0].summary or "")


def test_closure_exception_remains_original_exception_and_diagnostic(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    error = RuntimeError("private stage closure failure")

    def close(_stage: CanonicalEvidenceStage) -> None:
        clock.advance(7)
        raise error

    monkeypatch.setattr(CanonicalEvidenceStage, "close", close)
    acquirer = fixture(lambda _arguments, _child: found())
    with pytest.raises(RuntimeError) as raised:
        dispatch(acquirer)
    assert raised.value is error
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert snapshot.diagnostics["read_provision"]["error"].max_seconds == 7
    assert "status=error" in (steps[0].summary or "")


def test_default_off_does_not_read_clock_or_change_summary_args_outcome_or_receipts(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()

    def forbidden_clock() -> float:
        raise AssertionError("Default dispatch must not collect timing")

    monkeypatch.setattr(clock, "monotonic", forbidden_clock)
    monkeypatch.setattr(acquisition, "time", clock)
    seen: list[dict[str, JsonValue]] = []
    expected = found()

    def handler(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        seen.append(dict(arguments))
        return expected

    acquirer = fixture(handler, capture=False)
    action = read_action()
    assert dispatch(acquirer, action) is expected
    assert seen == [action.arguments]
    assert acquirer.task_timing_snapshot().completed == {}
    assert acquirer.task_timing_snapshot().diagnostics == {}
    assert steps[0].summary is None
    assert steps[0].output_value == {"status": "found", "original_count": 1}


def test_completed_signature_reuse_is_not_a_new_worker_timing(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)

    def handler(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        clock.advance(4)
        return found()

    acquirer = fixture(handler)
    first = acquirer.acquire([read_action()], plan())
    snapshot = acquirer.task_timing_snapshot()
    reused = acquirer.acquire([read_action()], plan())
    assert reused[0]["reused"] is True
    assert reused[0]["citations"] == first[0]["citations"]
    assert acquirer.task_timing_snapshot() == snapshot
    assert sum(step.operation == "legal_composite.source_task" for step in steps) == 1


def test_executor_queue_delay_is_not_in_the_next_worker_duration(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)

    def handler(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        clock.advance(90 if "query" in arguments else 4)
        return found()

    acquirer = fixture(handler)
    acquirer.policy = acquirer.policy.model_copy(update={"max_parallel_tools": 1})
    acquirer.acquire([search_action(), read_action()], plan())
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed["search_corpus"].max_seconds == 90
    assert snapshot.completed["read_provision"].max_seconds == 4
    assert sum(step.operation == "legal_composite.source_task" for step in steps) == 2


@pytest.mark.parametrize("mismatch", ["source", "chunk", "hash", "blank_text"])
def test_original_identity_and_text_integrity_are_required_for_timing_samples(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep], mismatch: str
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found()
    acquirer = fixture(lambda _arguments, _child: expected)
    recorded = acquirer.ledger.get

    def invalid_original(number: int) -> EvidenceItem | None:
        item = recorded(number)
        assert item is not None and item.search_doc is not None
        if mismatch == "source":
            item.search_doc.document_id = "private-unmatched-source"
        elif mismatch == "chunk":
            item.search_doc.metadata["regulatory_chunk_id"] = "private-unmatched-chunk"
        elif mismatch == "hash":
            item.text_hash = "0" * 64
        else:
            item.text = " "
        return item

    monkeypatch.setattr(acquirer.ledger, "get", invalid_original)
    assert dispatch(acquirer) is expected
    assert acquirer.task_timing_snapshot().completed == {}
    assert (
        acquirer.task_timing_snapshot()
        .diagnostics["read_provision"]["noncanonical"]
        .count
        == 1
    )
    assert "private" not in (steps[0].summary or "")


def test_timing_observation_failure_does_not_replace_dispatch_result(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found()
    acquirer = fixture(lambda _arguments, _child: expected)

    def unavailable(_number: int) -> EvidenceItem | None:
        raise RuntimeError("private diagnostic snapshot failure")

    monkeypatch.setattr(acquirer.ledger, "get", unavailable)
    assert dispatch(acquirer) is expected
    assert acquirer.task_timing_snapshot().completed == {}
    assert steps[0].summary is None


def test_start_clock_failure_cannot_prevent_existing_dispatch(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()

    def failed_clock() -> float:
        raise RuntimeError("private optional timing failure")

    monkeypatch.setattr(clock, "monotonic", failed_clock)
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found()
    acquirer = fixture(lambda _arguments, _child: expected)
    assert dispatch(acquirer) is expected
    assert acquirer.task_timing_snapshot().completed == {}
    assert steps[0].summary is None


def test_timing_summary_and_snapshot_do_not_export_private_tool_or_argument_values(
    monkeypatch: pytest.MonkeyPatch, steps: list[RecordedStep]
) -> None:
    clock = WorkerClock()
    monkeypatch.setattr(acquisition, "time", clock)
    expected = found("private original text")
    expected.summary = "private capability output summary"
    acquirer = fixture(lambda _arguments, _child: expected)
    action = read_action().model_copy(
        update={
            "tool": "private_tool_name",
            "need_ids": ["private_need_identity"],
            "arguments": {"private_argument": "private source query"},
        }
    )

    def successful_unknown(_call: CapabilityCall, _child: RunContext) -> ToolOutcome:
        return expected

    monkeypatch.setattr(acquirer.registry, "dispatch", successful_unknown)
    assert dispatch(acquirer, action) is expected
    snapshot = acquirer.task_timing_snapshot()
    assert snapshot.completed == {}
    assert snapshot.diagnostics["other"]["unsupported_tool"].count == 1
    assert "private" not in snapshot.model_dump_json()
    assert "private" not in (steps[0].summary or "")
    assert "tool=other status=unsupported_tool" in (steps[0].summary or "")
