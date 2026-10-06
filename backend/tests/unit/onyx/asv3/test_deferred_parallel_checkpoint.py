"""Coalesce aggregate capture without delaying independent semantic owners."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.harness import Harness
from onyx.asv3.models import CapabilityCall, Decision, RunContext
from onyx.asv3.parallel_checkpoint_writer import ParallelCheckpointWriter
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchState
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from onyx.server.query_and_chat.streaming_models import AgentResponseStart, SectionEnd
from tests.unit.onyx.asv3 import test_experimental_parallel_runtime as fixture
from tests.unit.onyx.asv3.test_runtime import packets, user_payload

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _objects(value: JsonValue) -> list[dict[str, JsonValue]]:
    assert isinstance(value, list)
    return [_object(row) for row in value]


def test_notifications_coalesce_blocked_capture_and_credit_only_successful_commit() -> (
    None
):
    entered, release, second_commit, release_commit = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    state_lock = threading.Lock()
    latest: dict[str, JsonValue] = {"original": "First exact source"}
    captured: list[dict[str, JsonValue]] = []
    committed: list[dict[str, JsonValue]] = []

    def capture() -> dict[str, JsonValue]:
        if not captured:
            entered.set()
            assert release.wait(3)
        with state_lock:
            snapshot = copy.deepcopy(latest)
        captured.append(snapshot)
        return snapshot

    def persist(snapshot: dict[str, JsonValue]) -> None:
        if committed:
            second_commit.set()
            assert release_commit.wait(3)
        committed.append(snapshot)

    writer = ParallelCheckpointWriter(persist, capture=capture)
    one = writer.notify()
    assert entered.wait(1)
    with state_lock:
        latest["original"] = "Latest original, scope and accepted owner intact"
    two, three = writer.notify(), writer.notify()
    assert (one, two, three) == (1, 2, 3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(writer.flush, two)
        try:
            assert not waiting.done()
            release.set()
            assert second_commit.wait(1)
            assert len(committed) == 1
            assert not waiting.done(), (
                "A capture alone cannot satisfy a durable barrier"
            )
            release_commit.set()
            waiting.result(timeout=2)
            writer.flush(three)
        finally:
            release.set()
            release_commit.set()
            writer.close()
    assert len(captured) == len(committed) == 2
    assert committed[-1] == latest


def test_deferred_capture_failure_poison_and_context_are_preserved() -> None:
    context: ContextVar[str] = ContextVar("deferred_capture_owner", default="unset")
    failure = ValueError("Aggregate binding failed")
    seen: list[str] = []
    token = context.set("captured-owner")

    def capture() -> dict[str, JsonValue]:
        seen.append(context.get())
        raise failure

    writer = ParallelCheckpointWriter(lambda _: None, capture=capture)
    context.reset(token)
    revision = writer.notify()
    with pytest.raises(ValueError) as result:
        writer.flush(revision)
    assert result.value is failure
    for entry in (writer.notify, writer.raise_if_failed, writer.close):
        with pytest.raises(ValueError) as poisoned:
            entry()
        assert poisoned.value is failure
    assert seen == ["captured-owner"]
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )


def test_control_shape_detaches_aliases_and_never_exports_shared_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext()
    context.services["research_state"] = ResearchState([], context)
    harness = Harness(
        request="Frozen request",
        context=context,
        registry=CapabilityRegistry([]),
        decide=lambda _: Decision(answer="Done"),
    )
    call = CapabilityCall(name="read", arguments={"nested": {"exact": "Original"}})
    harness._pending_calls[call.call_id] = call.model_dump(mode="json")
    harness.last_draft = "Exact body [1]."
    whole = copy.deepcopy(harness.snapshot())
    control = harness.snapshot_control()
    assert set(control) == set(whole) - {"budget", "evidence", "research_state"}
    assert control == {key: value for key, value in whole.items() if key in control}

    def forbidden() -> dict[str, JsonValue]:
        raise AssertionError("Shared export belongs to aggregate capture")

    monkeypatch.setattr(harness.evidence, "export", forbidden)
    monkeypatch.setattr(context.budget, "snapshot", forbidden)
    state = context.services["research_state"]
    monkeypatch.setattr(state, "export", forbidden)
    assert harness.snapshot_control() == control
    harness._pending_calls[call.call_id]["arguments"] = {"nested": "Changed"}
    harness.last_draft = "Later body"
    assert control == {key: value for key, value in whole.items() if key in control}


def test_blocked_aggregate_capture_does_not_block_children_and_final_retains_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, _, checkpoints, queue, _, _, bodies, _ = (
        fixture.script_two_children(monkeypatch)
    )
    entered, release, decisions_ready = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    terminal_requests: set[str] = set()
    lock = threading.Lock()
    original_invoke = selected.invoke.side_effect
    original_export = runtime.OutcomeMap.export
    original_snapshot, original_control = Harness.snapshot, Harness.snapshot_control
    root_controls: list[dict[str, JsonValue]] = []

    def export(state: Any) -> dict[str, JsonValue]:
        if (
            threading.current_thread().name == "asv3-checkpoint-writer"
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(5)
        return original_export(state)

    def snapshot(harness: Harness) -> dict[str, JsonValue]:
        assert threading.current_thread().name != "asv3-checkpoint-writer"
        return original_snapshot(harness)

    def control(harness: Harness) -> dict[str, JsonValue]:
        assert threading.current_thread().name != "asv3-checkpoint-writer"
        result = original_control(harness)
        if harness.context.services.get("experimental_parallel") is True:
            root_controls.append(result)
        return result

    def invoke(**arguments: Any) -> Any:
        result = original_invoke(**arguments)
        request = user_payload(arguments["prompt"][1])["request"]
        if request in bodies:
            with lock:
                terminal_requests.add(request)
                if terminal_requests == set(fixture.TASKS):
                    decisions_ready.set()
        return result

    monkeypatch.setattr(runtime.OutcomeMap, "export", export)
    monkeypatch.setattr(Harness, "snapshot", snapshot)
    monkeypatch.setattr(Harness, "snapshot_control", control)
    selected.invoke.side_effect = invoke
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(runtime.run_asv3_loop, **kwargs)
        try:
            assert entered.wait(5)
            assert decisions_ready.wait(5), (
                "Both owners advance during heavy aggregate capture"
            )
            assert not running.done()
            assert checkpoints == []
            assert not any(
                isinstance(row.obj, AgentResponseStart) for row in packets(queue)
            )
        finally:
            release.set()
        running.result(timeout=10)

    assert selected.invoke.call_count == 5
    assert root_controls and all(
        "evidence" not in item and "budget" not in item for item in root_controls
    )
    final = decode_asv3_checkpoint(encode_asv3_checkpoint(checkpoints[-1]))
    assert _object(final["budget"])["decisions"] == 5
    answers = _objects(_object(final["question_research"])["answers"])
    assert [row["answer"] for row in answers] == [
        bodies[task] for task in fixture.TASKS
    ]
    pins = _object(final["evidence"])["pinned_delivery_calls"]
    assert isinstance(pins, list)
    tasks = _objects(_object(final["workers"])["tasks"])
    for receipt in _objects(_object(final["parallel_answers"])["receipts"]):
        assert receipt["model_call_id"] in pins
        task = next(row for row in tasks if row["task_id"] == receipt["task_id"])
        child = _object(_object(task["child_checkpoint"])["snapshot"])
        assert _object(child["serial_experimental_session"])["accepted"] is not None
    assert final["last_draft"] == "\n\n".join(
        f"## {index}. {row['answer_title']}\n\n{row['answer']}"
        for index, row in enumerate(answers, 1)
    )
    assert any(isinstance(row.obj, SectionEnd) for row in packets(queue))
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )
