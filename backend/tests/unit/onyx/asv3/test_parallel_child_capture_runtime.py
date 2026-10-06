"""Keep shared ledger capture off independent semantic-owner threads."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.serial_experimental_session import SerialExperimentalSession
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


def test_blocked_ledger_cut_does_not_block_owned_controls_and_retains_complete_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, _, checkpoints, queue, _, _, bodies, _ = (
        fixture.script_two_children(monkeypatch)
    )
    entered, release, terminals_ready = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    originals_export, original_control, original_invoke = (
        EvidenceLedger.export,
        SerialExperimentalSession.snapshot_control,
        selected.invoke.side_effect,
    )
    state = threading.local()
    captured_controls: list[dict[str, JsonValue]] = []
    terminal_requests: set[str] = set()
    lock = threading.Lock()

    def export(ledger: EvidenceLedger) -> dict[str, JsonValue]:
        assert not getattr(state, "capturing_control", False), (
            "A child control must never export the shared ledger"
        )
        if (
            threading.current_thread().name == "asv3-checkpoint-writer"
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(5)
        return originals_export(ledger)

    def control(session: SerialExperimentalSession) -> dict[str, JsonValue]:
        state.capturing_control = True
        try:
            result = original_control(session)
        finally:
            state.capturing_control = False
        assert "evidence" not in result
        with lock:
            captured_controls.append(result)
        return result

    def invoke(**arguments: Any) -> Any:
        result = original_invoke(**arguments)
        request = user_payload(arguments["prompt"][1])["request"]
        if request in bodies:
            with lock:
                terminal_requests.add(request)
                if terminal_requests == set(fixture.TASKS):
                    terminals_ready.set()
        return result

    monkeypatch.setattr(EvidenceLedger, "export", export)
    monkeypatch.setattr(SerialExperimentalSession, "snapshot_control", control)
    selected.invoke.side_effect = invoke
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(runtime.run_asv3_loop, **kwargs)
        try:
            assert entered.wait(5)
            assert terminals_ready.wait(5)
            assert checkpoints == []
            assert not running.done()
            assert not any(
                isinstance(packet.obj, AgentResponseStart) for packet in packets(queue)
            )
        finally:
            release.set()
        running.result(timeout=15)

    assert captured_controls
    assert selected.invoke.call_count == 5
    final = decode_asv3_checkpoint(encode_asv3_checkpoint(checkpoints[-1]))
    canonical = _object(final["evidence"])
    assert canonical["records"]
    assert canonical["deliveries"]
    pins = canonical["pinned_delivery_calls"]
    assert isinstance(pins, list)
    tasks = _objects(_object(final["workers"])["tasks"])
    for receipt in _objects(_object(final["parallel_answers"])["receipts"]):
        task = next(item for item in tasks if item["task_id"] == receipt["task_id"])
        child = _object(_object(task["child_checkpoint"])["snapshot"])
        assert child["evidence"] == canonical
        assert child["last_draft"] == bodies[str(task["task"])] == receipt["answer"]
        assert receipt["model_call_id"] in pins
        accepted = _object(_object(child["serial_experimental_session"])["accepted"])
        assert accepted["model_call_id"] == receipt["model_call_id"]
        assert _object(child["budget"])["decisions"] == 2
    assert any(isinstance(packet.obj, SectionEnd) for packet in packets(queue))
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )


def test_failed_accepted_materialization_cannot_publish_or_leave_a_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _, _, queue, _, _, _, _ = fixture.script_two_children(monkeypatch)
    original_save = runtime.save_asv3_checkpoint
    accepted_failure = threading.Event()

    def persist(**arguments: Any) -> Any:
        snapshot = arguments["snapshot"]
        for task in snapshot.get("workers", {}).get("tasks", []):
            wrapper = task.get("child_checkpoint")
            if isinstance(wrapper, dict) and (
                wrapper["snapshot"]["serial_experimental_session"]["accepted"]
                is not None
            ):
                accepted_failure.set()
                raise RuntimeError("Accepted checkpoint commit failed")
        return original_save(**arguments)

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", persist)
    with pytest.raises(RuntimeError, match="Accepted checkpoint commit failed"):
        runtime.run_asv3_loop(**kwargs)
    assert accepted_failure.is_set()
    assert not any(
        isinstance(packet.obj, (AgentResponseStart, SectionEnd))
        for packet in packets(queue)
    )
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )
