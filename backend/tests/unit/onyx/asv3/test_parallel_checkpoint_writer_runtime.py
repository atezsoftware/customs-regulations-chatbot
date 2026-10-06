"""Durable barriers on actual hosted serial research and root publication."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from onyx.asv3 import runtime, serial_experimental_session
from onyx.asv3.models import RunStopped
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseStart,
    SectionEnd,
)
from tests.unit.onyx.asv3 import test_experimental_parallel_runtime as fixture
from tests.unit.onyx.asv3.test_runtime import (
    packets,
    response,
    run_independent,
    setup_run,
    user_payload,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def test_slow_persistence_does_not_hold_research_but_acceptance_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, _, checkpoints, queue, _, _, bodies, _ = (
        fixture.script_two_children(monkeypatch)
    )
    writing, release, terminals_ready = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    lock = threading.Lock()
    terminal_requests: set[str] = set()
    returned: list[str] = []
    original_invoke = selected.invoke.side_effect

    def invoke(**arguments: Any) -> Any:
        result = original_invoke(**arguments)
        request = user_payload(arguments["prompt"][1])["request"]
        if request in bodies:
            with lock:
                terminal_requests.add(request)
                if terminal_requests == set(fixture.TASKS):
                    terminals_ready.set()
        return result

    selected.invoke.side_effect = invoke
    original_run = serial_experimental_session.SerialExperimentalSession.run

    def run(session: serial_experimental_session.SerialExperimentalSession) -> Any:
        result = original_run(session)
        with lock:
            assert any(
                task["child_checkpoint"]["snapshot"]
                .get("serial_experimental_session", {})
                .get("accepted")
                is not None
                for saved in checkpoints
                for task in saved["workers"]["tasks"]
                if task["task_id"] == session.owner
            )
            returned.append(session.owner)
        return result

    monkeypatch.setattr(
        serial_experimental_session.SerialExperimentalSession, "run", run
    )

    def persist(**arguments: Any) -> None:
        if not writing.is_set():
            writing.set()
            assert release.wait(5)
        with lock:
            checkpoints.append(arguments["snapshot"])

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", persist)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.run_asv3_loop, **kwargs)
        try:
            assert writing.wait(5)
            assert terminals_ready.wait(5), (
                "Both research decisions advance during DB wait"
            )
            assert returned == []
            assert not future.done()
            assert not any(
                isinstance(item.obj, AgentResponseStart) for item in packets(queue)
            )
        finally:
            release.set()
        future.result(timeout=10)

    assert len(returned) == len(fixture.TASKS)
    assert selected.invoke.call_count == 5
    saved = decode_asv3_checkpoint(encode_asv3_checkpoint(checkpoints[-1]))
    questions = saved["question_research"]
    assert isinstance(questions, dict)
    answers = questions["answers"]
    assert isinstance(answers, list)
    assert all(isinstance(item, dict) for item in answers)
    assert [item["answer"] for item in answers if isinstance(item, dict)] == [
        bodies[task] for task in fixture.TASKS
    ]
    assert [item["sequence"] for item in checkpoints] == sorted(
        {item["sequence"] for item in checkpoints}
    )
    evidence = saved["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["included"] == [1]
    parallel_answers = saved["parallel_answers"]
    assert isinstance(parallel_answers, dict)
    receipts = parallel_answers["receipts"]
    assert isinstance(receipts, list)
    pins = evidence["pinned_delivery_calls"]
    assert isinstance(pins, list)
    assert all(
        isinstance(receipt, dict) and receipt["model_call_id"] in pins
        for receipt in receipts
    )


def test_failed_commit_blocks_final_start_and_joins_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, selected, _, queue, _ = fixture.parallel_setup(monkeypatch)
    selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("submit_answer", {"answer": "Merhaba!", "basis": "conversation"})]
    )
    failure = RuntimeError("Synthetic checkpoint commit failed")

    def persist(**_arguments: Any) -> None:
        raise failure

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", persist)
    with pytest.raises(RuntimeError) as rejected:
        runtime.run_asv3_loop(**kwargs)
    assert rejected.value is failure
    assert not any(
        isinstance(item.obj, (AgentResponseStart, SectionEnd))
        for item in packets(queue)
    )
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )


def test_completed_publication_waits_for_durable_checkpoint_before_section_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, selected, checkpoints, queue, _ = fixture.parallel_setup(monkeypatch)
    selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("submit_answer", {"answer": "Merhaba!", "basis": "conversation"})]
    )
    writing, release = threading.Event(), threading.Event()

    def persist(**arguments: Any) -> None:
        snapshot = arguments["snapshot"]
        if any(
            row.get("phase") == "completed" and row.get("task_id") is None
            for row in snapshot["progress_state"]["events"]
        ):
            writing.set()
            assert release.wait(5)
        checkpoints.append(snapshot)

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", persist)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.run_asv3_loop, **kwargs)
        try:
            assert writing.wait(5)
            emitted = packets(queue)
            assert any(isinstance(item.obj, AgentResponseStart) for item in emitted)
            assert not any(isinstance(item.obj, SectionEnd) for item in emitted)
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=5)
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
    assert any(isinstance(item.obj, SectionEnd) for item in packets(queue))


def test_stopped_research_drains_pending_capture_and_joins_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _, _ = fixture.parallel_setup(monkeypatch)
    calls = 0

    def invoke(**_arguments: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(calls=[("record_scenario", {"facts": []})])
        broker.cancelled.set()
        raise RunStopped("Synthetic research cancellation")

    selected.invoke.side_effect = invoke
    with pytest.raises(RunStopped):
        runtime.run_asv3_loop(**kwargs)
    assert checkpoints
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )


def test_worker_teardown_error_cannot_leave_the_writer_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, selected, _, queue, _ = fixture.parallel_setup(monkeypatch)
    selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("submit_answer", {"answer": "Merhaba!", "basis": "conversation"})]
    )
    failure = RuntimeError("Synthetic worker teardown failed")

    def close(_pool: Any) -> None:
        raise failure

    monkeypatch.setattr(runtime.WorkerPool, "close", close)
    with pytest.raises(RuntimeError) as rejected:
        runtime.run_asv3_loop(**kwargs)
    assert rejected.value is failure
    assert not any(isinstance(item.obj, SectionEnd) for item in packets(queue))
    assert not any(
        thread.name == "asv3-checkpoint-writer" for thread in threading.enumerate()
    )


def test_late_stale_capture_cannot_overwrite_newer_root_control_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, selected, checkpoints, _, _ = fixture.parallel_setup(monkeypatch)
    root_harnesses: list[Any] = []
    writers: list[Any] = []
    original_harness, original_writer = (
        runtime.Harness,
        runtime.ParallelCheckpointWriter,
    )

    def harness(**arguments: Any) -> Any:
        captured = original_harness(**arguments)
        root_harnesses.append(captured)
        return captured

    def writer(persist: Any) -> Any:
        captured = original_writer(persist)
        writers.append(captured)
        return captured

    monkeypatch.setattr(runtime, "Harness", harness)
    monkeypatch.setattr(runtime, "ParallelCheckpointWriter", writer)

    def invoke(**_arguments: Any) -> Any:
        captured = root_harnesses[0]
        captured.last_draft = "Older captured control state"
        stale = captured.snapshot()
        ready, proceed = threading.Event(), threading.Event()

        def late() -> None:
            ready.set()
            assert proceed.wait(5)
            captured.checkpoint(stale)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(late)
            try:
                assert ready.wait(5)
                captured.last_draft = "Current control state"
                captured.checkpoint(captured.snapshot())
            finally:
                proceed.set()
            future.result(timeout=5)
        writers[0].flush()
        assert checkpoints[-1]["last_draft"] == "Current control state"
        assert all(
            snapshot["last_draft"] != "Older captured control state"
            for snapshot in checkpoints
        )
        return response(
            calls=[("submit_answer", {"answer": "Merhaba!", "basis": "conversation"})]
        )

    selected.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert checkpoints[-1]["last_draft"] == "Merhaba!"


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_other_profiles_never_start_a_coalescing_writer(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    kwargs, _, _, checkpoints, _ = setup_run(monkeypatch)
    kwargs["research_profile"] = profile

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Other profiles retain synchronous persistence")

    monkeypatch.setattr(runtime, "ParallelCheckpointWriter", forbidden)
    if profile == "deep":
        run_independent(**kwargs)
    else:
        kwargs.pop("test_language")
        runtime.run_asv3_loop(**kwargs)
    assert checkpoints[-1]["publication_status"] == "found"
