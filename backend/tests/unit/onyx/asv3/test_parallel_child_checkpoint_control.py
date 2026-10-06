"""Keep deferred child producers independent of complete ledger capture."""

import copy
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
)
from onyx.asv3.serial_experimental_session import SerialExperimentalSession
from onyx.asv3.workers import WorkerCheckpointCapture, WorkerPool
from onyx.llm.models import ReasoningEffort
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    SCENARIO,
    SerialRun,
    serial_run,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def root_and_ledger(run: SerialRun) -> tuple[RunContext, EvidenceLedger]:
    ledger = EvidenceLedger()
    root = RunContext(
        run_id=run.harness.context.run_id,
        scope=copy.deepcopy(run.harness.context.scope),
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "evidence": ledger,
        },
    )
    deliveries: list[dict[str, JsonValue]] = []
    full_deliveries: list[dict[str, JsonValue]] = []
    for index in range(3):
        item = EvidenceItem(
            source_id=f"canonical-source-{index}",
            chunk_id=f"canonical-chunk-{index}",
            text="Operative condition, exception and continuation. İĞŞöç — 東京\n"
            * 500,
            metadata={
                "source_sha256": f"source-sha-{index}",
                "canonical_metadata": {
                    "document_type": "law",
                    "heading_path": ["Example Law", f"MADDE {index + 1}"],
                    "publication_revision_id": "revision-a",
                    "validity_start": "2026-01-01",
                },
                "provenance": {"binding_sha256": f"binding-{index}", "external": False},
            },
        )
        number = ledger.add([item], root)[0]
        full_deliveries.append({"citation": number, "text": item.text})
        middle = len(item.text) // 2
        deliveries.extend(
            [
                {
                    "citation": number,
                    "text": item.text[:middle],
                    "start_char": 0,
                    "end_char": middle,
                },
                {
                    "citation": number,
                    "text": item.text[middle:],
                    "start_char": middle,
                    "end_char": len(item.text),
                },
            ]
        )
    ledger.record_delivery("partial-original-call", "asv3_coordinator", deliveries)
    ledger.pin_delivery("partial-original-call")
    ledger.record_delivery("pinned-original-call", "asv3_coordinator", full_deliveries)
    ledger.pin_delivery("pinned-original-call")
    return root, ledger


def hosted_session(
    run: SerialRun,
    envelope: RunContext,
    ledger: EvidenceLedger,
    *,
    deferred: bool,
    callback: Callable[[dict[str, JsonValue]], None] | None = None,
) -> SerialExperimentalSession:
    return SerialExperimentalSession(
        outer_context=envelope,
        request=FOCUS,
        scenario_request=SCENARIO,
        history=run.history,
        ledger=ledger,
        llm=run.selected,
        reasoning_effort=ReasoningEffort.AUTO,
        token_counter=len,
        capability_factory=lambda _context: [],
        verify=lambda _context, _arguments: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="No provider verification"
        ),
        checkpoint_callback=callback,
        deferred_checkpoint_control=deferred,
    )


def completed_task(pool: WorkerPool) -> str:
    identity = pool.spawn(
        FOCUS,
        independent_question=True,
        assignment_id="independent-a",
        outcome_ids=["local-result"],
    )
    pool.wait_until_all([identity])
    return identity


def snapshot_from(payload: dict[str, JsonValue], task_id: str) -> dict[str, JsonValue]:
    rows = cast(list[dict[str, JsonValue]], payload["tasks"])
    task = next(row for row in rows if row["task_id"] == task_id)
    wrapper = cast(dict[str, JsonValue], task["child_checkpoint"])
    return cast(dict[str, JsonValue], wrapper["snapshot"])


def test_actual_foreground_save_finishes_while_writer_ledger_export_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "\n".join(["Selam! İĞŞçöü — 東京"] * 700)
    run = serial_run(monkeypatch, "conversation", body)
    run.calls.clear()
    root, ledger = root_and_ledger(run)
    with closing(
        WorkerPool(
            root, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = completed_task(pool)
        child = hosted_session(
            run,
            pool._contexts[task_id],
            ledger,
            deferred=True,
            callback=lambda value: pool.record_checkpoint_control(task_id, value),
        )
        try:
            assert child.run().summary == body
            full = child.snapshot()
            control = child.snapshot_control()
            assert control == {
                key: value for key, value in full.items() if key != "evidence"
            }
            pool.record_checkpoint_control(task_id, control)
            capture = pool.capture_checkpoint_state()
            entered, release = threading.Event(), threading.Event()
            real_export = ledger.export

            def blocked_export() -> dict[str, JsonValue]:
                entered.set()
                assert release.wait(timeout=5)
                return real_export()

            def write() -> dict[str, JsonValue]:
                return pool.materialize_checkpoint_state(capture, ledger.export())

            with patch.object(ledger, "export", side_effect=blocked_export):
                with ThreadPoolExecutor(max_workers=2) as threads:
                    writer = threads.submit(write)
                    assert entered.wait(timeout=2)
                    foreground = threads.submit(child.harness._save)
                    try:
                        foreground.result(timeout=1)
                        assert not writer.done()
                    finally:
                        release.set()
                    materialized = writer.result(timeout=5)
            restored = snapshot_from(materialized, task_id)
            assert restored == {**control, "evidence": real_export()}
            assert restored["last_draft"] == body
            assert restored["budget"] == control["budget"]
            assert (
                restored["serial_experimental_session"]
                == control["serial_experimental_session"]
            )
            verifier = EvidenceLedger()
            verifier.restore(cast(dict[str, JsonValue], restored["evidence"]), root)
            assert verifier.export() == real_export()
            assert verifier.completely_delivered("pinned-original-call") == {1, 2, 3}
            assert verifier.completely_delivered("partial-original-call") == set()
            pins = verifier.export()["pinned_delivery_calls"]
            assert isinstance(pins, list)
            assert "pinned-original-call" in pins
            replay = hosted_session(
                run, pool._contexts[task_id], verifier, deferred=True
            )
            try:
                replay.restore(restored)
                run.selected.reset_mock()
                assert replay.run().summary == body
                run.selected.invoke.assert_not_called()
            finally:
                replay.workers.close()
        finally:
            child.workers.close()


def test_old_capture_keeps_exact_control_without_overwriting_newer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    root, ledger = root_and_ledger(run)
    with closing(
        WorkerPool(
            root, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = completed_task(pool)
        child = hosted_session(run, pool._contexts[task_id], ledger, deferred=True)
        try:
            first = child.snapshot_control()
            first["last_draft"] = "First exact draft. İĞŞçöü — 東京"
            pool.record_checkpoint_control(task_id, first)
            capture = pool.capture_checkpoint_state()
            second = copy.deepcopy(first)
            second["last_draft"] = "Second exact draft. İĞŞçöü — 東京"
            pool.record_checkpoint_control(task_id, second)
            first["last_draft"] = "Caller mutation"
            old = pool.materialize_checkpoint_state(capture, ledger.export())
            assert (
                snapshot_from(old, task_id)["last_draft"]
                == "First exact draft. İĞŞçöü — 東京"
            )
            latest = pool.materialize_checkpoint_state(
                pool.capture_checkpoint_state(), ledger.export()
            )
            assert snapshot_from(latest, task_id)["last_draft"] == second["last_draft"]
            assert pool.checkpoint(task_id) == snapshot_from(latest, task_id)
        finally:
            child.workers.close()


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "foreign-run"), ("request", "Foreign question"), ("evidence", {})],
)
def test_control_rejects_wrong_run_request_or_embedded_ledger(
    monkeypatch: pytest.MonkeyPatch, field: str, value: JsonValue
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    root, ledger = root_and_ledger(run)
    with closing(
        WorkerPool(
            root, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = completed_task(pool)
        child = hosted_session(run, pool._contexts[task_id], ledger, deferred=True)
        try:
            control = child.snapshot_control()
            control[field] = value
            with pytest.raises(ValueError):
                pool.record_checkpoint_control(task_id, control)
            assert pool.checkpoint(task_id) is None
        finally:
            child.workers.close()


@pytest.mark.parametrize("target", ["task", "scope", "integrity", "missing-task"])
def test_materialization_rejects_tampered_bindings_and_missing_task(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    root, ledger = root_and_ledger(run)
    with closing(
        WorkerPool(
            root, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = completed_task(pool)
        child = hosted_session(run, pool._contexts[task_id], ledger, deferred=True)
        try:
            pool.record_checkpoint_control(task_id, child.snapshot_control())
            capture = pool.capture_checkpoint_state()
            controls = dict(capture.controls)
            payload = copy.deepcopy(capture.payload)
            if target == "missing-task":
                payload["tasks"] = []
            else:
                control = controls[task_id]
                changes = (
                    {"task_id": "foreign-task"}
                    if target == "task"
                    else {"scope_hash": "foreign-scope"}
                    if target == "scope"
                    else {"integrity": "0" * 64}
                )
                controls[task_id] = replace(control, **changes)
            with pytest.raises(ValueError):
                pool.materialize_checkpoint_state(
                    WorkerCheckpointCapture(payload, controls), ledger.export()
                )
            assert pool.capture_checkpoint_state() == capture
        finally:
            child.workers.close()


def test_default_session_callbacks_keep_full_legacy_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    root, ledger = root_and_ledger(run)
    with closing(
        WorkerPool(
            root, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = completed_task(pool)
        saves: list[dict[str, JsonValue]] = []
        child = hosted_session(
            run, pool._contexts[task_id], ledger, deferred=False, callback=saves.append
        )
        try:
            child.harness._save()
            assert saves[-1] == child.snapshot()
            assert saves[-1]["evidence"] == ledger.export()
            envelope = copy.copy(pool._contexts[task_id])
            envelope.services = {**envelope.services, "experimental_parallel": False}
            with pytest.raises(ValueError, match="require Experimental Parallel"):
                hosted_session(run, envelope, ledger, deferred=True)
        finally:
            child.workers.close()
