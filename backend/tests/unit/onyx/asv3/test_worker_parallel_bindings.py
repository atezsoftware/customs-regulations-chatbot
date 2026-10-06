import copy
import threading
from contextlib import closing
from typing import Callable, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.models import OutcomeStatus, RunContext, TaskStatus, ToolOutcome
from onyx.asv3.workers import WorkerPool


def test_checkpoint_restore_and_explicit_followup_preserve_assignment_and_outcome_bindings() -> (
    None
):
    root = RunContext(run_id="run", scope={"document_sets": [15]})
    observed: list[dict[str, object]] = []

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        observed.append(dict(child.services))
        save = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        save(
            {
                "version": 1,
                "run_id": child.run_id,
                "request": task,
                "turns": [{"content": "private checkpoint-only payload"}],
            }
        )
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Supported result [1].")

    with closing(WorkerPool(root, runner)) as pool:
        task_id = pool.spawn(
            "Assigned outcome",
            independent_question=True,
            assignment_id="q0",
            outcome_ids=["permission", "proof"],
        )
        pool.wait_until_all([task_id])
        raw = pool.checkpoint(task_id)
        assert raw is not None
        raw["request"] = "External mutation"
        retained = pool.checkpoint(task_id)
        assert retained is not None
        assert retained["request"] == "Assigned outcome"
        assert pool.list()[0].child_checkpoint is None
        assert pool.results()[0].child_checkpoint is None
        snapshot = pool.export()
    assert observed[0]["task_outcome_ids"] == ["permission", "proof"]
    assert observed[0]["assignment_id"] == "q0"

    with closing(WorkerPool(root, runner)) as restored:
        restored.restore(snapshot)
        assert restored._futures == {}
        assert len(observed) == 1
        old = restored.results()[0]
        assert old.assignment_id == "q0"
        assert old.outcome_ids == ["permission", "proof"]
        assert restored._contexts[task_id].services["restored_child_checkpoint"]
        next_id = restored.followup(task_id, "Find the missing exception only.")
        restored.wait_until_all([next_id])
        next_task = restored.results([next_id])[0]
        assert next_task.assignment_id == "q0"
        assert next_task.outcome_ids == ["permission", "proof"]
        assert observed[1]["previous_child_checkpoint"]
        assert "restored_child_checkpoint" not in observed[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("assignment_id", "q1"),
        ("outcome_ids", ["sibling"]),
        ("task_id", "sibling-task"),
        ("task", "Different request"),
    ],
)
def test_worker_restore_rejects_changed_checkpoint_bindings(
    field: str, value: JsonValue
) -> None:
    root = RunContext(run_id="run", scope={"document_sets": [15]})

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        save = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        save({"run_id": child.run_id, "request": task})
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(root, runner)) as pool:
        task_id = pool.spawn(
            "Assigned outcome",
            independent_question=True,
            assignment_id="q0",
            outcome_ids=["permission"],
        )
        pool.wait_until_all([task_id])
        snapshot = copy.deepcopy(pool.export())
    tasks = cast(list[dict[str, JsonValue]], snapshot["tasks"])
    tasks[0][field] = value
    with closing(WorkerPool(root, runner)) as restored:
        with pytest.raises(ValueError, match="checkpoint.*changed"):
            restored.restore(snapshot)
        assert restored.list() == []


def test_interrupted_checkpoint_is_retained_without_automatic_restart() -> None:
    root = RunContext(run_id="run", scope={"document_sets": [15]})
    entered, released = threading.Event(), threading.Event()

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        save = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        save({"run_id": child.run_id, "request": task, "pending_calls": []})
        entered.set()
        assert released.wait(5)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(root, runner)) as pool:
        task_id = pool.spawn(
            "Assigned outcome", independent_question=True, assignment_id="q0"
        )
        assert entered.wait(5)
        snapshot = pool.export()
        released.set()
        pool.wait_until_all([task_id])
    with closing(WorkerPool(root, runner)) as restored:
        restored.restore(snapshot)
        assert restored.results()[0].status == TaskStatus.INTERRUPTED
        assert restored.results()[0].outcome is None
        assert restored.checkpoint(task_id) is not None
        assert restored._futures == {}


def test_checkpoint_scope_and_private_payload_integrity_are_fenced() -> None:
    root = RunContext(run_id="run", scope={"document_sets": [15]})

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        save = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        save({"run_id": child.run_id, "request": task, "last_draft": "Original"})
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(root, runner)) as pool:
        task_id = pool.spawn(
            "Assigned outcome", independent_question=True, assignment_id="q0"
        )
        pool.wait_until_all([task_id])
        snapshot = pool.export()
        with pytest.raises(ValueError, match="request"):
            pool.record_checkpoint(task_id, {"run_id": "run", "request": "wrong"})
    wrong_scope = RunContext(run_id="run", scope={"document_sets": [16]})
    with closing(WorkerPool(wrong_scope, runner)) as restored:
        with pytest.raises(ValueError, match="scope"):
            restored.restore(snapshot)
    changed = copy.deepcopy(snapshot)
    tasks = cast(list[dict[str, JsonValue]], changed["tasks"])
    wrapper = cast(dict[str, JsonValue], tasks[0]["child_checkpoint"])
    payload = cast(dict[str, JsonValue], wrapper["snapshot"])
    payload["last_draft"] = "Changed"
    with closing(WorkerPool(root, runner)) as restored:
        with pytest.raises(ValueError, match="integrity"):
            restored.restore(changed)


def test_assignment_binding_survives_even_before_first_child_checkpoint() -> None:
    root = RunContext(run_id="run", scope={"document_sets": [15]})

    def runner(
        _task: str, _child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(root, runner)) as pool:
        task_id = pool.spawn(
            "Assigned outcome",
            independent_question=True,
            assignment_id="q0",
            outcome_ids=["permission"],
        )
        pool.wait_until_all([task_id])
        snapshot = pool.export()
    tasks = cast(list[dict[str, JsonValue]], snapshot["tasks"])
    assert tasks[0]["child_checkpoint"] is None
    tasks[0]["outcome_ids"] = ["sibling"]
    with closing(WorkerPool(root, runner)) as restored:
        with pytest.raises(ValueError, match="outcome binding"):
            restored.restore(snapshot)
