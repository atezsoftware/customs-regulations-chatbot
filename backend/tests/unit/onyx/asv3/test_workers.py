import copy
import threading
import time
from typing import Callable, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    LocalBudget,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    TaskStatus,
    ToolOutcome,
)
from onyx.asv3.workers import WorkerPool


def done(
    _task: str, _context: RunContext, _updates: Callable[[], list[str]]
) -> ToolOutcome:
    return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")


def test_local_spend_and_shared_physical_and_byte_resources() -> None:
    parent = SharedBudget(
        max_tools=1,
        max_decisions=1,
        unlimited_execution=True,
        max_evidence_bytes=6,
        max_artifact_bytes=5,
        max_inflight_tools=1,
        max_inflight_models=1,
        max_inflight_sources=1,
    )
    context = RunContext(budget=parent)
    first = context.independent_child()
    second = context.independent_child()
    first.budget.consume("tools", 2)
    first.budget.consume("tools", 100)
    second.budget.consume("tools")
    for _ in range(2):
        first.consume_research_decision()
    for _ in range(100):
        first.consume_research_decision()
    second.consume_research_decision()
    assert parent.snapshot()["tools"] == parent.snapshot()["decisions"] == 103
    assert parent.unlimited_execution is True
    for name in ("tool_slots", "model_slots", "source_slots"):
        shared = getattr(parent, name)
        assert getattr(first.budget, name) is getattr(second.budget, name) is shared
        assert shared.acquire(blocking=False)
        try:
            assert not getattr(second.budget, name).acquire(blocking=False)
        finally:
            shared.release()
    ledger = EvidenceLedger()
    original = EvidenceItem(source_id="law", text="source")
    assert ledger.add([original], first) == ledger.add([original], second) == [1]
    with pytest.raises(RunStopped, match="evidence_bytes"):
        ledger.add([EvidenceItem(source_id="other", text="x")], second)
    first.budget.consume("artifact_bytes", 3)
    second.budget.consume("artifact_bytes", 2)
    with pytest.raises(RunStopped, match="artifact_bytes"):
        first.budget.consume("artifact_bytes")
    second.budget.release("artifact_bytes", 2)
    assert (
        first.budget.snapshot()["artifact_bytes"]
        == parent.snapshot()["artifact_bytes"]
        == 3
    )
    assert first.child().budget is first.budget
    assert first.deadline == first.research_deadline == float("inf")
    assert first.budget.unlimited_execution is True
    first.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        first.check_active()
    second.check_active()
    context.check_active()


def test_independent_execution_ignores_legacy_limits_without_mutating_parent_policy() -> (
    None
):
    parent = SharedBudget(max_tools=1, max_decisions=1)
    first, second = LocalBudget(parent), LocalBudget(parent)
    first.consume("tools", 100)
    second.consume("tools", 100)
    for _ in range(100):
        first.consume_research_decision()
        second.consume_research_decision()
    assert first.snapshot()["tools"] == first.snapshot()["decisions"] == 100
    assert parent.snapshot()["tools"] == parent.snapshot()["decisions"] == 200
    assert parent.unlimited_execution is False
    with pytest.raises(RunStopped, match="Shared tools"):
        parent.consume("tools")
    assert first.allocation_snapshot()["limits"] == {"tools": None, "decisions": None}
    assert first.allocation_snapshot()["unlimited_execution"] is True
    with pytest.raises(ValueError, match="execution count"):
        first.consume("tools", -1)


def test_independent_results_and_checkpoint_preserve_complete_answers() -> None:
    answer = "\n".join(
        f"Condition {n}: operative detail and exception." for n in range(500)
    )
    data: dict[str, JsonValue] = {"details": [answer, "last requirement"]}
    context = RunContext()

    def runner(
        _task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        child.consume_research_decision()
        child.budget.consume("tools")
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=answer, data=data)

    pool = WorkerPool(context, runner)
    restored = WorkerPool(RunContext(run_id=context.run_id), done)
    try:
        task = pool.spawn("Question", independent_question=True)
        result = pool.wait_until_all([task])[0]
        assert result.outcome is not None
        assert result.outcome.summary == answer and result.outcome.data == data
        result.outcome.data["details"] = []
        assert pool.list()[0].outcome == pool.results([task])[0].outcome
        original = pool.results([task])[0].outcome
        assert original is not None and original.data == data
        snapshot = pool.export()
        restored.restore(snapshot)
        saved = restored.results([task])[0]
        assert saved.status == TaskStatus.COMPLETED
        assert saved.outcome is not None and saved.outcome.summary == answer
        assert saved.local_budget["used"] == {"tools": 1, "decisions": 1}
        legacy = pool.spawn("Legacy")
        pool.wait_until_all([legacy])
        bounded = pool.list()[-1].outcome
        complete = pool.results([legacy])[0].outcome
        assert bounded is not None and len(bounded.summary) == 3000
        assert complete is not None and complete.summary == answer
    finally:
        pool.close()
        restored.close()


def test_wait_deadline_preserves_completed_and_rejects_late_answer() -> None:
    release, started = threading.Event(), threading.Event()
    context = RunContext(timeout_seconds=10)

    def runner(
        task: str, _child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        if task == "Slow":
            started.set()
            assert release.wait(2)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    pool = WorkerPool(context, runner)
    try:
        complete = pool.spawn("Completed")
        slow = pool.spawn("Slow")
        assert started.wait(1)
        pool.wait(complete, timeout_seconds=1)
        context.research_deadline = time.monotonic() + 0.02
        results = pool.wait_until_all([complete, slow])
        assert results[0].status == TaskStatus.COMPLETED
        assert (
            results[0].outcome is not None and results[0].outcome.summary == "Completed"
        )
        assert results[1].status == TaskStatus.CANCELLED
        context.check_active()
        release.set()
        pool._futures[slow].result(timeout=1)
        assert pool.results([slow])[0].outcome is None
    finally:
        release.set()
        pool.close()


def test_independent_wait_ignores_elapsed_parent_clock_and_preserves_complete_answer() -> (
    None
):
    started, release = threading.Event(), threading.Event()
    context = RunContext(timeout_seconds=10)
    answer = "Kaynaklı koşul ve devamı. " * 1_000

    def runner(
        _task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        started.set()
        assert release.wait(2)
        child.check_research_active()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=answer)

    pool = WorkerPool(context, runner)
    results = []
    try:
        task = pool.spawn("Independent", independent_question=True)
        assert started.wait(1)
        context.deadline = context.research_deadline = time.monotonic() - 1
        waiter = threading.Thread(
            target=lambda: results.extend(pool.wait_until_all([task]))
        )
        waiter.start()
        waiter.join(0.05)
        assert waiter.is_alive()
        release.set()
        waiter.join(1)
        assert not waiter.is_alive()
        assert results[0].status == TaskStatus.COMPLETED
        assert results[0].outcome is not None and results[0].outcome.summary == answer
    finally:
        release.set()
        pool.close()


def test_independent_task_count_and_text_are_unbounded_and_restore_without_old_caps() -> (
    None
):
    context = RunContext(timeout_seconds=float("inf"))
    pool = WorkerPool(context, done, max_tasks=1)
    restored = WorkerPool(RunContext(run_id=context.run_id), done, max_tasks=1)
    try:
        tasks = [
            pool.spawn("Question " * 600, independent_question=True) for _ in range(30)
        ]
        assert len(pool.wait_until_all(tasks)) == 30
        ordinary = pool.spawn("Ordinary")
        pool.wait_until_all([ordinary])
        with pytest.raises(RunStopped, match="capacity"):
            pool.spawn("Another ordinary")
        restored.restore(pool.export())
        assert len(restored.results()) == 31
        assert all(
            item.local_budget["unlimited_execution"] is True
            for item in restored.results(tasks)
        )
    finally:
        pool.close()
        restored.close()


def test_restore_local_spend_and_interruption_without_reexecution() -> None:
    context = RunContext()
    pool = WorkerPool(context, done)
    restored_context = RunContext(run_id=context.run_id)
    restored = WorkerPool(restored_context, done)
    invalid = WorkerPool(RunContext(run_id=context.run_id), done)
    try:
        task = pool.spawn("Question", independent_question=True)
        pool.wait_until_all([task])
        snapshot = pool.export()
        tasks = cast(list[dict[str, JsonValue]], snapshot["tasks"])
        tasks[0]["status"] = TaskStatus.RUNNING.value
        tasks[0]["local_budget"] = {
            "limits": {"tools": 2, "decisions": 2},
            "used": {"tools": 1, "decisions": 2},
        }
        restored_context.budget.restore({"tools": 1, "decisions": 2})
        restored.restore(snapshot)
        result = restored.results([task])[0]
        assert result.status == TaskStatus.INTERRUPTED and result.outcome is None
        assert result.local_budget["used"] == {"tools": 1, "decisions": 2}
        assert restored_context.budget.snapshot()["decisions"] == 2
        assert not restored._futures
        # A .50 checkpoint's old quota is historical metadata, never a renewed cap.
        resumed = restored.followup(task, "Continue")
        followup = restored.wait_until_all([resumed])[0]
        assert followup.status == TaskStatus.COMPLETED
        assert followup.local_budget["unlimited_execution"] is True
        corrupt = copy.deepcopy(snapshot)
        corrupt_tasks = cast(list[dict[str, JsonValue]], corrupt["tasks"])
        corrupt_tasks[0]["local_budget"] = {
            "limits": {"tools": 2, "decisions": 2},
            "used": {"tools": 1, "decisions": -1},
        }
        with pytest.raises(ValueError, match="checkpoint budget"):
            invalid.restore(corrupt)
        assert invalid.results() == []
    finally:
        pool.close()
        restored.close()
        invalid.close()
