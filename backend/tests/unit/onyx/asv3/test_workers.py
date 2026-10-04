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
    first = context.independent_child(max_tools=2, max_decisions=2)
    second = context.independent_child(max_tools=2, max_decisions=2)
    first.budget.consume("tools", 2)
    with pytest.raises(RunStopped, match="Local tools"):
        first.budget.consume("tools")
    second.budget.consume("tools")
    for _ in range(2):
        first.consume_research_decision()
    with pytest.raises(RunStopped, match="Local decisions"):
        first.consume_research_decision()
    second.consume_research_decision()
    assert parent.snapshot()["tools"] == parent.snapshot()["decisions"] == 3
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
    assert first.deadline == context.research_deadline
    first.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        first.check_active()
    second.check_active()
    context.check_active()


def test_global_rejection_does_not_spend_local_quota() -> None:
    parent = SharedBudget(max_tools=1)
    first = LocalBudget(parent, max_tools=3, max_decisions=2)
    second = LocalBudget(parent, max_tools=3, max_decisions=2)
    first.consume("tools")
    with pytest.raises(RunStopped, match="Shared tools"):
        second.consume("tools")
    assert second.snapshot()["tools"] == 0
    assert parent.snapshot()["tools"] == 1


@pytest.mark.parametrize("limit", [0, -1, True])
def test_local_limits_are_strict_positive_integers(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        LocalBudget(SharedBudget(), max_tools=limit, max_decisions=2)


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
        complete = pool.spawn("Completed", independent_question=True)
        slow = pool.spawn("Slow", independent_question=True)
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


def test_restore_local_spend_and_interruption_without_reexecution() -> None:
    context = RunContext()
    pool = WorkerPool(context, done)
    restored_context = RunContext(run_id=context.run_id)
    restored = WorkerPool(restored_context, done)
    invalid = WorkerPool(RunContext(run_id=context.run_id), done)
    try:
        task = pool.spawn(
            "Question", independent_question=True, max_tools=2, max_decisions=2
        )
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
        with pytest.raises(RunStopped, match="Independent question budget"):
            restored.followup(task, "Continue")
        corrupt = copy.deepcopy(snapshot)
        corrupt_tasks = cast(list[dict[str, JsonValue]], corrupt["tasks"])
        corrupt_tasks[0]["local_budget"] = {
            "limits": {"tools": 2, "decisions": 2},
            "used": {"tools": 1, "decisions": 3},
        }
        with pytest.raises(ValueError, match="checkpoint budget"):
            invalid.restore(corrupt)
        assert invalid.results() == []
    finally:
        pool.close()
        restored.close()
        invalid.close()
