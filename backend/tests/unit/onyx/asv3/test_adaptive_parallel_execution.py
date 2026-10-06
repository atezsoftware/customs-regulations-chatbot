import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    TaskStatus,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.question_research import QuestionResearch
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool


def test_host_assignments_all_start_and_preserve_long_bodies_in_order() -> None:
    count = 5
    context = RunContext(language="en", budget=SharedBudget(unlimited_execution=True))
    entered = threading.Condition()
    started: set[str] = set()
    release = threading.Event()
    bodies = {
        f"Issue {index}": (f"Detail {index} \u0131\u015f\u0131k. " * 2000) + "  \n"
        for index in range(count)
    }

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        with entered:
            started.add(task)
            entered.notify_all()
        assert release.wait(5)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary=bodies[task],
            data={"parallel_answer_receipt": child.services["assignment_id"]},
        )

    with closing(WorkerPool(context, runner, max_workers=2)) as workers:
        research = QuestionResearch(
            context, workers, ["All outcomes"], host_assembly=True
        )
        research.accepted_answer_guard = lambda _item: None
        assignments: list[JsonValue] = [
            {
                "question_id": f"q{index}",
                "question": task,
                "answer_title": f"Topic {index}",
                "parent_question_ids": [1],
                "public_title": "Checking the assigned outcome",
                "public_message": "Examining its applicable original conditions.",
            }
            for index, task in enumerate(bodies)
        ]
        concurrent_calls = threading.Barrier(3)

        def research_batch() -> ToolOutcome:
            concurrent_calls.wait(timeout=5)
            return research.research_questions({"questions": assignments}, context)

        with ThreadPoolExecutor(max_workers=2) as control:
            futures = [control.submit(research_batch) for _ in range(2)]
            concurrent_calls.wait(timeout=5)
            try:
                with entered:
                    assert entered.wait_for(lambda: len(started) == count, timeout=5)
                assert all(
                    task.status == TaskStatus.RUNNING
                    for task in workers.results(full=True)
                )
            finally:
                release.set()
            assert all(
                future.result(timeout=5).status == OutcomeStatus.FOUND
                for future in futures
            )
        assert [item["answer"] for item in research.answers] == list(bodies.values())
        answer = cast(str, context.services["assembled_answer"])
        assert answer == "\n\n".join(
            f"## {index + 1}. Topic {index}\n\n{body}"
            for index, body in enumerate(bodies.values())
        )
        # Completed replay does not resize the pool or run a researcher again.
        assert research.research_questions({}, context).status == OutcomeStatus.FOUND
        assert len(started) == count


def test_legacy_worker_capacity_is_unchanged_without_preparation() -> None:
    release = threading.Event()
    entered = threading.Condition()
    started = 0

    def runner(
        _task: str, _child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        nonlocal started
        with entered:
            started += 1
            entered.notify_all()
        assert release.wait(5)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(RunContext(), runner, max_workers=2)) as workers:
        tasks = [
            workers.spawn(f"Outcome {index}", independent_question=True)
            for index in range(5)
        ]
        try:
            with entered:
                assert entered.wait_for(lambda: started == 2, timeout=5)
            statuses = [task.status for task in workers.results(full=True)]
            assert statuses.count(TaskStatus.RUNNING) == 2
            assert statuses.count(TaskStatus.QUEUED) == 3
            with pytest.raises(ValueError, match="before spawning"):
                workers.prepare_independent_batch(5)
        finally:
            release.set()
        assert len(workers.wait_until_all(tasks)) == 5


@pytest.mark.parametrize(
    "resource,capacity", [("model_slots", 4), ("tool_slots", 4), ("source_slots", 2)]
)
def test_adaptive_researchers_keep_shared_physical_capacity(
    resource: str, capacity: int
) -> None:
    count = 7
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    entered = threading.Condition()
    release = threading.Event()
    started = active = peak = 0

    def runner(
        _task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        nonlocal started, active, peak
        gate = cast(threading.BoundedSemaphore, getattr(child.budget, resource))
        assert gate is getattr(context.budget, resource)
        with entered:
            started += 1
            entered.notify_all()
        assert gate.acquire(timeout=5)
        try:
            with entered:
                active += 1
                peak = max(peak, active)
                entered.notify_all()
            assert release.wait(5)
        finally:
            with entered:
                active -= 1
            gate.release()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(context, runner, max_workers=2)) as workers:
        workers.prepare_independent_batch(count)
        tasks = [
            workers.spawn(f"Outcome {index}", independent_question=True)
            for index in range(count)
        ]
        try:
            with entered:
                assert entered.wait_for(
                    lambda: started == count and active == capacity, timeout=5
                )
            assert peak == capacity
        finally:
            release.set()
        workers.wait_until_all(tasks)
        assert peak == capacity


def test_waiting_source_researchers_do_not_block_independent_model_work() -> None:
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    sources_started = threading.Barrier(3)
    release = threading.Event()
    model_finished = threading.Event()

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        gate = (
            child.budget.source_slots
            if task.startswith("Source")
            else child.budget.model_slots
        )
        assert gate.acquire(timeout=5)
        try:
            if task.startswith("Source"):
                sources_started.wait(timeout=5)
                assert release.wait(5)
            else:
                model_finished.set()
        finally:
            gate.release()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete")

    with closing(WorkerPool(context, runner, max_workers=2)) as workers:
        workers.prepare_independent_batch(3)
        tasks = [
            workers.spawn(task, independent_question=True)
            for task in ["Source one", "Source two", "Model decision"]
        ]
        try:
            sources_started.wait(timeout=5)
            assert model_finished.wait(5)
            assert not release.is_set()
        finally:
            release.set()
        workers.wait_until_all(tasks)


def test_adaptive_tool_batch_uses_free_capacity_while_source_calls_wait() -> None:
    parent = RunContext(budget=SharedBudget(unlimited_execution=True))
    context = parent.independent_child()
    sources_started = threading.Barrier(3)
    release = threading.Event()
    entered = threading.Condition()
    model_finished = 0

    def read_source(
        _arguments: dict[str, JsonValue], caller: RunContext
    ) -> ToolOutcome:
        assert caller.budget.source_slots.acquire(timeout=5)
        try:
            sources_started.wait(timeout=5)
            assert release.wait(5)
        finally:
            caller.budget.source_slots.release()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Read complete")

    def model_step(_arguments: dict[str, JsonValue], caller: RunContext) -> ToolOutcome:
        nonlocal model_finished
        assert caller.budget.model_slots.acquire(timeout=5)
        try:
            with entered:
                model_finished += 1
                entered.notify_all()
        finally:
            caller.budget.model_slots.release()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Decision complete")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_source",
                description="Read original",
                parameters={"type": "object"},
                handler=read_source,
            ),
            ToolSpec(
                name="model_step",
                description="Prepare next step",
                parameters={"type": "object"},
                handler=model_step,
            ),
        ]
    )
    decisions = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return Decision(
                calls=[CapabilityCall(name="read_source") for _ in range(2)]
                + [CapabilityCall(name="model_step") for _ in range(3)]
            )
        return Decision(answer="Complete")

    harness = Harness(
        request="Independent outcome",
        context=context,
        registry=registry,
        decide=decide,
        max_workers=2,
        adaptive_tool_parallelism=True,
    )
    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(harness.run)
        try:
            sources_started.wait(timeout=5)
            with entered:
                assert entered.wait_for(lambda: model_finished == 3, timeout=5)
            assert not release.is_set()
        finally:
            release.set()
        assert future.result(timeout=5).answer == "Complete"
    assert decisions == 2
    assert len(harness.receipts) == 5


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_independent_batch_size_is_rejected(count: int) -> None:
    with closing(
        WorkerPool(
            RunContext(),
            lambda *_args: ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete"),
        )
    ) as workers:
        with pytest.raises(ValueError, match="positive integer"):
            workers.prepare_independent_batch(count)


def test_closed_or_cancelled_pool_cannot_prepare_new_capacity() -> None:
    context = RunContext()
    workers = WorkerPool(
        context,
        lambda *_args: ToolOutcome(status=OutcomeStatus.FOUND, summary="Complete"),
    )
    workers.close()
    with pytest.raises(RunStopped, match="closed"):
        workers.prepare_independent_batch(3)
    context.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        workers.prepare_independent_batch(3)
