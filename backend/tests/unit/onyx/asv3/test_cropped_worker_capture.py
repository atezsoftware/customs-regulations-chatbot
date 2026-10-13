"""Worker tasks must not inherit a caller's in-flight compose provenance."""

from collections.abc import Callable
from contextvars import ContextVar

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_article_bindings as articles
from onyx.asv3 import cropped_composed_topics as topics
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    RunStopped,
    TaskStatus,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.workers import WorkerPool
from tests.unit.onyx.asv3.test_cropped_composed_topics import setup


@pytest.mark.parametrize("child_compose", [False, True])
@pytest.mark.parametrize("nested_spawn", [False, True])
def test_actual_composed_spawn_keeps_child_and_parent_capture_independent(
    child_compose: bool, nested_spawn: bool
) -> None:
    current, ledger, _, store, registry, call, _ = setup()
    current.services["cropped_article_bindings"] = articles.CroppedArticleBindings(
        current, store.request
    )
    inherited: ContextVar[str] = ContextVar("worker_test_tenant_and_trace")
    token = inherited.set("tenant-and-trace-unchanged")
    seen: list[tuple[str, int, list[str]]] = []
    child_receipts: list[ToolReceipt] = []

    def run(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        assert articles._COMPOSED_OWNER.get() is None
        assert topics._OWNER.get() is None
        assert inherited.get() == "tenant-and-trace-unchanged"
        seen.append(
            (
                str(child.services["task_id"]),
                child.depth,
                child.services["task_need_ids"],
            )
        )
        child_call = (
            call.model_copy(update={"call_id": f"child-{child.depth}"}, deep=True)
            if child_compose
            else CapabilityCall(
                name="read_provision",
                arguments={"source_id": "source-2", "_need_id": "B"},
            )
        )
        outcome = registry.dispatch(child_call, child)
        assert outcome.status == OutcomeStatus.FOUND
        receipt = ToolReceipt(call=child_call, outcome=outcome, elapsed_seconds=0)
        child_receipts.append(receipt)
        if child_compose:
            captured = store.composed_capture.take(receipt, child)
            assert [step.step_id for step in captured] == ["a", "b"]
            assert [step.call.arguments["_need_id"] for step in captured] == ["A", "B"]
        if nested_spawn and task == "child":
            with (
                articles.cropped_composed_read_capture(call, child),
                topics.cropped_composed_topic_capture(call, child),
            ):
                owner = articles._COMPOSED_OWNER.get()
                topic_owner = topics._OWNER.get()
                nested = pool.spawn("grandchild", request_context=child, need_ids=["A"])
                result = pool.wait(nested, timeout_seconds=5)[0]
                assert result.status == TaskStatus.COMPLETED
                assert articles._COMPOSED_OWNER.get() is owner
                assert topics._OWNER.get() is topic_owner
                assert topic_owner is not None and not topic_owner.invalid
        return outcome

    pool = WorkerPool(current, run)

    def spawn(_args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        parent_owner = articles._COMPOSED_OWNER.get()
        topic_owner = topics._OWNER.get()
        assert parent_owner is not None and topic_owner is not None
        task_id = pool.spawn("child", request_context=context, need_ids=["B"])
        result = pool.wait(task_id, timeout_seconds=5)[0]
        assert result.status == TaskStatus.COMPLETED
        assert articles._COMPOSED_OWNER.get() is parent_owner
        assert topics._OWNER.get() is topic_owner
        assert not topic_owner.invalid
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Spawned", data={"task_id": task_id}
        )

    registry.register(
        ToolSpec(
            name="spawn_researcher",
            description="Run actual child task",
            parameters={"type": "object", "properties": {}},
            handler=spawn,
            parallel_safe=True,
        )
    )
    root_call = call.model_copy(deep=True)
    root_call.arguments["steps"] = [
        {
            "id": "a",
            "tool": "read_provision",
            "arguments": {"source_id": "source-1", "_need_id": "A"},
        },
        {"id": "spawn", "tool": "spawn_researcher", "arguments": {"_need_id": "B"}},
    ]
    before = ledger.export()
    try:
        outcome = registry.dispatch(root_call, current)
        assert outcome.status == OutcomeStatus.FOUND
        captured = store.composed_capture.take(
            ToolReceipt(call=root_call, outcome=outcome, elapsed_seconds=0), current
        )
        assert {step.step_id for step in captured} == {"a", "spawn"}
        assert len(seen) == (2 if nested_spawn else 1)
        assert [depth for _, depth, _ in seen] == ([1, 2] if nested_spawn else [1])
        assert [needs for _, _, needs in seen] == (
            [["B"], ["A"]] if nested_spawn else [["B"]]
        )
        assert all(receipt.outcome.evidence for receipt in child_receipts)
        assert ledger.export() == before
        assert articles._COMPOSED_OWNER.get() is None
        assert topics._OWNER.get() is None
    finally:
        pool.close()
        inherited.reset(token)


@pytest.mark.parametrize("exception", [RuntimeError, RunStopped])
def test_capture_isolation_restores_outer_owners_after_failure(
    exception: type[Exception],
) -> None:
    current, _, _, store, _, call, _ = setup()
    current.services["cropped_article_bindings"] = articles.CroppedArticleBindings(
        current, store.request
    )
    with (
        articles.cropped_composed_read_capture(call, current),
        topics.cropped_composed_topic_capture(call, current),
    ):
        owner = articles._COMPOSED_OWNER.get()
        topic_owner = topics._OWNER.get()
        with pytest.raises(exception):
            with (
                articles.isolated_cropped_read_capture(),
                topics.isolated_cropped_topic_capture(),
            ):
                assert articles._COMPOSED_OWNER.get() is None
                assert topics._OWNER.get() is None
                raise exception("stop")
        assert articles._COMPOSED_OWNER.get() is owner
        assert topics._OWNER.get() is topic_owner
        assert topic_owner is not None and not topic_owner.invalid


def test_foreign_owner_still_rejected_outside_worker_boundary() -> None:
    current, _, _, store, registry, call, _ = setup()
    current.services["cropped_article_bindings"] = articles.CroppedArticleBindings(
        current, store.request
    )
    child = current.child()
    child.services["task_id"] = "foreign"
    with articles.cropped_composed_read_capture(call, current):
        outcome = registry.dispatch(
            CapabilityCall(
                name="read_provision",
                arguments={"source_id": "source-2", "_need_id": "B"},
            ),
            child,
        )
        assert outcome.status == OutcomeStatus.ERROR
        assert outcome.evidence == []


@pytest.mark.parametrize(
    "cropped,cheap", [(False, False), ("true", False), (True, True)]
)
def test_other_workflows_keep_existing_copied_context(
    cropped: bool | str, cheap: bool
) -> None:
    current, _, _, store, _, call, _ = setup()
    current.services["cropped_article_bindings"] = articles.CroppedArticleBindings(
        current, store.request
    )
    with (
        articles.cropped_composed_read_capture(call, current),
        topics.cropped_composed_topic_capture(call, current),
    ):
        owner = articles._COMPOSED_OWNER.get()
        topic_owner = topics._OWNER.get()
        current.services.update(asv3_exp_cropped=cropped, asv3_exp_cheap=cheap)

        def run(
            _task: str, _child: RunContext, _updates: Callable[[], list[str]]
        ) -> ToolOutcome:
            assert articles._COMPOSED_OWNER.get() is owner
            assert topics._OWNER.get() is topic_owner
            return ToolOutcome(status=OutcomeStatus.FOUND, summary="Unchanged")

        pool = WorkerPool(current, run)
        try:
            task_id = pool.spawn("ordinary", need_ids=["B"])
            assert (
                pool.wait(task_id, timeout_seconds=5)[0].status == TaskStatus.COMPLETED
            )
        finally:
            pool.close()
