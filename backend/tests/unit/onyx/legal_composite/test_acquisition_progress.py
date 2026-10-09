"""Parallel completion reports one truthful batch update without losing originals."""

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite import acquisition
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from tests.unit.onyx.legal_composite.test_review_assessment import original


@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("coalesce_progress", [False, True])
def test_simultaneous_lane_completions_preserve_counts_and_originals(
    monkeypatch: pytest.MonkeyPatch, stopped: bool, coalesce_progress: bool
) -> None:
    progress: list[tuple[int, int, list[SourceKind | None]]] = []
    shutdown_calls: list[tuple[bool, bool]] = []
    ledger = EvidenceLedger()
    context = RunContext()

    def read(arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        name = str(arguments["source"])
        if stopped and name == SourceKind.UNKNOWN.value:
            raise RunStopped("Source deadline reached")
        child.check_active()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original received.",
            evidence=[original(f"Original for {name}.", name)],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read an original in the captured lane.",
                parameters={
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "additionalProperties": False,
                },
                handler=read,
                exposes_public_update=False,
            )
        ]
    )

    class CompletedExecutor:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["max_workers"] == 12

        def submit(
            self, function: Callable[..., ToolOutcome], *args: Any
        ) -> Future[ToolOutcome]:
            future: Future[ToolOutcome] = Future()
            try:
                future.set_result(function(*args))
            except RunStopped as error:
                future.set_exception(error)
            return future

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(acquisition, "ThreadPoolExecutor", CompletedExecutor)
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="need",
                question="Outcome?",
                governing_source="Law",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    actions = [
        SourceAction(
            need_ids=["need"],
            tool="read_original",
            source_kind=kind,
            arguments={"source": kind.value},
        )
        for kind in SourceKind
    ]
    instance = CanonicalAcquirer(
        registry,
        context,
        ledger,
        WorkflowPolicy(max_parallel_tools=12),
        on_batch_progress=lambda rows, pending, completed: progress.append(
            (pending, completed, [row.source_kind for row in rows])
        ),
        coalesce_progress=coalesce_progress,
    )
    instance.acquire(actions, plan)
    assert len(progress) == (2 if coalesce_progress else 13)
    assert progress[0][:2] == (12, 0)
    assert progress[-1][:2] == (0, 12)
    assert progress[0][2] == progress[-1][2] == list(SourceKind)
    if not coalesce_progress:
        assert [row[:2] for row in progress] == [
            (12 - completed, completed) for completed in range(13)
        ]
    assert len(instance.last_receipts) == 12
    assert len(ledger.citation_numbers()) == (11 if stopped else 12)
    assert sum(row["status"] == "truncated" for row in instance.last_receipts) == int(
        stopped
    )
    assert shutdown_calls == [(False, True)]


@pytest.mark.parametrize("coalesce_progress", [False, True])
def test_pending_lane_keeps_five_second_heartbeat(
    monkeypatch: pytest.MonkeyPatch, coalesce_progress: bool
) -> None:
    context = RunContext()
    ledger = EvidenceLedger()
    progress: list[tuple[int, int]] = []
    clock = [acquisition.time.monotonic()]
    scheduled: list[
        tuple[Future[ToolOutcome], Callable[..., ToolOutcome], tuple[Any, ...]]
    ] = []
    wait_count = 0

    def read(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original received.")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read an original.",
                parameters={"type": "object", "properties": {}},
                handler=read,
                exposes_public_update=False,
            )
        ]
    )

    class DeferredExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit(
            self, function: Callable[..., ToolOutcome], *args: Any
        ) -> Future[ToolOutcome]:
            future: Future[ToolOutcome] = Future()
            scheduled.append((future, function, args))
            return future

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            assert (wait, cancel_futures) == (False, True)

    def complete_after_heartbeat(
        _futures: Any, *, timeout: float, return_when: str
    ) -> tuple[set[Future[ToolOutcome]], set[Future[ToolOutcome]]]:
        nonlocal wait_count
        assert timeout == 0.05 and return_when == acquisition.FIRST_COMPLETED
        wait_count += 1
        clock[0] += 5.01
        future, function, args = scheduled[0]
        if wait_count == 1:
            return set(), {future}
        future.set_result(function(*args))
        return {future}, set()

    monkeypatch.setattr(acquisition, "ThreadPoolExecutor", DeferredExecutor)
    monkeypatch.setattr(acquisition, "wait", complete_after_heartbeat)
    monkeypatch.setattr(acquisition.time, "monotonic", lambda: clock[0])
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="need",
                question="Outcome?",
                governing_source="Law",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    instance = CanonicalAcquirer(
        registry,
        context,
        ledger,
        WorkflowPolicy(),
        on_batch_progress=lambda _rows, pending, completed: progress.append(
            (pending, completed)
        ),
        coalesce_progress=coalesce_progress,
    )
    instance.acquire(
        [SourceAction(need_ids=["need"], tool="read_original", arguments={})], plan
    )
    assert progress == [(1, 0), (1, 0), (0, 1)]
