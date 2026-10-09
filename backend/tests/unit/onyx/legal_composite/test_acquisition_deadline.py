"""A research deadline preserves already completed source receipts and originals."""

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
from onyx.legal_composite import acquisition
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from tests.unit.onyx.legal_composite.test_review_assessment import original


def test_deadline_retains_completed_receipts_and_cancels_pending_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = EvidenceLedger()
    context = RunContext()

    def check_research() -> None:
        if ledger.citation_numbers():
            raise RunStopped("Research deadline reached; finalization retained")

    monkeypatch.setattr(context, "check_research_active", check_research)

    def read_original(
        arguments: dict[str, JsonValue], child: RunContext
    ) -> ToolOutcome:
        assert arguments == {}
        child.check_active()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Source original delivered.",
            evidence=[original("Completed controlling original.", "completed")],
        )

    registry = CapabilityRegistry(
        ToolSpec(
            name=name,
            description="Read an authorized original.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=read_original,
            exposes_public_update=False,
        )
        for name in ("read_first", "read_second")
    )
    pending: Future[ToolOutcome] = Future()
    shutdown_calls: list[tuple[bool, bool]] = []
    progress: list[tuple[int, int]] = []

    class ControlledExecutor:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["max_workers"] == 4
            self.calls = 0

        def submit(
            self, function: Callable[..., ToolOutcome], *args: Any
        ) -> Future[ToolOutcome]:
            self.calls += 1
            if self.calls > 1:
                return pending
            completed: Future[ToolOutcome] = Future()
            completed.set_result(function(*args))
            return completed

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(acquisition, "ThreadPoolExecutor", ControlledExecutor)
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="rule",
                question="Kaynaklı sonuç?",
                governing_source="Law",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    actions = [
        SourceAction(need_ids=["rule"], tool=name, arguments={})
        for name in ("read_first", "read_second")
    ]
    acquirer = CanonicalAcquirer(
        registry,
        context,
        ledger,
        WorkflowPolicy(),
        on_batch_progress=lambda _rows, active, completed: progress.append(
            (active, completed)
        ),
    )
    with pytest.raises(RunStopped, match="Research deadline"):
        acquirer.acquire(actions, plan)
    receipts = acquirer.last_receipts
    assert len(receipts) == 2
    assert receipts[0]["tool"] == "read_first" and receipts[0]["citations"] == [1]
    assert receipts[1]["tool"] == "read_second" and receipts[1]["status"] == "truncated"
    assert receipts[1]["citations"] == []
    item = ledger.get(1)
    assert item is not None and item.text == "Completed controlling original."
    assert pending.cancelled()
    assert shutdown_calls == [(False, True)]
    assert progress == [(2, 0), (1, 1)]
