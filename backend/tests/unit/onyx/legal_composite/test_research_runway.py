"""Research admission leaves time to inspect completed work before finalization."""

from unittest.mock import Mock

import pytest

from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.models import (
    CompositeWorkflowResult,
    IssueResearchStep,
    SourceAction,
)
from tests.unit.legal_composite.test_staged_discovery import engine as staged_engine
from tests.unit.legal_composite.test_staged_discovery import plan
from tests.unit.onyx.legal_composite.test_repair_runway import engine


@pytest.mark.parametrize("remaining,allowed", [(47, False), (48, False), (49, True)])
def test_reading_reserves_observed_duration_and_selection(
    monkeypatch: pytest.MonkeyPatch, remaining: float, allowed: bool
) -> None:
    assert engine(monkeypatch, remaining)._reading_has_runway() is allowed


@pytest.mark.parametrize(
    "tool,remaining,allowed",
    [
        ("search_corpus", 158, False),
        ("search_corpus", 159, True),
        ("get_regulatory_provision", 66, False),
        ("get_regulatory_provision", 67, True),
        ("read_sections", 123, False),
        ("read_sections", 124, True),
    ],
)
def test_followup_reserves_acquisition_next_reading_and_selection(
    monkeypatch: pytest.MonkeyPatch, tool: str, remaining: float, allowed: bool
) -> None:
    value = engine(monkeypatch, remaining)
    step = IssueResearchStep(
        actions=[SourceAction(tool=tool, arguments={}, need_ids=["issue"])],
        ready_to_answer=False,
        remaining_gaps=["Missing operative support"],
    )
    assert value._followup_has_runway(step) is allowed


def test_last_partial_source_work_is_not_opened_or_marked_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr("onyx.legal_composite.engine.time.monotonic", lambda: now[0])
    instance, gateway, acquirer = staged_engine()
    frozen = plan()
    frozen_before = frozen.model_dump(mode="json")
    instance.evidence_context = Mock(research_deadline=140.0)
    gap = "Requested operative remedy remains unsupported"
    step = IssueResearchStep(
        actions=[frozen.initial_actions[1].model_copy(deep=True)],
        ready_to_answer=False,
        remaining_gaps=[gap],
        issue_gaps={"n2": [gap]},
    )
    calls = 0

    def complete(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            now[0] = 20.0
            return frozen
        now[0] += 8.0
        return step

    gateway.complete.side_effect = complete

    def acquire(*_args: object) -> list[object]:
        now[0] += 80.0
        return []

    acquirer.acquire.side_effect = acquire
    final_payload = []

    def finalize(*_args: object) -> CompositeWorkflowResult:
        final_payload.append(instance._payload("Complete request", ""))
        return CompositeWorkflowResult(answer=None, status="unavailable", gaps=[gap])

    monkeypatch.setattr(instance, "_finalize_semantic", finalize)
    result = instance.run("Complete request")
    assert result.status == "unavailable" and result.answer is None
    assert calls == 2
    acquirer.acquire.assert_called_once()
    assert (
        frozen_before["initial_actions"]
        == frozen.model_dump(mode="json")["initial_actions"]
    )
    assert {need.need_id for need in frozen.needs} == {"n1", "n2", "n3"}
    assert gap in frozen.needs[1].evidence_gaps
    assert (
        not instance.requirements.records() and not instance.ledger.citation_numbers()
    )
    deferred = final_payload[0]["deferred_followup_source_actions"]
    assert deferred == [
        {
            "actions": [step.actions[0].model_dump(mode="json")],
            "material_dependencies": [],
            "status": "unexecuted_navigation",
            "reason": "research_runway",
            "navigation_only": True,
        }
    ]
    assert instance.source_requests[0]["status"] == "completed"
    assert len(instance.source_requests) == 1


def test_closed_budget_cannot_be_bypassed_by_a_time_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = engine(monkeypatch, 1_000)
    value.research_available = lambda: False
    assert not value._reading_has_runway()
    assert not value._followup_has_runway(
        IssueResearchStep(actions=[], ready_to_answer=False, remaining_gaps=[])
    )


def test_multiple_expanded_waves_cannot_inherit_one_wave_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = engine(monkeypatch, 270)
    value.policy = value.policy.model_copy(update={"max_parallel_tools": 12})
    value.plan = plan()
    acquirer = Mock(spec=CanonicalAcquirer)
    acquirer.pending_call_counts.return_value = {"search_corpus": 36}
    value.acquirer = acquirer
    step = IssueResearchStep(
        actions=value.plan.initial_actions,
        ready_to_answer=False,
        remaining_gaps=["Unresolved operative basis"],
    )
    assert not value._followup_has_runway(step)
    acquirer.pending_call_counts.assert_called_once_with(step.actions, value.plan)
    acquirer.acquire.assert_not_called()
    assert value.plan.needs and not value.requirements.records()
