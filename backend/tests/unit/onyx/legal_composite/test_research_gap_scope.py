"""Focused research cannot clear another issue or close unread law without new support."""

from unittest.mock import Mock

import pytest

from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import GapResolution, IssueResearchStep, WorkflowPolicy
from tests.unit.onyx.legal_composite.test_source_requirements import fixture


def test_focused_gap_closure_requires_new_support_and_preserves_other_issue() -> None:
    ledger, plan, requirements, _draft = fixture()
    first, second = plan.needs
    first.evidence_gaps = ["A unresolved law"]
    second.evidence_gaps = ["B unresolved law"]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.requirements.update(requirements, plan, {1, 2})
    closing = IssueResearchStep(
        actions=[],
        ready_to_answer=True,
        remaining_gaps=[],
        issue_gaps={first.need_id: []},
    )
    engine._record_research_gaps(closing, plan, {first.need_id})
    assert first.evidence_gaps == ["A unresolved law"]
    assert second.evidence_gaps == ["B unresolved law"]
    closing.requirements = [requirements[0]]
    closing.gap_resolutions = [
        GapResolution(
            need_id=first.need_id,
            gap="A unresolved law",
            requirement_ids=[requirements[0].requirement_id],
        )
    ]
    engine._record_research_gaps(
        closing,
        plan,
        {first.need_id},
        new_requirement_ids={requirements[0].requirement_id},
    )
    assert first.evidence_gaps == []
    assert second.evidence_gaps == ["B unresolved law"]
    assert engine.research_gaps == ["B unresolved law"]


def test_absent_gap_key_is_not_a_closure_and_out_of_scope_updates_are_rejected() -> (
    None
):
    ledger, plan, _requirements, _draft = fixture()
    first, second = plan.needs
    first.evidence_gaps = ["A unread law"]
    second.evidence_gaps = ["B unread law"]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    step = IssueResearchStep(actions=[], ready_to_answer=True, remaining_gaps=[])
    engine._record_research_gaps(step, plan, {first.need_id})
    assert first.evidence_gaps and second.evidence_gaps
    step.issue_gaps = {second.need_id: []}
    with pytest.raises(InvalidSourceAction, match="unaffected"):
        engine._record_research_gaps(step, plan, {first.need_id})
    assert first.evidence_gaps == ["A unread law"]
    assert second.evidence_gaps == ["B unread law"]


def test_same_issue_new_rule_without_exact_gap_resolution_does_not_clear_missing_law() -> (
    None
):
    ledger, plan, requirements, _draft = fixture()
    first = plan.needs[0]
    first.evidence_gaps = ["Named limiting interaction unread"]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.requirements.update(requirements, plan, {1, 2})
    step = IssueResearchStep(
        actions=[],
        ready_to_answer=True,
        remaining_gaps=[],
        requirements=[requirements[0]],
        issue_gaps={first.need_id: []},
    )
    engine._record_research_gaps(
        step, plan, new_requirement_ids={requirements[0].requirement_id}
    )
    assert first.evidence_gaps == ["Named limiting interaction unread"]
    assert first.evidence_gap_resolutions == []
