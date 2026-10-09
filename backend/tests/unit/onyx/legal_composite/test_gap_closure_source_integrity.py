"""An exact quotation for one effect cannot erase an unread effect in the same issue."""

from unittest.mock import Mock

import pytest

from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    GapResolution,
    IssueResearchStep,
    WorkflowPolicy,
)
from tests.unit.onyx.legal_composite.test_source_requirements import fixture


def test_new_same_issue_requirement_does_not_hide_an_unread_interaction() -> None:
    ledger, plan, requirements, _draft = fixture()
    need = plan.needs[0]
    unread_interaction = "Sonraki denetim kararının sınırlayıcı etkisi henüz okunmadı."
    need.evidence_gaps = [unread_interaction]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    requirement = requirements[0]
    engine.requirements.update([requirement], plan, {1, 2})
    step = IssueResearchStep(
        actions=[],
        ready_to_answer=True,
        remaining_gaps=[],
        requirements=[requirement],
        issue_gaps={need.need_id: []},
    )

    # The quote establishes document requirements, not the still-unread decision.
    engine._record_research_gaps(
        step,
        plan,
        {need.need_id},
        new_requirement_ids={requirement.requirement_id},
    )

    assert unread_interaction in need.evidence_gaps
    assert unread_interaction in engine.research_gaps


def test_final_reading_closes_only_named_gap_and_keeps_originals_unchanged() -> None:
    ledger, plan, requirements, _draft = fixture()
    first, second = plan.needs
    solved = "A işlemine ait belge yükümlülüğü henüz okunmadı."
    remaining = "Sonraki denetim kararının sınırlayıcı etkisi henüz okunmadı."
    other_issue = "B başvurusunun süre başlangıcı henüz okunmadı."
    first.evidence_gaps = [solved, remaining]
    second.evidence_gaps = [other_issue]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.requirements.update([requirements[1]], plan, {1, 2})
    originals_before = ledger.export()
    resolution = GapResolution(
        need_id=first.need_id,
        gap=solved,
        requirement_ids=[requirements[0].requirement_id],
    )

    engine._accept_draft_reading(
        [requirements[0]], [resolution], plan, {1, 2}, {first.need_id}
    )

    assert first.evidence_gaps == [remaining]
    assert second.evidence_gaps == [other_issue]
    assert first.evidence_gap_resolutions == [resolution]
    assert second.evidence_gap_resolutions == []
    assert engine.research_gaps == [remaining, other_issue]
    assert engine.requirements.citations() == {1, 2}
    assert ledger.export() == originals_before
    assert engine.requirements.export()[1]["original_bindings"]


@pytest.mark.parametrize("fault", ["quotation", "delivery"])
def test_invalid_final_support_leaves_gap_history_and_requirements_unchanged(
    fault: str,
) -> None:
    ledger, plan, requirements, _draft = fixture()
    first, second = plan.needs
    first.evidence_gaps = ["Belge yükümlülüğü henüz okunmadı."]
    second.evidence_gaps = ["Başvuru süresi henüz okunmadı."]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.requirements.update([requirements[1]], plan, {1, 2})
    plan_before = plan.model_dump()
    requirements_before = engine.requirements.export()
    originals_before = ledger.export()
    requirement = requirements[0].model_copy(deep=True)
    delivered = {1, 2}
    if fault == "quotation":
        requirement.supports[0].quotation = "Orijinalde bulunmayan bir kural."
    else:
        delivered = {2}
    resolution = GapResolution(
        need_id=first.need_id,
        gap=first.evidence_gaps[0],
        requirement_ids=[requirement.requirement_id],
    )

    with pytest.raises(InvalidSourceAction, match="exact delivered original"):
        engine._accept_draft_reading(
            [requirement], [resolution], plan, delivered, {first.need_id}
        )

    assert plan.model_dump() == plan_before
    assert engine.requirements.export() == requirements_before
    assert ledger.export() == originals_before
