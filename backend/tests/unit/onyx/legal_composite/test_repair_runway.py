"""Focused research must retain time for its actual acquisition and selection."""

from typing import TypeVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.claim_edits import ClaimRepairEdits
from onyx.legal_composite.draft_composition import DraftComposition
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AuthorityDependency,
    IssueResearchPlan,
    IssueResearchStep,
    ReviewCheck,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for
from tests.unit.onyx.legal_composite.test_source_requirements import fixture

T = TypeVar("T", bound=BaseModel)


def engine(monkeypatch: pytest.MonkeyPatch, remaining: float) -> LegalCompositeEngine:
    monkeypatch.setattr("onyx.legal_composite.engine.time.monotonic", lambda: 100.0)
    value = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=EvidenceLedger(),
        policy=WorkflowPolicy(max_call_seconds=75, selection_reserve_seconds=12),
        check_active=lambda: None,
        research_available=lambda: True,
        evidence_context=Mock(research_deadline=100.0 + remaining),
    )
    value._research_seconds = [14.0, 36.0]
    value._source_seconds = {"search_corpus": 110.0, "get_regulatory_provision": 18.0}
    return value


@pytest.mark.parametrize(
    "remaining,expected", [(22.0, False), (66.0, False), (67.0, True)]
)
def test_planning_retains_observed_acquisition_and_selection(
    monkeypatch: pytest.MonkeyPatch, remaining: float, expected: bool
) -> None:
    assert engine(monkeypatch, remaining)._repair_research_has_runway() is expected


def test_new_search_is_rechecked_after_planning_consumes_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = engine(monkeypatch, 50)
    search = IssueResearchStep(
        actions=[
            SourceAction(
                tool="search_corpus",
                arguments={"query": "generic gap"},
                need_ids=["issue"],
            )
        ],
        ready_to_answer=False,
        remaining_gaps=[],
    )
    direct = IssueResearchStep(
        actions=[
            SourceAction(
                tool="get_regulatory_provision",
                arguments={"article": "source-backed"},
                need_ids=["issue"],
            )
        ],
        ready_to_answer=False,
        remaining_gaps=[],
    )
    assert not value._repair_actions_have_runway(search)
    assert value._repair_actions_have_runway(direct)


def test_unseen_tool_does_not_inherit_a_fast_other_tool_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = IssueResearchStep(
        actions=[SourceAction(tool="read_sections", arguments={}, need_ids=["issue"])],
        ready_to_answer=False,
        remaining_gaps=[],
    )
    assert not engine(monkeypatch, 50)._repair_actions_have_runway(step)


def test_spend_or_cancellation_stop_precedes_time_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = engine(monkeypatch, 1_000)
    value.research_available = lambda: False
    assert not value._repair_research_has_runway()
    assert value._repair_runway() == 0


def test_unrecorded_requirement_uses_existing_evidence_without_research() -> None:
    ledger, plan, requirements, draft = fixture()
    draft = StructuredDraftAnswer(
        sections=[
            section.model_copy(
                update={
                    "text": "",
                    "claim_ids": [
                        claim.claim_id
                        for claim in draft.claims
                        if claim.section_id == section.section_id
                    ],
                }
            )
            for section in draft.sections
        ],
        claims=[
            claim.model_copy(
                update={"answer_excerpt": claim.answer_excerpt + f" [[{index}]]"}
            )
            for index, claim in enumerate(draft.claims, start=1)
        ],
        unresolved_need_ids=[],
    )
    calls: list[type[BaseModel]] = []

    class Gateway:
        last_call_id: str | None = "test"
        last_delivered_citations = {1, 2}

        def complete(
            self,
            system: str,
            payload: dict[str, JsonValue],
            response_type: type[T],
            flow: LLMFlow,
            finalizing: bool = False,
        ) -> T:
            del system, payload, flow, finalizing
            calls.append(response_type)
            if response_type is DraftComposition:
                return response_type.model_validate(composition_for(draft).model_dump())
            assert response_type is ClaimRepairEdits
            return response_type.model_validate(
                {
                    "claims": [],
                    "unresolved_need_ids": [],
                    "requirements": [
                        requirement.model_dump() for requirement in requirements
                    ],
                }
            )

    class Reviewer:
        def expected_checks(
            self,
            request: str,
            plan: IssueResearchPlan,
            draft: StructuredDraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> dict[str, ReviewQuestion]:
            return build_checks(
                request,
                plan,
                draft,
                requirements,
                dependencies,
                delivered,
                previous=previous,
                affected_sections=affected_sections,
                ledger=ledger,
            )

        def review(
            self,
            request: str,
            plan: IssueResearchPlan,
            draft: StructuredDraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> SemanticReview:
            checks = self.expected_checks(
                request,
                plan,
                draft,
                requirements,
                dependencies,
                delivered,
                previous=previous,
                affected_sections=affected_sections,
            )
            return SemanticReview(
                checks=[
                    ReviewCheck(
                        check_id=check.check_id,
                        need_ids=check.need_ids,
                        section_ids=check.section_ids,
                        status="addressed",
                        confidence=0.99,
                    )
                    for check in checks.values()
                ]
            )

    acquirer = Mock()
    value = LegalCompositeEngine(
        gateway=Gateway(),
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
        reviewer=Reviewer(),
    )
    value.plan = plan
    result = value._finalize_semantic("Generic issues a and b", "", None, plan)
    assert result.status == "verified", result.gaps
    assert calls == [DraftComposition, ClaimRepairEdits]
    assert set(row.requirement_id for row in value.requirements.records()) == {
        "r_a",
        "r_b",
    }
    acquirer.acquire.assert_not_called()
