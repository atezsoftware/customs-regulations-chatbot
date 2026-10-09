"""Disclosed unread law permits a partial answer without waiving correctness or identity."""

from typing import TypeVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.claim_edits import ClaimRepairEdits
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AnswerSection,
    AuthorityDependency,
    DependencyOrigin,
    GapResolution,
    IssueResearchPlan,
    IssueResearchStep,
    ReviewCheck,
    SemanticReview,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.test_source_requirements import fixture

T = TypeVar("T", bound=BaseModel)
SOURCE_GAP = (
    "Sonraki denetim kararının bu işlem üzerindeki sınırlayıcı etkisi okunamadı."
)


class _Gateway:
    last_call_id: str | None = "partial-answer"
    last_delivered_citations = {1, 2}

    def __init__(self, draft: StructuredDraftAnswer) -> None:
        self.draft = draft
        self.calls: list[type[BaseModel]] = []

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[T],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> T:
        del system, flow, finalizing
        self.calls.append(response_type)
        if response_type is StructuredDraftAnswer:
            return response_type.model_validate(self.draft.model_dump())
        if response_type is IssueResearchStep:
            return response_type.model_validate(
                {
                    "actions": [],
                    "ready_to_answer": True,
                    "remaining_gaps": [SOURCE_GAP],
                    "issue_gaps": {"a": [SOURCE_GAP]},
                }
            )
        assert response_type is ClaimRepairEdits
        affected = payload["affected_section_ids"]
        assert isinstance(affected, list)
        return response_type.model_validate(
            {
                "claims": [],
                "unresolved_need_ids": self.draft.unresolved_need_ids,
            }
        )


class _Reviewer:
    def __init__(self, ledger: EvidenceLedger, fault: str | None = None) -> None:
        self.ledger = ledger
        self.fault = fault
        self.calls = 0

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
            previous,
            affected_sections,
            ledger=self.ledger,
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
        self.calls += 1
        questions = self.expected_checks(
            request,
            plan,
            draft,
            requirements,
            dependencies,
            delivered,
            previous,
            affected_sections,
        )
        checks = {
            identity: ReviewCheck(
                check_id=identity,
                need_ids=question.need_ids,
                section_ids=question.section_ids,
                status="gap" if identity == "evidence:a" else "addressed",
                confidence=0.99,
            )
            for identity, question in questions.items()
        }
        evidence = checks["evidence:a"]
        issue = checks["issue:a"]
        if self.fault in {"evidence:incorrect", "evidence:uncertain"}:
            evidence.status = (
                "incorrect" if self.fault == "evidence:incorrect" else "uncertain"
            )
        elif self.fault == "evidence:low_confidence":
            evidence.confidence = 0.79
        elif self.fault == "evidence:need_identity":
            evidence.need_ids = ["b"]
        elif self.fault == "evidence:section_identity":
            evidence.section_ids = ["s_b"]
        elif self.fault == "issue:incorrect":
            issue.status = "incorrect"
        elif self.fault == "issue:uncertain":
            issue.status = "uncertain"
        elif self.fault == "issue:not_applicable":
            issue.status = "not_applicable"
        elif self.fault == "issue:low_confidence":
            issue.confidence = 0.79
        elif self.fault == "issue:need_identity":
            issue.need_ids = ["b"]
        elif self.fault == "issue:section_identity":
            issue.section_ids = ["s_b"]
        elif self.fault == "gap_resolution":
            checks["gap-resolution:a:0"].status = "gap"
        elif self.fault == "missing_check":
            checks.pop("issue:a")
        rows = list(checks.values())
        if self.fault == "duplicate_check":
            rows.append(evidence.model_copy(deep=True))
        return SemanticReview(
            checks=rows,
            failure="review context unavailable" if self.fault == "provider" else None,
        )


def _engine(
    *,
    max_reviews: int = 1,
    research_available: bool = False,
    fault: str | None = None,
) -> tuple[LegalCompositeEngine, IssueResearchPlan, _Gateway, _Reviewer]:
    ledger, plan, requirements, draft = fixture()
    plan.needs[0].evidence_gaps = [SOURCE_GAP]
    draft.sections[0].text += f" Bu konuda kesin sonuç verilemiyor: {SOURCE_GAP}"
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id=section.section_id,
                need_ids=section.need_ids,
                text="",
                claim_ids=[
                    claim.claim_id
                    for claim in draft.claims
                    if claim.section_id == section.section_id
                ],
            )
            for section in draft.sections
        ],
        claims=[
            claim.model_copy(
                deep=True,
                update={
                    "answer_excerpt": next(
                        section.text
                        for section in draft.sections
                        if section.section_id == claim.section_id
                    )
                },
            )
            for claim in draft.claims
        ],
        unresolved_need_ids=["a"],
    )
    if fault == "unresolved_missing":
        draft.unresolved_need_ids = []
    elif fault == "recorded_gap_missing":
        plan.needs[0].evidence_gaps = []
    elif fault == "gap_resolution":
        plan.needs[0].evidence_gap_resolutions = [
            GapResolution(
                need_id="a", gap="Önceki izin etkileşimi", requirement_ids=["r_a"]
            )
        ]
    gateway = _Gateway(draft)
    reviewer = _Reviewer(ledger, fault)
    acquirer = Mock()
    acquirer.definitions.return_value = []
    acquirer.acquire.return_value = []
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=max_reviews),
        check_active=lambda: None,
        research_available=lambda: research_available,
        reviewer=reviewer,
    )
    engine.plan = plan
    engine.research_gaps = list(plan.needs[0].evidence_gaps)
    engine.requirements.update(requirements, plan, {1, 2})
    if fault == "original_integrity":
        ledger._items[1].text_hash = "0" * 64
    elif fault == "dependency_integrity":
        original = ledger.get(1)
        assert original is not None
        engine.dependencies = [
            AuthorityDependency(
                edge_id="material-rule",
                need_ids=["a"],
                instrument_name="Faaliyet Yönetmeliği",
                article="18",
                origins=[
                    DependencyOrigin(
                        citation=1,
                        source_id=original.source_id,
                        chunk_id=original.chunk_id,
                        text_hash="0" * 64,
                    )
                ],
                governing_citations=[1],
            )
        ]
    return engine, plan, gateway, reviewer


@pytest.mark.parametrize(
    "max_reviews,research_available",
    [(2, False), (1, True)],
    ids=["research-unavailable", "final-attempt"],
)
def test_precisely_disclosed_source_gap_publishes_partial_at_boundary(
    max_reviews: int, research_available: bool
) -> None:
    engine, plan, gateway, reviewer = _engine(
        max_reviews=max_reviews, research_available=research_available
    )
    originals_before = engine.ledger.export()
    result = engine._finalize_semantic("A ve B işlemlerini açıklayın.", "", None, plan)

    assert result.status == "partial"
    assert result.answer is not None and SOURCE_GAP in result.answer
    assert result.gaps == [SOURCE_GAP]
    assert plan.needs[0].evidence_gaps == [SOURCE_GAP]
    assert engine.ledger.export() == originals_before
    assert reviewer.calls == 1
    assert gateway.calls == [StructuredDraftAnswer]


def test_source_gap_is_researched_and_rechecked_before_last_allowed_review() -> None:
    engine, plan, gateway, reviewer = _engine(max_reviews=2, research_available=True)
    result = engine._finalize_semantic("A ve B işlemlerini açıklayın.", "", None, plan)

    assert result.status == "partial" and SOURCE_GAP in result.gaps
    assert reviewer.calls == 2
    assert gateway.calls == [StructuredDraftAnswer, IssueResearchStep, ClaimRepairEdits]


@pytest.mark.parametrize(
    "fault",
    [
        "unresolved_missing",
        "recorded_gap_missing",
        "evidence:incorrect",
        "evidence:uncertain",
        "evidence:low_confidence",
        "evidence:need_identity",
        "evidence:section_identity",
        "issue:incorrect",
        "issue:uncertain",
        "issue:not_applicable",
        "issue:low_confidence",
        "issue:need_identity",
        "issue:section_identity",
        "gap_resolution",
        "provider",
        "missing_check",
        "duplicate_check",
        "original_integrity",
        "dependency_integrity",
    ],
)
def test_partial_publication_never_waives_other_quality_or_identity_failures(
    fault: str,
) -> None:
    engine, plan, _gateway, reviewer = _engine(fault=fault)
    result = engine._finalize_semantic("A ve B işlemlerini açıklayın.", "", None, plan)

    assert result.status == "unavailable"
    assert result.answer is None
    assert result.gaps
    assert reviewer.calls == 1
