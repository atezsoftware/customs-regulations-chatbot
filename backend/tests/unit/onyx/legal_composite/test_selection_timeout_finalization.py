"""Unclassified originals survive a closed selection phase and still require review."""

from typing import Literal, TypeVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_composite.budget import (
    ResearchPhaseClosed,
    ResearchStopReason,
    WorkflowBudget,
)
from onyx.legal_composite.draft_composition import DraftComposition
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AnswerSection,
    AuthorityDependency,
    IssueResearchPlan,
    ReviewCheck,
    SemanticReview,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from onyx.legal_composite.selection import (
    SelectionObservation,
    SourceSelectionRequest,
    SourceSelector,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for
from tests.unit.onyx.legal_composite.test_review_assessment import original
from tests.unit.onyx.legal_composite.test_source_requirements import (
    RULE_A,
    RULE_B,
    fixture,
)

T = TypeVar("T", bound=BaseModel)


class _Gateway:
    last_call_id: str | None = None

    def __init__(
        self,
        budget: WorkflowBudget,
        plan: IssueResearchPlan,
        draft: StructuredDraftAnswer,
        preserve_research_finalization_on_timeout: bool = False,
    ) -> None:
        self.budget = budget
        self.plan = plan
        self.draft = draft
        self.calls: list[tuple[type[BaseModel], bool]] = []
        self.writer_originals: list[JsonValue] = []
        self.last_delivered_citations: set[int] = set()
        self.preserve_research_finalization_on_timeout = (
            preserve_research_finalization_on_timeout
        )

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[T],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> T:
        del system, flow
        reservation = self.budget.request(100, 10, 1, 1, finalizing)
        self.budget.settle(reservation, 90, 9)
        self.last_call_id = reservation.call_id
        self.calls.append((response_type, finalizing))
        if response_type is IssueResearchPlan:
            return response_type.model_validate(self.plan.model_dump())
        assert response_type is DraftComposition and finalizing
        records = payload["original_evidence"]
        assert isinstance(records, list)
        self.writer_originals = records
        self.last_delivered_citations = {
            record["citation"]
            for record in records
            if isinstance(record, dict) and isinstance(record["citation"], int)
        }
        required = payload["required_evidence_numbers"]
        assert isinstance(required, list) and set(required) == {1, 2}
        return response_type.model_validate(composition_for(self.draft).model_dump())


class _TimeoutClassifier:
    def __init__(self, budget: WorkflowBudget, reason: ResearchStopReason) -> None:
        self.budget = budget
        self.reason = reason
        self.calls = 0

    def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
        self.calls += 1
        assert {row.citation for row in request.candidates} == {1, 2}
        self.budget.request(100, 10, 1, 1)
        self.budget.close_research(self.reason)
        raise ResearchPhaseClosed(self.reason)


class _FullReviewer:
    def __init__(
        self, ledger: EvidenceLedger, budget: WorkflowBudget, fault: str | None
    ) -> None:
        self.ledger = ledger
        self.budget = budget
        self.fault = fault
        self.calls = 0
        self.reviewed_check_ids: set[str] = set()
        self.reviewed_originals: dict[int, tuple[str, str | None]] = {}

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
        reservation = self.budget.request(100, 10, 1, 1, finalizing=True)
        self.budget.settle(reservation, 90, 9)
        self.calls += 1
        assert delivered == {1, 2} and affected_sections is None
        for citation in delivered:
            item = self.ledger.get(citation)
            assert item is not None
            self.reviewed_originals[citation] = (item.text, item.text_hash)
        questions = self.expected_checks(
            request, plan, draft, requirements, dependencies, delivered, previous
        )
        self.reviewed_check_ids = set(questions)
        checks = {
            identity: ReviewCheck(
                check_id=identity,
                need_ids=question.need_ids,
                section_ids=question.section_ids,
                status="addressed",
                confidence=0.99,
            )
            for identity, question in questions.items()
        }
        if self.fault == "incorrect":
            checks["claim:c_a"].status = "incorrect"
        elif self.fault == "missing_check":
            del checks["claim:c_a"]
        return SemanticReview(
            checks=list(checks.values()),
            failure="Review unavailable" if self.fault == "provider" else None,
        )


@pytest.mark.parametrize(
    "reason", ["host_research_deadline", "host_call_timeout", "provider_timeout"]
)
@pytest.mark.parametrize("fault", [None, "incorrect", "missing_check", "provider"])
@pytest.mark.parametrize("large_originals", [False, True])
def test_selection_timeout_keeps_originals_for_reserved_writer_and_full_review(
    reason: ResearchStopReason, fault: str | None, large_originals: bool
) -> None:
    ledger, plan, requirements, original_draft = fixture()
    if large_originals:
        ledger = EvidenceLedger()
        originals = [
            original(rule + "\nCanonical supporting detail." * 1_100, need)
            for rule, need in [(RULE_A, "a"), (RULE_B, "b")]
        ]
        for item in originals:
            assert item.search_doc is not None and item.chunk_id is not None
            item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
        assert sum(len(item.text) for item in originals) > 50_000
        ledger.add(originals, RunContext())
    before = {
        citation: (item.text, item.text_hash)
        for citation in ledger.citation_numbers()
        if (item := ledger.get(citation)) is not None
    }
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id=row.section_id,
                need_ids=row.need_ids,
                claim_ids=[
                    claim.claim_id
                    for claim in original_draft.claims
                    if claim.section_id == row.section_id
                ],
            )
            for row in original_draft.sections
        ],
        claims=[
            claim.model_copy(
                deep=True,
                update={
                    "answer_excerpt": next(
                        row.text
                        for row in original_draft.sections
                        if row.section_id == claim.section_id
                    )
                },
            )
            for claim in original_draft.claims
        ],
        requirements=requirements,
        unresolved_need_ids=[],
    )
    policy = WorkflowPolicy(max_reviews=1)
    budget = WorkflowBudget(policy, lambda: 0.0)
    budget.configure_finalization(100, 10, 0.00011)
    gateway = _Gateway(budget, plan, draft)
    classifier = _TimeoutClassifier(budget, reason)
    reviewer = _FullReviewer(ledger, budget, fault)
    acquirer = Mock()
    acquirer.acquire.return_value = []
    acquirer.definitions.return_value = []
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=policy,
        check_active=lambda: None,
        research_available=budget.research_available,
        selector=SourceSelector(classifier),
        reviewer=reviewer,
    )

    result = engine.run("A ve B işlemlerinin koşulları ve süreleri?")

    assert gateway.calls == [(IssueResearchPlan, False), (DraftComposition, True)]
    assert classifier.calls == 1 and reviewer.calls == 1
    acquirer.acquire.assert_called_once()
    assert engine.selection is None
    assert engine.receipts[-1] == {
        "status": "selection_stopped",
        "reason": reason,
        "citations": [1, 2],
        "navigation_only": True,
    }
    assert reviewer.reviewed_originals == before
    assert {
        row["citation"]: (row["text"], row["text_hash"])
        for row in gateway.writer_originals
        if isinstance(row, dict)
    } == before
    assert {
        "issue:a",
        "issue:b",
        "claim:c_a",
        "claim:c_b",
        "original:1",
        "original:2",
    } <= reviewer.reviewed_check_ids
    assert budget.snapshot()["research_stop_reason"] == reason
    assert budget.snapshot()["stop_reason"] is None
    assert budget.snapshot()["unsettled_calls"] == 1
    assert budget.snapshot()["pending_final_calls"] == 0
    if fault is None:
        assert result.status == "verified" and result.answer == draft.answer
    else:
        assert result.status == "unavailable" and result.answer is None


PreAdmissionFailure = Literal["deadline", "cost", "cancel", "other", "early_deadline"]


class _PreAdmissionClassifier:
    def __init__(self, budget: WorkflowBudget, failure: PreAdmissionFailure) -> None:
        self.budget = budget
        self.failure = failure
        self.calls = 0

    def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
        self.calls += 1
        assert {row.citation for row in request.candidates} == {1, 2}
        if self.failure == "cost":
            self.budget.stop(
                "Workflow model budget exhausted; finalization allocation retained"
            )
        elif self.failure == "other":
            raise RunStopped("Unrelated canonical operation stopped")
        elif self.failure == "early_deadline":
            raise RunStopped("Workflow deadline reached; finalization time retained")
        self.budget.check_active()
        raise AssertionError("Expired research must fail before provider admission")


@pytest.mark.parametrize(
    "failure,opt_in",
    [
        ("deadline", True),
        ("cost", True),
        ("cancel", True),
        ("other", True),
        ("deadline", False),
        ("early_deadline", True),
    ],
)
def test_pre_admission_deadline_preserves_only_authorized_finalization(
    failure: PreAdmissionFailure, opt_in: bool
) -> None:
    ledger, plan, requirements, original_draft = fixture()
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id=row.section_id,
                need_ids=row.need_ids,
                claim_ids=[
                    claim.claim_id
                    for claim in original_draft.claims
                    if claim.section_id == row.section_id
                ],
            )
            for row in original_draft.sections
        ],
        claims=[
            claim.model_copy(
                deep=True,
                update={
                    "answer_excerpt": next(
                        row.text
                        for row in original_draft.sections
                        if row.section_id == claim.section_id
                    )
                },
            )
            for claim in original_draft.claims
        ],
        requirements=requirements,
        unresolved_need_ids=[],
    )
    now = [0.0]
    policy = WorkflowPolicy(max_reviews=1)
    budget = WorkflowBudget(policy, lambda: now[0])
    budget.configure_finalization(100, 10, 0.00011)
    gateway = _Gateway(budget, plan, draft, opt_in)
    classifier = _PreAdmissionClassifier(budget, failure)
    reviewer = _FullReviewer(ledger, budget, None)
    owner_context = RunContext()
    acquirer = Mock()

    def acquired(*_args: object) -> list[dict[str, JsonValue]]:
        if failure != "early_deadline":
            now[0] = policy.timeout_seconds - policy.finalization_reserve_seconds
        if failure == "cancel":
            owner_context.cancel()
        return []

    acquirer.acquire.side_effect = acquired
    acquirer.definitions.return_value = []
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=policy,
        check_active=owner_context.check_active,
        research_available=budget.research_available,
        selector=SourceSelector(classifier),
        reviewer=reviewer,
    )

    result = engine.run("A ve B işlemlerinin koşulları ve süreleri?")

    assert classifier.calls == 1 and engine.selection is None
    acquirer.acquire.assert_called_once()
    if failure == "deadline" and opt_in:
        assert result.status == "verified" and result.answer == draft.answer
        assert gateway.calls == [(IssueResearchPlan, False), (DraftComposition, True)]
        assert reviewer.calls == 1 and set(reviewer.reviewed_originals) == {1, 2}
        assert {"issue:a", "issue:b", "original:1", "original:2"} <= (
            reviewer.reviewed_check_ids
        )
        assert engine.pending_selection_citations == {1, 2}
        assert budget.snapshot()["research_stop_reason"] == "host_research_deadline"
        assert budget.snapshot()["unsettled_calls"] == 0
        assert budget.snapshot()["pending_final_calls"] == 0
    else:
        assert result.answer is None
        assert result.status == ("cancelled" if failure == "cancel" else "unavailable")
        assert gateway.calls == [(IssueResearchPlan, False)] and reviewer.calls == 0
        assert not engine.pending_selection_citations
        assert "research_stop_reason" not in budget.snapshot()
        assert budget.snapshot()["pending_final_calls"] == 2
        assert not any(
            row.get("status") == "selection_stopped" for row in engine.receipts
        )
