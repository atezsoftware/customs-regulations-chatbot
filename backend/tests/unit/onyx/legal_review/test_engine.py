from collections.abc import Sequence
from typing import TypeVar
from unittest.mock import patch

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, SharedBudget
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_review.engine import (
    LegalReviewEngine,
    recorded_legal_status,
    recorded_validity,
    validate_support,
)
from onyx.legal_review.models import (
    AnswerClaim,
    DimensionAssessment,
    DraftAnswer,
    Issue,
    IssuePlan,
    LegalDimension,
    PassageSupport,
    ReadingDecision,
    Requirement,
    ReviewCheck,
    ReviewResult,
    SourceAction,
    WorkflowPolicy,
)
from onyx.tracing.flows import LLMFlow

T = TypeVar("T", bound=BaseModel)
RULE = "Başvuru, belgenin ibraz edilmesi şartıyla kabul edilir."


def original() -> EvidenceItem:
    return EvidenceItem(
        source_id="source-1",
        chunk_id="chunk-1",
        text=RULE,
        metadata={"validity_start": "2025-01-01", "read_as_of_date": "2026-10-09"},
        search_doc=SearchDoc(
            document_id="source-1",
            chunk_ind=1,
            semantic_identifier="Özgün kaynak",
            source_type=DocumentSource.FILE,
            blurb=RULE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": "chunk-1"},
            match_highlights=[],
        ),
    )


def plan() -> IssuePlan:
    return IssuePlan(
        language="tr",
        issues=[
            Issue(
                issue_id="i1",
                question="Başvuru şartı nedir?",
                requested_outcome="Başvuru şartını açıklamak",
                research_queries=["Başvuru kabul belgenin ibrazı"],
            )
        ],
    )


def requirement(identity: str = "r1") -> Requirement:
    return Requirement(
        requirement_id=identity,
        issue_id="i1",
        dimension=LegalDimension.LEGAL_BASIS,
        rule=RULE,
        application="Belgenin ibrazı koşuldur.",
        supports=[PassageSupport(citation=1, quotation=RULE)],
    )


def reading(*, actions: bool = False) -> ReadingDecision:
    return ReadingDecision(
        requirements=[requirement()],
        dimensions=[
            DimensionAssessment(
                issue_id="i1",
                dimension=dimension,
                status="addressed"
                if dimension is LegalDimension.LEGAL_BASIS
                else "not_applicable",
                reason="Kullanıcı bu başvuru şartını sordu; diğer boyutların bu olaydaki etkisi yoktur.",
                requirement_ids=["r1"]
                if dimension is LegalDimension.LEGAL_BASIS
                else [],
            )
            for dimension in LegalDimension
        ],
        actions=[
            SourceAction(
                issue_ids=["i1"],
                tool="read_provision",
                arguments={"source_id": "source-1", "article": "2"},
            )
        ]
        if actions
        else [],
    )


def draft() -> DraftAnswer:
    text = RULE + " [1]"
    return DraftAnswer(
        answer=text,
        claims=[
            AnswerClaim(
                claim_id="c1",
                issue_ids=["i1"],
                answer_excerpt=text,
                supports=[PassageSupport(citation=1, quotation=RULE)],
            )
        ],
    )


class FakeGateway:
    def __init__(self, results: list[BaseModel]) -> None:
        self.results = list(results)
        self.calls: list[tuple[LLMFlow, dict[str, JsonValue]]] = []

    def complete(
        self,
        prompt: str,
        state: dict[str, JsonValue],
        response_model: type[T],
        flow: LLMFlow,
        *,
        finalizing: bool = False,
    ) -> T:
        del prompt, finalizing
        self.calls.append((flow, state))
        return response_model.model_validate(self.results.pop(0).model_dump())


class FakeAcquirer:
    def __init__(self, ledger: EvidenceLedger, context: RunContext) -> None:
        self.ledger = ledger
        self.context = context
        self.receipts: list[dict[str, JsonValue]] = []
        self.actions: list[SourceAction] = []
        self.searches = 0
        self.original = original()

    def definitions(self) -> list[dict[str, JsonValue]]:
        return []

    def acquire(
        self, actions: list[SourceAction], plan: IssuePlan, *, finalizing: bool = False
    ) -> None:
        del plan, finalizing
        self.searches += sum(action.tool == "search_corpus" for action in actions)
        self.actions.extend(actions)
        self.ledger.add([self.original], self.context)


class FakeReviewer:
    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = list(outcomes)
        self.states: list[dict[str, JsonValue]] = []

    def review(
        self,
        state: dict[str, JsonValue],
        checks: Sequence[ReviewCheck],
        timeout_seconds: float,
    ) -> ReviewResult:
        del timeout_seconds
        self.states.append(state)
        outcome = self.outcomes.pop(0)
        if outcome == "incomplete":
            return ReviewResult(
                completed=False, failure_reason="jev_review_timeout", input_tokens=20
            )
        flags = [checks[-1]] if outcome == "flag" else []
        return ReviewResult(
            completed=True,
            scores={check.id: 0.9 if check in flags else 0.1 for check in checks},
            flags=flags,
            input_tokens=20,
        )


def engine(
    outputs: list[BaseModel], reviews: list[str]
) -> tuple[LegalReviewEngine, FakeGateway, FakeReviewer]:
    context = RunContext(budget=SharedBudget(max_decisions=32))
    ledger = EvidenceLedger()
    gateway = FakeGateway(outputs)
    reviewer = FakeReviewer(reviews)
    return (
        LegalReviewEngine(
            gateway=gateway,
            acquirer=FakeAcquirer(ledger, context),
            reviewer=reviewer,
            ledger=ledger,
            context=context,
            policy=WorkflowPolicy(),
        ),
        gateway,
        reviewer,
    )


def test_main_path_has_one_early_and_one_draft_review_and_honest_validity() -> None:
    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "partial"
    assert result.answer is not None and "Yürürlük sınırı" in result.answer
    assert result.requirements[0].legal_status == "unknown"
    assert len(result.dimensions) == 12
    assert len(reviewer.states) == 2
    assert [flow for flow, _ in gateway.calls] == [
        LLMFlow.LEGAL_REVIEW_PLANNER,
        LLMFlow.LEGAL_REVIEW_READING,
        LLMFlow.LEGAL_REVIEW_DRAFT,
    ]
    assert reviewer.states[0]["draft"] is None
    assert isinstance(reviewer.states[1]["draft"], dict)
    assert "Yürürlük sınırı" in str(reviewer.states[1]["draft"])
    assert "tools" not in reviewer.states[1]


def test_missing_review_never_publishes_an_unchecked_draft() -> None:
    workflow, gateway, reviewer = engine([plan(), reading()], ["incomplete"])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable" and result.answer is None
    assert "jev_review_timeout" in result.gaps
    assert len(gateway.calls) == 2 and len(reviewer.states) == 1


def test_one_repair_uses_literal_draft_then_stops_on_remaining_flags() -> None:
    workflow, gateway, reviewer = engine(
        [plan(), reading(), draft(), reading(), draft()], ["pass", "flag", "flag"]
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable" and result.answer is None
    assert result.repair_used is True
    assert len(reviewer.states) == 3
    repair_read = gateway.calls[3][1]
    assert isinstance(repair_read["draft"], dict)
    assert repair_read["draft"]["answer"] is not None
    assert len(gateway.calls) == 5


def test_nonliteral_support_is_rejected_before_jev() -> None:
    bad = reading()
    bad.requirements[0].supports[0].quotation = "Bu şart aranmamaktadır."
    workflow, _, reviewer = engine([plan(), bad], [])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable" and result.answer is None
    assert reviewer.states == []


def test_source_text_is_not_truncated_to_fit_review() -> None:
    workflow, _, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    workflow.run("Başvuru şartı nedir?", "")
    originals = reviewer.states[0]["original_evidence"]
    assert isinstance(originals, list) and isinstance(originals[0], dict)
    assert originals[0]["text"] == RULE


def test_dimension_inventory_must_cover_all_twelve() -> None:
    bad = reading()
    bad.dimensions.pop()
    workflow, _, _ = engine([plan(), bad], [])
    assert workflow.run("Soru", "").status == "unavailable"


def test_dimension_cannot_close_using_another_dimensions_requirement() -> None:
    bad = reading()
    bad.dimensions[1].requirement_ids = ["r1"]
    bad.dimensions[1].status = "addressed"
    workflow, _, _ = engine([plan(), bad], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable"
    assert "match the assessed dimension" in result.gaps[-1]


def test_requirement_supersession_preserves_audit_and_removes_obsolete_rule() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(reading())
    replacement = reading()
    replacement.requirements[0].requirement_id = "r2"
    replacement.requirements[0].supersedes_requirement_ids = ["r1"]
    replacement.requirements[0].application = "Yeni yorum aynı belge şartını korur."
    replacement.dimensions[0].requirement_ids = ["r2"]
    workflow._accept_reading(replacement)
    assert set(workflow.requirements) == {"r2"}
    assert set(workflow.requirement_history) == {"r1"}
    assert "requirement_history" not in workflow.state("Soru", "")


def test_cancellation_terminates_without_review_or_publication() -> None:
    workflow, _, reviewer = engine([plan()], [])
    workflow.context.cancel()
    # A real gateway checks cancellation before every provider call.
    with patch.object(
        workflow.context, "check_active", side_effect=TimeoutError("cancelled")
    ):
        result = workflow.run("Soru", "")
    assert result.status == "cancelled" and result.answer is None
    assert reviewer.states == []


def test_lifecycle_active_does_not_prove_legal_in_force() -> None:
    ledger = EvidenceLedger()
    item = original()
    item.metadata["status"] = "active"
    item.metadata["legal_status"] = "in_force"
    item.metadata["legal_status_verified"] = True
    ledger.add([item], RunContext())
    assert recorded_legal_status(requirement(), ledger) == "unknown"


def test_read_date_does_not_silently_become_the_event_date() -> None:
    ledger = EvidenceLedger()
    ledger.add([original()], RunContext())
    assert recorded_validity(requirement(), ledger) == "unknown"
    assert recorded_validity(requirement(), ledger, "2024-10-09") == "unknown"
    assert (
        recorded_validity(requirement(), ledger, "2026-10-09")
        == "within_recorded_window"
    )


def source_issue(
    identity: str = "s1",
    *,
    query: bool = True,
    dimension: LegalDimension = LegalDimension.EXCEPTIONS,
) -> Issue:
    return Issue(
        issue_id=identity,
        question="Bu belge şartının olaya uygulanmasını değiştiren bir istisna var mı?",
        requested_outcome="Belge şartının maddi kapsamını çözmek",
        origin="source",
        parent_issue_id="i1",
        trigger_dimension=dimension,
        supporting_citations=[1],
        supporting_requirement_ids=["r1"],
        material_reason="Bir istisna başvuru şartının sonucunu değiştirebilir.",
        closure_criteria=[
            "Özgün metne göre şartın kapsamını ve maddi istisnayı belirle."
        ],
        research_queries=["Belge ibrazı başvuru şartı istisna"] if query else [],
    )


def reading_child(
    *,
    additional: bool,
    resolved: bool,
    query: bool = True,
    dimension: LegalDimension = LegalDimension.EXCEPTIONS,
) -> ReadingDecision:
    decision = reading()
    child = source_issue(query=query, dimension=dimension)
    decision.additional_issues = [child] if additional else []
    if resolved:
        child_requirement = requirement("r2")
        child_requirement.issue_id = child.issue_id
        decision.requirements.append(child_requirement)
    decision.dimensions.extend(
        DimensionAssessment(
            issue_id=child.issue_id,
            dimension=dimension,
            status="addressed"
            if resolved and dimension is LegalDimension.LEGAL_BASIS
            else "unresolved"
            if not resolved and dimension is LegalDimension.EXCEPTIONS
            else "not_applicable",
            reason="Kaynak istisnasının maddi kapsamı henüz incelenmedi."
            if not resolved
            else "Özgün metindeki belge ibrazı şartı bu dar kapsamlı meselede belirleyicidir.",
            requirement_ids=["r2"]
            if resolved and dimension is LegalDimension.LEGAL_BASIS
            else [],
        )
        for dimension in LegalDimension
    )
    return decision


def test_discovered_norm_does_not_automatically_force_search_or_child_issue() -> None:
    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    assert isinstance(workflow.acquirer, FakeAcquirer)
    workflow.acquirer.original.metadata.update(
        {
            "regulation_name": "Gümrük Kanunu",
            "regulation_number": "4458",
            "article_no": "241",
        }
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "partial"
    assert len(workflow.acquirer.actions) == 1
    assert result.plan is not None and len(result.plan.issues) == 1
    assert len(gateway.calls) == 3 and len(reviewer.states) == 2


@pytest.mark.parametrize(
    "dimension",
    [LegalDimension.EXCEPTIONS, LegalDimension.TAX, LegalDimension.PROCEDURE],
)
def test_model_opens_material_source_dependency_and_researches_it_once(
    dimension: LegalDimension,
) -> None:
    workflow, gateway, reviewer = engine(
        [
            plan(),
            reading_child(additional=True, resolved=False, dimension=dimension),
            reading_child(additional=False, resolved=True, dimension=dimension),
            draft(),
        ],
        ["pass", "pass"],
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert isinstance(workflow.acquirer, FakeAcquirer)
    assert len(workflow.acquirer.actions) == 2
    assert workflow.acquirer.actions[-1].issue_ids == ["s1"]
    assert result.plan is not None and len(result.plan.issues) == 2
    assert len(result.dimensions) == 24
    assert [row.status for row in result.issue_closures] == ["partial", "partial"]
    assert result.issue_closures[0].blocking_child_ids == ["s1"]
    assert len(gateway.calls) == 4 and len(reviewer.states) == 2
    assert result.status == "partial"


def test_parent_stays_open_and_disclosed_while_material_child_is_open() -> None:
    workflow, _, reviewer = engine(
        [plan(), reading_child(additional=True, resolved=False, query=False), draft()],
        ["pass", "pass"],
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert [row.status for row in result.issue_closures] == ["open", "open"]
    assert result.answer is not None and "İnceleme sınırı" in result.answer
    assert "İnceleme sınırı" in str(reviewer.states[-1]["draft"])
    assert result.status == "partial"


def test_source_issue_cannot_bind_unobserved_parent_requirement() -> None:
    bad = reading_child(additional=True, resolved=False, query=False)
    bad.additional_issues[0].supporting_requirement_ids = ["invented"]
    workflow, _, reviewer = engine([plan(), bad], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and reviewer.states == []
    assert "parent-backed requirement" in result.gaps[-1]


def test_duplicate_source_dependency_reuses_existing_issue() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    first = reading_child(additional=True, resolved=False, query=False)
    workflow._accept_reading(first)
    duplicate = reading_child(additional=False, resolved=False, query=False)
    duplicate.additional_issues = [source_issue("s2", query=False)]
    duplicate.dimensions.extend(
        row.model_copy(update={"issue_id": "s2"})
        for row in list(duplicate.dimensions)
        if row.issue_id == "s1"
    )
    with pytest.raises(ValueError, match="Duplicate parent/source/dimension"):
        workflow._accept_reading(duplicate)


def test_issue_count_has_no_arbitrary_cap_and_dependencies_remain_acyclic() -> None:
    issues = [
        plan().issues[0].model_copy(update={"issue_id": f"i{index}"})
        for index in range(31)
    ]
    assert len(IssuePlan(language="tr", issues=issues).issues) == 31
    assert (
        Issue.model_validate(
            {**issues[0].model_dump(), "research_queries": []}
        ).research_queries
        == []
    )
    child = source_issue(query=False)
    child.parent_issue_id = "s1"
    with pytest.raises(ValueError, match="acyclic"):
        IssuePlan(language="tr", issues=[plan().issues[0], child])


def test_distinct_initial_queries_respect_global_search_budget_before_dispatch() -> (
    None
):
    issues = [
        plan()
        .issues[0]
        .model_copy(
            update={
                "issue_id": f"i{index}",
                "research_queries": [f"Focused query {index}"],
            }
        )
        for index in range(25)
    ]
    workflow, _, _ = engine([IssuePlan(language="tr", issues=issues)], [])
    result = workflow.run("Soru", "")
    assert (
        isinstance(workflow.acquirer, FakeAcquirer) and workflow.acquirer.actions == []
    )
    assert result.status == "unavailable" and "total search budget" in result.gaps[-1]


def test_related_issues_share_initial_search_without_losing_requested_outcomes() -> (
    None
):
    issues = [
        plan()
        .issues[0]
        .model_copy(
            update={
                "issue_id": f"i{index}",
                "requested_outcome": f"Explicit requested outcome {index}",
            }
        )
        for index in range(31)
    ]
    issues[-1].research_queries = []
    grouped = IssuePlan(language="tr", issues=issues)
    extracted = reading()
    extracted.requirements[0].issue_id = "i0"
    extracted.dimensions = [
        row.model_copy(
            update={
                "issue_id": issue.issue_id,
                "status": row.status if issue.issue_id == "i0" else "not_applicable",
                "requirement_ids": row.requirement_ids
                if issue.issue_id == "i0"
                else [],
            }
        )
        for issue in issues
        for row in reading().dimensions
    ]
    written = draft()
    written.claims[0].issue_ids = ["i0"]
    workflow, _, _ = engine([grouped, extracted, written], ["pass", "pass"])
    result = workflow.run("Soru", "")
    assert (
        isinstance(workflow.acquirer, FakeAcquirer)
        and len(workflow.acquirer.actions) == 1
    )
    assert len(workflow.acquirer.actions[0].issue_ids) == 30
    assert result.plan is not None and len(result.plan.issues) == 31
    assert result.plan.issues[-1].requested_outcome == "Explicit requested outcome 30"


def test_initial_planner_cannot_invent_source_issues_before_retrieval() -> None:
    premature = IssuePlan(
        language="tr", issues=[plan().issues[0], source_issue(query=False)]
    )
    workflow, _, _ = engine([premature], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and "initial plan" in result.gaps[-1]
    assert (
        isinstance(workflow.acquirer, FakeAcquirer) and workflow.acquirer.actions == []
    )


def test_child_canonical_trigger_survives_same_round_parent_supersession() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(
        reading_child(additional=True, resolved=False, query=False)
    )
    revised = reading_child(additional=False, resolved=False, query=False)
    revised.requirements[0].requirement_id = "r3"
    revised.requirements[0].supersedes_requirement_ids = ["r1"]
    revised.requirements[0].application = "Kaynağın kapsamı düzeltildi."
    revised.dimensions[0].requirement_ids = ["r3"]
    workflow._accept_reading(revised)
    assert "r1" in workflow.requirement_history and "r1" not in workflow.requirements
    assert workflow.plan.issues[1].supporting_requirement_ids == ["r1"]


@pytest.mark.parametrize("quotation", [" ", "benzer ama kaynakta olmayan ifade"])
def test_empty_or_fabricated_quotation_cannot_bind_original(quotation: str) -> None:
    ledger = EvidenceLedger()
    ledger.add([original()], RunContext())
    with pytest.raises(ValueError):
        validate_support(PassageSupport(citation=1, quotation=quotation), ledger)
