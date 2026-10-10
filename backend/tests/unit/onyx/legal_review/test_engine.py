from collections.abc import Sequence
from typing import TypeVar
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, SharedBudget
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_review.drafting import DraftEdits, EditorialEdits, GeneratedDraft
from onyx.legal_review.engine import (
    LegalReviewEngine,
    recorded_legal_status,
    recorded_validity,
)
from onyx.legal_review.models import (
    AnswerClaim,
    DimensionAssessment,
    DiscoveryQuery,
    DraftAnswer,
    EvidenceResolutionDecision,
    InitialDiscoveryPlan,
    InitialReadingDecision,
    Issue,
    IssuePlan,
    LegalDimension,
    PassageSupport,
    PlannedIssue,
    PublicationFinding,
    PublicationReview,
    ReadingDecision,
    RepairReadingDecision,
    RepairResolution,
    RequestedOutcome,
    Requirement,
    ResearchResolution,
    ReviewCheck,
    ReviewResult,
    SourceAction,
    WorkflowPolicy,
)
from onyx.prompts.legal_review.prompts import READING_PROMPT, REPAIR_READING_PROMPT
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
            PlannedIssue(
                material_reason="The document condition changes application eligibility.",
                closure_criteria=[
                    "Establish the document condition and its application."
                ],
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
        rule=RULE,
        supports=[PassageSupport(citation=1, span_number=1)],
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
                supports=[PassageSupport(citation=1, span_number=1)],
            )
        ],
    )


def repair_reading(check_id: str = "claim:c1") -> RepairReadingDecision:
    return RepairReadingDecision(
        dimensions=[],
        repair_resolutions=[
            RepairResolution(
                check_ids=[check_id],
                diagnosis="The literal conclusion needs the original document condition.",
                correction="Preserve that condition in the repaired answer.",
                disposition="correct",
                scope="draft",
                supports=[PassageSupport(citation=1, span_number=1)],
            )
        ],
    )


class FakeGateway:
    def __init__(self, results: list[BaseModel]) -> None:
        self.results = list(results)
        self.calls: list[tuple[LLMFlow, dict[str, JsonValue]]] = []
        self.response_models: list[type[BaseModel]] = []
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        state: dict[str, JsonValue],
        response_model: type[T],
        flow: LLMFlow,
        *,
        finalizing: bool = False,
    ) -> T:
        del finalizing
        self.prompts.append(prompt)
        self.calls.append((flow, state))
        self.response_models.append(response_model)
        result = self.results.pop(0)
        payload = result.model_dump()
        if (
            issubclass(response_model, InitialDiscoveryPlan)
            and type(result) is IssuePlan
        ):
            payload["requested_outcomes"] = [
                {"request": issue.requested_outcome, "issue_ids": [issue.issue_id]}
                for issue in result.issues
            ]
            shared: dict[str, list[str]] = {}
            for issue in result.issues:
                for query in issue.research_queries:
                    shared.setdefault(query, []).append(issue.issue_id)
            payload["discovery_queries"] = [
                {"query": query, "issue_ids": identities}
                for query, identities in shared.items()
            ]
        if (
            issubclass(response_model, InitialDiscoveryPlan)
            and "request_coverage" in response_model.model_fields
        ):
            units = state["explicit_request_units"]
            assert isinstance(units, dict)
            payload["request_coverage"] = {
                slot: {
                    "requested_result": str(text),
                    "issue_ids": [issue["issue_id"] for issue in payload["issues"]],
                }
                for slot, text in units.items()
            }
        if response_model in {GeneratedDraft, DraftEdits} and isinstance(
            result, DraftAnswer
        ):
            payload = {
                "blocks": [
                    {
                        "block_id": "b1",
                        "text": result.answer,
                        "claims": [
                            {
                                **claim.model_dump(exclude={"answer_excerpt"}),
                                "application": {
                                    "source_conditions": "The original requires the document.",
                                    "fact_application": "Explain the stated application condition.",
                                    "remaining_uncertainty": "The event date is not supplied.",
                                },
                            }
                            for claim in result.claims
                        ],
                    }
                ],
                "unresolved_issue_ids": result.unresolved_issue_ids,
            }
            if response_model is DraftEdits:
                payload["replacements"] = payload.pop("blocks")
        return response_model.model_validate(payload)


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
        if outcome == "refusal":
            return ReviewResult(
                completed=False,
                failure_reason="openai_decision_refusal",
                scores={check.id: 0.1 for check in checks[:-1]},
                unassessed_checks=[checks[-1]],
            )
        flags = [checks[-1]] if outcome == "flag" else []
        if outcome.startswith("flag:"):
            flags = [
                check for check in checks if check.id == outcome.removeprefix("flag:")
            ]
            assert len(flags) == 1
        return ReviewResult(
            completed=True,
            scores={check.id: 0.9 if check in flags else 0.1 for check in checks},
            flags=flags,
            input_tokens=20,
        )


def engine(
    outputs: list[BaseModel], reviews: list[str]
) -> tuple[LegalReviewEngine, FakeGateway, FakeReviewer]:
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(max_decisions=32)
    )
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
            policy=WorkflowPolicy(
                always_review_research=True, assume_current_corpus=False
            ),
        ),
        gateway,
        reviewer,
    )


def test_main_path_has_one_early_and_one_draft_review_and_honest_validity() -> None:
    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "partial"
    assert result.answer == draft().answer
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
    assert reviewer.states[1]["draft"]["answer"] == draft().answer
    assert reviewer.states[1]["draft"]["unresolved_issue_ids"] == ["i1"]
    assert "tools" not in reviewer.states[1]
    assert "review_diagnoses" not in reviewer.states[1]
    assert "research_resolutions" not in reviewer.states[1]


def test_missing_review_never_publishes_an_unchecked_draft() -> None:
    workflow, gateway, reviewer = engine([plan(), reading()], ["incomplete"])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable" and result.answer is None
    assert "jev_review_timeout" in result.gaps
    assert len(gateway.calls) == 2 and len(reviewer.states) == 1


def test_research_continues_through_multiple_distinct_source_returns() -> None:
    first = reading(actions=True)
    second = reading(actions=True)
    second.actions[0].arguments["article"] = "3"
    workflow, gateway, _ = engine(
        [plan(), first, second, reading(), draft()], ["pass", "pass"]
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.answer is not None
    assert (
        len([flow for flow, _ in gateway.calls if flow is LLMFlow.LEGAL_REVIEW_READING])
        == 3
    )
    assert isinstance(workflow.acquirer, FakeAcquirer)
    assert [
        action.arguments["article"]
        for action in workflow.acquirer.actions
        if action.tool == "read_provision"
    ] == ["2", "3"]


def test_one_repair_uses_literal_draft_then_stops_on_remaining_flags() -> None:
    workflow, gateway, reviewer = engine(
        [plan(), reading(), draft(), repair_reading(), draft()],
        ["pass", "flag", "flag"],
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "partial" and result.answer
    assert result.publication_mode == "review_incomplete"
    assert result.repair_used is True
    assert len(reviewer.states) == 3
    repair_read = gateway.calls[3][1]
    assert isinstance(repair_read["draft"], dict)
    assert repair_read["draft"]["answer"] is not None
    assert len(gateway.calls) == 5


def test_unknown_passage_support_is_rejected_before_jev() -> None:
    bad = reading()
    bad.requirements[0].supports = [PassageSupport(citation=1, span_number=999)]
    workflow, _, reviewer = engine([plan(), bad, bad], [])
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable" and result.answer is None
    assert reviewer.states == []


def test_source_text_is_not_truncated_to_fit_review() -> None:
    workflow, _, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    workflow.run("Başvuru şartı nedir?", "")
    originals = reviewer.states[0]["original_evidence"]
    assert isinstance(originals, list) and isinstance(originals[0], dict)
    passages = originals[0]["passages"]
    assert isinstance(passages, list)
    assert (
        "".join(str(row["text"]) for row in passages if isinstance(row, dict)) == RULE
    )
    assert "text" not in originals[0]


def test_final_review_does_not_treat_private_interpretations_as_answer_claims() -> None:
    extracted = reading()
    private_interpretation = "Özel araştırma yorumu: aksi yönde karar bulunmamaktadır."
    extracted.dimensions[2].reason = private_interpretation
    workflow, _, reviewer = engine([plan(), extracted, draft()], ["pass", "pass"])
    result = workflow.run("Başvuru şartı nedir?", "")
    early, final = reviewer.states
    assert early["review_target"] == "research"
    assert private_interpretation in str(early["dimension_assessments"])
    assert final["review_target"] == "literal_answer"
    assert "dimension_assessments" not in final and "requirements" not in final
    assert private_interpretation not in str(final)
    assert final["original_evidence"] == early["original_evidence"]
    assert final["source_registry"] == early["source_registry"]
    assert final["finding_sources"] == [
        {
            "requirement_id": "r1",
            "issue_ids": ["i1"],
            "supports": [PassageSupport(citation=1, span_number=1).model_dump()],
        }
    ]
    assert isinstance(final["draft"], dict)
    assert final["draft"]["answer"] == result.answer
    assert result.dimensions[2].reason == private_interpretation


def test_initial_reader_cannot_skip_any_standard_dimension() -> None:
    sparse = reading()
    sparse.dimensions.pop()
    workflow, _, reviewer = engine([plan(), sparse], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and result.answer is None
    assert result.dimensions == [] and result.requirements == []
    assert reviewer.states == []


def test_finding_can_support_multiple_dimensions_without_duplicate_extraction() -> None:
    extracted = reading()
    extracted.dimensions[1].requirement_ids = ["r1"]
    extracted.dimensions[1].status = "addressed"
    extracted.dimensions[
        1
    ].reason = "Aynı hükmün uygulama zamanı ayrıca değerlendirilir."
    workflow, _, reviewer = engine([plan(), extracted, draft()], ["pass", "pass"])
    result = workflow.run("Soru", "")
    assert result.status == "partial"
    assert len(result.requirements) == 1
    assert len(reviewer.states) == 2
    assert result.dimensions[1].requirement_ids == ["r1"]


def test_known_finding_id_does_not_bypass_semantic_review() -> None:
    extracted = reading()
    extracted.dimensions[1].requirement_ids = ["r1"]
    extracted.dimensions[1].status = "addressed"
    workflow, _, reviewer = engine([plan(), extracted], ["incomplete"])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and result.answer is None
    assert len(reviewer.states) == 1
    checks = workflow._checks(draft=False)
    assert all(
        "ID link alone proves no entailment" in check.instructions
        for check in checks[:12]
    )


def test_requirement_supersession_preserves_audit_and_removes_obsolete_rule() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(reading())
    replacement = reading()
    replacement.requirements[0].requirement_id = "r2"
    replacement.requirements[0].supersedes_requirement_ids = ["r1"]
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


def test_parent_stays_open_without_rewriting_prose_while_material_child_is_open() -> (
    None
):
    workflow, _, reviewer = engine(
        [plan(), reading_child(additional=True, resolved=False, query=False), draft()],
        ["pass", "pass"],
    )
    result = workflow.run("Başvuru şartı nedir?", "")
    assert [row.status for row in result.issue_closures] == ["open", "open"]
    assert result.answer == draft().answer
    reviewed = reviewer.states[-1]["draft"]
    assert isinstance(reviewed, dict)
    assert reviewed["answer"] == draft().answer
    assert reviewed["unresolved_issue_ids"] == ["i1", "s1"]
    assert result.status == "partial"


def test_source_issue_cannot_bind_unobserved_parent_requirement() -> None:
    bad = reading_child(additional=True, resolved=False, query=False)
    bad.additional_issues[0].supporting_requirement_ids = ["invented"]
    workflow, _, reviewer = engine([plan(), bad, bad], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and reviewer.states == []
    assert "parent-backed requirement" in result.gaps[-1]


def test_distinct_material_dependencies_can_share_parent_source_and_dimension() -> None:
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
    duplicate.additional_issues[0].question = "Aynı şartın süreye etkisi nedir?"
    workflow._accept_reading(duplicate)
    assert workflow.plan is not None and len(workflow.plan.issues) == 3
    assert set(workflow.source_issue_triggers) == {"s1", "s2"}


@pytest.mark.parametrize("tool", ["search_corpus", "read_provision"])
def test_distinct_dependencies_can_share_one_source_operation(tool: str) -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    chosen = reading_child(
        additional=True, resolved=False, query=tool == "search_corpus"
    )
    second = source_issue(
        "s2", query=tool == "search_corpus", dimension=LegalDimension.PROCEDURE
    )
    chosen.additional_issues.append(second)
    chosen.dimensions.extend(
        row.model_copy(update={"issue_id": "s2"})
        for row in list(chosen.dimensions)
        if row.issue_id == "s1"
    )
    if tool == "read_provision":
        chosen.actions = [
            SourceAction(
                issue_ids=[identity],
                tool=tool,
                arguments={"source_id": "source-1", "article": "2"},
            )
            for identity in ["s1", "s2"]
        ]
    workflow._accept_reading(chosen)
    assert len(chosen.actions) == 1
    assert chosen.actions[0].issue_ids == ["s1", "s2"]
    assert chosen.actions[0].tool == tool


def test_issue_count_has_no_arbitrary_cap_and_dependencies_remain_acyclic() -> None:
    issues: list[Issue] = [
        PlannedIssue.model_validate(
            plan().issues[0].model_copy(update={"issue_id": f"i{index}"}).model_dump()
        )
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


def test_distinct_initial_queries_have_no_global_search_quota() -> None:
    discovery = InitialDiscoveryPlan(
        requested_outcomes=[
            RequestedOutcome(request=f"Outcome {index}", issue_ids=[f"i{index}"])
            for index in range(25)
        ],
        language="tr",
        issues=[
            PlannedIssue.model_validate(
                plan()
                .issues[0]
                .model_copy(update={"issue_id": f"i{index}"})
                .model_dump()
            )
            for index in range(25)
        ],
        discovery_queries=[
            DiscoveryQuery(query=f"Focused query {index}", issue_ids=[f"i{index}"])
            for index in range(25)
        ],
    )
    workflow, _, _ = engine([], [])
    workflow.plan = discovery
    workflow._acquire(
        [
            SourceAction(
                tool="search_corpus",
                arguments={"query": query.query},
                issue_ids=query.issue_ids,
            )
            for query in discovery.discovery_queries
        ]
    )
    assert isinstance(workflow.acquirer, FakeAcquirer)
    assert workflow.acquirer.searches == 25
    assert len(workflow.research.needs) == 25


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
    for issue in issues:
        issue.research_queries = []
    grouped = InitialDiscoveryPlan(
        requested_outcomes=[
            RequestedOutcome(
                request=issue.requested_outcome, issue_ids=[issue.issue_id]
            )
            for issue in issues
        ],
        language="tr",
        issues=[PlannedIssue.model_validate(issue.model_dump()) for issue in issues],
        discovery_queries=[
            DiscoveryQuery(
                query="Başvuru kabul belgenin ibrazı",
                issue_ids=[issue.issue_id for issue in issues],
            )
        ],
    )
    extracted = reading()
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
    assert len(workflow.acquirer.actions[0].issue_ids) == 31
    assert result.plan is not None and len(result.plan.issues) == 31
    assert result.plan.issues[-1].requested_outcome == "Explicit requested outcome 30"


def test_initial_planner_cannot_invent_source_issues_before_retrieval() -> None:
    premature = IssuePlan(
        language="tr", issues=[plan().issues[0], source_issue(query=False)]
    )
    workflow, _, _ = engine([premature], [])
    result = workflow.run("Soru", "")
    assert result.status == "unavailable" and "origin" in result.gaps[-1]
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
    revised.dimensions[0].requirement_ids = ["r3"]
    workflow._accept_reading(revised)
    assert "r1" in workflow.requirement_history and "r1" not in workflow.requirements
    assert workflow.plan.issues[1].supporting_requirement_ids == ["r1"]


@pytest.mark.parametrize("span_number", [0, -1, "1", True])
def test_invalid_passage_selector_cannot_bind_original(span_number: object) -> None:
    with pytest.raises(ValidationError):
        PassageSupport.model_validate({"citation": 1, "span_number": span_number})


def test_model_cannot_supply_or_paraphrase_the_source_quotation() -> None:
    with pytest.raises(ValidationError):
        PassageSupport.model_validate(
            {"citation": 1, "span_number": 1, "quotation": "fabricated"}
        )


def unresolved_reading(issue_ids: tuple[str, ...] = ("i1",)) -> ReadingDecision:
    return ReadingDecision(
        dimensions=[
            DimensionAssessment(
                issue_id=identity,
                dimension=dimension,
                status="unresolved",
                reason="Bu boyutun maddi etkisi eldeki kaynaklarla henüz çözülemedi.",
                requirement_ids=[],
            )
            for identity in issue_ids
            for dimension in LegalDimension
        ]
    )


def complete_assessments(
    decision: ReadingDecision, issue_ids: tuple[str, ...]
) -> ReadingDecision:
    existing = {(row.issue_id, row.dimension) for row in decision.dimensions}
    decision.dimensions.extend(
        row
        for row in unresolved_reading(issue_ids).dimensions
        if (row.issue_id, row.dimension) not in existing
    )
    return decision


def prepared_engine(*, shared: bool = False) -> LegalReviewEngine:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    if shared:
        workflow.plan.issues.append(
            workflow.plan.issues[0].model_copy(update={"issue_id": "i2"}, deep=True)
        )
    workflow.ledger.add([original()], workflow.context)
    return workflow


def assessment(
    issue_id: str,
    dimension: LegalDimension,
    identities: list[str],
) -> DimensionAssessment:
    return DimensionAssessment(
        issue_id=issue_id,
        dimension=dimension,
        status="addressed",
        reason="Özgün bulgunun bu meseledeki maddi etkisi ayrıca değerlendirilir.",
        requirement_ids=identities,
    )


def test_global_findings_can_be_shared_across_issues_and_dimensions() -> None:
    workflow = prepared_engine(shared=True)
    extracted = reading()
    extracted.dimensions.extend(
        [
            assessment("i2", LegalDimension.PENALTIES, ["r1"]),
            assessment("i2", LegalDimension.PROCEDURE, ["r1"]),
        ]
    )
    workflow._accept_reading(complete_assessments(extracted, ("i1", "i2")))
    assert len(workflow.requirements) == 1
    assert len(workflow.dimensions) == 24
    assert workflow._issue_requirements("i1") == workflow._issue_requirements("i2")
    assert workflow._uncertain_issue_ids() == {"i1", "i2"}
    assert ("i2", "r1") in workflow.requirement_associations


def test_sparse_updates_preserve_previous_assessments_and_complete_new_child() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    before = workflow.dimensions[0].model_copy(deep=True)
    workflow._accept_reading(
        ReadingDecision(
            dimensions=unresolved_reading(("s1",)).dimensions,
            additional_issues=[source_issue(query=False)],
        )
    )
    assert workflow.dimensions[0] == before
    child_rows = [row for row in workflow.dimensions if row.issue_id == "s1"]
    assert len(child_rows) == 12 and all(
        row.status == "unresolved" for row in child_rows
    )
    assert [row.status for row in workflow.issue_closures()] == ["open", "open"]


@pytest.mark.parametrize("reversed_order", [False, True])
def test_batch_supersession_is_order_independent_and_replay_is_idempotent(
    reversed_order: bool,
) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    records = [requirement(), replacement]
    if reversed_order:
        records.reverse()
    row = assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])
    workflow._accept_reading(ReadingDecision(requirements=records, dimensions=[row]))
    assert set(workflow.requirements) == {"r2"}
    assert set(workflow.requirement_history) == {"r1"}
    before = workflow.state("Soru", "")["dimension_assessments"]
    workflow._accept_reading(ReadingDecision(requirements=records, dimensions=[row]))
    assert set(workflow.requirements) == {"r2"}
    assert set(workflow.requirement_history) == {"r1"}
    assert workflow.state("Soru", "")["dimension_assessments"] == before


@pytest.mark.parametrize("reversed_order", [False, True])
def test_new_same_batch_supersession_chain_keeps_only_final_finding(
    reversed_order: bool,
) -> None:
    workflow = prepared_engine()
    first, second, third = requirement(), requirement("r2"), requirement("r3")
    second.supersedes_requirement_ids = ["r1"]
    third.supersedes_requirement_ids = ["r2"]
    records = [first, second, third]
    if reversed_order:
        records.reverse()
    workflow._accept_reading(
        ReadingDecision(
            requirements=records,
            dimensions=complete_assessments(
                ReadingDecision(
                    dimensions=[assessment("i1", LegalDimension.LEGAL_BASIS, ["r3"])]
                ),
                ("i1",),
            ).dimensions,
        )
    )
    assert set(workflow.requirements) == {"r3"}
    assert set(workflow.requirement_history) == {"r1", "r2"}


def stable_engine_state(workflow: LegalReviewEngine) -> dict[str, JsonValue]:
    state = workflow.state("Soru", "Önceki olgular")
    state.pop("limits")
    return {
        **state,
        "finding_history": {
            key: value.model_dump(mode="json")
            for key, value in workflow.requirement_history.items()
        },
    }


@pytest.mark.parametrize("failure", ["self", "cycle", "fork", "unknown", "row"])
def test_invalid_batch_is_atomic_for_findings_rows_and_child_triggers(
    failure: str,
) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(
        reading_child(additional=True, resolved=False, query=False)
    )
    before = stable_engine_state(workflow)
    second, third = requirement("r2"), requirement("r3")
    second.supersedes_requirement_ids = ["r1"]
    records = [second]
    rows = [assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])]
    if failure == "self":
        second.supersedes_requirement_ids = ["r2"]
    elif failure == "cycle":
        second.supersedes_requirement_ids = ["r3"]
        third.supersedes_requirement_ids = ["r2"]
        records.append(third)
    elif failure == "fork":
        third.supersedes_requirement_ids = ["r1"]
        records.append(third)
    elif failure == "unknown":
        second.supersedes_requirement_ids = ["invented"]
    else:
        rows.append(rows[0].model_copy(deep=True))
    with pytest.raises(ValueError):
        workflow._accept_reading(ReadingDecision(requirements=records, dimensions=rows))
    assert stable_engine_state(workflow) == before


def test_supersession_invalidates_unupdated_shared_application_without_rebinding() -> (
    None
):
    workflow = prepared_engine(shared=True)
    initial = reading()
    initial.dimensions.append(assessment("i2", LegalDimension.VALIDITY, ["r1"]))
    workflow._accept_reading(complete_assessments(initial, ("i1", "i2")))
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(
        ReadingDecision(
            requirements=[replacement],
            dimensions=[assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])],
        )
    )
    row = next(
        row
        for row in workflow.dimensions
        if row.issue_id == "i2" and row.dimension == LegalDimension.VALIDITY
    )
    assert row.status == "unresolved" and row.requirement_ids == []
    assert "superseded" in row.reason
    assert workflow._issue_requirements("i2") == []
    workflow._accept_reading(
        ReadingDecision(dimensions=[assessment("i2", LegalDimension.VALIDITY, ["r1"])])
    )
    assert (
        next(
            row
            for row in workflow.dimensions
            if row.issue_id == "i2" and row.dimension == LegalDimension.VALIDITY
        ).status
        == "unresolved"
    )


@pytest.mark.parametrize("failure", ["unknown_issue", "unknown_finding", "duplicate"])
def test_sparse_dimension_updates_still_reject_invalid_relations(failure: str) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    row = assessment("i1", LegalDimension.EXCEPTIONS, ["r1"])
    if failure == "unknown_issue":
        row.issue_id = "invented"
    elif failure == "unknown_finding":
        row.requirement_ids = ["invented"]
    rows = [row, row.model_copy(deep=True)] if failure == "duplicate" else [row]
    with pytest.raises(ValueError):
        workflow._accept_reading(ReadingDecision(dimensions=rows))


def test_source_trigger_parent_binding_can_use_another_dimension_and_is_immutable() -> (
    None
):
    workflow = prepared_engine()
    chosen = reading_child(additional=True, resolved=False, query=False)
    workflow._accept_reading(chosen)
    before = workflow.state("Soru", "")["source_issue_triggers"]
    assert isinstance(before, list) and isinstance(before[0], dict)
    findings = before[0]["findings"]
    assert isinstance(findings, list) and isinstance(findings[0], dict)
    assert isinstance(findings[0]["parent_assessments"], list)
    assert (
        findings[0]["parent_assessments"][0]["dimension"]
        == LegalDimension.LEGAL_BASIS.value
    )
    passages = findings[0]["canonical_passages"]
    assert isinstance(passages, list) and isinstance(passages[0], dict)
    assert passages[0]["source_id"] == "source-1"
    assert passages[0]["chunk_id"] == "chunk-1"
    assert passages[0]["text_hash"] == original().text_hash
    assert "quotation" not in passages[0]
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(
        ReadingDecision(
            requirements=[replacement],
            dimensions=[assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])],
        )
    )
    assert workflow.state("Soru", "")["source_issue_triggers"] == before
    assert "r1" in workflow.requirement_history


def test_source_child_can_use_accepted_parent_association_history() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(
        ReadingDecision(
            requirements=[replacement],
            dimensions=[assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])],
        )
    )
    workflow._accept_reading(
        ReadingDecision(
            dimensions=unresolved_reading(("s1",)).dimensions,
            additional_issues=[source_issue(query=False)],
        )
    )
    assert "s1" in workflow.source_issue_triggers
    assert workflow.source_issue_triggers["s1"][0][0].requirement_id == "r1"


def test_global_finding_associated_only_to_other_parent_cannot_trigger_child() -> None:
    workflow = prepared_engine(shared=True)
    workflow._accept_reading(
        complete_assessments(
            ReadingDecision(
                requirements=[requirement()],
                dimensions=[assessment("i2", LegalDimension.PROCEDURE, ["r1"])],
            ),
            ("i1", "i2"),
        )
    )
    with pytest.raises(ValueError, match="parent-backed requirement association"):
        workflow._accept_reading(
            ReadingDecision(
                dimensions=unresolved_reading(("s1",)).dimensions,
                additional_issues=[source_issue(query=False)],
            )
        )


def test_unbound_unknown_finding_does_not_taint_unrelated_issue_closure_or_disclosure() -> (
    None
):
    workflow = prepared_engine(shared=True)
    chosen = reading()
    chosen.requirements.append(requirement("unbound"))
    chosen.dimensions.extend(
        DimensionAssessment(
            issue_id="i2",
            dimension=dimension,
            status="not_applicable",
            reason="Bu dar mesele bakımından bu boyutun maddi etkisi yoktur.",
            requirement_ids=[],
        )
        for dimension in LegalDimension
    )
    workflow._accept_reading(chosen)
    assert workflow._uncertain_issue_ids() == {"i1"}
    assert [row.status for row in workflow.issue_closures()] == ["partial", "closed"]
    disclosed = workflow._bind_unresolved_issues(draft())
    assert disclosed.unresolved_issue_ids == ["i1"]
    assert disclosed.answer == draft().answer
    assert disclosed.claims == draft().claims


def test_honest_claimless_limitation_still_receives_entire_answer_review() -> None:
    limited = DraftAnswer(
        answer="Başvuru şartı eldeki araştırmayla kesinleştirilemedi; sonuç açık kaldı.",
        claims=[],
        unresolved_issue_ids=["i1"],
    )
    workflow, gateway, reviewer = engine(
        [
            plan(),
            unresolved_reading(),
            limited,
        ],
        ["pass", "pass"],
    )
    result = workflow.run("Soru", "")
    assert result.status == "partial" and result.answer is not None
    assert result.final_review is not None and result.final_review.completed
    assert len(gateway.calls) == 3 and len(reviewer.states) == 2
    assert "all_answer_claims" in result.final_review.scores
    assert reviewer.states[-1]["draft"] is not None


def test_unresolved_positive_assertion_is_explicitly_unverified_after_limit() -> None:
    unsupported = DraftAnswer(
        answer="Başvuru kesinlikle kabul edilir.",
        claims=[],
        unresolved_issue_ids=["i1"],
    )
    workflow, _, reviewer = engine(
        [
            plan(),
            unresolved_reading(),
            unsupported,
            repair_reading("all_answer_claims"),
            unsupported,
        ],
        ["pass", "flag:all_answer_claims", "flag:all_answer_claims"],
    )
    result = workflow.run("Soru", "")
    assert result.status == "partial" and result.answer
    assert result.publication_mode == "review_incomplete"
    assert result.repair_used and len(reviewer.states) == 3


def test_claimless_draft_needs_known_unresolved_issues_and_cannot_cite_unbound_source() -> (
    None
):
    workflow = prepared_engine()
    for limited in [
        DraftAnswer(answer="Araştırma tamamlanmadı.", claims=[]),
        DraftAnswer(
            answer="Araştırma tamamlanmadı.",
            claims=[],
            unresolved_issue_ids=["invented"],
        ),
        DraftAnswer(
            answer="Araştırma tamamlanmadı. [1]", claims=[], unresolved_issue_ids=["i1"]
        ),
    ]:
        with pytest.raises(ValueError):
            workflow._validate_draft(limited)


def test_archived_echo_cannot_resurrect_or_rewrite_obsolete_finding() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(ReadingDecision(dimensions=[], requirements=[replacement]))
    workflow._accept_reading(
        ReadingDecision(dimensions=[], requirements=[requirement()])
    )
    assert set(workflow.requirements) == {"r2"}
    assert set(workflow.requirement_history) == {"r1"}
    modified = requirement()
    modified.rule = "Eski kimlikle yeni ve çelişkili bir yorum."
    before = stable_engine_state(workflow)
    with pytest.raises(ValueError, match="immutable"):
        workflow._accept_reading(
            ReadingDecision(dimensions=[], requirements=[modified])
        )
    assert stable_engine_state(workflow) == before


def test_result_preserves_archived_source_trigger_without_private_runtime_dependency() -> (
    None
):
    workflow = prepared_engine()
    workflow._accept_reading(
        reading_child(additional=True, resolved=False, query=False)
    )
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(
        ReadingDecision(
            requirements=[replacement],
            dimensions=[assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])],
        )
    )
    result = workflow._result("partial", "Araştırma tamamlanmadı.")
    assert (
        result.source_issue_triggers
        == workflow.state("Soru", "")["source_issue_triggers"]
    )
    encoded = result.model_dump(mode="json")["source_issue_triggers"]
    assert encoded[0]["findings"][0]["finding"]["requirement_id"] == "r1"
    assert encoded[0]["findings"][0]["canonical_passages"][0]["chunk_id"] == "chunk-1"


def test_corrupted_current_original_fails_closed_and_preserves_frozen_child_audit() -> (
    None
):
    workflow = prepared_engine()
    workflow._accept_reading(
        reading_child(additional=True, resolved=False, query=False)
    )
    before = workflow.state("Soru", "")["source_issue_triggers"]
    item = workflow.ledger.get(1)
    assert item is not None
    item.text += " Yetkisiz kaynak değişikliği."
    with patch.object(workflow.ledger, "get", return_value=item):
        result = workflow.run("Soru", "")
    assert result.status == "unavailable" and result.answer is None
    assert result.source_issue_triggers == before
    assert "authorized original passage" in result.gaps[-1]


@pytest.mark.parametrize("schema", [ReadingDecision, InitialReadingDecision])
def test_every_reading_schema_requires_explicit_dimensions_array(
    schema: type[ReadingDecision],
) -> None:
    assert "dimensions" in schema.model_json_schema()["required"]
    assert "default" not in schema.model_json_schema()["properties"]["dimensions"]
    with pytest.raises(ValidationError):
        schema.model_validate({"requirements": [], "actions": []})
    if schema is InitialReadingDecision:
        with pytest.raises(ValidationError):
            schema.model_validate({"dimensions": []})
    else:
        assert schema.model_validate({"dimensions": []}).dimensions == []


def test_initial_full_unresolved_assessment_advances_to_explicit_sparse_update() -> (
    None
):
    workflow = prepared_engine()
    assert isinstance(workflow.gateway, FakeGateway)
    workflow.gateway.results = [unresolved_reading(), ReadingDecision(dimensions=[])]
    contract = workflow.state("Soru", "")["reading_contract"]
    assert isinstance(contract, dict)
    assert contract["mode"] == "initial_assessment"
    assert contract["full_matrix_required_issue_ids"] == ["i1"]
    workflow._reading("Soru", "")
    before = workflow.dimensions.copy()
    workflow._reading("Soru", "")
    assert workflow.gateway.response_models == [InitialReadingDecision, ReadingDecision]
    assert workflow.dimensions == before and workflow.requirements == {}
    assert all(row.status == "unresolved" for row in workflow.dimensions)
    contract = workflow.state("Soru", "")["reading_contract"]
    assert isinstance(contract, dict)
    assert contract["mode"] == "assessment_update"
    assert contract["full_matrix_required_issue_ids"] == []
    assert contract["sparse_update_issue_ids"] == ["i1"]


def test_rejected_initial_matrix_does_not_advance_reader_phase() -> None:
    workflow = prepared_engine()
    assert isinstance(workflow.gateway, FakeGateway)
    workflow.gateway.results = [ReadingDecision(dimensions=[]), unresolved_reading()]
    with pytest.raises(ValidationError):
        workflow._reading("Soru", "")
    assert workflow.dimensions == [] and workflow.requirements == {}
    workflow._reading("Soru", "")
    assert workflow.gateway.response_models == [
        InitialReadingDecision,
        InitialReadingDecision,
    ]


def test_first_assessment_must_cover_every_known_issue_atomically() -> None:
    workflow = prepared_engine(shared=True)
    before = stable_engine_state(workflow)
    with pytest.raises(ValueError, match="every standard dimension"):
        workflow._accept_reading(reading())
    assert stable_engine_state(workflow) == before


def test_new_source_issue_needs_complete_model_assessment_before_adoption() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    before = stable_engine_state(workflow)
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    incomplete = unresolved_reading(("s1",))
    incomplete.dimensions.pop()
    incomplete.dimensions.append(assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"]))
    incomplete.additional_issues = [source_issue(query=False)]
    incomplete.requirements = [replacement]
    with pytest.raises(ValueError, match="every standard dimension"):
        workflow._accept_reading(incomplete)
    assert stable_engine_state(workflow) == before


def test_new_child_full_matrix_preserves_existing_issues_without_reemission() -> None:
    workflow = prepared_engine(shared=True)
    workflow._accept_reading(complete_assessments(reading(), ("i1", "i2")))
    old_rows = [row.model_copy(deep=True) for row in workflow.dimensions]
    child = unresolved_reading(("s1",))
    child.additional_issues = [source_issue(query=False)]
    workflow._accept_reading(child)
    assert workflow.dimensions[:24] == old_rows
    assert len(workflow.dimensions) == 36
    assert all(row.status == "unresolved" for row in workflow.dimensions[24:])
    assert all("not yet been assessed" not in row.reason for row in workflow.dimensions)


def test_supersession_unresolved_application_is_assessed_and_accepts_empty_delta() -> (
    None
):
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(ReadingDecision(requirements=[replacement], dimensions=[]))
    assert workflow.dimensions[0].status == "unresolved"
    assert workflow.plan is not None
    assert workflow._required_assessment_issue_ids(workflow.plan) == []
    workflow._accept_reading(ReadingDecision(dimensions=[]))
    assert workflow.dimensions[0].status == "unresolved"


def test_active_finding_and_literal_claim_checks_are_bound_in_existing_batches() -> (
    None
):
    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "pass"])
    result = workflow.run("Soru", "")
    assert result.status == "partial"
    assert len(gateway.calls) == 3 and len(reviewer.states) == 2
    assert (
        result.early_review is not None and "finding:r1" in result.early_review.scores
    )
    assert result.final_review is not None
    assert {
        "finding:r1",
        "claim:c1",
        "all_answer_claims",
    } <= result.final_review.scores.keys()
    finding = next(
        check for check in workflow._checks(draft=False) if check.requirement_id
    )
    assert finding.id == "finding:r1" and finding.requirement_id == "r1"
    assert "r1" in finding.instructions and "operative scope" in finding.instructions
    assert (
        "annulment" in finding.instructions
        and "temporal effect" in finding.instructions
    )
    claim = next(
        check
        for check in workflow._checks(draft=True, answer=draft())
        if check.claim_id
    )
    assert claim.id == "claim:c1" and claim.claim_id == "c1" and claim.issue_id == "i1"
    assert "c1" in claim.instructions and "unrelated items" in claim.instructions


def test_finding_checks_follow_active_versions_only() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    workflow._accept_reading(ReadingDecision(requirements=[replacement], dimensions=[]))
    checks = workflow._checks(draft=False)
    assert {check.requirement_id for check in checks if check.requirement_id} == {"r2"}
    assert "finding:r1" not in {check.id for check in checks}


@pytest.mark.parametrize("tamper", ["requirement_id", "instructions", "duplicate"])
def test_review_flag_must_retain_exact_host_owned_finding_binding(tamper: str) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    checks = workflow._checks(draft=False)
    finding = next(check for check in checks if check.requirement_id == "r1")
    scores = {check.id: 0.9 if check == finding else 0.1 for check in checks}
    flags = [finding]
    if tamper == "duplicate":
        flags.append(finding)
    else:
        flags = [finding.model_copy(update={tamper: "different"})]
    with patch.object(
        workflow.reviewer,
        "review",
        return_value=ReviewResult(completed=True, scores=scores, flags=flags),
    ):
        result = workflow._review("Soru", "", None)
    assert not result.completed and result.failure_reason == "review_inventory_invalid"


def prepare_repair(*flag_ids: str) -> LegalReviewEngine:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    checks = workflow._checks(draft=True, answer=draft())
    flags = [check for check in checks if check.id in flag_ids]
    assert len(flags) == len(flag_ids)
    workflow.final_review = ReviewResult(
        completed=True,
        scores={check.id: 0.9 if check in flags else 0.1 for check in checks},
        flags=flags,
    )
    workflow._start_repair(draft())
    return workflow


def test_repair_schema_requires_explicit_diagnosis_contract() -> None:
    with pytest.raises(ValidationError, match="repair_resolutions"):
        RepairReadingDecision.model_validate({"dimensions": []})
    assert "repair_resolutions" not in ReadingDecision.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "check_ids", [["claim:c1"], ["claim:c1", "unknown"], ["claim:c1", "claim:c1"]]
)
def test_repair_resolution_inventory_is_exact_and_atomic(check_ids: list[str]) -> None:
    workflow = prepare_repair("claim:c1", "source_conditions")
    decision = repair_reading()
    decision.repair_resolutions[0].check_ids = check_ids
    replacement = requirement("r2")
    replacement.supersedes_requirement_ids = ["r1"]
    decision.requirements = [replacement]
    with pytest.raises(ValueError, match="every flagged check exactly once"):
        workflow._accept_reading(decision)
    assert set(workflow.requirements) == {"r1"}
    assert workflow.requirement_history == {} and workflow.repair_resolutions == []


def test_grouped_draft_resolution_is_canonical_and_not_an_automatic_pass() -> None:
    workflow = prepare_repair("claim:c1", "source_conditions")
    decision = repair_reading()
    decision.repair_resolutions[0].check_ids.append("source_conditions")
    workflow._accept_reading(decision)
    assert len(workflow.repair_resolutions) == 1
    assert set(workflow.requirements) == {"r1"}
    result = workflow._result("unavailable", gap="Final legal review did not pass")
    assert result.answer is None
    assert {check.id for check in result.repair_checks} == {
        "claim:c1",
        "source_conditions",
    }
    assert result.repair_resolutions[0].check_ids == ["claim:c1", "source_conditions"]


def test_repair_support_cannot_invent_a_canonical_passage() -> None:
    workflow = prepare_repair("claim:c1")
    decision = repair_reading()
    decision.repair_resolutions[0].supports = [
        PassageSupport(citation=1, span_number=999)
    ]
    with pytest.raises(ValueError):
        workflow._accept_reading(decision)
    assert workflow.repair_resolutions == []


@pytest.mark.parametrize("scope", ["research", "draft"])
def test_finding_correction_cannot_leave_research_unchanged(scope: str) -> None:
    workflow = prepare_repair("finding:r1")
    decision = repair_reading("finding:r1")
    decision.repair_resolutions[0] = decision.repair_resolutions[0].model_copy(
        update={"scope": scope}
    )
    with pytest.raises(ValueError, match="changed finding or linked assessment"):
        workflow._accept_reading(decision)
    assert workflow.repair_resolutions == []


def test_finding_correction_and_affected_assessment_are_persistent_across_source_return() -> (
    None
):
    workflow = prepare_repair("finding:r1", "draft:i1:legal_basis_and_hierarchy")
    decision = repair_reading("finding:r1")
    decision.repair_resolutions[0].scope = "research"
    decision.repair_resolutions[0].check_ids.append(
        "draft:i1:legal_basis_and_hierarchy"
    )
    replacement = requirement("r2")
    replacement.rule = "The document condition must be satisfied before acceptance."
    replacement.supersedes_requirement_ids = ["r1"]
    decision.requirements = [replacement]
    decision.dimensions = [assessment("i1", LegalDimension.LEGAL_BASIS, ["r2"])]
    workflow._accept_reading(decision)
    followup = RepairReadingDecision(
        dimensions=[], repair_resolutions=decision.repair_resolutions
    )
    workflow._accept_reading(followup)
    assert set(workflow.requirements) == {"r2"}
    assert "r1" in workflow.requirement_history
    assert workflow.dimensions[0].requirement_ids == ["r2"]
    assert workflow.repair_resolutions[0].scope == "research"


def test_disputed_flag_retains_grounded_rebuttal_without_changing_the_gate() -> None:
    workflow = prepare_repair("finding:r1")
    decision = repair_reading("finding:r1")
    decision.repair_resolutions[0].disposition = "disputed"
    workflow._accept_reading(decision)
    assert set(workflow.requirements) == {"r1"}
    assert workflow.final_review is not None and workflow.final_review.flags
    assert workflow._result("unavailable").answer is None


def test_unresolved_repair_gap_keeps_issue_open_and_must_be_disclosed() -> None:
    workflow = prepare_repair("claim:c1")
    decision = repair_reading()
    decision.repair_resolutions[0].disposition = "unresolved"
    decision.repair_resolutions[0].supports = []
    decision.repair_resolutions[
        0
    ].correction = "The operative document requirement could not be established."
    workflow._accept_reading(decision)
    assert workflow.issue_closures()[0].status == "open"
    assert decision.repair_resolutions[0].correction in workflow.gaps
    with pytest.raises(ValueError, match="all unresolved"):
        workflow._validate_draft(draft())
    disclosed = workflow._bind_unresolved_issues(draft())
    workflow._validate_draft(disclosed)
    assert disclosed.unresolved_issue_ids == ["i1"]


def test_repair_diagnostics_reach_writer_but_not_independent_review_input() -> None:
    workflow, gateway, reviewer = engine(
        [plan(), reading(), draft(), repair_reading(), draft()],
        ["pass", "flag", "pass"],
    )
    result = workflow.run("Soru", "")
    assert result.status == "partial" and result.repair_used
    assert gateway.response_models[3] is RepairReadingDecision
    assert gateway.prompts[1] == READING_PROMPT
    assert gateway.prompts[3] == REPAIR_READING_PROMPT
    assert gateway.calls[3][1]["repair_contract"] == {
        "required_check_ids": ["claim:c1"],
        "grouped_check_ids_allowed": True,
    }
    assert gateway.calls[4][1]["repair_resolutions"]
    assert len(gateway.calls) == 5 and len(reviewer.states) == 3
    assert all(
        "repair_resolutions" not in state and "repair_contract" not in state
        for state in reviewer.states
    )
    assert result.repair_resolutions and result.repair_checks


@pytest.mark.parametrize("last_review", ["pass", "refusal"])
def test_unassessed_review_check_can_be_repaired_but_cannot_pass_unassessed(
    last_review: str,
) -> None:
    workflow, gateway, reviewer = engine(
        [plan(), reading(), draft(), repair_reading(), draft()],
        ["pass", "refusal", last_review],
    )
    result = workflow.run("Soru", "")
    assert result.repair_used
    assert len(reviewer.states) == 3
    assert gateway.calls[3][1]["repair_contract"] == {
        "required_check_ids": ["claim:c1"],
        "grouped_check_ids_allowed": True,
    }
    assert result.status == "partial"
    if last_review == "refusal":
        assert result.answer
        assert result.publication_mode == "review_incomplete"
        assert result.non_publication_reason is None


def test_completed_rejected_final_review_is_distinct_from_provider_failure() -> None:
    workflow, _, _ = engine(
        [plan(), reading(), draft(), repair_reading(), draft()],
        ["pass", "flag", "flag"],
    )
    result = workflow.run("Soru", "")
    assert result.status == "partial" and result.answer
    assert result.publication_mode == "review_incomplete"
    assert result.non_publication_reason is None
    assert result.final_review is not None and result.final_review.completed


def test_early_review_research_is_bound_executed_and_reassessed() -> None:
    from onyx.legal_review.models import (
        ReviewDiagnosis,
        ReviewDiagnosisBatch,
        ReviewResearchTask,
    )

    workflow, gateway, _ = engine([], ["pass"])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(reading())
    workflow._acquire(
        [
            SourceAction(
                issue_ids=["i1"],
                tool="search_corpus",
                arguments={"query": "Başvuru koşulları"},
            )
        ]
    )
    check = ReviewCheck(
        id="early-validity",
        instructions="Missing operative version?",
        issue_id="i1",
        dimension=LegalDimension.VALIDITY.value,
    )
    review = ReviewResult(completed=True, flags=[check], scores={check.id: 0.9})
    diagnosis = ReviewDiagnosisBatch(
        diagnoses=[
            ReviewDiagnosis(
                kind="research",
                assertion="The condition is stated as currently operative.",
                reason="The event-date version is unverified.",
                required_change="Read the amendment history.",
                supports=[PassageSupport(citation=1, span_number=1)],
                research_task_ids=["validity"],
                dimension=LegalDimension.VALIDITY,
                check_ids=[check.id],
            )
        ],
        research_tasks=[
            ReviewResearchTask(
                task_id="validity",
                subject="Application condition",
                question="Was this application condition amended by the event date?",
                query="başvuru şartı değişiklik yürürlük",
                existing_need_id=None,
                supports=[PassageSupport(citation=1, span_number=1)],
                dimension=LegalDimension.VALIDITY,
            )
        ],
    )

    class IndependentExaminer:
        def edit(
            self, state: dict[str, JsonValue], timeout_seconds: float
        ) -> EditorialEdits:
            del state, timeout_seconds
            raise AssertionError("Research diagnosis does not edit a draft")

        def examine(
            self,
            state: dict[str, JsonValue],
            checks: Sequence[ReviewCheck],
            timeout_seconds: float,
        ) -> PublicationReview:
            del state, checks, timeout_seconds
            raise AssertionError("This research test must not adjudicate an answer")

        def diagnose(
            self,
            state: dict[str, JsonValue],
            checks: Sequence[ReviewCheck],
            timeout_seconds: float,
        ) -> ReviewDiagnosisBatch:
            assert state["request"] == "Soru"
            assert list(checks) == [check]
            assert timeout_seconds > 0
            return diagnosis

    workflow.diagnoser = IndependentExaminer()
    workflow._diagnose_review("Soru", "", None, review)
    assert workflow.acquirer.searches == 2
    assert not gateway.calls
    searched_need_id = diagnosis.research_tasks[0].existing_need_id
    assert searched_need_id is not None
    request = RepairReadingDecision(
        dimensions=[
            DimensionAssessment(
                issue_id="i1",
                dimension=LegalDimension.VALIDITY,
                status="unresolved",
                reason="Operative amendment history is unread.",
                requirement_ids=[],
            )
        ],
        actions=[
            SourceAction(
                issue_ids=["i1"],
                tool="search_corpus",
                arguments={"query": "başvuru şartı değişiklik yürürlük"},
                research_need_ids=[searched_need_id],
            )
        ],
        repair_resolutions=[
            RepairResolution(
                check_ids=[check.id],
                diagnosis="Operative amendment history is unread.",
                correction="Read it before closing the issue.",
                disposition="unresolved",
                scope="research",
                supports=[],
            )
        ],
        research_resolutions=[
            ResearchResolution(
                check_ids=[check.id],
                issue_ids=["i1"],
                disposition="request_sources",
                reason="Read the operative amendment history.",
                supports=[],
                missing_user_facts=[],
            )
        ],
    )
    from onyx.legal_review.evidence_resolution import EvidenceResolutionPlan

    resolution_plan = EvidenceResolutionPlan(diagnosis, {check.id: {"i1"}})
    gateway.results = [
        resolution_plan.response_model().model_validate(
            {
                **request.model_dump(
                    exclude={"repair_resolutions", "research_resolutions"}
                ),
                "research_resolutions": [
                    {
                        "slot": "r0001",
                        **request.research_resolutions[0].model_dump(
                            exclude={"check_ids", "issue_ids"}
                        ),
                    }
                ],
            }
        )
    ]
    decision = workflow._reading("Soru", "")
    assert gateway.response_models[-1].__name__ == "BoundEvidenceResolutionDecision"
    assert "repair_contract" not in gateway.calls[-1][1]
    assert gateway.calls[-1][1]["review_diagnoses"] == diagnosis.model_dump(mode="json")
    assert workflow.issue_closures()[0].status == "open"
    workflow._acquire(decision.actions)
    assert workflow.acquirer.searches == 2
    assert not workflow.pending_actions
    assert workflow.issue_closures()[0].status == "open"
    # Merely executing retrieval is not proof that the research question is resolved.
    assert not workflow.repair_resolutions
    assert workflow.research_resolutions

    omitted_work = EvidenceResolutionDecision(dimensions=[], research_resolutions=[])
    with pytest.raises(ValueError, match="Every independent research question"):
        workflow._accept_reading(omitted_work)

    missing_work = request.model_copy(deep=True)
    missing_work.actions = []
    missing_work.repair_resolutions[0].disposition = "correct"
    with pytest.raises(ValueError, match="executable source operations"):
        workflow._accept_reading(missing_work)

    claimed_done = request.model_copy(deep=True)
    claimed_done.actions = []
    claimed_done.research_resolutions[0].disposition = "resolved"
    claimed_done.research_resolutions[0].supports = [
        PassageSupport(citation=1, span_number=1)
    ]
    with pytest.raises(ValueError, match="executed source operation"):
        workflow._accept_reading(claimed_done)

    dismissed = claimed_done.model_copy(deep=True)
    dismissed.research_resolutions[0].disposition = "disputed"
    with pytest.raises(ValueError, match="executed source operation"):
        workflow._accept_reading(dismissed)

    # Re-reading a known original is genuine source work; a novel citation is not required.
    workflow.acquirer.receipts.append(
        {
            "tool": "read_provision",
            "status": "found",
            "issue_ids": ["i1"],
            "evidence_ids": [1],
        }
    )
    workflow._accept_reading(claimed_done)
    assert workflow.research_resolutions[0].disposition == "resolved"

    # The next review reuses this exact completed need instead of searching again.
    workflow.acquirer.receipts[-1]["research_need_ids"] = [searched_need_id]
    workflow._review_receipts_start = len(workflow.acquirer.receipts)
    workflow._accept_reading(claimed_done)
    assert workflow.research_resolutions[0].disposition == "resolved"
    assert workflow.acquirer.searches == 2

    # An older or newer unrelated search cannot justify this research disposition.
    workflow.acquirer.receipts[-1]["research_need_ids"] = ["another-need"]
    with pytest.raises(ValueError, match="executed source operation"):
        workflow._accept_reading(claimed_done)
    workflow.acquirer.receipts.append(
        {
            "tool": "search_corpus",
            "status": "found",
            "issue_ids": ["i1"],
            "research_need_ids": ["another-need"],
            "evidence_ids": [1],
        }
    )
    with pytest.raises(ValueError, match="executed source operation"):
        workflow._accept_reading(claimed_done)


def test_final_review_starts_fresh_work_without_losing_early_audit() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(reading())
    early = ReviewCheck(id="early", instructions="Early check", issue_id="i1")
    final = ReviewCheck(id="final", instructions="Final check", claim_id="c1")
    workflow._start_repair(None, ReviewResult(completed=True, flags=[early]))
    workflow._start_repair(draft(), ReviewResult(completed=True, flags=[final]))
    assert set(workflow.repair_checks) == {"final"}
    assert set(workflow._repair_check_issues) == {"final"}
    assert len(workflow.review_work_history) == 1


def test_completed_contract_correction_is_atomic_and_available_only_once() -> None:
    from onyx.legal_review.contracts import ReadingContractError

    invalid = reading()
    invalid.dimensions[0].requirement_ids = []
    workflow, gateway, _ = engine([invalid, reading()], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._reading("Soru", "")
    assert workflow.reading_correction_used
    assert len(gateway.calls) == 2
    corrected_state = gateway.calls[1][1]
    assert corrected_state["requirements"] == []
    assert corrected_state["dimension_assessments"] == []
    assert corrected_state["reading_contract_correction"]
    accepted = workflow.state("Soru", "")
    gateway.results = [invalid]
    with pytest.raises(ReadingContractError):
        workflow._reading("Soru", "")
    assert len(gateway.calls) == 3
    assert workflow.state("Soru", "")["requirements"] == accepted["requirements"]
    assert (
        workflow.state("Soru", "")["dimension_assessments"]
        == accepted["dimension_assessments"]
    )


def test_grouped_review_searches_are_dispatched_in_one_batch_before_reader() -> None:
    from onyx.legal_review.models import (
        ReviewDiagnosis,
        ReviewDiagnosisBatch,
        ReviewResearchTask,
    )

    workflow, gateway, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    workflow._accept_reading(reading())
    checks = [
        ReviewCheck(id="basis", instructions="Check basis", issue_id="i1"),
        ReviewCheck(id="validity", instructions="Check validity", issue_id="i1"),
        ReviewCheck(id="exception", instructions="Check exception", issue_id="i1"),
    ]
    diagnoses = ReviewDiagnosisBatch(
        diagnoses=[
            ReviewDiagnosis(
                kind="research",
                assertion="The requirement is unconditional.",
                reason="Material conditions are unread.",
                required_change="Read the controlling conditions.",
                supports=[PassageSupport(citation=1, span_number=1)],
                research_task_ids=task_ids,
                dimension=None,
                check_ids=ids,
            )
            for ids, task_ids in [
                (["basis", "validity"], ["operative", "exception"]),
                (["exception"], ["exception"]),
            ]
        ],
        research_tasks=[
            ReviewResearchTask(
                task_id=identity,
                subject=subject,
                question=question,
                query=query,
                existing_need_id=None,
                dimension=dimension,
                supports=[PassageSupport(citation=1, span_number=1)],
            )
            for identity, subject, question, query, dimension in [
                (
                    "operative",
                    "Primary basis",
                    "What is its operative scope?",
                    "primary basis amendments",
                    LegalDimension.VALIDITY,
                ),
                (
                    "exception",
                    "Eligibility exception",
                    "When does this exception apply?",
                    "eligibility exception conditions",
                    LegalDimension.EXCEPTIONS,
                ),
            ]
        ],
    )
    workflow.diagnoser = Mock()
    workflow.diagnoser.diagnose.return_value = diagnoses
    with patch.object(
        workflow.acquirer, "acquire", wraps=workflow.acquirer.acquire
    ) as acquire:
        workflow._diagnose_review(
            "Soru", "", None, ReviewResult(completed=True, flags=checks)
        )
    acquire.assert_called_once()
    assert len(acquire.call_args.args[0]) == 2
    assert workflow.acquirer.searches == 2
    assert len(workflow.research.needs) == 2
    assert all(need.attempted for need in workflow.research.needs.values())
    assert not gateway.calls


@pytest.mark.parametrize("disposition", ["defect", "rebutted", "disclosed_limitation"])
def test_final_adjudication_preserves_scores_and_blocks_actual_defects(
    disposition: str,
) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    check = ReviewCheck(
        id="claim:c1",
        instructions="Does the claim omit the document condition?",
        claim_id="c1",
    )
    workflow.final_review = ReviewResult(
        completed=True, scores={check.id: 0.9}, flags=[check]
    )
    examiner = Mock()
    finding = PublicationFinding.model_validate(
        {
            "check_id": check.id,
            "disposition": disposition,
            "target": "assertion",
            "answer_quotes": [draft().answer],
            "reason": "Compare the actual answer condition with the original.",
            "required_change": "Restore the omitted condition."
            if disposition == "defect"
            else None,
            "supports": [{"citation": 1, "span_number": 1}],
        }
    )
    adjudication = PublicationReview(findings=[finding])
    examiner.examine.return_value = adjudication
    workflow.diagnoser = examiner
    assert workflow._final_review_is_publishable("Soru", "", draft()) is (
        disposition != "defect"
    )
    assert workflow.final_adjudication == adjudication
    assert workflow.final_review.scores == {
        check.id: 0.9
    } and workflow.final_review.flags == [check]
    examiner.diagnose.assert_not_called()
    assert examiner.examine.call_args.args[0]["draft"] == draft().model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    "invalid", ["missing_check", "unknown_passage", "private_quote"]
)
def test_final_adjudication_cannot_bypass_evidence_or_inventory(invalid: str) -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    check = ReviewCheck(id="dimension", instructions="Check")
    workflow.final_review = ReviewResult(
        completed=True, scores={check.id: 0.9}, flags=[check]
    )
    examiner = Mock()
    examiner.examine.return_value = PublicationReview(
        findings=[
            PublicationFinding(
                check_id="wrong" if invalid == "missing_check" else check.id,
                disposition="rebutted",
                target="assertion",
                required_change=None,
                answer_quotes=[
                    "Private research claim absent from answer"
                    if invalid == "private_quote"
                    else draft().answer
                ],
                reason="The condition is preserved.",
                supports=[
                    PassageSupport(
                        citation=1,
                        span_number=999 if invalid == "unknown_passage" else 1,
                    )
                ],
            )
        ]
    )
    workflow.diagnoser = examiner
    with pytest.raises(ValueError):
        workflow._final_review_is_publishable("Soru", "", draft())
    assert workflow.final_adjudication is None


def test_rebutted_draft_flags_do_not_trigger_research_or_rewrite() -> None:
    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "flag"])
    examiner = Mock()
    examiner.examine.return_value = PublicationReview(
        findings=[
            PublicationFinding(
                check_id="claim:c1",
                disposition="rebutted",
                target="assertion",
                answer_quotes=[draft().answer],
                reason="The required submission is already stated.",
                required_change=None,
                supports=[PassageSupport(citation=1, span_number=1)],
            )
        ]
    )
    workflow.diagnoser = examiner
    result = workflow.run("Soru", "")
    assert result.answer is not None and result.status == "partial"
    assert not result.repair_used and len(gateway.calls) == 3
    assert len(reviewer.states) == 2 and result.final_adjudication is not None
    assert result.final_review is not None and result.final_review.flags
    examiner.examine.assert_called_once()
    examiner.diagnose.assert_not_called()


def test_source_usage_is_in_the_same_review_inventory_without_new_issues() -> None:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    assert workflow.plan is not None
    original_plan = workflow.plan.model_dump()
    before = dict(workflow.context.budget.used)
    research_checks = workflow._checks(draft=False)
    answer_checks = workflow._checks(draft=True, answer=draft())
    assert not any(check.source_citation for check in research_checks)
    source_checks = [check for check in answer_checks if check.source_citation]
    assert [(check.id, check.source_citation) for check in source_checks] == [
        ("source_use:1", 1)
    ]
    assert {check.id for check in answer_checks}.issuperset(
        {"claim:c1", "source_conditions", "all_answer_claims", "request_coverage"}
    )
    assert workflow.plan.model_dump() == original_plan
    assert workflow.context.budget.used == before


def test_independent_editor_changes_wording_without_research_or_repair_loop() -> None:
    from onyx.legal_review.drafting import EditorialEdits

    workflow, gateway, reviewer = engine([plan(), reading(), draft()], ["pass", "flag"])
    finding = PublicationFinding(
        check_id="claim:c1",
        disposition="defect",
        target="assertion",
        reason="The prerequisite must qualify the conclusion.",
        required_change="State the document condition in the conclusion.",
        supports=[PassageSupport(citation=1, span_number=1)],
        repair_kind="correction",
        answer_quotes=[draft().answer],
    )
    examiner = Mock()
    examiner.examine.return_value = PublicationReview(findings=[finding])

    def edit(state: dict[str, JsonValue], _timeout: float) -> EditorialEdits:
        base = GeneratedDraft.model_validate(state["repair_base"])
        replacement = base.blocks[0].model_copy(deep=True)
        replacement.text = "Belge ibraz edilirse başvuru kabul edilir. [1]"
        return EditorialEdits(
            replacements=[replacement],
            unresolved_issue_ids=["i1"],
            resolved_check_ids=["claim:c1"],
            unresolved_check_ids=[],
        )

    examiner.edit.side_effect = edit
    workflow.diagnoser = examiner
    result = workflow.run("Soru", "")
    assert result.status == "partial" and result.publication_mode == "editor_adjusted"
    assert result.answer and "Belge ibraz edilirse" in result.answer
    assert "Açık kalan kontrol bulguları" not in result.answer
    assert result.editorial_changes[0]["before"] == draft().answer
    assert not result.repair_used and len(gateway.calls) == 3
    assert len(reviewer.states) == 2
    examiner.edit.assert_called_once()
    examiner.diagnose.assert_not_called()


@pytest.mark.parametrize("expired", [True, False])
def test_editor_failure_or_deadline_retains_answer_and_visible_findings(
    expired: bool,
) -> None:
    workflow, _, _ = engine([plan(), reading(), draft()], ["pass", "pass"])
    workflow.run("Soru", "")
    workflow.final_adjudication = PublicationReview(
        findings=[
            PublicationFinding(
                check_id="claim:c1",
                disposition="defect",
                target="assertion",
                reason="Missing document condition",
                required_change="Belge şartı doğrulanmalı.",
                supports=[],
                repair_kind="research",
                research_query="Belge şartı",
                answer_quotes=[draft().answer],
            )
        ]
    )
    examiner = Mock()
    examiner.edit.side_effect = TimeoutError("Editor exhausted")
    workflow.diagnoser = examiner
    if expired:
        workflow.context.deadline = 0
    result = workflow._publish_partial("Soru", "", "Deadline", time_exhausted=True)
    assert result.status == "partial" and result.publication_mode == "limit_reached"
    assert result.answer and draft().answer in result.answer
    assert "Belge şartı doğrulanmalı." in result.answer
    assert "doğrulanmadı" in result.answer
    assert examiner.edit.call_count == (0 if expired else 1)


def test_timeout_before_first_draft_preserves_findings_but_cancellation_never_publishes() -> (
    None
):
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    workflow.context.deadline = 0
    with patch.object(workflow, "_run", side_effect=TimeoutError("Deadline")):
        result = workflow.run("Soru", "")
    assert result.status == "partial" and result.answer and RULE in result.answer
    assert "olaya uygulanması tamamlanmadı" in result.answer
    workflow.context.cancel()
    with patch.object(workflow, "_run", side_effect=TimeoutError("Deadline")):
        stopped = workflow.run("Soru", "")
    assert stopped.status == "cancelled" and stopped.answer is None


def test_issue_plan_updates_and_publication_decision_are_visible_in_graph() -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    steps: dict[str, list[SimpleNamespace]] = {}

    def capture(operation: str, _input: object) -> object:
        node = SimpleNamespace(output_value=None)
        steps.setdefault(operation, []).append(node)
        return nullcontext(node)

    workflow, _, _ = engine([plan(), reading(), draft()], ["pass", "pass"])
    with patch("onyx.legal_review.engine.graph_step", side_effect=capture):
        result = workflow.run("Soru", "")
    assert steps["legal_review.issue_plan"][0].output_value["discovery_queries"]
    assert steps["legal_review.issue_update"][0].output_value["closures"]
    assert (
        steps["legal_review.publication_decision"][0].output_value["status"]
        == result.status
    )
