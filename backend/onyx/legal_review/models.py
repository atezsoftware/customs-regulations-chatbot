from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)

from onyx.legal_review.passages import PassageReference as PassageSupport
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS

DIAGNOSIS_MODEL: Literal["gpt-6.1-sol"] = "gpt-6.1-sol"


class LegalDimension(StrEnum):
    LEGAL_BASIS = "legal_basis_and_hierarchy"
    VALIDITY = "validity_and_timing"
    CASE_LAW = "case_law_and_rulings"
    EXCEPTIONS = "exceptions_and_exemptions"
    PENALTIES = "penalties_and_reductions"
    TAX = "tax_and_financial_consequences"
    ALTERNATIVES = "alternative_routes"
    PROCEDURE = "procedure_and_deadlines"
    DOCUMENTS = "evidence_and_documents"
    OPERATIONS = "operational_steps"
    FACTS = "missing_facts"
    LIABILITY = "liability_and_conflicts"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewCheck(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    id: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    issue_id: str | None = None
    dimension: str | None = None
    requirement_id: str | None = Field(default=None, min_length=1)
    claim_id: str | None = Field(default=None, min_length=1)
    source_citation: int | None = Field(default=None, gt=0)


class ReviewResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    completed: bool
    scores: dict[str, float] = Field(default_factory=dict)
    flags: list[ReviewCheck] = Field(default_factory=list)
    unassessed_checks: list[ReviewCheck] = Field(default_factory=list)
    failure_reason: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    error_code: str | None = None
    error_param: str | None = None
    request_id: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class JevProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    route: Literal["typesafe", "openrouter"]
    api_key: SecretStr = Field(exclude=True, repr=False)
    provider_name: str | None = None


class DecisionProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    route: Literal["openai_decisions"] = "openai_decisions"
    api_key: SecretStr = Field(exclude=True, repr=False)
    provider_name: str | None = None


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    timeout_seconds: float | None = Field(default=600, gt=0)
    finalization_reserve_seconds: float = Field(default=300, gt=0)
    publication_reserve_seconds: float = Field(default=120, gt=0)
    editor_reserve_seconds: float = Field(default=60, gt=0)
    assume_current_corpus: bool = True
    always_review_research: bool = False
    max_call_seconds: int = Field(default=45, gt=0)
    max_model_calls: int = Field(default=32, ge=12)
    max_tools: int = Field(default=96, gt=0)
    max_parallel_tools: int = Field(default=4, ge=1, le=8)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_context_tokens: int | None = Field(default=None, gt=0)
    max_evidence_bytes: int = Field(default=2_000_000, gt=0)
    max_generation_output_tokens: int = Field(default=65_536, gt=0, le=65_536)

    @model_validator(mode="after")
    def reserve_fits(self) -> WorkflowPolicy:
        if (
            self.timeout_seconds is not None
            and self.finalization_reserve_seconds >= self.timeout_seconds
        ):
            raise ValueError("Finalization reserve must fit within the deadline")
        return self


class Issue(StrictModel):
    issue_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_outcome: str = Field(min_length=1)
    supplied_facts: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(default_factory=list)
    origin: Literal["question", "source"] = "question"
    parent_issue_id: str | None = None
    trigger_dimension: LegalDimension | None = None
    supporting_requirement_ids: list[str] = Field(default_factory=list)
    material_reason: str | None = None
    closure_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bounded_queries(self) -> Issue:
        if self.origin == "source" and not (
            self.parent_issue_id
            and self.trigger_dimension is not None
            and self.supporting_requirement_ids
            and self.material_reason
            and self.material_reason.strip()
            and self.closure_criteria
            and all(criterion.strip() for criterion in self.closure_criteria)
        ):
            raise ValueError(
                "Source issues need a parent, canonical triggers, material reason and closure criteria"
            )
        if self.origin == "question" and self.parent_issue_id is not None:
            raise ValueError("Question issues cannot have a source dependency parent")
        if any(
            not query.strip() or len(query) > REGULATORY_MAX_SEARCH_QUERY_CHARS
            for query in self.research_queries
        ):
            raise ValueError("Every issue needs bounded, nonempty research queries")
        return self


class IssuePlan(StrictModel):
    language: str = Field(min_length=2, max_length=35)
    issues: list[Issue] = Field(min_length=1)
    missing_user_facts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_issues(self) -> IssuePlan:
        ids = [issue.issue_id for issue in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("Issue identities must be unique")
        by_id = {issue.issue_id: issue for issue in self.issues}
        for issue in self.issues:
            seen = {issue.issue_id}
            parent = issue.parent_issue_id
            while parent is not None:
                if parent in seen or parent not in by_id:
                    raise ValueError("Issue dependencies must be known and acyclic")
                seen.add(parent)
                parent = by_id[parent].parent_issue_id
        return self


class DiscoveryQuery(StrictModel):
    query: str = Field(min_length=1, max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS)
    issue_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def bounded_query(self) -> DiscoveryQuery:
        if not self.query.strip() or any(
            not identity.strip() for identity in self.issue_ids
        ):
            raise ValueError(
                "Discovery queries need nonempty text and issue identities"
            )
        return self


class PlannedIssue(Issue):
    origin: Literal["question"] = "question"
    material_reason: str = Field(min_length=1)
    closure_criteria: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def actionable_issue(self) -> PlannedIssue:
        if not self.material_reason.strip() or any(
            not item.strip() for item in self.closure_criteria
        ):
            raise ValueError(
                "Planned issues need a material reason and actionable closure criteria"
            )
        return self


class RequestedOutcome(StrictModel):
    request: str = Field(
        min_length=1,
        description="A distinct requested result or alternative from the complete user question, without assuming the legal answer.",
    )
    issue_ids: list[str] = Field(min_length=1)


class InitialDiscoveryPlan(IssuePlan):
    issues: list[PlannedIssue] = Field(min_length=1)
    requested_outcomes: list[RequestedOutcome] = Field(
        min_length=1,
        description="Cover every explicit requested result and alternative. Many requests can map to one issue, and one request can map to several. Do not omit later subquestions.",
    )

    @field_validator("issues", mode="before")
    @classmethod
    def read_issue_models(cls, value: object) -> object:
        if isinstance(value, list):
            return [
                row.model_dump() if isinstance(row, Issue) else row for row in value
            ]
        return value

    discovery_queries: list[DiscoveryQuery] = Field(
        min_length=1,
        description="Required shared initial searches, each bound to the covered issue IDs.",
    )

    @model_validator(mode="after")
    def known_query_issues(self) -> InitialDiscoveryPlan:
        known = {issue.issue_id for issue in self.issues}
        covered = {
            identity for row in self.requested_outcomes for identity in row.issue_ids
        }
        if covered != known:
            raise ValueError("Requested outcomes must cover exactly the planned issues")
        if any(set(query.issue_ids) - known for query in self.discovery_queries):
            raise ValueError("Discovery queries must refer to known issue identities")
        if {
            identity for query in self.discovery_queries for identity in query.issue_ids
        } != known:
            raise ValueError("Every planned issue must be covered by initial discovery")
        return self


class SourceAction(StrictModel):
    issue_ids: list[str] = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    research_need_ids: list[str] = Field(default_factory=list)
    retry_reason: str | None = None


class ResearchNeed(StrictModel):
    need_id: str
    issue_ids: list[str]
    question: str
    subject: str | None = None
    origin: Literal["question", "source", "review"]
    dimension: LegalDimension | None = None
    covered_dimensions: list[LegalDimension] = Field(default_factory=list)
    parent_need_id: str | None = None
    trigger_supports: list[PassageSupport] = Field(default_factory=list)
    attempted: bool = False
    query: str | None = None
    attempted_queries: list[str] = Field(default_factory=list)
    receipt_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[int] = Field(default_factory=list)


class Requirement(StrictModel):
    requirement_id: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    supports: list[PassageSupport] = Field(min_length=1)
    missing_user_facts: list[str] = Field(default_factory=list)
    supersedes_requirement_ids: list[str] = Field(default_factory=list)


class RequirementRecord(Requirement):
    # A recorded version window is distinct from a legal annulment determination.
    validity: Literal["unknown", "within_recorded_window", "outside_recorded_window"]
    legal_status: Literal["in_force", "amended", "annulled", "unknown"] = "unknown"


class DimensionAssessment(StrictModel):
    issue_id: str = Field(min_length=1)
    dimension: LegalDimension
    status: Literal["addressed", "not_applicable", "unresolved"]
    reason: str = Field(min_length=1)
    requirement_ids: list[str] = Field(
        description="Always include this array. For addressed, link at least one existing or newly extracted original-backed finding; an empty array is allowed only for unresolved or not_applicable."
    )


class ReadingDecision(StrictModel):
    additional_issues: list[Issue] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    dimensions: list[DimensionAssessment] = Field(
        description="Required assessment array: full matrix for initial or new issues; explicit sparse updates for previously assessed issues."
    )
    actions: list[SourceAction] = Field(default_factory=list, max_length=16)
    evidence_gaps: list[str] = Field(default_factory=list)


class InitialReadingDecision(ReadingDecision):
    dimensions: list[DimensionAssessment] = Field(
        min_length=len(LegalDimension),
        description="Initial assessment: exactly all twelve dimensions for every known issue and any new source issue, with explicit addressed, not_applicable or unresolved reasoning.",
    )


class RepairResolution(StrictModel):
    check_ids: list[str] = Field(min_length=1)
    diagnosis: str = Field(min_length=1)
    correction: str = Field(min_length=1)
    disposition: Literal["correct", "unresolved", "disputed"]
    scope: Literal["research", "draft"]
    supports: list[PassageSupport]


class ResearchDisposition(StrictModel):
    disposition: Literal[
        "request_sources",
        "resolved",
        "needs_user_facts",
        "disputed",
        "exhausted",
        "unresolved",
    ]
    reason: str = Field(min_length=1)
    supports: list[PassageSupport]
    missing_user_facts: list[str]


class ResearchResolution(ResearchDisposition):
    check_ids: list[str] = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)


class EvidenceResolutionDecision(ReadingDecision):
    research_resolutions: list[ResearchResolution] = Field(
        description="Required dispositions for every independent research diagnosis, bound to its check_ids. Executing discovery alone does not resolve any missing legal effect."
    )


class RepairReadingDecision(ReadingDecision):
    repair_resolutions: list[RepairResolution] = Field(min_length=1)
    research_resolutions: list[ResearchResolution] = Field(
        default_factory=list,
        description="Account separately for every research diagnosis. A prose caveat does not resolve a source question. Request sources, resolve with newly read evidence, or explain a grounded rebuttal or a specific missing user fact.",
    )


class ResearchReadingDecision(RepairReadingDecision):
    research_resolutions: list[ResearchResolution] = Field(min_length=1)


class DiagnosisStatement(StrictModel):
    assertion: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    required_change: str = Field(min_length=1)
    supports: list[PassageSupport]
    dimension: LegalDimension | None


class DiagnosisFinding(DiagnosisStatement):
    kind: Literal["correction", "research", "disputed"]
    research_task_ids: list[str]

    @model_validator(mode="after")
    def actionable_research(self) -> DiagnosisFinding:
        if (self.kind == "research") != bool(self.research_task_ids):
            raise ValueError(
                "Only research diagnoses must bind material research tasks"
            )
        if len(self.research_task_ids) != len(set(self.research_task_ids)):
            raise ValueError("Diagnosis repeats a research task")
        return self


class ReviewDiagnosis(DiagnosisFinding):
    check_ids: list[str] = Field(min_length=1)


class DiagnosisReference(StrictModel):
    same_as: str = Field(
        min_length=1,
        description="An eligible earlier check slot in the same dimension with the same concrete diagnosis. Different dimensions may share tasks without replacing their diagnoses.",
    )


class ResearchQuestion(StrictModel):
    subject: str = Field(
        min_length=1,
        description="The single controlling norm, provision or source whose missing legal effect this task investigates. Do not list independent targets together.",
    )
    question: str = Field(min_length=1)
    dimension: LegalDimension
    supports: list[PassageSupport]
    query: str | None = Field(
        min_length=1, max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS
    )
    existing_need_id: str | None = Field(
        description="Eligible specific gap with the same subject, dimension and legal question, or null for a new gap requiring its one query. Initial broad discovery is never eligible.",
    )

    @model_validator(mode="after")
    def actionable_task(self) -> ResearchQuestion:
        if not self.existing_need_id and not (self.query and self.query.strip()):
            raise ValueError("A new material research task requires its one query")
        return self


class ReviewResearchTask(ResearchQuestion):
    task_id: str = Field(min_length=1)


class FreshResearchQuestion(ResearchQuestion):
    query: str = Field(
        min_length=1, max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS, pattern=r"\S"
    )
    existing_need_id: None


class AttemptedResearchQuestion(ResearchQuestion):
    query: str | None = Field(
        min_length=1, max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS
    )
    existing_need_id: str = Field(min_length=1)


class ReviewDiagnosisBatch(StrictModel):
    diagnoses: list[ReviewDiagnosis] = Field(min_length=1)
    research_tasks: list[ReviewResearchTask]
    research_coverage: dict[str, list[LegalDimension]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bound_tasks(self) -> ReviewDiagnosisBatch:
        identities = [task.task_id for task in self.research_tasks]
        referenced = {
            task_id for row in self.diagnoses for task_id in row.research_task_ids
        }
        if len(identities) != len(set(identities)) or set(identities) != referenced:
            raise ValueError(
                "Every unique research task must bind a diagnosis, with no unknown task references"
            )
        if self.research_coverage.keys() - set(identities):
            raise ValueError("Research coverage must bind known investigations")
        return self


class PublicationDecision(StrictModel):
    disposition: Literal["defect", "rebutted", "disclosed_limitation"]
    target: Literal["assertion", "omission", "not_applicable"]
    reason: str = Field(min_length=1)
    required_change: str | None
    supports: list[PassageSupport]
    repair_kind: Literal["correction", "research"] | None = None
    research_query: str | None = Field(
        default=None, min_length=1, max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS
    )

    @model_validator(mode="after")
    def grounded_disposition(self) -> "PublicationDecision":
        if self.repair_kind == "research" and not self.research_query:
            raise ValueError("A research defect needs its focused discovery query")
        if self.repair_kind != "research" and self.research_query is not None:
            raise ValueError("Only a research defect may request discovery")
        if self.disposition == "defect":
            if self.target == "not_applicable" or not (
                self.required_change and self.required_change.strip()
            ):
                raise ValueError(
                    "A publication defect requires an applicable correction"
                )
        elif self.required_change is not None:
            raise ValueError(
                "An acceptable answer must not carry a required correction"
            )
        return self


class PublicationAssessment(PublicationDecision):
    answer_quotes: list[str]

    @model_validator(mode="after")
    def literal_assertion(self) -> "PublicationAssessment":
        if self.target == "assertion" and not self.answer_quotes:
            raise ValueError("An assertion assessment requires literal answer text")
        if any(not quote.strip() for quote in self.answer_quotes):
            raise ValueError("Answer quotations must be nonempty")
        return self


class PublicationFinding(PublicationAssessment):
    check_id: str = Field(min_length=1)


class PublicationReview(StrictModel):
    findings: list[PublicationFinding] = Field(min_length=1)


class AnswerClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    answer_excerpt: str = Field(min_length=1)
    supports: list[PassageSupport] = Field(min_length=1)


class DraftAnswer(StrictModel):
    answer: str = Field(min_length=1)
    claims: list[AnswerClaim]
    unresolved_issue_ids: list[str] = Field(default_factory=list)


class IssueClosure(StrictModel):
    issue_id: str
    status: Literal["open", "partial", "closed"]
    blocking_child_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class WorkflowResult(StrictModel):
    status: Literal["verified", "partial", "unavailable", "cancelled"]
    non_publication_reason: Literal["review_rejected", "time_exhausted"] | None = None
    publication_mode: Literal[
        "reviewed", "editor_adjusted", "limit_reached", "review_incomplete"
    ] = "reviewed"
    editorial_changes: list[dict[str, JsonValue]] = Field(default_factory=list)
    reading_correction_used: bool = False
    answer: str | None = None
    plan: IssuePlan | None = None
    requested_outcomes: list[RequestedOutcome] = Field(default_factory=list)
    requirements: list[RequirementRecord] = Field(default_factory=list)
    dimensions: list[DimensionAssessment] = Field(default_factory=list)
    issue_closures: list[IssueClosure] = Field(default_factory=list)
    source_issue_triggers: list[dict[str, JsonValue]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    early_review: ReviewResult | None = None
    final_review: ReviewResult | None = None
    repair_used: bool = False
    repair_checks: list[ReviewCheck] = Field(default_factory=list)
    repair_resolutions: list[RepairResolution] = Field(default_factory=list)
    review_diagnoses: ReviewDiagnosisBatch | None = None
    final_adjudication: PublicationReview | None = None
    draft_adjudication: PublicationReview | None = None
    review_work_history: list[JsonValue] = Field(default_factory=list)
    research_resolutions: list[ResearchResolution] = Field(default_factory=list)
    research_needs: list[ResearchNeed] = Field(default_factory=list)
    skipped_searches: list[dict[str, JsonValue]] = Field(default_factory=list)
    source_journey: list[dict[str, JsonValue]] = Field(default_factory=list)

    @model_validator(mode="after")
    def rejected_review_has_no_published_answer(self) -> "WorkflowResult":
        if self.publication_mode != "reviewed" and self.status == "verified":
            raise ValueError(
                "Unfinished review or editorial publication must remain partial"
            )
        if self.non_publication_reason is not None and (
            self.status != "unavailable"
            or self.answer is not None
            or (
                self.non_publication_reason == "review_rejected"
                and (self.final_review is None or not self.final_review.completed)
            )
        ):
            raise ValueError(
                "A rejected review requires a completed review and no answer"
            )
        return self
