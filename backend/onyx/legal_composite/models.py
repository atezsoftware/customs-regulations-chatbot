from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.db.legal_composite_sources import SourceKind
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: float = Field(default=120, gt=0)
    finalization_reserve_seconds: float = Field(default=40, gt=0)
    selection_reserve_seconds: float = Field(default=0, ge=0)
    max_model_calls: int = Field(default=8, ge=4)
    max_input_tokens: int = Field(default=120_000, gt=0)
    max_output_tokens: int = Field(default=24_000, gt=0)
    max_cost_usd: float = Field(default=0.10, gt=0)
    max_call_seconds: float = Field(default=45, gt=0)
    max_context_tokens: int = Field(default=32_000, gt=0)
    final_output_tokens: int = Field(default=4_096, gt=0)
    max_tools: int = Field(default=24, gt=0)
    max_parallel_tools: int = Field(default=4, ge=1, le=12)
    max_search_calls: int = Field(default=8, gt=0)
    max_research_rounds: int = Field(default=4, ge=1)
    max_reviews: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def reserve_fits(self) -> WorkflowPolicy:
        if (
            self.finalization_reserve_seconds + self.selection_reserve_seconds
            >= self.timeout_seconds
        ):
            raise ValueError("Finalization reserve must fit within the deadline")
        return self


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchNeed(StrictModel):
    need_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    governing_source: str
    conditions_to_check: list[str]
    source_kinds: list[SourceKind] = Field(default_factory=list)


class GapResolution(StrictModel):
    need_id: str = Field(min_length=1)
    gap: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)


class IssueResearchNeed(ResearchNeed):
    required_outcome: str = ""
    research_dimensions: list[str] = Field(default_factory=list)
    relevant_facts: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    evidence_gap_resolutions: list[GapResolution] = Field(default_factory=list)


class SourceAction(StrictModel):
    need_ids: list[str] = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    source_kind: SourceKind | None = None


class ResearchPlan(StrictModel):
    language: str = Field(min_length=2, max_length=35)
    requires_sources: bool
    needs: list[ResearchNeed] = Field(min_length=1)
    discovery_query: str = Field(
        default="", max_length=REGULATORY_MAX_SEARCH_QUERY_CHARS
    )
    initial_actions: list[SourceAction]
    missing_user_facts: list[str]

    @model_validator(mode="after")
    def unique_ids(self) -> ResearchPlan:
        ids = [need.need_id for need in self.needs]
        if len(ids) != len(set(ids)):
            raise ValueError("Research needs must have unique identities")
        known = set(ids)
        if any(set(action.need_ids) - known for action in self.initial_actions):
            raise ValueError("Action refers to an unknown research need")
        return self


class IssueResearchPlan(ResearchPlan):
    needs: list[IssueResearchNeed] = Field(min_length=1)


class PassageSupport(StrictModel):
    citation: int = Field(gt=0)
    quotation: str = Field(min_length=1)


class SpanSupport(PassageSupport):
    span_id: str | None = Field(default=None, min_length=1)
    quotation: str = ""

    @model_validator(mode="before")
    @classmethod
    def accept_literal_support(cls, value: object) -> object:
        if isinstance(value, PassageSupport) and not isinstance(value, cls):
            return value.model_dump(mode="python")
        return value

    @model_validator(mode="after")
    def witness_present(self) -> SpanSupport:
        if self.span_id is None and not self.quotation.strip():
            raise ValueError("Support needs an original span ID or literal quotation")
        return self


class SourceRequirement(StrictModel):
    supersedes_requirement_ids: list[str] = Field(default_factory=list)
    requirement_id: str = Field(min_length=1)
    need_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    application: str = Field(min_length=1)
    supports: list[SpanSupport] = Field(min_length=1)
    missing_user_facts: list[str] = Field(default_factory=list)


class MaterialDependencyRequest(StrictModel):
    need_ids: list[str] = Field(min_length=1)
    origin_citation: int = Field(gt=0, strict=True)
    instrument_name: str = Field(min_length=1)
    instrument_number: str | None = None
    article: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ResearchStep(StrictModel):
    actions: list[SourceAction]
    ready_to_answer: bool
    remaining_gaps: list[str]
    related_citations: list[int] = Field(default_factory=list)


class IssueResearchStep(ResearchStep):
    reconsider_citations: list[Annotated[int, Field(gt=0, strict=True)]] = Field(
        default_factory=list
    )
    gap_resolutions: list[GapResolution] = Field(default_factory=list)
    requirements: list[SourceRequirement] = Field(default_factory=list)
    material_dependencies: list[MaterialDependencyRequest] = Field(default_factory=list)
    issue_gaps: dict[str, list[str]] = Field(default_factory=dict)


class AnswerSection(StrictModel):
    section_id: str = Field(min_length=1)
    need_ids: list[str] = Field(min_length=1)
    text: str = ""
    claim_ids: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


class DraftClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    need_ids: list[str] = Field(min_length=1)
    answer_excerpt: str = Field(min_length=1)
    supports: list[SpanSupport] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)


def render_claim_sections(
    sections: list[AnswerSection], claims: list[DraftClaim]
) -> list[AnswerSection]:
    """Render addressed claims exactly once while retaining literal-text sections."""
    section_ids = [section.section_id for section in sections]
    claim_ids = [claim.claim_id for claim in claims]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Answer sections must have unique identities")
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Claims must have unique identities")
    by_id = {claim.claim_id: claim for claim in claims}
    rendered: list[AnswerSection] = []
    for section in sections:
        if not section.claim_ids:
            if not section.text.strip():
                raise ValueError("Literal section text cannot be empty")
            rendered.append(section.model_copy(deep=True))
            continue
        if len(section.claim_ids) != len(set(section.claim_ids)):
            raise ValueError("A section cannot reference a claim more than once")
        selected: list[DraftClaim] = []
        for identity in section.claim_ids:
            claim = by_id.get(identity)
            if claim is None:
                raise ValueError("Section refers to an unknown claim")
            if claim.section_id != section.section_id:
                raise ValueError("Section cannot render another section's claim")
            if set(claim.need_ids) - set(section.need_ids):
                raise ValueError("Rendered claim refers to another section's issue")
            selected.append(claim)
        own_claims = {
            claim.claim_id for claim in claims if claim.section_id == section.section_id
        }
        if set(section.claim_ids) != own_claims:
            raise ValueError("Rendered section must include every claim assigned to it")
        body = "\n\n".join(claim.answer_excerpt for claim in selected)
        if section.text == body or section.text.endswith("\n\n" + body):
            text = section.text
        else:
            text = "\n\n".join(part for part in (section.text, body) if part)
        rendered.append(section.model_copy(deep=True, update={"text": text}))
    return rendered


class DraftAnswer(StrictModel):
    answer: str = Field(min_length=1)
    unresolved_need_ids: list[str]


class StructuredDraftAnswer(DraftAnswer):
    gap_resolutions: list[GapResolution] = Field(default_factory=list)
    answer: str = ""
    unresolved_need_ids: list[str]
    sections: list[AnswerSection]
    claims: list[DraftClaim]
    requirements: list[SourceRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def section_identity(self) -> StructuredDraftAnswer:
        self.sections = render_claim_sections(self.sections, self.claims)
        joined = "\n\n".join(section.text for section in self.sections)
        if self.sections:
            if self.answer and self.answer != joined:
                raise ValueError(
                    "Answer must exactly join its sections with two newlines"
                )
            self.answer = joined
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        return self


class DraftPatch(StrictModel):
    gap_resolutions: list[GapResolution] = Field(default_factory=list)
    sections: list[AnswerSection]
    claims: list[DraftClaim]
    unresolved_need_ids: list[str]
    requirements: list[SourceRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def render_sections(self) -> DraftPatch:
        self.sections = render_claim_sections(self.sections, self.claims)
        return self


class ReviewCheck(StrictModel):
    check_id: str = Field(min_length=1)
    need_ids: list[str]
    section_ids: list[str]
    status: Literal["addressed", "not_applicable", "gap", "incorrect", "uncertain"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    finding: str = Field(default="", max_length=600)


class SemanticReview(StrictModel):
    checks: list[ReviewCheck]
    repair_actions: list[SourceAction] = Field(default_factory=list)
    failure: str | None = None


class ConditionReview(StrictModel):
    condition_index: int = Field(ge=0)
    status: Literal["preserved", "unresolved", "missing", "incorrect"]
    answer_excerpt: str
    support_citations: list[int]


class NeedReview(StrictModel):
    need_id: str
    status: Literal["supported", "conditional", "unresolved", "incorrect"]
    supports: list[PassageSupport]
    conditions_preserved: bool
    condition_reviews: list[ConditionReview]
    explanation: str
    gap_disclosure: str | None = None


class DependencyOrigin(StrictModel):
    citation: int = Field(gt=0, strict=True)
    source_id: str
    chunk_id: str | None
    text_hash: str


class AuthorityDependency(StrictModel):
    edge_id: str
    need_ids: list[str]
    instrument_name: str
    instrument_number: str | None = None
    article: str
    qualifier: str | None = None
    origins: list[DependencyOrigin]
    governing_citations: list[int] = Field(default_factory=list)
    candidate_citations: list[int] = Field(default_factory=list)
    candidate_source_ids: list[str] = Field(default_factory=list)
    judicial_source_ids: list[str] = Field(default_factory=list)
    incomplete_source_ids: list[str] = Field(default_factory=list)
    discovery_gaps: list[str] = Field(default_factory=list)
    discovery_limits: list[str] = Field(default_factory=list)


class DependencyWitness(PassageSupport):
    citation: int = Field(gt=0, strict=True)
    role: Literal["governing", "operative", "scope", "date", "nonmaterial"]


class DependencyAssessment(StrictModel):
    edge_id: str
    need_ids: list[str]
    status: Literal["examined_applicable", "examined_nonmaterial", "unresolved"]
    witnesses: list[DependencyWitness]
    explanation: str = Field(min_length=1)
    scope_and_date: str = Field(min_length=1)
    temporal_status: Literal["established", "conditional", "unresolved"]
    conditional_excerpt: str | None = None
    gap_disclosure: str | None = None


class AnswerReview(StrictModel):
    request_coverage_complete: bool
    material_claims_supported: bool
    counter_authority_checked: bool
    selection_uncertainty_resolved: bool = False
    needs: list[NeedReview]
    defects: list[str]
    repair_actions: list[SourceAction]
    dependency_assessments: list[DependencyAssessment] | None = None


class WorkflowResult(StrictModel):
    answer: str | None
    status: Literal["verified", "partial", "cancelled", "unavailable"]
    gaps: list[str]
    plan: ResearchPlan | None = None
    review: AnswerReview | None = None


class CompositeWorkflowResult(WorkflowResult):
    plan: IssueResearchPlan | None = None
    semantic_review: SemanticReview | None = None
    source_requirements: list[SourceRequirement] = Field(default_factory=list)
