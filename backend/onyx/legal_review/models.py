from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS


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


class ReviewResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    completed: bool
    scores: dict[str, float] = Field(default_factory=dict)
    flags: list[ReviewCheck] = Field(default_factory=list)
    failure_reason: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    timeout_seconds: float = Field(default=240, gt=0)
    finalization_reserve_seconds: float = Field(default=110, gt=0)
    max_call_seconds: int = Field(default=45, gt=0)
    max_model_calls: int = Field(default=32, ge=12)
    max_tools: int = Field(default=96, gt=0)
    max_searches: int = Field(default=24, gt=0)
    max_parallel_tools: int = Field(default=4, ge=1, le=8)
    max_research_rounds: int = Field(default=1, ge=1, le=3)
    max_input_tokens: int = Field(default=2_000_000, gt=0)
    max_output_tokens: int = Field(default=128_000, gt=0)
    max_context_tokens: int = Field(default=192_000, gt=0)
    max_evidence_bytes: int = Field(default=2_000_000, gt=0)
    max_generation_output_tokens: int = Field(default=16_384, gt=0)

    @model_validator(mode="after")
    def reserve_fits(self) -> WorkflowPolicy:
        if self.finalization_reserve_seconds >= self.timeout_seconds:
            raise ValueError("Finalization reserve must fit within the deadline")
        return self


class Issue(StrictModel):
    issue_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_outcome: str = Field(min_length=1)
    supplied_facts: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def bounded_queries(self) -> Issue:
        if any(
            not query.strip() or len(query) > REGULATORY_MAX_SEARCH_QUERY_CHARS
            for query in self.research_queries
        ):
            raise ValueError("Every issue needs bounded, nonempty research queries")
        return self


class IssuePlan(StrictModel):
    language: str = Field(min_length=2, max_length=35)
    issues: list[Issue] = Field(min_length=1, max_length=24)
    missing_user_facts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_issues(self) -> IssuePlan:
        ids = [issue.issue_id for issue in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("Issue identities must be unique")
        return self


class SourceAction(StrictModel):
    issue_ids: list[str] = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class PassageSupport(StrictModel):
    citation: int = Field(gt=0, strict=True)
    quotation: str = Field(min_length=1)


class Requirement(StrictModel):
    requirement_id: str = Field(min_length=1)
    issue_id: str = Field(min_length=1)
    dimension: LegalDimension
    rule: str = Field(min_length=1)
    application: str = Field(min_length=1)
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
    requirement_ids: list[str] = Field(default_factory=list)


class ReadingDecision(StrictModel):
    additional_issues: list[Issue] = Field(default_factory=list, max_length=8)
    requirements: list[Requirement] = Field(default_factory=list)
    dimensions: list[DimensionAssessment]
    actions: list[SourceAction] = Field(default_factory=list, max_length=16)
    evidence_gaps: list[str] = Field(default_factory=list)


class AnswerClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    answer_excerpt: str = Field(min_length=1)
    supports: list[PassageSupport] = Field(min_length=1)


class DraftAnswer(StrictModel):
    answer: str = Field(min_length=1)
    claims: list[AnswerClaim] = Field(min_length=1)
    unresolved_issue_ids: list[str] = Field(default_factory=list)


class WorkflowResult(StrictModel):
    status: Literal["verified", "partial", "unavailable", "cancelled"]
    answer: str | None = None
    plan: IssuePlan | None = None
    requirements: list[RequirementRecord] = Field(default_factory=list)
    dimensions: list[DimensionAssessment] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    early_review: ReviewResult | None = None
    final_review: ReviewResult | None = None
    repair_used: bool = False
