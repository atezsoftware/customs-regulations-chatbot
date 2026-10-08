from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: float = Field(default=120, gt=0)
    finalization_reserve_seconds: float = Field(default=40, gt=0)
    max_model_calls: int = Field(default=8, ge=4)
    max_input_tokens: int = Field(default=120_000, gt=0)
    max_output_tokens: int = Field(default=24_000, gt=0)
    max_cost_usd: float = Field(default=0.10, gt=0)
    max_call_seconds: float = Field(default=45, gt=0)
    max_context_tokens: int = Field(default=32_000, gt=0)
    max_tools: int = Field(default=24, gt=0)
    max_parallel_tools: int = Field(default=4, ge=1, le=4)
    max_search_calls: int = Field(default=8, gt=0)
    max_research_rounds: int = Field(default=4, ge=1)
    max_reviews: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def reserve_fits(self) -> WorkflowPolicy:
        if self.finalization_reserve_seconds >= self.timeout_seconds:
            raise ValueError("Finalization reserve must fit within the deadline")
        return self


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchNeed(StrictModel):
    need_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    governing_source: str
    conditions_to_check: list[str]


class SourceAction(StrictModel):
    need_ids: list[str] = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class ResearchPlan(StrictModel):
    language: str = Field(min_length=2, max_length=35)
    requires_sources: bool
    needs: list[ResearchNeed] = Field(min_length=1)
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


class ResearchStep(StrictModel):
    actions: list[SourceAction]
    ready_to_answer: bool
    remaining_gaps: list[str]


class DraftAnswer(StrictModel):
    answer: str = Field(min_length=1)
    unresolved_need_ids: list[str]


class PassageSupport(StrictModel):
    citation: int = Field(gt=0)
    quotation: str = Field(min_length=1)


class NeedReview(StrictModel):
    need_id: str
    status: Literal["supported", "conditional", "unresolved", "incorrect"]
    supports: list[PassageSupport]
    conditions_preserved: bool
    explanation: str
    gap_disclosure: str | None = None


class AnswerReview(StrictModel):
    request_coverage_complete: bool
    material_claims_supported: bool
    counter_authority_checked: bool
    needs: list[NeedReview]
    defects: list[str]
    repair_actions: list[SourceAction]


class WorkflowResult(StrictModel):
    answer: str | None
    status: Literal["verified", "partial", "cancelled", "unavailable"]
    gaps: list[str]
    plan: ResearchPlan | None = None
    review: AnswerReview | None = None
