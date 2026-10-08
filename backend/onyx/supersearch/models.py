from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from onyx.legal_composite.models import (
    AnswerReview,
    PassageSupport,
    SourceAction,
    StrictModel,
)


class NeedFocusAssessment(StrictModel):
    subject_id: str
    need_id: str
    status: Literal["material", "incidental", "pending"]
    explanation: str = Field(min_length=1)
    witnesses: list[PassageSupport]


class NeedFocusDecision(StrictModel):
    assessments: list[NeedFocusAssessment]


class FocusReviewAssessment(StrictModel):
    subject_id: str
    need_id: str
    status: Literal["nonmaterial", "reopen", "unresolved"]
    explanation: str = Field(min_length=1)
    witnesses: list[PassageSupport]


class FocusAnswerReview(AnswerReview):
    focus_reviews: list[FocusReviewAssessment]


class WriterDecision(StrictModel):
    answer: str | None
    unresolved_need_ids: list[str]
    actions: list[SourceAction]

    @model_validator(mode="after")
    def answer_or_research(self) -> WriterDecision:
        if bool(self.answer and self.answer.strip()) == bool(self.actions):
            raise ValueError("Writer must either request originals or return an answer")
        return self


class PassagePatch(StrictModel):
    old_text: str = Field(min_length=1)
    new_text: str


class AnswerRepair(StrictModel):
    patches: list[PassagePatch]
    unresolved_need_ids: list[str]
