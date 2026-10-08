from __future__ import annotations

from pydantic import Field, model_validator

from onyx.legal_composite.models import SourceAction, StrictModel


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
