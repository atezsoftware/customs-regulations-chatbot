"""Compile one integrated draft without asking the model to recopy its claims."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from onyx.legal_review.models import AnswerClaim, DraftAnswer, StrictModel
from onyx.legal_review.passages import PassageReference


class GeneratedClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    supports: list[PassageReference] = Field(min_length=1)

    @field_validator("claim_id")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim identity must be nonempty")
        return value

    @field_validator("issue_ids")
    @classmethod
    def nonempty_issue_identities(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Claim issue identities must be nonempty")
        return values


class GeneratedBlock(StrictModel):
    block_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    # A heading can have no material legal assertion; the field remains required.
    claims: list[GeneratedClaim]

    @field_validator("block_id", "text")
    @classmethod
    def nonempty_block(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Draft block identity and text must be nonempty")
        return value


class GeneratedDraft(StrictModel):
    blocks: list[GeneratedBlock] = Field(min_length=1)
    unresolved_issue_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_identities(self) -> GeneratedDraft:
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Draft block identities must be unique")
        claim_ids = [claim.claim_id for block in self.blocks for claim in block.claims]
        if not claim_ids:
            raise ValueError("Draft must contain at least one supported claim")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Draft claim identities must be unique across blocks")
        return self


def compile_draft(generated: GeneratedDraft) -> DraftAnswer:
    """Preserve ordered prose and bind every claim to its host-owned literal block."""
    # Revalidate mutable model instances before compiling the publication inventory.
    checked = GeneratedDraft.model_validate(generated.model_dump(mode="python"))
    return DraftAnswer(
        answer="\n\n".join(block.text for block in checked.blocks),
        claims=[
            AnswerClaim(
                claim_id=claim.claim_id,
                issue_ids=list(claim.issue_ids),
                answer_excerpt=block.text,
                supports=list(claim.supports),
            )
            for block in checked.blocks
            for claim in block.claims
        ],
        unresolved_issue_ids=list(checked.unresolved_issue_ids),
    )
