"""Compile one integrated draft without asking the model to recopy its claims."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_review.models import AnswerClaim, DraftAnswer, StrictModel
from onyx.legal_review.passages import PassageReference, source_passage_references


class ClaimApplication(StrictModel):
    source_conditions: str = Field(min_length=1)
    fact_application: str = Field(min_length=1)
    remaining_uncertainty: str | None

    @field_validator("source_conditions", "fact_application", "remaining_uncertainty")
    @classmethod
    def nonempty_explanation(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Claim application explanations must be nonempty")
        return value


class GeneratedClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    supports: list[PassageReference] = Field(min_length=1)
    application: ClaimApplication

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
    # Select and apply the original before generating the publication text.
    claims: list[GeneratedClaim]
    text: str = Field(min_length=1)

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
        if not claim_ids and not self.unresolved_issue_ids:
            raise ValueError("A claimless draft must identify unresolved issues")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Draft claim identities must be unique across blocks")
        return self


def compile_draft(
    generated: GeneratedDraft, *, ledger: EvidenceLedger | None = None
) -> DraftAnswer:
    """Preserve ordered prose and bind every claim to its host-owned literal block."""
    # Revalidate mutable model instances before compiling the publication inventory.
    checked = GeneratedDraft.model_validate(generated.model_dump(mode="python"))
    claims = [claim for block in checked.blocks for claim in block.claims]
    claim_ids = {claim.claim_id for claim in claims}
    source_issues: dict[int, set[str]] = {}
    for claim in claims:
        for support in claim.supports:
            source_issues.setdefault(support.citation, set()).update(claim.issue_ids)
    rendered: list[tuple[GeneratedBlock, str]] = []
    inline_claims: list[AnswerClaim] = []
    for block in checked.blocks:
        cited = set(extract_citation_numbers(block.text))
        selected = dict.fromkeys(
            support.citation for claim in block.claims for support in claim.supports
        )
        missing = [citation for citation in selected if citation not in cited]
        markers = " ".join(f"[{citation}]" for citation in missing)
        # A suffix on a closing fence or table row would change Markdown structure.
        text = block.text + ("\n\n" + markers if markers else "")
        rendered.append((block, text))
        inline_only = sorted(cited - selected.keys())
        if ledger is not None and inline_only:
            # Both structured selectors and inline markers are model source selections.
            # Bind the latter to full originals; entailment still requires review.
            supports = [
                support
                for citation in inline_only
                for support in source_passage_references(citation, ledger)
            ]
            issues = {
                issue_id for claim in block.claims for issue_id in claim.issue_ids
            } or {
                issue_id
                for citation in inline_only
                for issue_id in source_issues.get(citation, set())
            }
            if not issues:
                raise ValueError(
                    "Inline source selections require a known issue binding"
                )
            identity = f"inline:{block.block_id}"
            while identity in claim_ids:
                identity += ":inline"
            claim_ids.add(identity)
            inline_claims.append(
                AnswerClaim(
                    claim_id=identity,
                    issue_ids=sorted(issues),
                    answer_excerpt=text,
                    supports=supports,
                )
            )
    return DraftAnswer(
        answer="\n\n".join(text for _, text in rendered),
        claims=[
            AnswerClaim(
                claim_id=claim.claim_id,
                issue_ids=list(claim.issue_ids),
                answer_excerpt=text,
                supports=list(claim.supports),
            )
            for block, text in rendered
            for claim in block.claims
        ]
        + inline_claims,
        unresolved_issue_ids=list(checked.unresolved_issue_ids),
    )
