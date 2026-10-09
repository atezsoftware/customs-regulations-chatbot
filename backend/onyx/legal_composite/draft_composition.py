"""Compose every returned claim through host-owned section membership and order."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, ValidationError

from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    AnswerSection,
    DraftClaim,
    GapResolution,
    SourceRequirement,
    StrictModel,
    StructuredDraftAnswer,
)
from onyx.legal_composite.prompts import (
    ANSWER_CONTENT_HEAD,
    ANSWER_CONTENT_TAIL,
    COMMON,
)

Identity = Annotated[str, Field(min_length=1)]


class CompositionSection(StrictModel):
    section_id: Identity
    need_ids: list[Identity] = Field(min_length=1)
    heading: str = Field(default="", max_length=600)


class DraftComposition(StrictModel):
    sections: list[CompositionSection] = Field(min_length=1)
    claims: list[DraftClaim]
    unresolved_need_ids: list[Identity]
    requirements: list[SourceRequirement] = Field(default_factory=list)
    gap_resolutions: list[GapResolution] = Field(default_factory=list)


DRAFT_COMPOSITION_PROMPT = (
    COMMON
    + ANSWER_CONTENT_HEAD
    + """The sections and claims arrays are mandatory. Return sections containing only
section_id, need_ids and heading. heading is only the heading/nonlegal introduction, at
most 600 characters; it may be empty when that section has claims. Do not return answer,
section text or claim_ids: the host derives every section's ordered claim inventory from
your global claims array and composes the entire answer. A social answer may explicitly
return claims=[] with a nonempty heading containing its useful source-free response.
Write each complete publishable legal passage ONCE in DraftClaim.answer_excerpt, including
its global [n] citations and operative qualifiers. Return claims in the desired order within
each section. Every claim must name an existing section_id and nonempty unique need_ids
that are a subset of that section's need_ids and identify the actual issues it addresses.
Never place legal claim prose only in a heading, duplicate it there or omit an assertion
from claims. Tables, conclusions and conditional branches with legal content are also
complete claim passages. Keep separate requested alternatives distinguishable. The host
preserves your exact passages; original support and full semantic review remain mandatory.
"""
    + ANSWER_CONTENT_TAIL
)


def compose_draft(value: DraftComposition) -> StructuredDraftAnswer:
    """Bind all claim memberships without changing prose, support or either inventory."""
    try:
        value = DraftComposition.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise InvalidSourceAction(
            "Draft composition failed the transport schema"
        ) from error
    sections = {section.section_id: section for section in value.sections}
    claim_ids = [claim.claim_id for claim in value.claims]
    if len(sections) != len(value.sections) or len(claim_ids) != len(set(claim_ids)):
        raise InvalidSourceAction("Draft composition identities must be unique")
    if any(
        len(section.need_ids) != len(set(section.need_ids))
        for section in value.sections
    ):
        raise InvalidSourceAction(
            "Draft composition section issue bindings must be unique"
        )
    for claim in value.claims:
        section = sections.get(claim.section_id)
        if (
            section is None
            or not claim.need_ids
            or len(claim.need_ids) != len(set(claim.need_ids))
            or set(claim.need_ids) - set(section.need_ids)
        ):
            raise InvalidSourceAction(
                "Draft composition claim exceeds its section issue scope"
            )
    try:
        return StructuredDraftAnswer(
            sections=[
                AnswerSection(
                    section_id=section.section_id,
                    need_ids=list(section.need_ids),
                    text=section.heading,
                    claim_ids=[
                        claim.claim_id
                        for claim in value.claims
                        if claim.section_id == section.section_id
                    ],
                )
                for section in value.sections
            ],
            claims=[claim.model_copy(deep=True) for claim in value.claims],
            unresolved_need_ids=list(value.unresolved_need_ids),
            requirements=[row.model_copy(deep=True) for row in value.requirements],
            gap_resolutions=[
                row.model_copy(deep=True) for row in value.gap_resolutions
            ],
        )
    except ValidationError as error:
        raise InvalidSourceAction(
            "Draft composition cannot form a valid answer"
        ) from error
