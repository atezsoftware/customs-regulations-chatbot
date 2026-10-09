"""Translate only exact rendered test fixtures without losing their literal prose."""

from onyx.legal_composite.draft_composition import (
    CompositionSection,
    DraftComposition,
)
from onyx.legal_composite.models import StructuredDraftAnswer


def composition_for(draft: StructuredDraftAnswer) -> DraftComposition:
    if draft.answer != "\n\n".join(section.text for section in draft.sections):
        raise ValueError("Fixture answer must exactly join its sections")
    by_id = {claim.claim_id: claim for claim in draft.claims}
    if len(by_id) != len(draft.claims):
        raise ValueError("Fixture claims must have unique identities")
    sections: list[CompositionSection] = []
    ordered: list[str] = []
    for section in draft.sections:
        own = [
            claim.claim_id
            for claim in draft.claims
            if claim.section_id == section.section_id
        ]
        identities = section.claim_ids or own
        if len(identities) != len(set(identities)) or set(identities) != set(own):
            raise ValueError(
                "Fixture section must name all its own claims exactly once"
            )
        body = "\n\n".join(by_id[identity].answer_excerpt for identity in identities)
        if not body:
            heading = section.text
        elif section.text == body:
            heading = ""
        elif section.text.endswith("\n\n" + body):
            heading = section.text[: -len("\n\n" + body)]
        else:
            raise ValueError("Fixture legal prose must be its exact ordered claim body")
        sections.append(
            CompositionSection(
                section_id=section.section_id,
                need_ids=list(section.need_ids),
                heading=heading,
            )
        )
        ordered.extend(identities)
    if set(ordered) != set(by_id):
        raise ValueError("Fixture cannot omit an orphan claim")
    return DraftComposition(
        sections=sections,
        claims=[by_id[identity].model_copy(deep=True) for identity in ordered],
        unresolved_need_ids=list(draft.unresolved_need_ids),
        requirements=[row.model_copy(deep=True) for row in draft.requirements],
        gap_resolutions=[row.model_copy(deep=True) for row in draft.gap_resolutions],
    )
