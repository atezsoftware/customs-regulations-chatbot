"""Resolve only uniquely located whitespace variants for new intake quotes."""

from hashlib import sha256

from onyx.regulatory.labeling.provider import (
    LabelingOutcome,
    TaxonomyDefinition,
    validate_labeling_response,
)


def _whitespace_view(text: str) -> tuple[str, list[tuple[int, int]]]:
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character.isspace():
            if characters and characters[-1] == " ":
                offsets[-1] = (offsets[-1][0], index + 1)
                continue
            character = " "
        characters.append(character)
        offsets.append((index, index + 1))
    return "".join(characters), offsets


def resolve_intake_labeling_response(
    response_text: str, *, text: str, taxonomy: TaxonomyDefinition
) -> tuple[LabelingOutcome, list[dict[str, object]]]:
    """Return strict assignments and audits; never rewrite source or raw output."""
    try:
        return validate_labeling_response(
            response_text, text=text, taxonomy=taxonomy
        ), []
    except ValueError as error:
        if (
            str(error)
            != "Label evidence must be an exact quote from the canonical target"
        ):
            raise
    outcome = LabelingOutcome.model_validate_json(response_text)
    source, offsets = _whitespace_view(text)
    assignments = []
    audits: list[dict[str, object]] = []
    for assignment in outcome.labels:
        quote = assignment.evidence_quote
        if quote in text:
            assignments.append(assignment)
            continue
        normalized, _quote_offsets = _whitespace_view(quote)
        position = source.find(normalized) if normalized.strip() else -1
        if position < 0 or source.find(normalized, position + 1) >= 0:
            raise ValueError(
                "Intake label quote has no unique whitespace-only source match"
            )
        start = offsets[position][0]
        end = offsets[position + len(normalized) - 1][1]
        resolved = text[start:end]
        if not 1 <= len(resolved) <= 1024:
            raise ValueError(
                "Resolved intake label quote exceeds its source span limit"
            )
        assignments.append(assignment.model_copy(update={"evidence_quote": resolved}))
        audits.append(
            {
                "label_id": assignment.label_id,
                "version": 1,
                "method": "unicode_whitespace_unique",
                "original_quote_sha256": sha256(quote.encode()).hexdigest(),
                "resolved_quote_sha256": sha256(resolved.encode()).hexdigest(),
                "source_text_sha256": sha256(text.encode()).hexdigest(),
                "source_start_char": start,
                "source_end_char": end,
            }
        )
    canonical = outcome.model_copy(update={"labels": assignments})
    return validate_labeling_response(
        canonical.model_dump_json(), text=text, taxonomy=taxonomy
    ), audits
