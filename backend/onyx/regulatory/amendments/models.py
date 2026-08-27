"""Pydantic models for the amendment (update) analysis pipeline.

Every LLM output here is schema-validated via
`onyx.regulatory.structured_llm.generate_structured` — no freeform JSON
parsing.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _require_iso_date_or_none(value: str | None) -> str | None:
    """Reject anything that isn't a real `YYYY-MM-DD` date or null.

    Structured output only guarantees the *type* (`str`), not the *format* —
    without this, a model that drifts from its instructions could write a
    natural-language phrase (e.g. "yayımı tarihinden itibaren") straight into
    a date field, which would either fail at insert (`regulatory_chunk`'s
    validity columns are real DATE columns) or silently misbehave upstream.
    Fail validation immediately instead.
    """
    if value is None:
        return None
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Expected a YYYY-MM-DD date or null, got {value!r} — natural-language "
            "date phrases are not valid here."
        ) from exc
    return value


class AmendmentInstruction(BaseModel):
    """One atomic, single-article change extracted from a pasted amendment text."""

    instruction_text: str = Field(
        description="The exact text of this single amendment instruction"
    )
    article_reference: str | None = Field(
        default=None,
        description="The article/provision this instruction refers to (e.g. 'Madde 3'), if stated",
    )
    target_source: str | None = Field(
        default=None,
        description=(
            "The full name/identity of the existing regulation being changed. "
            "Resolve references such as 'Aynı Tebliğ' to the concrete source "
            "named earlier in the pasted amendment text."
        ),
    )
    raw_date_phrase: str | None = Field(
        default=None,
        description="The natural-language effective-date phrase for this instruction, if any",
    )
    search_query: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "One focused question in the likely language of the indexed source "
            "that retrieves the existing rule affected by this instruction"
        ),
    )
    recovery_query: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "One alternate, concise lexical query for a single bounded recovery "
            "attempt if the first retrieval cannot be confirmed"
        ),
    )


class SegmentationResult(BaseModel):
    """Output of splitting a pasted amendment text into atomic instructions."""

    reference_date: str | None = Field(
        default=None,
        description="The amendment text's own publication/reference date (YYYY-MM-DD), if stated",
    )
    instructions: list[AmendmentInstruction]

    _validate_reference_date = field_validator("reference_date")(
        _require_iso_date_or_none
    )


class MatchResult(BaseModel):
    """Which (if any) hybrid-search candidate an instruction amends."""

    old_chunk_id: str | None = Field(
        description=(
            "id of the matched candidate chunk to amend, or null if this "
            "instruction adds a new provision with no existing match"
        )
    )
    confidence: float = Field(description="Confidence in this match, 0.0-1.0")
    rationale: str = Field(
        description="Brief explanation of why this candidate was (or wasn't) matched"
    )


class ChunkFieldsDraft(BaseModel):
    """The amended chunk's content. The LLM has authority over every field
    here, including any key inside `metadata_changes`.

    `metadata_changes` is a *patch*, not the full metadata dict — the
    pipeline merges it onto the old chunk's metadata (`{**old, **changes}`)
    rather than trusting the LLM to faithfully reproduce every unrelated
    field verbatim.
    """

    text: str = Field(description="Full amended chunk text")
    chunk_type: str | None = Field(
        default=None, description="Chunk type, e.g. 'paragraph', 'article', 'table'"
    )
    heading_path: list[str] | None = Field(
        default=None,
        description=(
            "Full heading path for the new chunk. Required when there is no "
            "old chunk (brand new article) — base it on sibling_reference's "
            "heading_path. Leave null to carry over the old chunk's path "
            "unchanged."
        ),
    )
    metadata_changes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "ONLY the metadata fields this amendment actually changes (e.g. "
            "article_no if the article was renumbered). Do NOT repeat "
            "fields that stay the same — they are carried over automatically "
            "from the old chunk if omitted here. Leave empty ({}) if nothing "
            "in metadata changes."
        ),
    )


class DateResolution(BaseModel):
    """Effective dates resolved from natural-language phrasing in the
    amendment text, anchored to the amendment's reference date."""

    effective_start_date: str | None = Field(
        default=None,
        description="YYYY-MM-DD the amended text starts applying, or null if not stated",
    )
    effective_end_date: str | None = Field(
        default=None,
        description=(
            "YYYY-MM-DD the amended text stops applying — only set this if "
            "the amendment itself is explicitly temporary, otherwise null"
        ),
    )
    rationale: str = Field(
        description="Brief explanation of how these dates were derived from the text"
    )

    _validate_dates = field_validator("effective_start_date", "effective_end_date")(
        _require_iso_date_or_none
    )


class DraftResult(BaseModel):
    new_chunk: ChunkFieldsDraft
    dates: DateResolution


class ProposalDraft(BaseModel):
    """One fully-assembled amendment proposal, ready to persist for review."""

    instruction_index: int
    instruction_text: str
    instruction_indices: list[int]
    instruction_texts: list[str]
    old_chunk_id: str | None
    old_chunk_snapshot: dict[str, Any]
    new_chunk_draft: dict[str, Any]
    match_confidence: float | None = None
    match_rationale: str | None = None
    date_rationale: str | None = None

    @model_validator(mode="after")
    def _validate_instruction_group(self) -> "ProposalDraft":
        if not self.instruction_indices:
            raise ValueError("instruction_indices must be non-empty")
        if any(
            current <= previous
            for previous, current in zip(
                self.instruction_indices, self.instruction_indices[1:]
            )
        ):
            raise ValueError(
                "instruction_indices must be strictly increasing and duplicate-free"
            )
        if self.instruction_indices[0] != self.instruction_index:
            raise ValueError(
                "instruction_index must be the first instruction_indices value"
            )
        if len(self.instruction_texts) != len(self.instruction_indices):
            raise ValueError(
                "instruction_texts must have one entry per instruction index"
            )
        if self.instruction_texts[0] != self.instruction_text:
            raise ValueError(
                "instruction_text must be the first instruction_texts value"
            )
        return self


class AnalysisResult(BaseModel):
    """Full output of analyzing one pasted amendment text."""

    reference_date: str | None
    proposals: list[ProposalDraft]
    unmatched_instructions: list[AmendmentInstruction]
