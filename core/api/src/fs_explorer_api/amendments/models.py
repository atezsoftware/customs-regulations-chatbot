"""Pydantic models for the amendment (Resmi Gazete update) analysis pipeline.

Every LLM output here is a typed, schema-validated structured response
(`LLMClient.generate_structured`) — no freeform JSON parsing, following the
`Action`-union pattern in `fs_explorer_api.models`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# LLM-facing models (segmentation -> matching -> drafting)
# =============================================================================


class AmendmentInstruction(BaseModel):
    """One atomic, single-article change extracted from a pasted gazette text."""

    instruction_text: str = Field(
        description="The exact text of this single amendment instruction"
    )
    article_reference: str | None = Field(
        default=None,
        description="The article/provision this instruction refers to (e.g. 'Madde 3'), if stated",
    )
    raw_date_phrase: str | None = Field(
        default=None,
        description="The natural-language effective-date phrase for this instruction, if any",
    )


class SegmentationResult(BaseModel):
    """Output of splitting a pasted gazette text into atomic instructions."""

    reference_date: str | None = Field(
        default=None,
        description="The gazette text's own publication/reference date (YYYY-MM-DD), if stated",
    )
    instructions: list[AmendmentInstruction]


class MatchResult(BaseModel):
    """Which (if any) hybrid-search candidate an instruction amends."""

    old_chunk_id: str | None = Field(
        description=(
            "chunk_id of the matched candidate to amend, or null if this "
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
    field (article_no, heading_path, document_date, ...) verbatim. A
    structured-output model asked to copy a whole dict has no guarantee it
    won't drop or alter fields it wasn't actually asked to change; asking
    for only the diff removes that failure mode by construction.
    """

    text: str = Field(description="Full amended chunk text")
    chunk_type: str | None = Field(
        default=None, description="Chunk type, e.g. 'paragraph', 'article', 'table'"
    )
    metadata_changes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "ONLY the metadata fields this amendment actually changes (e.g. "
            "article_no if the article was renumbered). Do NOT repeat "
            "fields that stay the same — heading_path, document_date, etc. "
            "are carried over automatically from the old chunk if omitted "
            "here. Leave empty ({}) if nothing in metadata changes."
        ),
    )


class DateResolution(BaseModel):
    """Effective dates resolved from natural-language phrasing in the
    amendment text, anchored to the gazette's reference date."""

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


class DraftResult(BaseModel):
    new_chunk: ChunkFieldsDraft
    dates: DateResolution


# =============================================================================
# Pipeline-assembled models
# =============================================================================


class ProposalDraft(BaseModel):
    """One fully-assembled amendment proposal, ready to persist for review."""

    instruction_index: int
    instruction_text: str
    old_chunk_id: str | None
    old_chunk_snapshot: dict[str, Any]
    new_chunk_draft: dict[str, Any]
    match_confidence: float | None = None
    match_rationale: str | None = None
    date_rationale: str | None = None
    duplicate_target: bool = False

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "instruction_index": self.instruction_index,
            "instruction_text": self.instruction_text,
            "old_chunk_id": self.old_chunk_id,
            "old_chunk_snapshot": self.old_chunk_snapshot,
            "new_chunk_draft": self.new_chunk_draft,
            "match_confidence": self.match_confidence,
            "match_rationale": self.match_rationale,
            "date_rationale": self.date_rationale,
        }


class AnalysisResult(BaseModel):
    """Full output of analyzing one pasted amendment text."""

    reference_date: str | None
    proposals: list[ProposalDraft]
    unmatched_instructions: list[AmendmentInstruction]


# =============================================================================
# REST request/response models
# =============================================================================


class AnalyzeAmendmentRequest(BaseModel):
    corpus_folder: str
    raw_text: str
    database_url: str | None = None


class ProposalRecord(BaseModel):
    """A persisted amendment proposal, as returned to callers.

    `duplicate_target` is not a stored column — it's recomputed whenever
    proposals are served together (see `flag_duplicate_targets_in_records`),
    since two proposals racing to supersede the same old chunk is only ever
    meaningful relative to the other pending proposals at read time.
    """

    id: str
    batch_id: str
    instruction_index: int
    instruction_text: str
    old_chunk_id: str | None
    old_chunk_snapshot: dict[str, Any]
    new_chunk_draft: dict[str, Any]
    match_confidence: float | None
    match_rationale: str | None
    date_rationale: str | None
    status: str
    applied_new_chunk_id: str | None
    decided_by: str | None
    decided_at: str | None
    created_at: str
    updated_at: str
    duplicate_target: bool = False


class AnalyzeAmendmentResponse(BaseModel):
    batch_id: str
    reference_date: str | None
    proposals: list[ProposalRecord]
    unmatched_instructions: list[str]


class ApproveProposalsRequest(BaseModel):
    proposal_ids: list[str]
    database_url: str | None = None
    decided_by: str | None = None


class ProposalFailure(BaseModel):
    proposal_id: str
    reason: str


class ApproveProposalsResponse(BaseModel):
    applied: list[ProposalRecord]
    failed: list[ProposalFailure]


class DecideProposalRequest(BaseModel):
    database_url: str | None = None
    decided_by: str | None = None
