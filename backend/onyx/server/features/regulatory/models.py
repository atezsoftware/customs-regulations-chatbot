import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from onyx.configs.app_configs import MAX_AMENDMENT_SOURCE_TEXT_CHARS
from onyx.db.models import RegulatoryChunk


class RegulatoryChunkSnapshot(BaseModel):
    id: str
    user_file_id: str
    text: str
    position: int
    chunk_type: str | None
    heading_path: list[str]
    chunk_metadata: dict[str, Any]
    validity_start_date: datetime.date | None
    validity_end_date: datetime.date | None
    status: str
    source: str
    supersedes_chunk_id: str | None
    superseded_by_chunk_id: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_model(cls, chunk: RegulatoryChunk) -> "RegulatoryChunkSnapshot":
        return cls(
            id=chunk.id,
            user_file_id=str(chunk.user_file_id),
            text=chunk.text,
            position=chunk.position,
            chunk_type=chunk.chunk_type,
            heading_path=list(chunk.heading_path),
            chunk_metadata=dict(chunk.chunk_metadata),
            validity_start_date=chunk.validity_start_date,
            validity_end_date=chunk.validity_end_date,
            status=chunk.status,
            source=chunk.source,
            supersedes_chunk_id=chunk.supersedes_chunk_id,
            superseded_by_chunk_id=chunk.superseded_by_chunk_id,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
        )


class RegulatoryChunkUpdateRequest(BaseModel):
    """Partial chunk edit. Omitted fields stay unchanged; validity dates may
    be explicitly nulled to clear them."""

    text: str | None = None
    heading_path: list[str] | None = None
    chunk_metadata: dict[str, Any] | None = None
    # Pydantic can't distinguish omitted from null with plain `| None`, so the
    # date fields ride alongside explicit "clear" flags.
    validity_start_date: datetime.date | None = None
    clear_validity_start_date: bool = False
    validity_end_date: datetime.date | None = None
    clear_validity_end_date: bool = False


class RegulatoryFileValidityUpdateRequest(BaseModel):
    """Explicit source-snapshot window applied across unversioned chunks."""

    validity_start_date: datetime.date | None = None
    clear_validity_start_date: bool = False
    validity_end_date: datetime.date | None = None
    clear_validity_end_date: bool = False


class RegulatoryFileValidityUpdateResponse(BaseModel):
    updated_chunk_count: int
    skipped_versioned_chunk_count: int


class UserFileRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=500)


# =============================================================================
# Amendment (update) mechanism
# =============================================================================


class AnalyzeAmendmentRequest(BaseModel):
    document_set_id: int
    raw_text: str = Field(min_length=1, max_length=MAX_AMENDMENT_SOURCE_TEXT_CHARS)


class AmendmentSourceUrlRequest(BaseModel):
    url: str = Field(min_length=1)


class AmendmentSourceExtractionSnapshot(BaseModel):
    text: str
    source_type: Literal["html", "pdf"]
    display_name: str


class AmendmentProposalSnapshot(BaseModel):
    id: int
    batch_id: int
    instruction_index: int
    instruction_text: str
    instruction_indices: list[int]
    instruction_texts: list[str]
    old_chunk_id: str | None
    old_chunk_snapshot: dict[str, Any]
    new_chunk_draft: dict[str, Any]
    match_confidence: float | None
    match_rationale: str | None
    date_rationale: str | None
    status: str
    applied_new_chunk_id: str | None
    decided_by: str | None
    decided_at: datetime.datetime | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    duplicate_target: bool = False

    @classmethod
    def from_model(
        cls, proposal: Any, *, duplicate_target: bool = False
    ) -> "AmendmentProposalSnapshot":
        instruction_indices = list(
            getattr(proposal, "instruction_indices", None)
            or [proposal.instruction_index]
        )
        instruction_texts = list(
            getattr(proposal, "instruction_texts", None) or [proposal.instruction_text]
        )
        return cls(
            id=proposal.id,
            batch_id=proposal.batch_id,
            instruction_index=proposal.instruction_index,
            instruction_text=proposal.instruction_text,
            instruction_indices=instruction_indices,
            instruction_texts=instruction_texts,
            old_chunk_id=proposal.old_chunk_id,
            old_chunk_snapshot=dict(proposal.old_chunk_snapshot),
            new_chunk_draft=dict(proposal.new_chunk_draft),
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
            status=proposal.status,
            applied_new_chunk_id=proposal.applied_new_chunk_id,
            decided_by=str(proposal.decided_by) if proposal.decided_by else None,
            decided_at=proposal.decided_at,
            created_at=proposal.created_at,
            updated_at=proposal.updated_at,
            duplicate_target=duplicate_target,
        )


class AmendmentBatchSnapshot(BaseModel):
    id: int
    document_set_id: int
    raw_text: str
    reference_date: datetime.date | None
    status: str
    stage: str = "queued"
    instruction_count: int = 0
    processed_instruction_count: int = 0
    error_message: str | None
    created_by: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None = None
    heartbeat_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None

    @classmethod
    def from_model(cls, batch: Any) -> "AmendmentBatchSnapshot":
        return cls(
            id=batch.id,
            document_set_id=batch.document_set_id,
            raw_text=batch.raw_text,
            reference_date=batch.reference_date,
            status=batch.status,
            stage=getattr(batch, "stage", "queued"),
            instruction_count=getattr(batch, "instruction_count", 0),
            processed_instruction_count=getattr(
                batch, "processed_instruction_count", 0
            ),
            error_message=batch.error_message,
            created_by=str(batch.created_by) if batch.created_by else None,
            created_at=batch.created_at,
            updated_at=batch.updated_at,
            started_at=getattr(batch, "started_at", None),
            heartbeat_at=getattr(batch, "heartbeat_at", None),
            completed_at=getattr(batch, "completed_at", None),
        )


class AnalyzeAmendmentResponse(BaseModel):
    batch: AmendmentBatchSnapshot
    proposals: list[AmendmentProposalSnapshot]
    unmatched_instructions: list[str]
