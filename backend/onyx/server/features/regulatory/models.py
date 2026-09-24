import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    model_validator,
)

from onyx.configs.app_configs import MAX_AMENDMENT_SOURCE_TEXT_CHARS
from onyx.db.models import AnnexChangeSet, RegulatoryChunk
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
    AnnexElementCorrection,
)
from onyx.regulatory.approval_execution_state import (
    ApprovalExecutionState,
    read_execution_state,
)


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


class RegulatoryChunkPage(BaseModel):
    items: list[RegulatoryChunkSnapshot]
    total: int
    offset: int
    limit: int


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
    source_package_id: UUID | None = None
    document_set_id: int
    raw_text: str = Field(min_length=1, max_length=MAX_AMENDMENT_SOURCE_TEXT_CHARS)


class ApproveAmendmentProposalRequest(BaseModel):
    new_chunk_draft: dict[str, Any]
    chunk_changes: list[dict[str, Any]] | None = None


class AmendmentSourceUrlRequest(BaseModel):
    url: str = Field(min_length=1)


class AmendmentSourceExtractionSnapshot(BaseModel):
    text: str
    source_type: Literal["html", "pdf", "docx"]
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
    chunk_changes: list[dict[str, Any]]
    match_confidence: float | None
    match_rationale: str | None
    date_rationale: str | None
    status: str
    applied_new_chunk_id: str | None
    applied_new_chunk_ids: list[str]
    approval_indexing_job_id: str | None
    approval_error: str | None
    approval_execution: ApprovalExecutionState | None = None
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
        raw_error = getattr(proposal, "approval_error", None)
        execution = read_execution_state(raw_error)
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
            chunk_changes=[
                dict(change)
                for change in (getattr(proposal, "chunk_changes", None) or [])
            ],
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
            status=proposal.status,
            applied_new_chunk_id=proposal.applied_new_chunk_id,
            applied_new_chunk_ids=list(
                getattr(proposal, "applied_new_chunk_ids", None) or []
            ),
            approval_indexing_job_id=(
                str(proposal.approval_indexing_job_id)
                if getattr(proposal, "approval_indexing_job_id", None)
                else None
            ),
            approval_error=(execution.message if execution.state == "failed" else None)
            if execution
            else raw_error,
            approval_execution=execution,
            decided_by=str(proposal.decided_by) if proposal.decided_by else None,
            decided_at=proposal.decided_at,
            created_at=proposal.created_at,
            updated_at=proposal.updated_at,
            duplicate_target=duplicate_target,
        )


class AnnexReviewPreparationSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    completed_chunks: int
    total_chunks: int
    error_message: str | None
    result_review_id: UUID | None


class AnnexReviewSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    include_evidence: bool = Field(default=False, exclude=True)
    id: UUID
    logical_group_id: UUID
    review_revision: int
    batch_id: int
    status: Literal[
        "pending",
        "blocked",
        "approving",
        "preparing",
        "publishing",
        "approved",
        "rejected",
        "failed",
    ]
    review_sha256: str
    publication_generation: int
    review_payload: AnnexChangeDraft
    preparation: AnnexReviewPreparationSnapshot | None = None

    @computed_field
    @property
    def dependency_summary(self) -> dict[str, Any]:
        impact = self.review_payload.dependency_impact
        return {
            "unresolved": len(impact.unresolved) if impact else 0,
            "affected": len(impact.affected_ids) if impact else 0,
            "reasons": sorted(
                {reason for reasons in impact.unresolved.values() for reason in reasons}
            )[:5]
            if impact
            else [],
        }

    @field_serializer("review_payload")
    def serialize_review_payload(self, payload: AnnexChangeDraft) -> dict[str, Any]:
        # Prompts and full-file projection receipts stay in the immutable DB review.
        if not self.include_evidence:
            if payload.impact is not None:
                payload = payload.model_copy(
                    update={
                        "impact": payload.impact.model_copy(
                            update={
                                "direct_canonical_changes": [],
                                "contextual_candidates": [],
                                "embedding_changes": [],
                                "context_only": [],
                                "metadata_only": [],
                                "retire_history": [],
                                "unchanged": [],
                                "reasons": {},
                            }
                        )
                    }
                )
            if payload.publication is not None:
                payload = payload.model_copy(
                    update={
                        "publication": payload.publication.model_copy(
                            update={"projection_ids": []}
                        )
                    }
                )
            payload = payload.model_copy(
                update={
                    "items": [],
                    "evidence": [],
                    "old_extraction": None,
                    "new_extraction": None,
                    "raw_new_extraction": None,
                    "comparison": None,
                    "patch_plan": None,
                    "submitted_source_text": None,
                    "baseline": None,
                    "source_graph": [],
                    "new_evidence_remapping": None,
                    "instruction_texts": [],
                }
            )
        return payload.model_dump(
            mode="json",
            exclude={
                "baseline_scope": True,
                "baseline_context": True,
                "impact": {"prepared": True},
                "dependency_impact": True,
            },
        )

    error_message: str | None
    created_at: datetime.datetime


class AnnexReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_review_sha256: str = Field(min_length=64, max_length=64)


class AnnexSelectionRequest(AnnexReviewDecisionRequest):
    item_ids: list[UUID] = Field(min_length=1, max_length=1000)


class AnnexChunkReviewItem(BaseModel):
    id: UUID
    position: int
    operation: str
    old_chunks: list[AnnexCanonicalSnapshot]
    new_chunks: list[AnnexCanonicalSnapshot]
    selection: AnnexReviewSnapshot | None = None
    old_image_evidence_ids: list[UUID] = Field(default_factory=list)
    new_image_evidence_ids: list[UUID] = Field(default_factory=list)


class AnnexChunkReviewPage(BaseModel):
    selection_count: int = 0
    items: list[AnnexChunkReviewItem]
    total: int
    offset: int
    limit: int


class AnnexReviewEditRequest(AnnexReviewDecisionRequest):
    corrections: list[AnnexElementCorrection] | None = None


class AnnexCapabilities(BaseModel):
    enabled: bool
    grouped_review: bool
    immutable_review_revisions: bool
    asynchronous_source_preparation: bool
    publication_requires_verified_index: bool = True


class AnnexSourceTextSnapshot(BaseModel):
    package_id: UUID
    manifest_sha256: str
    original_text: str
    original_text_sha256: str


class AmendmentSourceTextRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_source_text_sha256: str = Field(min_length=64, max_length=64)
    raw_text: str = Field(min_length=1, max_length=MAX_AMENDMENT_SOURCE_TEXT_CHARS)
    source_package_id: UUID | None = None


class AmendmentBatchSnapshot(BaseModel):
    source_package_id: UUID | None = None
    source_text_sha256: str | None = None
    source_parent_batch_id: int | None = None
    superseded_by_batch_id: int | None = None
    annex_group_count: int = 0
    annex_review_revision_count: int = 0
    annex_pending_count: int = 0
    id: int
    document_set_id: int
    raw_text: str
    reference_date: datetime.date | None
    status: str
    stage: str = "queued"
    instruction_count: int = 0
    processed_instruction_count: int = 0
    matched_instruction_count: int = 0
    error_message: str | None
    created_by: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None = None
    heartbeat_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None

    @classmethod
    def from_model(
        cls,
        batch: Any,
        annex_groups: list[AnnexChangeSet] | None = None,
        matched_instruction_count: int = 0,
    ) -> "AmendmentBatchSnapshot":
        return cls(
            source_package_id=getattr(batch, "source_package_id", None),
            source_text_sha256=getattr(batch, "source_text_sha256", None),
            source_parent_batch_id=getattr(batch, "source_parent_batch_id", None),
            superseded_by_batch_id=getattr(batch, "superseded_by_batch_id", None),
            annex_group_count=len(annex_groups or []),
            annex_review_revision_count=sum(
                group.review_revision for group in annex_groups or []
            ),
            annex_pending_count=sum(
                group.status == "pending" for group in annex_groups or []
            ),
            matched_instruction_count=matched_instruction_count,
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
    annex_groups: list[AnnexReviewSnapshot] = Field(default_factory=list)
    batch: AmendmentBatchSnapshot
    proposals: list[AmendmentProposalSnapshot]
    unmatched_instructions: list[str]
    # Ordered record of what the analysis actually did, shown on the Updates
    # screen so a run can be diagnosed without server log access.
    analysis_log: list[dict[str, Any]] = Field(default_factory=list)


class CreateAmendmentSourcePackageRequest(BaseModel):
    document_set_id: int
    idempotency_key: str = Field(min_length=1, max_length=200)
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    text: str | None = Field(
        default=None, min_length=1, max_length=MAX_AMENDMENT_SOURCE_TEXT_CHARS
    )

    @model_validator(mode="after")
    def exactly_one_source(self) -> Self:
        if (self.url is None) == (self.text is None):
            raise ValueError("Provide exactly one URL or text source")
        return self


class AmendmentSourceAssetSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sha256: str
    mime_type: str
    display_name: str
    byte_count: int
    original_url: str | None
    final_url: str | None
    text_sha256: str | None


class AmendmentSourcePackageSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_set_id: int
    status: Literal["processing", "ready", "partial", "blocked", "failed"]
    asset_count: int
    total_bytes: int
    issues: list[dict[str, Any]]
    manifest_sha256: str | None
    assets: list[AmendmentSourceAssetSnapshot] = Field(default_factory=list)
    created_at: datetime.datetime
    updated_at: datetime.datetime
