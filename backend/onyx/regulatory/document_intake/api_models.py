import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class IntakeCapabilities(BaseModel):
    enabled: bool
    target_document_set_id: int | None = None
    target_document_set_name: str = "PC Külliyatı"
    error: str | None = None


class IntakeCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_package_id: UUID
    idempotency_key: UUID


class IntakeApproveRequest(BaseModel):
    expected_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IntakeReprepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: UUID


class IntakeReprepareSnapshot(BaseModel):
    eligible: bool
    expected_sha256: str | None = None
    reason: str | None = None
    request_id: UUID | None = None
    status: str | None = None
    previous_labeling_run_id: UUID | None = None
    documents: dict[str, JsonValue] = Field(default_factory=dict)


class IntakeChunkUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=1, max_length=2_000_000)
    heading_path: list[str] | None = Field(default=None, max_length=64)
    chunk_type: str | None = Field(default=None, max_length=200)
    chunk_metadata: dict[str, JsonValue] | None = Field(default=None, max_length=64)
    validity_start_date: datetime.date | None = None
    validity_end_date: datetime.date | None = None


class IntakeChunkEditRequest(IntakeChunkUpdate, IntakeApproveRequest):
    pass


class IntakeChunkBulkEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    chunk_id: str = Field(min_length=1, max_length=512)
    updates: IntakeChunkUpdate

    @model_validator(mode="after")
    def require_updates(self) -> "IntakeChunkBulkEdit":
        if not self.updates.model_fields_set:
            raise ValueError("Supply at least one editable chunk field")
        return self


class IntakeChunksEditRequest(IntakeApproveRequest):
    model_config = ConfigDict(extra="forbid")

    edits: list[IntakeChunkBulkEdit] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def require_unique_chunks(self) -> "IntakeChunksEditRequest":
        if len({edit.chunk_id for edit in self.edits}) != len(self.edits):
            raise ValueError("Each canonical chunk can be edited only once per review")
        return self


class IntakePredecessorSelectionRequest(IntakeApproveRequest):
    model_config = ConfigDict(extra="forbid")

    predecessor_user_file_id: UUID | None
    relation: Literal["successor", "repeals"] = "successor"
    effective_date: datetime.date | None = None
    evidence: str | None = Field(default=None, max_length=4000)


class IntakePredecessorCandidate(BaseModel):
    user_file_id: UUID
    title: str
    reason: str
    score: float
    status: str


class IntakeLineageReview(BaseModel):
    predecessor_user_file_id: UUID
    title: str
    relation: Literal["successor", "repeals"]
    effective_date: datetime.date | None = None
    evidence: str | None = None
    status: Literal["draft", "frozen", "active"]


class DocumentLineageEdge(BaseModel):
    user_file_id: UUID
    title: str
    relation: Literal["successor", "repeals"]
    effective_date: datetime.date | None = None
    evidence: str | None = None


class DocumentLineageSnapshot(BaseModel):
    user_file_id: UUID
    title: str
    status: Literal["current", "repeal_scheduled", "repealed"]
    effective_date: datetime.date | None = None
    predecessors: list[DocumentLineageEdge] = Field(default_factory=list)
    successors: list[DocumentLineageEdge] = Field(default_factory=list)


class IntakeSourceIssue(BaseModel):
    code: str
    locator: str | None = None
    retryable: bool = True
    failure_detail: str | None = None


class IntakeDocumentLink(BaseModel):
    href: str
    label: str
    target_document_id: UUID


class IntakeDocumentSnapshot(BaseModel):
    id: UUID
    asset_id: UUID
    display_name: str
    role: Literal["root", "annex"]
    parent_document_id: UUID | None = None
    source_url: str | None = None
    status: str
    chunk_count: int = Field(ge=0)
    labeling_failed_count: int = Field(default=0, ge=0)
    labels: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None
    user_file_id: UUID | None = None
    document_links: list[IntakeDocumentLink] = Field(default_factory=list)
    lineage_review: IntakeLineageReview | None = None


class IntakeBatchSnapshot(BaseModel):
    id: UUID
    source_package_id: UUID
    status: Literal[
        "preparing", "ready", "approving", "completed", "partial", "blocked", "failed"
    ]
    stage: str
    target_document_set_id: int
    target_document_set_name: str
    review_sha256: str | None = None
    edit_review_sha256: str | None = None
    error: str | None = None
    issues: list[IntakeSourceIssue] = Field(default_factory=list)
    documents: list[IntakeDocumentSnapshot] = Field(default_factory=list)
    labeling_run_id: UUID | None = None
    labeling_failed_count: int = Field(default=0, ge=0)
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None
