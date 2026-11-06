"""Public storage-only Markdown draft contract."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MarkdownDraftCreateRequest(BaseModel):
    idempotency_key: UUID


class MarkdownDraftPrepareRequest(BaseModel):
    expected_revision: int = Field(ge=0, strict=True)


class MarkdownDraftDocumentSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    display_name: str
    parent_document_id: UUID | None
    byte_count: int
    sha256: str


class MarkdownDraftSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    revision: int
    status: Literal["draft", "prepared"]
    documents: list[MarkdownDraftDocumentSnapshot] = Field(default_factory=list)
    source_package_id: UUID | None
    updated_at: datetime


class FrozenMarkdownDraftDocument(MarkdownDraftDocumentSnapshot):
    model_config = ConfigDict(extra="forbid", frozen=True)
    file_id: str


class FrozenMarkdownDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    draft_id: UUID
    revision: int = Field(ge=0, strict=True)
    created_by: UUID
    document_set_id: int
    environment: str
    documents: list[FrozenMarkdownDraftDocument] = Field(min_length=1, max_length=21)
