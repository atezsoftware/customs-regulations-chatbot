"""Exact source authority for an administrator-reviewed unpublished intake edit."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewedMetadataDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_file_id: UUID
    previous_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revisions: dict[str, UUID] = Field(min_length=1)
    refresh_metadata: bool

    @model_validator(mode="after")
    def require_actual_context_change(self) -> "ReviewedMetadataDocument":
        if self.refresh_metadata != (self.previous_input_sha256 != self.input_sha256):
            raise ValueError(
                "Reviewed metadata refresh must match an actual source change"
            )
        return self


class ReviewedMetadataEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    intake_id: UUID
    package_id: UUID
    previous_labeling_run_id: UUID | None
    expected_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    documents: dict[UUID, ReviewedMetadataDocument] = Field(min_length=1, max_length=21)
