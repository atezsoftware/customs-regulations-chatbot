"""Source identity is independent of editable legal titles and extraction metadata."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.amendments.annexes.models import SourceLink


class IntakeOriginalSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: UUID
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_name: str
    original_url: str | None
    final_url: str | None


class IntakeRelatedFile(BaseModel):
    user_file_id: UUID
    document_id: UUID
    title: str
    source: IntakeOriginalSource
    source_links: list[SourceLink]


class IntakeFileSourceLineage(BaseModel):
    version: Literal[1] = 1
    intake_id: UUID
    package_id: UUID
    document_set_id: int
    document_id: UUID
    role: Literal["root", "annex"]
    source: IntakeOriginalSource
    parents: list[IntakeRelatedFile] = Field(default_factory=list)
    attachments: list[IntakeRelatedFile] = Field(default_factory=list)


class ReviewedRelatedFile(BaseModel):
    user_file_id: UUID
    title: str
    original_name: str | None
    source_url: str | None
    document_set_id: int
    intake_id: UUID
    review_decision_id: UUID
    basis: Literal["reviewed_existing_parent"] = "reviewed_existing_parent"


class ReviewedFileRelations(BaseModel):
    parents: list[ReviewedRelatedFile] = Field(default_factory=list)
    attachments: list[ReviewedRelatedFile] = Field(default_factory=list)
