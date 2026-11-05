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
