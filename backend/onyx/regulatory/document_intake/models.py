from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.amendments.annexes.models import SourceLink

INTAKE_DELIVERY_SECONDS = 15 * 60


class IntakeSourceAsset(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: UUID
    package_id: UUID
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_id: str
    text_file_id: str | None
    text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mime_type: str
    display_name: str
    byte_count: int = Field(ge=1)
    original_url: str | None = None
    final_url: str | None = None


class IntakeSourceContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunking_version: Literal["1"] = "1"
    package_id: UUID
    asset_id: UUID
    asset_sha256: str
    parent_asset_hashes: list[str]
    parent_asset_ids: list[UUID]
    parent_titles: list[str]
    source_links: list[SourceLink]


class IntakeFilePreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_file_id: UUID
    source_asset_id: UUID
    documents_sha256: str
    plaintext_sha256: str
    initial_chunks_created: bool
