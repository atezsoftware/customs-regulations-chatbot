from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.regulatory.amendments.annexes.models import SourceLink

INTAKE_DELIVERY_SECONDS = 15 * 60
IntakeChunkingVersion = Literal["1", "2", "3", "4"]


class IntakePreparationPolicy(BaseModel):
    """Immutable model and budgets selected when a new intake is admitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    chunking_version: Literal["4"] = "4"
    provider: Literal["google"] = "google"
    model_name: Literal["gemini-embedding-2"] = "gemini-embedding-2"
    body_token_limit: Literal[512] = 512
    table_token_limit: Literal[2048] = 2048


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

    chunking_version: IntakeChunkingVersion = "3"
    preparation_policy: IntakePreparationPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    package_id: UUID
    asset_id: UUID
    asset_sha256: str
    parent_asset_hashes: list[str]
    parent_asset_ids: list[UUID]
    parent_titles: list[str]
    source_links: list[SourceLink]

    @model_validator(mode="after")
    def require_matching_policy(self) -> "IntakeSourceContext":
        if (self.chunking_version == "4") != (self.preparation_policy is not None):
            raise ValueError("Intake v4 requires its frozen preparation policy")
        return self


class IntakeFilePreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_file_id: UUID
    source_asset_id: UUID
    documents_sha256: str
    plaintext_sha256: str
    initial_chunks_created: bool
