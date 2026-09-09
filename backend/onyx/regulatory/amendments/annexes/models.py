from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourcePackageStatus = Literal["processing", "ready", "partial", "blocked", "failed"]


class SourceIssue(BaseModel):
    code: str
    locator: str | None = None
    retryable: bool = True


class SourceLink(BaseModel):
    parent_asset_hash: str
    target_asset_hash: str | None = None
    source_page: int | None = None
    source_field: str
    label: str
    original_url: str | None = None
    final_url: str | None = None
    kind: Literal["url", "internal", "embedded"]


class AcquiredAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256: str
    content: bytes = Field(exclude=True)
    mime_type: str
    display_name: str
    original_url: str | None = None
    final_url: str | None = None
    text: str = ""


class AcquisitionResult(BaseModel):
    status: SourcePackageStatus
    assets: list[AcquiredAsset] = Field(default_factory=list)
    links: list[SourceLink] = Field(default_factory=list)
    issues: list[SourceIssue] = Field(default_factory=list)


class DiscoveredLink(BaseModel):
    kind: Literal["url", "internal", "embedded"]
    target: str
    label: str
    source_page: int | None = None
    source_field: str
    embedded_base64: str | None = None


class SourceInspection(BaseModel):
    mime_type: str
    text: str = ""
    links: list[DiscoveredLink] = Field(default_factory=list)
