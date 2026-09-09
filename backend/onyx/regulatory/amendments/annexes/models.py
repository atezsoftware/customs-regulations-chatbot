from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourcePackageStatus = Literal["processing", "ready", "partial", "blocked", "failed"]


class SourceIssue(BaseModel):
    code: str
    locator: str | None = None
    retryable: bool = True


class SourceLink(BaseModel):
    parent_url: str | None = None
    requested_url: str | None = None
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


class AnnexModelSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model_provider: str
    model_name: str


class AnnexLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    cell: str | None = None
    row: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)
    path: str | None = None
    original_box: tuple[float, float, float, float] | None = None
    normalized_box: tuple[float, float, float, float] | None = None
    original_width: float | None = None
    original_height: float | None = None
    coordinate_system: (
        Literal["top_left_pixels", "top_left_points", "pdf_user_space"] | None
    ) = None


AnnexElementKind = Literal[
    "text", "table_row", "table_cell", "footnote", "image_region"
]


class ExtractedAnnexElement(BaseModel):
    canonical_chunk_id: str | None = None
    bound_to_regulatory_chunk_id: str | None = None
    canonical_role: Literal["authoritative", "supporting"] = "authoritative"
    model_config = ConfigDict(extra="forbid")
    kind: AnnexElementKind
    text: str
    semantic_key: str | None = None
    locator: AnnexLocator = Field(default_factory=AnnexLocator)
    formula: str | None = None
    value: str | int | float | bool | None = None
    evidence_kind: Literal["original", "rendered_preview", "canonical"] = "original"
    status: Literal["readable", "uncertain", "unreadable"] = "readable"
    issues: list[str] = Field(default_factory=list)
    aggregate: bool = False
    image_file_id: str | None = None
    source_asset_id: str | None = None


class AnnexExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    source_sha256: str
    mime_type: str
    elements: list[ExtractedAnnexElement] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    model_snapshot: AnnexModelSnapshot | None = None


class AnnexRenderedPage(BaseModel):
    original_orientation: int = 1
    page: int
    width: float
    height: float
    png: bytes
    text_elements: list[ExtractedAnnexElement] = Field(default_factory=list)


class AnnexVisionElement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: AnnexElementKind
    text: str = Field(max_length=20000)
    box: tuple[float, float, float, float]
    status: Literal["readable", "uncertain", "unreadable"]
    issues: list[str] = Field(default_factory=list, max_length=20)


class AnnexVisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[AnnexVisionElement] = Field(max_length=2000)


class AnnexOriginalEvidence(BaseModel):
    canonical_chunk_ids: list[str] = Field(default_factory=list)
    file_id: str
    mime_type: str | None = None
    sha256: str | None = None
    available: bool = False
    issue: str | None = None


class AnnexBaseline(BaseModel):
    canonical_amendment_chunk_ids: list[str] = Field(default_factory=list)
    revision_id: str | None = None
    baseline_sha256: str
    canonical_text: str
    elements: list[ExtractedAnnexElement]
    originals: list[AnnexOriginalEvidence]
    visual_evidence_available: bool
    issues: list[str] = Field(default_factory=list)


class RegulatoryChunkEvidence(BaseModel):
    bound_to_regulatory_chunk_id: str | None = None
    image_file_id: str | None = None
    image_file_ids: list[str] = Field(default_factory=list)
    source_links: dict[int, str] = Field(default_factory=lambda: {0: ""})
    source_asset_ids: list[str] = Field(default_factory=list)
    annex_element_ids: list[str] = Field(default_factory=list)
