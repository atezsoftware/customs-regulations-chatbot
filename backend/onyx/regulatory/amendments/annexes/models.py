from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    page_count: int | None = Field(default=None, ge=1)
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
    kind: Literal["text", "table_cell", "footnote", "image_region"]
    text: str = Field(max_length=20000)
    box: tuple[float, float, float, float]
    status: Literal["readable", "uncertain", "unreadable"]
    issues: list[
        Literal[
            "low_readability",
            "missing_evidence",
            "ambiguous_structure",
            "uncertain_value",
            "unsupported_visual",
        ]
    ] = Field(default_factory=list, max_length=20)


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


class AnnexElementReference(BaseModel):
    """An ordinal inside a frozen extraction, never a model-authored DB identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    position: int = Field(ge=0)
    text: str
    locator: AnnexLocator


class AnnexDifference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal[
        "replace", "insert", "remove", "move", "split", "merge", "visual"
    ]
    old: list[AnnexElementReference] = Field(default_factory=list)
    new: list[AnnexElementReference] = Field(default_factory=list)
    explanation: str
    uncertain: bool = False


class AnnexComparisonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: list[AnnexDifference] = Field(default_factory=list, max_length=2000)
    old_positions: list[int] = Field(default_factory=list)
    new_positions: list[int] = Field(default_factory=list)
    old_pages: list[int] = Field(default_factory=list)
    new_pages: list[int] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_overlapping_operations(self) -> "AnnexComparisonResponse":
        for side in ("old", "new"):
            positions = [
                reference.position
                for change in self.changes
                for reference in getattr(change, side)
            ]
            if len(positions) != len(set(positions)):
                raise ValueError(
                    "overlapping change references: each physical change must appear once"
                )
        return self


class AnnexCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    old_positions: list[int]
    new_positions: list[int]
    old_pages: list[int]
    new_pages: list[int]
    method: Literal["identical_asset", "native_structure", "simultaneous_vision"]


class AnnexComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = 1
    old_source_sha256: str
    new_source_sha256: str
    old_snapshot_sha256: str
    new_snapshot_sha256: str
    changes: list[AnnexDifference]
    coverage: AnnexCoverage
    model_snapshot: AnnexModelSnapshot | None = None
    prompt_version: str = "annex-comparison-v1"
    issues: list[str]
    ready: bool


class AnnexCanonicalSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_chunk_id: str
    old_position: int
    start: int
    end: int
    old_text: str


class AnnexCanonicalPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_role: Literal["authoritative", "supporting"] = "authoritative"
    validated_spans: list[AnnexCanonicalSpan] = Field(default_factory=list)
    old_chunk_id: str | None
    old_text: str | None
    new_text: str | None
    old_positions: list[int]
    new_positions: list[int]
    operation: Literal["replace", "insert", "remove", "move", "visual"]


class AnnexPatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_sha256: str
    comparison_sha256: str
    effective_date: date | None
    patches: list[AnnexCanonicalPatch]
    direct_canonical_changes: list[str]
    metadata_only: list[str]
    retire_history: list[str]
    unchanged: list[str]
    issues: list[str]
    ready: bool


class ContextSourceRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_chunk_id: str
    start: int
    end: int


class ContextSourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str
    selector: str
    reference_date: date
    text: str
    ordered_ranges: list[ContextSourceRange]


class ContextGenerationCall(BaseModel):
    generation_input_sha256: str = ""
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_sha256: str
    stage: Literal["summary", "chunk", "fallback_summary", "durable_chunk"]
    prompt_json: str
    config_sha256: str
    output: str
    source_text: str
    token_budget: int
    tokenizer: str
    generation_path: Literal["normal", "durable"] = "normal"


class FrozenContextProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_chunk_id: str
    source_snapshot_sha256: str
    generation_path: Literal["normal", "durable"]
    request_hashes: list[str]
    embedding_input_sha256: str
    embedding_config_sha256: str
    embedding_texts: list[str]
    canonical_text_sha256: str
    metadata_sha256: str
    doc_summary: str = ""
    chunk_context: str = ""
    title: str | None = None
    mini_chunk_texts: list[str] = Field(default_factory=list)
    contextual_config: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    embedding_config: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    validity_start: date | None = None
    validity_end: date | None = None
    projection_id: str | None = None
    vector_reuse_verified: bool = False
    canonical_dependency_ids: list[str] = Field(default_factory=list)
    existing_index_evidence: dict[str, str | int] = Field(default_factory=dict)


class PreparedContextView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    projections: list[FrozenContextProjection] = Field(default_factory=list)
    snapshots: list[ContextSourceSnapshot] = Field(default_factory=list)
    calls: list[ContextGenerationCall] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class AnnexContextImpact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    direct_canonical_changes: list[str]
    contextual_candidates: list[str]
    embedding_changes: list[str]
    context_only: list[str]
    metadata_only: list[str]
    retire_history: list[str]
    unchanged: list[str]
    reasons: dict[str, list[str]]
    prepared: PreparedContextView
    ready: bool


class ExistingIndexEmbeddingEvidence(BaseModel):
    """Server-reconstructed evidence from an existing index, never fresh generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index_name: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    canonical_chunk_id: str
    projection_ordinal: int = Field(ge=0)
    embedding_input_sha256: str
    embedding_config_sha256: str
    canonical_text_sha256: str
    vector_dimension: int = Field(gt=0)
    expected_dimension: int = Field(gt=0)
