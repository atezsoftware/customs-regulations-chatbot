from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.regulatory.amendments.models import DateResolution

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
    extraction_method: Literal["native", "vision", "canonical", "unknown"] = "unknown"
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


class AnnexEvidenceParent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    file_id: str
    sha256: str
    mime_type: str
    extraction_sha256: str
    element_count: int
    page_count: int | None
    canonical_chunk_ids: list[str]


class AnnexEvidencePage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_index: int
    original_page: int
    view_page: int
    normalized_box: tuple[float, float, float, float]


class AnnexEvidenceElementMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_index: int
    original_position: int
    original_locator: AnnexLocator
    view_position: int


class AnnexEvidenceView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str
    label: str
    parents: list[AnnexEvidenceParent]
    pages: list[AnnexEvidencePage]
    selected_positions: list[int]
    element_mappings: list[AnnexEvidenceElementMap]
    boundary_positions: list[int]
    selection_method: Literal[
        "native_boundaries",
        "bound_whole_original",
        "native_sheet",
        "ordered_bound_originals",
        "ordered_source_occurrences",
    ]
    source_occurrences: list[SourceLink] = Field(default_factory=list)
    extraction_version: str = "annex-extraction-v1"
    renderer_version: str = "annex-rendering-v1"


class AnnexExtraction(BaseModel):
    evidence_view: AnnexEvidenceView | None = None
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
    issues: list[
        Literal[
            "low_readability",
            "missing_evidence",
            "ambiguous_structure",
            "uncertain_value",
            "unsupported_visual",
            "incomplete_coverage",
            "formula_change_requires_review",
        ]
    ] = Field(default_factory=list)

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


class AnnexComparedImage(BaseModel):
    """Exact image submitted to comparison, in extraction-local page coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    side: Literal["old", "new"]
    page: int
    kind: Literal["comparison_page", "comparison_tile", "comparison_region"]
    normalized_box: tuple[float, float, float, float]
    sha256: str
    byte_count: int


class AnnexComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = 1
    old_source_sha256: str
    new_source_sha256: str
    old_snapshot_sha256: str
    new_snapshot_sha256: str
    changes: list[AnnexDifference]
    image_manifest: list[AnnexComparedImage] = Field(default_factory=list)
    coverage: AnnexCoverage
    model_snapshot: AnnexModelSnapshot | None = None
    prompt_version: str = "annex-comparison-v2"
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


class AnnexCanonicalSnapshot(BaseModel):
    """Complete canonical publication input, including historical/source metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    user_file_id: str
    chunk_type: str | None
    status: str
    projection_ordinal: int
    supersedes_chunk_id: str | None
    superseded_by_chunk_id: str | None
    position: int
    text: str
    heading_path: list[str]
    metadata: dict[str, JsonValue]
    source: str
    validity_start_date: date | None
    validity_end_date: date | None


class AnnexChangeItemDraft(BaseModel):
    insertion_after_chunk_id: str | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal[
        "replace", "insert", "remove", "split", "merge", "move", "visual"
    ]
    old_chunk_ids: list[str]
    new_chunks: list[AnnexCanonicalSnapshot]
    old_positions: list[int]
    new_positions: list[int]


class AnnexReviewEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: UUID
    side: Literal["old", "new"]
    kind: Literal["original", "comparison_page", "comparison_tile", "comparison_region"]
    file_id: str
    sha256: str
    mime_type: str
    byte_count: int
    parent_file_id: str
    parent_sha256: str
    source_asset_id: UUID | None = None
    locator: AnnexLocator = Field(default_factory=AnnexLocator)


class AnnexNewElementEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    position: int
    element_id: UUID
    source_asset_id: UUID
    parent_file_id: str
    evidence_ids: list[UUID]
    image_file_ids: list[str]


class AnnexNewEvidenceRemapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    extraction_sha256: str
    elements: list[AnnexNewElementEvidence]


class AnnexInstructionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    annex_label: str
    instruction_indices: list[int]
    instruction_texts: list[str]
    target_sources: list[str]


class AnnexElementCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    position: int = Field(ge=0)
    before_text: str
    corrected_text: str = Field(min_length=1, max_length=100000)
    reason: str = Field(min_length=1, max_length=2000)


class AnnexCorrectionReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    supported: bool
    rationale: str
    input_sha256: str | None = None
    model_snapshot: AnnexModelSnapshot | None = None


class AnnexChangeDraft(BaseModel):
    batch_id: int | None = None
    date_resolution: DateResolution | None = None
    target_sources: list[str] = Field(default_factory=list)
    indexing_configuration: dict[str, JsonValue] | None = None
    source_only_canonical_ids: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", frozen=True)
    instruction_indices: list[int]
    instruction_texts: list[str]
    annex_label: str
    user_file_id: UUID | None = None
    effective_date: date | None = None
    source_package_id: UUID | None = None
    source_text_sha256: str | None = None
    source_manifest_sha256: str | None = None
    original_source_text_sha256: str | None = None
    source_graph_sha256: str | None = None
    source_graph: list[SourceLink] = Field(default_factory=list)
    submitted_source_text: str | None = None
    preparation_configuration: dict[str, str] = Field(default_factory=dict)
    new_evidence_remapping: AnnexNewEvidenceRemapping | None = None
    raw_new_extraction: AnnexExtraction | None = None
    corrections: list[AnnexElementCorrection] = Field(default_factory=list)
    correction_reconciliation: AnnexCorrectionReconciliation | None = None
    corrected_by: UUID | None = None
    insertion_after_chunk_id: str | None = None
    baseline_scope: list[AnnexCanonicalSnapshot] = Field(default_factory=list)
    baseline: AnnexBaseline | None = None
    old_extraction: AnnexExtraction | None = None
    new_extraction: AnnexExtraction | None = None
    comparison: AnnexComparison | None = None
    patch_plan: AnnexPatchPlan | None = None
    items: list[AnnexChangeItemDraft] = Field(default_factory=list)
    baseline_context: PreparedContextView | None = None
    impact: AnnexContextImpact | None = None
    evidence: list[AnnexReviewEvidence] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_instruction_coverage(self) -> "AnnexChangeDraft":
        if (
            not self.instruction_indices
            or min(self.instruction_indices) < 0
            or self.instruction_indices != sorted(set(self.instruction_indices))
            or len(self.instruction_texts) != len(self.instruction_indices)
        ):
            raise ValueError("invalid grouped instruction coverage")
        return self


class AnnexReviewEvidenceScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_id: int
    document_set_id: int
    user_file_id: UUID
    created_by: UUID | None
    environment: str
    old_original_file_ids: list[str]
    new_original_file_ids: list[str]

    def storage_identity(self) -> dict[str, JsonValue]:
        return self.model_dump(
            mode="json", exclude={"old_original_file_ids", "new_original_file_ids"}
        )


class AnnexComparisonImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    png: bytes
    kind: Literal["comparison_page", "comparison_tile", "comparison_region"]
    normalized_box: tuple[float, float, float, float]
