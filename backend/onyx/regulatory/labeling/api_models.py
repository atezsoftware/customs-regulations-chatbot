from __future__ import annotations

import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.labeling.provider import MAX_LABELS, LabelDefinition


class TaxonomyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    labels: list[LabelDefinition] = Field(min_length=1, max_length=MAX_LABELS)


class TaxonomySummary(BaseModel):
    id: str
    name: str
    version_hash: str
    label_count: int = Field(ge=1, le=MAX_LABELS)
    created_at: datetime.datetime


class LabelingProviderSummary(BaseModel):
    id: int
    name: str


class LabelingCounts(BaseModel):
    files: int
    canonical_chunks: int
    derived_chunks: int


class LabelingSetup(BaseModel):
    model: str
    default_label_count: int = Field(ge=1, le=MAX_LABELS)
    taxonomies: list[TaxonomySummary]
    providers: list[LabelingProviderSummary]
    counts: LabelingCounts
    active_run_id: str | None
    warnings: list[str]


class LabelSettingsSnapshot(BaseModel):
    revision: int
    taxonomy_id: str
    labels: list[LabelDefinition] = Field(min_length=1, max_length=MAX_LABELS)
    updated_at: datetime.datetime


class LabelSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    labels: list[LabelDefinition] = Field(min_length=1, max_length=MAX_LABELS)


class LabelingRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_id: UUID | None = None
    model_configuration_id: int
    idempotency_key: UUID


class LabelingRunSnapshot(BaseModel):
    id: str
    document_set_id: int
    taxonomy_id: str
    taxonomy_name: str
    model: str
    status: str
    stage: str
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    stale_chunks: int
    derived_chunks: int
    unresolved_derived_chunks: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    finished_at: datetime.datetime | None
    error: str | None
    cancel_requested: bool


class LabelingItemSnapshot(BaseModel):
    chunk_id: str
    file_id: str
    status: str
    labels: list[str] = Field(max_length=MAX_LABELS)
    error: str | None


class LabelingItemsPage(BaseModel):
    items: list[LabelingItemSnapshot]
    total: int
