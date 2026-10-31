"""Public admin catalog request and response models."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from onyx.server.features.document_set.file_models import (
    RegulatoryIndexingProgressSnapshot,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot

CATALOG_SEARCH_CANDIDATES = 500


class CatalogDocumentSet(BaseModel):
    id: int
    name: str


class CatalogIndexRequest(BaseModel):
    attempt_id: UUID
    status: str
    updated_at: datetime
    failure_code: str | None


class CatalogFile(BaseModel):
    id: UUID
    name: str
    folder_path: str
    user_id: UUID | None
    file_id: str
    status: str
    file_type: str
    created_at: datetime
    chunk_count: int
    document_sets: list[CatalogDocumentSet]
    pending_review: bool
    regulatory_indexing_progress: RegulatoryIndexingProgressSnapshot | None = None
    index_request: CatalogIndexRequest | None = None


class CatalogFilePage(BaseModel):
    items: list[CatalogFile]
    total: int
    offset: int
    limit: int


class CatalogFolder(BaseModel):
    name: str
    path: str
    file_count: int


class CatalogFolderPage(BaseModel):
    items: list[CatalogFolder]
    total: int
    offset: int
    limit: int


class CatalogSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["bm25", "semantic", "hybrid"] = "bm25"
    offset: int = Field(default=0, ge=0, le=CATALOG_SEARCH_CANDIDATES - 1)
    limit: int = Field(default=25, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Search query cannot be blank")
        return value


class CatalogSearchHit(BaseModel):
    file: CatalogFile
    chunk: RegulatoryChunkSnapshot
    score: float | None


class CatalogSearchPage(BaseModel):
    items: list[CatalogSearchHit]
    total: int
    offset: int
    limit: int
    has_more: bool
    # Retrieval is a bounded ranked candidate set, not an exact corpus count.
    total_is_exact: bool = False
    candidate_limit: int = CATALOG_SEARCH_CANDIDATES
