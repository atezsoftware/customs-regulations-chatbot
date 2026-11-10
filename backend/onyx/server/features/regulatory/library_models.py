"""Minimal, read-only legislation library responses for non-admin clients."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from onyx.server.features.regulatory.simple_search_models import SimpleSearchRequest

LibraryLanguage = Literal["tr", "en", "de"]


class LibrarySearchTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=640)
    overview: str | None = Field(default=None, max_length=800)
    legislations: list[Annotated[str, StringConstraints(max_length=180)]] = Field(
        default_factory=list, max_length=4
    )


class LibrarySearchRequest(SimpleSearchRequest):
    model_config = ConfigDict(extra="forbid")

    language: LibraryLanguage = "tr"
    context: list[LibrarySearchTurn] = Field(default_factory=list, max_length=3)


class LibraryFile(BaseModel):
    id: UUID
    name: str
    folder_path: str
    chunk_count: int


class LibraryFilePage(BaseModel):
    items: list[LibraryFile]
    total: int
    offset: int
    limit: int


class LibraryRepeal(BaseModel):
    heading: str
    source_label: str
    source_url: str
    kind: Literal["article", "clause"] = "article"


class LibraryContent(BaseModel):
    file_id: UUID
    name: str
    markdown: str
    content_sha256: str
    verified_repeals: list[LibraryRepeal] = Field(default_factory=list)


class LibraryChunk(BaseModel):
    id: str
    text: str
    chunk_type: str | None
    heading_path: list[str]
    chunk_metadata: dict[str, str]


class LibrarySearchHit(BaseModel):
    file: LibraryFile
    chunk: LibraryChunk


class LibrarySearchResults(BaseModel):
    items: list[LibrarySearchHit]


class LibrarySearchResponse(BaseModel):
    kind: Literal["search", "memory"] = "search"
    search: LibrarySearchResults
    summary: str | None
    summary_error: str | None
