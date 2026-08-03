from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import Tag
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tools.models import ChatMinimalTextMessage


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)

    sources: list[DocumentSource] | None = None
    document_sets: list[str] | None = None
    tags: list[Tag] | None = None
    # ISO 8601 timestamp. Only documents updated on or after this moment are
    # returned. Naive (timezone-less) values are treated as UTC server-side.
    time_cutoff: datetime | None = None
    # Explicit opt-in for the regulatory chunk index path. This is intentionally
    # independent of USER_FILE because the public search API also serves ordinary
    # uploaded knowledge.
    regulatory_chunks_only: bool = False
    as_of_date: date | None = None

    persona_id: int | None = None

    provider: str | None = None
    model: str | None = None

    skip_query_expansion: bool = False
    search_mode: Literal["hybrid", "keyword", "full_text"] = "hybrid"

    message_history: list[ChatMinimalTextMessage] | None = None

    @model_validator(mode="after")
    def validate_provider_model_pair(self) -> "SearchRequest":
        if self.model and not self.provider:
            raise ValueError("provider is required when model is specified")
        if self.provider and not self.model:
            raise ValueError("model is required when provider is specified")
        if (
            self.regulatory_chunks_only
            and len(self.query.strip()) > REGULATORY_MAX_SEARCH_QUERY_CHARS
        ):
            raise ValueError(
                "regulatory query must be a focused fragment no longer than "
                f"{REGULATORY_MAX_SEARCH_QUERY_CHARS} characters"
            )
        return self


class SearchResult(BaseModel):
    citation_id: int | None
    title: str
    # Full chunk text the LLM saw for this section. Multiple results may
    # share a citation_id when the LLM selected multiple non-overlapping
    # sections of the same document.
    content: str
    link: str | None
    source_type: str
    updated_at: str | None


class SearchResponse(BaseModel):
    results: list[SearchResult]
