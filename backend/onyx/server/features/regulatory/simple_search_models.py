"""Admin simple-search wire models."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from onyx.regulatory.catalog_summary_usage import CatalogSummaryUsage
from onyx.server.features.regulatory.catalog_models import CatalogSearchPage


class SimpleSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Search query cannot be blank")
        return value


class SimpleSearchResponse(BaseModel):
    search: CatalogSearchPage
    summary: str | None
    summary_model: str
    summary_error: str | None = None
    search_query: str
    query_rewritten: bool = False
    request_id: UUID
    summary_usage: CatalogSummaryUsage | None = None
    summary_error_code: str | None = None
