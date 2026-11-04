"""Admin simple-search wire models."""

from pydantic import BaseModel, Field, field_validator

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
