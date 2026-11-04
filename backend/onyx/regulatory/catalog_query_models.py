"""Prepared retrieval input for the admin's simple search."""

from pydantic import BaseModel, Field


class CatalogQueryPreparation(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    rewritten: bool
