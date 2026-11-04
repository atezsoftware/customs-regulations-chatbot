"""Non-sensitive provider-reported usage for a single summary generation."""

from pydantic import BaseModel, Field


class CatalogSummaryUsage(BaseModel):
    response_id: str | None
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    provider_cost_usd: float | None = Field(ge=0, allow_inf_nan=False)
