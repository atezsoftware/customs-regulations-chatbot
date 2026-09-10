"""Durable, review-bound execution contracts; no provider or DB authority from clients."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


class AnnexPublicationDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent_id: UUID
    change_set_id: UUID
    logical_group_id: UUID
    review_revision: int
    review_sha256: str
    publication_generation: int
    tenant_id: str
    environment: str
    database_identity: str


class AnnexPublicationOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index_uuid: str
    ordinal: int
    kind: Literal["upsert", "tombstone"]
    binding: AnnexTemporalProjection | None = None


class AnnexPublicationOperations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operations: list[AnnexPublicationOperation]


class AnnexEmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    projection_id: UUID
    custom_id: str = Field(pattern=r"^annex-[0-9a-f]{32}$")
    inputs: list[str]
    configuration: dict[str, str | int | float | bool | None]
    dimension: int


class AnnexBatchCorrelationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    remote_id: str
    custom_id: str
    request_sha256: str
    response_json: str


class AnnexEmbeddingCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: AnnexEmbeddingRequest
    status: Literal["pending", "submitting", "submitted", "complete", "indeterminate"]
    remote_id: str | None = None
    vectors: list[list[float]] | None = None
    provider_receipt: AnnexBatchCorrelationReceipt | None = None


class AnnexPublicationNeedsReconciliation(ValueError):
    """Provider outcome cannot be retried without positively resolved remote authority."""
