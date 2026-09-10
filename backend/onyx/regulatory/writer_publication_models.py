"""Frozen legacy writer operations use the same qualified publication authority."""

from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationScope,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
    PreparedContextView,
)


class WriterPublicationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    scope: PublicationScope
    user_file_id: UUID
    kind: Literal[
        "correction",
        "reindex",
        "metadata",
        "delete",
        "amendment",
        "validity",
        "durable",
        "cancellation",
    ]
    index_state_sha256: str | None = None
    canonical_before_sha256: str
    canonical_after: list[AnnexCanonicalSnapshot] | None = None
    name_after: str | None = None
    amendment_proposal_id: int | None = None
    amendment_review_sha256: str | None = None
    durable_job_id: UUID | None = None
    durable_input_sha256: str | None = None
    cancellation_job_id: UUID | None = None
    cancelled_manifest_sha256: str | None = None
    complete_file: bool = False
    chunk_count_after: int | None = Field(default=None, ge=0)
    secondary_reconcile_pending: bool | None = None
    indexes: list[PublicationIndexSnapshot] = Field(min_length=1)
    previous_binding_ids: list[UUID]
    bindings: list[AnnexTemporalProjection]
    canonical_revisions: dict[UUID, UUID] = Field(default_factory=dict)
    views: list[PreparedContextView] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if (self.kind == "durable") != (
            self.durable_job_id is not None and self.durable_input_sha256 is not None
        ):
            raise ValueError("durable writer requires its exact job input identity")
        if (self.kind == "cancellation") != (self.cancellation_job_id is not None):
            raise ValueError("cancellation writer requires its exact job identity")
        indexes = {index.index_uuid: index for index in self.indexes}
        if len(indexes) != len(self.indexes):
            raise ValueError("duplicate writer physical index")
        identities = {
            (binding.index.index_uuid, binding.projection.ordinal)
            for binding in self.bindings
        }
        if len(identities) != len(self.bindings):
            raise ValueError("duplicate writer projection identity")
        for binding in self.bindings:
            if binding.index.index_uuid not in indexes:
                raise ValueError("writer binding is outside target indexes")
        if self.kind == "delete" and (
            self.bindings or self.canonical_after is not None
        ):
            raise ValueError("deletion cannot publish live canonical representations")
        if self.canonical_after is not None and any(
            UUID(row.user_file_id) != self.user_file_id for row in self.canonical_after
        ):
            raise ValueError("writer canonical file scope mismatch")
        return self
