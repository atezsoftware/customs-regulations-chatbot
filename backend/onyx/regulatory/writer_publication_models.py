"""Frozen legacy writer operations use the same qualified publication authority."""

from datetime import date
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationScope,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
    PreparedContextView,
)


class AmendmentConsumer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    consumer_id: str
    relation: Literal["aggregate", "image", "text_occurrence", "context_input"]
    path: list[str]
    quote: str = ""


class AmendmentContextEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    consumer_ids: list[str]
    effective_start: date
    effective_end: date
    context_sha256: str
    audit_input_sha256: str
    outcome: Literal["affected", "unchanged", "unresolved"]
    quote: str
    reason: str
    source_id: str | None = None
    source_quote: str = ""
    source_side: Literal["before", "after"] | None = None


class AmendmentImpactReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    canonical_before_sha256: str
    canonical_after_sha256: str
    index_uuid: str | None = None
    affected_windows: dict[str, list[tuple[date, date]]]
    dependencies: list[AmendmentConsumer]
    context_evidence: list[AmendmentContextEvidence]
    unchanged_ids: list[str]
    unresolved: list[str] = Field(default_factory=list)


class AmendmentSourceUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_file_id: UUID
    index_uuid: str | None
    as_of_date: date
    source_ids: list[str]
    canonical_sha256: str
    canonical_count: int
    consumers: list[AmendmentConsumer]
    contextual_consumer_count: int
    warnings: list[str]


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
        "baseline",
    ]
    index_state_sha256: str | None = None
    canonical_before_sha256: str
    canonical_after: list[AnnexCanonicalSnapshot] | None = None
    name_after: str | None = None
    amendment_proposal_id: int | None = None
    baseline_origin_proposal_id: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
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
    structure_repair: dict[str, JsonValue] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    views: list[PreparedContextView] = Field(default_factory=list)
    amendment_impacts: list[AmendmentImpactReport] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.structure_repair is not None:
            from onyx.regulatory.structure_metadata_repair import StructureRepairPlan

            repair = StructureRepairPlan.model_validate(self.structure_repair)
            desired = {row.id: row for row in self.canonical_after or []}
            if (
                self.kind != "metadata"
                or not repair.changes
                or any(
                    desired.get(change.after.id) != change.after
                    for change in repair.changes
                )
            ):
                raise ValueError(
                    "Structure repair source authority differs from its canonical transition"
                )
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
