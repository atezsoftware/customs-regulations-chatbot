"""Frozen file-publication contracts, independent of legacy contiguous verification."""

import json
from datetime import datetime
from hashlib import sha256
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.document_index.elasticsearch.constants import DEFAULT_MAX_CHUNK_SIZE


class PublicationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PublicationScope(PublicationModel):
    tenant_id: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    database_identity: str = Field(min_length=1)


class FileOwnership(PublicationModel):
    scope: PublicationScope
    user_file_id: UUID
    owner_id: UUID
    fencing_token: int = Field(gt=0)
    expires_at: datetime


class FileReservations(PublicationModel):
    ownership: FileOwnership
    ordinals: tuple[int, ...]
    gate_closed: bool


class ReadObservation(PublicationModel):
    scope: PublicationScope
    committed_epoch: int = Field(ge=0)


class PublicationIndexSnapshot(PublicationModel):
    index_name: str = Field(min_length=1)
    index_uuid: str = Field(min_length=1)
    search_settings_id: int
    model_provider: str
    model_name: str = Field(min_length=1)
    vector_dimension: int = Field(gt=0)
    embedding_config_sha256: str = Field(min_length=64, max_length=64)
    multitenant: bool


def publication_digest(value: JsonValue) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


class SerializedPublicationSource(PublicationModel):
    """Strict ES JSON shapes emitted by DocumentChunk's serializer.

    Timestamps are epoch seconds and tenant identity is an explicit string here;
    validation must not use DocumentChunk's process-global tenant deserializer.
    The original JSON is retained by FrozenPublicationProjection without coercion.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    max_chunk_size: int = Field(default=DEFAULT_MAX_CHUNK_SIZE, gt=0)
    title: str | None = None
    title_vector: list[float] | None = None
    content: str
    content_vector: list[float] = Field(min_length=1)
    source_type: str
    metadata_list: list[str] | None = None
    last_updated: int | None = None
    created_at: int | None = None
    public: bool
    access_control_list: list[str]
    hidden: bool = False
    written_by_port: bool | None = None
    global_boost: int
    semantic_identifier: str
    image_file_id: str | None = None
    source_links: str | None = None
    blurb: str
    doc_summary: str
    chunk_context: str
    metadata_suffix: str | None = None
    document_sets: list[str] | None = None
    user_projects: list[int] | None = None
    personas: list[int] | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    ancestor_hierarchy_node_ids: list[int] | None = None
    regulatory_chunk_id: str = Field(min_length=1)
    heading_path: list[str] | None = None
    provision_identifiers: list[str] | None = None
    decision_numbers: list[str] | None = None
    legal_dates: list[str] | None = None
    validity_start_date: int | None = None
    validity_end_date: int | None = None
    tenant_id: str | None = None

    @model_validator(mode="after")
    def validate_title_vector_pair(self) -> Self:
        if (self.title is None) != (self.title_vector is None):
            raise ValueError("title and title vector must both be present or absent")
        return self


class FrozenPublicationProjection(PublicationModel):
    """Exact serialized source and actual encoder inputs/configuration.

    Source contains all search fields (vectors, image/source and temporal metadata),
    without publication controls. JSON strings keep nested state immutable.
    """

    ordinal: int = Field(ge=0)
    context_projection_id: str = Field(min_length=1)
    source_json: str
    embedding_inputs: tuple[str, ...] = Field(min_length=1)
    embedding_config_json: str

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        source = json.loads(self.source_json)
        config = json.loads(self.embedding_config_json)
        if not isinstance(source, dict) or not isinstance(config, dict):
            raise ValueError("projection source/config must be JSON objects")
        serialized = SerializedPublicationSource.model_validate(source)
        if serialized.chunk_index != self.ordinal:
            raise ValueError("projection ordinal/canonical identity mismatch")
        publication_digest(source)
        publication_digest(config)
        return self


class PublicationVerification(PublicationModel):
    reservations: FileReservations
    index: PublicationIndexSnapshot
    live_ordinals: tuple[int, ...]
    canonical_chunk_ids: frozenset[str]
    manifest_sha256: str


class IndexedProjectionEvidence(PublicationModel):
    index: PublicationIndexSnapshot
    source_json: str
    frozen_projection: FrozenPublicationProjection | None
    payload_sha256: str | None
