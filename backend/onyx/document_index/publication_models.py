"""Frozen file-publication contracts, independent of legacy contiguous verification."""

import json
from datetime import datetime
from hashlib import sha256
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.document_index.elasticsearch.schema import DocumentChunk


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
        required = {
            name
            for name, field in DocumentChunk.model_fields.items()
            if field.is_required()
        }
        if not required.issubset(source):
            raise ValueError("full projection is missing required search fields")
        if (
            source.get("chunk_index") != self.ordinal
            or not isinstance(source.get("regulatory_chunk_id"), str)
            or not source["regulatory_chunk_id"]
        ):
            raise ValueError("projection ordinal/canonical identity mismatch")
        for key in ("content", "doc_summary", "chunk_context"):
            if not isinstance(source.get(key), str):
                raise ValueError("projection text/context missing")
        if any(key.startswith("publication_") for key in source):
            raise ValueError("reserved publication field in source")
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
