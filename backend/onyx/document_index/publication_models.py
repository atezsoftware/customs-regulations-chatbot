"""Frozen file-publication contracts, independent of legacy contiguous verification."""

import json
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from onyx.document_index.encoder_authority import EffectiveEncoderAuthority
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


class PublicationReadEvidence(PublicationModel):
    observation: ReadObservation
    user_file_ids: tuple[UUID, ...] = Field(default_factory=tuple)


class PublicationEncoderAuthority(PublicationModel):
    provider: str | None
    model: str
    effective_dimension: int = Field(gt=0)
    endpoint_sha256: str
    deployment_name: str | None
    api_version: str | None
    normalize: bool | None
    passage_prefix: str | None


class PublicationEncoderReceipt(PublicationModel):
    """Full transport receipt plus explicit resolution of omitted authority facts."""

    configuration_json: str
    authority: PublicationEncoderAuthority
    resolution_sha256: str = Field(min_length=64, max_length=64)
    resolved_fields: dict[str, JsonValue] = Field(default_factory=dict)

    def effective_authority(self) -> "EffectiveEncoderAuthority":
        from onyx.document_index.encoder_authority import effective_receipt_authority

        return effective_receipt_authority(self)

    @model_validator(mode="after")
    def validate_encoder_authority(self) -> Self:
        configuration = json.loads(self.configuration_json)
        if not isinstance(configuration, dict) or set(configuration).intersection(
            self.resolved_fields
        ):
            raise ValueError(
                "encoder authority resolution conflicts with actual receipt"
            )
        resolved = {**configuration, **self.resolved_fields}
        expected = self.authority.model_dump(mode="json")
        dimension = resolved.get("reduced_dimension") or resolved.get("dimension")
        for key, value in expected.items():
            actual_key = "dimension" if key == "effective_dimension" else key
            if (
                actual_key not in resolved
                or (dimension if key == "effective_dimension" else resolved[actual_key])
                != value
            ):
                raise ValueError(f"unresolved or incompatible encoder authority: {key}")
        return self


class PublicationIndexSnapshot(PublicationModel):
    index_name: str = Field(min_length=1)
    index_uuid: str = Field(min_length=1)
    search_settings_id: int
    model_provider: str
    model_name: str = Field(min_length=1)
    vector_dimension: int = Field(gt=0)
    embedding_config_sha256: str = Field(min_length=64, max_length=64)
    multitenant: bool
    encoder_authority: PublicationEncoderAuthority | None = None
    encoder_receipts: tuple[PublicationEncoderReceipt, ...] = ()

    @model_validator(mode="after")
    def validate_receipts(self) -> Self:
        if self.encoder_authority is None:
            if self.encoder_receipts:
                raise ValueError("encoder receipts require compatible authority")
            return self
        authority = self.encoder_authority
        if (
            authority.provider or "",
            authority.model,
            authority.effective_dimension,
        ) != (self.model_provider, self.model_name, self.vector_dimension):
            raise ValueError("index encoder authority mismatch")
        representative = next(
            (
                receipt
                for receipt in self.encoder_receipts
                if receipt.authority == authority
            ),
            None,
        )
        if representative is None or any(
            receipt.effective_authority() != representative.effective_authority()
            for receipt in self.encoder_receipts
        ):
            raise ValueError("incompatible accepted encoder receipt")
        if self.embedding_config_sha256 not in {
            publication_digest(json.loads(receipt.configuration_json))
            for receipt in self.encoder_receipts
        }:
            raise ValueError("default encoder configuration receipt missing")
        return self

    def effective_authority(self) -> "EffectiveEncoderAuthority | None":
        representative = next(
            (
                receipt
                for receipt in self.encoder_receipts
                if receipt.authority == self.encoder_authority
            ),
            None,
        )
        return (
            representative.effective_authority() if representative is not None else None
        )

    def temporal_lookup_identity(self) -> str:
        """Physical/model selector remains stable as approved formatter receipts grow."""
        return publication_digest(
            self.model_dump(
                mode="json",
                exclude={
                    "embedding_config_sha256",
                    "encoder_authority",
                    "encoder_receipts",
                },
            )
        )

    def temporal_lookup_identities(self) -> tuple[str, ...]:
        """Retain stored hashes; recognize only proven OpenRouter enum/value spelling."""
        from shared_configs.enums import EmbeddingProvider

        identity = self.temporal_lookup_identity()
        effective = self.effective_authority()
        spellings = (
            EmbeddingProvider.OPENROUTER.value,
            str(EmbeddingProvider.OPENROUTER),
        )
        if (
            effective is None
            or effective.version != "openrouter-native-v1"
            or self.model_provider not in spellings
            or (self.model_name, self.vector_dimension)
            != (effective.authority.model, effective.authority.effective_dimension)
        ):
            return (identity,)
        return tuple(
            dict.fromkeys(
                [
                    identity,
                    *[
                        self.model_copy(
                            update={"model_provider": spelling}
                        ).temporal_lookup_identity()
                        for spelling in spellings
                    ],
                ]
            )
        )

    def matches_temporal_index(self, other: "PublicationIndexSnapshot") -> bool:
        return bool(
            set(self.temporal_lookup_identities()).intersection(
                other.temporal_lookup_identities()
            )
        )

    def accepts_encoder_configuration(self, configuration: JsonValue) -> bool:
        digest = publication_digest(configuration)
        if self.encoder_authority is None:
            return digest == self.embedding_config_sha256
        return any(
            json.loads(receipt.configuration_json) == configuration
            for receipt in self.encoder_receipts
        )


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
