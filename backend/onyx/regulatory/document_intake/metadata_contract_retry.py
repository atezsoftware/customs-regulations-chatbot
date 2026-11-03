"""Frozen, owner-authorized repair of specifically disproved metadata receipts."""

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.regulatory.document_intake.metadata import (
    LegalMetadataFact,
    LegalMetadataResult,
    MetadataChunk,
    build_metadata_evidence_catalog,
    build_metadata_output_reserve_request,
    build_metadata_request,
    clause_body_identity,
    metadata_evidence_scope,
    metadata_input_hash,
    validate_metadata_transport_response,
)
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    metadata_digest,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest

CONTRACT_RETRY_KEY = "metadata_contract_retry"
CONTRACT_RETRY_POLICY = "own_body_clause_prefix_or_omission"
MAX_CONTRACT_TARGETS = 16
MAX_CONTRACT_BYTES = 16 * 1024 * 1024
ClauseFailureReason = Literal[
    "selected_clause_contradiction", "target_clause_prefix_contradiction"
]


class ContractInvalidClause(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: Literal["clause_label"] = "clause_label"
    reason: ClauseFailureReason
    previous_value: str
    previous_fact: dict[str, Any]
    previous_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_base_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_repair_attempt: int = Field(ge=0, le=1)
    previous_raw_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_cache_file_id: str
    before_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pending_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContractRetryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_file_id: UUID
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preserved_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: list[MetadataChunk] = Field(min_length=1)
    targets: dict[str, ContractInvalidClause] = Field(min_length=1)
    request_hashes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_target_scope(self) -> "ContractRetryDocument":
        ids = [chunk.chunk_id for chunk in self.chunks]
        if (
            len(ids) != len(set(ids))
            or not set(self.targets) <= set(ids)
            or set(self.request_hashes) != set(self.targets)
            or metadata_input_hash(self.chunks) != self.input_sha256
        ):
            raise ValueError("Contract retry source/target request scope changed")
        return self


class ContractAppliedClause(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    intent_id: UUID
    intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    omitted: bool
    policy: Literal["own_body_clause_prefix_or_omission"]
    output_reserve: bool = False
    validation_feedback: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class MetadataContractRetry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    policy: Literal["own_body_clause_prefix_or_omission"] = CONTRACT_RETRY_POLICY
    id: UUID
    intake_id: UUID
    package_id: UUID
    owner_id: UUID
    previous_labeling_run_id: UUID
    previous_checkpoint: MetadataCheckpoint
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_metadata_sha256: dict[str, str]
    documents: dict[str, ContractRetryDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_frozen_requests(self) -> "MetadataContractRetry":
        self.previous_checkpoint.validate_checksum()
        source = self.previous_checkpoint.source_snapshot
        if (
            source.get("intake_id") != str(self.intake_id)
            or source.get("package_id") != str(self.package_id)
            or source.get("created_by") != str(self.owner_id)
            or not set(self.documents) <= set(source["documents"])
            or any(
                str(document.user_file_id) != source["documents"][key]["user_file_id"]
                or document.input_sha256 != source["documents"][key]["input_sha256"]
                or document.canonical_source_sha256
                != source["documents"][key]["canonical_source_sha256"]
                for key, document in self.documents.items()
            )
        ):
            raise ValueError(
                "Contract retry intake, owner or document identity changed"
            )
        if set(self.scope_metadata_sha256) != set(
            self.previous_checkpoint.source_snapshot["documents"]
        ):
            raise ValueError("Contract retry full metadata ownership scope changed")
        targets = [target for doc in self.documents.values() for target in doc.targets]
        if not 1 <= len(targets) <= MAX_CONTRACT_TARGETS or len(set(targets)) != len(
            targets
        ):
            raise ValueError("Contract retry target limit or uniqueness changed")
        for key, document in self.documents.items():
            for target in document.targets:
                if (
                    document.targets[target].previous_checkpoint_request_hash
                    not in self.previous_checkpoint.request_hashes[key]
                ):
                    raise ValueError(
                        "Contract retry prior ACK left its checkpoint request ancestry"
                    )
                if (
                    build_contract_retry_request(
                        self.id, document.chunks, target
                    ).request_hash
                    != document.request_hashes[target]
                ):
                    raise ValueError("Contract retry frozen request identity changed")
        if len(self.model_dump_json().encode()) > MAX_CONTRACT_BYTES:
            raise ValueError("Contract retry proof exceeds its bounded size")
        return self

    @property
    def intent_sha256(self) -> str:
        return metadata_digest(self.model_dump(mode="json"))


def metadata_ack_file_id(
    *,
    tenant_id: str,
    intake_id: UUID,
    document_id: UUID,
    binding_sha256: str,
    request_hash: str,
) -> str:
    identity = f"metadata-v2:{tenant_id}:{intake_id}:{document_id}:{binding_sha256}"
    submission = hashlib.sha256(f"{identity}:{request_hash}".encode()).hexdigest()
    return (
        f"regulatory-labeling-direct/{binding_sha256}/{submission}/{request_hash}.json"
    )


def build_contract_retry_request(
    contract_id: UUID, chunks: list[MetadataChunk], target_id: str
) -> VertexBatchRequest:
    base = build_metadata_request(chunks, [target_id])
    payload = json.loads(base.prompt)
    payload[CONTRACT_RETRY_KEY] = {
        "version": 1,
        "id": str(contract_id),
        "policy": CONTRACT_RETRY_POLICY,
        "target_chunk_id": target_id,
        "fields": ["clause_label"],
        "instruction": (
            "Review only clause_label for this exact target. Select BODY evidence from "
            "the target itself, beginning with its own unambiguous formal clause marker. "
            "Its value must agree with the target body's own starting marker. Inherited "
            "headings and other chunks cannot support this repaired field. If the target "
            "has no own unambiguous clause prefix, return no fields for it. Never invent "
            "a prefix, quote, evidence ID or replacement. All normal strict checks apply."
        ),
    }
    config = deepcopy(base.generation_config)
    assert isinstance(config, dict)
    schema = config["responseJsonSchema"]
    assert isinstance(schema, dict)
    schema["properties"]["chunks"]["items"]["properties"]["fields"]["items"][
        "properties"
    ]["field"]["enum"] = ["clause_label"]
    request = VertexBatchRequest(
        prompt=json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        system_instruction=base.system_instruction,
        generation_config=config,
    )
    if (
        len(
            json.dumps(
                request.to_generate_content_request(), ensure_ascii=False
            ).encode()
        )
        > 8 * 1024 * 1024
    ):
        raise ValueError("Contract retry request exceeds its frozen byte budget")
    return request


def require_contract_base_request(
    request: VertexBatchRequest, *, chunks: list[MetadataChunk], target_id: str
) -> None:
    payload = json.loads(request.prompt)
    contract_id = UUID(payload[CONTRACT_RETRY_KEY]["id"])
    expected = build_contract_retry_request(contract_id, chunks, target_id)
    if request != expected:
        raise ValueError("Contract retry requires its exact frozen policy request")


def require_contract_accepted_request(
    request: VertexBatchRequest,
    *,
    contract_id: UUID,
    chunks: list[MetadataChunk],
    target_id: str,
) -> None:
    base = build_contract_retry_request(contract_id, chunks, target_id)
    reserved = build_metadata_output_reserve_request(base)
    if request == base or request == reserved:
        return
    payload, original = json.loads(request.prompt), json.loads(base.prompt)
    feedback = payload.pop("validation_feedback", None)
    if (
        payload != original
        or not isinstance(feedback, dict)
        or feedback.get("original_request_hash")
        not in {
            base.request_hash,
            reserved.request_hash if reserved is not None else None,
        }
        or feedback.get("repair_attempt") != 1
        or request.system_instruction != base.system_instruction
        or request.generation_config
        not in [
            base.generation_config,
            reserved.generation_config if reserved else None,
        ]
    ):
        raise ValueError("Contract retry accepted request left its declared lineage")


def validate_contract_retry_result(
    result: LegalMetadataResult, *, chunks: list[MetadataChunk], target_id: str
) -> None:
    if (
        len(result.chunks) != 1
        or result.chunks[0].chunk_id != target_id
        or len(result.chunks[0].fields) > 1
    ):
        raise ValueError("Contract retry result changed its single target")
    source = next((chunk for chunk in chunks if chunk.chunk_id == target_id), None)
    if source is None:
        raise ValueError("Contract retry target source is absent")
    for fact in result.chunks[0].fields:
        validate_repaired_clause_fact(fact, source=source)


def validate_repaired_clause_fact(
    fact: LegalMetadataFact, *, source: MetadataChunk
) -> None:
    own = clause_body_identity(source.text)
    if (
        fact.field != "clause_label"
        or fact.source_chunk_id != source.chunk_id
        or metadata_evidence_scope(fact, {source.chunk_id: source}) != "body"
        or own is None
        or not source.text.lstrip().startswith(fact.evidence_quote.lstrip())
        or clause_body_identity(fact.evidence_quote) != own
        or not isinstance(fact.value, str)
        or fact.value.replace("I", "ı").replace("İ", "i").lower().strip("() ") != own
    ):
        raise ValueError(
            "Contract repaired clause requires its own BODY prefix or omission"
        )


def validate_contract_retry_response(
    response: str, *, chunks: list[MetadataChunk], target_id: str
) -> LegalMetadataResult:
    result = validate_metadata_transport_response(
        response, chunks=chunks, target_ids=[target_id]
    )
    catalog = {
        entry.evidence_id: entry for entry in build_metadata_evidence_catalog(chunks)
    }
    for target in json.loads(response)["chunks"]:
        for fact in target["fields"]:
            selected = catalog[fact["evidence_id"]]
            if selected.scope != "body" or selected.source_chunk_id != target_id:
                raise ValueError(
                    "Contract repaired clause requires its own BODY prefix or omission"
                )
    validate_contract_retry_result(result, chunks=chunks, target_id=target_id)
    return result
