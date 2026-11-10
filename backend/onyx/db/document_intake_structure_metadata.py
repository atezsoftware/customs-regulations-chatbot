"""Persist explicit local source verification without inventing provider receipts."""

from collections.abc import Sequence
from copy import deepcopy
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.regulatory.document_intake import metadata
from onyx.regulatory.document_intake.metadata_checkpoint import metadata_digest
from onyx.regulatory.document_intake.metadata_structure import (
    CORE_LOCATORS,
    STRUCTURAL_FIELDS,
    VerifiedStructuralFact,
    positive_locator_evidence,
)


def record_structural_verification(
    session: Session,
    *,
    user_file_id: UUID,
    results: Sequence[tuple[str, metadata.LegalMetadataResult]],
    input_sha256: str,
) -> None:
    rows = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
        )
    }
    for _request_hash, result in results:
        for target in result.chunks:
            row = rows[target.chunk_id]
            stored = deepcopy(row.chunk_metadata)
            audit = stored[metadata.METADATA_AUDIT_KEY]
            proofs = {}
            for fact in target.fields:
                if (
                    not isinstance(fact, VerifiedStructuralFact)
                    or audit["fields"].get(fact.field, {}).get("source") == "manual"
                ):
                    continue
                witness = {
                    "target_chunk_id": row.id,
                    **fact.model_dump(mode="json"),
                    "input_sha256": input_sha256,
                }
                proofs[fact.field] = {
                    "witness": witness,
                    "sha256": metadata_digest(witness),
                }
                field = audit["fields"][fact.field]
                field["source"] = "structural"
                field["verified_by"] = "strict_source_validator_v1"
                field.pop("request_hash", None)
            audit["structural_verification"] = {
                "version": 1,
                "input_sha256": input_sha256,
                "fields": proofs,
            }
            row.chunk_metadata = stored
    session.flush()


def validate_structural_verification(
    rows: Sequence[RegulatoryChunk],
    chunks: Sequence[metadata.MetadataChunk],
) -> None:
    input_sha256 = metadata.metadata_input_hash(chunks)
    source_ids = {chunk.chunk_id for chunk in chunks}
    catalog = metadata.build_metadata_evidence_catalog(chunks)
    for row in rows:
        if row.id not in source_ids:
            continue
        audit = row.chunk_metadata.get(metadata.METADATA_AUDIT_KEY, {})
        proof = audit.get("structural_verification")
        if proof is None:
            continue
        if proof.get("version") != 1 or proof.get("input_sha256") != input_sha256:
            raise ValueError("Local structural verification source changed")
        for field, entry in proof["fields"].items():
            if audit["fields"].get(field, {}).get("source") == "manual":
                continue
            witness = dict(entry["witness"])
            if (
                entry["sha256"] != metadata_digest(witness)
                or witness.pop("target_chunk_id") != row.id
                or witness.pop("input_sha256") != input_sha256
            ):
                raise ValueError("Local structural verification witness changed")
            fact = metadata.LegalMetadataFact.model_validate(witness)
            if row.chunk_metadata.get(field) != fact.value or fact.field != field:
                raise ValueError("Local structural verification value changed")
            if field not in STRUCTURAL_FIELDS or not any(
                entry.source_chunk_id == row.id
                and entry.text == fact.evidence_quote
                and positive_locator_evidence(field, fact.value, entry)
                and (
                    field in CORE_LOCATORS
                    or (
                        entry.scope == "heading"
                        and metadata._fold(" ".join(str(fact.value).split()))
                        == metadata._fold(" ".join(entry.text.split()))
                    )
                )
                for entry in catalog
            ):
                raise ValueError(
                    "Local structural verification has no positive source marker"
                )
            result = metadata.LegalMetadataResult(
                chunks=[metadata.ChunkLegalMetadata(chunk_id=row.id, fields=[fact])]
            )
            metadata.validate_metadata_response(
                result.model_dump_json(), chunks=chunks, target_ids=[row.id]
            )
