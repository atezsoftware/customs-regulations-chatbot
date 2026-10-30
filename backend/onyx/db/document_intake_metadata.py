"""Metadata writes are limited to unpublished documents owned by one intake."""

import datetime
import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_chunks import is_hierarchical_aggregate_chunk
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.document_names import (
    UNKNOWN_DOCUMENT_TITLE,
    intake_document_name,
)
from onyx.regulatory.document_intake.metadata import (
    DATE_FIELDS,
    INTEGER_FIELDS,
    LEGAL_METADATA_FIELDS,
    METADATA_AUDIT_KEY,
    METADATA_MODEL,
    METADATA_VERSION,
    VALIDITY_FIELDS,
    LegalMetadataResult,
    MetadataChunk,
    metadata_evidence_scope,
    metadata_input_hash,
    validate_metadata_response,
)


def _audit(chunk: RegulatoryChunk) -> dict[str, Any]:
    existing = chunk.chunk_metadata.get(METADATA_AUDIT_KEY)
    if isinstance(existing, dict):
        return deepcopy(existing)
    fields = {
        key: {
            "source": "structural",
            "basis": "chunker",
            "evidence_quote": None,
            "source_chunk_id": chunk.id,
            "stale": False,
        }
        for key in LEGAL_METADATA_FIELDS
        if chunk.chunk_metadata.get(key) is not None
    }
    return {
        "version": METADATA_VERSION,
        "model": None,
        "request_hash": None,
        "input_sha256": None,
        "fields": fields,
        "warnings": [],
        "input_stale": False,
    }


def manual_edit_metadata(
    chunk: RegulatoryChunk, updates: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a legal metadata patch and retain protected, server-owned provenance."""
    normalized = deepcopy(dict(updates))
    metadata = deepcopy(chunk.chunk_metadata)
    audit = _audit(chunk)
    fields = audit["fields"]
    changed = False
    if "chunk_metadata" in updates:
        patch = updates["chunk_metadata"]
        if (
            not isinstance(patch, dict)
            or len(patch) > len(LEGAL_METADATA_FIELDS)
            or set(patch) - LEGAL_METADATA_FIELDS
        ):
            raise ValueError(
                "Only legal chunk metadata fields can be edited; source, graph and audit fields are protected"
            )
        for key, value in patch.items():
            if value is not None:
                if key in INTEGER_FIELDS:
                    if type(value) is not int or value < 0:
                        raise ValueError(
                            "Table indexes must be nonnegative integers or null"
                        )
                elif (
                    not isinstance(value, str) or not value.strip() or len(value) > 2000
                ):
                    raise ValueError(
                        "Legal metadata must be a nonempty string of at most 2000 characters, or null"
                    )
                if key in DATE_FIELDS:
                    date = datetime.date.fromisoformat(value)
                    if value != date.isoformat():
                        raise ValueError("Metadata dates must use ISO YYYY-MM-DD")
            if metadata.get(key) == value:
                continue
            if value is None:
                metadata.pop(key, None)
            else:
                metadata[key] = value
            fields[key] = {
                "source": "manual",
                "basis": "user_review",
                "evidence_quote": None,
                "source_chunk_id": chunk.id,
                "stale": False,
            }
            changed = True
    for key in VALIDITY_FIELDS:
        if key in updates and updates[key] != getattr(chunk, key):
            fields[key] = {
                "source": "manual",
                "basis": "user_review",
                "evidence_quote": None,
                "source_chunk_id": chunk.id,
                "stale": False,
            }
            changed = True
    if "heading_path" in updates and updates["heading_path"] != chunk.heading_path:
        metadata["heading_path"] = deepcopy(updates["heading_path"])
        fields["heading_path"] = {
            "source": "manual",
            "basis": "user_review",
            "evidence_quote": None,
            "source_chunk_id": chunk.id,
            "stale": False,
        }
        changed = True
    if changed:
        metadata[METADATA_AUDIT_KEY] = audit
        normalized["chunk_metadata"] = metadata
    else:
        normalized.pop("chunk_metadata", None)
    return normalized


def invalidate_metadata_file_context(rows: Sequence[RegulatoryChunk]) -> None:
    """After a context edit, inferred dates lose authority; manual values win."""
    for row in rows:
        if not isinstance(row.chunk_metadata.get(METADATA_AUDIT_KEY), dict):
            continue
        audit, metadata = _audit(row), deepcopy(row.chunk_metadata)
        audit["input_stale"] = True
        for key, entry in audit["fields"].items():
            if not isinstance(entry, dict) or entry.get("source") != "llm":
                continue
            entry["stale"] = True
            if key in VALIDITY_FIELDS:
                setattr(row, key, None)
            elif key in DATE_FIELDS:
                metadata.pop(key, None)
        warning = "Document context changed after metadata extraction; model evidence is stale and inferred dates were cleared. Manually reviewed values are retained."
        audit["warnings"] = list(dict.fromkeys([*audit.get("warnings", []), warning]))[
            :32
        ]
        metadata[METADATA_AUDIT_KEY] = audit
        row.chunk_metadata = metadata


def metadata_chunks(rows: Sequence[RegulatoryChunk]) -> list[MetadataChunk]:
    return [
        MetadataChunk(
            chunk_id=row.id,
            text=row.text,
            heading_path=list(row.heading_path),
            chunk_type=row.chunk_type,
            structural_metadata={
                key: deepcopy(value)
                for key, value in row.chunk_metadata.items()
                if key in LEGAL_METADATA_FIELDS
            },
        )
        for row in rows
        if not is_hierarchical_aggregate_chunk(row)
    ]


def _load_rows(
    session: Session, batch: RegulatoryDocumentIntake, document_id: UUID
) -> list[RegulatoryChunk]:
    member = session.get_one(RegulatoryDocumentIntakeFile, document_id)
    file = session.get_one(UserFile, member.user_file_id)
    if member.intake_id != batch.id or file.status != UserFileStatus.CHUNKED:
        raise intake.IntakeConflictError(
            "Metadata extraction can change only this intake's unpublished files"
        )
    return list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == file.id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        )
    )


def load_metadata_work(
    session: Session, intake_id: UUID, lease_token: UUID
) -> tuple[RegulatoryDocumentIntake, list[tuple[UUID, list[MetadataChunk]]]]:
    batch, _package, _assets = intake.load_intake_job(session, intake_id, lease_token)
    batch.stage = "metadata"
    batch.lease_expires_at = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=10)
    # Label retries and manually edited reviews never overwrite already reviewed metadata.
    if batch.labeling_run_id is not None:
        return batch, []
    work = []
    for member in intake.intake_files(session, batch):
        rows = _load_rows(session, batch, member.id)
        chunks = metadata_chunks(rows)
        if not chunks:
            raise ValueError("Metadata requires at least one canonical source chunk")
        input_hash = metadata_input_hash(chunks)
        if all(
            isinstance(row.chunk_metadata.get(METADATA_AUDIT_KEY), dict)
            and row.chunk_metadata[METADATA_AUDIT_KEY].get("input_sha256") == input_hash
            for row in rows
            if not is_hierarchical_aggregate_chunk(row)
        ):
            continue
        work.append((member.id, chunks))
    return batch, work


def refresh_intake_document_name(
    session: Session, batch: RegulatoryDocumentIntake, document_id: UUID
) -> None:
    """Rename only the unpublished file owned by this intake; source assets stay immutable."""
    rows = _load_rows(session, batch, document_id)
    member = session.get_one(RegulatoryDocumentIntakeFile, document_id)
    file = session.get_one(UserFile, member.user_file_id)
    asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
    if (
        asset.package_id != batch.source_package_id
        or not asset.text_file_id
        or not asset.text_sha256
    ):
        raise intake.IntakeConflictError(
            "The document title requires its frozen source"
        )
    store = get_default_file_store()
    stream = (
        store.read_file(asset.text_file_id, db_session=session)
        if isinstance(store, PostgresBackedFileStore)
        else store.read_file(asset.text_file_id)
    )
    with stream:
        prepared = stream.read()
    if hashlib.sha256(prepared).hexdigest() != asset.text_sha256:
        raise intake.IntakeConflictError("The document title source integrity changed")
    file.name = (
        intake_document_name(
            [
                row.chunk_metadata
                for row in rows
                if not is_hierarchical_aggregate_chunk(row)
            ],
            source_text=prepared.decode("utf-8")
            + "\n"
            + "\n".join(row.text for row in rows),
        )
        or UNKNOWN_DOCUMENT_TITLE
    )
    first = next(row for row in rows if not is_hierarchical_aggregate_chunk(row))
    first.chunk_metadata = {
        **first.chunk_metadata,
        "intake_display_name": {"version": 1, "title": file.name},
    }


def heartbeat_metadata(session: Session, intake_id: UUID, lease_token: UUID) -> None:
    batch, _package, _assets = intake.load_intake_job(session, intake_id, lease_token)
    batch.lease_expires_at = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=10)
    session.flush()


def apply_metadata_results(
    session: Session,
    *,
    intake_id: UUID,
    lease_token: UUID,
    document_id: UUID,
    expected_input_sha256: str,
    results: Sequence[tuple[str, LegalMetadataResult]],
) -> None:
    batch, _package, _assets = intake.load_intake_job(session, intake_id, lease_token)
    rows = _load_rows(session, batch, document_id)
    chunks = metadata_chunks(rows)
    if metadata_input_hash(chunks) != expected_input_sha256:
        raise intake.IntakeConflictError(
            "Canonical document changed during metadata extraction"
        )
    by_id = {row.id: row for row in rows if not is_hierarchical_aggregate_chunk(row)}
    evidence_sources = {chunk.chunk_id: chunk for chunk in chunks}
    received = [
        target.chunk_id for _request_hash, result in results for target in result.chunks
    ]
    if len(set(received)) != len(received) or set(received) != set(by_id):
        raise ValueError("Metadata windows do not cover the exact document")
    for request_hash, result in results:
        validate_metadata_response(
            result.model_dump_json(),
            chunks=chunks,
            target_ids=[target.chunk_id for target in result.chunks],
        )
        for target in result.chunks:
            row = by_id[target.chunk_id]
            metadata, audit = deepcopy(row.chunk_metadata), _audit(row)
            for fact in target.fields:
                if audit["fields"].get(fact.field, {}).get("source") == "manual":
                    continue
                if fact.field in VALIDITY_FIELDS:
                    setattr(
                        row, fact.field, datetime.date.fromisoformat(str(fact.value))
                    )
                else:
                    metadata[fact.field] = fact.value
                audit["fields"][fact.field] = {
                    "source": "llm",
                    "request_hash": fact.retained_request_hash or request_hash,
                    **(
                        {"retained_from_validated_fact": True}
                        if fact.retained_request_hash
                        else {}
                    ),
                    "basis": fact.basis,
                    "evidence_quote": fact.evidence_quote,
                    "source_chunk_id": fact.source_chunk_id,
                    "evidence_scope": metadata_evidence_scope(fact, evidence_sources),
                    "reference_quote": fact.reference_quote,
                    "reference_chunk_id": fact.reference_chunk_id,
                    "stale": False,
                }
            if (
                row.validity_start_date
                and row.validity_end_date
                and row.validity_start_date >= row.validity_end_date
            ):
                raise ValueError("Metadata validity end must be after its start")
            audit.update(
                {
                    "version": METADATA_VERSION,
                    "model": METADATA_MODEL,
                    "request_hash": request_hash,
                    "input_sha256": expected_input_sha256,
                    "warnings": list(result.warnings),
                    "input_stale": False,
                }
            )
            if any(fact.retained_request_hash for fact in target.fields):
                audit["composed_from_validated_facts"] = True
            else:
                audit.pop("composed_from_validated_facts", None)
            metadata[METADATA_AUDIT_KEY] = audit
            row.chunk_metadata = metadata
    refresh_intake_document_name(session, batch, document_id)
    heartbeat_metadata(session, intake_id, lease_token)
    session.flush()
