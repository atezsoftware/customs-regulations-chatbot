"""Metadata writes are limited to unpublished documents owned by one intake."""

import datetime
import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata_checkpoint import require_metadata_lease
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
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    MetadataChunk,
    metadata_evidence_scope,
    metadata_input_hash,
    validate_metadata_response,
)
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from onyx.regulatory.source_document_identity import (
    DocumentIdentityEvidence,
    derive_document_identity,
)

_AUTOMATED_SOURCES = frozenset({"llm", "source_opening"})


def _own_opening_identity(
    rows: Sequence[RegulatoryChunk],
) -> DocumentIdentityEvidence | None:
    first = next(
        (row for row in rows if not is_hierarchical_aggregate_chunk(row)), None
    )
    if first is None or first.position != 0 or not first.heading_path:
        return None
    return derive_document_identity(
        user_file_id=str(first.user_file_id),
        source_chunk_id=first.id,
        text=first.text,
        root_heading=first.heading_path[0],
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
            if (
                not isinstance(entry, dict)
                or entry.get("source") not in _AUTOMATED_SOURCES
            ):
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


def _stale_model_fields(row: RegulatoryChunk) -> set[str]:
    audit = row.chunk_metadata.get(METADATA_AUDIT_KEY)
    fields = audit.get("fields", {}) if isinstance(audit, dict) else {}
    if not isinstance(fields, dict):
        return set()
    return {
        key
        for key, entry in fields.items()
        if isinstance(entry, dict)
        and entry.get("source") in _AUTOMATED_SOURCES
        and entry.get("stale") is True
    }


def metadata_receipts_current(rows: Sequence[RegulatoryChunk], input_hash: str) -> bool:
    canonical = [row for row in rows if not is_hierarchical_aggregate_chunk(row)]
    return bool(canonical) and all(
        isinstance(audit := row.chunk_metadata.get(METADATA_AUDIT_KEY), dict)
        and audit.get("model") == METADATA_MODEL
        and audit.get("version") == METADATA_VERSION
        and audit.get("input_sha256") == input_hash
        and audit.get("input_stale") is False
        and isinstance(audit.get("fields"), dict)
        and all(
            isinstance(entry, dict)
            and (
                entry.get("source") not in _AUTOMATED_SOURCES
                or entry.get("stale") is False
            )
            for entry in audit["fields"].values()
        )
        for row in canonical
    )


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
                if key in LEGAL_METADATA_FIELDS and key not in _stale_model_fields(row)
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
    require_metadata_lease(
        session, intake_id, lease_token, allow_preparation_stage=True
    )
    batch, _package, _assets = intake.load_intake_job(session, intake_id, lease_token)
    batch.stage = "metadata"
    batch.lease_expires_at = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=10)
    from onyx.db.document_intake_metadata_review import (
        require_reviewed_document,
        reviewed_metadata_edit,
    )
    from onyx.db.document_intake_review import effective_intake_files

    reviewed = reviewed_metadata_edit(batch)
    if (
        reviewed is not None
        and reviewed.previous_labeling_run_id != batch.labeling_run_id
    ):
        reviewed = None

    members = (
        effective_intake_files(session, batch)
        if batch.labeling_run_id is not None
        else intake.intake_files(session, batch)
    )
    work = []
    for member in members:
        rows = _load_rows(session, batch, member.id)
        chunks = metadata_chunks(rows)
        if not chunks:
            raise ValueError("Metadata requires at least one canonical source chunk")
        input_hash = metadata_input_hash(chunks)
        complete = metadata_receipts_current(rows, input_hash)
        authorized = (
            reviewed is not None
            and member.id in reviewed.documents
            and reviewed.documents[member.id].refresh_metadata
        )
        if authorized and not complete:
            assert reviewed is not None
            require_reviewed_document(
                reviewed.documents[member.id], member.user_file_id, rows
            )
        else:
            try:
                validate_metadata_receipt_evidence(rows, chunks)
            except ValueError as error:
                if batch.labeling_run_id is not None:
                    raise intake.IntakeConflictError(
                        "Saved locator evidence changed its validation contract; review this import before labeling or approval"
                    ) from error
                complete = False
        if batch.labeling_run_id is not None and not complete and not authorized:
            raise intake.IntakeConflictError(
                "Unreviewed post-label metadata cannot be regenerated"
            )
        # Label retries and manually edited reviews retain their reviewed metadata.
        if complete:
            continue
        work.append((member.id, chunks))
    return batch, work


def validate_metadata_receipt_evidence(
    rows: Sequence[RegulatoryChunk], chunks: Sequence[MetadataChunk]
) -> None:
    own_identity = _own_opening_identity(rows)
    for row in rows:
        if is_hierarchical_aggregate_chunk(row):
            continue
        audit = row.chunk_metadata.get(METADATA_AUDIT_KEY)
        fields = audit.get("fields") if isinstance(audit, dict) else None
        if not isinstance(fields, dict):
            continue
        for field, entry in fields.items():
            if not isinstance(entry, dict) or entry.get("source") != "source_opening":
                continue
            if (
                own_identity is None
                or field not in own_identity.fields
                or entry.get("source_chunk_id") != own_identity.source_chunk_id
                or entry.get("source_sha256") != own_identity.source_sha256
                or row.chunk_metadata.get(field) != own_identity.fields[field]
            ):
                raise ValueError(
                    "Saved own document identity differs from its source opening"
                )
        facts = []
        for field in ("article_no", "paragraph_no", "clause_label"):
            entry = fields.get(field)
            if (
                not isinstance(entry, dict)
                or entry.get("source") not in _AUTOMATED_SOURCES
            ):
                continue
            facts.append(
                LegalMetadataFact.model_validate(
                    {
                        "field": field,
                        "value": row.chunk_metadata.get(field),
                        "source_chunk_id": entry.get("source_chunk_id"),
                        "evidence_quote": entry.get("evidence_quote"),
                        "basis": entry.get("basis", "explicit"),
                    }
                )
            )
        if not facts:
            continue
        result = LegalMetadataResult(
            chunks=[ChunkLegalMetadata(chunk_id=row.id, fields=facts)]
        )
        validate_metadata_response(
            result.model_dump_json(), chunks=chunks, target_ids=[row.id]
        )
        if isinstance(audit, dict) and "metadata_contract_retry" in audit:
            from onyx.regulatory.document_intake.metadata_contract_retry import (
                ContractAppliedClause,
                validate_repaired_clause_fact,
            )

            stamp = ContractAppliedClause.model_validate(
                audit["metadata_contract_retry"]
            )
            clause = next(
                (fact for fact in facts if fact.field == "clause_label"), None
            )
            if stamp.omitted:
                if clause is not None or "clause_label" in row.chunk_metadata:
                    raise ValueError("An omitted contract clause was restored")
            elif clause is None:
                raise ValueError("The contract repaired clause receipt is missing")
            else:
                if fields["clause_label"].get("evidence_scope") != "body":
                    raise ValueError(
                        "Contract clause receipt requires actual BODY scope"
                    )
                validate_repaired_clause_fact(
                    clause, source=next(c for c in chunks if c.chunk_id == row.id)
                )


def validate_intake_metadata_receipts(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    verify_contract_ack_bytes: bool = True,
) -> None:
    from onyx.db.document_intake_metadata_contract_retry import (
        contract_retry,
        require_current_contract,
    )

    if contract_retry(batch) is not None:
        require_current_contract(
            session,
            batch,
            require_complete=True,
            verify_ack_bytes=verify_contract_ack_bytes,
            require_fresh_labeling=True,
        )
    from onyx.db.document_intake_review import effective_intake_files

    for member in effective_intake_files(session, batch):
        rows = _load_rows(session, batch, member.id)
        try:
            validate_metadata_receipt_evidence(rows, metadata_chunks(rows))
        except ValueError as error:
            raise intake.IntakeConflictError(
                "Saved locator evidence failed source validation; retry or correct this import before approval"
            ) from error


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
    batch = require_metadata_lease(session, intake_id, lease_token)
    batch.lease_expires_at = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=10)
    session.flush()


def heartbeat_metadata_with_timeout(
    session: Session, intake_id: UUID, lease_token: UUID
) -> None:
    """Bound an isolated renewal transaction before checking its existing owner."""
    try:
        session.execute(text("SET LOCAL lock_timeout = '5s'"))
        session.execute(text("SET LOCAL statement_timeout = '10s'"))
        heartbeat_metadata(session, intake_id, lease_token)
    except (OperationalError, SQLAlchemyTimeoutError) as error:
        raise MetadataLeaseLost(
            "Metadata heartbeat ownership could not be renewed"
        ) from error


def apply_metadata_results(
    session: Session,
    *,
    intake_id: UUID,
    lease_token: UUID,
    document_id: UUID,
    expected_input_sha256: str,
    results: Sequence[tuple[str, LegalMetadataResult]],
) -> None:
    require_metadata_lease(session, intake_id, lease_token)
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
    own_identity = _own_opening_identity(rows)
    for request_hash, result in results:
        validate_metadata_response(
            result.model_dump_json(),
            chunks=chunks,
            target_ids=[target.chunk_id for target in result.chunks],
        )
        for target in result.chunks:
            row = by_id[target.chunk_id]
            metadata, audit = deepcopy(row.chunk_metadata), _audit(row)
            for field, entry in list(audit["fields"].items()):
                if (
                    entry.get("source") in _AUTOMATED_SOURCES
                    and entry.get("stale") is True
                ):
                    audit["fields"].pop(field)
                    if field in VALIDITY_FIELDS:
                        setattr(row, field, None)
                    elif field in LEGAL_METADATA_FIELDS:
                        metadata.pop(field, None)
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
            if own_identity is not None:
                for field, value in own_identity.fields.items():
                    if audit["fields"].get(field, {}).get("source") == "manual":
                        continue
                    if value is None:
                        metadata.pop(field, None)
                    else:
                        metadata[field] = value
                    audit["fields"][field] = {
                        "source": "source_opening",
                        "basis": "own_source_opening",
                        "source_chunk_id": own_identity.source_chunk_id,
                        "source_sha256": own_identity.source_sha256,
                        "evidence_quote": value,
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
