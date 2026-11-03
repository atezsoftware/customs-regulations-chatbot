"""Freeze reviewed source edits before their metadata and labels are regenerated."""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
)
from onyx.regulatory.document_intake.metadata import (
    LEGAL_METADATA_FIELDS,
    METADATA_AUDIT_KEY,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    metadata_digest,
)
from onyx.regulatory.document_intake.metadata_review import (
    ReviewedMetadataDocument,
    ReviewedMetadataEdit,
)


def prelabel_metadata_edit_digest(
    session: Session, batch: RegulatoryDocumentIntake
) -> str | None:
    if (
        batch.status != "failed"
        or batch.stage not in {"metadata", "chunking"}
        or batch.labeling_run_id is not None
        or batch.lease_token is not None
        or batch.lease_expires_at is not None
        or labeling.get_active_run_id(session, batch.document_set_id) is not None
    ):
        return None
    from onyx.db.document_intake_metadata_checkpoint import _source_snapshot

    try:
        source, _requests, _completed = _source_snapshot(
            session, batch, for_review=True
        )
        canonical = intake.review_digest(session, batch)
    except (ValueError, intake.IntakeConflictError):
        return None
    return metadata_digest(
        {
            "purpose": "unpublished_prelabel_metadata_edit",
            "source": source,
            "canonical_review": canonical,
            "status": batch.status,
            "stage": batch.stage,
            "error": batch.error,
        }
    )


def canonical_metadata_source_digest(rows: Sequence[RegulatoryChunk]) -> str:
    return metadata_digest(
        [
            {
                "id": row.id,
                "text": row.text,
                "heading_path": row.heading_path,
                "chunk_type": row.chunk_type,
                "position": row.position,
                "source_metadata": {
                    key: value
                    for key, value in row.chunk_metadata.items()
                    if key
                    not in {
                        *LEGAL_METADATA_FIELDS,
                        METADATA_AUDIT_KEY,
                        "intake_display_name",
                    }
                },
            }
            for row in rows
        ]
    )


def reviewed_metadata_edit(
    batch: RegulatoryDocumentIntake,
) -> ReviewedMetadataEdit | None:
    if batch.metadata_checkpoint is None:
        return None
    saved = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    saved.validate_checksum()
    value = saved.source_snapshot.get("reviewed_edit")
    return ReviewedMetadataEdit.model_validate(value) if value is not None else None


def require_reviewed_metadata_owner(
    session: Session,
    batch: RegulatoryDocumentIntake,
    edit: ReviewedMetadataEdit,
    files: set[str],
) -> None:
    if (
        edit.intake_id != batch.id
        or edit.package_id != batch.source_package_id
        or edit.previous_labeling_run_id != batch.labeling_run_id
    ):
        raise intake.IntakeConflictError("The reviewed metadata owner changed")
    if edit.previous_labeling_run_id is not None:
        previous = session.get_one(
            RegulatoryLabelingRun, edit.previous_labeling_run_id, populate_existing=True
        )
        if (
            previous.status
            not in {"completed", "completed_with_errors", "failed", "cancelled"}
            or set(previous.file_ids) != files
            or previous.lease_token is not None
            and previous.lease_expires_at is not None
            and previous.lease_expires_at > datetime.now(timezone.utc)
        ):
            raise intake.IntakeConflictError(
                "The prior reviewed labeling owner is not idle"
            )
    active_id = labeling.get_active_run_id(session, batch.document_set_id)
    if active_id is not None:
        active = session.get_one(
            RegulatoryLabelingRun, active_id, populate_existing=True
        )
        if set(active.file_ids) & files:
            raise intake.IntakeConflictError(
                "Active labeling overlaps reviewed metadata"
            )


def require_reviewed_document(
    document: ReviewedMetadataDocument, file_id: UUID, rows: Sequence[RegulatoryChunk]
) -> None:
    from onyx.db.document_intake_metadata import metadata_chunks

    by_id = {row.id: row for row in rows}
    if (
        document.user_file_id != file_id
        or document.input_sha256 != metadata_input_hash(metadata_chunks(rows))
        or document.canonical_source_sha256 != canonical_metadata_source_digest(rows)
        or any(
            identifier not in by_id
            or by_id[identifier].chunk_metadata.get("intake_edit_revision")
            != str(revision)
            for identifier, revision in document.revisions.items()
        )
    ):
        raise intake.IntakeConflictError(
            "The reviewed metadata source or revision changed"
        )
    for row in rows:
        audit = row.chunk_metadata.get(METADATA_AUDIT_KEY)
        if not isinstance(audit, dict) or (
            audit.get("input_sha256") == document.input_sha256
            and audit.get("input_stale") is False
            and all(
                entry.get("source") != "llm" or entry.get("stale") is False
                for entry in audit.get("fields", {}).values()
            )
        ):
            continue
        if document.refresh_metadata and (
            audit.get("input_stale") is not True
            or any(
                entry.get("source") == "llm" and entry.get("stale") is not True
                for entry in audit.get("fields", {}).values()
            )
        ):
            raise intake.IntakeConflictError(
                "Reviewed source edits require stale model evidence"
            )


def capture_reviewed_metadata_checkpoint(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    expected_review_sha256: str,
    documents: dict[UUID, ReviewedMetadataDocument],
    binding: LabelingProviderBinding,
) -> None:
    from onyx.db.document_intake_metadata_checkpoint import _source_snapshot
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

    predecessor = (
        MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
        if batch.metadata_checkpoint
        else None
    )
    if predecessor is not None:
        predecessor.validate_checksum()
    edit = ReviewedMetadataEdit(
        intake_id=batch.id,
        package_id=batch.source_package_id,
        previous_labeling_run_id=batch.labeling_run_id,
        expected_review_sha256=expected_review_sha256,
        previous_checkpoint_sha256=predecessor.checkpoint_sha256
        if predecessor
        else None,
        documents=documents,
    )
    source, requests, _completed = _source_snapshot(session, batch, reviewed=edit)
    checkpoint = MetadataCheckpoint.build(
        recipe_sha256=metadata_recipe_sha256(),
        provider_binding=binding.model_dump(mode="json"),
        source_snapshot=source,
        request_hashes=requests,
        predecessor=predecessor,
    )
    batch.metadata_checkpoint = checkpoint.model_dump(mode="json")


def require_fresh_reviewed_metadata(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    from onyx.db.document_intake_metadata import (
        _load_rows,
        metadata_chunks,
        metadata_receipts_current,
        validate_metadata_receipt_evidence,
    )
    from onyx.db.document_intake_review import effective_intake_files

    for member in effective_intake_files(session, batch):
        rows = _load_rows(session, batch, member.id)
        if not metadata_receipts_current(
            rows, metadata_input_hash(metadata_chunks(rows))
        ):
            raise intake.IntakeConflictError(
                "Reviewed metadata must be current before labeling"
            )
        validate_metadata_receipt_evidence(rows, metadata_chunks(rows))
