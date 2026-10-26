"""Restore only a reviewed, unpublished intake's failed initial index request."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_original_ingestion import (
    has_initial_intake_canonical_state,
)
from onyx.db.document_intake_reprepare import reprepared_original_receipt
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryAnnexElementChunk,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.utils.logger import setup_logger

logger = setup_logger()


def restore_failed_intake_index_request(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    file: UserFile,
) -> dict[str, object]:
    """Caller owns the batch/file locks and a rollback boundary for all members."""
    repair = session.get(
        UserFileProjectionRepair,
        file.id,
        with_for_update=True,
        populate_existing=True,
    )
    publication = session.get(
        RegulatoryFilePublication, file.id, populate_existing=True
    )
    if (
        batch.status not in {"failed", "partial"}
        or batch.stage != "indexing"
        or not batch.review_sha256
        or member.intake_id != batch.id
        or member.user_file_id != file.id
        or member.status != "failed"
        or member.approval_attempt_id is None
        or file.status != UserFileStatus.FAILED
        or file.user_id != batch.created_by
        or repair is None
        or not repair.initial_index
        or repair.status != UserFileProjectionRepairStatus.FAILED
        or repair.attempt_id != member.approval_attempt_id
        or publication is None
        or publication.epoch != 0
        or publication.gate_closed
        or publication.writer_manifest is not None
        or publication.writer_manifest_sha256 is not None
        or publication.lease_expires_at > datetime.now(timezone.utc)
        or not publication.original_ingestion_receipt
    ):
        raise intake.IntakeConflictError(
            "Only a failed, approved initial index request can be retried"
        )
    asset = session.get(
        RegulatorySourceAsset, member.source_asset_id, populate_existing=True
    )
    package = session.get(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    try:
        original = OriginalIngestionReceipt.model_validate(
            publication.original_ingestion_receipt
        )
        effective = reprepared_original_receipt(session, file.id, original)
        if (
            file.file_id != original.file_id
            or asset is None
            or asset.package_id != batch.source_package_id
            or asset.sha256 != original.raw_sha256
            or package is None
            or package.status != "ready"
            or package.issues
            or package.created_by != batch.created_by
            or package.document_set_id != batch.document_set_id
            or effective.file_id != original.file_id
            or effective.raw_sha256 != original.raw_sha256
            or effective.plaintext_sha256 != original.plaintext_sha256
            or effective.generation_hash != file.regulatory_chunk_generation_hash
            or not has_initial_intake_canonical_state(session, [file.id])
            or intake.review_digest(session, batch) != batch.review_sha256
        ):
            raise ValueError("The approved source or canonical snapshot changed")
    except ValueError as error:
        raise intake.IntakeConflictError(
            "The approved source or canonical snapshot changed"
        ) from error
    if (
        session.scalar(
            select(RegulatoryTemporalProjection.id)
            .where(RegulatoryTemporalProjection.user_file_id == file.id)
            .limit(1)
        )
        is not None
        or session.scalar(
            select(RegulatoryAnnexElementChunk.chunk_id)
            .join(
                RegulatoryChunk,
                RegulatoryChunk.id == RegulatoryAnnexElementChunk.chunk_id,
            )
            .where(RegulatoryChunk.user_file_id == file.id)
            .limit(1)
        )
        is not None
        or session.scalar(
            select(RegulatoryIndexingJob.id)
            .where(
                RegulatoryIndexingJob.user_file_id == file.id,
                RegulatoryIndexingJob.status.not_in(("FAILED", "CANCELLED")),
            )
            .limit(1)
        )
        is not None
    ):
        raise intake.IntakeConflictError(
            "Published history or active indexing prevents initial intake retry"
        )
    evidence: dict[str, object] = {
        "intake_id": str(batch.id),
        "user_file_id": str(file.id),
        "previous_attempt_id": str(repair.attempt_id),
        "previous_failure_code": repair.failure_code,
        "previous_attempt_updated_at": repair.updated_at.isoformat(),
        "approved_review_sha256": batch.review_sha256,
    }
    file.status = UserFileStatus.CHUNKED
    return evidence


def log_failed_intake_index_retry(
    evidence: dict[str, object], replacement_attempt_id: UUID
) -> None:
    logger.info(
        "Approved initial intake index retry requested",
        extra={**evidence, "replacement_attempt_id": str(replacement_attempt_id)},
    )
