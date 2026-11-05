"""Restore only a reviewed, unpublished intake's failed initial index request."""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast
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
    KVStore,
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

_DELIVERY_SUCCESSION_PREFIX = "intake_index_delivery_succession:"
_MAX_DELIVERY_SUCCESSIONS = 64


def _delivery_scope(
    batch: RegulatoryDocumentIntake, member: RegulatoryDocumentIntakeFile
) -> dict[str, str | int]:
    return {
        "version": 1,
        "intake_id": str(batch.id),
        "document_id": str(member.id),
        "user_file_id": str(member.user_file_id),
        "source_package_id": str(batch.source_package_id),
        "source_asset_id": str(member.source_asset_id),
        "created_by": str(batch.created_by),
        "document_set_id": batch.document_set_id,
        "approved_review_sha256": batch.review_sha256 or "",
        "approval_attempt_id": str(member.approval_attempt_id),
    }


def intake_index_attempt_chain(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    attempt_id: UUID,
) -> tuple[UUID, ...] | None:
    if not batch.review_sha256 or member.approval_attempt_id is None:
        return None
    scope = _delivery_scope(batch, member)
    current, chain = attempt_id, []
    for depth in range(_MAX_DELIVERY_SUCCESSIONS + 1):
        if current in chain:
            return None
        chain.append(current)
        if current == member.approval_attempt_id:
            return tuple(chain)
        if depth == _MAX_DELIVERY_SUCCESSIONS:
            return None
        receipt = session.get(KVStore, _DELIVERY_SUCCESSION_PREFIX + str(current))
        if receipt is None or not isinstance(receipt.value, Mapping):
            return None
        value = cast(Mapping[str, object], receipt.value)
        try:
            previous = UUID(str(value.get("previous_attempt_id")))
        except ValueError:
            return None
        if value != {
            **scope,
            "attempt_id": str(current),
            "previous_attempt_id": str(previous),
        }:
            return None
        current = previous
    return None


def intake_index_attempt_is_authorized(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    attempt_id: UUID,
) -> bool:
    """Accept only the approved attempt or its recorded, same-review successors."""
    return intake_index_attempt_chain(session, batch, member, attempt_id) is not None


def record_intake_index_delivery_succession(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    previous_attempt_id: UUID,
    replacement_attempt_id: UUID,
) -> bool:
    """Caller holds publication, batch, file and intent locks before rotating."""
    chain = intake_index_attempt_chain(session, batch, member, previous_attempt_id)
    key = _DELIVERY_SUCCESSION_PREFIX + str(replacement_attempt_id)
    if (
        chain is None
        or len(chain) > _MAX_DELIVERY_SUCCESSIONS
        or replacement_attempt_id == previous_attempt_id
        or session.get(KVStore, key) is not None
    ):
        return False
    session.add(
        KVStore(
            key=key,
            value={
                **_delivery_scope(batch, member),
                "attempt_id": str(replacement_attempt_id),
                "previous_attempt_id": str(previous_attempt_id),
            },
        )
    )
    return True


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
        or not intake_index_attempt_is_authorized(
            session, batch, member, repair.attempt_id
        )
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
    from onyx.db.document_intake_durable_indexing import (
        bound_intake_durable_job,
        require_failed_durable_cleanup,
    )

    durable = bound_intake_durable_job(
        session, batch, member, repair.attempt_id, strict=True
    )
    if durable is not None:
        require_failed_durable_cleanup(durable)
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
