"""Read-only original-source authority for the approved intake snapshot."""

from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_reprepare import reprepared_original_receipt
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryPublicationOrdinal,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt


def has_initial_intake_canonical_state(session: Session, file_ids: list[UUID]) -> bool:
    rows = session.execute(
        select(RegulatoryChunk, RegulatoryPublicationOrdinal.ordinal)
        .outerjoin(
            RegulatoryPublicationOrdinal,
            and_(
                RegulatoryPublicationOrdinal.user_file_id
                == RegulatoryChunk.user_file_id,
                RegulatoryPublicationOrdinal.allocation_key
                == "canonical:" + RegulatoryChunk.id,
            ),
        )
        .where(RegulatoryChunk.user_file_id.in_(file_ids))
        .execution_options(populate_existing=True)
    ).all()
    return (
        bool(rows)
        and {chunk.user_file_id for chunk, _ordinal in rows} == set(file_ids)
        and all(
            chunk.source == "indexed"
            and chunk.supersedes_chunk_id is None
            and chunk.superseded_by_chunk_id is None
            and ordinal is not None
            and chunk.projection_ordinal == ordinal
            for chunk, ordinal in rows
        )
    )


def approved_intake_original_receipt(
    session: Session,
    user_file_id: UUID,
    original: OriginalIngestionReceipt,
    *,
    canonical_sha256: str,
) -> OriginalIngestionReceipt | None:
    """Derive current canonical authority without changing extraction evidence."""
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile)
        .where(RegulatoryDocumentIntakeFile.user_file_id == user_file_id)
        .execution_options(populate_existing=True)
    )
    if member is None:
        return None
    batch = session.get(
        RegulatoryDocumentIntake, member.intake_id, populate_existing=True
    )
    if (
        batch is None
        or batch.status != "completed"
        or batch.stage != "completed"
        or not batch.review_sha256
    ):
        return None
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .execution_options(populate_existing=True)
        )
    )
    if not members or any(
        item.status != "completed" or item.approval_attempt_id is None
        for item in members
    ):
        return None
    file = session.get(UserFile, user_file_id, populate_existing=True)
    publication = session.get(
        RegulatoryFilePublication, user_file_id, populate_existing=True
    )
    asset = session.get(
        RegulatorySourceAsset, member.source_asset_id, populate_existing=True
    )
    package = session.get(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    if (
        file is None
        or file.status != UserFileStatus.COMPLETED
        or file.user_id != batch.created_by
        or file.file_id != original.file_id
        or publication is None
        or publication.gate_closed
        or publication.epoch <= 0
        or publication.writer_manifest is not None
        or publication.original_ingestion_receipt != original.model_dump(mode="json")
        or asset is None
        or asset.package_id != batch.source_package_id
        or asset.sha256 != original.raw_sha256
        or package is None
        or package.status != "ready"
        or package.issues
        or package.created_by != batch.created_by
        or package.document_set_id != batch.document_set_id
    ):
        return None
    try:
        effective = reprepared_original_receipt(session, user_file_id, original)
        if (
            effective.file_id != original.file_id
            or effective.raw_sha256 != original.raw_sha256
            or effective.plaintext_sha256 != original.plaintext_sha256
            or effective.generation_hash != file.regulatory_chunk_generation_hash
            or not has_initial_intake_canonical_state(
                session, [item.user_file_id for item in members]
            )
            or intake.review_digest(session, batch) != batch.review_sha256
        ):
            return None
    except (ValueError, intake.IntakeConflictError):
        return None
    # The approved review binds metadata and manual validity dates added after
    # extraction/reprepare; those immutable receipts retain their original hash.
    return effective.model_copy(update={"canonical_sha256": canonical_sha256})
