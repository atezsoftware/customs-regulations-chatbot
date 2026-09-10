"""Positive current-original eligibility, independent of permanent ownership gates."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import RegulatoryFilePublication, UserFile
from onyx.db.regulatory_public_reads import protected_file_ids
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt


def original_attachment_receipt(
    session: Session, user_file_id: UUID
) -> OriginalIngestionReceipt | None:
    from onyx.db.regulatory_writer_publication import canonical_scope_digest

    publication = session.get(
        RegulatoryFilePublication, user_file_id, populate_existing=True
    )
    file = session.get(UserFile, user_file_id, populate_existing=True)
    if (
        publication is None
        or publication.original_ingestion_receipt is None
        or file is None
    ):
        return None
    receipt = OriginalIngestionReceipt.model_validate(
        publication.original_ingestion_receipt
    )
    if (
        publication.gate_closed
        or file.status != UserFileStatus.COMPLETED
        or file.file_id != receipt.file_id
        or canonical_scope_digest(session, user_file_id) != receipt.canonical_sha256
    ):
        return None
    return receipt


def unavailable_original_file_ids(
    session: Session, file_ids: tuple[UUID, ...]
) -> frozenset[UUID]:
    return frozenset(
        identifier
        for identifier in protected_file_ids(session, file_ids)
        if original_attachment_receipt(session, identifier) is None
    )


def current_unavailable_original_file_ids(
    file_ids: tuple[UUID, ...],
) -> frozenset[UUID]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return unavailable_original_file_ids(session, file_ids)


def current_original_receipts(file_id: str) -> list[OriginalIngestionReceipt | None]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        originals = tuple(
            session.scalars(select(UserFile.id).where(UserFile.file_id == file_id))
        )
        return [
            original_attachment_receipt(session, identifier)
            for identifier in protected_file_ids(session, originals)
        ]
