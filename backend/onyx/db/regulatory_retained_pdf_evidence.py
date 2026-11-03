"""Append retained raw renditions without editing the canonical publication."""

from typing import cast
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from onyx.auth.permissions import get_effective_permissions
from onyx.db.document_intake import resolve_intake_target
from onyx.db.enums import Permission
from onyx.db.models import (
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryRetainedPdfEvidence,
    User,
    UserFile,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.retained_pdf_evidence import (
    PdfRenditionRelation,
    RetainedPdfDescriptor,
)


class RetainedPdfConflictError(ValueError):
    pass


def retained_pdf_scope(
    session: Session,
    user: User,
    user_file_id: UUID,
    document_set_id: int,
    *,
    lock: bool = False,
    require_active: bool = True,
) -> tuple[UserFile, str]:
    if Permission.FULL_ADMIN_PANEL_ACCESS not in get_effective_permissions(user):
        raise PermissionError("Administrator access is required")
    target = resolve_intake_target(session, user)
    file = session.get(
        UserFile, user_file_id, populate_existing=True, with_for_update=lock
    )
    if (
        target.id != document_set_id
        or file is None
        or session.get(DocumentSet__UserFile, (target.id, user_file_id)) is None
        or (
            require_active
            and session.scalar(
                select(RegulatoryChunk.id)
                .where(
                    RegulatoryChunk.user_file_id == user_file_id,
                    RegulatoryChunk.status == "active",
                    RegulatoryChunk.superseded_by_chunk_id.is_(None),
                )
                .limit(1)
            )
            is None
        )
    ):
        raise ValueError("Canonical document not found in editable PC Külliyatı")
    return file, canonical_scope_digest(session, user_file_id)


def retained_pdf_request_hash(
    descriptor: RetainedPdfDescriptor,
    user_file_id: UUID,
    sha256: str,
    byte_count: int,
    generation_hash: str | None,
) -> str:
    return publication_digest(
        {
            **descriptor.model_dump(mode="json"),
            "user_file_id": str(user_file_id),
            "sha256": sha256,
            "byte_count": byte_count,
            "mime_type": "application/pdf",
            "canonical_generation_hash": generation_hash,
        }
    )


def validate_retained_pdf_receipt(row: RegulatoryRetainedPdfEvidence) -> None:
    descriptor = RetainedPdfDescriptor(
        document_set_id=row.document_set_id,
        expected_canonical_sha256=row.canonical_sha256,
        idempotency_key=row.idempotency_key,
        original_filename=row.original_filename,
        relation=cast(PdfRenditionRelation, row.relation),
        reason=row.reason,
        source_url=row.source_url,
    )
    if (
        row.mime_type != "application/pdf"
        or row.request_sha256
        != retained_pdf_request_hash(
            descriptor,
            row.user_file_id,
            row.sha256,
            row.byte_count,
            row.canonical_generation_hash,
        )
    ):
        raise RetainedPdfConflictError("The retained original receipt changed")


def existing_retained_pdf(
    session: Session,
    *,
    user: User,
    user_file_id: UUID,
    descriptor: RetainedPdfDescriptor,
    sha256: str,
    byte_count: int,
) -> RegulatoryRetainedPdfEvidence | None:
    row = session.scalar(
        select(RegulatoryRetainedPdfEvidence).where(
            RegulatoryRetainedPdfEvidence.user_file_id == user_file_id,
            or_(
                (RegulatoryRetainedPdfEvidence.created_by == user.id)
                & (
                    RegulatoryRetainedPdfEvidence.idempotency_key
                    == descriptor.idempotency_key
                ),
                RegulatoryRetainedPdfEvidence.sha256 == sha256,
            ),
        )
    )
    if row is not None:
        validate_retained_pdf_receipt(row)
        if row.request_sha256 != retained_pdf_request_hash(
            descriptor, user_file_id, sha256, byte_count, row.canonical_generation_hash
        ):
            raise RetainedPdfConflictError(
                "This retained original or request key already has a different receipt"
            )
    return row


def append_retained_pdf(
    session: Session,
    *,
    user: User,
    user_file_id: UUID,
    descriptor: RetainedPdfDescriptor,
    sha256: str,
    byte_count: int,
    raw_file_id: str,
) -> tuple[RegulatoryRetainedPdfEvidence, bool]:
    file, digest = retained_pdf_scope(
        session, user, user_file_id, descriptor.document_set_id, lock=True
    )
    existing = existing_retained_pdf(
        session,
        user=user,
        user_file_id=user_file_id,
        descriptor=descriptor,
        sha256=sha256,
        byte_count=byte_count,
    )
    if existing is not None:
        return existing, False
    if digest != descriptor.expected_canonical_sha256:
        raise RetainedPdfConflictError(
            "The canonical document changed; reload before attaching an original"
        )
    row = RegulatoryRetainedPdfEvidence(
        document_set_id=descriptor.document_set_id,
        user_file_id=user_file_id,
        raw_file_id=raw_file_id,
        sha256=sha256,
        byte_count=byte_count,
        mime_type="application/pdf",
        original_filename=descriptor.original_filename,
        relation=descriptor.relation,
        reason=descriptor.reason,
        source_url=descriptor.source_url,
        canonical_sha256=digest,
        canonical_generation_hash=file.regulatory_chunk_generation_hash,
        idempotency_key=descriptor.idempotency_key,
        request_sha256=retained_pdf_request_hash(
            descriptor,
            user_file_id,
            sha256,
            byte_count,
            file.regulatory_chunk_generation_hash,
        ),
        created_by=user.id,
    )
    session.add(row)
    session.flush()
    return row, True


def list_retained_pdfs(
    session: Session, user_file_id: UUID
) -> list[RegulatoryRetainedPdfEvidence]:
    rows = list(
        session.scalars(
            select(RegulatoryRetainedPdfEvidence)
            .where(
                RegulatoryRetainedPdfEvidence.user_file_id == user_file_id,
            )
            .order_by(
                RegulatoryRetainedPdfEvidence.created_at,
                RegulatoryRetainedPdfEvidence.id,
            )
        )
    )
    for row in rows:
        validate_retained_pdf_receipt(row)
    return rows


def retained_pdf_blob_is_attached(session: Session, raw_file_id: str) -> bool:
    return (
        session.scalar(
            select(RegulatoryRetainedPdfEvidence.id)
            .where(
                RegulatoryRetainedPdfEvidence.raw_file_id == raw_file_id,
            )
            .limit(1)
        )
        is not None
    )


def get_retained_pdf(
    session: Session,
    user_file_id: UUID,
    evidence_id: UUID,
    document_set_id: int,
) -> RegulatoryRetainedPdfEvidence:
    row = session.get(RegulatoryRetainedPdfEvidence, evidence_id)
    if (
        row is None
        or row.user_file_id != user_file_id
        or row.document_set_id != document_set_id
    ):
        raise ValueError("Retained original not found")
    validate_retained_pdf_receipt(row)
    return row
