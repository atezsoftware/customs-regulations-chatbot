"""Opt in only the current approved, unpublished intake index request."""

from uuid import UUID

from sqlalchemy import select

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    DocumentSet__UserFile,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.document_index.publication_models import FileOwnership


def current_intake_embedding_scope(
    owner: FileOwnership, attempt_id: UUID, *, canonical_sha256: str
) -> bool:
    """Read under the existing publication lease; retain no lock over provider I/O."""
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        publication = session.get(
            RegulatoryFilePublication, owner.user_file_id, populate_existing=True
        )
        associations = session.execute(
            select(
                RegulatoryDocumentIntake,
                RegulatoryDocumentIntakeFile,
                UserFile,
                UserFileProjectionRepair,
            )
            .join(
                RegulatoryDocumentIntakeFile,
                RegulatoryDocumentIntakeFile.intake_id == RegulatoryDocumentIntake.id,
            )
            .join(UserFile, UserFile.id == RegulatoryDocumentIntakeFile.user_file_id)
            .join(
                UserFileProjectionRepair,
                UserFileProjectionRepair.user_file_id == UserFile.id,
            )
            .join(
                DocumentSet__UserFile,
                (DocumentSet__UserFile.user_file_id == UserFile.id)
                & (
                    DocumentSet__UserFile.document_set_id
                    == RegulatoryDocumentIntake.document_set_id
                ),
            )
            .where(UserFile.id == owner.user_file_id)
            .execution_options(populate_existing=True)
            .limit(2)
        ).all()
        if len(associations) != 1 or publication is None:
            return False
        batch, member, file, request = associations[0]
        if (
            reservations.gate_closed
            or publication.epoch != 0
            or publication.writer_manifest is not None
            or batch.status != "approving"
            or batch.stage != "indexing"
            or not batch.review_sha256
            or member.status != "indexing"
            or member.approval_attempt_id != attempt_id
            or request.attempt_id != attempt_id
            or not request.initial_index
            or request.status != UserFileProjectionRepairStatus.RUNNING
            or file.status != UserFileStatus.CHUNKED
            or file.user_id != batch.created_by
            or session.scalar(
                select(RegulatoryTemporalProjection.id)
                .where(RegulatoryTemporalProjection.user_file_id == file.id)
                .limit(1)
            )
            is not None
        ):
            return False
        from onyx.db import document_intake as intake

        try:
            return (
                canonical_scope_digest(session, file.id) == canonical_sha256
                and intake.review_digest(session, batch) == batch.review_sha256
            )
        except (ValueError, intake.IntakeConflictError):
            return False
