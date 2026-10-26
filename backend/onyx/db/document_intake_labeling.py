"""Identify only the current unpublished intake scope for labeling policy."""

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet__UserFile,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingRun,
    UserFile,
)


def is_current_intake_labeling_run(
    session: Session, run: RegulatoryLabelingRun
) -> bool:
    """Read under the caller's run lease; never lock a batch after its run."""
    associations = session.execute(
        select(
            RegulatoryDocumentIntake.id,
            RegulatoryDocumentIntake.document_set_id,
            RegulatoryDocumentIntake.created_by,
            RegulatoryLabelingRun.file_ids,
        )
        .select_from(RegulatoryDocumentIntake)
        .join(
            RegulatoryLabelingRun,
            RegulatoryDocumentIntake.labeling_run_id == RegulatoryLabelingRun.id,
        )
        .where(
            RegulatoryLabelingRun.id == run.id,
            RegulatoryLabelingRun.status.in_(("queued", "running")),
            RegulatoryLabelingRun.cancel_requested.is_(False),
            RegulatoryDocumentIntake.status == "preparing",
            RegulatoryDocumentIntake.stage == "labeling",
            RegulatoryDocumentIntake.document_set_id
            == RegulatoryLabelingRun.document_set_id,
            RegulatoryDocumentIntake.labeling_model_configuration_id
            == RegulatoryLabelingRun.model_configuration_id,
            RegulatoryDocumentIntake.created_by
            == RegulatoryLabelingRun.requested_by_id,
        )
        .limit(2)
    ).all()
    if len(associations) != 1:
        return False
    intake_id, document_set_id, owner_id, file_ids = associations[0]
    if (
        not isinstance(file_ids, list)
        or not file_ids
        or any(not isinstance(identifier, str) for identifier in file_ids)
        or len(set(file_ids)) != len(file_ids)
    ):
        return False
    members = session.execute(
        select(
            RegulatoryDocumentIntakeFile.user_file_id,
            RegulatoryDocumentIntakeFile.status,
            RegulatoryDocumentIntakeFile.approval_attempt_id,
            UserFile.user_id,
            UserFile.status,
            DocumentSet__UserFile.document_set_id,
        )
        .select_from(RegulatoryDocumentIntakeFile)
        .outerjoin(UserFile, UserFile.id == RegulatoryDocumentIntakeFile.user_file_id)
        .outerjoin(
            DocumentSet__UserFile,
            and_(
                DocumentSet__UserFile.user_file_id
                == RegulatoryDocumentIntakeFile.user_file_id,
                DocumentSet__UserFile.document_set_id == document_set_id,
            ),
        )
        .where(RegulatoryDocumentIntakeFile.intake_id == intake_id)
    ).all()
    return (
        bool(members)
        and set(file_ids) == {str(member.user_file_id) for member in members}
        and all(
            status == "labeling"
            and approval_attempt_id is None
            and user_id == owner_id
            and file_status == UserFileStatus.CHUNKED
            and membership_id == document_set_id
            for (
                _file_id,
                status,
                approval_attempt_id,
                user_id,
                file_status,
                membership_id,
            ) in members
        )
    )
