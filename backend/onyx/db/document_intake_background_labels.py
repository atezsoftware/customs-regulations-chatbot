"""Independent durable label intent; legal review/publication never wait for it."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_metadata import (
    load_metadata_work,
    metadata_chunks,
    metadata_receipts_current,
    validate_metadata_receipt_evidence,
)
from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
    User,
)
from onyx.regulatory.document_intake.metadata import metadata_input_hash

_TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled"}


def uses_background_labels(batch: RegulatoryDocumentIntake) -> bool:
    return getattr(batch, "background_labeling_enabled", False) is True


def validate_legal_source(session: Session, batch: RegulatoryDocumentIntake) -> None:
    from onyx.db.document_intake_review import effective_intake_files
    from onyx.db.document_intake_structure_metadata import (
        validate_structural_verification,
    )

    for member in effective_intake_files(session, batch):
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == member.user_file_id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            )
        )
        chunks = metadata_chunks(rows)
        if not chunks or not metadata_receipts_current(
            rows, metadata_input_hash(chunks)
        ):
            raise intake.IntakeConflictError(
                "The legal metadata source is incomplete or stale"
            )
        try:
            validate_metadata_receipt_evidence(rows, chunks)
            validate_structural_verification(rows, chunks)
        except ValueError as error:
            raise intake.IntakeConflictError(str(error)) from error


def schedule_background_labels(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    batch.background_label_status = "pending"
    batch.background_label_error = None
    batch.background_label_attempts = 0
    batch.background_label_next_retry_at = None
    batch.review_sha256 = intake.review_digest(session, batch)
    batch.background_label_source_sha256 = batch.review_sha256


def finish_legal_preparation(
    session: Session, batch: RegulatoryDocumentIntake
) -> UUID | None:
    if not uses_background_labels(batch) or batch.lease_token is None:
        raise intake.IntakeConflictError(
            "Background labels require a current legal preparation"
        )
    _batch, unfinished = load_metadata_work(session, batch.id, batch.lease_token)
    if unfinished:
        raise intake.IntakeConflictError(
            "Legal metadata must finish before document review"
        )
    validate_legal_source(session, batch)
    files = intake.intake_files(session, batch)
    if not files or any(member.status != "chunked" for member in files):
        raise intake.IntakeConflictError("Every source must be chunked before review")
    batch.status, batch.stage, batch.error = "ready", "review", None
    batch.lease_token, batch.lease_expires_at = None, None
    for member in files:
        member.status = "ready"
    schedule_background_labels(session, batch)
    session.flush()
    return None


def _create_run(session: Session, batch: RegulatoryDocumentIntake) -> UUID:
    from onyx.db.document_intake_review import effective_intake_files
    from onyx.db.document_intake_structural_readers import (
        load_intake_structural_reader_proofs,
    )

    user = session.get_one(User, batch.created_by)
    binding = resolve_intake_labeling_provider_binding(
        session, batch.labeling_model_configuration_id, user=user
    )
    members = effective_intake_files(session, batch)
    readers = load_intake_structural_reader_proofs(session, intake_id=batch.id)
    run, _created = labeling.create_labeling_run(
        session,
        document_set_id=batch.document_set_id,
        taxonomy=labeling.get_label_settings(session).taxonomy,
        model_configuration_id=batch.labeling_model_configuration_id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=batch.created_by,
        idempotency_key=uuid5(
            batch.id,
            f"background-labels:{batch.background_label_source_sha256}:{batch.background_label_attempts}",
        ),
        uses_current_labels=True,
        target_file_ids=[member.user_file_id for member in members],
        structural_readers=readers or None,
    )
    return run.id


def start_background_labels(
    session: Session, batch: RegulatoryDocumentIntake
) -> UUID | None:
    if not uses_background_labels(batch) or batch.background_label_status not in {
        "pending",
        "waiting",
    }:
        return None
    try:
        if (
            not batch.background_label_source_sha256
            or intake.review_digest(session, batch)
            != batch.background_label_source_sha256
        ):
            raise intake.IntakeConflictError(
                "The background label source review changed"
            )
        validate_legal_source(session, batch)
        from onyx.db.models import DocumentSet

        session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
        if labeling.get_active_run_id(session, batch.document_set_id) is not None:
            batch.background_label_status = "waiting"
            return None
        if batch.background_label_attempts >= 3:
            raise ValueError("Background labeling exhausted its bounded retry budget")
        batch.background_label_attempts += 1
        with session.begin_nested():
            run_id = _create_run(session, batch)
        batch.labeling_run_id = run_id
        batch.background_label_status, batch.background_label_error = "running", None
        batch.background_label_next_retry_at = None
        session.flush()
        return run_id
    except (ValueError, intake.IntakeConflictError) as error:
        batch.background_label_error = str(error)[:2000]
        batch.background_label_status = "failed"
        batch.background_label_next_retry_at = None
        session.flush()
        return None


def reconcile_background_labels(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    if (
        not uses_background_labels(batch)
        or batch.labeling_run_id is None
        or batch.background_label_status not in {"running", "completed"}
    ):
        return
    run = session.get(
        RegulatoryLabelingRun, batch.labeling_run_id, populate_existing=True
    )
    from onyx.db.document_intake_review import effective_intake_files

    if run is None or set(run.file_ids) != {
        str(m.user_file_id) for m in effective_intake_files(session, batch)
    }:
        batch.background_label_status, batch.background_label_error = (
            "failed",
            "The background labeling scope changed",
        )
        return
    if run.status not in _TERMINAL:
        batch.background_label_status = "running"
        return
    if run.status == "completed" and not (
        run.failed_chunks or run.stale_chunks or run.unresolved_derived_chunks
    ):
        batch.background_label_status, batch.background_label_error = "completed", None
    else:
        batch.background_label_status = (
            "cancelled" if run.status == "cancelled" else "failed"
        )
        batch.background_label_error = (
            run.error
            or "Background labels have unresolved targets; legal review remains available"
        )


def background_label_snapshot(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[str, object] | None:
    if not uses_background_labels(batch) or batch.background_label_status is None:
        return None
    reconcile_background_labels(session, batch)
    run = (
        session.get(RegulatoryLabelingRun, batch.labeling_run_id)
        if batch.labeling_run_id
        and batch.background_label_status not in {"pending", "waiting"}
        else None
    )
    classified, unlabelled = (
        labeling.completed_classification_counts(session, run.id) if run else (0, 0)
    )
    return {
        "status": batch.background_label_status,
        "error": batch.background_label_error,
        "total_chunks": run.total_chunks if run else 0,
        "completed_chunks": run.completed_chunks if run else 0,
        "classified_chunks": classified,
        "unlabelled_chunks": unlabelled,
        "failed_chunks": run.failed_chunks if run else 0,
    }


def resume_background_labeling(
    session: Session, *, environment: str, limit: int = 10
) -> list[UUID]:
    if not environment.strip() or not 1 <= limit <= 100:
        raise ValueError("Background labeling requires a bounded environment scope")
    batches = list(
        session.scalars(
            select(RegulatoryDocumentIntake)
            .join(
                AmendmentSourcePackage,
                AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
            )
            .where(
                AmendmentSourcePackage.environment == environment,
                RegulatoryDocumentIntake.background_labeling_enabled.is_(True),
                RegulatoryDocumentIntake.background_label_status.in_(
                    ("pending", "waiting", "running")
                ),
                RegulatoryDocumentIntake.status.in_(
                    ("ready", "approving", "completed", "failed")
                ),
                or_(
                    RegulatoryDocumentIntake.background_label_next_retry_at.is_(None),
                    RegulatoryDocumentIntake.background_label_next_retry_at
                    <= datetime.now(timezone.utc),
                ),
            )
            .order_by(RegulatoryDocumentIntake.created_at, RegulatoryDocumentIntake.id)
            .limit(limit)
            .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
            .execution_options(populate_existing=True)
        )
    )
    deliveries = []
    for batch in batches:
        if batch.background_label_status == "running":
            reconcile_background_labels(session, batch)
            continue
        run_id = start_background_labels(session, batch)
        if run_id is not None:
            deliveries.append(run_id)
    session.flush()
    return deliveries


def label_delivery_pending(batch: RegulatoryDocumentIntake) -> None:
    batch.background_label_error = (
        "Labeling delivery is pending; automatic recovery will retry"
    )
    batch.background_label_next_retry_at = datetime.now(timezone.utc) + timedelta(
        seconds=30
    )


def is_current_background_label_run(
    session: Session, run: RegulatoryLabelingRun
) -> bool:
    if run.status not in {"queued", "running"} or run.cancel_requested:
        return False
    batches = list(
        session.scalars(
            select(RegulatoryDocumentIntake)
            .where(
                RegulatoryDocumentIntake.labeling_run_id == run.id,
                RegulatoryDocumentIntake.background_labeling_enabled.is_(True),
                RegulatoryDocumentIntake.background_label_status == "running",
                RegulatoryDocumentIntake.document_set_id == run.document_set_id,
                RegulatoryDocumentIntake.labeling_model_configuration_id
                == run.model_configuration_id,
                RegulatoryDocumentIntake.created_by == run.requested_by_id,
            )
            .limit(2)
        )
    )
    if len(batches) != 1:
        return False
    batch = batches[0]
    if batch.status not in {"ready", "approving", "completed", "failed"}:
        return False
    from onyx.db.document_intake_review import effective_intake_files
    from onyx.db.models import DocumentSet__UserFile, UserFile

    members = effective_intake_files(session, batch)
    ids = {member.user_file_id for member in members}
    if not ids or set(run.file_ids) != {str(identifier) for identifier in ids}:
        return False
    owned = set(
        session.scalars(
            select(UserFile.id)
            .join(
                DocumentSet__UserFile, DocumentSet__UserFile.user_file_id == UserFile.id
            )
            .where(
                UserFile.id.in_(ids),
                UserFile.user_id == batch.created_by,
                DocumentSet__UserFile.document_set_id == batch.document_set_id,
            )
        )
    )
    if owned != ids:
        return False
    try:
        return batch.background_label_source_sha256 == intake.review_digest(
            session, batch
        )
    except (ValueError, intake.IntakeConflictError):
        return False


def adopt_completed_legal_preparation(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    """Explicit maintenance can release an old label wait without changing metadata."""
    if uses_background_labels(batch):
        return
    if (
        batch.status != "preparing"
        or batch.stage not in {"labeling", "waiting_labeling"}
        or batch.lease_token is not None
        or batch.lease_expires_at is not None
    ):
        raise intake.IntakeConflictError(
            "Only completed legal preparation can adopt background labels"
        )
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import UserFile

    members = intake.intake_files(session, batch)
    if not members or any(
        m.approval_attempt_id is not None
        or session.get_one(UserFile, m.user_file_id).status != UserFileStatus.CHUNKED
        for m in members
    ):
        raise intake.IntakeConflictError(
            "Background label adoption requires unpublished owned files"
        )
    validate_legal_source(session, batch)
    batch.background_labeling_enabled = True
    batch.status, batch.stage, batch.error = "ready", "review", None
    for member in members:
        member.status = "ready"
    batch.review_sha256 = intake.review_digest(session, batch)
    batch.background_label_source_sha256 = batch.review_sha256
    batch.background_label_status = "running" if batch.labeling_run_id else "pending"
    batch.background_label_error = None
    reconcile_background_labels(session, batch)
    session.flush()
