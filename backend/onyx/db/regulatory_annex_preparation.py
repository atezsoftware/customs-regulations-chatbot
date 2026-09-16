"""Fenced, durable review preparation independent of publication intent."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import AnnexChangeSet, AnnexReviewPreparation
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexElementCorrection,
)

STALE_AFTER = timedelta(minutes=10)


def queue_review_preparation(
    session: Session,
    *,
    review_id: UUID,
    expected_review_sha256: str,
    corrections: list[AnnexElementCorrection] | None,
    corrected_by: UUID,
    tenant_id: str,
    environment: str,
    database_identity: str,
    initial_checkpoint: AnnexChangeDraft | None = None,
) -> AnnexChangeSet:
    from onyx.db.regulatory_annex_changes import require_current_annex_review

    review = require_current_annex_review(
        session,
        change_set_id=review_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
        allow_preparation=True,
    )
    if (
        review.status not in ("pending", "blocked", "rejected", "failed")
        or review.publication_generation
    ):
        raise ValueError("review state does not allow edits")
    from onyx.db.regulatory_annex_selection import require_unpartitioned_review

    require_unpartitioned_review(session, review)
    draft = AnnexChangeDraft.model_validate(review.review_payload)
    payload = [
        item.model_dump(mode="json")
        for item in (corrections if corrections is not None else draft.corrections)
    ]
    request_hash = context_hash([expected_review_sha256, payload, str(corrected_by)])
    job = session.scalar(
        select(AnnexReviewPreparation)
        .where(AnnexReviewPreparation.review_id == review.id)
        .with_for_update()
    )
    if job is not None:
        if (job.tenant_id, job.environment, job.database_identity) != (
            tenant_id,
            environment,
            database_identity,
        ):
            raise ValueError("review preparation scope mismatch")
        if job.status in ("queued", "running"):
            if job.request_sha256 != request_hash:
                raise ValueError("another correction is already being prepared")
            session.commit()
            return review
        if job.request_sha256 != request_hash:
            job.checkpoint = None
        job.request_sha256 = request_hash
        job.corrections = payload
        job.corrected_by = corrected_by
        job.status = "queued"
        job.stage = "queued"
        job.error_message = None
        job.completed_chunks = 0
        job.total_chunks = 0
        job.heartbeat_at = datetime.now(timezone.utc)
    else:
        job = AnnexReviewPreparation(
            review_id=review.id,
            expected_review_sha256=expected_review_sha256,
            request_sha256=request_hash,
            corrections=payload,
            corrected_by=corrected_by,
            tenant_id=tenant_id,
            environment=environment,
            database_identity=database_identity,
            status="queued",
            stage="queued",
            generation=0,
            heartbeat_at=datetime.now(timezone.utc),
            completed_chunks=0,
            total_chunks=0,
        )
        session.add(job)
    if (
        job.checkpoint is None
        and payload == [item.model_dump(mode="json") for item in draft.corrections]
        and (
            draft.selection_parent_id is not None
            or (not draft.issues and draft.impact is not None and draft.impact.ready)
        )
    ):
        job.checkpoint = draft.model_dump(mode="json")
    if initial_checkpoint is not None:
        if initial_checkpoint.model_dump(mode="json") != review.review_payload:
            raise ValueError("initial selection checkpoint differs from review")
        job.checkpoint = initial_checkpoint.model_dump(mode="json")
    session.commit()
    session.refresh(review)
    return review


def claim_review_preparation(
    session: Session,
    *,
    review_id: UUID,
    tenant_id: str,
    environment: str,
    database_identity: str,
) -> AnnexReviewPreparation | None:
    job = session.scalar(
        select(AnnexReviewPreparation)
        .where(
            AnnexReviewPreparation.review_id == review_id,
            AnnexReviewPreparation.tenant_id == tenant_id,
            AnnexReviewPreparation.environment == environment,
            AnnexReviewPreparation.database_identity == database_identity,
        )
        .with_for_update()
    )
    now = datetime.now(timezone.utc)
    if (
        job is None
        or job.status not in ("queued", "running")
        or (job.status == "running" and job.heartbeat_at > now - STALE_AFTER)
    ):
        session.rollback()
        return None
    job.status = "running"
    job.generation += 1
    job.heartbeat_at = now
    session.commit()
    session.refresh(job)
    session.expunge(job)
    return job


def touch_review_preparation(
    *,
    review_id: UUID,
    generation: int,
    stage: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    checkpoint: AnnexChangeDraft | None = None,
    error: str | None = None,
) -> None:
    with get_session_with_current_tenant() as session:
        values: dict[str, object] = {"heartbeat_at": datetime.now(timezone.utc)}
        if stage is not None:
            values["stage"] = stage
        if completed is not None:
            values["completed_chunks"] = completed
        if total is not None:
            values["total_chunks"] = total
        if checkpoint is not None:
            values["checkpoint"] = checkpoint.model_dump(mode="json")
        if error is not None:
            values.update(status="failed", error_message=error)
        updated = session.scalar(
            update(AnnexReviewPreparation)
            .where(
                AnnexReviewPreparation.review_id == review_id,
                AnnexReviewPreparation.generation == generation,
                AnnexReviewPreparation.status == "running",
            )
            .values(**values)
            .returning(AnnexReviewPreparation.review_id)
        )
        if updated is None:
            raise ValueError("review preparation lease lost")
        session.commit()


def pending_review_preparations(
    *, tenant_id: str, environment: str, database_identity: str
) -> list[UUID]:
    with get_session_with_current_tenant() as session:
        return list(
            session.scalars(
                select(AnnexReviewPreparation.review_id)
                .where(
                    AnnexReviewPreparation.tenant_id == tenant_id,
                    AnnexReviewPreparation.environment == environment,
                    AnnexReviewPreparation.database_identity == database_identity,
                    or_(
                        AnnexReviewPreparation.status == "queued",
                        (AnnexReviewPreparation.status == "running")
                        & (
                            AnnexReviewPreparation.heartbeat_at
                            < datetime.now(timezone.utc) - STALE_AFTER
                        ),
                    ),
                )
                .order_by(AnnexReviewPreparation.heartbeat_at)
                .limit(20)
            )
        )
