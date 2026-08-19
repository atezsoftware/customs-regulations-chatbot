import datetime
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
)
from onyx.db.models import RegulatoryIndexingItem, RegulatoryIndexingJob

_MAX_ERROR_MESSAGE_LENGTH = 4000


@dataclass(frozen=True)
class RegulatoryIndexingJobClaim:
    job_id: UUID
    stage: RegulatoryIndexingStage
    lease_generation: int


def create_or_get_regulatory_indexing_job(
    db_session: Session,
    *,
    user_file_id: UUID,
    content_hash: str,
    search_settings_id: int,
    prompt_hash: str,
    config_snapshot: dict[str, object],
    now: datetime.datetime,
) -> RegulatoryIndexingJob:
    """Create one durable row for an immutable file/config revision."""
    proposed_id = uuid4()
    created_id = db_session.scalar(
        pg_insert(RegulatoryIndexingJob)
        .values(
            id=proposed_id,
            user_file_id=user_file_id,
            content_hash=content_hash,
            search_settings_id=search_settings_id,
            prompt_hash=prompt_hash,
            config_snapshot=config_snapshot,
            status=RegulatoryIndexingJobStatus.QUEUED.value,
            stage=RegulatoryIndexingStage.PREPARING.value,
            lease_generation=0,
            attempt_count=0,
            heartbeat_at=now,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(constraint="uq_regulatory_indexing_job_idempotency")
        .returning(RegulatoryIndexingJob.id)
    )
    job_id = created_id or db_session.scalar(
        select(RegulatoryIndexingJob.id).where(
            RegulatoryIndexingJob.user_file_id == user_file_id,
            RegulatoryIndexingJob.content_hash == content_hash,
            RegulatoryIndexingJob.search_settings_id == search_settings_id,
            RegulatoryIndexingJob.prompt_hash == prompt_hash,
        )
    )
    if job_id is None:
        raise RuntimeError("regulatory indexing job disappeared during creation")
    db_session.commit()
    job = db_session.get(RegulatoryIndexingJob, job_id)
    if job is None:
        raise RuntimeError(f"regulatory indexing job {job_id} was not persisted")
    return job


def claim_regulatory_indexing_job(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    now: datetime.datetime,
) -> bool:
    """Claim a queued or due-retry job with a monotonic lease fence."""
    runnable_statuses = (
        RegulatoryIndexingJobStatus.QUEUED.value,
        RegulatoryIndexingJobStatus.RETRY_WAIT.value,
    )
    claimed_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status.in_(runnable_statuses),
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.stage == expected_stage.value,
            or_(
                RegulatoryIndexingJob.status
                == RegulatoryIndexingJobStatus.QUEUED.value,
                and_(
                    RegulatoryIndexingJob.status
                    == RegulatoryIndexingJobStatus.RETRY_WAIT.value,
                    RegulatoryIndexingJob.next_retry_at.is_not(None),
                    RegulatoryIndexingJob.next_retry_at <= now,
                ),
            ),
        )
        .values(
            status=RegulatoryIndexingJobStatus.RUNNING.value,
            lease_generation=RegulatoryIndexingJob.lease_generation + 1,
            heartbeat_at=now,
            next_retry_at=None,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return claimed_id is not None


def advance_regulatory_indexing_job(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    next_stage: RegulatoryIndexingStage,
    next_status: RegulatoryIndexingJobStatus = RegulatoryIndexingJobStatus.QUEUED,
    error_code: str | None = None,
    error_message: str | None = None,
    now: datetime.datetime,
) -> bool:
    """Advance only the worker that owns the current stage lease."""
    advanced_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .values(
            status=next_status.value,
            stage=next_stage.value,
            attempt_count=0,
            heartbeat_at=now,
            next_retry_at=None,
            error_code=error_code,
            error_message=(
                error_message[:_MAX_ERROR_MESSAGE_LENGTH]
                if error_message is not None
                else None
            ),
            updated_at=now,
            completed_at=(
                now
                if next_status
                in {
                    RegulatoryIndexingJobStatus.SUCCEEDED,
                    RegulatoryIndexingJobStatus.FAILED,
                    RegulatoryIndexingJobStatus.CANCELLED,
                }
                else None
            ),
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return advanced_id is not None


def schedule_regulatory_indexing_retry(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    next_retry_at: datetime.datetime,
    error_code: str,
    error_message: str,
) -> bool:
    """Persist retry eligibility without allowing an older lease to reschedule."""
    scheduled_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .values(
            status=RegulatoryIndexingJobStatus.RETRY_WAIT.value,
            attempt_count=RegulatoryIndexingJob.attempt_count + 1,
            next_retry_at=next_retry_at,
            error_code=error_code,
            error_message=error_message[:_MAX_ERROR_MESSAGE_LENGTH],
            updated_at=func.now(),
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return scheduled_id is not None


def claim_stale_regulatory_indexing_jobs(
    db_session: Session,
    *,
    stale_before: datetime.datetime,
    claimed_at: datetime.datetime,
    limit: int = 20,
) -> list[RegulatoryIndexingJobClaim]:
    """Lock and claim stale runnable jobs for durable queue re-delivery."""
    stale_queued_or_running = and_(
        RegulatoryIndexingJob.status.in_(
            (
                RegulatoryIndexingJobStatus.QUEUED.value,
                RegulatoryIndexingJobStatus.RUNNING.value,
            )
        ),
        RegulatoryIndexingJob.heartbeat_at.is_not(None),
        RegulatoryIndexingJob.heartbeat_at <= stale_before,
    )
    due_retry = and_(
        RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RETRY_WAIT.value,
        RegulatoryIndexingJob.next_retry_at.is_not(None),
        RegulatoryIndexingJob.next_retry_at <= claimed_at,
    )
    jobs = list(
        db_session.scalars(
            select(RegulatoryIndexingJob)
            .where(or_(stale_queued_or_running, due_retry))
            .order_by(
                func.coalesce(
                    RegulatoryIndexingJob.next_retry_at,
                    RegulatoryIndexingJob.heartbeat_at,
                    RegulatoryIndexingJob.created_at,
                ),
                RegulatoryIndexingJob.id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
    )
    claims: list[RegulatoryIndexingJobClaim] = []
    for job in jobs:
        job.status = RegulatoryIndexingJobStatus.RUNNING.value
        job.lease_generation += 1
        job.heartbeat_at = claimed_at
        job.next_retry_at = None
        claims.append(
            RegulatoryIndexingJobClaim(
                job_id=job.id,
                stage=RegulatoryIndexingStage(job.stage),
                lease_generation=job.lease_generation,
            )
        )
    if jobs:
        db_session.commit()
    return claims


def create_or_get_regulatory_indexing_item(
    db_session: Session,
    *,
    job_id: UUID,
    regulatory_chunk_id: str,
    request_hash: str,
) -> RegulatoryIndexingItem:
    """Create one result row per canonical chunk, tolerating redelivery."""
    proposed_id = uuid4()
    created_id = db_session.scalar(
        pg_insert(RegulatoryIndexingItem)
        .values(
            id=proposed_id,
            job_id=job_id,
            regulatory_chunk_id=regulatory_chunk_id,
            request_hash=request_hash,
            status=RegulatoryIndexingItemStatus.PENDING.value,
        )
        .on_conflict_do_nothing(constraint="uq_regulatory_indexing_item_job_chunk")
        .returning(RegulatoryIndexingItem.id)
    )
    item_id = created_id or db_session.scalar(
        select(RegulatoryIndexingItem.id).where(
            RegulatoryIndexingItem.job_id == job_id,
            RegulatoryIndexingItem.regulatory_chunk_id == regulatory_chunk_id,
        )
    )
    if item_id is None:
        raise RuntimeError("regulatory indexing item disappeared during creation")
    db_session.commit()
    item = db_session.get(RegulatoryIndexingItem, item_id)
    if item is None:
        raise RuntimeError(f"regulatory indexing item {item_id} was not persisted")
    return item


def _persist_regulatory_indexing_item_values(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
    allowed_statuses: tuple[str, ...],
    values: dict[str, object],
) -> bool:
    persisted_id = db_session.scalar(
        update(RegulatoryIndexingItem)
        .where(
            RegulatoryIndexingItem.id == item_id,
            RegulatoryIndexingItem.status.in_(allowed_statuses),
            RegulatoryIndexingItem.job_id.in_(
                select(RegulatoryIndexingJob.id).where(
                    RegulatoryIndexingJob.status
                    == RegulatoryIndexingJobStatus.RUNNING.value,
                    RegulatoryIndexingJob.lease_generation == expected_generation,
                )
            ),
        )
        .values(**values)
        .returning(RegulatoryIndexingItem.id)
    )
    db_session.commit()
    return persisted_id is not None


def persist_regulatory_indexing_item_context(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
    context: dict[str, object],
) -> bool:
    return _persist_regulatory_indexing_item_values(
        db_session,
        item_id=item_id,
        expected_generation=expected_generation,
        allowed_statuses=(RegulatoryIndexingItemStatus.PENDING.value,),
        values={
            "status": RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            "context": context,
            "error_code": None,
            "error_message": None,
            "updated_at": func.now(),
        },
    )


def persist_regulatory_indexing_item_vector(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
    vector: list[float],
) -> bool:
    return _persist_regulatory_indexing_item_values(
        db_session,
        item_id=item_id,
        expected_generation=expected_generation,
        allowed_statuses=(
            RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            RegulatoryIndexingItemStatus.SKIPPED.value,
        ),
        values={
            "status": RegulatoryIndexingItemStatus.EMBEDDED.value,
            "vector": vector,
            "error_code": None,
            "error_message": None,
            "updated_at": func.now(),
        },
    )


def persist_regulatory_indexing_item_skipped(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
) -> bool:
    return _persist_regulatory_indexing_item_values(
        db_session,
        item_id=item_id,
        expected_generation=expected_generation,
        allowed_statuses=(RegulatoryIndexingItemStatus.PENDING.value,),
        values={
            "status": RegulatoryIndexingItemStatus.SKIPPED.value,
            "context": None,
            "error_code": None,
            "error_message": None,
            "updated_at": func.now(),
        },
    )


def persist_regulatory_indexing_item_failure(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
    error_code: str,
    error_message: str,
) -> bool:
    return _persist_regulatory_indexing_item_values(
        db_session,
        item_id=item_id,
        expected_generation=expected_generation,
        allowed_statuses=(
            RegulatoryIndexingItemStatus.PENDING.value,
            RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            RegulatoryIndexingItemStatus.SKIPPED.value,
        ),
        values={
            "status": RegulatoryIndexingItemStatus.FAILED.value,
            "error_code": error_code,
            "error_message": error_message[:_MAX_ERROR_MESSAGE_LENGTH],
            "updated_at": func.now(),
        },
    )
