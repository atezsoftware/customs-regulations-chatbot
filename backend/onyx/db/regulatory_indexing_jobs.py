import datetime
import math
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
_SECRET_KEY_FRAGMENTS = (
    "apikey",
    "accesstoken",
    "refreshtoken",
    "bearertoken",
    "password",
    "secret",
    "credential",
    "privatekey",
    "accesskey",
    "authorization",
)
_NEXT_STAGE = {
    RegulatoryIndexingStage.PREPARING: RegulatoryIndexingStage.CONTEXT_SUBMIT,
    RegulatoryIndexingStage.CONTEXT_SUBMIT: RegulatoryIndexingStage.CONTEXT_WAIT,
    RegulatoryIndexingStage.CONTEXT_WAIT: RegulatoryIndexingStage.CONTEXT_APPLY,
    RegulatoryIndexingStage.CONTEXT_APPLY: RegulatoryIndexingStage.EMBEDDING,
    RegulatoryIndexingStage.EMBEDDING: RegulatoryIndexingStage.INDEX_WRITE,
    RegulatoryIndexingStage.INDEX_WRITE: RegulatoryIndexingStage.VERIFY,
    RegulatoryIndexingStage.VERIFY: RegulatoryIndexingStage.PUBLISH,
}

type RegulatoryIndexingJSONScalar = str | int | float | bool | None
type RegulatoryIndexingJSONValue = (
    RegulatoryIndexingJSONScalar
    | list[RegulatoryIndexingJSONValue]
    | dict[str, RegulatoryIndexingJSONValue]
)
type RegulatoryIndexingConfigSnapshot = dict[str, RegulatoryIndexingJSONValue]


@dataclass(frozen=True)
class RegulatoryIndexingJobClaim:
    job_id: UUID
    stage: RegulatoryIndexingStage
    lease_generation: int


def _validate_config_snapshot(
    value: object,
    *,
    path: str = "config_snapshot",
    active_container_ids: set[int] | None = None,
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain only JSON-safe values")

    container_ids = active_container_ids if active_container_ids is not None else set()
    container_id = id(value)
    if container_id in container_ids:
        raise ValueError(f"{path} must not contain recursive containers")
    container_ids.add(container_id)
    try:
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_config_snapshot(
                    item,
                    path=f"{path}[{index}]",
                    active_container_ids=container_ids,
                )
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must use string JSON object keys")
            normalized_key = "".join(
                character for character in key.lower() if character.isalnum()
            )
            if normalized_key == "token" or any(
                fragment in normalized_key for fragment in _SECRET_KEY_FRAGMENTS
            ):
                raise ValueError(f"{path}.{key} is a secret-like key")
            _validate_config_snapshot(
                item,
                path=f"{path}.{key}",
                active_container_ids=container_ids,
            )
    finally:
        container_ids.remove(container_id)


def _legal_transition_source_status(
    expected_stage: RegulatoryIndexingStage,
    next_stage: RegulatoryIndexingStage,
    next_status: RegulatoryIndexingJobStatus,
) -> RegulatoryIndexingJobStatus | None:
    if next_status is RegulatoryIndexingJobStatus.QUEUED:
        if _NEXT_STAGE.get(expected_stage) is next_stage or (
            expected_stage is RegulatoryIndexingStage.CONTEXT_WAIT
            and next_stage is RegulatoryIndexingStage.CONTEXT_WAIT
        ):
            return RegulatoryIndexingJobStatus.RUNNING
        return None
    if next_status is RegulatoryIndexingJobStatus.SUCCEEDED:
        if (
            expected_stage is RegulatoryIndexingStage.PUBLISH
            and next_stage is RegulatoryIndexingStage.PUBLISH
        ):
            return RegulatoryIndexingJobStatus.RUNNING
        return None
    if next_status in {
        RegulatoryIndexingJobStatus.FAILED,
        RegulatoryIndexingJobStatus.CANCELLING,
    }:
        return (
            RegulatoryIndexingJobStatus.RUNNING
            if next_stage is expected_stage
            else None
        )
    if next_status is RegulatoryIndexingJobStatus.CANCELLED:
        return (
            RegulatoryIndexingJobStatus.CANCELLING
            if next_stage is expected_stage
            else None
        )
    return None


def create_or_get_regulatory_indexing_job(
    db_session: Session,
    *,
    user_file_id: UUID,
    content_hash: str,
    search_settings_id: int,
    prompt_hash: str,
    config_snapshot: RegulatoryIndexingConfigSnapshot,
    now: datetime.datetime,
) -> RegulatoryIndexingJob:
    """Create one durable row for an immutable file/config revision."""
    _validate_config_snapshot(config_snapshot)
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
    source_status = _legal_transition_source_status(
        expected_stage, next_stage, next_status
    )
    if source_status is None:
        return False
    advanced_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == source_status.value,
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
    expected_generation: int,
) -> RegulatoryIndexingItem | None:
    """Create one result row per canonical chunk, tolerating redelivery."""
    if not _lock_running_regulatory_indexing_job_lease(
        db_session,
        job_id=job_id,
        expected_generation=expected_generation,
    ):
        db_session.commit()
        return None
    proposed_id = uuid4()
    db_session.scalar(
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
    item = db_session.scalar(
        select(RegulatoryIndexingItem).where(
            RegulatoryIndexingItem.job_id == job_id,
            RegulatoryIndexingItem.regulatory_chunk_id == regulatory_chunk_id,
        )
    )
    if item is None:
        raise RuntimeError("regulatory indexing item disappeared during creation")
    if item.request_hash != request_hash:
        db_session.rollback()
        raise ValueError(
            "regulatory indexing item request hash does not match persisted request hash"
        )
    db_session.commit()
    return item


def _lock_running_regulatory_indexing_job_lease(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
) -> bool:
    locked_id = db_session.scalar(
        select(RegulatoryIndexingJob.id)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    return locked_id is not None


def _persist_regulatory_indexing_item_values(
    db_session: Session,
    *,
    item_id: UUID,
    expected_generation: int,
    allowed_statuses: tuple[str, ...],
    values: dict[str, object],
) -> bool:
    locked_job_id = db_session.scalar(
        select(RegulatoryIndexingJob.id)
        .join(
            RegulatoryIndexingItem,
            RegulatoryIndexingItem.job_id == RegulatoryIndexingJob.id,
        )
        .where(
            RegulatoryIndexingItem.id == item_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update(of=RegulatoryIndexingJob)
    )
    if locked_job_id is None:
        db_session.commit()
        return False
    persisted_id = db_session.scalar(
        update(RegulatoryIndexingItem)
        .where(
            RegulatoryIndexingItem.id == item_id,
            RegulatoryIndexingItem.status.in_(allowed_statuses),
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
