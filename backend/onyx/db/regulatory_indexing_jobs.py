import datetime
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
    UserFileStatus,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    UserFile,
)

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
class RegulatoryIndexingExternalMutationLease:
    """Fresh locked state held for the duration of one external mutation."""

    job_id: UUID
    user_file_id: UUID
    lease_generation: int
    stage: RegulatoryIndexingStage
    config_snapshot: RegulatoryIndexingConfigSnapshot
    search_settings_id: int
    search_settings: SearchSettings | None
    user_file_name: str
    user_file_status: UserFileStatus
    user_file_chunk_count: int | None
    regulatory_chunks: tuple[RegulatoryChunk, ...]
    indexing_items: tuple[RegulatoryIndexingItem, ...]


@dataclass(frozen=True)
class RegulatoryIndexingJobClaim:
    job_id: UUID
    stage: RegulatoryIndexingStage
    lease_generation: int


@dataclass(frozen=True)
class RegulatoryIndexingPreparedItem:
    regulatory_chunk_id: str
    request_hash: str
    skip_context: bool


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


def get_regulatory_indexing_job(
    db_session: Session,
    job_id: UUID,
) -> RegulatoryIndexingJob | None:
    return db_session.get(RegulatoryIndexingJob, job_id)


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


def persist_regulatory_indexing_preparation(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    prepare_items: Callable[[], Sequence[RegulatoryIndexingPreparedItem]],
    now: datetime.datetime,
) -> bool:
    """Atomically replace chunks, create items, and finish PREPARING."""

    locked_job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.PREPARING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if locked_job is None:
        db_session.rollback()
        return False

    try:
        prepared_items = list(prepare_items())
        if not prepared_items:
            raise ValueError("regulatory indexing preparation produced no items")
        chunk_ids = [item.regulatory_chunk_id for item in prepared_items]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                "regulatory indexing preparation contains duplicate chunks"
            )
        request_hashes = [item.request_hash for item in prepared_items]
        if len(set(request_hashes)) != len(request_hashes):
            raise ValueError(
                "regulatory indexing preparation contains duplicate hashes"
            )

        db_session.flush()
        canonical_chunk_ids = set(
            db_session.scalars(
                select(RegulatoryChunk.id).where(
                    RegulatoryChunk.user_file_id == locked_job.user_file_id,
                )
            ).all()
        )
        if canonical_chunk_ids != set(chunk_ids):
            raise ValueError(
                "regulatory indexing preparation does not cover canonical chunks"
            )
        db_session.execute(
            delete(RegulatoryIndexingItem).where(
                RegulatoryIndexingItem.job_id == job_id
            )
        )
        for item in prepared_items:
            db_session.add(
                RegulatoryIndexingItem(
                    id=uuid4(),
                    job_id=job_id,
                    regulatory_chunk_id=item.regulatory_chunk_id,
                    request_hash=item.request_hash,
                    status=(
                        RegulatoryIndexingItemStatus.SKIPPED.value
                        if item.skip_context
                        else RegulatoryIndexingItemStatus.PENDING.value
                    ),
                )
            )

        locked_job.status = RegulatoryIndexingJobStatus.QUEUED.value
        locked_job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
        locked_job.attempt_count = 0
        locked_job.heartbeat_at = now
        locked_job.next_retry_at = None
        locked_job.error_code = None
        locked_job.error_message = None
        locked_job.completed_at = None
        locked_job.updated_at = now
        db_session.commit()
        return True
    except Exception:
        db_session.rollback()
        raise


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


def persist_regulatory_indexing_item_vectors(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    item_vectors: Sequence[tuple[UUID, list[float]]],
) -> bool:
    """Persist one validated embedding response as a fenced transaction."""

    if not item_vectors:
        raise ValueError("item_vectors must not be empty")
    item_ids = [item_id for item_id, _vector in item_vectors]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item_vectors contains duplicate item ids")
    for _item_id, vector in item_vectors:
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("item_vectors must contain finite non-empty vectors")

    locked_job_id = db_session.scalar(
        select(RegulatoryIndexingJob.id)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.EMBEDDING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if locked_job_id is None:
        db_session.rollback()
        return False

    eligible_item_ids = set(
        db_session.scalars(
            select(RegulatoryIndexingItem.id).where(
                RegulatoryIndexingItem.job_id == job_id,
                RegulatoryIndexingItem.id.in_(item_ids),
                RegulatoryIndexingItem.status.in_(
                    (
                        RegulatoryIndexingItemStatus.CONTEXT_READY.value,
                        RegulatoryIndexingItemStatus.SKIPPED.value,
                        RegulatoryIndexingItemStatus.EMBEDDED.value,
                    )
                ),
            )
        ).all()
    )
    if eligible_item_ids != set(item_ids):
        db_session.rollback()
        return False

    for item_id, vector in item_vectors:
        db_session.execute(
            update(RegulatoryIndexingItem)
            .where(
                RegulatoryIndexingItem.id == item_id,
                RegulatoryIndexingItem.job_id == job_id,
            )
            .values(
                status=RegulatoryIndexingItemStatus.EMBEDDED.value,
                vector=vector,
                error_code=None,
                error_message=None,
                updated_at=func.now(),
            )
        )
    db_session.commit()
    return True


@contextmanager
def regulatory_indexing_external_mutation_lease(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
) -> Iterator[RegulatoryIndexingExternalMutationLease | None]:
    """Fence an external mutation with locked current job and file rows."""

    locked_job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if locked_job is None:
        db_session.rollback()
        yield None
        return

    locked_user_file = db_session.scalar(
        select(UserFile).where(UserFile.id == locked_job.user_file_id).with_for_update()
    )
    if locked_user_file is None:
        db_session.rollback()
        yield None
        return

    locked_search_settings = db_session.scalar(
        select(SearchSettings)
        .where(SearchSettings.id == locked_job.search_settings_id)
        .with_for_update()
    )
    regulatory_chunks = tuple(
        db_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == locked_user_file.id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            .with_for_update()
        ).all()
    )
    indexing_items = tuple(
        db_session.scalars(
            select(RegulatoryIndexingItem)
            .where(RegulatoryIndexingItem.job_id == locked_job.id)
            .with_for_update()
        ).all()
    )
    lease = RegulatoryIndexingExternalMutationLease(
        job_id=locked_job.id,
        user_file_id=locked_user_file.id,
        lease_generation=locked_job.lease_generation,
        stage=RegulatoryIndexingStage(locked_job.stage),
        config_snapshot=cast(
            RegulatoryIndexingConfigSnapshot,
            deepcopy(locked_job.config_snapshot),
        ),
        search_settings_id=locked_job.search_settings_id,
        search_settings=locked_search_settings,
        user_file_name=locked_user_file.name,
        user_file_status=locked_user_file.status,
        user_file_chunk_count=locked_user_file.chunk_count,
        regulatory_chunks=regulatory_chunks,
        indexing_items=indexing_items,
    )
    try:
        yield lease
        heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
        locked_job.heartbeat_at = heartbeat_at
        locked_job.updated_at = heartbeat_at
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise


def complete_regulatory_indexing_user_file(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    chunk_count: int,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    """Mark a published file complete only for the current PUBLISH lease."""

    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    locked_job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.PUBLISH.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if locked_job is None:
        db_session.rollback()
        return False

    completed_id = db_session.scalar(
        update(UserFile)
        .where(
            UserFile.id == locked_job.user_file_id,
            UserFile.status.in_((UserFileStatus.INDEXING, UserFileStatus.COMPLETED)),
        )
        .values(
            status=UserFileStatus.COMPLETED,
            chunk_count=chunk_count,
            last_project_sync_at=now,
        )
        .returning(UserFile.id)
    )
    if completed_id is None:
        db_session.rollback()
        return False
    if commit:
        db_session.commit()
    else:
        db_session.flush()
    return True


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
