import datetime
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingCancellationIntent,
    RegulatoryIndexingCancellationPhase,
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingProviderCleanupPhase,
    RegulatoryIndexingProviderCleanupState,
    RegulatoryIndexingStage,
    RegulatoryIndexingSubmissionState,
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
_MAX_ERROR_CODE_LENGTH = 128
_ACTIVE_REGULATORY_INDEXING_JOB_STATUSES = (
    RegulatoryIndexingJobStatus.QUEUED.value,
    RegulatoryIndexingJobStatus.RUNNING.value,
    RegulatoryIndexingJobStatus.RETRY_WAIT.value,
    RegulatoryIndexingJobStatus.CANCELLING.value,
)
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


@dataclass
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
    _db_session: Session = field(repr=False)
    _locked_job: RegulatoryIndexingJob = field(repr=False)
    _committed: bool = field(default=False, init=False, repr=False)

    @property
    def committed(self) -> bool:
        return self._committed

    def commit(self) -> None:
        """Refresh the heartbeat and synchronously commit the locked transaction."""

        if self._committed:
            raise RuntimeError("external mutation transaction is already committed")
        heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
        self._locked_job.heartbeat_at = heartbeat_at
        self._locked_job.updated_at = heartbeat_at
        self._db_session.commit()
        self._committed = True


@dataclass(frozen=True)
class RegulatoryIndexingJobClaim:
    job_id: UUID
    stage: RegulatoryIndexingStage
    lease_generation: int
    recovery_token: UUID


@dataclass(frozen=True)
class RegulatoryIndexingCancellationDelivery:
    job_id: UUID
    expected_generation: int


@dataclass(frozen=True)
class RegulatoryIndexingProviderCleanupClaim:
    job_id: UUID
    cleanup_generation: int
    cleanup_token: UUID


@dataclass(frozen=True)
class UserFileDeletionCleanupPlan:
    ready_to_delete: bool
    deliveries: tuple[RegulatoryIndexingCancellationDelivery, ...]


@dataclass(frozen=True)
class RegulatoryIndexingPreparedItem:
    regulatory_chunk_id: str
    request_hash: str
    skip_context: bool


@dataclass(frozen=True)
class RegulatoryIndexingRuntime:
    job: RegulatoryIndexingJob
    user_file: UserFile
    search_settings: SearchSettings | None
    regulatory_chunks: tuple[RegulatoryChunk, ...]
    indexing_items: tuple[RegulatoryIndexingItem, ...]


def _lock_user_file_for_regulatory_job(
    db_session: Session,
    job_id: UUID,
) -> UserFile | None:
    """Lock the job's parent before the job to keep lifecycle lock order stable."""

    user_file_id = db_session.scalar(
        select(RegulatoryIndexingJob.user_file_id).where(
            RegulatoryIndexingJob.id == job_id
        )
    )
    if user_file_id is None:
        return None
    return db_session.scalar(
        select(UserFile).where(UserFile.id == user_file_id).with_for_update()
    )


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
        if (
            _NEXT_STAGE.get(expected_stage) is next_stage
            or (
                expected_stage is RegulatoryIndexingStage.CONTEXT_WAIT
                and next_stage is RegulatoryIndexingStage.CONTEXT_WAIT
            )
            or (
                expected_stage is RegulatoryIndexingStage.CONTEXT_SUBMIT
                and next_stage is RegulatoryIndexingStage.EMBEDDING
            )
            or (
                expected_stage is RegulatoryIndexingStage.CONTEXT_SUBMIT
                and next_stage is RegulatoryIndexingStage.CONTEXT_SUBMIT
            )
            or (
                expected_stage is RegulatoryIndexingStage.CONTEXT_APPLY
                and next_stage is RegulatoryIndexingStage.CONTEXT_SUBMIT
            )
            or (
                expected_stage is RegulatoryIndexingStage.EMBEDDING
                and next_stage is RegulatoryIndexingStage.EMBEDDING
            )
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
    chunk_generation_hash: str,
    config_snapshot: RegulatoryIndexingConfigSnapshot,
    now: datetime.datetime,
) -> RegulatoryIndexingJob:
    """Create one active job, or reuse the file's existing active generation.

    Once every earlier generation is terminal, a different immutable identity may
    create a new reindex job. An identical terminal identity remains idempotent.
    """
    _validate_config_snapshot(config_snapshot)
    if config_snapshot.get("input_content_hash") != content_hash:
        raise ValueError("config snapshot input hash does not match content hash")
    if config_snapshot.get("input_hash_version") not in {
        "legacy-v1",
        "canonical-v2",
        "chunk-rows-v3",
    }:
        raise ValueError("config snapshot input hash version is unsupported")
    if config_snapshot.get("chunk_generation_hash") != chunk_generation_hash:
        raise ValueError(
            "config snapshot generation hash does not match chunk generation hash"
        )
    if len(chunk_generation_hash) != 64 or any(
        character not in "0123456789abcdef" for character in chunk_generation_hash
    ):
        raise ValueError("chunk generation hash must be a lowercase SHA-256 hash")
    locked_user_file = db_session.scalar(
        select(UserFile).where(UserFile.id == user_file_id).with_for_update()
    )
    if locked_user_file is None:
        db_session.rollback()
        raise ValueError("cannot create regulatory indexing job for missing user file")
    if locked_user_file.status is UserFileStatus.DELETING:
        db_session.rollback()
        raise ValueError("cannot create regulatory indexing job for deleting user file")
    if locked_user_file.status is UserFileStatus.CANCELED:
        db_session.rollback()
        raise ValueError(
            "cannot create regulatory indexing job for cancelled user file"
        )
    active_job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.user_file_id == user_file_id,
            RegulatoryIndexingJob.status.in_(_ACTIVE_REGULATORY_INDEXING_JOB_STATUSES),
        )
        .order_by(
            RegulatoryIndexingJob.updated_at.desc(),
            RegulatoryIndexingJob.created_at.desc(),
            RegulatoryIndexingJob.id.desc(),
        )
        .limit(1)
        .with_for_update()
    )
    if active_job is not None:
        if (
            active_job.chunk_generation_hash != chunk_generation_hash
            and active_job.status != RegulatoryIndexingJobStatus.CANCELLING.value
        ):
            active_job.status = RegulatoryIndexingJobStatus.CANCELLING.value
            active_job.cancellation_intent = (
                RegulatoryIndexingCancellationIntent.SUPERSEDE.value
            )
            active_job.cancellation_phase = (
                RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
                if active_job.remote_vertex_job_name
                or active_job.remote_openrouter_batch_id
                else RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
            )
            active_job.lease_generation += 1
            active_job.recovery_token = None
            active_job.attempt_count = 0
            active_job.next_retry_at = None
            active_job.error_code = None
            active_job.error_message = None
            active_job.completed_at = None
            active_job.heartbeat_at = now
            active_job.updated_at = now
        active_job_id = active_job.id
        db_session.commit()
        persisted_active_job = db_session.get(RegulatoryIndexingJob, active_job_id)
        if persisted_active_job is None:
            raise RuntimeError(
                f"active regulatory indexing job {active_job_id} disappeared"
            )
        return persisted_active_job
    db_session.scalar(
        select(SearchSettings)
        .where(SearchSettings.id == search_settings_id)
        .with_for_update()
    )
    proposed_id = uuid4()
    created_id = db_session.scalar(
        pg_insert(RegulatoryIndexingJob)
        .values(
            id=proposed_id,
            user_file_id=user_file_id,
            content_hash=content_hash,
            chunk_generation_hash=chunk_generation_hash,
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
            RegulatoryIndexingJob.chunk_generation_hash == chunk_generation_hash,
        )
    )
    if job_id is None:
        raise RuntimeError("regulatory indexing job disappeared during creation")
    persisted_before_commit = db_session.get(RegulatoryIndexingJob, job_id)
    if persisted_before_commit is None:
        raise RuntimeError(f"regulatory indexing job {job_id} was not persisted")
    if (
        persisted_before_commit.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
        and locked_user_file.status
        in {UserFileStatus.PROCESSING, UserFileStatus.INDEXING}
    ):
        locked_user_file.status = UserFileStatus.COMPLETED
    db_session.commit()
    job = db_session.get(RegulatoryIndexingJob, job_id)
    if job is None:
        raise RuntimeError(f"regulatory indexing job {job_id} was not persisted")
    return job


def supersede_regulatory_indexing_job_for_generation_drift(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    current_chunk_generation_hash: str,
    now: datetime.datetime,
) -> RegulatoryIndexingCancellationDelivery | None:
    """Fence a claimed stale generation and start its durable cleanup."""

    if len(current_chunk_generation_hash) != 64 or any(
        character not in "0123456789abcdef"
        for character in current_chunk_generation_hash
    ):
        raise ValueError("chunk generation hash must be a lowercase SHA-256 hash")
    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None:
        db_session.rollback()
        return None
    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if (
        job is None
        or job.user_file_id != locked_user_file.id
        or job.chunk_generation_hash == current_chunk_generation_hash
        or locked_user_file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}
    ):
        db_session.rollback()
        return None

    job.status = RegulatoryIndexingJobStatus.CANCELLING.value
    job.cancellation_intent = RegulatoryIndexingCancellationIntent.SUPERSEDE.value
    job.cancellation_phase = (
        RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
        if job.remote_vertex_job_name or job.remote_openrouter_batch_id
        else RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
    )
    job.lease_generation += 1
    job.recovery_token = None
    job.attempt_count = 0
    job.next_retry_at = None
    job.error_code = None
    job.error_message = None
    job.completed_at = None
    job.heartbeat_at = now
    job.updated_at = now
    next_generation = job.lease_generation
    db_session.commit()
    return RegulatoryIndexingCancellationDelivery(
        job_id=job_id,
        expected_generation=next_generation,
    )


def get_regulatory_indexing_job(
    db_session: Session,
    job_id: UUID,
) -> RegulatoryIndexingJob | None:
    return db_session.get(RegulatoryIndexingJob, job_id)


def get_regulatory_indexing_runtime(
    db_session: Session,
    job_id: UUID,
) -> RegulatoryIndexingRuntime | None:
    job = db_session.get(RegulatoryIndexingJob, job_id)
    if job is None:
        return None
    user_file = db_session.get(UserFile, job.user_file_id)
    if user_file is None:
        return None
    return RegulatoryIndexingRuntime(
        job=job,
        user_file=user_file,
        search_settings=db_session.get(SearchSettings, job.search_settings_id),
        regulatory_chunks=tuple(
            db_session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == user_file.id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            ).all()
        ),
        indexing_items=tuple(
            db_session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job.id
                )
            ).all()
        ),
    )


def has_active_regulatory_indexing_jobs_for_search_settings(
    db_session: Session,
    search_settings_id: int,
) -> bool:
    return bool(
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id) > 0).where(
                RegulatoryIndexingJob.search_settings_id == search_settings_id,
                RegulatoryIndexingJob.status.in_(
                    _ACTIVE_REGULATORY_INDEXING_JOB_STATUSES
                ),
            )
        )
    )


def mark_regulatory_indexing_user_file(
    db_session: Session,
    *,
    job_id: UUID,
) -> bool:
    user_file_id = db_session.scalar(
        select(RegulatoryIndexingJob.user_file_id).where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status.not_in(
                (
                    RegulatoryIndexingJobStatus.SUCCEEDED.value,
                    RegulatoryIndexingJobStatus.FAILED.value,
                    RegulatoryIndexingJobStatus.CANCELLED.value,
                )
            ),
        )
    )
    if user_file_id is None:
        db_session.rollback()
        return False
    updated_id = db_session.scalar(
        update(UserFile)
        .where(
            UserFile.id == user_file_id,
            UserFile.status.in_((UserFileStatus.PROCESSING, UserFileStatus.INDEXING)),
        )
        .values(status=UserFileStatus.INDEXING)
        .returning(UserFile.id)
    )
    db_session.commit()
    return updated_id is not None


def request_user_file_deletion_cleanup(
    db_session: Session,
    *,
    user_file_id: UUID,
    now: datetime.datetime,
) -> UserFileDeletionCleanupPlan:
    """Tombstone a file and clean one durable generation at a time."""

    locked_user_file = db_session.scalar(
        select(UserFile).where(UserFile.id == user_file_id).with_for_update()
    )
    if locked_user_file is None:
        db_session.rollback()
        return UserFileDeletionCleanupPlan(ready_to_delete=True, deliveries=())

    locked_jobs = tuple(
        db_session.scalars(
            select(RegulatoryIndexingJob)
            .where(RegulatoryIndexingJob.user_file_id == user_file_id)
            .order_by(
                RegulatoryIndexingJob.updated_at.desc(),
                RegulatoryIndexingJob.created_at.desc(),
                RegulatoryIndexingJob.id.desc(),
            )
            .with_for_update()
        ).all()
    )
    locked_user_file.status = UserFileStatus.DELETING
    deliveries: list[RegulatoryIndexingCancellationDelivery] = []
    active_jobs = tuple(
        job
        for job in locked_jobs
        if job.status in _ACTIVE_REGULATORY_INDEXING_JOB_STATUSES
    )
    if len(active_jobs) > 1:
        db_session.rollback()
        raise RuntimeError("user file has multiple active regulatory indexing jobs")
    for terminal_job in locked_jobs:
        if (
            terminal_job.status == RegulatoryIndexingJobStatus.CANCELLED.value
            and terminal_job.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.NONE.value
        ):
            _schedule_provider_cleanup(terminal_job, now=now, cancel_first=True)
    job = (
        active_jobs[0]
        if active_jobs
        else next(
            (
                candidate
                for candidate in locked_jobs
                if candidate.status != RegulatoryIndexingJobStatus.CANCELLED.value
            ),
            None,
        )
    )
    if job is not None:
        entering_delete_cancellation = (
            job.status != RegulatoryIndexingJobStatus.CANCELLING.value
            or job.cancellation_intent
            != RegulatoryIndexingCancellationIntent.USER_DELETE.value
        )
        if entering_delete_cancellation:
            job.status = RegulatoryIndexingJobStatus.CANCELLING.value
            job.cancellation_intent = (
                RegulatoryIndexingCancellationIntent.USER_DELETE.value
            )
            job.cancellation_phase = (
                RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
                if job.remote_vertex_job_name or job.remote_openrouter_batch_id
                else RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
            )
            job.lease_generation += 1
            job.recovery_token = None
            job.attempt_count = 0
            job.next_retry_at = None
            job.error_code = None
            job.error_message = None
            job.completed_at = None
            job.heartbeat_at = now
            job.updated_at = now
        if job.next_retry_at is None or job.next_retry_at <= now:
            deliveries.append(
                RegulatoryIndexingCancellationDelivery(
                    job_id=job.id,
                    expected_generation=job.lease_generation,
                )
            )

    db_session.commit()
    return UserFileDeletionCleanupPlan(
        ready_to_delete=(
            job is None
            and all(
                candidate.status == RegulatoryIndexingJobStatus.CANCELLED.value
                and candidate.provider_cleanup_state
                == RegulatoryIndexingProviderCleanupState.SUCCEEDED.value
                and candidate.provider_cleanup_phase
                == RegulatoryIndexingProviderCleanupPhase.COMPLETE.value
                for candidate in locked_jobs
            )
        ),
        deliveries=tuple(deliveries),
    )


def _update_vertex_submission_state(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    allowed_states: tuple[RegulatoryIndexingSubmissionState, ...],
    next_state: RegulatoryIndexingSubmissionState,
    now: datetime.datetime,
    extra_values: dict[str, object] | None = None,
    commit: bool = True,
) -> bool:
    values: dict[str, object] = {
        "vertex_submission_key": submission_key,
        "vertex_submission_state": next_state.value,
        "heartbeat_at": now,
        "updated_at": now,
    }
    if extra_values:
        values.update(extra_values)
    updated_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.vertex_submission_state.in_(
                tuple(state.value for state in allowed_states)
            ),
            or_(
                RegulatoryIndexingJob.vertex_submission_key.is_(None),
                RegulatoryIndexingJob.vertex_submission_key == submission_key,
            ),
        )
        .values(**values)
        .returning(RegulatoryIndexingJob.id)
    )
    if commit:
        db_session.commit()
    else:
        db_session.flush()
    return updated_id is not None


def record_vertex_submission_intent(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    submission_attempt: int,
    now: datetime.datetime,
) -> bool:
    """Commit deterministic create intent before the provider create call."""

    if submission_attempt < 1:
        raise ValueError("Vertex submission attempt must be positive")
    updated_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.vertex_submission_state
            == RegulatoryIndexingSubmissionState.NONE.value,
            RegulatoryIndexingJob.vertex_submission_key.is_(None),
            RegulatoryIndexingJob.vertex_submission_attempt_count
            == submission_attempt - 1,
        )
        .values(
            vertex_submission_key=submission_key,
            vertex_submission_state=RegulatoryIndexingSubmissionState.SUBMITTING.value,
            vertex_submission_attempt_count=submission_attempt,
            vertex_submission_charged=False,
            vertex_reconcile_miss_count=0,
            vertex_reconcile_until=None,
            heartbeat_at=now,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return updated_id is not None


def _charge_vertex_submission_items(
    db_session: Session,
    *,
    job_id: UUID,
    request_hashes: Sequence[str],
    now: datetime.datetime,
) -> None:
    hashes = tuple(request_hashes)
    if not hashes or len(set(hashes)) != len(hashes):
        raise ValueError("charged Vertex request hashes must be unique and non-empty")
    db_session.execute(
        update(RegulatoryIndexingItem)
        .where(
            RegulatoryIndexingItem.job_id == job_id,
            RegulatoryIndexingItem.status == RegulatoryIndexingItemStatus.PENDING.value,
            RegulatoryIndexingItem.request_hash.in_(hashes),
        )
        .values(
            context_attempt_count=RegulatoryIndexingItem.context_attempt_count + 1,
            updated_at=now,
        )
    )


def require_vertex_submission_reconciliation(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    request_hashes: Sequence[str],
    reconcile_until: datetime.datetime,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    persisted = _update_vertex_submission_state(
        db_session,
        job_id=job_id,
        expected_generation=expected_generation,
        submission_key=submission_key,
        allowed_states=(RegulatoryIndexingSubmissionState.SUBMITTING,),
        next_state=RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED,
        now=now,
        extra_values={
            "vertex_submission_charged": True,
            "vertex_reconcile_miss_count": 0,
            "vertex_reconcile_until": reconcile_until,
        },
        commit=False,
    )
    if persisted:
        _charge_vertex_submission_items(
            db_session,
            job_id=job_id,
            request_hashes=request_hashes,
            now=now,
        )
        if commit:
            db_session.commit()
        else:
            db_session.flush()
    return persisted


def record_vertex_reconciliation_miss(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    now: datetime.datetime,
) -> bool:
    updated_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.vertex_submission_state
            == RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED.value,
            RegulatoryIndexingJob.vertex_reconcile_until.is_not(None),
            RegulatoryIndexingJob.vertex_reconcile_until > now,
        )
        .values(
            vertex_reconcile_miss_count=(
                RegulatoryIndexingJob.vertex_reconcile_miss_count + 1
            ),
            heartbeat_at=now,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return updated_id is not None


def record_vertex_submission_not_sent(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    return _update_vertex_submission_state(
        db_session,
        job_id=job_id,
        expected_generation=expected_generation,
        submission_key=submission_key,
        allowed_states=(RegulatoryIndexingSubmissionState.SUBMITTING,),
        next_state=RegulatoryIndexingSubmissionState.NONE,
        now=now,
        extra_values={
            "vertex_submission_key": None,
            "vertex_submission_charged": False,
            "vertex_reconcile_miss_count": 0,
            "vertex_reconcile_until": None,
        },
        commit=commit,
    )


def record_vertex_submission(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    request_hashes: Sequence[str],
    charge_items: bool,
    remote_job_name: str,
    input_uri: str | None,
    output_uri: str | None,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    persisted = _update_vertex_submission_state(
        db_session,
        job_id=job_id,
        expected_generation=expected_generation,
        submission_key=submission_key,
        allowed_states=(
            RegulatoryIndexingSubmissionState.SUBMITTING,
            RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED,
        ),
        next_state=RegulatoryIndexingSubmissionState.SUBMITTED,
        now=now,
        extra_values={
            "remote_vertex_job_name": remote_job_name,
            "vertex_input_uri": input_uri,
            "vertex_output_uri": output_uri,
            "vertex_submission_charged": True,
            "vertex_reconcile_until": None,
        },
        commit=False,
    )
    if persisted and charge_items:
        _charge_vertex_submission_items(
            db_session,
            job_id=job_id,
            request_hashes=request_hashes,
            now=now,
        )
    if persisted:
        if commit:
            db_session.commit()
        else:
            db_session.flush()
    return persisted


def persist_vertex_poll_state(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    remote_job_name: str,
    output_uri: str | None,
    now: datetime.datetime,
) -> bool:
    updated_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.CONTEXT_WAIT.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.remote_vertex_job_name == remote_job_name,
        )
        .values(vertex_output_uri=output_uri, heartbeat_at=now, updated_at=now)
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return updated_id is not None


def reset_vertex_submission_for_partial_retry(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    now: datetime.datetime,
) -> bool:
    updated_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.CONTEXT_APPLY.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .values(
            vertex_submission_key=None,
            vertex_submission_state=RegulatoryIndexingSubmissionState.NONE.value,
            remote_vertex_job_name=None,
            vertex_input_uri=None,
            vertex_output_uri=None,
            vertex_submission_charged=False,
            vertex_reconcile_miss_count=0,
            vertex_reconcile_until=None,
            heartbeat_at=now,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return updated_id is not None


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
        RegulatoryIndexingJobStatus.CANCELLING.value,
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
                and_(
                    RegulatoryIndexingJob.status
                    == RegulatoryIndexingJobStatus.CANCELLING.value,
                    or_(
                        RegulatoryIndexingJob.next_retry_at.is_(None),
                        RegulatoryIndexingJob.next_retry_at <= now,
                    ),
                ),
            ),
        )
        .values(
            status=case(
                (
                    RegulatoryIndexingJob.status
                    == RegulatoryIndexingJobStatus.CANCELLING.value,
                    RegulatoryIndexingJobStatus.CANCELLING.value,
                ),
                else_=RegulatoryIndexingJobStatus.RUNNING.value,
            ),
            lease_generation=RegulatoryIndexingJob.lease_generation + 1,
            recovery_token=None,
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
    values: dict[str, object] = {
        "status": next_status.value,
        "stage": next_stage.value,
        "attempt_count": 0,
        "heartbeat_at": now,
        "next_retry_at": None,
        "error_code": (
            error_code[:_MAX_ERROR_CODE_LENGTH] if error_code is not None else None
        ),
        "error_message": (
            error_message[:_MAX_ERROR_MESSAGE_LENGTH]
            if error_message is not None
            else None
        ),
        "updated_at": now,
        "completed_at": (
            now
            if next_status
            in {
                RegulatoryIndexingJobStatus.SUCCEEDED,
                RegulatoryIndexingJobStatus.FAILED,
                RegulatoryIndexingJobStatus.CANCELLED,
            }
            else None
        ),
    }
    if next_status is RegulatoryIndexingJobStatus.CANCELLING:
        values["cancellation_phase"] = case(
            (
                or_(
                    RegulatoryIndexingJob.remote_vertex_job_name.is_not(None),
                    RegulatoryIndexingJob.remote_openrouter_batch_id.is_not(None),
                ),
                RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value,
            ),
            else_=RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value,
        )
    advanced_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == source_status.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .values(**values)
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
            error_code=error_code[:_MAX_ERROR_CODE_LENGTH],
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
    cancellable = and_(
        RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.CANCELLING.value,
        or_(
            and_(
                RegulatoryIndexingJob.next_retry_at.is_(None),
                RegulatoryIndexingJob.heartbeat_at.is_not(None),
                RegulatoryIndexingJob.heartbeat_at <= stale_before,
            ),
            and_(
                RegulatoryIndexingJob.next_retry_at.is_not(None),
                RegulatoryIndexingJob.next_retry_at <= claimed_at,
            ),
        ),
    )
    jobs = list(
        db_session.scalars(
            select(RegulatoryIndexingJob)
            .where(or_(stale_queued_or_running, due_retry, cancellable))
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
        recovery_token = uuid4()
        if job.status != RegulatoryIndexingJobStatus.CANCELLING.value:
            job.status = RegulatoryIndexingJobStatus.RUNNING.value
        job.lease_generation += 1
        job.recovery_token = recovery_token
        job.heartbeat_at = claimed_at
        job.next_retry_at = None
        claims.append(
            RegulatoryIndexingJobClaim(
                job_id=job.id,
                stage=RegulatoryIndexingStage(job.stage),
                lease_generation=job.lease_generation,
                recovery_token=recovery_token,
            )
        )
    if jobs:
        db_session.commit()
    return claims


def consume_preclaimed_regulatory_indexing_delivery(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    recovery_token: UUID,
    consumed_at: datetime.datetime,
) -> bool:
    """Atomically consume the scanner-issued one-use delivery token."""

    consumed_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status.in_(
                (
                    RegulatoryIndexingJobStatus.RUNNING.value,
                    RegulatoryIndexingJobStatus.CANCELLING.value,
                )
            ),
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.recovery_token == recovery_token,
        )
        .values(
            recovery_token=None,
            heartbeat_at=consumed_at,
            updated_at=consumed_at,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return consumed_id is not None


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
    resolved_input_hash_version: str,
    now: datetime.datetime,
) -> bool:
    """Atomically replace chunks, create items, and finish PREPARING."""

    if resolved_input_hash_version not in {
        "legacy-v1",
        "canonical-v2",
        "chunk-rows-v3",
    }:
        raise ValueError("resolved input hash version is unsupported")

    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None or locked_user_file.status not in {
        UserFileStatus.PROCESSING,
        UserFileStatus.INDEXING,
        UserFileStatus.FAILED,
        UserFileStatus.CHUNKED,
    }:
        db_session.rollback()
        return False

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
    if locked_job.user_file_id != locked_user_file.id:
        db_session.rollback()
        return False
    persisted_input_hash_version = locked_job.config_snapshot.get("input_hash_version")
    if persisted_input_hash_version not in {
        "legacy-or-canonical",
        resolved_input_hash_version,
    }:
        db_session.rollback()
        raise ValueError("resolved input hash version conflicts with snapshot")

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

        locked_job.config_snapshot = {
            **locked_job.config_snapshot,
            "input_hash_version": resolved_input_hash_version,
        }
        locked_job.status = RegulatoryIndexingJobStatus.QUEUED.value
        locked_job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
        locked_job.attempt_count = 0
        locked_job.heartbeat_at = now
        locked_job.next_retry_at = None
        locked_job.error_code = None
        locked_job.error_message = None
        locked_job.completed_at = None
        locked_job.updated_at = now
        locked_user_file.status = UserFileStatus.INDEXING
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


def _lock_openrouter_embedding_job(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
) -> RegulatoryIndexingJob | None:
    return db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == RegulatoryIndexingStage.EMBEDDING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )


def record_openrouter_submission_intent(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    submission_attempt: int,
    active_item_ids: Sequence[UUID],
    now: datetime.datetime,
) -> bool:
    if submission_attempt < 1 or not active_item_ids:
        raise ValueError("OpenRouter submission intent is incomplete")
    if len(set(active_item_ids)) != len(active_item_ids):
        raise ValueError("OpenRouter submission contains duplicate item ids")
    job = _lock_openrouter_embedding_job(
        db_session, job_id=job_id, expected_generation=expected_generation
    )
    if job is None or job.openrouter_submission_state != "NONE":
        db_session.rollback()
        return False
    eligible_ids = set(
        db_session.scalars(
            select(RegulatoryIndexingItem.id).where(
                RegulatoryIndexingItem.job_id == job_id,
                RegulatoryIndexingItem.id.in_(active_item_ids),
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
    if eligible_ids != set(active_item_ids):
        db_session.rollback()
        return False
    job.openrouter_submission_key = submission_key
    job.openrouter_submission_state = "SUBMITTING"
    job.openrouter_submission_attempt_count = submission_attempt
    job.openrouter_submission_charged = False
    job.openrouter_reconcile_miss_count = 0
    job.openrouter_reconcile_until = None
    job.openrouter_completion_deadline = None
    job.remote_openrouter_batch_id = None
    job.openrouter_active_item_ids = [str(item_id) for item_id in active_item_ids]
    job.heartbeat_at = now
    job.updated_at = now
    db_session.commit()
    return True


def record_openrouter_submission_ambiguous(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    job = _lock_openrouter_embedding_job(
        db_session, job_id=job_id, expected_generation=expected_generation
    )
    if (
        job is None
        or job.openrouter_submission_key != submission_key
        or job.openrouter_submission_state
        not in {
            "SUBMITTING",
            "RECONCILE_REQUIRED",
            "RECONCILED_ABSENT",
            "MANUAL_RECONCILE_REQUIRED",
        }
    ):
        db_session.rollback()
        return False
    if not job.openrouter_submission_charged:
        active_ids = [UUID(item_id) for item_id in job.openrouter_active_item_ids]
        result = db_session.execute(
            update(RegulatoryIndexingItem)
            .where(
                RegulatoryIndexingItem.job_id == job_id,
                RegulatoryIndexingItem.id.in_(active_ids),
            )
            .values(
                embedding_attempt_count=(
                    RegulatoryIndexingItem.embedding_attempt_count + 1
                ),
                updated_at=func.now(),
            )
        )
        if result.rowcount != len(active_ids):  # ty: ignore[unresolved-attribute]
            db_session.rollback()
            return False
        job.openrouter_submission_charged = True
    job.openrouter_submission_state = "MANUAL_RECONCILE_REQUIRED"
    job.openrouter_reconcile_until = None
    job.heartbeat_at = now
    job.updated_at = now
    if commit:
        db_session.commit()
    return True


def record_openrouter_submission_not_sent(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    job = _lock_openrouter_embedding_job(
        db_session, job_id=job_id, expected_generation=expected_generation
    )
    if (
        job is None
        or job.openrouter_submission_key != submission_key
        or job.openrouter_submission_state != "SUBMITTING"
    ):
        db_session.rollback()
        return False
    job.openrouter_submission_key = None
    job.openrouter_submission_state = "NONE"
    job.openrouter_submission_charged = False
    job.openrouter_reconcile_until = None
    job.openrouter_completion_deadline = None
    job.remote_openrouter_batch_id = None
    job.openrouter_active_item_ids = []
    job.heartbeat_at = now
    job.updated_at = now
    if commit:
        db_session.commit()
    return True


def record_openrouter_submission(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    submission_key: str,
    remote_batch_id: str,
    completion_deadline: datetime.datetime,
    charge_items: bool,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    job = _lock_openrouter_embedding_job(
        db_session, job_id=job_id, expected_generation=expected_generation
    )
    if (
        job is None
        or job.openrouter_submission_key != submission_key
        or job.openrouter_submission_state != "SUBMITTING"
        or not job.openrouter_active_item_ids
    ):
        db_session.rollback()
        return False
    if charge_items and not job.openrouter_submission_charged:
        active_ids = [UUID(item_id) for item_id in job.openrouter_active_item_ids]
        result = db_session.execute(
            update(RegulatoryIndexingItem)
            .where(
                RegulatoryIndexingItem.job_id == job_id,
                RegulatoryIndexingItem.id.in_(active_ids),
            )
            .values(
                embedding_attempt_count=(
                    RegulatoryIndexingItem.embedding_attempt_count + 1
                ),
                updated_at=func.now(),
            )
        )
        if result.rowcount != len(active_ids):  # ty: ignore[unresolved-attribute]
            db_session.rollback()
            return False
        job.openrouter_submission_charged = True
    job.openrouter_submission_state = "SUBMITTED"
    job.remote_openrouter_batch_id = remote_batch_id
    job.openrouter_completion_deadline = completion_deadline
    job.openrouter_reconcile_until = None
    job.heartbeat_at = now
    job.updated_at = now
    if commit:
        db_session.commit()
    return True


def apply_openrouter_embedding_batch(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    remote_batch_id: str,
    item_vectors: Sequence[tuple[UUID, list[float]]],
    failed_item_ids: Sequence[UUID],
    now: datetime.datetime,
) -> bool:
    """Atomically apply one complete provider batch and clear its remote state."""

    successful_ids = [item_id for item_id, _vector in item_vectors]
    covered_ids = [*successful_ids, *failed_item_ids]
    if len(set(covered_ids)) != len(covered_ids):
        raise ValueError("OpenRouter Batch results contain duplicate item ids")
    job = _lock_openrouter_embedding_job(
        db_session, job_id=job_id, expected_generation=expected_generation
    )
    if (
        job is None
        or job.openrouter_submission_state != "SUBMITTED"
        or job.remote_openrouter_batch_id != remote_batch_id
        or set(job.openrouter_active_item_ids)
        != {str(item_id) for item_id in covered_ids}
    ):
        db_session.rollback()
        return False
    for item_id, vector in item_vectors:
        if not vector or any(not math.isfinite(value) for value in vector):
            db_session.rollback()
            raise ValueError("OpenRouter Batch returned an invalid vector")
        persisted_id = db_session.scalar(
            update(RegulatoryIndexingItem)
            .where(
                RegulatoryIndexingItem.id == item_id,
                RegulatoryIndexingItem.job_id == job_id,
                RegulatoryIndexingItem.status.in_(
                    (
                        RegulatoryIndexingItemStatus.CONTEXT_READY.value,
                        RegulatoryIndexingItemStatus.SKIPPED.value,
                        RegulatoryIndexingItemStatus.EMBEDDED.value,
                    )
                ),
            )
            .values(
                status=RegulatoryIndexingItemStatus.EMBEDDED.value,
                vector=vector,
                error_code=None,
                error_message=None,
                updated_at=func.now(),
            )
            .returning(RegulatoryIndexingItem.id)
        )
        if persisted_id is None:
            db_session.rollback()
            return False
    job.openrouter_submission_key = None
    job.openrouter_submission_state = "NONE"
    job.openrouter_submission_charged = False
    job.openrouter_reconcile_miss_count = 0
    job.openrouter_reconcile_until = None
    job.openrouter_completion_deadline = None
    job.remote_openrouter_batch_id = None
    job.openrouter_active_item_ids = []
    job.heartbeat_at = now
    job.updated_at = now
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

    parent_ids = db_session.execute(
        select(
            RegulatoryIndexingJob.user_file_id,
            RegulatoryIndexingJob.search_settings_id,
        ).where(RegulatoryIndexingJob.id == job_id)
    ).one_or_none()
    if parent_ids is None:
        db_session.rollback()
        yield None
        return

    locked_search_settings = db_session.scalar(
        select(SearchSettings)
        .where(SearchSettings.id == parent_ids.search_settings_id)
        .with_for_update()
    )
    locked_user_file = db_session.scalar(
        select(UserFile).where(UserFile.id == parent_ids.user_file_id).with_for_update()
    )
    if locked_user_file is None:
        db_session.rollback()
        yield None
        return

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
    if locked_job.user_file_id != locked_user_file.id or (
        locked_search_settings is not None
        and locked_job.search_settings_id != locked_search_settings.id
    ):
        db_session.rollback()
        yield None
        return
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
        _db_session=db_session,
        _locked_job=locked_job,
    )
    try:
        yield lease
    finally:
        if not lease.committed:
            db_session.rollback()


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
    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None or locked_user_file.status not in {
        UserFileStatus.INDEXING,
        UserFileStatus.COMPLETED,
    }:
        if commit:
            db_session.rollback()
        return False
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
        if commit:
            db_session.rollback()
        return False
    if locked_job.user_file_id != locked_user_file.id:
        if commit:
            db_session.rollback()
        return False
    locked_user_file.status = UserFileStatus.COMPLETED
    locked_user_file.chunk_count = chunk_count
    locked_user_file.last_project_sync_at = now
    locked_user_file.secondary_reconcile_pending = True
    if commit:
        db_session.commit()
    else:
        db_session.flush()
    return True


def _schedule_provider_cleanup(
    job: RegulatoryIndexingJob,
    *,
    now: datetime.datetime,
    cancel_first: bool,
) -> None:
    if cancel_first and (job.remote_vertex_job_name or job.remote_openrouter_batch_id):
        phase = RegulatoryIndexingProviderCleanupPhase.VERTEX_CANCEL
    elif job.remote_vertex_job_name:
        phase = RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE
    elif job.vertex_submission_key:
        phase = RegulatoryIndexingProviderCleanupPhase.VERTEX_RECONCILE
    else:
        phase = RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP
    job.provider_cleanup_state = RegulatoryIndexingProviderCleanupState.PENDING.value
    job.provider_cleanup_phase = phase.value
    job.provider_cleanup_attempt_count = 0
    job.provider_cleanup_token = None
    job.provider_cleanup_next_retry_at = None
    job.provider_cleanup_heartbeat_at = now
    job.provider_cleanup_error_code = None
    job.provider_cleanup_error_message = None
    job.provider_cleanup_completed_at = None


def complete_regulatory_indexing_publication(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    chunk_count: int,
    now: datetime.datetime,
    commit: bool = True,
) -> bool:
    """Atomically complete the published file and its fenced durable job."""

    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None or locked_user_file.status not in {
        UserFileStatus.INDEXING,
        UserFileStatus.COMPLETED,
    }:
        if commit:
            db_session.rollback()
        return False
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
        if commit:
            db_session.rollback()
        return False
    if locked_job.user_file_id != locked_user_file.id:
        if commit:
            db_session.rollback()
        return False
    locked_user_file.status = UserFileStatus.COMPLETED
    locked_user_file.chunk_count = chunk_count
    locked_user_file.last_project_sync_at = now
    locked_user_file.secondary_reconcile_pending = True
    locked_job.status = RegulatoryIndexingJobStatus.SUCCEEDED.value
    locked_job.attempt_count = 0
    locked_job.next_retry_at = None
    locked_job.error_code = None
    locked_job.error_message = None
    locked_job.heartbeat_at = now
    locked_job.updated_at = now
    locked_job.completed_at = now
    _schedule_provider_cleanup(locked_job, now=now, cancel_first=False)
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
            "error_code": error_code[:_MAX_ERROR_CODE_LENGTH],
            "error_message": error_message[:_MAX_ERROR_MESSAGE_LENGTH],
            "updated_at": func.now(),
        },
    )


def fail_regulatory_indexing_job(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    error_code: str,
    error_message: str,
    now: datetime.datetime,
) -> bool:
    """Atomically terminally fail the fenced job and its live user file."""

    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None:
        db_session.rollback()
        return False
    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if job is None:
        db_session.rollback()
        return False
    if job.user_file_id != locked_user_file.id:
        db_session.rollback()
        return False
    job.status = RegulatoryIndexingJobStatus.FAILED.value
    job.error_code = error_code[:_MAX_ERROR_CODE_LENGTH]
    job.error_message = error_message[:_MAX_ERROR_MESSAGE_LENGTH]
    job.completed_at = now
    job.updated_at = now
    _schedule_provider_cleanup(job, now=now, cancel_first=True)
    if locked_user_file.status in {
        UserFileStatus.PROCESSING,
        UserFileStatus.INDEXING,
    }:
        locked_user_file.status = UserFileStatus.FAILED
    db_session.commit()
    return True


def cancel_regulatory_indexing_job(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    now: datetime.datetime,
) -> bool:
    """Clear sensitive derived payloads and finish cancellation under the fence."""

    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None:
        db_session.rollback()
        return False
    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if job is None:
        db_session.rollback()
        return False
    if job.user_file_id != locked_user_file.id:
        db_session.rollback()
        return False
    db_session.execute(
        update(RegulatoryIndexingItem)
        .where(RegulatoryIndexingItem.job_id == job_id)
        .values(vector=None, updated_at=now)
    )
    cancellation_intent = RegulatoryIndexingCancellationIntent(job.cancellation_intent)
    terminal_failure = (
        cancellation_intent is RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE
    )
    job.status = (
        RegulatoryIndexingJobStatus.FAILED.value
        if terminal_failure
        else RegulatoryIndexingJobStatus.CANCELLED.value
    )
    job.completed_at = now
    job.next_retry_at = None
    if not terminal_failure:
        job.error_code = None
        job.error_message = None
    job.updated_at = now
    _schedule_provider_cleanup(job, now=now, cancel_first=False)
    if terminal_failure:
        if locked_user_file.status in {
            UserFileStatus.PROCESSING,
            UserFileStatus.INDEXING,
        }:
            locked_user_file.status = UserFileStatus.FAILED
    elif cancellation_intent is RegulatoryIndexingCancellationIntent.SUPERSEDE:
        if locked_user_file.status not in {
            UserFileStatus.CANCELED,
            UserFileStatus.DELETING,
        }:
            locked_user_file.status = UserFileStatus.PROCESSING
    elif cancellation_intent is RegulatoryIndexingCancellationIntent.USER_DELETE:
        locked_user_file.status = UserFileStatus.DELETING
    elif locked_user_file.status is not UserFileStatus.DELETING:
        locked_user_file.status = UserFileStatus.CANCELED
    db_session.commit()
    return True


def request_regulatory_indexing_cancellation(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    cancellation_intent: RegulatoryIndexingCancellationIntent,
    now: datetime.datetime,
) -> bool:
    """Enter the resumable cancellation state under the active stage fence."""

    if cancellation_intent not in {
        RegulatoryIndexingCancellationIntent.USER_CANCEL,
        RegulatoryIndexingCancellationIntent.USER_DELETE,
    }:
        raise ValueError("explicit cancellation intent is unsupported")

    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if job is None:
        db_session.rollback()
        return False
    job.status = RegulatoryIndexingJobStatus.CANCELLING.value
    job.cancellation_intent = cancellation_intent.value
    job.cancellation_phase = (
        RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
        if job.remote_vertex_job_name or job.remote_openrouter_batch_id
        else RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
    )
    job.attempt_count = 0
    job.next_retry_at = None
    job.error_code = None
    job.error_message = None
    job.heartbeat_at = now
    job.updated_at = now
    db_session.commit()
    return True


def request_regulatory_indexing_terminal_failure_cleanup(
    db_session: Session,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
    error_code: str,
    error_message: str,
    now: datetime.datetime,
) -> bool:
    """Fail closed through the durable external and index cleanup phases."""

    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status == RegulatoryIndexingJobStatus.RUNNING.value,
            RegulatoryIndexingJob.stage == expected_stage.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
        )
        .with_for_update()
    )
    if job is None:
        db_session.rollback()
        return False
    job.status = RegulatoryIndexingJobStatus.CANCELLING.value
    job.cancellation_intent = (
        RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE.value
    )
    job.cancellation_phase = (
        RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
        if job.remote_vertex_job_name or job.remote_openrouter_batch_id
        else RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
    )
    job.attempt_count = 0
    job.next_retry_at = None
    job.error_code = error_code[:_MAX_ERROR_CODE_LENGTH]
    job.error_message = error_message[:_MAX_ERROR_MESSAGE_LENGTH]
    job.heartbeat_at = now
    job.updated_at = now
    db_session.commit()
    return True


def advance_regulatory_indexing_cancellation(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    expected_phase: RegulatoryIndexingCancellationPhase,
    next_phase: RegulatoryIndexingCancellationPhase,
    now: datetime.datetime,
) -> bool:
    advanced_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status
            == RegulatoryIndexingJobStatus.CANCELLING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.cancellation_phase == expected_phase.value,
        )
        .values(
            cancellation_phase=next_phase.value,
            attempt_count=0,
            next_retry_at=None,
            error_code=case(
                (
                    RegulatoryIndexingJob.cancellation_intent
                    == RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE.value,
                    RegulatoryIndexingJob.error_code,
                ),
                else_=None,
            ),
            error_message=case(
                (
                    RegulatoryIndexingJob.cancellation_intent
                    == RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE.value,
                    RegulatoryIndexingJob.error_message,
                ),
                else_=None,
            ),
            heartbeat_at=now,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return advanced_id is not None


def schedule_regulatory_indexing_cancellation_retry(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    expected_phase: RegulatoryIndexingCancellationPhase,
    next_retry_at: datetime.datetime,
    error_code: str,
    error_message: str,
) -> bool:
    scheduled_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status
            == RegulatoryIndexingJobStatus.CANCELLING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.cancellation_phase == expected_phase.value,
        )
        .values(
            attempt_count=RegulatoryIndexingJob.attempt_count + 1,
            next_retry_at=next_retry_at,
            error_code=case(
                (
                    RegulatoryIndexingJob.cancellation_intent
                    == RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE.value,
                    RegulatoryIndexingJob.error_code,
                ),
                else_=error_code[:_MAX_ERROR_CODE_LENGTH],
            ),
            error_message=case(
                (
                    RegulatoryIndexingJob.cancellation_intent
                    == RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE.value,
                    RegulatoryIndexingJob.error_message,
                ),
                else_=error_message[:_MAX_ERROR_MESSAGE_LENGTH],
            ),
            updated_at=func.now(),
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return scheduled_id is not None


def finalize_regulatory_indexing_cancellation(
    db_session: Session,
    *,
    job_id: UUID,
    expected_generation: int,
    now: datetime.datetime,
) -> bool:
    """Clear derived payloads and terminalize the final cancellation phase."""

    locked_user_file = _lock_user_file_for_regulatory_job(db_session, job_id)
    if locked_user_file is None:
        db_session.rollback()
        return False
    job = db_session.scalar(
        select(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.status
            == RegulatoryIndexingJobStatus.CANCELLING.value,
            RegulatoryIndexingJob.lease_generation == expected_generation,
            RegulatoryIndexingJob.cancellation_phase
            == RegulatoryIndexingCancellationPhase.FINALIZE.value,
        )
        .with_for_update()
    )
    if job is None:
        db_session.rollback()
        return False
    if job.user_file_id != locked_user_file.id:
        db_session.rollback()
        return False
    db_session.execute(
        update(RegulatoryIndexingItem)
        .where(RegulatoryIndexingItem.job_id == job_id)
        .values(vector=None, updated_at=now)
    )
    cancellation_intent = RegulatoryIndexingCancellationIntent(job.cancellation_intent)
    terminal_failure = (
        cancellation_intent is RegulatoryIndexingCancellationIntent.TERMINAL_FAILURE
    )
    job.status = (
        RegulatoryIndexingJobStatus.FAILED.value
        if terminal_failure
        else RegulatoryIndexingJobStatus.CANCELLED.value
    )
    job.completed_at = now
    job.next_retry_at = None
    if not terminal_failure:
        job.error_code = None
        job.error_message = None
    job.heartbeat_at = now
    job.updated_at = now
    _schedule_provider_cleanup(job, now=now, cancel_first=False)
    if terminal_failure:
        if locked_user_file.status in {
            UserFileStatus.PROCESSING,
            UserFileStatus.INDEXING,
        }:
            locked_user_file.status = UserFileStatus.FAILED
    elif cancellation_intent is RegulatoryIndexingCancellationIntent.SUPERSEDE:
        if locked_user_file.status not in {
            UserFileStatus.CANCELED,
            UserFileStatus.DELETING,
        }:
            locked_user_file.status = UserFileStatus.PROCESSING
    elif cancellation_intent is RegulatoryIndexingCancellationIntent.USER_DELETE:
        locked_user_file.status = UserFileStatus.DELETING
    elif locked_user_file.status is not UserFileStatus.DELETING:
        locked_user_file.status = UserFileStatus.CANCELED
    db_session.commit()
    return True


def claim_due_regulatory_provider_cleanups(
    db_session: Session,
    *,
    stale_before: datetime.datetime,
    claimed_at: datetime.datetime,
    limit: int = 20,
) -> list[RegulatoryIndexingProviderCleanupClaim]:
    """Claim terminal provider cleanup with a one-use delivery token."""

    due = or_(
        RegulatoryIndexingJob.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.PENDING.value,
        and_(
            RegulatoryIndexingJob.provider_cleanup_state.in_(
                (
                    RegulatoryIndexingProviderCleanupState.RETRY_WAIT.value,
                    RegulatoryIndexingProviderCleanupState.EXHAUSTED.value,
                )
            ),
            RegulatoryIndexingJob.provider_cleanup_next_retry_at.is_not(None),
            RegulatoryIndexingJob.provider_cleanup_next_retry_at <= claimed_at,
        ),
        and_(
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_heartbeat_at.is_not(None),
            RegulatoryIndexingJob.provider_cleanup_heartbeat_at <= stale_before,
        ),
    )
    jobs = list(
        db_session.scalars(
            select(RegulatoryIndexingJob)
            .where(
                RegulatoryIndexingJob.status.in_(
                    (
                        RegulatoryIndexingJobStatus.SUCCEEDED.value,
                        RegulatoryIndexingJobStatus.FAILED.value,
                        RegulatoryIndexingJobStatus.CANCELLED.value,
                    )
                ),
                due,
            )
            .order_by(
                func.coalesce(
                    RegulatoryIndexingJob.provider_cleanup_next_retry_at,
                    RegulatoryIndexingJob.provider_cleanup_heartbeat_at,
                    RegulatoryIndexingJob.completed_at,
                ),
                RegulatoryIndexingJob.id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
    )
    claims: list[RegulatoryIndexingProviderCleanupClaim] = []
    for job in jobs:
        token = uuid4()
        if (
            job.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.EXHAUSTED.value
        ):
            job.provider_cleanup_attempt_count = 0
        job.provider_cleanup_state = (
            RegulatoryIndexingProviderCleanupState.RUNNING.value
        )
        job.provider_cleanup_generation += 1
        job.provider_cleanup_token = token
        job.provider_cleanup_next_retry_at = None
        job.provider_cleanup_heartbeat_at = claimed_at
        claims.append(
            RegulatoryIndexingProviderCleanupClaim(
                job_id=job.id,
                cleanup_generation=job.provider_cleanup_generation,
                cleanup_token=token,
            )
        )
    db_session.commit()
    return claims


def consume_regulatory_provider_cleanup_delivery(
    db_session: Session,
    *,
    job_id: UUID,
    cleanup_generation: int,
    cleanup_token: UUID,
    consumed_at: datetime.datetime,
) -> bool:
    consumed_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_generation == cleanup_generation,
            RegulatoryIndexingJob.provider_cleanup_token == cleanup_token,
        )
        .values(
            provider_cleanup_token=None,
            provider_cleanup_heartbeat_at=consumed_at,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return consumed_id is not None


def advance_regulatory_provider_cleanup(
    db_session: Session,
    *,
    job_id: UUID,
    cleanup_generation: int,
    expected_phase: RegulatoryIndexingProviderCleanupPhase,
    next_phase: RegulatoryIndexingProviderCleanupPhase,
    now: datetime.datetime,
) -> bool:
    advanced_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_generation == cleanup_generation,
            RegulatoryIndexingJob.provider_cleanup_token.is_(None),
            RegulatoryIndexingJob.provider_cleanup_phase == expected_phase.value,
        )
        .values(
            provider_cleanup_state=RegulatoryIndexingProviderCleanupState.PENDING.value,
            provider_cleanup_phase=next_phase.value,
            provider_cleanup_attempt_count=0,
            provider_cleanup_next_retry_at=None,
            provider_cleanup_heartbeat_at=now,
            provider_cleanup_error_code=None,
            provider_cleanup_error_message=None,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return advanced_id is not None


def record_reconciled_provider_cleanup_vertex_job(
    db_session: Session,
    *,
    job_id: UUID,
    cleanup_generation: int,
    submission_key: str,
    remote_job_name: str,
    input_uri: str | None,
    output_uri: str | None,
    now: datetime.datetime,
) -> bool:
    """Persist a late-visible indeterminate job before deleting it."""

    recorded_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_generation == cleanup_generation,
            RegulatoryIndexingJob.provider_cleanup_token.is_(None),
            RegulatoryIndexingJob.provider_cleanup_phase
            == RegulatoryIndexingProviderCleanupPhase.VERTEX_RECONCILE.value,
            RegulatoryIndexingJob.vertex_submission_key == submission_key,
        )
        .values(
            remote_vertex_job_name=remote_job_name,
            vertex_input_uri=input_uri,
            vertex_output_uri=output_uri,
            provider_cleanup_state=RegulatoryIndexingProviderCleanupState.PENDING.value,
            provider_cleanup_phase=(
                RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE.value
            ),
            provider_cleanup_attempt_count=0,
            provider_cleanup_next_retry_at=None,
            provider_cleanup_heartbeat_at=now,
            provider_cleanup_error_code=None,
            provider_cleanup_error_message=None,
            updated_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return recorded_id is not None


def schedule_regulatory_provider_cleanup_retry(
    db_session: Session,
    *,
    job_id: UUID,
    cleanup_generation: int,
    next_retry_at: datetime.datetime,
    error_code: str,
    error_message: str,
    exhausted: bool,
) -> bool:
    scheduled_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_generation == cleanup_generation,
            RegulatoryIndexingJob.provider_cleanup_token.is_(None),
        )
        .values(
            provider_cleanup_state=(
                RegulatoryIndexingProviderCleanupState.EXHAUSTED.value
                if exhausted
                else RegulatoryIndexingProviderCleanupState.RETRY_WAIT.value
            ),
            provider_cleanup_attempt_count=(
                RegulatoryIndexingJob.provider_cleanup_attempt_count + 1
            ),
            provider_cleanup_next_retry_at=next_retry_at,
            provider_cleanup_had_failure=True,
            provider_cleanup_error_code=error_code[:_MAX_ERROR_CODE_LENGTH],
            provider_cleanup_error_message=error_message[:_MAX_ERROR_MESSAGE_LENGTH],
            provider_cleanup_heartbeat_at=func.now(),
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return scheduled_id is not None


def complete_regulatory_provider_cleanup(
    db_session: Session,
    *,
    job_id: UUID,
    cleanup_generation: int,
    now: datetime.datetime,
) -> bool:
    completed_id = db_session.scalar(
        update(RegulatoryIndexingJob)
        .where(
            RegulatoryIndexingJob.id == job_id,
            RegulatoryIndexingJob.provider_cleanup_state
            == RegulatoryIndexingProviderCleanupState.RUNNING.value,
            RegulatoryIndexingJob.provider_cleanup_generation == cleanup_generation,
            RegulatoryIndexingJob.provider_cleanup_token.is_(None),
            RegulatoryIndexingJob.provider_cleanup_phase
            == RegulatoryIndexingProviderCleanupPhase.COMPLETE.value,
        )
        .values(
            provider_cleanup_state=RegulatoryIndexingProviderCleanupState.SUCCEEDED.value,
            provider_cleanup_next_retry_at=None,
            provider_cleanup_heartbeat_at=now,
            provider_cleanup_error_code=None,
            provider_cleanup_error_message=None,
            provider_cleanup_completed_at=now,
        )
        .returning(RegulatoryIndexingJob.id)
    )
    db_session.commit()
    return completed_id is not None
