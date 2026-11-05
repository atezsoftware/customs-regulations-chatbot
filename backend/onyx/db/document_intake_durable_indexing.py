"""Bind durable jobs and explicit successors to one approved intake intent."""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_indexing_retry import (
    _delivery_scope,
    intake_index_attempt_chain,
    intake_index_attempt_is_authorized,
)
from onyx.db.enums import (
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingProviderCleanupState,
    UserFileProjectionRepairStatus,
    UserFileStatus,
)
from onyx.db.models import (
    KVStore,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryIndexingJob,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.document_index.publication_models import publication_digest

_JOB_PREFIX = "intake_durable_index_job:"
_RETRY_PREFIX = "intake_durable_index_retry:"


def _owners(
    session: Session, file_id: UUID
) -> tuple[RegulatoryDocumentIntake, RegulatoryDocumentIntakeFile, UserFile] | None:
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.user_file_id == file_id
        )
    )
    if member is None:
        return None
    batch = session.get_one(RegulatoryDocumentIntake, member.intake_id)
    file = session.get_one(UserFile, file_id)
    if not batch.review_sha256 or file.user_id != batch.created_by:
        raise intake.IntakeConflictError("The approved durable intake owner changed")
    return batch, member, file


def _job_identity(job: RegulatoryIndexingJob) -> dict[str, str | int]:
    return {
        "job_id": str(job.id),
        "content_hash": job.content_hash,
        "chunk_generation_hash": job.chunk_generation_hash,
        "search_settings_id": job.search_settings_id,
        "prompt_hash": job.prompt_hash,
        "config_sha256": publication_digest(job.config_snapshot),
    }


def _put(session: Session, key: str, value: dict[str, str | int]) -> None:
    previous = session.get(KVStore, key)
    if previous is not None:
        if previous.value != value:
            raise intake.IntakeConflictError(
                "The retained durable intent binding changed"
            )
        return
    session.add(KVStore(key=key, value=value))


def bound_intake_durable_job(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    attempt_id: UUID,
    *,
    strict: bool = False,
) -> RegulatoryIndexingJob | None:
    chain = intake_index_attempt_chain(session, batch, member, attempt_id)
    if chain is None:
        return None
    receipt = None
    token = attempt_id
    for token in chain:
        receipt = session.get(KVStore, _JOB_PREFIX + str(token))
        if receipt is not None:
            break
    if receipt is None:
        return None

    def invalid() -> None:
        if strict:
            raise intake.IntakeConflictError(
                "The retained durable intent binding changed"
            )

    if not isinstance(receipt.value, Mapping):
        invalid()
        return None
    value = cast(Mapping[str, object], receipt.value)
    try:
        identifier = UUID(str(value.get("job_id")))
    except ValueError:
        invalid()
        return None
    job = session.get(RegulatoryIndexingJob, identifier, populate_existing=True)
    if (
        job is None
        or job.user_file_id != member.user_file_id
        or value
        != {
            **_delivery_scope(batch, member),
            "attempt_id": str(token),
            **_job_identity(job),
        }
    ):
        invalid()
        return None
    return job


def bind_intake_durable_job(
    session: Session, file_id: UUID, attempt_id: UUID, job: RegulatoryIndexingJob
) -> None:
    owners = _owners(session, file_id)
    if owners is None:
        return
    batch, member, _file = owners
    job = session.get_one(RegulatoryIndexingJob, job.id, populate_existing=True)
    repair = session.get(UserFileProjectionRepair, file_id, with_for_update=True)
    if (
        job.user_file_id != file_id
        or repair is None
        or not repair.initial_index
        or repair.attempt_id != attempt_id
        or repair.status != UserFileProjectionRepairStatus.RUNNING
        or not intake_index_attempt_is_authorized(session, batch, member, attempt_id)
    ):
        raise intake.IntakeConflictError(
            "The durable job escaped its approved index intent"
        )
    _put(
        session,
        _JOB_PREFIX + str(attempt_id),
        {
            **_delivery_scope(batch, member),
            "attempt_id": str(attempt_id),
            **_job_identity(job),
        },
    )
    session.flush()
    if job.status == RegulatoryIndexingJobStatus.FAILED.value:
        fail_bound_intake_index_request(session, job)


def fail_bound_intake_index_request(
    session: Session, job: RegulatoryIndexingJob
) -> None:
    """The terminal job caller holds its file lock before the intent is locked."""
    if job.status != RegulatoryIndexingJobStatus.FAILED.value:
        return
    owners = _owners(session, job.user_file_id)
    if owners is None:
        return
    batch, member, file = owners
    repair = session.get(UserFileProjectionRepair, file.id, with_for_update=True)
    if (
        repair is None
        or not repair.initial_index
        or repair.status
        not in {
            UserFileProjectionRepairStatus.PENDING,
            UserFileProjectionRepairStatus.RUNNING,
            UserFileProjectionRepairStatus.SUCCEEDED,
        }
    ):
        return
    bound = bound_intake_durable_job(session, batch, member, repair.attempt_id)
    if bound is None or bound.id != job.id:
        return
    repair.status = UserFileProjectionRepairStatus.FAILED
    repair.failure_code = ("durable:" + (job.error_code or "FAILED"))[:128]
    repair.updated_at = datetime.now(timezone.utc)
    if file.status in {UserFileStatus.CHUNKED, UserFileStatus.INDEXING}:
        file.status = UserFileStatus.FAILED


def require_failed_durable_cleanup(job: RegulatoryIndexingJob) -> None:
    if (
        job.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.SUCCEEDED.value
    ):
        return
    if (
        job.provider_cleanup_state == RegulatoryIndexingProviderCleanupState.NONE.value
        and not any(
            (
                job.remote_vertex_job_name,
                job.vertex_submission_key,
                job.vertex_input_uri,
                job.vertex_output_uri,
                job.remote_openrouter_batch_id,
                job.openrouter_submission_key,
            )
        )
    ):
        return
    raise intake.IntakeConflictError(
        "The failed durable job must finish provider cleanup before Retry"
    )


def record_intake_durable_retry(
    session: Session,
    file_id: UUID,
    previous_attempt_id: UUID,
    replacement_attempt_id: UUID,
) -> None:
    owners = _owners(session, file_id)
    if owners is None:
        return
    batch, member, _file = owners
    job = bound_intake_durable_job(
        session, batch, member, previous_attempt_id, strict=True
    )
    if job is None:
        pending_restart = intake_durable_retry_job(
            session, file_id, previous_attempt_id
        )
        if pending_restart is None:
            return
        job, _anchor = pending_restart
    if job.status != RegulatoryIndexingJobStatus.FAILED.value:
        raise intake.IntakeConflictError(
            "Only the bound failed durable job can be restarted"
        )
    require_failed_durable_cleanup(job)
    scope = _delivery_scope(batch, member)
    scope["approval_attempt_id"] = str(replacement_attempt_id)
    _put(
        session,
        _RETRY_PREFIX + str(replacement_attempt_id),
        {
            **scope,
            "attempt_id": str(replacement_attempt_id),
            **_job_identity(job),
        },
    )


def intake_durable_retry_job(
    session: Session, file_id: UUID, attempt_id: UUID
) -> tuple[RegulatoryIndexingJob, UUID] | None:
    owners = _owners(session, file_id)
    if owners is None:
        return None
    batch, member, file = owners
    if member.approval_attempt_id is None or not intake_index_attempt_is_authorized(
        session, batch, member, attempt_id
    ):
        raise intake.IntakeConflictError("The durable retry intent is not approved")
    anchor = member.approval_attempt_id
    receipt = session.get(KVStore, _RETRY_PREFIX + str(anchor))
    if receipt is None:
        return None
    if not isinstance(receipt.value, Mapping):
        raise intake.IntakeConflictError("The durable retry receipt is invalid")
    value = cast(Mapping[str, object], receipt.value)
    try:
        identifier = UUID(str(value.get("job_id")))
    except ValueError as error:
        raise intake.IntakeConflictError(
            "The durable retry job identity changed"
        ) from error
    job = session.get_one(RegulatoryIndexingJob, identifier, populate_existing=True)
    if (
        job.user_file_id != file.id
        or value
        != {
            **_delivery_scope(batch, member),
            "attempt_id": str(anchor),
            **_job_identity(job),
        }
        or job.status != RegulatoryIndexingJobStatus.FAILED.value
    ):
        raise intake.IntakeConflictError(
            "The failed durable job or approved retry changed"
        )
    from onyx.regulatory.indexing_jobs.preparation import regulatory_chunks_content_hash

    rows = list(
        session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
    )
    if (
        file.regulatory_chunk_generation_hash != job.chunk_generation_hash
        or regulatory_chunks_content_hash(rows) != job.content_hash
        or intake.review_digest(session, batch) != batch.review_sha256
    ):
        raise intake.IntakeConflictError("The approved durable retry source changed")
    require_failed_durable_cleanup(job)
    return job, anchor
