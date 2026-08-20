import datetime
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.db import regulatory_indexing_jobs as regulatory_indexing_job_repository
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
    User,
    UserFile,
)
from onyx.db.regulatory_chunks import replace_indexed_chunks_for_file
from onyx.db.regulatory_indexing_jobs import (
    RegulatoryIndexingConfigSnapshot,
    advance_regulatory_indexing_cancellation,
    advance_regulatory_indexing_job,
    advance_regulatory_provider_cleanup,
    claim_due_regulatory_provider_cleanups,
    claim_regulatory_indexing_job,
    claim_stale_regulatory_indexing_jobs,
    complete_regulatory_indexing_publication,
    complete_regulatory_indexing_user_file,
    complete_regulatory_provider_cleanup,
    complete_vertex_partial_retry_cleanup,
    consume_preclaimed_regulatory_indexing_delivery,
    consume_regulatory_provider_cleanup_delivery,
    create_or_get_regulatory_indexing_item,
    create_or_get_regulatory_indexing_job,
    fail_regulatory_indexing_job,
    finalize_regulatory_indexing_cancellation,
    mark_vertex_partial_retry_cleanup_required,
    persist_regulatory_indexing_item_context,
    persist_regulatory_indexing_item_failure,
    persist_regulatory_indexing_item_skipped,
    persist_regulatory_indexing_item_vector,
    persist_regulatory_indexing_item_vectors,
    record_openrouter_submission_ambiguous,
    record_openrouter_submission_intent,
    record_reconciled_provider_cleanup_vertex_job,
    record_vertex_reconciliation_miss,
    record_vertex_submission,
    record_vertex_submission_intent,
    regulatory_indexing_external_mutation_lease,
    request_regulatory_indexing_cancellation,
    request_user_file_deletion_cleanup,
    require_vertex_submission_reconciliation,
    schedule_regulatory_indexing_cancellation_retry,
    schedule_regulatory_indexing_retry,
    schedule_regulatory_provider_cleanup_retry,
)
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.indexing_jobs.publisher import stage_regulatory_job_in_index
from tests.external_dependency_unit.conftest import create_test_user

_NOW = datetime.datetime(2026, 8, 19, 10, 0, tzinfo=datetime.timezone.utc)
_SNAPSHOT: RegulatoryIndexingConfigSnapshot = {
    "input_content_hash": "a" * 64,
    "input_hash_version": "canonical-v2",
    "chunk_generation_hash": "b" * 64,
    "embedding_provider": "openrouter",
    "embedding_model": "openai/text-embedding-3-large",
    "effective_dimension": 1536,
    "vertex_model": "gemini-3.6-flash",
}


@pytest.fixture
def regulatory_user_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> Generator[UserFile, None, None]:
    user = create_test_user(db_session, "regulatory-indexing-job")
    user_file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id=f"regulatory-indexing-{uuid4().hex}",
        name="mevzuat.md",
        file_type="text/markdown",
    )
    db_session.add(user_file)
    db_session.commit()
    try:
        yield user_file
    finally:
        db_session.rollback()
        persisted_files = db_session.scalars(
            select(UserFile).where(UserFile.user_id == user.id)
        ).all()
        for persisted_file in persisted_files:
            db_session.delete(persisted_file)
        db_session.commit()
        persisted_user = db_session.get(User, user.id)
        if persisted_user is not None:
            db_session.delete(persisted_user)
            db_session.commit()


def _create_job(
    db_session: Session,
    user_file_id: UUID,
    *,
    content_hash: str | None = None,
    chunk_generation_hash: str = "b" * 64,
) -> RegulatoryIndexingJob:
    resolved_content_hash = content_hash or uuid4().hex
    snapshot = {
        **_SNAPSHOT,
        "input_content_hash": resolved_content_hash,
        "chunk_generation_hash": chunk_generation_hash,
    }
    return create_or_get_regulatory_indexing_job(
        db_session,
        user_file_id=user_file_id,
        content_hash=resolved_content_hash,
        search_settings_id=17,
        prompt_hash="prompt-v1",
        chunk_generation_hash=chunk_generation_hash,
        config_snapshot=snapshot,
        now=_NOW,
    )


def _create_sibling_user_file(
    db_session: Session,
    user_file: UserFile,
) -> UserFile:
    sibling = UserFile(
        id=uuid4(),
        user_id=user_file.user_id,
        file_id=f"regulatory-indexing-{uuid4().hex}",
        name="mevzuat-sibling.md",
        file_type="text/markdown",
    )
    db_session.add(sibling)
    db_session.commit()
    return sibling


def test_duplicate_job_creation_is_atomic_and_idempotent(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    engine = db_session.get_bind()
    content_hash = uuid4().hex

    def create_in_independent_session() -> UUID:
        with Session(engine) as independent_session:
            return _create_job(
                independent_session,
                regulatory_user_file.id,
                content_hash=content_hash,
            ).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        created_ids = list(
            executor.map(lambda _: create_in_independent_session(), range(2))
        )

    assert created_ids[0] == created_ids[1]
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id,
                RegulatoryIndexingJob.content_hash == content_hash,
                RegulatoryIndexingJob.search_settings_id == 17,
                RegulatoryIndexingJob.prompt_hash == "prompt-v1",
            )
        )
        == 1
    )


def test_job_creation_rejects_a_tombstoned_user_file(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    deletion = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )
    assert deletion.ready_to_delete is True

    with pytest.raises(ValueError, match="deleting user file"):
        _create_job(db_session, regulatory_user_file.id)

    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id
            )
        )
        == 0
    )


@pytest.mark.parametrize(
    "stage,status,has_remote_job",
    [
        (RegulatoryIndexingStage.PREPARING, RegulatoryIndexingJobStatus.QUEUED, False),
        (
            RegulatoryIndexingStage.CONTEXT_SUBMIT,
            RegulatoryIndexingJobStatus.RUNNING,
            False,
        ),
        (
            RegulatoryIndexingStage.CONTEXT_WAIT,
            RegulatoryIndexingJobStatus.RUNNING,
            True,
        ),
        (
            RegulatoryIndexingStage.CONTEXT_APPLY,
            RegulatoryIndexingJobStatus.RETRY_WAIT,
            True,
        ),
        (RegulatoryIndexingStage.EMBEDDING, RegulatoryIndexingJobStatus.RUNNING, True),
        (
            RegulatoryIndexingStage.INDEX_WRITE,
            RegulatoryIndexingJobStatus.RUNNING,
            True,
        ),
        (RegulatoryIndexingStage.VERIFY, RegulatoryIndexingJobStatus.RUNNING, True),
        (RegulatoryIndexingStage.PUBLISH, RegulatoryIndexingJobStatus.RUNNING, True),
    ],
)
def test_changed_chunk_generation_identity_supersedes_and_fences_active_job(
    db_session: Session,
    regulatory_user_file: UserFile,
    stage: RegulatoryIndexingStage,
    status: RegulatoryIndexingJobStatus,
    has_remote_job: bool,
) -> None:
    content_hash = uuid4().hex

    first = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="1" * 64,
    )
    first.stage = stage.value
    first.status = status.value
    first.remote_vertex_job_name = "remote-generation-1" if has_remote_job else None
    db_session.commit()
    second = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )

    assert second.id == first.id
    db_session.refresh(first)
    assert second.chunk_generation_hash == "1" * 64
    assert first.status == RegulatoryIndexingJobStatus.CANCELLING.value
    assert first.lease_generation == 1
    assert first.cancellation_intent == "SUPERSEDE"
    expected_phase = (
        RegulatoryIndexingCancellationPhase.VERTEX_CANCEL
        if has_remote_job
        else RegulatoryIndexingCancellationPhase.GCS_CLEANUP
    )
    assert first.cancellation_phase == expected_phase.value
    repeated = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )
    db_session.refresh(first)
    assert repeated.id == first.id
    assert first.lease_generation == 1
    assert first.cancellation_intent == "SUPERSEDE"
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id
            )
        )
        == 1
    )


def test_supersession_finalization_requeues_file_and_allows_one_successor(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    content_hash = uuid4().hex
    old_job = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="1" * 64,
    )
    reused = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )
    assert reused.id == old_job.id
    old_job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    assert finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=old_job.id,
        expected_generation=1,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(old_job)
    db_session.refresh(regulatory_user_file)
    assert old_job.status == RegulatoryIndexingJobStatus.CANCELLED.value
    assert old_job.cancellation_intent == "SUPERSEDE"
    assert regulatory_user_file.status is UserFileStatus.PROCESSING
    assert not finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=old_job.id,
        expected_generation=1,
        now=_NOW + datetime.timedelta(minutes=2),
    )

    successor = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )
    assert successor.id != old_job.id
    assert successor.status == RegulatoryIndexingJobStatus.QUEUED.value
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id,
                RegulatoryIndexingJob.status.in_(
                    [
                        RegulatoryIndexingJobStatus.QUEUED.value,
                        RegulatoryIndexingJobStatus.RUNNING.value,
                        RegulatoryIndexingJobStatus.RETRY_WAIT.value,
                        RegulatoryIndexingJobStatus.CANCELLING.value,
                    ]
                ),
            )
        )
        == 1
    )


def test_claimed_generation_drift_is_atomically_superseded_and_fenced(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    delivery = regulatory_indexing_job_repository.supersede_regulatory_indexing_job_for_generation_drift(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        current_chunk_generation_hash="2" * 64,
        now=_NOW + datetime.timedelta(minutes=1),
    )

    assert delivery is not None
    assert delivery.job_id == job.id
    assert delivery.expected_generation == 2
    db_session.refresh(job)
    assert job.status == RegulatoryIndexingJobStatus.CANCELLING.value
    assert job.cancellation_intent == "SUPERSEDE"
    assert (
        job.cancellation_phase == RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
    )
    assert job.lease_generation == 2
    assert (
        regulatory_indexing_job_repository.supersede_regulatory_indexing_job_for_generation_drift(
            db_session,
            job_id=job.id,
            expected_stage=RegulatoryIndexingStage.PREPARING,
            expected_generation=1,
            current_chunk_generation_hash="2" * 64,
            now=_NOW + datetime.timedelta(minutes=2),
        )
        is None
    )


def test_deletion_monotonically_overrides_in_progress_supersession(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    old_job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    db_session.refresh(old_job)
    assert old_job.cancellation_intent == "SUPERSEDE"
    assert old_job.lease_generation == 1

    first = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(old_job)
    db_session.refresh(regulatory_user_file)
    assert old_job.cancellation_intent == "USER_DELETE"
    assert old_job.lease_generation == 2
    assert [(row.job_id, row.expected_generation) for row in first.deliveries] == [
        (old_job.id, 2)
    ]
    assert regulatory_user_file.status is UserFileStatus.DELETING

    repeated = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW + datetime.timedelta(minutes=2),
    )
    db_session.refresh(old_job)
    assert old_job.lease_generation == 2
    assert repeated.deliveries == first.deliveries

    old_job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    db_session.commit()
    assert finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=old_job.id,
        expected_generation=2,
        now=_NOW + datetime.timedelta(minutes=3),
    )
    db_session.refresh(regulatory_user_file)
    assert regulatory_user_file.status is UserFileStatus.DELETING


def test_deletion_cleans_multiple_generations_sequentially(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    first = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    first.status = RegulatoryIndexingJobStatus.SUCCEEDED.value
    first.completed_at = _NOW
    db_session.commit()
    second = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    second.status = RegulatoryIndexingJobStatus.FAILED.value
    second.completed_at = _NOW + datetime.timedelta(seconds=1)
    db_session.commit()
    active = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="3" * 64,
    )
    active.status = RegulatoryIndexingJobStatus.CANCELLING.value
    active.cancellation_intent = RegulatoryIndexingCancellationIntent.SUPERSEDE.value
    active.cancellation_phase = RegulatoryIndexingCancellationPhase.INDEX_DELETE.value
    active.lease_generation = 4
    db_session.commit()

    remaining_ids = {first.id, second.id, active.id}
    expected_first_id = active.id
    generation = 4
    for cleanup_number in range(3):
        plan = request_user_file_deletion_cleanup(
            db_session,
            user_file_id=regulatory_user_file.id,
            now=_NOW + datetime.timedelta(minutes=cleanup_number + 1),
        )
        assert plan.ready_to_delete is False
        assert len(plan.deliveries) == 1
        delivery = plan.deliveries[0]
        if cleanup_number == 0:
            assert delivery.job_id == expected_first_id
            assert delivery.expected_generation == generation + 1
        activated = db_session.get(RegulatoryIndexingJob, delivery.job_id)
        assert activated is not None
        assert activated.status == RegulatoryIndexingJobStatus.CANCELLING.value
        assert (
            activated.cancellation_intent
            == RegulatoryIndexingCancellationIntent.USER_DELETE.value
        )
        assert (
            db_session.scalar(
                select(func.count(RegulatoryIndexingJob.id)).where(
                    RegulatoryIndexingJob.user_file_id == regulatory_user_file.id,
                    RegulatoryIndexingJob.status.in_(
                        [
                            RegulatoryIndexingJobStatus.QUEUED.value,
                            RegulatoryIndexingJobStatus.RUNNING.value,
                            RegulatoryIndexingJobStatus.RETRY_WAIT.value,
                            RegulatoryIndexingJobStatus.CANCELLING.value,
                        ]
                    ),
                )
            )
            == 1
        )
        assert {
            row.id
            for row in db_session.scalars(
                select(RegulatoryIndexingJob).where(
                    RegulatoryIndexingJob.user_file_id == regulatory_user_file.id,
                    RegulatoryIndexingJob.status
                    == RegulatoryIndexingJobStatus.CANCELLED.value,
                )
            )
        } == ({first.id, second.id, active.id} - remaining_ids)

        repeated = request_user_file_deletion_cleanup(
            db_session,
            user_file_id=regulatory_user_file.id,
            now=_NOW + datetime.timedelta(minutes=cleanup_number + 1),
        )
        assert repeated.deliveries == plan.deliveries

        activated.cancellation_phase = (
            RegulatoryIndexingCancellationPhase.FINALIZE.value
        )
        db_session.commit()
        assert finalize_regulatory_indexing_cancellation(
            db_session,
            job_id=activated.id,
            expected_generation=delivery.expected_generation,
            now=_NOW + datetime.timedelta(minutes=cleanup_number + 2),
        )
        remaining_ids.remove(activated.id)

    blocked = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW + datetime.timedelta(minutes=5),
    )
    assert blocked.ready_to_delete is False
    assert blocked.deliveries == ()

    cleanup_claims = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW,
        claimed_at=_NOW + datetime.timedelta(minutes=5),
        limit=10,
    )
    assert {claim.job_id for claim in cleanup_claims} == {
        first.id,
        second.id,
        active.id,
    }
    for claim in cleanup_claims:
        assert consume_regulatory_provider_cleanup_delivery(
            db_session,
            job_id=claim.job_id,
            cleanup_generation=claim.cleanup_generation,
            cleanup_token=claim.cleanup_token,
            consumed_at=_NOW + datetime.timedelta(minutes=5),
        )
        assert advance_regulatory_provider_cleanup(
            db_session,
            job_id=claim.job_id,
            cleanup_generation=claim.cleanup_generation,
            expected_phase=RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP,
            next_phase=RegulatoryIndexingProviderCleanupPhase.COMPLETE,
            now=_NOW + datetime.timedelta(minutes=5),
        )

    completion_claims = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW,
        claimed_at=_NOW + datetime.timedelta(minutes=6),
        limit=10,
    )
    assert len(completion_claims) == 3
    for claim in completion_claims:
        assert consume_regulatory_provider_cleanup_delivery(
            db_session,
            job_id=claim.job_id,
            cleanup_generation=claim.cleanup_generation,
            cleanup_token=claim.cleanup_token,
            consumed_at=_NOW + datetime.timedelta(minutes=6),
        )
        assert complete_regulatory_provider_cleanup(
            db_session,
            job_id=claim.job_id,
            cleanup_generation=claim.cleanup_generation,
            now=_NOW + datetime.timedelta(minutes=6),
        )

    ready = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW + datetime.timedelta(minutes=7),
    )
    assert ready.ready_to_delete is True
    assert ready.deliveries == ()
    db_session.refresh(regulatory_user_file)
    assert regulatory_user_file.status is UserFileStatus.DELETING


def test_terminal_job_allows_a_new_chunk_generation_reindex(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    content_hash = uuid4().hex
    first = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="1" * 64,
    )
    first.status = RegulatoryIndexingJobStatus.SUCCEEDED.value
    first.completed_at = _NOW
    db_session.commit()

    second = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )

    assert second.id != first.id
    assert second.chunk_generation_hash == "2" * 64


def test_identical_successful_current_generation_is_idempotent_and_completed(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    content_hash = uuid4().hex
    completed = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )
    completed.status = RegulatoryIndexingJobStatus.SUCCEEDED.value
    completed.completed_at = _NOW
    regulatory_user_file.status = UserFileStatus.PROCESSING
    db_session.commit()

    repeated = _create_job(
        db_session,
        regulatory_user_file.id,
        content_hash=content_hash,
        chunk_generation_hash="2" * 64,
    )

    db_session.refresh(regulatory_user_file)
    assert repeated.id == completed.id
    assert regulatory_user_file.status is UserFileStatus.COMPLETED
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id
            )
        )
        == 1
    )


def test_concurrent_different_generations_create_only_one_active_job(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    engine = db_session.get_bind()
    barrier = Barrier(2)

    def create_in_independent_session(generation: str) -> UUID:
        with Session(engine) as independent_session:
            barrier.wait(timeout=5)
            return _create_job(
                independent_session,
                regulatory_user_file.id,
                content_hash=uuid4().hex,
                chunk_generation_hash=generation * 64,
            ).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(create_in_independent_session, "1")
        second = executor.submit(create_in_independent_session, "2")
        created_ids = [first.result(timeout=10), second.result(timeout=10)]

    assert created_ids[0] == created_ids[1]
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id,
                RegulatoryIndexingJob.status.in_(
                    [
                        RegulatoryIndexingJobStatus.QUEUED.value,
                        RegulatoryIndexingJobStatus.RUNNING.value,
                        RegulatoryIndexingJobStatus.RETRY_WAIT.value,
                        RegulatoryIndexingJobStatus.CANCELLING.value,
                    ]
                ),
            )
        )
        == 1
    )


def test_partial_unique_index_defends_against_a_second_active_job(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    first = _create_job(db_session, regulatory_user_file.id)
    second = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=regulatory_user_file.id,
        content_hash=uuid4().hex,
        chunk_generation_hash="9" * 64,
        search_settings_id=17,
        prompt_hash="prompt-v2",
        config_snapshot={
            **_SNAPSHOT,
            "input_content_hash": "f" * 64,
            "chunk_generation_hash": "9" * 64,
        },
        status=RegulatoryIndexingJobStatus.RUNNING.value,
        stage=RegulatoryIndexingStage.PREPARING.value,
        heartbeat_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )
    db_session.add(second)

    with pytest.raises(
        IntegrityError,
        match="uq_regulatory_indexing_job_active_user_file",
    ):
        db_session.commit()
    db_session.rollback()

    assert db_session.get(RegulatoryIndexingJob, first.id) is not None


def test_stale_failure_from_terminal_generation_cannot_overwrite_new_reindex(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    old_job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    assert fail_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        error_code="old_generation",
        error_message="terminal before reindex",
        now=_NOW,
    )
    db_session.refresh(old_job)
    assert (
        old_job.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.PENDING.value
    )

    new_job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    assert new_job.id != old_job.id
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=new_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    assert not fail_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        error_code="stale_delivery",
        error_message="must not affect the new generation",
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(new_job)
    db_session.refresh(regulatory_user_file)
    assert new_job.status == RegulatoryIndexingJobStatus.RUNNING.value
    assert regulatory_user_file.status is UserFileStatus.INDEXING


def test_superseded_lease_cannot_fail_file_before_or_after_successor_creation(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    old_job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    old_job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    reused = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    assert reused.id == old_job.id
    db_session.refresh(old_job)
    assert old_job.lease_generation == 2
    assert not fail_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
        expected_generation=1,
        error_code="stale_before_cleanup",
        error_message="must be fenced by supersession",
        now=_NOW,
    )
    db_session.refresh(regulatory_user_file)
    assert regulatory_user_file.status is UserFileStatus.INDEXING

    old_job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    db_session.commit()
    assert finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=old_job.id,
        expected_generation=2,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    successor = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    assert not fail_regulatory_indexing_job(
        db_session,
        job_id=old_job.id,
        expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
        expected_generation=2,
        error_code="stale_after_successor",
        error_message="must not overwrite successor state",
        now=_NOW + datetime.timedelta(minutes=2),
    )
    db_session.refresh(successor)
    db_session.refresh(regulatory_user_file)
    assert successor.status == RegulatoryIndexingJobStatus.QUEUED.value
    assert regulatory_user_file.status is UserFileStatus.INDEXING


def test_job_lease_can_be_claimed_only_once(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)

    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    assert not claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )

    db_session.refresh(job)
    assert job.status == RegulatoryIndexingJobStatus.RUNNING.value
    assert job.lease_generation == 1
    assert job.heartbeat_at == _NOW


def test_simultaneous_claims_have_exactly_one_winner(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    engine = db_session.get_bind()
    barrier = Barrier(2)

    def claim_in_independent_session() -> bool:
        with Session(engine) as independent_session:
            barrier.wait(timeout=5)
            return claim_regulatory_indexing_job(
                independent_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.PREPARING,
                expected_generation=0,
                now=_NOW,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim_in_independent_session(), range(2)))

    assert sorted(results) == [False, True]
    db_session.refresh(job)
    assert job.lease_generation == 1


def test_older_lease_cannot_advance_job_state(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )

    assert not advance_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        now=_NOW,
    )
    assert advance_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        now=_NOW,
    )

    db_session.refresh(job)
    assert job.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    assert job.status == RegulatoryIndexingJobStatus.QUEUED.value


def test_vertex_submission_identity_and_reconciliation_state_are_generation_fenced(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert job.vertex_submission_key is None
    assert job.vertex_submission_state == RegulatoryIndexingSubmissionState.NONE.value
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    submitted_chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 1 - Submitted item.",
        position=0,
        heading_path=["MADDE 1"],
        chunk_metadata={},
    )
    omitted_chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 2 - Omitted item.",
        position=1,
        heading_path=["MADDE 2"],
        chunk_metadata={},
    )
    db_session.add_all((submitted_chunk, omitted_chunk))
    db_session.commit()
    submitted_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=submitted_chunk.id,
        request_hash="f" * 64,
        expected_generation=1,
    )
    omitted_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=omitted_chunk.id,
        request_hash="e" * 64,
        expected_generation=1,
    )
    assert submitted_item is not None and omitted_item is not None
    job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    db_session.commit()
    submission_key = "regulatory-context-" + "a" * 64
    request_hashes = ("f" * 64,)

    assert not record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=0,
        submission_key=submission_key,
        submission_attempt=1,
        now=_NOW,
    )
    assert record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        submission_attempt=1,
        now=_NOW,
    )
    db_session.refresh(job)
    assert job.vertex_submission_key == submission_key
    assert (
        job.vertex_submission_state
        == RegulatoryIndexingSubmissionState.SUBMITTING.value
    )

    assert require_vertex_submission_reconciliation(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        request_hashes=request_hashes,
        reconcile_until=_NOW + datetime.timedelta(minutes=5),
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.vertex_submission_state
        == RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED.value
    )
    db_session.refresh(submitted_item)
    db_session.refresh(omitted_item)
    assert submitted_item.context_attempt_count == 1
    assert omitted_item.context_attempt_count == 0

    assert record_vertex_reconciliation_miss(
        db_session,
        job_id=job.id,
        expected_generation=1,
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.vertex_submission_state
        == RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED.value
    )
    assert job.vertex_reconcile_miss_count == 1

    assert record_vertex_submission(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        request_hashes=request_hashes,
        charge_items=False,
        remote_job_name="projects/p/locations/l/batchJobs/1",
        input_uri="gs://bucket/input.jsonl",
        output_uri="gs://bucket/output",
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.vertex_submission_state == RegulatoryIndexingSubmissionState.SUBMITTED.value
    )
    assert job.remote_vertex_job_name == "projects/p/locations/l/batchJobs/1"
    db_session.refresh(submitted_item)
    assert submitted_item.context_attempt_count == 1

    job.stage = RegulatoryIndexingStage.CONTEXT_APPLY.value
    db_session.commit()
    assert mark_vertex_partial_retry_cleanup_required(
        db_session,
        job_id=job.id,
        expected_generation=1,
        remote_job_name="projects/p/locations/l/batchJobs/1",
        now=_NOW + datetime.timedelta(minutes=1),
    )
    assert complete_vertex_partial_retry_cleanup(
        db_session,
        job_id=job.id,
        expected_generation=1,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(job)
    assert job.status == RegulatoryIndexingJobStatus.QUEUED.value
    assert job.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    assert job.remote_vertex_job_name is None
    assert job.vertex_output_uri is None
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=1,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    repeated_key = "regulatory-context-" + "b" * 64
    assert record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=2,
        submission_key=repeated_key,
        submission_attempt=2,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    assert record_vertex_submission(
        db_session,
        job_id=job.id,
        expected_generation=2,
        submission_key=repeated_key,
        request_hashes=request_hashes,
        charge_items=True,
        remote_job_name="projects/p/locations/l/batchJobs/2",
        input_uri="gs://bucket/input-2.jsonl",
        output_uri="gs://bucket/output-2",
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(submitted_item)
    db_session.refresh(omitted_item)
    assert submitted_item.context_attempt_count == 2
    assert omitted_item.context_attempt_count == 0


def test_openrouter_ambiguous_submission_is_persisted_and_charged_once(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 1 - Embedding item.",
        position=0,
        heading_path=["MADDE 1"],
        chunk_metadata={},
    )
    db_session.add(chunk)
    db_session.commit()
    item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="d" * 64,
        expected_generation=1,
    )
    assert item is not None
    item.status = RegulatoryIndexingItemStatus.SKIPPED.value
    job.stage = RegulatoryIndexingStage.EMBEDDING.value
    db_session.commit()
    submission_key = "regulatory-embedding-" + "c" * 64

    assert record_openrouter_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        submission_attempt=1,
        active_item_ids=[item.id],
        now=_NOW,
    )
    assert record_openrouter_submission_ambiguous(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        now=_NOW,
    )
    db_session.refresh(job)
    db_session.refresh(item)
    assert job.openrouter_submission_state == "MANUAL_RECONCILE_REQUIRED"
    assert job.remote_openrouter_batch_id is None
    assert job.openrouter_submission_charged is True
    assert item.embedding_attempt_count == 1

    assert record_openrouter_submission_ambiguous(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        now=_NOW,
    )
    db_session.refresh(item)
    assert item.embedding_attempt_count == 1


def test_illegal_stage_skip_cannot_persist_terminal_job_state(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )

    assert not advance_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.PUBLISH,
        next_status=RegulatoryIndexingJobStatus.SUCCEEDED,
        now=_NOW,
    )

    db_session.refresh(job)
    assert job.stage == RegulatoryIndexingStage.PREPARING.value
    assert job.status == RegulatoryIndexingJobStatus.RUNNING.value
    assert job.completed_at is None


def test_publish_stage_can_succeed_and_cancellation_can_finish(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    succeeded_job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=succeeded_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    succeeded_job.stage = RegulatoryIndexingStage.PUBLISH.value
    db_session.commit()
    assert advance_regulatory_indexing_job(
        db_session,
        job_id=succeeded_job.id,
        expected_stage=RegulatoryIndexingStage.PUBLISH,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.PUBLISH,
        next_status=RegulatoryIndexingJobStatus.SUCCEEDED,
        now=_NOW,
    )
    db_session.refresh(succeeded_job)
    assert succeeded_job.stage == RegulatoryIndexingStage.PUBLISH.value
    assert succeeded_job.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
    assert succeeded_job.completed_at == _NOW

    cancelled_job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=cancelled_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    assert advance_regulatory_indexing_job(
        db_session,
        job_id=cancelled_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.PREPARING,
        next_status=RegulatoryIndexingJobStatus.CANCELLING,
        now=_NOW,
    )
    db_session.refresh(cancelled_job)
    assert (
        cancelled_job.cancellation_phase
        == RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value
    )
    assert advance_regulatory_indexing_job(
        db_session,
        job_id=cancelled_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.PREPARING,
        next_status=RegulatoryIndexingJobStatus.CANCELLED,
        now=_NOW,
    )
    db_session.refresh(cancelled_job)
    assert cancelled_job.status == RegulatoryIndexingJobStatus.CANCELLED.value
    assert cancelled_job.completed_at == _NOW


def test_only_due_retries_are_claimable(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    due_job = _create_job(db_session, regulatory_user_file.id)
    future_file = _create_sibling_user_file(db_session, regulatory_user_file)
    future_job = _create_job(db_session, future_file.id)
    for job in (due_job, future_job):
        assert claim_regulatory_indexing_job(
            db_session,
            job_id=job.id,
            expected_stage=RegulatoryIndexingStage.PREPARING,
            expected_generation=0,
            now=_NOW,
        )

    assert schedule_regulatory_indexing_retry(
        db_session,
        job_id=due_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_retry_at=_NOW,
        error_code="rate_limited",
        error_message="retry now",
    )
    assert schedule_regulatory_indexing_retry(
        db_session,
        job_id=future_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_retry_at=_NOW + datetime.timedelta(minutes=5),
        error_code="rate_limited",
        error_message="retry later",
    )

    assert claim_regulatory_indexing_job(
        db_session,
        job_id=due_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        now=_NOW,
    )
    assert not claim_regulatory_indexing_job(
        db_session,
        job_id=future_job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        now=_NOW,
    )


def test_retry_error_code_is_bounded_to_persisted_column(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )

    assert schedule_regulatory_indexing_retry(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_retry_at=_NOW,
        error_code="x" * 200,
        error_message="safe diagnostic",
    )

    db_session.refresh(job)
    assert job.error_code == "x" * 128


def test_stale_recovery_claims_due_and_abandoned_jobs_only(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    stale_running = _create_job(db_session, regulatory_user_file.id)
    fresh_file = _create_sibling_user_file(db_session, regulatory_user_file)
    due_file = _create_sibling_user_file(db_session, regulatory_user_file)
    future_file = _create_sibling_user_file(db_session, regulatory_user_file)
    fresh_running = _create_job(db_session, fresh_file.id)
    due_retry = _create_job(db_session, due_file.id)
    future_retry = _create_job(db_session, future_file.id)

    for job in (stale_running, fresh_running, due_retry, future_retry):
        assert claim_regulatory_indexing_job(
            db_session,
            job_id=job.id,
            expected_stage=RegulatoryIndexingStage.PREPARING,
            expected_generation=0,
            now=_NOW,
        )
    stale_running.heartbeat_at = _NOW - datetime.timedelta(minutes=3)
    fresh_running.heartbeat_at = _NOW
    db_session.commit()
    assert schedule_regulatory_indexing_retry(
        db_session,
        job_id=due_retry.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_retry_at=_NOW,
        error_code="temporary",
        error_message="due",
    )
    assert schedule_regulatory_indexing_retry(
        db_session,
        job_id=future_retry.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_retry_at=_NOW + datetime.timedelta(minutes=5),
        error_code="temporary",
        error_message="future",
    )

    claims = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=_NOW - datetime.timedelta(minutes=2),
        claimed_at=_NOW,
        limit=10,
    )

    assert {claim.job_id for claim in claims} == {stale_running.id, due_retry.id}
    assert all(claim.lease_generation == 2 for claim in claims)


def test_simultaneous_recovery_claims_stale_job_once(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW - datetime.timedelta(minutes=3),
    )
    engine = db_session.get_bind()
    barrier = Barrier(2)

    def recover_in_independent_session() -> list[UUID]:
        with Session(engine) as independent_session:
            barrier.wait(timeout=5)
            return [
                claim.job_id
                for claim in claim_stale_regulatory_indexing_jobs(
                    independent_session,
                    stale_before=_NOW - datetime.timedelta(minutes=2),
                    claimed_at=_NOW,
                    limit=1,
                )
            ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: recover_in_independent_session(), range(2))
        )

    assert [job_id for claimed_ids in results for job_id in claimed_ids] == [job.id]
    db_session.refresh(job)
    assert job.lease_generation == 2


def test_preclaimed_recovery_token_has_exactly_one_consumer(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW - datetime.timedelta(minutes=3),
    )
    claims = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=_NOW - datetime.timedelta(minutes=2),
        claimed_at=_NOW,
        limit=1,
    )
    assert len(claims) == 1
    claim = claims[0]
    engine = db_session.get_bind()
    barrier = Barrier(2)

    def consume_in_independent_session() -> bool:
        with Session(engine) as independent_session:
            barrier.wait(timeout=5)
            return consume_preclaimed_regulatory_indexing_delivery(
                independent_session,
                job_id=claim.job_id,
                expected_generation=claim.lease_generation,
                recovery_token=claim.recovery_token,
                consumed_at=_NOW + datetime.timedelta(seconds=1),
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: consume_in_independent_session(), range(2))
        )

    assert sorted(results) == [False, True]


def test_submission_external_lease_prevents_stale_recovery_during_create(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    db_session.commit()
    assert record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key="regulatory-context-" + "c" * 64,
        submission_attempt=1,
        now=_NOW,
    )
    engine = db_session.get_bind()
    entered = Event()
    release = Event()

    def hold_create_lease() -> None:
        with Session(engine) as create_session:
            with regulatory_indexing_external_mutation_lease(
                create_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
                expected_generation=1,
            ) as lease:
                assert lease is not None
                entered.set()
                assert release.wait(timeout=5)
                lease.commit()

    future_time = _NOW + datetime.timedelta(days=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        held = executor.submit(hold_create_lease)
        assert entered.wait(timeout=5)
        assert (
            claim_stale_regulatory_indexing_jobs(
                db_session,
                stale_before=future_time,
                claimed_at=future_time,
                limit=1,
            )
            == []
        )
        release.set()
        held.result(timeout=5)

    claims = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=future_time,
        claimed_at=future_time,
        limit=1,
    )
    assert len(claims) == 1
    assert claims[0].job_id == job.id


def test_item_creation_requires_current_running_lease_and_matching_request_hash(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 1 - Olusturma fence testi.",
        position=0,
        heading_path=["MADDE 1"],
        chunk_metadata={},
    )
    db_session.add(chunk)
    db_session.commit()

    assert (
        create_or_get_regulatory_indexing_item(
            db_session,
            job_id=job.id,
            regulatory_chunk_id=chunk.id,
            request_hash="request-v1",
            expected_generation=0,
        )
        is None
    )
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    assert (
        create_or_get_regulatory_indexing_item(
            db_session,
            job_id=job.id,
            regulatory_chunk_id=chunk.id,
            request_hash="request-v1",
            expected_generation=0,
        )
        is None
    )
    item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="request-v1",
        expected_generation=1,
    )
    assert item is not None
    with pytest.raises(ValueError, match="request hash"):
        create_or_get_regulatory_indexing_item(
            db_session,
            job_id=job.id,
            regulatory_chunk_id=chunk.id,
            request_hash="request-v2",
            expected_generation=1,
        )


def test_item_results_are_unique_and_fenced_by_the_job_lease(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 1 - Kalici test metni.",
        position=0,
        heading_path=["MADDE 1"],
        chunk_metadata={"article_no": "1"},
    )
    db_session.add(chunk)
    db_session.commit()
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="request-v1",
        expected_generation=1,
    )
    assert item is not None
    duplicate = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="request-v1",
        expected_generation=1,
    )
    assert duplicate is not None
    assert duplicate.id == item.id
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingItem.id)).where(
                RegulatoryIndexingItem.job_id == job.id
            )
        )
        == 1
    )

    assert not persist_regulatory_indexing_item_context(
        db_session,
        item_id=item.id,
        expected_generation=0,
        context={"text": "Gecersiz eski sonuc"},
    )
    assert persist_regulatory_indexing_item_context(
        db_session,
        item_id=item.id,
        expected_generation=1,
        context={"text": "Kalici baglam"},
    )
    assert persist_regulatory_indexing_item_vector(
        db_session,
        item_id=item.id,
        expected_generation=1,
        vector=[0.25, -0.5, 0.75],
    )

    db_session.refresh(item)
    assert item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
    assert item.context == {"text": "Kalici baglam"}
    assert item.vector == [0.25, -0.5, 0.75]

    failed_chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 2 - Hata sonucu test metni.",
        position=1,
        heading_path=["MADDE 2"],
        chunk_metadata={"article_no": "2"},
    )
    db_session.add(failed_chunk)
    db_session.commit()
    failed_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=failed_chunk.id,
        request_hash="request-v2",
        expected_generation=1,
    )
    assert failed_item is not None
    assert persist_regulatory_indexing_item_failure(
        db_session,
        item_id=failed_item.id,
        expected_generation=1,
        error_code="malformed_output",
        error_message="invalid result",
    )
    db_session.refresh(failed_item)
    assert failed_item.status == RegulatoryIndexingItemStatus.FAILED.value
    assert failed_item.error_code == "malformed_output"
    assert failed_item.error_message == "invalid result"

    skipped_chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 3 - Baglamsiz gomulecek test metni.",
        position=2,
        heading_path=["MADDE 3"],
        chunk_metadata={"article_no": "3"},
    )
    db_session.add(skipped_chunk)
    db_session.commit()
    skipped_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=skipped_chunk.id,
        request_hash="request-v3",
        expected_generation=1,
    )
    assert skipped_item is not None
    assert persist_regulatory_indexing_item_skipped(
        db_session,
        item_id=skipped_item.id,
        expected_generation=1,
    )
    assert persist_regulatory_indexing_item_vector(
        db_session,
        item_id=skipped_item.id,
        expected_generation=1,
        vector=[1.0, 0.0, 0.0],
    )
    db_session.refresh(skipped_item)
    assert skipped_item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
    assert skipped_item.context is None
    assert skipped_item.vector == [1.0, 0.0, 0.0]


def test_item_write_waits_for_takeover_then_rejects_stale_generation(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    chunk = RegulatoryChunk(
        id=f"regulatory-chunk-{uuid4().hex}",
        user_file_id=regulatory_user_file.id,
        text="MADDE 4 - Eski lease sonucu yazilamaz.",
        position=0,
        heading_path=["MADDE 4"],
        chunk_metadata={},
    )
    db_session.add(chunk)
    db_session.commit()
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW - datetime.timedelta(minutes=3),
    )
    item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="request-v4",
        expected_generation=1,
    )
    assert item is not None
    assert persist_regulatory_indexing_item_context(
        db_session,
        item_id=item.id,
        expected_generation=1,
        context={"text": "Kalici baglam"},
    )
    engine = db_session.get_bind()
    takeover_session = Session(engine)
    locked_job = takeover_session.scalar(
        select(RegulatoryIndexingJob)
        .where(RegulatoryIndexingJob.id == job.id)
        .with_for_update()
    )
    assert locked_job is not None
    locked_job.lease_generation = 2
    locked_job.heartbeat_at = _NOW
    takeover_session.flush()

    def persist_stale_vector() -> bool:
        with Session(engine) as stale_session:
            return persist_regulatory_indexing_item_vector(
                stale_session,
                item_id=item.id,
                expected_generation=1,
                vector=[9.0, 9.0, 9.0],
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            stale_write = executor.submit(persist_stale_vector)
            try:
                stale_write.result(timeout=0.3)
            except TimeoutError:
                pass
            else:
                raise AssertionError(
                    "stale item write did not wait for job-row takeover"
                )
            takeover_session.commit()
            assert stale_write.result(timeout=5) is False
    finally:
        takeover_session.close()

    assert persist_regulatory_indexing_item_vector(
        db_session,
        item_id=item.id,
        expected_generation=2,
        vector=[1.0, 2.0, 3.0],
    )
    db_session.refresh(item)
    assert item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
    assert item.vector == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "preparation_file_status",
    (UserFileStatus.PROCESSING, UserFileStatus.FAILED),
)
def test_atomic_preparation_repairs_partial_state_and_advances_once(
    db_session: Session,
    regulatory_user_file: UserFile,
    preparation_file_status: UserFileStatus,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    partial_chunks = (
        RegulatoryChunker(min_chunk_chars=0)
        .chunk_text("MADDE 1 - Yarım kalmış hazırlık.", source_file="mevzuat.md")
        .chunks
    )
    partial_rows = replace_indexed_chunks_for_file(
        db_session, regulatory_user_file.id, partial_chunks
    )
    db_session.flush()
    partial_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=partial_rows[0].id,
        request_hash="partial-request",
        expected_generation=1,
    )
    assert partial_item is not None
    partial_row_id = partial_rows[0].id
    partial_item_id = partial_item.id
    regulatory_user_file.status = preparation_file_status
    job.config_snapshot = {
        **job.config_snapshot,
        "input_hash_version": "legacy-or-canonical",
    }
    db_session.commit()

    def prepare_recovered_items() -> list[
        regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem
    ]:
        recovered_chunks = (
            RegulatoryChunker(min_chunk_chars=0)
            .chunk_text("MADDE 2 - Kurtarılmış hazırlık.", source_file="mevzuat.md")
            .chunks
        )
        recovered_rows = replace_indexed_chunks_for_file(
            db_session, regulatory_user_file.id, recovered_chunks
        )
        db_session.flush()
        return [
            regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem(
                regulatory_chunk_id=recovered_rows[0].id,
                request_hash="recovered-request",
                skip_context=True,
            )
        ]

    persisted = (
        regulatory_indexing_job_repository.persist_regulatory_indexing_preparation(
            db_session,
            job_id=job.id,
            expected_generation=1,
            prepare_items=prepare_recovered_items,
            resolved_input_hash_version="canonical-v2",
            now=_NOW,
        )
    )

    assert persisted
    db_session.expire_all()
    recovered_job = db_session.get(RegulatoryIndexingJob, job.id)
    assert recovered_job is not None
    assert recovered_job.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    assert recovered_job.status == RegulatoryIndexingJobStatus.QUEUED.value
    assert recovered_job.config_snapshot["input_hash_version"] == "canonical-v2"
    recovered_file = db_session.get(UserFile, regulatory_user_file.id)
    assert recovered_file is not None
    assert recovered_file.status is UserFileStatus.INDEXING
    assert db_session.get(RegulatoryChunk, partial_row_id) is None
    assert db_session.get(RegulatoryIndexingItem, partial_item_id) is None
    recovered_items = list(
        db_session.scalars(
            select(RegulatoryIndexingItem).where(
                RegulatoryIndexingItem.job_id == job.id
            )
        ).all()
    )
    assert len(recovered_items) == 1
    assert recovered_items[0].status == RegulatoryIndexingItemStatus.SKIPPED.value
    assert recovered_items[0].request_hash == "recovered-request"


def test_atomic_preparation_rolls_back_callback_writes_on_validation_failure(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    original_chunks = (
        RegulatoryChunker(min_chunk_chars=0)
        .chunk_text("MADDE 1 - Korunacak hüküm.", source_file="mevzuat.md")
        .chunks
    )
    original_rows = replace_indexed_chunks_for_file(
        db_session, regulatory_user_file.id, original_chunks
    )
    job.config_snapshot = {
        **job.config_snapshot,
        "input_hash_version": "legacy-or-canonical",
    }
    db_session.commit()

    def prepare_invalid_items() -> list[
        regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem
    ]:
        replacement_chunks = (
            RegulatoryChunker(min_chunk_chars=0)
            .chunk_text("MADDE 9 - Geri alınacak hüküm.", source_file="mevzuat.md")
            .chunks
        )
        replacement_rows = replace_indexed_chunks_for_file(
            db_session, regulatory_user_file.id, replacement_chunks
        )
        db_session.flush()
        return [
            regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem(
                regulatory_chunk_id=replacement_rows[0].id,
                request_hash="duplicate-request",
                skip_context=False,
            ),
            regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem(
                regulatory_chunk_id=replacement_rows[0].id,
                request_hash="duplicate-request",
                skip_context=False,
            ),
        ]

    with pytest.raises(ValueError, match="duplicate chunks"):
        regulatory_indexing_job_repository.persist_regulatory_indexing_preparation(
            db_session,
            job_id=job.id,
            expected_generation=1,
            prepare_items=prepare_invalid_items,
            resolved_input_hash_version="canonical-v2",
            now=_NOW,
        )

    db_session.expire_all()
    persisted_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == regulatory_user_file.id
            )
        ).all()
    )
    persisted_job = db_session.get(RegulatoryIndexingJob, job.id)
    assert [(row.id, row.text) for row in persisted_rows] == [
        (original_rows[0].id, "MADDE 1 - Korunacak hüküm.")
    ]
    assert persisted_job is not None
    assert persisted_job.stage == RegulatoryIndexingStage.PREPARING.value
    assert persisted_job.status == RegulatoryIndexingJobStatus.RUNNING.value
    assert persisted_job.config_snapshot["input_hash_version"] == "legacy-or-canonical"


def test_lease_takeover_fences_stale_atomic_chunk_replacement(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    current_chunks = (
        RegulatoryChunker(min_chunk_chars=0)
        .chunk_text("MADDE 1 - Yeni neslin korunan hükmü.", source_file="mevzuat.md")
        .chunks
    )
    current_rows = replace_indexed_chunks_for_file(
        db_session, regulatory_user_file.id, current_chunks
    )
    db_session.flush()
    current_item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=current_rows[0].id,
        request_hash="current-request",
        expected_generation=1,
    )
    assert current_item is not None
    engine = db_session.get_bind()
    claims = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=_NOW + datetime.timedelta(seconds=1),
        claimed_at=_NOW + datetime.timedelta(seconds=2),
    )
    assert [(claim.job_id, claim.lease_generation) for claim in claims] == [(job.id, 2)]

    newer_prepared = Event()
    allow_newer_commit = Event()
    stale_callback_started = Event()

    def takeover_prepare() -> bool:
        with Session(engine) as takeover_session:

            def replace_with_newer_chunks() -> list[
                regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem
            ]:
                newer_chunks = (
                    RegulatoryChunker(min_chunk_chars=0)
                    .chunk_text(
                        "MADDE 2 - Yeni neslin korunan hükmü.",
                        source_file="mevzuat.md",
                    )
                    .chunks
                )
                newer_rows = replace_indexed_chunks_for_file(
                    takeover_session, regulatory_user_file.id, newer_chunks
                )
                takeover_session.flush()
                newer_prepared.set()
                assert allow_newer_commit.wait(timeout=5)
                return [
                    regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem(
                        regulatory_chunk_id=newer_rows[0].id,
                        request_hash="newer-request",
                        skip_context=False,
                    )
                ]

            return regulatory_indexing_job_repository.persist_regulatory_indexing_preparation(
                takeover_session,
                job_id=job.id,
                expected_generation=2,
                prepare_items=replace_with_newer_chunks,
                resolved_input_hash_version="canonical-v2",
                now=_NOW + datetime.timedelta(seconds=3),
            )

    def stale_prepare() -> bool:
        with Session(engine) as stale_session:

            def replace_with_stale_chunks() -> list[
                regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem
            ]:
                stale_callback_started.set()
                stale_chunks = (
                    RegulatoryChunker(min_chunk_chars=0)
                    .chunk_text(
                        "MADDE 9 - Eski neslin yazmaması gereken hükmü.",
                        source_file="mevzuat.md",
                    )
                    .chunks
                )
                stale_rows = replace_indexed_chunks_for_file(
                    stale_session, regulatory_user_file.id, stale_chunks
                )
                stale_session.flush()
                return [
                    regulatory_indexing_job_repository.RegulatoryIndexingPreparedItem(
                        regulatory_chunk_id=stale_rows[0].id,
                        request_hash="stale-request",
                        skip_context=False,
                    )
                ]

            return regulatory_indexing_job_repository.persist_regulatory_indexing_preparation(
                stale_session,
                job_id=job.id,
                expected_generation=1,
                prepare_items=replace_with_stale_chunks,
                resolved_input_hash_version="canonical-v2",
                now=_NOW,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        takeover_future = executor.submit(takeover_prepare)
        assert newer_prepared.wait(timeout=5)
        stale_future = executor.submit(stale_prepare)
        with pytest.raises(TimeoutError):
            stale_future.result(timeout=0.3)
        assert not stale_callback_started.is_set()
        allow_newer_commit.set()
        assert takeover_future.result(timeout=5) is True
        assert stale_future.result(timeout=5) is False
        assert not stale_callback_started.is_set()

    db_session.expire_all()
    chunk_texts = set(
        db_session.scalars(
            select(RegulatoryChunk.text).where(
                RegulatoryChunk.user_file_id == regulatory_user_file.id
            )
        ).all()
    )
    assert chunk_texts == {"MADDE 2 - Yeni neslin korunan hükmü."}
    item_hashes = set(
        db_session.scalars(
            select(RegulatoryIndexingItem.request_hash).where(
                RegulatoryIndexingItem.job_id == job.id
            )
        ).all()
    )
    assert item_hashes == {"newer-request"}


def _create_vector_items(
    db_session: Session,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
) -> list[RegulatoryIndexingItem]:
    items: list[RegulatoryIndexingItem] = []
    for position in range(2):
        chunk = RegulatoryChunk(
            id=f"vector-chunk-{uuid4().hex}",
            user_file_id=user_file.id,
            text=f"MADDE {position + 1} - Vektor testi.",
            position=position,
            heading_path=[f"MADDE {position + 1}"],
            chunk_metadata={},
        )
        item = RegulatoryIndexingItem(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=chunk.id,
            request_hash=f"request-{position}",
            status=RegulatoryIndexingItemStatus.CONTEXT_READY.value,
        )
        db_session.add_all([chunk, item])
        items.append(item)
    db_session.commit()
    return items


def test_multi_vector_persistence_is_atomic_and_generation_fenced(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.EMBEDDING.value
    db_session.commit()
    items = _create_vector_items(db_session, job, regulatory_user_file)

    assert persist_regulatory_indexing_item_vectors(
        db_session,
        job_id=job.id,
        expected_generation=1,
        item_vectors=[
            (items[0].id, [0.1, 0.2, 0.3]),
            (items[1].id, [0.4, 0.5, 0.6]),
        ],
    )
    for item, expected in zip(items, ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]), strict=True):
        db_session.refresh(item)
        assert item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
        assert item.vector == expected

    items[0].status = RegulatoryIndexingItemStatus.CONTEXT_READY.value
    items[0].vector = None
    db_session.commit()
    assert not persist_regulatory_indexing_item_vectors(
        db_session,
        job_id=job.id,
        expected_generation=1,
        item_vectors=[
            (items[0].id, [9.0, 9.0, 9.0]),
            (uuid4(), [8.0, 8.0, 8.0]),
        ],
    )
    db_session.refresh(items[0])
    assert items[0].status == RegulatoryIndexingItemStatus.CONTEXT_READY.value
    assert items[0].vector is None

    job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    db_session.commit()
    assert not persist_regulatory_indexing_item_vectors(
        db_session,
        job_id=job.id,
        expected_generation=1,
        item_vectors=[(items[0].id, [1.0, 1.0, 1.0])],
    )
    job.stage = RegulatoryIndexingStage.EMBEDDING.value
    db_session.commit()
    assert not persist_regulatory_indexing_item_vectors(
        db_session,
        job_id=job.id,
        expected_generation=0,
        item_vectors=[(items[0].id, [1.0, 1.0, 1.0])],
    )


def test_external_mutation_lock_excludes_recovery_and_refreshes_heartbeat(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW - datetime.timedelta(minutes=5),
    )
    job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()
    engine = db_session.get_bind()
    entered = Event()
    release = Event()

    def hold_external_mutation() -> None:
        with Session(engine) as mutation_session:
            with regulatory_indexing_external_mutation_lease(
                mutation_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                expected_generation=1,
            ) as lease:
                assert lease is not None
                entered.set()
                assert release.wait(timeout=5)
                lease.commit()

    with ThreadPoolExecutor(max_workers=1) as executor:
        held = executor.submit(hold_external_mutation)
        assert entered.wait(timeout=5)
        with Session(engine) as recovery_session:
            claims = claim_stale_regulatory_indexing_jobs(
                recovery_session,
                stale_before=_NOW - datetime.timedelta(minutes=2),
                claimed_at=_NOW,
                limit=10,
            )
        assert claims == []
        release.set()
        held.result(timeout=5)

    db_session.refresh(job)
    assert job.lease_generation == 1
    assert job.heartbeat_at is not None
    assert job.heartbeat_at > _NOW


def test_external_mutation_lock_blocks_cancellation_until_es_finishes(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()
    engine = db_session.get_bind()
    entered = Event()
    release = Event()

    def hold_external_mutation() -> None:
        with Session(engine) as mutation_session:
            with regulatory_indexing_external_mutation_lease(
                mutation_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                expected_generation=1,
            ) as lease:
                assert lease is not None
                entered.set()
                assert release.wait(timeout=5)
                lease.commit()

    def cancel_job() -> bool:
        with Session(engine) as cancel_session:
            return advance_regulatory_indexing_job(
                cancel_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                expected_generation=1,
                next_stage=RegulatoryIndexingStage.INDEX_WRITE,
                next_status=RegulatoryIndexingJobStatus.CANCELLING,
                now=_NOW,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        held = executor.submit(hold_external_mutation)
        assert entered.wait(timeout=5)
        cancellation = executor.submit(cancel_job)
        with pytest.raises(TimeoutError):
            cancellation.result(timeout=0.3)
        release.set()
        held.result(timeout=5)
        assert cancellation.result(timeout=5)


def test_external_mutation_lock_blocks_durable_user_file_deletion_cleanup(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()
    engine = db_session.get_bind()
    entered = Event()
    release = Event()

    def hold_external_mutation() -> None:
        with Session(engine) as mutation_session:
            with regulatory_indexing_external_mutation_lease(
                mutation_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                expected_generation=1,
            ) as lease:
                assert lease is not None
                entered.set()
                assert release.wait(timeout=5)
                lease.commit()

    def request_deletion_cleanup() -> tuple[bool, tuple[tuple[UUID, int], ...]]:
        with Session(engine) as delete_session:
            plan = request_user_file_deletion_cleanup(
                delete_session,
                user_file_id=regulatory_user_file.id,
                now=_NOW,
            )
            return plan.ready_to_delete, tuple(
                (delivery.job_id, delivery.expected_generation)
                for delivery in plan.deliveries
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        held = executor.submit(hold_external_mutation)
        assert entered.wait(timeout=5)
        deletion = executor.submit(request_deletion_cleanup)
        with pytest.raises(TimeoutError):
            deletion.result(timeout=0.3)
        release.set()
        held.result(timeout=5)
        ready, deliveries = deletion.result(timeout=5)

    db_session.refresh(regulatory_user_file)
    db_session.refresh(job)
    assert regulatory_user_file.status is UserFileStatus.DELETING
    assert ready is False
    assert deliveries == ((job.id, 2),)
    assert job.status == RegulatoryIndexingJobStatus.CANCELLING.value
    assert job.cancellation_intent == "USER_DELETE"
    assert job.lease_generation == 2


def test_user_file_completion_preserves_cancelled_and_deleting_states(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    cases = (
        (UserFileStatus.INDEXING, True),
        (UserFileStatus.COMPLETED, True),
        (UserFileStatus.CANCELED, False),
        (UserFileStatus.DELETING, False),
    )
    for index, (initial_status, should_complete) in enumerate(cases):
        case_user_file = (
            regulatory_user_file
            if index == 0
            else _create_sibling_user_file(db_session, regulatory_user_file)
        )
        case_user_file.status = UserFileStatus.PROCESSING
        db_session.commit()
        job = _create_job(db_session, case_user_file.id)
        assert claim_regulatory_indexing_job(
            db_session,
            job_id=job.id,
            expected_stage=RegulatoryIndexingStage.PREPARING,
            expected_generation=0,
            now=_NOW,
        )
        job.stage = RegulatoryIndexingStage.PUBLISH.value
        case_user_file.status = initial_status
        db_session.commit()

        assert (
            complete_regulatory_indexing_user_file(
                db_session,
                job_id=job.id,
                expected_generation=1,
                chunk_count=2,
                now=_NOW,
            )
            is should_complete
        )
        db_session.refresh(case_user_file)
        assert case_user_file.status is (
            UserFileStatus.COMPLETED if should_complete else initial_status
        )


def test_publication_atomically_completes_user_file_and_job(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.PUBLISH.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    assert complete_regulatory_indexing_publication(
        db_session,
        job_id=job.id,
        expected_generation=1,
        chunk_count=2,
        now=_NOW,
        commit=True,
    )

    db_session.refresh(job)
    db_session.refresh(regulatory_user_file)
    assert job.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
    assert job.completed_at == _NOW
    assert regulatory_user_file.status is UserFileStatus.COMPLETED
    assert regulatory_user_file.chunk_count == 2
    assert regulatory_user_file.secondary_reconcile_pending is True
    assert (
        job.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.PENDING.value
    )
    assert (
        job.provider_cleanup_phase
        == RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP.value
    )

    claims = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW - datetime.timedelta(minutes=1),
        claimed_at=_NOW,
        limit=1,
    )
    assert len(claims) == 1
    claim = claims[0]
    assert claim.job_id == job.id
    assert consume_regulatory_provider_cleanup_delivery(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        cleanup_token=claim.cleanup_token,
        consumed_at=_NOW,
    )
    assert not consume_regulatory_provider_cleanup_delivery(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        cleanup_token=claim.cleanup_token,
        consumed_at=_NOW,
    )
    sweep_at = _NOW + datetime.timedelta(minutes=15)
    assert schedule_regulatory_provider_cleanup_retry(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        next_retry_at=sweep_at,
        error_code="provider_unavailable",
        error_message="IndexingGatewayConnectionError",
        exhausted=True,
    )
    db_session.refresh(job)
    assert (
        job.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.EXHAUSTED.value
    )
    assert (
        claim_due_regulatory_provider_cleanups(
            db_session,
            stale_before=_NOW,
            claimed_at=sweep_at - datetime.timedelta(seconds=1),
            limit=1,
        )
        == []
    )
    swept = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW,
        claimed_at=sweep_at,
        limit=1,
    )
    assert len(swept) == 1
    db_session.refresh(job)
    assert job.provider_cleanup_attempt_count == 0


def test_terminal_cleanup_persists_a_late_visible_vertex_job(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    submission_key = "regulatory-context-" + "f" * 64
    job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    job.vertex_submission_key = submission_key
    job.vertex_submission_state = (
        RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED.value
    )
    db_session.commit()
    assert fail_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=1,
        error_code="visibility_horizon",
        error_message="VertexBatchContractError",
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.provider_cleanup_phase
        == RegulatoryIndexingProviderCleanupPhase.VERTEX_RECONCILE.value
    )
    claim = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW - datetime.timedelta(minutes=1),
        claimed_at=_NOW,
        limit=1,
    )[0]
    assert consume_regulatory_provider_cleanup_delivery(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        cleanup_token=claim.cleanup_token,
        consumed_at=_NOW,
    )
    assert record_reconciled_provider_cleanup_vertex_job(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        submission_key=submission_key,
        remote_job_name="projects/p/locations/l/batchJobs/late",
        input_uri="gs://bucket/input.jsonl",
        output_uri="gs://bucket/output",
        now=_NOW,
    )
    db_session.refresh(job)
    assert job.remote_vertex_job_name is not None
    assert job.remote_vertex_job_name.endswith("/late")
    assert (
        job.provider_cleanup_phase
        == RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE.value
    )


@pytest.mark.parametrize(
    "stage,status",
    [
        (RegulatoryIndexingStage.PREPARING, RegulatoryIndexingJobStatus.QUEUED),
        (RegulatoryIndexingStage.CONTEXT_WAIT, RegulatoryIndexingJobStatus.RUNNING),
        (RegulatoryIndexingStage.EMBEDDING, RegulatoryIndexingJobStatus.RETRY_WAIT),
        (RegulatoryIndexingStage.PUBLISH, RegulatoryIndexingJobStatus.SUCCEEDED),
        (RegulatoryIndexingStage.INDEX_WRITE, RegulatoryIndexingJobStatus.FAILED),
    ],
)
def test_deletion_tombstone_generation_fences_every_non_cancelled_job(
    db_session: Session,
    regulatory_user_file: UserFile,
    stage: RegulatoryIndexingStage,
    status: RegulatoryIndexingJobStatus,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    job.stage = stage.value
    job.status = status.value
    job.lease_generation = 7
    job.remote_vertex_job_name = "projects/p/locations/l/batchJobs/1"
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()

    first = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )

    db_session.refresh(job)
    db_session.refresh(regulatory_user_file)
    assert first.ready_to_delete is False
    assert [
        (delivery.job_id, delivery.expected_generation) for delivery in first.deliveries
    ] == [(job.id, 8)]
    assert regulatory_user_file.status is UserFileStatus.DELETING
    assert job.status == RegulatoryIndexingJobStatus.CANCELLING.value
    assert job.cancellation_intent == "USER_DELETE"
    assert job.lease_generation == 8
    assert (
        job.cancellation_phase
        == RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
    )

    repeated = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )
    db_session.refresh(job)
    assert job.lease_generation == 8
    assert job.cancellation_intent == "USER_DELETE"
    assert repeated.deliveries == first.deliveries


@pytest.mark.parametrize(
    "cleanup_state,cleanup_phase",
    [
        (
            RegulatoryIndexingProviderCleanupState.PENDING,
            RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP,
        ),
        (
            RegulatoryIndexingProviderCleanupState.RETRY_WAIT,
            RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP,
        ),
        (
            RegulatoryIndexingProviderCleanupState.EXHAUSTED,
            RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP,
        ),
        (
            RegulatoryIndexingProviderCleanupState.PENDING,
            RegulatoryIndexingProviderCleanupPhase.VERTEX_RECONCILE,
        ),
    ],
)
def test_deletion_blocks_every_incomplete_provider_cleanup_state(
    db_session: Session,
    regulatory_user_file: UserFile,
    cleanup_state: RegulatoryIndexingProviderCleanupState,
    cleanup_phase: RegulatoryIndexingProviderCleanupPhase,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    job.status = RegulatoryIndexingJobStatus.CANCELLED.value
    job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    job.provider_cleanup_state = cleanup_state.value
    job.provider_cleanup_phase = cleanup_phase.value
    if cleanup_state in {
        RegulatoryIndexingProviderCleanupState.RETRY_WAIT,
        RegulatoryIndexingProviderCleanupState.EXHAUSTED,
    }:
        job.provider_cleanup_next_retry_at = _NOW + datetime.timedelta(minutes=5)
    regulatory_user_file.status = UserFileStatus.DELETING
    db_session.commit()

    plan = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )

    assert plan.ready_to_delete is False
    assert plan.deliveries == ()
    assert db_session.get(UserFile, regulatory_user_file.id) is not None
    assert db_session.get(RegulatoryIndexingJob, job.id) is not None


def test_deletion_becomes_ready_only_after_cancelled_job_cleanup_completes(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    job.status = RegulatoryIndexingJobStatus.CANCELLED.value
    job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    job.provider_cleanup_state = RegulatoryIndexingProviderCleanupState.PENDING.value
    job.provider_cleanup_phase = RegulatoryIndexingProviderCleanupPhase.COMPLETE.value
    job.completed_at = _NOW
    regulatory_user_file.status = UserFileStatus.DELETING
    db_session.commit()

    blocked = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )
    assert blocked.ready_to_delete is False
    assert blocked.deliveries == ()

    claim = claim_due_regulatory_provider_cleanups(
        db_session,
        stale_before=_NOW - datetime.timedelta(minutes=1),
        claimed_at=_NOW,
        limit=1,
    )[0]
    assert consume_regulatory_provider_cleanup_delivery(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        cleanup_token=claim.cleanup_token,
        consumed_at=_NOW,
    )
    assert complete_regulatory_provider_cleanup(
        db_session,
        job_id=job.id,
        cleanup_generation=claim.cleanup_generation,
        now=_NOW + datetime.timedelta(seconds=1),
    )

    db_session.expire_all()
    ready = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW + datetime.timedelta(seconds=2),
    )
    assert ready.ready_to_delete is True
    assert ready.deliveries == ()


def test_deletion_schedules_cleanup_for_a_legacy_cancelled_job(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    job.status = RegulatoryIndexingJobStatus.CANCELLED.value
    job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    job.provider_cleanup_state = RegulatoryIndexingProviderCleanupState.NONE.value
    job.provider_cleanup_phase = RegulatoryIndexingProviderCleanupPhase.NONE.value
    job.completed_at = _NOW
    regulatory_user_file.status = UserFileStatus.DELETING
    db_session.commit()

    blocked = request_user_file_deletion_cleanup(
        db_session,
        user_file_id=regulatory_user_file.id,
        now=_NOW,
    )

    db_session.refresh(job)
    assert blocked.ready_to_delete is False
    assert blocked.deliveries == ()
    assert (
        job.provider_cleanup_state
        == RegulatoryIndexingProviderCleanupState.PENDING.value
    )
    assert (
        job.provider_cleanup_phase
        == RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP.value
    )


def test_user_cancel_intent_remains_distinct_from_supersession_and_deletion(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    regulatory_user_file.status = UserFileStatus.CANCELED
    db_session.commit()

    assert request_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        cancellation_intent=RegulatoryIndexingCancellationIntent.USER_CANCEL,
        now=_NOW,
    )
    db_session.refresh(job)
    assert job.cancellation_intent == "USER_CANCEL"
    job.cancellation_phase = RegulatoryIndexingCancellationPhase.FINALIZE.value
    db_session.commit()
    assert finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_generation=1,
        now=_NOW + datetime.timedelta(minutes=1),
    )
    db_session.refresh(regulatory_user_file)
    assert regulatory_user_file.status is UserFileStatus.CANCELED


def test_cancellation_phases_are_durable_and_generation_fenced(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.remote_vertex_job_name = "projects/p/locations/l/batchJobs/1"
    regulatory_user_file.status = UserFileStatus.DELETING
    db_session.commit()
    assert request_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        cancellation_intent=RegulatoryIndexingCancellationIntent.USER_DELETE,
        now=_NOW,
    )
    db_session.refresh(job)
    assert job.status == RegulatoryIndexingJobStatus.CANCELLING.value
    assert (
        job.cancellation_phase
        == RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value
    )

    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        now=_NOW,
    )
    assert not advance_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_generation=1,
        expected_phase=RegulatoryIndexingCancellationPhase.VERTEX_CANCEL,
        next_phase=RegulatoryIndexingCancellationPhase.GCS_CLEANUP,
        now=_NOW,
    )
    generation = 2
    for expected_phase, next_phase in (
        (
            RegulatoryIndexingCancellationPhase.VERTEX_CANCEL,
            RegulatoryIndexingCancellationPhase.GCS_CLEANUP,
        ),
        (
            RegulatoryIndexingCancellationPhase.GCS_CLEANUP,
            RegulatoryIndexingCancellationPhase.INDEX_DELETE,
        ),
        (
            RegulatoryIndexingCancellationPhase.INDEX_DELETE,
            RegulatoryIndexingCancellationPhase.FINALIZE,
        ),
    ):
        assert advance_regulatory_indexing_cancellation(
            db_session,
            job_id=job.id,
            expected_generation=generation,
            expected_phase=expected_phase,
            next_phase=next_phase,
            now=_NOW,
        )
        if next_phase is not RegulatoryIndexingCancellationPhase.FINALIZE:
            assert claim_regulatory_indexing_job(
                db_session,
                job_id=job.id,
                expected_stage=RegulatoryIndexingStage.PREPARING,
                expected_generation=generation,
                now=_NOW,
            )
            generation += 1

    assert finalize_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_generation=generation,
        now=_NOW,
    )
    db_session.refresh(job)
    db_session.refresh(regulatory_user_file)
    assert job.status == RegulatoryIndexingJobStatus.CANCELLED.value
    assert regulatory_user_file.status is UserFileStatus.DELETING


def test_cancellation_retry_deadline_is_respected_by_recovery(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW - datetime.timedelta(minutes=3),
    )
    job.remote_vertex_job_name = "projects/p/locations/l/batchJobs/1"
    db_session.commit()
    assert request_regulatory_indexing_cancellation(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        cancellation_intent=RegulatoryIndexingCancellationIntent.USER_CANCEL,
        now=_NOW - datetime.timedelta(minutes=3),
    )
    retry_at = _NOW + datetime.timedelta(minutes=5)
    assert schedule_regulatory_indexing_cancellation_retry(
        db_session,
        job_id=job.id,
        expected_generation=1,
        expected_phase=RegulatoryIndexingCancellationPhase.VERTEX_CANCEL,
        next_retry_at=retry_at,
        error_code="network",
        error_message="temporary",
    )

    assert (
        claim_stale_regulatory_indexing_jobs(
            db_session,
            stale_before=_NOW - datetime.timedelta(minutes=2),
            claimed_at=_NOW,
        )
        == []
    )
    claims = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=retry_at,
        claimed_at=retry_at,
    )
    assert len(claims) == 1
    assert claims[0].job_id == job.id


def test_required_index_cleanup_retry_survives_crash_and_blocks_successor(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="1" * 64,
    )
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.status = RegulatoryIndexingJobStatus.CANCELLING.value
    job.cancellation_intent = RegulatoryIndexingCancellationIntent.SUPERSEDE.value
    job.cancellation_phase = RegulatoryIndexingCancellationPhase.INDEX_DELETE.value
    job.attempt_count = 3
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()
    retry_at = _NOW + datetime.timedelta(minutes=5)

    assert schedule_regulatory_indexing_cancellation_retry(
        db_session,
        job_id=job.id,
        expected_generation=1,
        expected_phase=RegulatoryIndexingCancellationPhase.INDEX_DELETE,
        next_retry_at=retry_at,
        error_code="UNKNOWN",
        error_message="RuntimeError",
    )
    db_session.refresh(job)
    assert (
        job.cancellation_phase == RegulatoryIndexingCancellationPhase.INDEX_DELETE.value
    )
    assert job.attempt_count == 4
    assert job.error_code == "UNKNOWN"
    assert job.next_retry_at == retry_at

    blocked_successor = _create_job(
        db_session,
        regulatory_user_file.id,
        chunk_generation_hash="2" * 64,
    )
    assert blocked_successor.id == job.id
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == regulatory_user_file.id
            )
        )
        == 1
    )
    assert (
        claim_stale_regulatory_indexing_jobs(
            db_session,
            stale_before=_NOW,
            claimed_at=_NOW,
        )
        == []
    )

    recovered = claim_stale_regulatory_indexing_jobs(
        db_session,
        stale_before=retry_at,
        claimed_at=retry_at,
    )
    assert len(recovered) == 1
    assert recovered[0].job_id == job.id
    db_session.refresh(job)
    assert (
        job.cancellation_phase == RegulatoryIndexingCancellationPhase.INDEX_DELETE.value
    )
    assert job.lease_generation == 2
    assert not claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        now=retry_at,
    )


def test_stale_generation_never_calls_recording_index(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    job = _create_job(db_session, regulatory_user_file.id)
    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
    )
    job.stage = RegulatoryIndexingStage.INDEX_WRITE.value
    regulatory_user_file.status = UserFileStatus.INDEXING
    db_session.commit()
    calls: list[object] = []
    recording_index = cast(
        DocumentIndex,
        SimpleNamespace(index=lambda *_args, **_kwargs: calls.append(object())),
    )
    stale_job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(id=job.id, lease_generation=0),
    )

    with pytest.raises(RuntimeError, match="lease was lost"):
        stage_regulatory_job_in_index(
            job=stale_job,
            user_file=regulatory_user_file,
            rows=[],
            items=[],
            search_settings=cast(SearchSettings, SimpleNamespace()),
            tenant_id="tenant-a",
            db_session=db_session,
            document_index=recording_index,
        )

    assert calls == []
