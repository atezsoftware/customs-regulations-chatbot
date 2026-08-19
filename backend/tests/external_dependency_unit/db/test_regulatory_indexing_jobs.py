import datetime
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from onyx.db import regulatory_indexing_jobs as regulatory_indexing_job_repository
from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
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
    advance_regulatory_indexing_job,
    claim_regulatory_indexing_job,
    claim_stale_regulatory_indexing_jobs,
    complete_regulatory_indexing_user_file,
    create_or_get_regulatory_indexing_item,
    create_or_get_regulatory_indexing_job,
    persist_regulatory_indexing_item_context,
    persist_regulatory_indexing_item_failure,
    persist_regulatory_indexing_item_skipped,
    persist_regulatory_indexing_item_vector,
    persist_regulatory_indexing_item_vectors,
    record_vertex_submission,
    record_vertex_submission_absent,
    record_vertex_submission_intent,
    regulatory_indexing_external_mutation_lease,
    require_vertex_submission_reconciliation,
    schedule_regulatory_indexing_retry,
)
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.indexing_jobs.publisher import stage_regulatory_job_in_index
from tests.external_dependency_unit.conftest import create_test_user

_NOW = datetime.datetime(2026, 8, 19, 10, 0, tzinfo=datetime.timezone.utc)
_SNAPSHOT: RegulatoryIndexingConfigSnapshot = {
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
        persisted_file = db_session.get(UserFile, user_file.id)
        if persisted_file is not None:
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
) -> RegulatoryIndexingJob:
    return create_or_get_regulatory_indexing_job(
        db_session,
        user_file_id=user_file_id,
        content_hash=content_hash or uuid4().hex,
        search_settings_id=17,
        prompt_hash="prompt-v1",
        config_snapshot=_SNAPSHOT,
        now=_NOW,
    )


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
    job.stage = RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    db_session.commit()
    submission_key = "regulatory-context-" + "a" * 64

    assert not record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=0,
        submission_key=submission_key,
        now=_NOW,
    )
    assert record_vertex_submission_intent(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
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
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.vertex_submission_state
        == RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED.value
    )

    assert record_vertex_submission_absent(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
        now=_NOW,
    )
    db_session.refresh(job)
    assert (
        job.vertex_submission_state
        == RegulatoryIndexingSubmissionState.RECONCILED_ABSENT.value
    )

    assert record_vertex_submission(
        db_session,
        job_id=job.id,
        expected_generation=1,
        submission_key=submission_key,
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
    future_job = _create_job(db_session, regulatory_user_file.id)
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


def test_stale_recovery_claims_due_and_abandoned_jobs_only(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    stale_running = _create_job(db_session, regulatory_user_file.id)
    fresh_running = _create_job(db_session, regulatory_user_file.id)
    due_retry = _create_job(db_session, regulatory_user_file.id)
    future_retry = _create_job(db_session, regulatory_user_file.id)

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


def test_atomic_preparation_repairs_partial_state_and_advances_once(
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
            now=_NOW,
        )
    )

    assert persisted
    db_session.expire_all()
    recovered_job = db_session.get(RegulatoryIndexingJob, job.id)
    assert recovered_job is not None
    assert recovered_job.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
    assert recovered_job.status == RegulatoryIndexingJobStatus.QUEUED.value
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
                now=_NOW,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        takeover_future = executor.submit(takeover_prepare)
        assert newer_prepared.wait(timeout=5)
        stale_future = executor.submit(stale_prepare)
        assert stale_future.result(timeout=1) is False
        assert not stale_callback_started.is_set()
        allow_newer_commit.set()
        assert takeover_future.result(timeout=5) is True

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


def test_external_mutation_lock_blocks_user_file_deletion_status(
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

    def mark_deleting() -> None:
        with Session(engine) as delete_session:
            delete_session.execute(
                update(UserFile)
                .where(UserFile.id == regulatory_user_file.id)
                .values(status=UserFileStatus.DELETING)
            )
            delete_session.commit()

    with ThreadPoolExecutor(max_workers=2) as executor:
        held = executor.submit(hold_external_mutation)
        assert entered.wait(timeout=5)
        deletion = executor.submit(mark_deleting)
        with pytest.raises(TimeoutError):
            deletion.result(timeout=0.3)
        release.set()
        held.result(timeout=5)
        deletion.result(timeout=5)

    db_session.refresh(regulatory_user_file)
    assert regulatory_user_file.status is UserFileStatus.DELETING


def test_user_file_completion_preserves_cancelled_and_deleting_states(
    db_session: Session,
    regulatory_user_file: UserFile,
) -> None:
    for initial_status, should_complete in (
        (UserFileStatus.INDEXING, True),
        (UserFileStatus.COMPLETED, True),
        (UserFileStatus.CANCELED, False),
        (UserFileStatus.DELETING, False),
    ):
        job = _create_job(db_session, regulatory_user_file.id)
        assert claim_regulatory_indexing_job(
            db_session,
            job_id=job.id,
            expected_stage=RegulatoryIndexingStage.PREPARING,
            expected_generation=0,
            now=_NOW,
        )
        job.stage = RegulatoryIndexingStage.PUBLISH.value
        regulatory_user_file.status = initial_status
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
        db_session.refresh(regulatory_user_file)
        assert regulatory_user_file.status is (
            UserFileStatus.COMPLETED if should_complete else initial_status
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
