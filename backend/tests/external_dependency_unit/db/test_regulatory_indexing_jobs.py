import datetime
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    User,
    UserFile,
)
from onyx.db.regulatory_indexing_jobs import (
    advance_regulatory_indexing_job,
    claim_regulatory_indexing_job,
    claim_stale_regulatory_indexing_jobs,
    create_or_get_regulatory_indexing_item,
    create_or_get_regulatory_indexing_job,
    persist_regulatory_indexing_item_context,
    persist_regulatory_indexing_item_failure,
    persist_regulatory_indexing_item_skipped,
    persist_regulatory_indexing_item_vector,
    schedule_regulatory_indexing_retry,
)
from tests.external_dependency_unit.conftest import create_test_user

_NOW = datetime.datetime(2026, 8, 19, 10, 0, tzinfo=datetime.timezone.utc)
_SNAPSHOT: dict[str, object] = {
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


def test_current_lease_can_persist_terminal_job_state(
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

    assert advance_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=1,
        next_stage=RegulatoryIndexingStage.PUBLISH,
        next_status=RegulatoryIndexingJobStatus.SUCCEEDED,
        now=_NOW,
    )

    db_session.refresh(job)
    assert job.stage == RegulatoryIndexingStage.PUBLISH.value
    assert job.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
    assert job.completed_at == _NOW


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
    item = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="request-v1",
    )
    duplicate = create_or_get_regulatory_indexing_item(
        db_session,
        job_id=job.id,
        regulatory_chunk_id=chunk.id,
        request_hash="ignored-duplicate-hash",
    )
    assert duplicate.id == item.id
    assert (
        db_session.scalar(
            select(func.count(RegulatoryIndexingItem.id)).where(
                RegulatoryIndexingItem.job_id == job.id
            )
        )
        == 1
    )

    assert claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=0,
        now=_NOW,
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
    )
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
    )
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
