import datetime
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from onyx.db.enums import (
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
    RegulatoryIndexingSubmissionState,
)
from onyx.db.models import RegulatoryIndexingJob
from onyx.db.regulatory_indexing_jobs import (
    _build_regulatory_indexing_progress,
    fetch_latest_regulatory_indexing_progress_for_user_files,
)


def _job(
    *,
    status: RegulatoryIndexingJobStatus,
    stage: RegulatoryIndexingStage,
    attempt_count: int = 0,
    next_retry_at: datetime.datetime | None = None,
    vertex_state: RegulatoryIndexingSubmissionState = (
        RegulatoryIndexingSubmissionState.NONE
    ),
    openrouter_state: RegulatoryIndexingSubmissionState = (
        RegulatoryIndexingSubmissionState.NONE
    ),
) -> RegulatoryIndexingJob:
    return RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=uuid4(),
        content_hash="content",
        chunk_generation_hash="a" * 64,
        search_settings_id=1,
        prompt_hash="prompt",
        config_snapshot={},
        status=status.value,
        stage=stage.value,
        attempt_count=attempt_count,
        next_retry_at=next_retry_at,
        vertex_submission_state=vertex_state.value,
        openrouter_submission_state=openrouter_state.value,
    )


def test_context_wait_progress_counts_only_persisted_context_outcomes() -> None:
    retry_at = datetime.datetime(2026, 8, 20, 12, tzinfo=datetime.timezone.utc)
    progress = _build_regulatory_indexing_progress(
        _job(
            status=RegulatoryIndexingJobStatus.RETRY_WAIT,
            stage=RegulatoryIndexingStage.CONTEXT_WAIT,
            attempt_count=2,
            next_retry_at=retry_at,
            vertex_state=RegulatoryIndexingSubmissionState.SUBMITTED,
        ),
        total_items=10,
        context_ready_items=5,
        embedded_items=2,
        failed_items=1,
        skipped_items=1,
    )

    assert progress.completed_items == 6
    assert progress.context_ready_items == 5
    assert progress.embedded_items == 2
    assert progress.failed_items == 1
    assert progress.attempt_count == 2
    assert progress.next_retry_at == retry_at
    assert progress.provider_batch_state == "vertex:SUBMITTED"
    assert progress.error_summary == ("Dizinleme otomatik yeniden denemeyi bekliyor.")


def test_embedding_progress_and_manual_reconciliation_are_fail_closed() -> None:
    progress = _build_regulatory_indexing_progress(
        _job(
            status=RegulatoryIndexingJobStatus.FAILED,
            stage=RegulatoryIndexingStage.EMBEDDING,
            openrouter_state=(
                RegulatoryIndexingSubmissionState.MANUAL_RECONCILE_REQUIRED
            ),
        ),
        total_items=10,
        context_ready_items=9,
        embedded_items=4,
        failed_items=1,
        skipped_items=1,
    )

    assert progress.completed_items == 4
    assert progress.provider_batch_state == ("openrouter:MANUAL_RECONCILE_REQUIRED")
    assert progress.error_summary == (
        "Embedding toplu işi için operatör kontrolü gerekiyor."
    )


def test_succeeded_progress_reports_all_items_complete() -> None:
    progress = _build_regulatory_indexing_progress(
        _job(
            status=RegulatoryIndexingJobStatus.SUCCEEDED,
            stage=RegulatoryIndexingStage.PUBLISH,
        ),
        total_items=10,
        context_ready_items=10,
        embedded_items=10,
        failed_items=0,
        skipped_items=0,
    )

    assert progress.completed_items == 10
    assert progress.error_summary is None


def test_file_progress_uses_one_ranked_aggregate_query() -> None:
    job = _job(
        status=RegulatoryIndexingJobStatus.RUNNING,
        stage=RegulatoryIndexingStage.EMBEDDING,
    )
    db_session = MagicMock()
    db_session.execute.return_value.all.return_value = [(job, 10, 7, 3, 1, 1)]

    progress_by_file_id = fetch_latest_regulatory_indexing_progress_for_user_files(
        db_session,
        user_file_ids=[job.user_file_id, job.user_file_id],
    )

    assert progress_by_file_id[job.user_file_id].completed_items == 3
    db_session.execute.assert_called_once()
    statement = db_session.execute.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "row_number() OVER" in sql
    assert "regulatory_indexing_item_counts" in sql
