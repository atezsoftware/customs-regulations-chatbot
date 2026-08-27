import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from onyx.db.enums import AmendmentBatchStage, AmendmentBatchStatus
from onyx.db.regulatory_amendments import (
    claim_batch_for_analysis,
    claim_stale_batches_for_recovery,
    create_batch,
    mark_batch_analyzed,
    mark_batch_failed,
    persist_proposal_checkpoint,
    persist_unmatched_checkpoint,
    reset_failed_batch_for_retry,
)

NOW = datetime.datetime(2026, 8, 27, 12, 0, tzinfo=datetime.timezone.utc)


def test_create_batch_snapshots_scope_and_starts_queued() -> None:
    db_session = MagicMock()
    first_file_id = uuid4()
    second_file_id = uuid4()

    batch = create_batch(
        db_session,
        document_set_id=7,
        user_file_ids=[first_file_id, second_file_id],
        raw_text="MADDE 1 değiştirilmiştir.",
        created_by=uuid4(),
    )

    assert batch.status == AmendmentBatchStatus.QUEUED.value
    assert batch.stage == AmendmentBatchStage.QUEUED.value
    assert batch.user_file_ids == [str(first_file_id), str(second_file_id)]
    assert batch.processed_instruction_count == 0
    assert batch.instruction_count == 0
    db_session.add.assert_called_once_with(batch)
    db_session.flush.assert_called_once_with()


def test_claim_moves_queued_batch_to_analyzing_and_increments_lease() -> None:
    batch = SimpleNamespace(
        id=11,
        status=AmendmentBatchStatus.QUEUED.value,
        stage=AmendmentBatchStage.QUEUED.value,
        lease_generation=4,
        started_at=None,
        completed_at=None,
        heartbeat_at=None,
        error_message="stale error",
        segmented_instructions=[],
    )
    db_session = MagicMock()
    db_session.scalar.return_value = batch

    lease = claim_batch_for_analysis(db_session, batch_id=11, now=NOW)

    assert lease is not None
    assert lease.batch_id == 11
    assert lease.generation == 5
    assert batch.status == AmendmentBatchStatus.ANALYZING.value
    assert batch.stage == AmendmentBatchStage.SEGMENTING.value
    assert batch.started_at == NOW
    assert batch.heartbeat_at == NOW
    assert batch.error_message is None
    db_session.commit.assert_called_once_with()


def test_retry_preserves_checkpoints_and_queues_failed_batch() -> None:
    batch = SimpleNamespace(
        id=12,
        status=AmendmentBatchStatus.FAILED.value,
        stage=AmendmentBatchStage.PROCESSING.value,
        lease_generation=8,
        error_message="provider failed",
        heartbeat_at=NOW,
        completed_at=NOW,
        segmented_instructions=[{"instruction_text": "MADDE 1"}],
        processed_instruction_count=1,
    )
    db_session = MagicMock()
    db_session.scalar.return_value = batch

    retried = reset_failed_batch_for_retry(db_session, batch_id=12)

    assert retried is batch
    assert batch.status == AmendmentBatchStatus.QUEUED.value
    assert batch.stage == AmendmentBatchStage.PROCESSING.value
    assert batch.lease_generation == 9
    assert batch.error_message is None
    assert batch.heartbeat_at is None
    assert batch.completed_at is None
    assert batch.segmented_instructions == [{"instruction_text": "MADDE 1"}]
    assert batch.processed_instruction_count == 1
    db_session.commit.assert_called_once_with()


def test_finalization_rejects_inconsistent_checkpoint_totals() -> None:
    batch = SimpleNamespace(
        id=13,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=3,
        instruction_count=2,
        processed_instruction_count=2,
        unmatched_instructions=[],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, 1]

    finalized = mark_batch_analyzed(
        db_session,
        batch_id=13,
        lease_generation=3,
        now=NOW,
    )

    assert finalized is False
    db_session.rollback.assert_called_once_with()
    db_session.commit.assert_not_called()


def test_stale_lease_cannot_advance_progress() -> None:
    batch = SimpleNamespace(
        id=14,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=9,
        processed_instruction_count=0,
        unmatched_instructions=[],
    )
    db_session = MagicMock()
    db_session.scalar.return_value = batch

    persisted = persist_unmatched_checkpoint(
        db_session,
        batch_id=14,
        lease_generation=8,
        instruction_index=0,
        instruction_text="MADDE 1",
    )

    assert persisted is False
    assert batch.processed_instruction_count == 0
    assert batch.unmatched_instructions == []
    db_session.rollback.assert_called_once_with()


def test_duplicate_delivery_does_not_create_second_proposal() -> None:
    batch = SimpleNamespace(
        id=15,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        processed_instruction_count=1,
        heartbeat_at=NOW,
    )
    existing = SimpleNamespace(id=99)
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, existing]
    proposal = SimpleNamespace(instruction_index=0)

    persisted = persist_proposal_checkpoint(
        db_session,
        batch_id=15,
        lease_generation=4,
        proposal=proposal,
    )

    assert persisted is True
    assert batch.processed_instruction_count == 1
    db_session.add.assert_not_called()
    db_session.commit.assert_called_once_with()


def test_terminal_batch_cannot_be_overwritten_as_failed() -> None:
    batch = SimpleNamespace(
        id=16,
        status=AmendmentBatchStatus.ANALYZED.value,
        lease_generation=5,
    )
    db_session = MagicMock()
    db_session.scalar.return_value = batch

    failed = mark_batch_failed(
        db_session,
        batch_id=16,
        lease_generation=5,
        error_message="late worker error",
        now=NOW,
    )

    assert failed is False
    assert batch.status == AmendmentBatchStatus.ANALYZED.value
    db_session.rollback.assert_called_once_with()


def test_recovery_throttles_queued_batches_with_a_dispatch_timestamp() -> None:
    queued = SimpleNamespace(
        id=17,
        document_set_id=7,
        user_file_ids=[str(uuid4())],
        status=AmendmentBatchStatus.QUEUED.value,
        stage=AmendmentBatchStage.QUEUED.value,
        segmented_instructions=[],
        heartbeat_at=None,
    )
    scalar_result = MagicMock()
    scalar_result.all.return_value = [queued]
    db_session = MagicMock()
    db_session.scalars.return_value = scalar_result

    recovered = claim_stale_batches_for_recovery(
        db_session,
        stale_before=NOW - datetime.timedelta(minutes=10),
        claimed_at=NOW,
    )

    assert recovered == [17]
    assert queued.heartbeat_at == NOW
    statement = str(db_session.scalars.call_args.args[0])
    assert "heartbeat_at IS NULL" in statement
    assert "heartbeat_at <=" in statement


def test_recovery_repairs_scope_for_legacy_rolling_deployment_batch() -> None:
    legacy_file_id = uuid4()
    legacy = SimpleNamespace(
        id=18,
        document_set_id=7,
        user_file_ids=[],
        status=AmendmentBatchStatus.ANALYZING.value,
        stage=AmendmentBatchStage.QUEUED.value,
        segmented_instructions=[],
        heartbeat_at=None,
        lease_generation=0,
    )
    batch_result = MagicMock()
    batch_result.all.return_value = [legacy]
    file_result = MagicMock()
    file_result.all.return_value = [legacy_file_id]
    db_session = MagicMock()
    db_session.scalars.side_effect = [batch_result, file_result]

    recovered = claim_stale_batches_for_recovery(
        db_session,
        stale_before=NOW - datetime.timedelta(minutes=10),
        claimed_at=NOW,
    )

    assert recovered == [18]
    assert legacy.user_file_ids == [str(legacy_file_id)]
    assert legacy.status == AmendmentBatchStatus.QUEUED.value
    assert legacy.lease_generation == 1
