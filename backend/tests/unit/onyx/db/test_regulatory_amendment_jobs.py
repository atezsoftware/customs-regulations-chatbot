import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from onyx.db.enums import (
    AmendmentBatchStage,
    AmendmentBatchStatus,
    AmendmentProposalStatus,
)
from onyx.db.regulatory_amendments import (
    claim_batch_for_analysis,
    claim_stale_batches_for_recovery,
    create_batch,
    mark_batch_analyzed,
    mark_batch_failed,
    persist_proposal_checkpoint,
    persist_unmatched_checkpoint,
    recover_stale_amendment_proposal_approvals,
    reset_failed_batch_for_retry,
)
from onyx.regulatory.amendments.models import ProposalDraft

NOW = datetime.datetime(2026, 8, 27, 12, 0, tzinfo=datetime.timezone.utc)


def test_stale_approval_recovery_resets_legacy_and_resumes_applied() -> None:
    legacy = SimpleNamespace(
        id=70,
        status=AmendmentProposalStatus.APPROVING.value,
        applied_new_chunk_id=None,
        decided_by=uuid4(),
        decided_at=None,
        approval_error=None,
        updated_at=NOW - datetime.timedelta(hours=2),
    )
    applied = SimpleNamespace(
        id=71,
        status=AmendmentProposalStatus.APPROVING.value,
        applied_new_chunk_id="new-chunk",
        decided_by=uuid4(),
        decided_at=None,
        approval_error=None,
        updated_at=NOW - datetime.timedelta(hours=2),
    )
    db_session = MagicMock()
    db_session.scalars.return_value.all.return_value = [legacy, applied]

    resume_ids = recover_stale_amendment_proposal_approvals(
        db_session,
        stale_before=NOW - datetime.timedelta(minutes=10),
        recovered_at=NOW,
    )

    assert resume_ids == [71]
    assert legacy.status == AmendmentProposalStatus.PENDING.value
    assert legacy.decided_by is None
    assert applied.status == AmendmentProposalStatus.APPROVING.value
    assert legacy.updated_at == NOW
    assert applied.updated_at == NOW
    db_session.commit.assert_called_once_with()


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
    assert batch.processed_instruction_indices == []
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
        instruction_count=1,
        processed_instruction_count=1,
        heartbeat_at=NOW,
    )
    existing = SimpleNamespace(id=99, instruction_index=0)
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, existing]
    proposal = cast(ProposalDraft, SimpleNamespace(instruction_index=0))

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


def test_grouped_proposal_checkpoint_tracks_exact_non_contiguous_indices() -> None:
    batch = SimpleNamespace(
        id=19,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=5,
        processed_instruction_count=1,
        processed_instruction_indices=[0],
        heartbeat_at=NOW,
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, None]
    proposal = ProposalDraft(
        instruction_index=2,
        instruction_text="MADDE 3",
        instruction_indices=[2, 4],
        instruction_texts=["MADDE 3", "MADDE 5"],
        old_chunk_id="old-chunk",
        old_chunk_snapshot={},
        new_chunk_draft={},
    )

    persisted = persist_proposal_checkpoint(
        db_session,
        batch_id=19,
        lease_generation=4,
        proposal=proposal,
    )

    assert persisted is True
    assert batch.processed_instruction_indices == [0, 2, 4]
    assert batch.processed_instruction_count == 3
    inserted = db_session.add.call_args.args[0]
    assert inserted.instruction_indices == [2, 4]
    assert inserted.instruction_texts == ["MADDE 3", "MADDE 5"]


def test_grouped_proposal_replay_requires_the_exact_existing_group() -> None:
    batch = SimpleNamespace(
        id=20,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=5,
        processed_instruction_count=3,
        processed_instruction_indices=[0, 2, 3],
        heartbeat_at=NOW,
    )
    existing = SimpleNamespace(
        instruction_index=2,
        instruction_indices=[2, 3],
        instruction_texts=["MADDE 3", "MADDE 4"],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, existing]
    proposal = ProposalDraft(
        instruction_index=2,
        instruction_text="MADDE 3",
        instruction_indices=[2, 4],
        instruction_texts=["MADDE 3", "MADDE 5"],
        old_chunk_id="old-chunk",
        old_chunk_snapshot={},
        new_chunk_draft={},
    )

    persisted = persist_proposal_checkpoint(
        db_session,
        batch_id=20,
        lease_generation=4,
        proposal=proposal,
    )

    assert persisted is False
    assert batch.processed_instruction_indices == [0, 2, 3]
    db_session.add.assert_not_called()
    db_session.rollback.assert_called_once_with()


def test_legacy_singleton_proposal_replay_uses_scalar_checkpoint_fields() -> None:
    batch = SimpleNamespace(
        id=26,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=2,
        processed_instruction_count=1,
        processed_instruction_indices=[],
        heartbeat_at=NOW,
    )
    existing = SimpleNamespace(
        instruction_index=0,
        instruction_indices=[],
        instruction_text="MADDE 1",
        instruction_texts=[],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, existing]
    proposal = ProposalDraft(
        instruction_index=0,
        instruction_text="MADDE 1",
        instruction_indices=[0],
        instruction_texts=["MADDE 1"],
        old_chunk_id="old-chunk",
        old_chunk_snapshot={"id": "old-chunk"},
        new_chunk_draft={},
    )

    persisted = persist_proposal_checkpoint(
        db_session,
        batch_id=26,
        lease_generation=4,
        proposal=proposal,
    )

    assert persisted is True
    assert batch.processed_instruction_indices == []
    assert batch.processed_instruction_count == 1
    db_session.add.assert_not_called()
    db_session.commit.assert_called_once_with()


def test_unmatched_checkpoint_records_non_contiguous_exact_coverage_once() -> None:
    batch = SimpleNamespace(
        id=21,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=5,
        processed_instruction_count=1,
        processed_instruction_indices=[0],
        unmatched_instructions=[],
        heartbeat_at=NOW,
    )
    db_session = MagicMock()
    db_session.scalar.return_value = batch

    persisted = persist_unmatched_checkpoint(
        db_session,
        batch_id=21,
        lease_generation=4,
        instruction_index=3,
        instruction_text="MADDE 4",
    )

    assert persisted is True
    assert batch.processed_instruction_indices == [0, 3]
    assert batch.processed_instruction_count == 2
    assert batch.unmatched_instructions == ["MADDE 4"]


def test_unmatched_checkpoint_replays_only_an_unmatched_covered_index() -> None:
    batch = SimpleNamespace(
        id=24,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=5,
        processed_instruction_count=2,
        processed_instruction_indices=[0, 3],
        unmatched_instructions=["MADDE 4"],
        heartbeat_at=NOW,
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, None]

    persisted = persist_unmatched_checkpoint(
        db_session,
        batch_id=24,
        lease_generation=4,
        instruction_index=3,
        instruction_text="MADDE 4",
    )

    assert persisted is True
    assert batch.unmatched_instructions == ["MADDE 4"]
    db_session.commit.assert_called_once_with()


def test_unmatched_checkpoint_rejects_an_index_owned_by_a_proposal() -> None:
    batch = SimpleNamespace(
        id=25,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=5,
        processed_instruction_count=2,
        processed_instruction_indices=[0, 3],
        unmatched_instructions=[],
        heartbeat_at=NOW,
    )
    proposal = SimpleNamespace(id=99, instruction_indices=[3])
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, proposal]

    persisted = persist_unmatched_checkpoint(
        db_session,
        batch_id=25,
        lease_generation=4,
        instruction_index=3,
        instruction_text="MADDE 4",
    )

    assert persisted is False
    assert batch.unmatched_instructions == []
    db_session.rollback.assert_called_once_with()
    proposal_lookup = str(db_session.scalar.call_args_list[1].args[0])
    assert "instruction_indices" in proposal_lookup


def test_legacy_singleton_proposal_owns_its_scalar_index_during_unmatched_replay() -> (
    None
):
    batch = SimpleNamespace(
        id=27,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=2,
        processed_instruction_count=1,
        processed_instruction_indices=[],
        unmatched_instructions=[],
        heartbeat_at=NOW,
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, 99]

    persisted = persist_unmatched_checkpoint(
        db_session,
        batch_id=27,
        lease_generation=4,
        instruction_index=0,
        instruction_text="MADDE 1",
    )

    assert persisted is False
    assert batch.unmatched_instructions == []
    proposal_lookup = str(db_session.scalar.call_args_list[1].args[0])
    assert "instruction_index" in proposal_lookup
    assert "jsonb_array_length" in proposal_lookup
    db_session.rollback.assert_called_once_with()


def test_finalization_requires_exact_index_coverage_and_group_member_total() -> None:
    batch = SimpleNamespace(
        id=22,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=3,
        processed_instruction_count=3,
        processed_instruction_indices=[0, 2, 3],
        unmatched_instructions=[],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, 3]

    finalized = mark_batch_analyzed(
        db_session,
        batch_id=22,
        lease_generation=4,
        now=NOW,
    )

    assert finalized is False
    db_session.rollback.assert_called_once_with()
    statement = str(db_session.scalar.call_args_list[0].args[0])
    assert "amendment_batch" in statement


def test_finalization_sums_group_members_not_proposal_rows() -> None:
    batch = SimpleNamespace(
        id=23,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=3,
        processed_instruction_count=3,
        processed_instruction_indices=[0, 1, 2],
        unmatched_instructions=["MADDE 3"],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, 2]

    finalized = mark_batch_analyzed(
        db_session,
        batch_id=23,
        lease_generation=4,
        now=NOW,
    )

    assert finalized is True
    proposal_coverage_query = str(db_session.scalar.call_args_list[1].args[0])
    assert "jsonb_array_length" in proposal_coverage_query


def test_finalization_uses_legacy_scalar_coverage_for_empty_group_arrays() -> None:
    batch = SimpleNamespace(
        id=28,
        status=AmendmentBatchStatus.ANALYZING.value,
        lease_generation=4,
        instruction_count=2,
        processed_instruction_count=2,
        processed_instruction_indices=[],
        unmatched_instructions=["MADDE 2"],
    )
    db_session = MagicMock()
    db_session.scalar.side_effect = [batch, 1]

    finalized = mark_batch_analyzed(
        db_session,
        batch_id=28,
        lease_generation=4,
        now=NOW,
    )

    assert finalized is True
    proposal_coverage_query = str(db_session.scalar.call_args_list[1].args[0])
    assert "CASE" in proposal_coverage_query
    assert "jsonb_array_length" in proposal_coverage_query
    db_session.commit.assert_called_once_with()


def test_proposal_draft_rejects_unsorted_or_duplicate_group_indices() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ProposalDraft(
            instruction_index=1,
            instruction_text="MADDE 2",
            instruction_indices=[1, 1],
            instruction_texts=["MADDE 2", "MADDE 2 tekrar"],
            old_chunk_id=None,
            old_chunk_snapshot={},
            new_chunk_draft={},
        )


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
