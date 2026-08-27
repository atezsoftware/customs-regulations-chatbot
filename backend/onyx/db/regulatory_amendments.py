"""DB operations for the amendment (update) mechanism.

`approve_amendment_proposal` is the one place that ever writes an
amendment-sourced row into `regulatory_chunk` — everything upstream
(segmenting, candidate search, drafting) is pure analysis that only produces
`AmendmentProposal` rows for review.
"""

import datetime
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from onyx.db.enums import (
    AmendmentBatchStage,
    AmendmentBatchStatus,
    RegulatoryChunkSource,
    RegulatoryChunkStatus,
)
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet__UserFile,
    RegulatoryChunk,
)
from onyx.db.regulatory_chunks import (
    delete_hierarchical_aggregates_referencing_chunk,
    is_hierarchical_aggregate_chunk,
    make_regulatory_chunk_id,
)
from onyx.regulatory.amendments.models import ProposalDraft
from onyx.regulatory.chunker import ATOMIC_CHUNK_VARIANT

_MAX_ERROR_MESSAGE_LENGTH = 4000


@dataclass(frozen=True)
class AmendmentBatchLease:
    batch_id: int
    generation: int


def create_batch(
    db_session: Session,
    *,
    document_set_id: int,
    user_file_ids: list[UUID],
    raw_text: str,
    created_by: UUID | None,
) -> AmendmentBatch:
    batch = AmendmentBatch(
        document_set_id=document_set_id,
        user_file_ids=[str(user_file_id) for user_file_id in user_file_ids],
        raw_text=raw_text,
        created_by=created_by,
        status=AmendmentBatchStatus.QUEUED.value,
        stage=AmendmentBatchStage.QUEUED.value,
        instruction_count=0,
        processed_instruction_count=0,
        lease_generation=0,
        segmented_instructions=[],
        unmatched_instructions=[],
    )
    db_session.add(batch)
    db_session.flush()
    return batch


def get_batch(db_session: Session, batch_id: int) -> AmendmentBatch | None:
    return db_session.get(AmendmentBatch, batch_id)


def _get_batch_for_update(db_session: Session, batch_id: int) -> AmendmentBatch | None:
    return db_session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )


def claim_batch_for_analysis(
    db_session: Session,
    *,
    batch_id: int,
    now: datetime.datetime | None = None,
) -> AmendmentBatchLease | None:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    batch = _get_batch_for_update(db_session, batch_id)
    if batch is None or batch.status != AmendmentBatchStatus.QUEUED.value:
        db_session.rollback()
        return None

    batch.lease_generation += 1
    batch.status = AmendmentBatchStatus.ANALYZING.value
    batch.stage = (
        AmendmentBatchStage.PROCESSING.value
        if batch.segmented_instructions
        else AmendmentBatchStage.SEGMENTING.value
    )
    batch.started_at = batch.started_at or now
    batch.heartbeat_at = now
    batch.completed_at = None
    batch.error_message = None
    generation = batch.lease_generation
    db_session.commit()
    return AmendmentBatchLease(batch_id=batch.id, generation=generation)


def persist_segmentation_checkpoint(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    reference_date: str | None,
    instructions: list[dict[str, object]],
    now: datetime.datetime | None = None,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    if not batch.segmented_instructions:
        batch.segmented_instructions = instructions
        batch.instruction_count = len(instructions)
        batch.reference_date = (
            datetime.date.fromisoformat(reference_date) if reference_date else None
        )
    batch.stage = AmendmentBatchStage.PROCESSING.value
    batch.heartbeat_at = now or datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    return True


def touch_batch_heartbeat(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    now: datetime.datetime | None = None,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    batch.heartbeat_at = now or datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    return True


def persist_proposal_checkpoint(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    proposal: ProposalDraft,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    existing = db_session.scalar(
        select(AmendmentProposal).where(
            AmendmentProposal.batch_id == batch_id,
            AmendmentProposal.instruction_index == proposal.instruction_index,
        )
    )
    if existing is None:
        if proposal.instruction_index != batch.processed_instruction_count:
            db_session.rollback()
            return False
        db_session.add(
            AmendmentProposal(
                batch_id=batch_id,
                instruction_index=proposal.instruction_index,
                instruction_text=proposal.instruction_text,
                old_chunk_id=proposal.old_chunk_id,
                old_chunk_snapshot=proposal.old_chunk_snapshot,
                new_chunk_draft=proposal.new_chunk_draft,
                match_confidence=proposal.match_confidence,
                match_rationale=proposal.match_rationale,
                date_rationale=proposal.date_rationale,
            )
        )
        batch.processed_instruction_count += 1
    batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    return True


def persist_unmatched_checkpoint(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    instruction_index: int,
    instruction_text: str,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    if instruction_index < batch.processed_instruction_count:
        db_session.rollback()
        return True
    if instruction_index != batch.processed_instruction_count:
        db_session.rollback()
        return False
    batch.unmatched_instructions = [
        *batch.unmatched_instructions,
        instruction_text,
    ]
    batch.processed_instruction_count += 1
    batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    return True


def mark_batch_analyzed(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    now: datetime.datetime | None = None,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
        or batch.processed_instruction_count != batch.instruction_count
    ):
        db_session.rollback()
        return False
    proposal_count = db_session.scalar(
        select(func.count(AmendmentProposal.id)).where(
            AmendmentProposal.batch_id == batch_id
        )
    )
    if (proposal_count or 0) + len(
        batch.unmatched_instructions
    ) != batch.instruction_count:
        db_session.rollback()
        return False
    completed_at = now or datetime.datetime.now(datetime.timezone.utc)
    batch.stage = AmendmentBatchStage.FINALIZING.value
    batch.status = AmendmentBatchStatus.ANALYZED.value
    batch.heartbeat_at = completed_at
    batch.completed_at = completed_at
    db_session.commit()
    return True


def mark_batch_failed(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    error_message: str,
    now: datetime.datetime | None = None,
) -> bool:
    batch = _get_batch_for_update(db_session, batch_id)
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    completed_at = now or datetime.datetime.now(datetime.timezone.utc)
    batch.status = AmendmentBatchStatus.FAILED.value
    batch.error_message = error_message[:_MAX_ERROR_MESSAGE_LENGTH]
    batch.heartbeat_at = completed_at
    batch.completed_at = completed_at
    db_session.commit()
    return True


def reset_failed_batch_for_retry(
    db_session: Session, *, batch_id: int
) -> AmendmentBatch | None:
    batch = _get_batch_for_update(db_session, batch_id)
    if batch is None or batch.status != AmendmentBatchStatus.FAILED.value:
        db_session.rollback()
        return None
    batch.status = AmendmentBatchStatus.QUEUED.value
    batch.stage = (
        AmendmentBatchStage.PROCESSING.value
        if batch.segmented_instructions
        else AmendmentBatchStage.QUEUED.value
    )
    batch.lease_generation += 1
    batch.error_message = None
    batch.heartbeat_at = None
    batch.completed_at = None
    db_session.commit()
    return batch


def claim_stale_batches_for_recovery(
    db_session: Session,
    *,
    stale_before: datetime.datetime,
    claimed_at: datetime.datetime,
    limit: int = 100,
) -> list[int]:
    batches = list(
        db_session.scalars(
            select(AmendmentBatch)
            .where(
                or_(
                    (
                        (AmendmentBatch.status == AmendmentBatchStatus.QUEUED.value)
                        & (
                            AmendmentBatch.heartbeat_at.is_(None)
                            | (AmendmentBatch.heartbeat_at <= stale_before)
                        )
                    ),
                    (
                        (AmendmentBatch.status == AmendmentBatchStatus.ANALYZING.value)
                        & (
                            (AmendmentBatch.heartbeat_at <= stale_before)
                            | (
                                AmendmentBatch.heartbeat_at.is_(None)
                                & (AmendmentBatch.created_at <= stale_before)
                            )
                        )
                    ),
                )
            )
            .order_by(AmendmentBatch.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
    )
    recovered_batch_ids: list[int] = []
    for batch in batches:
        if not batch.user_file_ids:
            legacy_file_ids = list(
                db_session.scalars(
                    select(DocumentSet__UserFile.user_file_id).where(
                        DocumentSet__UserFile.document_set_id == batch.document_set_id
                    )
                ).all()
            )
            if not legacy_file_ids:
                batch.status = AmendmentBatchStatus.FAILED.value
                batch.error_message = (
                    "Analysis cannot resume because the document set has no files."
                )
                batch.heartbeat_at = claimed_at
                batch.completed_at = claimed_at
                continue
            batch.user_file_ids = [str(file_id) for file_id in legacy_file_ids]
        if batch.status == AmendmentBatchStatus.ANALYZING.value:
            batch.lease_generation += 1
        batch.status = AmendmentBatchStatus.QUEUED.value
        batch.stage = (
            AmendmentBatchStage.PROCESSING.value
            if batch.segmented_instructions
            else AmendmentBatchStage.QUEUED.value
        )
        batch.heartbeat_at = claimed_at
        recovered_batch_ids.append(batch.id)
    db_session.commit()
    return recovered_batch_ids


def list_batches_for_document_set(
    db_session: Session, document_set_id: int
) -> list[AmendmentBatch]:
    stmt = (
        select(AmendmentBatch)
        .where(AmendmentBatch.document_set_id == document_set_id)
        .order_by(AmendmentBatch.created_at.desc())
    )
    return list(db_session.scalars(stmt).all())


def get_proposal(db_session: Session, proposal_id: int) -> AmendmentProposal | None:
    return db_session.get(AmendmentProposal, proposal_id)


def list_proposals_for_batch(
    db_session: Session, batch_id: int
) -> list[AmendmentProposal]:
    stmt = (
        select(AmendmentProposal)
        .where(AmendmentProposal.batch_id == batch_id)
        .order_by(AmendmentProposal.instruction_index)
    )
    return list(db_session.scalars(stmt).all())


def compute_duplicate_targets(
    proposals: list[AmendmentProposal],
) -> dict[int, bool]:
    """Flag proposals from the same batch that target the same old chunk —
    approving both would race on the approval transaction's
    `status == 'active'` guard, so it's better to warn the admin up front
    than let the second approval fail later."""
    counts: dict[str, int] = {}
    for proposal in proposals:
        if proposal.old_chunk_id and proposal.status == "pending":
            counts[proposal.old_chunk_id] = counts.get(proposal.old_chunk_id, 0) + 1
    return {
        proposal.id: bool(
            proposal.old_chunk_id and counts.get(proposal.old_chunk_id, 0) > 1
        )
        for proposal in proposals
    }


def reject_proposal(
    proposal: AmendmentProposal, *, decided_by: UUID | None
) -> AmendmentProposal:
    if proposal.status != "pending":
        raise ValueError(
            f"Amendment proposal {proposal.id} is already {proposal.status}."
        )
    proposal.status = "rejected"
    proposal.decided_by = decided_by
    proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
    return proposal


class ApprovalResult:
    def __init__(
        self,
        *,
        proposal: AmendmentProposal,
        new_chunk: RegulatoryChunk,
        old_chunk: RegulatoryChunk | None,
    ) -> None:
        self.proposal = proposal
        self.new_chunk = new_chunk
        self.old_chunk = old_chunk


def approve_amendment_proposal(
    db_session: Session,
    proposal: AmendmentProposal,
    *,
    decided_by: UUID | None,
) -> ApprovalResult:
    """Apply one pending proposal:

    1. Verify the proposal is still 'pending'.
    2. Insert the new chunk (source='amendment').
    3. If old_chunk_id is set, mark it superseded — guarded by
       `status == 'active'` so two proposals racing to supersede the same
       chunk can't both succeed silently. The new chunk's
       `validity_start_date` doubles as the old chunk's supersession
       boundary (when the new text starts applying is when the old text
       stops).
    4. Mark the proposal approved and record which chunk it produced.

    Date fallback: if the drafted `effective_start_date` is null (no
    explicit date in the amendment text, and no reference/publication date
    to resolve "yayım tarihinden itibaren" against), both the new chunk's
    start and the old chunk's end default to today — the day the admin
    approves it.

    Raises ValueError on any conflict (already decided, old chunk already
    superseded) so callers can surface a clear per-proposal failure reason.

    Does not re-project the affected file into Elasticsearch — that is a
    service-layer concern (see onyx.regulatory.projection); the caller does
    it after a successful approval, same as chunk edits from the Files
    panel.
    """
    if proposal.status != "pending":
        raise ValueError(
            f"Amendment proposal {proposal.id} is already {proposal.status}."
        )

    draft = proposal.new_chunk_draft
    user_file_id = UUID(draft["user_file_id"])
    today = datetime.date.today()
    start_date_str = draft.get("effective_start_date")
    end_date_str = draft.get("effective_end_date")
    start_date = (
        datetime.date.fromisoformat(start_date_str) if start_date_str else today
    )
    end_date = datetime.date.fromisoformat(end_date_str) if end_date_str else None

    old_chunk: RegulatoryChunk | None = None
    if proposal.old_chunk_id:
        old_chunk = db_session.get(RegulatoryChunk, proposal.old_chunk_id)
        if old_chunk is not None and old_chunk.status != "active":
            raise ValueError(
                f"Old chunk {proposal.old_chunk_id} is already "
                f"{old_chunk.status}; cannot approve."
            )
        if old_chunk is not None and is_hierarchical_aggregate_chunk(old_chunk):
            raise ValueError("Derived aggregate chunks cannot be amended directly.")

    new_chunk_metadata = dict(draft.get("metadata") or {})
    new_chunk_metadata.setdefault("chunk_variant", ATOMIC_CHUNK_VARIANT)
    new_chunk_metadata.setdefault("source_chunk_orders", [])
    new_chunk_metadata.setdefault("source_regulatory_chunk_ids", [])

    new_chunk_id = make_regulatory_chunk_id(
        user_file_id, draft["position"], draft["text"]
    )
    new_chunk = RegulatoryChunk(
        id=new_chunk_id,
        user_file_id=user_file_id,
        text=draft["text"],
        position=draft["position"],
        chunk_type=draft.get("chunk_type"),
        heading_path=draft.get("heading_path") or [],
        chunk_metadata=new_chunk_metadata,
        status=RegulatoryChunkStatus.ACTIVE.value,
        source=RegulatoryChunkSource.AMENDMENT.value,
        validity_start_date=start_date,
        validity_end_date=end_date,
        supersedes_chunk_id=old_chunk.id if old_chunk else None,
    )
    db_session.add(new_chunk)
    db_session.flush()

    if old_chunk is not None:
        delete_hierarchical_aggregates_referencing_chunk(
            db_session,
            user_file_id=user_file_id,
            source_chunk_id=old_chunk.id,
        )
        old_chunk.status = RegulatoryChunkStatus.SUPERSEDED.value
        old_chunk.validity_end_date = start_date
        old_chunk.superseded_by_chunk_id = new_chunk.id
        db_session.add(old_chunk)

    proposal.status = "approved"
    proposal.applied_new_chunk_id = new_chunk.id
    proposal.decided_by = decided_by
    proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
    db_session.add(proposal)
    db_session.flush()

    return ApprovalResult(proposal=proposal, new_chunk=new_chunk, old_chunk=old_chunk)
