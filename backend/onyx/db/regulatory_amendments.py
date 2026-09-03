"""DB operations for the amendment (update) mechanism.

`approve_amendment_proposal` is the one place that ever writes an
amendment-sourced row into `regulatory_chunk` — everything upstream
(segmenting, candidate search, drafting) is pure analysis that only produces
`AmendmentProposal` rows for review.
"""

import datetime
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.orm import Session

from onyx.db.enums import (
    AmendmentBatchStage,
    AmendmentBatchStatus,
    AmendmentProposalStatus,
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
    has_active_structural_descendants,
    is_hierarchical_aggregate_chunk,
    make_regulatory_chunk_id,
    supersede_hierarchical_aggregates_referencing_chunk,
)
from onyx.regulatory.amendments.draft_integrity import (
    explicit_replacement_body,
    reconcile_existing_heading_path,
    reject_unsupported_descendant_replacement_texts,
    validate_explicit_replacement_texts,
)
from onyx.regulatory.amendments.models import ProposalDraft, ReviewedAmendmentChunkDraft
from onyx.regulatory.chunker import ATOMIC_CHUNK_VARIANT

_MAX_ERROR_MESSAGE_LENGTH = 4000
_AMENDMENT_PROJECTION_ORDINAL_BASE = 1_000_000_000
_MAX_AMENDMENT_PROPOSAL_ID = 999_999_999


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
        processed_instruction_indices=[],
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


def _processed_instruction_indices(batch: AmendmentBatch) -> set[int]:
    indices = getattr(batch, "processed_instruction_indices", None)
    if not indices:
        return set(range(batch.processed_instruction_count))
    return set(indices)


def _proposal_instruction_indices(
    proposal: AmendmentProposal | ProposalDraft,
) -> list[int]:
    indices = getattr(proposal, "instruction_indices", None)
    if not indices:
        return [proposal.instruction_index]
    return list(indices)


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
    instruction_indices = _proposal_instruction_indices(proposal)
    covered_indices = _processed_instruction_indices(batch)
    if not instruction_indices or any(
        index < 0 or index >= batch.instruction_count for index in instruction_indices
    ):
        db_session.rollback()
        return False
    existing = db_session.scalar(
        select(AmendmentProposal).where(
            AmendmentProposal.batch_id == batch_id,
            AmendmentProposal.instruction_index == proposal.instruction_index,
        )
    )
    if existing is not None:
        if _proposal_instruction_indices(existing) != instruction_indices or not set(
            instruction_indices
        ).issubset(covered_indices):
            db_session.rollback()
            return False
        batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
        db_session.commit()
        return True
    if covered_indices.intersection(instruction_indices):
        db_session.rollback()
        return False
    instruction_texts = list(getattr(proposal, "instruction_texts", None) or [])
    if not instruction_texts:
        instruction_texts = [proposal.instruction_text]
    db_session.add(
        AmendmentProposal(
            batch_id=batch_id,
            instruction_index=proposal.instruction_index,
            instruction_text=proposal.instruction_text,
            instruction_indices=instruction_indices,
            instruction_texts=instruction_texts,
            old_chunk_id=proposal.old_chunk_id,
            old_chunk_snapshot=proposal.old_chunk_snapshot,
            new_chunk_draft=proposal.new_chunk_draft,
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
        )
    )
    covered_indices.update(instruction_indices)
    batch.processed_instruction_indices = sorted(covered_indices)
    batch.processed_instruction_count = len(covered_indices)
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
    if instruction_index < 0 or instruction_index >= batch.instruction_count:
        db_session.rollback()
        return False
    covered_indices = _processed_instruction_indices(batch)
    if instruction_index in covered_indices:
        proposal_owns_index = db_session.scalar(
            select(AmendmentProposal.id).where(
                AmendmentProposal.batch_id == batch_id,
                or_(
                    AmendmentProposal.instruction_indices.contains([instruction_index]),
                    and_(
                        func.jsonb_array_length(AmendmentProposal.instruction_indices)
                        == 0,
                        AmendmentProposal.instruction_index == instruction_index,
                    ),
                ),
            )
        )
        if proposal_owns_index is not None:
            db_session.rollback()
            return False
        batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
        db_session.commit()
        return True
    batch.unmatched_instructions = [
        *batch.unmatched_instructions,
        instruction_text,
    ]
    covered_indices.add(instruction_index)
    batch.processed_instruction_indices = sorted(covered_indices)
    batch.processed_instruction_count = len(covered_indices)
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
    ):
        db_session.rollback()
        return False
    covered_indices = _processed_instruction_indices(batch)
    expected_indices = set(range(batch.instruction_count))
    if covered_indices != expected_indices or batch.processed_instruction_count != len(
        covered_indices
    ):
        db_session.rollback()
        return False
    grouped_instruction_count = func.jsonb_array_length(
        AmendmentProposal.instruction_indices
    )
    proposal_coverage = db_session.scalar(
        select(
            func.coalesce(
                func.sum(
                    case(
                        (grouped_instruction_count > 0, grouped_instruction_count),
                        else_=1,
                    )
                ),
                0,
            )
        ).where(AmendmentProposal.batch_id == batch_id)
    )
    if (proposal_coverage or 0) + len(
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


def _lock_proposal_for_transition(
    db_session: Session, proposal_id: int
) -> AmendmentProposal:
    proposal = db_session.scalar(
        select(AmendmentProposal)
        .where(AmendmentProposal.id == proposal_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if proposal is None:
        raise ValueError(f"Amendment proposal {proposal_id} no longer exists.")
    return proposal


def reject_proposal(
    db_session: Session,
    proposal: AmendmentProposal,
    *,
    decided_by: UUID | None,
) -> AmendmentProposal:
    proposal = _lock_proposal_for_transition(db_session, proposal.id)
    if proposal.status != AmendmentProposalStatus.PENDING.value:
        raise ValueError(
            f"Amendment proposal {proposal.id} is already {proposal.status}."
        )
    proposal.status = AmendmentProposalStatus.REJECTED.value
    proposal.decided_by = decided_by
    proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
    return proposal


def queue_amendment_proposal_approval(
    db_session: Session,
    proposal: AmendmentProposal,
    *,
    decided_by: UUID | None,
    reviewed_new_chunk_draft: dict[str, Any] | None = None,
) -> AmendmentProposal:
    """Validate and durably claim a proposal before dispatching its projection."""

    proposal = _lock_proposal_for_transition(db_session, proposal.id)
    if proposal.status != AmendmentProposalStatus.PENDING.value:
        raise ValueError(
            f"Amendment proposal {proposal.id} is already {proposal.status}."
        )
    effective_draft = reviewed_new_chunk_draft or proposal.new_chunk_draft
    reviewed_draft = _validated_reviewed_chunk_draft(
        proposal.new_chunk_draft,
        effective_draft,
        old_chunk_snapshot=getattr(proposal, "old_chunk_snapshot", None) or {},
    )
    validate_explicit_replacement_texts(
        _proposal_instruction_texts(proposal),
        reviewed_draft["text"],
    )
    proposal.new_chunk_draft = reviewed_draft
    proposal.status = AmendmentProposalStatus.APPROVING.value
    proposal.decided_by = decided_by
    proposal.decided_at = None
    db_session.add(proposal)
    db_session.flush()
    return proposal


def reset_amendment_proposal_approval(
    db_session: Session,
    *,
    proposal_id: int,
) -> bool:
    """Return a failed queued approval to review without undoing later decisions."""

    proposal = _lock_proposal_for_transition(db_session, proposal_id)
    if proposal.status != AmendmentProposalStatus.APPROVING.value:
        db_session.rollback()
        return False
    proposal.status = AmendmentProposalStatus.PENDING.value
    proposal.decided_by = None
    proposal.decided_at = None
    db_session.commit()
    return True


def retry_amendment_proposal_projection(
    db_session: Session,
    *,
    proposal_id: int,
) -> tuple[AmendmentProposal, bool]:
    """Retry search projection without creating another legal chunk version."""

    proposal = _lock_proposal_for_transition(db_session, proposal_id)
    if (
        proposal.status == AmendmentProposalStatus.APPROVING.value
        and proposal.applied_new_chunk_id
        and proposal.approval_indexing_job_id is None
    ):
        return proposal, False
    if proposal.status != AmendmentProposalStatus.APPROVAL_FAILED.value:
        raise ValueError(
            f"Amendment proposal {proposal.id} is not awaiting an indexing retry."
        )
    if not proposal.applied_new_chunk_id:
        raise ValueError(
            f"Amendment proposal {proposal.id} has no applied chunk to reindex."
        )
    proposal.status = AmendmentProposalStatus.APPROVING.value
    proposal.approval_error = None
    proposal.approval_indexing_job_id = None
    proposal.decided_at = None
    proposal.updated_at = datetime.datetime.now(datetime.timezone.utc)
    db_session.add(proposal)
    db_session.flush()
    return proposal, True


def link_amendment_proposal_indexing_job(
    db_session: Session,
    *,
    proposal_id: int,
    job_id: UUID,
) -> bool:
    """Link an applied proposal to the durable projection that will publish it."""

    result = db_session.execute(
        update(AmendmentProposal)
        .where(
            AmendmentProposal.id == proposal_id,
            AmendmentProposal.status == AmendmentProposalStatus.APPROVING.value,
            AmendmentProposal.applied_new_chunk_id.is_not(None),
        )
        .values(
            approval_indexing_job_id=job_id,
            approval_error=None,
            updated_at=func.now(),
        )
    )
    return bool(result.rowcount)  # ty: ignore[unresolved-attribute]


def finalize_amendment_proposals_for_indexing_job(
    db_session: Session,
    *,
    job_id: UUID,
    succeeded: bool,
    error_message: str | None = None,
) -> int:
    """Reflect one terminal projection outcome on every linked proposal."""

    values: dict[str, Any]
    if succeeded:
        values = {
            "status": AmendmentProposalStatus.APPROVED.value,
            "approval_error": None,
            "decided_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": func.now(),
        }
    else:
        values = {
            "status": AmendmentProposalStatus.APPROVAL_FAILED.value,
            "approval_error": (
                error_message or "Indexing failed. The approval was not published."
            )[:_MAX_ERROR_MESSAGE_LENGTH],
            "decided_at": None,
            "updated_at": func.now(),
        }
    result = db_session.execute(
        update(AmendmentProposal)
        .where(
            AmendmentProposal.approval_indexing_job_id == job_id,
            AmendmentProposal.status == AmendmentProposalStatus.APPROVING.value,
        )
        .values(**values)
    )
    return int(result.rowcount or 0)  # ty: ignore[unresolved-attribute]


def finalize_amendment_proposal_projection(
    db_session: Session,
    *,
    proposal_id: int,
    succeeded: bool,
    error_message: str | None = None,
) -> bool:
    """Persist the terminal state of an active-index amendment projection."""

    values: dict[str, Any]
    if succeeded:
        values = {
            "status": AmendmentProposalStatus.APPROVED.value,
            "approval_error": None,
            "decided_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": func.now(),
        }
    else:
        values = {
            "status": AmendmentProposalStatus.APPROVAL_FAILED.value,
            "approval_error": (
                error_message or "Indexing failed. The approval was not published."
            )[:_MAX_ERROR_MESSAGE_LENGTH],
            "decided_at": None,
            "updated_at": func.now(),
        }
    result = db_session.execute(
        update(AmendmentProposal)
        .where(
            AmendmentProposal.id == proposal_id,
            AmendmentProposal.status == AmendmentProposalStatus.APPROVING.value,
            AmendmentProposal.applied_new_chunk_id.is_not(None),
        )
        .values(**values)
    )
    return bool(result.rowcount)  # ty: ignore[unresolved-attribute]


def touch_amendment_proposal_approval(
    db_session: Session,
    *,
    proposal_id: int,
) -> bool:
    """Keep an applied active-index projection out of stale recovery."""

    result = db_session.execute(
        update(AmendmentProposal)
        .where(
            AmendmentProposal.id == proposal_id,
            AmendmentProposal.status == AmendmentProposalStatus.APPROVING.value,
            AmendmentProposal.applied_new_chunk_id.is_not(None),
            AmendmentProposal.approval_indexing_job_id.is_(None),
        )
        .values(updated_at=func.now())
    )
    db_session.commit()
    return bool(result.rowcount)  # ty: ignore[unresolved-attribute]


def recover_stale_amendment_proposal_approvals(
    db_session: Session,
    *,
    stale_before: datetime.datetime,
    recovered_at: datetime.datetime,
    limit: int = 100,
) -> list[int]:
    """Repair pre-job approvals and claim applied-but-unlinked approvals."""

    proposals = list(
        db_session.scalars(
            select(AmendmentProposal)
            .where(
                AmendmentProposal.status == AmendmentProposalStatus.APPROVING.value,
                AmendmentProposal.approval_indexing_job_id.is_(None),
                AmendmentProposal.updated_at < stale_before,
            )
            .order_by(AmendmentProposal.updated_at, AmendmentProposal.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
    )
    resume_ids: list[int] = []
    for proposal in proposals:
        if proposal.applied_new_chunk_id:
            proposal.updated_at = recovered_at
            resume_ids.append(proposal.id)
            continue
        proposal.status = AmendmentProposalStatus.PENDING.value
        proposal.decided_by = None
        proposal.decided_at = None
        proposal.approval_error = None
        proposal.updated_at = recovered_at
    db_session.commit()
    return resume_ids


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


def _validated_reviewed_chunk_draft(
    stored_draft: dict[str, Any],
    reviewed_draft: dict[str, Any],
    *,
    old_chunk_snapshot: dict[str, Any],
) -> dict[str, Any]:
    try:
        stored = ReviewedAmendmentChunkDraft.model_validate(stored_draft)
        reviewed = ReviewedAmendmentChunkDraft.model_validate(reviewed_draft)
    except ValueError as error:
        raise ValueError(f"Invalid reviewed chunk draft: {error}") from error

    if reviewed.user_file_id != stored.user_file_id:
        raise ValueError("Reviewed chunk draft cannot change user_file_id")
    if reviewed.position != stored.position:
        raise ValueError("Reviewed chunk draft cannot change position")
    canonical_chunk_type = old_chunk_snapshot.get("chunk_type")
    if (
        canonical_chunk_type is not None
        and reviewed.chunk_type is not None
        and reviewed.chunk_type != canonical_chunk_type
    ):
        raise ValueError("Reviewed chunk draft cannot change chunk_type")
    chunk_type = canonical_chunk_type or reviewed.chunk_type or stored.chunk_type
    metadata = dict(reviewed.metadata)
    canonical_metadata = dict(old_chunk_snapshot.get("metadata") or {})
    if "metadata" in old_chunk_snapshot:
        for key in (
            "article_no",
            "paragraph_no",
            "clause_label",
            "subclause_label",
        ):
            if canonical_metadata.get(key) is None:
                metadata.pop(key, None)
            else:
                metadata[key] = canonical_metadata[key]
    canonical_heading_path = old_chunk_snapshot.get("heading_path")
    heading_path = reconcile_existing_heading_path(
        (
            canonical_heading_path
            if canonical_heading_path is not None
            else reviewed.heading_path or stored.heading_path
        ),
        amended_text=reviewed.text,
        chunk_type=chunk_type,
        article_no=(
            str(metadata["article_no"])
            if metadata.get("article_no") is not None
            else None
        ),
        article_title=(
            str(metadata["article_title"])
            if metadata.get("article_title") is not None
            else None
        ),
        paragraph_no=(
            str(metadata["paragraph_no"])
            if metadata.get("paragraph_no") is not None
            else None
        ),
        clause_label=(
            str(metadata["clause_label"])
            if metadata.get("clause_label") is not None
            else None
        ),
        subclause_label=(
            str(metadata["subclause_label"])
            if metadata.get("subclause_label") is not None
            else None
        ),
    )
    metadata["heading_path"] = list(heading_path)
    payload = reviewed.model_dump(mode="json")
    payload.update(
        chunk_type=chunk_type,
        heading_path=heading_path,
        metadata=metadata,
    )
    return payload


def _proposal_instruction_texts(proposal: AmendmentProposal) -> list[str]:
    return list(
        getattr(proposal, "instruction_texts", None)
        or [getattr(proposal, "instruction_text", "")]
    )


def _review_snapshot_value(chunk: RegulatoryChunk, key: str) -> Any:
    """Return the live value for one field persisted in a review snapshot."""
    if key == "user_file_id":
        return str(chunk.user_file_id)
    if key == "metadata":
        return dict(chunk.chunk_metadata)
    if key == "heading_path":
        return list(chunk.heading_path)
    if key in {"validity_start_date", "validity_end_date"}:
        value = getattr(chunk, key)
        return value.isoformat() if value is not None else None
    if key == "created_at":
        return getattr(chunk, key).isoformat()
    return getattr(chunk, key)


def _ensure_old_chunk_matches_review_snapshot(
    chunk: RegulatoryChunk,
    snapshot: dict[str, Any],
) -> None:
    supported_keys = {
        "id",
        "user_file_id",
        "position",
        "text",
        "chunk_type",
        "heading_path",
        "metadata",
        "validity_start_date",
        "validity_end_date",
        "status",
        "source",
        "supersedes_chunk_id",
        "superseded_by_chunk_id",
        "created_at",
    }
    for key in snapshot.keys() & supported_keys:
        if snapshot[key] != _review_snapshot_value(chunk, key):
            raise ValueError(
                f"Old chunk {chunk.id} changed after analysis; "
                "refresh or reanalyze before approval."
            )


def approve_amendment_proposal(
    db_session: Session,
    proposal: AmendmentProposal,
    *,
    decided_by: UUID | None = None,
) -> ApprovalResult:
    """Apply one durably queued proposal:

    1. Verify the proposal is still 'approving'.
    2. Insert the new chunk (source='amendment').
    3. If old_chunk_id is set, mark it superseded — guarded by
       `status == 'active'` so two proposals racing to supersede the same
       chunk can't both succeed silently. The new chunk's
       `validity_start_date` doubles as the old chunk's supersession
       boundary (when the new text starts applying is when the old text
       stops).
    4. Record which chunk it produced while leaving the proposal `approving`.
       Durable index publication is the only operation that marks it approved.

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
    proposal = _lock_proposal_for_transition(db_session, proposal.id)
    if proposal.status != AmendmentProposalStatus.APPROVING.value:
        raise ValueError(
            f"Amendment proposal {proposal.id} is not queued for approval "
            f"(status: {proposal.status})."
        )

    applied_new_chunk_id = getattr(proposal, "applied_new_chunk_id", None)
    if isinstance(applied_new_chunk_id, str) and applied_new_chunk_id:
        new_chunk = db_session.get(RegulatoryChunk, applied_new_chunk_id)
        if new_chunk is None:
            raise ValueError(
                f"Applied chunk {applied_new_chunk_id} no longer exists; "
                "cannot resume approval."
            )
        old_chunk = (
            db_session.get(RegulatoryChunk, new_chunk.supersedes_chunk_id)
            if new_chunk.supersedes_chunk_id
            else None
        )
        return ApprovalResult(
            proposal=proposal,
            new_chunk=new_chunk,
            old_chunk=old_chunk,
        )

    snapshot_target_id = (getattr(proposal, "old_chunk_snapshot", None) or {}).get("id")
    if (
        isinstance(snapshot_target_id, str)
        and snapshot_target_id
        and proposal.old_chunk_id != snapshot_target_id
    ):
        raise ValueError(
            f"Old chunk {snapshot_target_id} no longer exists; cannot approve."
        )

    draft = _validated_reviewed_chunk_draft(
        proposal.new_chunk_draft,
        proposal.new_chunk_draft,
        old_chunk_snapshot=getattr(proposal, "old_chunk_snapshot", None) or {},
    )
    validate_explicit_replacement_texts(
        _proposal_instruction_texts(proposal),
        draft["text"],
    )
    user_file_id = UUID(draft["user_file_id"])
    today = datetime.date.today()
    start_date_str = draft.get("effective_start_date")
    end_date_str = draft.get("effective_end_date")
    start_date = (
        datetime.date.fromisoformat(start_date_str) if start_date_str else today
    )
    end_date = datetime.date.fromisoformat(end_date_str) if end_date_str else None
    if end_date is not None and end_date <= start_date:
        raise ValueError("effective_end_date must be after effective_start_date")

    old_chunk: RegulatoryChunk | None = None
    if proposal.old_chunk_id:
        old_chunk = db_session.scalar(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.id == proposal.old_chunk_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if old_chunk is None:
            raise ValueError(
                f"Old chunk {proposal.old_chunk_id} no longer exists; cannot approve."
            )
        if old_chunk.status != "active":
            raise ValueError(
                f"Old chunk {proposal.old_chunk_id} is already "
                f"{old_chunk.status}; cannot approve."
            )
        if is_hierarchical_aggregate_chunk(old_chunk):
            raise ValueError("Derived aggregate chunks cannot be amended directly.")
        _ensure_old_chunk_matches_review_snapshot(
            old_chunk,
            getattr(proposal, "old_chunk_snapshot", None) or {},
        )
        instruction_texts = _proposal_instruction_texts(proposal)
        if any(explicit_replacement_body(text) for text in instruction_texts):
            reject_unsupported_descendant_replacement_texts(
                instruction_texts,
                has_active_descendants=has_active_structural_descendants(
                    db_session, old_chunk
                ),
            )

    new_chunk_metadata = dict(draft.get("metadata") or {})
    new_chunk_metadata.setdefault("chunk_variant", ATOMIC_CHUNK_VARIANT)
    new_chunk_metadata.setdefault("source_chunk_orders", [])
    new_chunk_metadata.setdefault("source_regulatory_chunk_ids", [])

    new_chunk_id = make_regulatory_chunk_id(
        user_file_id,
        draft["position"],
        draft["text"],
        version_key=f"amendment:{proposal.id}",
    )
    projection_ordinal = _AMENDMENT_PROJECTION_ORDINAL_BASE + proposal.id
    if proposal.id > _MAX_AMENDMENT_PROPOSAL_ID:
        raise ValueError("Amendment projection ordinal range is exhausted")
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
        projection_ordinal=projection_ordinal,
        validity_start_date=start_date,
        validity_end_date=end_date,
        supersedes_chunk_id=old_chunk.id if old_chunk else None,
    )
    db_session.add(new_chunk)
    db_session.flush()

    if old_chunk is not None:
        supersede_hierarchical_aggregates_referencing_chunk(
            db_session,
            user_file_id=user_file_id,
            source_chunk_id=old_chunk.id,
            superseded_at=start_date,
        )
        old_chunk.status = RegulatoryChunkStatus.SUPERSEDED.value
        old_chunk.validity_end_date = start_date
        old_chunk.superseded_by_chunk_id = new_chunk.id
        db_session.add(old_chunk)

    proposal.new_chunk_draft = draft
    proposal.status = AmendmentProposalStatus.APPROVING.value
    proposal.applied_new_chunk_id = new_chunk.id
    proposal.approval_error = None
    if decided_by is not None:
        proposal.decided_by = decided_by
    proposal.decided_at = None
    db_session.add(proposal)
    db_session.flush()

    return ApprovalResult(proposal=proposal, new_chunk=new_chunk, old_chunk=old_chunk)
