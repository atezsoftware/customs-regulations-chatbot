"""DB operations for the amendment (update) mechanism.

`approve_amendment_proposal` is the one place that ever writes an
amendment-sourced row into `regulatory_chunk` — everything upstream
(segmenting, candidate search, drafting) is pure analysis that only produces
`AmendmentProposal` rows for review.
"""

import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import RegulatoryChunkSource, RegulatoryChunkStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    RegulatoryChunk,
)
from onyx.db.regulatory_chunks import make_regulatory_chunk_id


def create_batch(
    db_session: Session,
    *,
    project_id: int,
    raw_text: str,
    created_by: UUID | None,
) -> AmendmentBatch:
    batch = AmendmentBatch(
        project_id=project_id,
        raw_text=raw_text,
        created_by=created_by,
    )
    db_session.add(batch)
    db_session.flush()
    return batch


def get_batch(db_session: Session, batch_id: int) -> AmendmentBatch | None:
    return db_session.get(AmendmentBatch, batch_id)


def list_batches_for_project(
    db_session: Session, project_id: int
) -> list[AmendmentBatch]:
    stmt = (
        select(AmendmentBatch)
        .where(AmendmentBatch.project_id == project_id)
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
        chunk_metadata=draft.get("metadata") or {},
        status=RegulatoryChunkStatus.ACTIVE.value,
        source=RegulatoryChunkSource.AMENDMENT.value,
        validity_start_date=start_date,
        validity_end_date=end_date,
        supersedes_chunk_id=old_chunk.id if old_chunk else None,
    )
    db_session.add(new_chunk)
    db_session.flush()

    if old_chunk is not None:
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
