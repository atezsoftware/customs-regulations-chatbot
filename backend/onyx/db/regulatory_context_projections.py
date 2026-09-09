"""Scoped immutable context inputs and independently effective retrieval versions."""

import datetime
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryContextGeneration,
    RegulatoryContextProjection,
    RegulatoryContextProjectionCall,
    RegulatoryContextSnapshot,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    ContextGenerationCall,
    ContextSourceSnapshot,
    FrozenContextProjection,
    PreparedContextView,
)


def persist_context_view(
    session: Session, *, user_file_id: UUID, view: PreparedContextView
) -> list[str]:
    """Persist preparation without making it effective or changing legal content."""
    if view.issues:
        raise ValueError("cannot persist incomplete context preparation")
    ids = {projection.canonical_chunk_id for projection in view.projections}
    ids.update(
        item.canonical_chunk_id
        for snapshot in view.snapshots
        for item in snapshot.ordered_ranges
    )
    rows = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.id.in_(ids),
                RegulatoryChunk.user_file_id == user_file_id,
            )
        )
    }
    if set(rows) != ids:
        raise ValueError("context canonical scope mismatch")
    snapshots: dict[str, UUID] = {}
    for snapshot in view.snapshots:
        payload = snapshot.model_dump(mode="json")
        session.execute(
            insert(RegulatoryContextSnapshot)
            .values(
                id=uuid4(),
                user_file_id=user_file_id,
                sha256=snapshot.sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_snapshot_file_hash")
        )
        stored = session.scalars(
            select(RegulatoryContextSnapshot).where(
                RegulatoryContextSnapshot.user_file_id == user_file_id,
                RegulatoryContextSnapshot.sha256 == snapshot.sha256,
            )
        ).one()
        if stored.payload != payload:
            raise ValueError("context snapshot hash conflicts with frozen payload")
        snapshots[snapshot.sha256] = stored.id
    calls: dict[str, UUID] = {}
    for call in view.calls:
        payload = call.model_dump(mode="json")
        session.execute(
            insert(RegulatoryContextGeneration)
            .values(
                id=uuid4(),
                user_file_id=user_file_id,
                request_sha256=call.request_sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_generation_file_hash")
        )
        stored_call = session.scalars(
            select(RegulatoryContextGeneration).where(
                RegulatoryContextGeneration.user_file_id == user_file_id,
                RegulatoryContextGeneration.request_sha256 == call.request_sha256,
            )
        ).one()
        if stored_call.payload != payload:
            raise ValueError("context request conflicts with frozen output")
        calls[call.request_sha256] = stored_call.id
    result: list[str] = []
    for projection in view.projections:
        row = rows[projection.canonical_chunk_id]
        if projection.canonical_text_sha256 != context_hash(row.text):
            raise ValueError("canonical text changed during context preparation")
        if projection.embedding_input_sha256 != context_hash(
            projection.embedding_texts
        ) or projection.embedding_config_sha256 != context_hash(
            projection.embedding_config
        ):
            raise ValueError("frozen embedding input or configuration hash mismatch")
        if projection.source_snapshot_sha256 not in snapshots or not set(
            projection.request_hashes
        ).issubset(calls):
            raise ValueError("context projection has incomplete source dependencies")
        payload = projection.model_dump(mode="json", exclude={"projection_id"})
        digest = context_hash(payload)
        session.execute(
            insert(RegulatoryContextProjection)
            .values(
                id=uuid4(),
                canonical_chunk_id=row.id,
                source_snapshot_id=snapshots[projection.source_snapshot_sha256],
                payload_sha256=digest,
                embedding_input_sha256=projection.embedding_input_sha256,
                embedding_config_sha256=projection.embedding_config_sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_projection_payload")
        )
        stored_projection = session.scalars(
            select(RegulatoryContextProjection).where(
                RegulatoryContextProjection.canonical_chunk_id == row.id,
                RegulatoryContextProjection.payload_sha256 == digest,
            )
        ).one()
        for request_hash in projection.request_hashes:
            session.execute(
                insert(RegulatoryContextProjectionCall)
                .values(
                    projection_id=stored_projection.id,
                    generation_id=calls[request_hash],
                )
                .on_conflict_do_nothing()
            )
        result.append(str(stored_projection.id))
    session.flush()
    return result


def activate_context_projection(
    session: Session,
    projection_id: UUID,
    *,
    effective_start: datetime.date | None,
    effective_end: datetime.date | None,
) -> None:
    """Publication boundary: intersect context lifetime with canonical legal life."""
    projection = session.get(RegulatoryContextProjection, projection_id)
    if projection is None:
        raise ValueError("context projection does not exist")
    row = session.scalars(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.id == projection.canonical_chunk_id)
        .with_for_update()
    ).one()
    starts = [
        value
        for value in (effective_start, row.validity_start_date)
        if value is not None
    ]
    ends = [
        value for value in (effective_end, row.validity_end_date) if value is not None
    ]
    start, end = max(starts) if starts else None, min(ends) if ends else None
    if start is not None and end is not None and start >= end:
        raise ValueError("context projection has no canonical validity intersection")
    if projection.published_at is not None:
        if projection.effective_start != start or projection.effective_end != end:
            raise ValueError("published context projection interval is immutable")
        return
    overlapping = list(
        session.scalars(
            select(RegulatoryContextProjection)
            .where(
                RegulatoryContextProjection.canonical_chunk_id == row.id,
                RegulatoryContextProjection.published_at.is_not(None),
                or_(
                    RegulatoryContextProjection.effective_end.is_(None),
                    RegulatoryContextProjection.effective_end
                    > (start or datetime.date.min),
                ),
                or_(
                    RegulatoryContextProjection.effective_start.is_(None),
                    RegulatoryContextProjection.effective_start
                    < (end or datetime.date.max),
                ),
            )
            .with_for_update()
        )
    )
    if overlapping:
        raise ValueError("context projection overlaps published history")
    projection.effective_start, projection.effective_end = start, end
    projection.published_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()


def get_effective_context_projection(
    session: Session, canonical_chunk_id: str, *, as_of_date: datetime.date
) -> RegulatoryContextProjection | None:
    return session.scalars(
        select(RegulatoryContextProjection)
        .join(
            RegulatoryChunk,
            RegulatoryChunk.id == RegulatoryContextProjection.canonical_chunk_id,
        )
        .where(
            RegulatoryChunk.id == canonical_chunk_id,
            RegulatoryContextProjection.published_at.is_not(None),
            or_(
                RegulatoryContextProjection.effective_start.is_(None),
                RegulatoryContextProjection.effective_start <= as_of_date,
            ),
            or_(
                RegulatoryContextProjection.effective_end.is_(None),
                RegulatoryContextProjection.effective_end > as_of_date,
            ),
            or_(
                RegulatoryChunk.validity_start_date.is_(None),
                RegulatoryChunk.validity_start_date <= as_of_date,
            ),
            or_(
                RegulatoryChunk.validity_end_date.is_(None),
                RegulatoryChunk.validity_end_date > as_of_date,
            ),
        )
    ).one_or_none()


def load_context_view(
    session: Session, *, user_file_id: UUID, as_of_date: datetime.date
) -> PreparedContextView:
    rows = session.scalars(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
    ).all()
    projections: list[FrozenContextProjection] = []
    snapshots: dict[UUID, ContextSourceSnapshot] = {}
    calls: dict[UUID, ContextGenerationCall] = {}
    for row in rows:
        projection = get_effective_context_projection(
            session, row.id, as_of_date=as_of_date
        )
        if projection is None:
            continue
        projections.append(
            FrozenContextProjection.model_validate(
                {
                    **projection.payload,
                    "projection_id": str(projection.id),
                    "vector_reuse_verified": True,
                }
            )
        )
        if projection.source_snapshot_id not in snapshots:
            snapshot = session.get(
                RegulatoryContextSnapshot, projection.source_snapshot_id
            )
            if snapshot is None:
                raise ValueError("context snapshot missing")
            snapshots[snapshot.id] = ContextSourceSnapshot.model_validate(
                snapshot.payload
            )
        for call in session.scalars(
            select(RegulatoryContextGeneration)
            .join(
                RegulatoryContextProjectionCall,
                RegulatoryContextProjectionCall.generation_id
                == RegulatoryContextGeneration.id,
            )
            .where(RegulatoryContextProjectionCall.projection_id == projection.id)
        ):
            calls[call.id] = ContextGenerationCall.model_validate(call.payload)
    return PreparedContextView(
        projections=projections,
        snapshots=list(snapshots.values()),
        calls=list(calls.values()),
    )


def load_context_generation_calls(
    session: Session, *, user_file_id: UUID
) -> list[ContextGenerationCall]:
    return [
        ContextGenerationCall.model_validate(row.payload)
        for row in session.scalars(
            select(RegulatoryContextGeneration).where(
                RegulatoryContextGeneration.user_file_id == user_file_id
            )
        )
    ]


def retire_context_projection(
    session: Session, projection_id: UUID, *, effective_end: datetime.date
) -> None:
    """Close an existing context version; its legal identity and payload stay intact."""
    projection = session.scalars(
        select(RegulatoryContextProjection)
        .where(RegulatoryContextProjection.id == projection_id)
        .with_for_update()
    ).one_or_none()
    if projection is None or projection.published_at is None:
        raise ValueError("published context projection does not exist")
    if (
        projection.effective_start is not None
        and effective_end <= projection.effective_start
    ):
        raise ValueError("cannot erase context projection history")
    if (
        projection.effective_end is not None
        and effective_end > projection.effective_end
    ):
        raise ValueError("cannot reopen context projection history")
    projection.effective_end = effective_end
    session.flush()
