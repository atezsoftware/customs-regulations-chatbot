"""Read-only source identity and bounded canonical context for amendments."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, UserFile
from onyx.db.regulatory_chunks import RegulatoryChunkStructuralMatch


@dataclass(frozen=True)
class AmendmentSourceIdentity:
    user_file_id: UUID
    name: str
    root_heading: str


def load_amendment_source_identities(
    db_session: Session,
    user_file_ids: Sequence[UUID],
) -> list[AmendmentSourceIdentity]:
    if not user_file_ids:
        return []
    roots = (
        select(RegulatoryChunk.user_file_id, RegulatoryChunk.heading_path)
        .where(
            RegulatoryChunk.user_file_id.in_(user_file_ids),
            RegulatoryChunk.status == "active",
        )
        .distinct(RegulatoryChunk.user_file_id)
        .order_by(
            RegulatoryChunk.user_file_id, RegulatoryChunk.position, RegulatoryChunk.id
        )
        .subquery()
    )
    rows = db_session.execute(
        select(UserFile.id, UserFile.name, roots.c.heading_path)
        .outerjoin(roots, roots.c.user_file_id == UserFile.id)
        .where(UserFile.id.in_(user_file_ids))
    ).all()
    return [
        AmendmentSourceIdentity(row[0], row[1], row[2][0] if row[2] else "")
        for row in rows
    ]


def load_amendment_source_chunks(
    db_session: Session,
    user_file_id: UUID,
    *,
    limit: int = 4096,
) -> list[RegulatoryChunkStructuralMatch]:
    """Return a complete bounded file; an oversized file cannot imply a boundary."""
    rows = db_session.execute(
        select(RegulatoryChunk, UserFile.name)
        .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
        .where(
            RegulatoryChunk.user_file_id == user_file_id,
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
            RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].astext.is_(
                None
            ),
        )
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(limit + 1)
    ).all()
    if len(rows) > limit:
        return []
    return [
        RegulatoryChunkStructuralMatch(chunk=row[0], source_name=row[1]) for row in rows
    ]
