"""Revision preconditions for canonical edits under publication ownership."""

from datetime import datetime

from sqlalchemy import select

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_chunks import is_hierarchical_aggregate_chunk
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import FileOwnership


class ChunkEditConflict(ValueError):
    """The inspected chunk changed before the editor acquired its file."""


def require_owned_chunk_revision(
    owner: FileOwnership, chunk_id: str, expected_updated_at: datetime
) -> None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        chunk = session.get(RegulatoryChunk, chunk_id)
        if (
            chunk is None
            or chunk.user_file_id != owner.user_file_id
            or chunk.updated_at != expected_updated_at
        ):
            raise ChunkEditConflict(
                "This chunk has changed. Reload its latest version before saving."
            )


def require_owned_chunk_revisions(
    owner: FileOwnership,
    expected: dict[str, datetime],
    *,
    replacement_texts: dict[str, str | None],
) -> None:
    """Check the entire edit set before any canonical or publication work."""
    if not 1 <= len(expected) <= 32 or set(replacement_texts) != set(expected):
        raise ValueError("A correction requires 1 to 32 distinct canonical chunks")
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        chunks = {
            row.id: row
            for row in session.scalars(
                select(RegulatoryChunk).where(RegulatoryChunk.id.in_(expected))
            )
        }
        for chunk_id, updated_at in expected.items():
            chunk = chunks.get(chunk_id)
            replacement = replacement_texts[chunk_id]
            if (
                chunk is None
                or chunk.user_file_id != owner.user_file_id
                or chunk.updated_at != updated_at
                or is_hierarchical_aggregate_chunk(chunk)
                or not (chunk.text if replacement is None else replacement).strip()
            ):
                raise ChunkEditConflict(
                    "A target changed or is not an editable canonical chunk in this file. "
                    "Reload all target revisions before saving."
                )
