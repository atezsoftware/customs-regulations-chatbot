"""Revision preconditions for canonical edits under publication ownership."""

from datetime import datetime

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import RegulatoryChunk
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
