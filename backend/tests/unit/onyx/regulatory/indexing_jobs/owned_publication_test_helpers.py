"""Typed owned-service fixtures; no database, provider or index connections."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from onyx.access.models import DocumentAccess
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.db.regulatory_writer_publication import OwnedWriterInputs
from onyx.document_index.publication_models import FileOwnership, PublicationScope
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.models import (
    AnnexProjectionAccess,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot


class OwnedAuthority:
    """Explicit scoped owner and deterministic allocations for caller unit tests."""

    def __init__(self, file_id: UUID, tenant_id: str = "tenant-a") -> None:
        self.scope = PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
        self.owner = FileOwnership(
            scope=self.scope,
            user_file_id=file_id,
            owner_id=uuid4(),
            fencing_token=7,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        self.allocations: dict[str, int] = {}
        self.released: list[FileOwnership] = []

    def for_scope(self, scope: PublicationScope) -> "OwnedAuthority":
        assert scope == self.scope
        return self

    def acquire(
        self, file_id: UUID, *, owner_id: UUID, ttl: timedelta
    ) -> FileOwnership:
        assert isinstance(owner_id, UUID)
        assert file_id == self.owner.user_file_id
        assert ttl > timedelta(0)
        return self.owner

    def release(self, owner: FileOwnership) -> None:
        assert owner == self.owner
        self.released.append(owner)

    def allocate(self, owner: FileOwnership, allocation_key: str) -> int:
        assert owner == self.owner
        return self.allocations.setdefault(
            allocation_key, 1_000_000_043 + len(self.allocations)
        )


def writer_inputs(file_id: UUID, rows: list[RegulatoryChunk]) -> OwnedWriterInputs:
    return OwnedWriterInputs(
        file=UserFile(id=file_id, name="Regulation.md"),
        settings=[],
        access=AnnexProjectionAccess(
            access=DocumentAccess.build(
                user_emails=[],
                user_groups=[],
                external_user_emails=[],
                external_user_group_ids=[],
                is_public=False,
            ),
            project_ids=[],
            persona_ids=[],
            document_sets=[],
        ),
        canonical=[_snapshot(row) for row in rows],
        bindings=[],
        revisions={},
        canonical_revisions={row.id: uuid4() for row in rows},
        cached=PreparedContextView(),
        index_state_sha256="a" * 64,
    )


def canonical_row(file_id: UUID, position: int, text: str) -> RegulatoryChunk:
    return RegulatoryChunk(
        id=f"row-{position}",
        user_file_id=file_id,
        position=position,
        projection_ordinal=position,
        text=text,
        chunk_type="article",
        status="active",
        source="indexed",
        heading_path=[f"MADDE {position + 1}"],
        chunk_metadata={"article_no": str(position + 1)},
        validity_start_date=None,
        validity_end_date=None,
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
    )
