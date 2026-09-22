"""Bounded revision reads retain every authority check in a rolled-back transaction."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.engine import Connection, ExecutionContext
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.models import (
    DocumentSet,
    RegulatoryCanonicalRevision,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
)
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.document_index.publication_models import (
    FileOwnership,
    IndexedProjectionEvidence,
    PublicationScope,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.publication_baseline import observed_baseline_binding
from tests.external_dependency_unit.regulatory.test_annex_baseline import _file
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    canonical_row,
)


@pytest.fixture
def revision_session() -> Generator[Session, None, None]:
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    with get_session_with_tenant(tenant_id="public") as session:
        try:
            yield session
        finally:
            session.rollback()


@contextmanager
def counted_queries(session: Session) -> Generator[list[str], None, None]:
    statements: list[str] = []
    connection = session.connection()

    def count(
        _connection: Connection,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: ExecutionContext,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(connection, "before_cursor_execute", count)
    try:
        yield statements
    finally:
        event.remove(connection, "before_cursor_execute", count)


def revision_snapshot(file_id: UUID, ordinal: int) -> AnnexCanonicalSnapshot:
    row = canonical_row(file_id, ordinal, f"Original source {ordinal}")
    row.id = f"revision-batching-{file_id}-{ordinal}"
    return _snapshot(row)


def seed_bindings(
    session: Session,
    count: int,
    *,
    corruption: str | None = None,
    retain_revision: bool = True,
    seed_revisions: bool = True,
) -> tuple[UUID, list[AnnexTemporalProjection]]:
    file_id = uuid4()
    bindings = []
    template = observed()
    for ordinal in range(count):
        snapshot = revision_snapshot(file_id, ordinal)
        payload = snapshot.model_dump(mode="json")
        digest = publication_digest(payload)
        revision_id = uuid5(NAMESPACE_URL, "canonical-revision:" + digest)
        corrupt = ordinal == count - 1
        if corrupt and corruption == "hash":
            payload = {**payload, "text": "Unapproved source"}
        if seed_revisions:
            session.add(
                RegulatoryCanonicalRevision(
                    id=revision_id,
                    user_file_id=uuid4()
                    if corrupt and corruption == "scope"
                    else file_id,
                    canonical_chunk_id=snapshot.id,
                    payload_sha256=digest,
                    payload=payload,
                )
            )
        source = json.loads(template.source_json)
        source.update(
            document_id=str(file_id),
            chunk_index=ordinal,
            regulatory_chunk_id=snapshot.id,
            content=snapshot.text,
            heading_path=snapshot.heading_path,
            doc_summary="",
            chunk_context="",
            metadata_suffix="",
            validity_start_date=None,
            validity_end_date=None,
            source_links=json.dumps({0: ""}),
            image_file_id=None,
        )
        binding = observed_baseline_binding(
            IndexedProjectionEvidence(
                index=template.observed_index,
                source_json=json.dumps(source),
                frozen_projection=None,
                payload_sha256=None,
            ),
            [snapshot],
        )
        bindings.append(binding)
        payload = binding.model_dump(mode="json")
        if corrupt and corruption == "authority":
            payload = {**payload, "canonical_base_sha256": "0" * 64}
        session.add(
            RegulatoryTemporalProjection(
                id=binding.id,
                user_file_id=file_id,
                canonical_chunk_id=snapshot.id,
                canonical_revision_id=revision_id if retain_revision else None,
                index_uuid=binding.index.index_uuid,
                index_identity_sha256=binding.index.temporal_lookup_identity(),
                projection_ordinal=ordinal,
                payload=payload,
                payload_sha256=publication_digest(payload),
            )
        )
    session.flush()
    session.expunge_all()
    return file_id, bindings


@pytest.mark.parametrize("count", [1, 32, 257])
def test_binding_read_does_not_query_once_per_revision(
    revision_session: Session, count: int
) -> None:
    file_id, expected = seed_bindings(revision_session, count)
    with counted_queries(revision_session) as statements:
        result = load_file_temporal_bindings(revision_session, file_id)
    assert {binding.id: binding for binding in result} == {
        binding.id: binding for binding in expected
    }
    budget = 2 if count <= 32 else 4
    assert len(statements) <= budget, f"{len(statements)} queries for {count} bindings"


@pytest.mark.parametrize("corruption", ["hash", "scope", "authority"])
def test_batched_read_rejects_corrupt_revision(
    revision_session: Session, corruption: str
) -> None:
    file_id, _ = seed_bindings(revision_session, 4, corruption=corruption)
    with pytest.raises(ValueError, match="revision|authority"):
        load_file_temporal_bindings(revision_session, file_id)


def owned_canonical(
    session: Session, count: int
) -> tuple[FileOwnership, list[AnnexCanonicalSnapshot]]:
    from onyx.regulatory.amendments.annexes import config

    group = DocumentSet(name=f"revision-batching-{uuid4()}", description="")
    session.add(group)
    session.flush()
    file = _file(session, group)
    scope = PublicationScope(
        tenant_id="public",
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    owner = FileOwnership(
        scope=scope,
        user_file_id=file.id,
        owner_id=uuid4(),
        fencing_token=1,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    session.add(
        RegulatoryFilePublication(
            user_file_id=file.id,
            scope_key=publication_digest(scope.model_dump(mode="json")),
            owner_id=owner.owner_id,
            fencing_token=owner.fencing_token,
            lease_expires_at=owner.expires_at,
            gate_closed=False,
        )
    )
    snapshots = []
    for ordinal in range(count):
        row = canonical_row(file.id, ordinal, f"Original source {ordinal}")
        row.id = f"revision-batching-{file.id}-{ordinal}"
        session.add(row)
        snapshots.append(_snapshot(row))
    session.flush()
    session.expunge_all()
    return owner, snapshots


@pytest.mark.parametrize("count", [32, 257])
def test_owned_archive_batches_revision_writes_and_reuses_exact_authority(
    revision_session: Session, count: int
) -> None:
    from onyx.db.regulatory_canonical_revisions import get_canonical_revisions
    from onyx.db.regulatory_publication import archive_canonical_revisions

    owner, snapshots = owned_canonical(revision_session, count)
    with counted_queries(revision_session) as statements:
        first = archive_canonical_revisions(revision_session, owner)
    budget = 8 if count == 32 else 12
    assert len(statements) <= budget, f"{len(statements)} queries for {count} revisions"
    retained = get_canonical_revisions(revision_session, list(first.values()))
    for snapshot in snapshots:
        assert retained[first[snapshot.id]].snapshot == snapshot
    with counted_queries(revision_session) as repeated:
        assert archive_canonical_revisions(revision_session, owner) == first
    assert len(repeated) <= budget


@pytest.mark.parametrize("corrupt", [False, True])
def test_batch_retention_imports_only_proven_old_bindings(
    revision_session: Session, corrupt: bool
) -> None:
    from onyx.db.regulatory_canonical_revisions import retain_canonical_revisions

    file_id, _ = seed_bindings(
        revision_session,
        4,
        corruption="authority" if corrupt else None,
        retain_revision=False,
    )
    snapshots = [revision_snapshot(file_id, i) for i in range(4)]
    query = select(RegulatoryTemporalProjection).where(
        RegulatoryTemporalProjection.user_file_id == file_id
    )
    before = {row.id: row.payload_sha256 for row in revision_session.scalars(query)}
    if corrupt:
        with pytest.raises(ValueError, match="authority has changed"):
            with revision_session.begin_nested():
                retain_canonical_revisions(revision_session, snapshots)
        revision_session.expire_all()
        assert all(
            row.canonical_revision_id is None for row in revision_session.scalars(query)
        )
    else:
        revisions = retain_canonical_revisions(revision_session, snapshots)
        assert all(
            row.canonical_revision_id == revisions[row.canonical_chunk_id]
            for row in revision_session.scalars(query)
        )
    assert {
        row.id: row.payload_sha256 for row in revision_session.scalars(query)
    } == before


def test_batch_retention_keeps_prior_revision_after_a_correction(
    revision_session: Session,
) -> None:
    from onyx.db.regulatory_canonical_revisions import (
        get_canonical_revisions,
        retain_canonical_revisions,
    )

    before = revision_snapshot(uuid4(), 0)
    after = before.model_copy(update={"text": "Corrected source"})
    old_id = retain_canonical_revisions(revision_session, [before])[before.id]
    new_id = retain_canonical_revisions(revision_session, [after])[after.id]
    assert old_id != new_id
    revisions = get_canonical_revisions(revision_session, [old_id, new_id])
    assert revisions[old_id].snapshot == before
    assert revisions[new_id].snapshot == after
    with pytest.raises(ValueError, match="does not exist"):
        get_canonical_revisions(revision_session, [old_id, uuid4()])


def test_later_batch_failure_rolls_back_new_revisions_and_earlier_imports(
    revision_session: Session,
) -> None:
    from onyx.db.regulatory_canonical_revisions import retain_canonical_revisions

    file_id, _ = seed_bindings(
        revision_session,
        257,
        corruption="authority",
        retain_revision=False,
        seed_revisions=False,
    )
    snapshots = [revision_snapshot(file_id, i) for i in range(257)]
    with pytest.raises(ValueError, match="authority has changed"):
        with revision_session.begin_nested():
            retain_canonical_revisions(revision_session, snapshots)
    revision_session.expire_all()
    assert (
        revision_session.scalar(
            select(func.count())
            .select_from(RegulatoryCanonicalRevision)
            .where(RegulatoryCanonicalRevision.user_file_id == file_id)
        )
        == 0
    )
    links = list(
        revision_session.scalars(
            select(RegulatoryTemporalProjection.canonical_revision_id).where(
                RegulatoryTemporalProjection.user_file_id == file_id
            )
        )
    )
    assert len(links) == 257 and all(link is None for link in links)
