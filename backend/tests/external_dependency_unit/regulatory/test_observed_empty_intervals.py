"""Retain same-day superseded evidence without inventing a valid legal day."""

import importlib.util
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import CheckConstraint, Table, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
)
from onyx.db.regulatory_context_projections import (
    activate_temporal_projection,
    get_indexed_temporal_projection,
)
from onyx.document_index.publication_models import (
    FileOwnership,
    IndexedProjectionEvidence,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.publication_baseline import (
    audit_baseline_inventory,
    observed_baseline_binding,
)
from tests.external_dependency_unit.regulatory.test_revision_batching import (
    owned_canonical,
)
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)


@pytest.fixture(params=["model", "migration"])
def temporal_session(request: pytest.FixtureRequest) -> Generator[Session, None, None]:
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_session_with_tenant(tenant_id="public") as session:
        try:
            # Shadow only this connection's table; DEV data/schema stay unchanged.
            session.execute(
                text(
                    "CREATE TEMP TABLE regulatory_temporal_projection "
                    "(LIKE public.regulatory_temporal_projection INCLUDING ALL) ON COMMIT DROP"
                )
            )
            if request.param == "model":
                constraint = next(
                    c
                    for c in cast(
                        Table, RegulatoryTemporalProjection.__table__
                    ).constraints
                    if isinstance(c, CheckConstraint)
                    and c.name == "temporal_projection_dates_check"
                )
                session.execute(
                    text(
                        "ALTER TABLE pg_temp.regulatory_temporal_projection "
                        "DROP CONSTRAINT temporal_projection_dates_check"
                    )
                )
                session.execute(
                    text(
                        "ALTER TABLE pg_temp.regulatory_temporal_projection "
                        f"ADD CONSTRAINT temporal_projection_dates_check CHECK ({constraint.sqltext})"
                    )
                )
            else:
                path = (
                    Path(__file__).resolve().parents[3]
                    / "alembic/versions/f43b8c129e70_retain_observed_empty_temporal_intervals.py"
                )
                spec = importlib.util.spec_from_file_location(
                    "empty_interval_migration", path
                )
                assert spec is not None and spec.loader is not None
                migration = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(migration)
                setattr(
                    migration,
                    "op",
                    Operations(MigrationContext.configure(session.connection())),
                )
                migration.upgrade()
            yield session
        finally:
            session.rollback()


def empty_case(
    session: Session,
) -> tuple[RegulatoryChunk, IndexedProjectionEvidence, FileOwnership]:
    owner, snapshots = owned_canonical(session, 1)
    row = session.get(RegulatoryChunk, snapshots[0].id)
    assert row is not None
    row.status = "superseded"
    row.source = "amendment"
    row.validity_start_date = row.validity_end_date = date(2026, 7, 4)
    session.flush()
    template = observed()
    source = json.loads(template.source_json)
    source.update(
        document_id=str(owner.user_file_id),
        regulatory_chunk_id=row.id,
        chunk_index=row.projection_ordinal,
        content=row.text,
        heading_path=row.heading_path,
        doc_summary="",
        chunk_context="",
        metadata_suffix="",
        source_links=json.dumps({0: ""}),
        image_file_id=None,
        validity_start_date=1783123200,
        validity_end_date=1783123200,
    )
    return (
        row,
        IndexedProjectionEvidence(
            index=template.observed_index,
            source_json=json.dumps(source),
            frozen_projection=None,
            payload_sha256=None,
        ),
        owner,
    )


def test_same_day_archive_retains_source_and_is_never_valid_as_of_a_day(
    temporal_session: Session,
) -> None:
    row, evidence, _ = empty_case(temporal_session)
    snapshot = _snapshot(row)
    before = audit_baseline_inventory([snapshot], [evidence], [])
    binding = observed_baseline_binding(evidence, [snapshot])
    activate_temporal_projection(
        temporal_session, user_file_id=row.user_file_id, binding=binding
    )
    stored = temporal_session.get(RegulatoryTemporalProjection, binding.id)
    assert stored is not None and stored.canonical_revision_id is not None
    retained = AnnexTemporalProjection.model_validate(stored.payload)
    assert json.loads(retained.projection.source_json) == json.loads(
        evidence.source_json
    )
    after = audit_baseline_inventory(
        [snapshot],
        [evidence.model_copy(update={"observed_projection": retained.projection})],
        [retained],
    )
    assert after.state == "ready" and after.binding_count == after.indexed_count == 1
    assert (after.source_sha256, after.vectors_sha256) == (
        before.source_sha256,
        before.vectors_sha256,
    )
    for day in (date(2026, 7, 3), date(2026, 7, 4), date(2026, 7, 5)):
        assert (
            get_indexed_temporal_projection(
                temporal_session, row.id, index=binding.index, as_of_date=day
            )
            is None
        )


@pytest.mark.parametrize("kind", ["active", "reversed"])
def test_baseline_preflight_rejects_invalid_dates_before_index_writes(
    temporal_session: Session,
    kind: str,
) -> None:
    row, evidence, _ = empty_case(temporal_session)
    if kind == "active":
        row.status = "active"
    else:
        row.validity_end_date = date(2026, 7, 3)
        source = json.loads(evidence.source_json)
        source["validity_end_date"] = 1783036800
        evidence = evidence.model_copy(update={"source_json": json.dumps(source)})
    result = audit_baseline_inventory([_snapshot(row)], [evidence], [])
    assert result.state == "unresolved"
    assert result.issues and result.issues[0].code == "source_integrity"


@pytest.mark.parametrize("kind", ["active", "contracted"])
def test_activation_rejects_fabricated_empty_legal_history(
    temporal_session: Session,
    kind: str,
) -> None:
    row, evidence, _ = empty_case(temporal_session)
    binding = observed_baseline_binding(evidence, [_snapshot(row)])
    if kind == "active":
        row.status = "active"
        temporal_session.flush()
    else:
        binding = binding.model_copy(
            update={
                "projection": binding.projection.model_copy(
                    update={"observed_start": 1783036800}
                )
            }
        )
    with pytest.raises(ValueError, match="empty|interval"):
        activate_temporal_projection(
            temporal_session, user_file_id=row.user_file_id, binding=binding
        )
    assert temporal_session.scalar(select(RegulatoryTemporalProjection.id)) is None


@pytest.mark.parametrize(
    "kind", ["missing_kind", "contracted", "reversed", "wrong_day"]
)
def test_database_rejects_unproven_empty_intervals(
    temporal_session: Session,
    kind: str,
) -> None:
    row, evidence, _ = empty_case(temporal_session)
    binding = observed_baseline_binding(evidence, [_snapshot(row)])
    payload = binding.model_dump(mode="json")
    start = end = date(2026, 7, 4)
    if kind == "missing_kind":
        del payload["projection"]["evidence_kind"]
    elif kind == "contracted":
        payload["projection"]["observed_start"] = 1783036800
    elif kind == "wrong_day":
        start = end = date(2026, 7, 5)
    else:
        end = date(2026, 7, 3)
    with pytest.raises(IntegrityError):
        with temporal_session.begin_nested():
            temporal_session.execute(
                text(
                    "INSERT INTO regulatory_temporal_projection "
                    "(id,user_file_id,canonical_chunk_id,index_uuid,index_identity_sha256,"
                    "projection_ordinal,effective_start,effective_end,payload,payload_sha256) "
                    "VALUES (:id,:file,:chunk,'index','identity',0,:start,:end,CAST(:payload AS jsonb),'hash')"
                ),
                dict(
                    id=binding.id,
                    file=row.user_file_id,
                    chunk=row.id,
                    start=start,
                    end=end,
                    payload=json.dumps(payload),
                ),
            )


@pytest.mark.parametrize(
    "corruption", [None, "active", "source_interval", "revision_scope"]
)
def test_frozen_recovery_checks_archival_dates_before_returning_index_work(
    temporal_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str | None,
) -> None:
    from onyx.db import regulatory_writer_publication as writer
    from onyx.db.regulatory_canonical_revisions import retain_canonical_revision
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

    row, evidence, owner = empty_case(temporal_session)
    binding = observed_baseline_binding(evidence, [_snapshot(row)])
    if corruption == "active":
        row.status = "active"
    if corruption == "source_interval":
        source = json.loads(binding.projection.source_json)
        source["validity_end_date"] = 1783036800
        binding = binding.model_copy(
            update={
                "projection": binding.projection.model_copy(
                    update={"source_json": json.dumps(source)}
                )
            }
        )
    snapshot = _snapshot(row)
    if corruption == "revision_scope":
        snapshot = snapshot.model_copy(update={"user_file_id": str(uuid4())})
    revision = retain_canonical_revision(temporal_session, snapshot)
    manifest = WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="baseline",
        canonical_before_sha256="a" * 64,
        indexes=[binding.index],
        previous_binding_ids=[],
        bindings=[binding],
        canonical_revisions={binding.id: revision},
    )
    payload = manifest.model_dump(mode="json")
    state = temporal_session.get(RegulatoryFilePublication, owner.user_file_id)
    assert state is not None
    state.writer_manifest = payload
    state.writer_manifest_sha256 = publication_digest(payload)
    state.gate_closed = True
    temporal_session.flush()

    @contextmanager
    def sessions(*, tenant_id: str) -> Generator[Session, None, None]:
        assert tenant_id == "public"
        with Session(
            bind=temporal_session.connection(), join_transaction_mode="create_savepoint"
        ) as session:
            yield session

    monkeypatch.setattr(writer, "get_session_with_tenant", sessions)
    if corruption:
        with pytest.raises(ValueError, match="interval"):
            writer.pending_writer_manifest(owner)
    else:
        assert writer.pending_writer_manifest(owner) == manifest
    temporal_session.refresh(state)
    assert state.writer_manifest == payload
    assert state.writer_manifest_sha256 == publication_digest(payload)


@pytest.mark.parametrize(
    "start,end,when,visible",
    [
        (None, None, date(2026, 7, 4), True),
        (None, date(2026, 7, 4), date(2026, 7, 3), True),
        (None, date(2026, 7, 4), date(2026, 7, 4), False),
        (date(2026, 7, 4), None, date(2026, 7, 4), True),
    ],
)
def test_nullable_legal_boundaries_keep_existing_lookup_semantics(
    temporal_session: Session,
    start: date | None,
    end: date | None,
    when: date,
    visible: bool,
) -> None:
    from onyx.regulatory.amendments.annexes.publication_representations import _epoch

    row, evidence, _ = empty_case(temporal_session)
    row.status = "active"
    row.validity_start_date, row.validity_end_date = start, end
    temporal_session.flush()
    source = json.loads(evidence.source_json)
    source.update(validity_start_date=_epoch(start), validity_end_date=_epoch(end))
    evidence = evidence.model_copy(update={"source_json": json.dumps(source)})
    binding = observed_baseline_binding(evidence, [_snapshot(row)])
    activate_temporal_projection(
        temporal_session, user_file_id=row.user_file_id, binding=binding
    )
    result = get_indexed_temporal_projection(
        temporal_session, row.id, index=binding.index, as_of_date=when
    )
    assert (result is not None) is visible
    assert row.validity_start_date == start and row.validity_end_date == end
