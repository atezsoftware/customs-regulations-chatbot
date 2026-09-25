"""Bound vector hydration while retaining the public reader's validation rules."""

import json
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4
from weakref import WeakSet

import pytest
from sqlalchemy import Column, MetaData, Table, create_engine, event
from sqlalchemy.orm import Session

from onyx.db import regulatory_public_reads as reads
from onyx.db.models import (
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
)
from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)

AS_OF = date(2026, 9, 25)
FILE_ID = UUID("00000000-0000-0000-0000-000000000001")
READ_BATCH_SIZE = reads._PUBLIC_TEMPORAL_READ_BATCH_SIZE
INDEX = PublicationIndexSnapshot(
    index_name="current-index",
    index_uuid="physical-index",
    search_settings_id=9,
    model_provider="fixture",
    model_name="fixture",
    vector_dimension=3,
    embedding_config_sha256=publication_digest({}),
    multitenant=False,
)


class _TrackingCursor(sqlite3.Cursor):
    closed = False

    def fetchmany(self, size: int | None = 1) -> list[Any]:
        driver = cast(_TrackingConnection, self.connection)
        batch_size = self.arraysize if size is None else size
        driver.fetch_sizes.append(batch_size)
        return super().fetchmany(batch_size)

    def fetchall(self) -> list[Any]:
        driver = cast(_TrackingConnection, self.connection)
        driver.fetchall_count += 1
        return super().fetchall()

    def close(self) -> None:
        self.closed = True
        super().close()


class _TrackingConnection(sqlite3.Connection):
    def __init__(self) -> None:
        super().__init__(":memory:")
        self.fetch_sizes: list[int] = []
        self.fetchall_count = 0
        self.cursors: list[_TrackingCursor] = []

    def cursor(self, factory: Any = _TrackingCursor) -> _TrackingCursor:
        cursor = super().cursor(factory)
        self.cursors.append(cursor)
        return cursor


@pytest.fixture
def session() -> Iterator[Session]:
    # SQLite substitutes only the external database; ORM queries and all source,
    # canonical revision, index, date and encoder validation remain real.
    driver = _TrackingConnection()
    engine = create_engine("sqlite://", creator=lambda: driver)
    metadata = MetaData()
    for model in (
        RegulatoryTemporalProjection,
        RegulatoryCanonicalRevision,
        RegulatoryChunk,
    ):
        Table(
            model.__tablename__,
            metadata,
            *(
                Column(
                    column.name,
                    column.type.as_generic(),
                    primary_key=column.primary_key,
                    nullable=column.nullable,
                    server_default=column.server_default,
                )
                for column in model.__table__.columns
            ),
        )
    metadata.create_all(engine)
    with Session(engine) as db_session:
        db_session.info["driver"] = driver
        yield db_session
    engine.dispose()


def _add_binding(
    session: Session,
    ordinal: int,
    *,
    file_id: UUID = FILE_ID,
    index: PublicationIndexSnapshot = INDEX,
    start: date | None = None,
    end: date | None = None,
    canonical_end: date | None = None,
    retired: bool = False,
) -> RegulatoryTemporalProjection:
    chunk_id = f"{file_id}:{index.index_uuid}:{ordinal}"
    text = f"Provision {ordinal}"
    snapshot = AnnexCanonicalSnapshot(
        id=chunk_id,
        user_file_id=str(file_id),
        chunk_type="text",
        status="active",
        projection_ordinal=ordinal,
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
        position=200 - ordinal,
        text=text,
        heading_path=["MADDE 1"],
        metadata={},
        source="indexed",
        validity_start_date=None,
        validity_end_date=canonical_end,
    )
    revision_id = uuid4()
    session.add(
        RegulatoryCanonicalRevision(
            id=revision_id,
            user_file_id=file_id,
            canonical_chunk_id=chunk_id,
            payload=snapshot.model_dump(mode="json"),
            payload_sha256=publication_digest(snapshot.model_dump(mode="json")),
        )
    )
    session.add(
        RegulatoryChunk(
            **snapshot.model_dump(exclude={"metadata", "user_file_id"}),
            user_file_id=file_id,
            chunk_metadata={},
        )
    )
    binding = AnnexTemporalProjection(
        id=uuid4(),
        index=index,
        projection=FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id="context",
            source_json=json.dumps(
                {
                    "document_id": str(file_id),
                    "chunk_index": ordinal,
                    "regulatory_chunk_id": chunk_id,
                    "content": text,
                    "content_vector": [0.1, 0.2, 0.3],
                    "source_type": "file",
                    "public": False,
                    "access_control_list": [],
                    "global_boost": 1,
                    "semantic_identifier": "fixture",
                    "blurb": text,
                    "doc_summary": "",
                    "chunk_context": "",
                }
            ),
            embedding_inputs=(text,),
            embedding_config_json="{}",
        ),
        canonical_base_sha256=context_hash(text),
        derived_role="canonical",
        dependency_ids=[],
        representation_text=text,
        reference_date=None,
        effective_start=start,
        effective_end=end,
        semantic_position=200 - ordinal,
    )
    row = RegulatoryTemporalProjection(
        id=binding.id,
        user_file_id=file_id,
        canonical_chunk_id=chunk_id,
        canonical_revision_id=revision_id,
        index_uuid=index.index_uuid,
        index_identity_sha256=index.temporal_lookup_identity(),
        projection_ordinal=ordinal,
        effective_start=start,
        effective_end=end,
        payload=binding.model_dump(mode="json"),
        payload_sha256=publication_digest(binding.model_dump(mode="json")),
        retired_at=datetime.now(timezone.utc) if retired else None,
    )
    session.add(row)
    return row


def test_iterator_bounds_hydration_and_preserves_the_full_projection_set(
    session: Session,
) -> None:
    binding_count = 2 * READ_BATCH_SIZE + 2
    for ordinal in range(binding_count):
        _add_binding(session, ordinal)
    session.flush()
    session.expunge_all()
    expected = reads.load_public_temporal_bindings(
        session, FILE_ID, index=INDEX, as_of_date=AS_OF
    )
    session.expunge_all()
    driver = cast(_TrackingConnection, session.info["driver"])
    driver.fetch_sizes.clear()
    driver.fetchall_count = 0
    driver.cursors.clear()
    resident: WeakSet[RegulatoryTemporalProjection] = WeakSet()
    peak_resident = 0

    @event.listens_for(session, "loaded_as_persistent")
    def count_hydration(_session: Session, instance: object) -> None:
        nonlocal peak_resident
        if isinstance(instance, RegulatoryTemporalProjection):
            resident.add(instance)
            peak_resident = max(peak_resident, len(resident))

    iterator = getattr(reads, "iter_public_temporal_bindings", None)
    assert iterator is not None, "public reads need a bounded hydration iterator"
    actual = list(iterator(session, FILE_ID, index=INDEX, as_of_date=AS_OF))

    assert {binding.projection.ordinal for binding in actual} == set(
        range(binding_count)
    )
    assert {binding.id: binding for binding in actual} == {
        binding.id: binding for binding in expected
    }
    assert len(driver.cursors) == 1, "one streamed join must serve every batch"
    assert driver.fetchall_count == 0
    assert driver.fetch_sizes and max(driver.fetch_sizes) <= READ_BATCH_SIZE
    # SQLAlchemy briefly retains the old batch while constructing the next.
    assert 0 < peak_resident <= 2 * READ_BATCH_SIZE
    assert not resident
    assert driver.cursors[0].closed


@pytest.mark.parametrize(
    "ordinals,canonical_ids,expected",
    [
        (None, None, {0, 1}),
        ((0, 1, 2), None, {0, 1}),
        ((), None, set()),
        (None, (), set()),
        ((0,), (f"{FILE_ID}:physical-index:1",), set()),
        (None, (f"{FILE_ID}:physical-index:1",), {1}),
    ],
)
def test_iterator_preserves_index_date_retirement_and_identity_filters(
    session: Session,
    ordinals: tuple[int, ...] | None,
    canonical_ids: tuple[str, ...] | None,
    expected: set[int],
) -> None:
    _add_binding(session, 0)
    _add_binding(session, 1, start=AS_OF, end=date(2027, 1, 1))
    _add_binding(session, 2, end=AS_OF)
    _add_binding(session, 3, start=date(2027, 1, 1))
    _add_binding(session, 4, retired=True)
    _add_binding(session, 5, canonical_end=AS_OF)
    _add_binding(session, 6, index=INDEX.model_copy(update={"model_name": "other"}))
    _add_binding(session, 7, index=INDEX.model_copy(update={"index_uuid": "other"}))
    _add_binding(session, 8, file_id=uuid4())
    session.flush()
    session.expunge_all()
    iterator = getattr(reads, "iter_public_temporal_bindings", None)
    assert iterator is not None, "public reads need a bounded hydration iterator"
    actual = list(
        iterator(
            session,
            FILE_ID,
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals=ordinals,
            canonical_chunk_ids=canonical_ids,
        )
    )
    assert {binding.projection.ordinal for binding in actual} == expected


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("payload", "temporal binding payload changed"),
        ("encoder", "encoder receipt is not accepted"),
        ("canonical", "canonical revision payload changed"),
        ("missing_revision", "canonical revision does not exist"),
        ("revision_scope", "canonical revision scope mismatch"),
        ("revision_authority", "temporal canonical revision authority mismatch"),
        ("lookup_identity", "temporal binding lookup identity mismatch"),
    ],
)
def test_iterator_propagates_full_validation_errors(
    session: Session, corruption: str, error: str
) -> None:
    row = _add_binding(session, 0)
    session.flush()
    if corruption == "payload":
        row.payload_sha256 = "0" * 64
    elif corruption == "encoder":
        payload = dict(row.payload)
        payload["projection"] = {
            **payload["projection"],
            "embedding_config_json": '{"provider":"unaccepted"}',
        }
        row.payload = payload
        row.payload_sha256 = publication_digest(payload)
    elif corruption == "lookup_identity":
        row.canonical_chunk_id = "wrong-canonical"
        row.canonical_revision_id = None
    else:
        revision = session.get(RegulatoryCanonicalRevision, row.canonical_revision_id)
        assert revision is not None
        if corruption == "missing_revision":
            session.delete(revision)
        elif corruption == "revision_scope":
            revision.canonical_chunk_id = "wrong-canonical"
        elif corruption == "revision_authority":
            row.payload = {**row.payload, "canonical_base_sha256": "0" * 64}
            row.payload_sha256 = publication_digest(row.payload)
        else:
            revision.payload_sha256 = "0" * 64
    session.flush()
    session.expunge_all()
    iterator = getattr(reads, "iter_public_temporal_bindings", None)
    assert iterator is not None, "public reads need a bounded hydration iterator"
    driver = cast(_TrackingConnection, session.info["driver"])
    driver.cursors.clear()
    with pytest.raises(ValueError, match=error):
        list(iterator(session, FILE_ID, index=INDEX, as_of_date=AS_OF))
    assert all(cursor.closed for cursor in driver.cursors)


def test_closing_iterator_releases_the_streaming_cursor(session: Session) -> None:
    for ordinal in range(2 * READ_BATCH_SIZE + 2):
        _add_binding(session, ordinal)
    session.flush()
    session.expunge_all()
    driver = cast(_TrackingConnection, session.info["driver"])
    driver.cursors.clear()
    iterator = reads.iter_public_temporal_bindings(
        session, FILE_ID, index=INDEX, as_of_date=AS_OF
    )
    next(iterator)
    assert len(driver.cursors) == 1
    assert not driver.cursors[0].closed
    iterator.close()
    assert driver.cursors[0].closed
    assert not session.identity_map


def test_missing_current_canonical_still_validates_retained_revision(
    session: Session,
) -> None:
    row = _add_binding(session, 0)
    session.flush()
    canonical = session.get(RegulatoryChunk, row.canonical_chunk_id)
    assert canonical is not None
    session.delete(canonical)
    session.flush()
    assert (
        list(
            reads.iter_public_temporal_bindings(
                session, FILE_ID, index=INDEX, as_of_date=AS_OF
            )
        )
        == []
    )
    row.canonical_revision_id = uuid4()
    session.flush()
    with pytest.raises(ValueError, match="canonical revision does not exist"):
        list(
            reads.iter_public_temporal_bindings(
                session, FILE_ID, index=INDEX, as_of_date=AS_OF
            )
        )
