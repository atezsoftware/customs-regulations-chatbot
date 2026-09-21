"""Real local PostgreSQL/ES acceptance; owns every fixture and index it mutates."""

import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from datetime import timedelta
from threading import Event
from uuid import UUID, uuid4

import pytest
from elasticsearch import BadRequestError, Elasticsearch, NotFoundError
from sqlalchemy import delete

from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.models import DocumentSet, User, UserFile
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
    PublicationScope,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


@contextmanager
def create_owned_file(tenant_id: str = "public") -> Generator[UUID, None, None]:
    assert POSTGRES_HOST in ("localhost", "127.0.0.1") and str(POSTGRES_PORT) == "25432"
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
        session.add(group)
        session.flush()
        file = _file(session, group)
        _chunk(session, file, 0, "existing zero")
        _chunk(session, file, 1_000_000_042, "existing sparse")
        file_id, user_id, group_id = file.id, file.user_id, group.id
        session.commit()
    yield file_id
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        from onyx.db.models import (
            RegulatoryCanonicalRevision,
            RegulatoryFilePublication,
            RegulatoryPublicationOrdinal,
            RegulatoryTemporalProjection,
        )

        session.execute(
            delete(RegulatoryPublicationOrdinal).where(
                RegulatoryPublicationOrdinal.user_file_id == file_id
            )
        )
        session.execute(
            delete(RegulatoryFilePublication).where(
                RegulatoryFilePublication.user_file_id == file_id
            )
        )
        session.execute(
            delete(RegulatoryTemporalProjection).where(
                RegulatoryTemporalProjection.user_file_id == file_id
            )
        )
        session.execute(
            delete(RegulatoryCanonicalRevision).where(
                RegulatoryCanonicalRevision.user_file_id == file_id
            )
        )
        session.execute(delete(UserFile).where(UserFile.id == file_id))
        user = session.get(User, user_id)
        assert user is not None
        session.delete(user)
        session.execute(delete(DocumentSet).where(DocumentSet.id == group_id))
        session.commit()


@pytest.fixture
def owned_file() -> Generator[UUID, None, None]:
    with create_owned_file() as file_id:
        yield file_id


def store(tenant_id: str = "public") -> PublicationStore:
    return PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment="task5a-fixture",
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )


def adapter_for(es: tuple[Elasticsearch, str]) -> FencedPublicationIndex:
    client, name = es
    snapshot = PublicationIndexSnapshot(
        index_name=name,
        index_uuid=client.indices.get(index=name)[name]["settings"]["index"]["uuid"],
        search_settings_id=1,
        model_provider="fixture",
        model_name="fixture",
        vector_dimension=3,
        embedding_config_sha256=publication_digest({"model": "fixture"}),
        multitenant=False,
    )
    return FencedPublicationIndex(client, snapshot)


def frozen_projection(
    file_id: UUID, ordinal: int, content: str = "approved text"
) -> FrozenPublicationProjection:
    return FrozenPublicationProjection(
        ordinal=ordinal,
        context_projection_id=f"context-{ordinal}",
        source_json=json.dumps(
            {
                "document_id": str(file_id),
                "chunk_index": ordinal,
                "regulatory_chunk_id": "same-canonical",
                "source_type": "file",
                "public": False,
                "access_control_list": [],
                "global_boost": 1,
                "semantic_identifier": "Regulation",
                "blurb": "approved text",
                "content": content,
                "doc_summary": "summary",
                "chunk_context": "context",
                "content_vector": [0.1, 0.2, 0.3],
                "hidden": False,
                "image_file_id": "image-evidence",
                "source_links": "{0: 'immutable-source'}",
                "validity_start_date": 1767225600 + ordinal,
                "validity_end_date": 1893456000 + ordinal,
            }
        ),
        embedding_inputs=(content + "context",),
        embedding_config_json='{"model":"fixture"}',
    )


def test_concurrent_lease_and_shared_sparse_allocator(owned_file: UUID) -> None:
    authority = store()

    def claim() -> FileOwnership | None:
        try:
            return authority.acquire(
                owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30)
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(2) as executor:
        claims = list(executor.map(lambda _: claim(), range(2)))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    owner = winners[0]
    with ThreadPoolExecutor(2) as executor:
        ordinals = list(
            executor.map(
                lambda _: authority.allocate(owner, "new-annex-element"), range(2)
            )
        )
    assert ordinals == [1_000_000_043, 1_000_000_043]
    assert (
        authority.allocate(owner, "canonical:" + authority.owned_chunks(owner)[0].id)
        == 0
    )
    assert authority.reservations(owner).ordinals == (0, 1_000_000_042, 1_000_000_043)


def test_expiry_never_opens_gate_and_old_owner_loses_access(owned_file: UUID) -> None:
    authority = store()
    owner = authority.acquire(
        owned_file, owner_id=uuid4(), ttl=timedelta(milliseconds=500)
    )
    authority.close_gate(owner)
    # Waiting on a real DB roundtrip avoids making client wall-clock the lease authority.
    with get_session_with_tenant(tenant_id="public") as session:
        from sqlalchemy import text

        session.execute(text("SELECT pg_sleep(0.6)"))
    assert owned_file in authority.unavailable(authority.observe(), (owned_file,))
    newer = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    assert newer.fencing_token == owner.fencing_token + 1
    with pytest.raises(ValueError, match="ownership"):
        authority.allocate(owner, "stale")
    with pytest.raises(ValueError, match="ownership"):
        authority.owned_chunks(owner)


def test_activation_retains_ownership_while_transaction_holds_row_lock(
    owned_file: UUID,
) -> None:
    from sqlalchemy import text

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=1))
    authority.close_gate(owner)
    with get_session_with_tenant(tenant_id="public") as session:
        initial = authority.lock_owned_snapshot(session, owner)
        session.execute(text("SELECT pg_sleep(1.1)"))
        # A competing acquisition must remain blocked until activation commits.
        with ThreadPoolExecutor(1) as executor:
            competing = executor.submit(
                authority.acquire,
                owned_file,
                owner_id=uuid4(),
                ttl=timedelta(seconds=30),
            )
            try:
                with pytest.raises(TimeoutError):
                    competing.result(timeout=0.1)
                assert authority.lock_owned_snapshot(session, owner) == initial
            finally:
                session.rollback()
            replacement = competing.result(timeout=5)
    assert replacement.fencing_token > owner.fencing_token
    with pytest.raises(ValueError, match="ownership"):
        authority.reservations(owner)


def test_ownership_checks_do_not_reload_frozen_vector_payload(owned_file: UUID) -> None:
    from sqlalchemy import event

    from onyx.db.engine.sql_engine import get_sqlalchemy_engine
    from onyx.db.models import RegulatoryFilePublication

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    payload = {"frozen_vectors": [0.1] * 1024}
    with get_session_with_tenant(tenant_id="public") as session:
        authority.lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owned_file)
        assert row is not None
        row.writer_manifest = payload
        session.commit()
    statements: list[str] = []

    def capture(
        _connection: object, _cursor: object, statement: str, *_args: object
    ) -> None:
        statements.append(statement)

    engine = get_sqlalchemy_engine()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        authority.reservations(owner)
        authority.heartbeat(owner, ttl=timedelta(seconds=30))
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not any(
        "regulatory_file_publication.writer_manifest," in sql for sql in statements
    )
    # A publication caller can still load the exact frozen payload under the lock.
    with get_session_with_tenant(tenant_id="public") as session:
        authority.lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owned_file)
        assert row is not None and row.writer_manifest == payload


@pytest.mark.parametrize("boundary", ["commit", "rollback", "savepoint"])
def test_activation_lock_authority_cannot_survive_transaction_boundary(
    owned_file: UUID, boundary: str
) -> None:
    from sqlalchemy import text

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=1))
    with get_session_with_tenant(tenant_id="public") as session:
        nested = session.begin_nested() if boundary == "savepoint" else None
        authority.lock_owned_snapshot(session, owner)
        session.execute(text("SELECT pg_sleep(1.1)"))
        if nested is not None:
            nested.rollback()
        elif boundary == "commit":
            session.commit()
        else:
            session.rollback()
        with pytest.raises(ValueError, match="ownership"):
            authority.lock_owned_snapshot(session, owner)


def test_committed_observation_does_not_skip_uncommitted_lower_epoch(
    owned_file: UUID,
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    observation = authority.observe()
    entered, release, other_entered = Event(), Event(), Event()

    def pending_event() -> None:
        with get_session_with_tenant(tenant_id="public") as session:
            authority.record_event(session, owner)
            entered.set()
            assert release.wait(5)
            session.commit()

    with create_owned_file() as other_file:
        other_owner = authority.acquire(
            other_file, owner_id=uuid4(), ttl=timedelta(seconds=30)
        )

        def higher_event() -> None:
            other_entered.set()
            authority.close_gate(other_owner)

        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(pending_event)
            assert entered.wait(5)
            second = executor.submit(higher_event)
            assert other_entered.wait(5)
            try:
                # B cannot commit a higher epoch while A holds its uncommitted lower one.
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.1)
                middle = authority.observe()
                assert middle == observation
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        assert authority.observe().committed_epoch == observation.committed_epoch + 2
        assert authority.unavailable(middle, (owned_file, other_file)) == {
            owned_file,
            other_file,
        }
        # B publishing does not make an unrelated, open third file unavailable.
        with create_owned_file() as unrelated:
            assert not authority.unavailable(middle, (unrelated,))


@pytest.fixture
def es() -> Generator[tuple[Elasticsearch, str], None, None]:
    client = Elasticsearch("http://127.0.0.1:29200")
    name = "annex-5a-" + uuid4().hex
    from onyx.document_index.elasticsearch.schema import DocumentSchema

    client.indices.create(
        index=name, mappings=DocumentSchema.get_document_schema(3, False)
    )
    yield client, name
    client.indices.delete(index=name)
    client.close()


def test_late_unseen_create_equal_token_identity_and_tombstones(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        PublicationIndexSnapshot,
        publication_digest,
    )

    client, name = es
    index = PublicationIndexSnapshot(
        index_name=name,
        index_uuid=client.indices.get(index=name)[name]["settings"]["index"]["uuid"],
        search_settings_id=1,
        model_provider="fixture",
        model_name="fixture",
        vector_dimension=3,
        embedding_config_sha256=publication_digest({"model": "fixture"}),
        multitenant=False,
    )
    adapter = FencedPublicationIndex(client, index)
    old = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    extra = authority.allocate(old, "old-larger-file-unseen")
    authority.close_gate(old)
    old_inventory = authority.reservations(old)

    def projection(ordinal: int, content: str) -> FrozenPublicationProjection:
        return FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id=f"context-{ordinal}",
            source_json=json.dumps(
                {
                    "document_id": str(owned_file),
                    "chunk_index": ordinal,
                    "regulatory_chunk_id": "same-canonical",
                    "source_type": "file",
                    "public": False,
                    "access_control_list": [],
                    "global_boost": 1,
                    "semantic_identifier": "Regulation",
                    "blurb": "approved text",
                    "content": content,
                    "doc_summary": "summary",
                    "chunk_context": "context",
                    "content_vector": [0.1, 0.2, 0.3],
                    "hidden": False,
                }
            ),
            embedding_inputs=(content + "context",),
            embedding_config_json='{"model":"fixture"}',
        )

    old_late = projection(extra, "old uncreated text")
    # No initialization means a content update cannot create a searchable ID.
    with pytest.raises(NotFoundError):
        adapter.upsert(old_inventory, old_late)
    authority.release(old)
    new = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    inventory = authority.reservations(new)
    adapter.seal(inventory)
    live = projection(0, "new text")
    adapter.upsert(inventory, live)
    adapter.upsert(inventory, live)
    with pytest.raises(BadRequestError) as rejected:
        adapter.upsert(inventory, projection(0, "same token different text"))
    assert "different equal-token" in json.dumps(rejected.value.body)
    for ordinal in inventory.ordinals:
        if ordinal != 0:
            adapter.tombstone(inventory, ordinal)
    proof = adapter.verify(inventory, (live,))
    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, new, proof)
        session.commit()
    with pytest.raises(BadRequestError) as rejected:
        adapter.upsert(old_inventory, old_late)
    assert "stale or unsealed" in json.dumps(rejected.value.body)
    # Even delayed initialization cannot replace a newer permanent tombstone.
    adapter.initialize(old_inventory)
    adapter.verify(inventory, (live,))
    assert owned_file not in authority.unavailable(authority.observe(), (owned_file,))
    authority.release(new)
    newest = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(newest)
    recreated = authority.reservations(newest)
    adapter.seal(recreated)
    extra_live = projection(extra, "recreated new content")
    adapter.upsert(recreated, live)
    adapter.upsert(recreated, extra_live)
    adapter.tombstone(recreated, 1_000_000_042)
    result = adapter.verify(recreated, (live, extra_live))
    assert result.live_ordinals == (0, extra)


def test_index_evidence_is_actual_and_missing_legacy_inputs_are_unverified(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import TenantState
    from onyx.document_index.publication_models import (
        PublicationIndexSnapshot,
        publication_digest,
    )

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    client, name = es
    snapshot = PublicationIndexSnapshot(
        index_name=name,
        index_uuid=client.indices.get(index=name)[name]["settings"]["index"]["uuid"],
        search_settings_id=1,
        model_provider="fixture",
        model_name="fixture",
        vector_dimension=3,
        embedding_config_sha256=publication_digest({"model": "fixture"}),
        multitenant=False,
    )
    adapter = FencedPublicationIndex(client, snapshot)
    identity = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False), str(owned_file), 0
    )
    client.index(
        index=name,
        id=identity,
        document={
            "document_id": str(owned_file),
            "chunk_index": 0,
            "content": "actual legacy content",
            "doc_summary": "stored summary",
            "chunk_context": "stored context",
            "content_vector": [0.1, 0.2, 0.3],
            "regulatory_chunk_id": "legacy-canonical",
            "hidden": False,
        },
    )
    evidence = adapter.read_evidence(inventory, 0)
    assert json.loads(evidence.source_json)["content"] == "actual legacy content"
    assert evidence.frozen_projection is None
    assert evidence.payload_sha256 is None
    adapter.seal(inventory)
    assert adapter.read_evidence(inventory, 0).frozen_projection is None


def test_seal_alone_cannot_verify_and_post_open_token_cannot_change_payload(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    from elasticsearch import BadRequestError

    from onyx.document_index.interfaces_new import DocumentChunkVerificationError

    authority = store()
    before = authority.observe()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    with pytest.raises(DocumentChunkVerificationError):
        adapter.verify(inventory, ())
    for ordinal in inventory.ordinals:
        adapter.tombstone(inventory, ordinal)
    proof = adapter.verify(inventory, ())
    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, owner, proof)
        session.rollback()
    assert owned_file in authority.unavailable(authority.observe(), (owned_file,))
    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, owner, proof)
        session.commit()
    assert owned_file in authority.unavailable(before, (owned_file,))
    assert not authority.unavailable(authority.observe(), (owned_file, uuid4()))
    with pytest.raises(BadRequestError):
        adapter.upsert(inventory, frozen_projection(owned_file, 0))
    adapter.seal(inventory)
    with pytest.raises(BadRequestError):
        adapter.upsert(inventory, frozen_projection(owned_file, 0))


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "wrong"),
        ("doc_summary", "wrong"),
        ("chunk_context", "wrong"),
        ("content_vector", [0.2, 0.1, 0.3]),
        ("image_file_id", "wrong-image"),
        ("source_links", "wrong-source"),
        ("validity_end_date", 1999999999),
    ],
)
def test_verification_reads_actual_sparse_temporal_payload(
    owned_file: UUID, es: tuple[Elasticsearch, str], field: str, value: object
) -> None:
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import (
        DocumentChunkVerificationError,
        TenantState,
    )

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    projections = tuple(
        frozen_projection(owned_file, ordinal) for ordinal in inventory.ordinals
    )
    for projection in projections:
        adapter.upsert(inventory, projection)
    verified = adapter.verify(inventory, projections)
    assert verified.live_ordinals == (0, 1_000_000_042)
    assert verified.canonical_chunk_ids == {"same-canonical"}
    client, name = es
    identity = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False),
        str(owned_file),
        1_000_000_042,
    )
    client.update(index=name, id=identity, doc={field: value})
    with pytest.raises(DocumentChunkVerificationError, match="exact"):
        adapter.verify(inventory, projections)


def test_targeted_metadata_retains_ownership_and_exact_input_identity(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    from elasticsearch import BadRequestError

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    projection = frozen_projection(owned_file, 0)
    adapter.upsert(inventory, projection)
    evidence = adapter.read_evidence(inventory, 0)
    assert (
        evidence.frozen_projection is not None and evidence.payload_sha256 is not None
    )
    assert evidence.frozen_projection.embedding_inputs == ("approved textcontext",)
    authority.release(owner)
    newer = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    current = authority.reservations(newer)
    adapter.seal(current)
    changed = json.loads(projection.source_json)
    changed["document_sets"] = ["new-access-set"]
    metadata = projection.model_copy(update={"source_json": json.dumps(changed)})
    with pytest.raises(BadRequestError):
        adapter.update_metadata(
            inventory,
            previous=projection,
            updated=metadata,
            previous_payload_sha256=evidence.payload_sha256,
        )
    adapter.update_metadata(
        current,
        previous=projection,
        updated=metadata,
        previous_payload_sha256=evidence.payload_sha256,
    )
    adapter.update_metadata(
        current,
        previous=projection,
        updated=metadata,
        previous_payload_sha256=evidence.payload_sha256,
    )
    different_result = json.loads(metadata.source_json)
    different_result["document_sets"] = ["different-access-set"]
    with pytest.raises(BadRequestError) as rejected:
        adapter.update_metadata(
            current,
            previous=projection,
            updated=FrozenPublicationProjection.model_validate(
                {**metadata.model_dump(), "source_json": json.dumps(different_result)}
            ),
            previous_payload_sha256=evidence.payload_sha256,
        )
    assert "different equal-token" in json.dumps(rejected.value.body)
    adapter.tombstone(current, 1_000_000_042)
    adapter.verify(current, (metadata,))
    actual = adapter.read_evidence(current, 0)
    assert json.loads(actual.source_json)["content_vector"] == [0.1, 0.2, 0.3]
    assert actual.frozen_projection is not None
    assert (
        actual.frozen_projection.embedding_inputs
        == evidence.frozen_projection.embedding_inputs
    )
    with pytest.raises(ValueError, match="identity"):
        adapter.update_metadata(
            current,
            previous=projection,
            updated=frozen_projection(owned_file, 0, "changed content"),
            previous_payload_sha256=evidence.payload_sha256,
        )


def test_finalization_rejects_new_reservations_after_verification(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    for ordinal in inventory.ordinals:
        adapter.tombstone(inventory, ordinal)
    proof = adapter.verify(inventory, ())
    authority.allocate(owner, "reserved-after-verification")
    with get_session_with_tenant(tenant_id="public") as session:
        with pytest.raises(ValueError, match="inventory"):
            authority.finalize(session, owner, proof)
    assert owned_file in authority.unavailable(authority.observe(), (owned_file,))


def test_bulk_publication_preserves_exact_fencing_retry_and_tombstones(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    reserved = authority.reservations(owner)
    adapter = adapter_for(es)
    projection = frozen_projection(owned_file, 0)
    adapter.publish_inventory(
        reserved, (projection,), before_batch=lambda: authority.reservations(owner)
    )
    assert adapter.verify(reserved, (projection,)).live_ordinals == (0,)
    adapter.publish_inventory(
        reserved, (projection,), before_batch=lambda: authority.reservations(owner)
    )
    assert adapter.verify(reserved, (projection,)).live_ordinals == (0,)
    with pytest.raises(ValueError, match="bulk"):
        adapter.publish_inventory(
            reserved,
            (frozen_projection(owned_file, 0, "unreviewed change"),),
            before_batch=lambda: authority.reservations(owner),
        )
    authority.release(owner)
    newer = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    current = authority.reservations(newer)
    adapter.publish_inventory(
        current, (), before_batch=lambda: authority.reservations(newer)
    )
    with pytest.raises(ValueError, match="bulk"):
        adapter.publish_inventory(reserved, (projection,), before_batch=lambda: None)
    assert adapter.verify(current, ()).live_ordinals == ()


def test_inventory_reads_scanned_vectors_without_per_record_gets(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    reserved = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(reserved)
    projection = frozen_projection(owned_file, 0)
    adapter.upsert(reserved, projection)
    adapter.tombstone(reserved, reserved.ordinals[-1])
    get = MagicMock(side_effect=AssertionError("inventory must use scanned evidence"))
    monkeypatch.setattr(es[0], "get", get)
    evidence = adapter.inventory_evidence(reserved)
    assert len(evidence) == 1
    frozen = evidence[0].frozen_projection
    assert frozen is not None
    assert frozen.context_projection_id == projection.context_projection_id
    assert frozen.embedding_inputs == projection.embedding_inputs
    assert json.loads(frozen.source_json) == json.loads(projection.source_json)
    assert json.loads(frozen.embedding_config_json) == json.loads(
        projection.embedding_config_json
    )
    get.assert_not_called()


def test_tenant_file_and_index_identity_cannot_be_adopted_from_other_scope(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import TenantState

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    with pytest.raises(ValueError, match="scope"):
        store("tenant_annex_local").acquire(
            owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30)
        )
    foreign = PublicationStore(
        authority.scope.model_copy(update={"environment": "other"})
    )
    with pytest.raises(ValueError, match="scope"):
        foreign.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    with pytest.raises(ValueError, match="scope"):
        foreign.unavailable(authority.observe(), (owned_file,))
    adapter = adapter_for(es)
    client, name = es
    identity = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False), str(owned_file), 0
    )
    client.index(
        index=name,
        id=identity,
        document={"document_id": str(uuid4()), "chunk_index": 0, "hidden": False},
    )
    with pytest.raises(BadRequestError):
        adapter.seal(inventory)
    with pytest.raises(ValueError, match="scope"):
        adapter.read_evidence(inventory, 0)
    with pytest.raises(ValueError, match="scope"):
        adapter.upsert(inventory, frozen_projection(uuid4(), 0))
    changed_index = FencedPublicationIndex(
        client, adapter.snapshot.model_copy(update={"index_uuid": "wrong-uuid"})
    )
    with pytest.raises(ValueError, match="identity"):
        changed_index.seal(inventory)
    changed_config = FencedPublicationIndex(
        client,
        adapter.snapshot.model_copy(update={"embedding_config_sha256": "a" * 64}),
    )
    with pytest.raises(ValueError, match="configuration"):
        changed_config.upsert(inventory, frozen_projection(owned_file, 0))


def test_single_tenant_index_refuses_other_tenant_authority(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    foreign_owner = owner.model_copy(
        update={"scope": owner.scope.model_copy(update={"tenant_id": "tenant_other"})}
    )
    foreign_inventory = inventory.model_copy(update={"ownership": foreign_owner})
    with pytest.raises(ValueError, match="tenant"):
        adapter_for(es).initialize(foreign_inventory)


@pytest.mark.parametrize(
    "changed_field", ["embedding_inputs", "embedding_config", "index"]
)
def test_actual_encoder_evidence_corruption_cannot_hide_behind_digest(
    owned_file: UUID, es: tuple[Elasticsearch, str], changed_field: str
) -> None:
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import (
        DocumentChunkVerificationError,
        TenantState,
    )

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    projection = frozen_projection(owned_file, 0)
    adapter.upsert(inventory, projection)
    adapter.tombstone(inventory, 1_000_000_042)
    evidence = adapter.read_evidence(inventory, 0)
    altered = json.loads(evidence.source_json)["publication_evidence"]
    altered[changed_field] = (
        ["wrong input"] if changed_field == "embedding_inputs" else {"model": "wrong"}
    )
    client, name = es
    identity = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False), str(owned_file), 0
    )
    client.update(index=name, id=identity, doc={"publication_evidence": altered})
    with pytest.raises(DocumentChunkVerificationError):
        adapter.verify(inventory, (projection,))
    with pytest.raises(DocumentChunkVerificationError):
        adapter.read_evidence(inventory, 0)


@pytest.mark.parametrize("corruption", ["extra", "missing"])
def test_verification_rejects_unmanifested_and_missing_projection_ids(
    owned_file: UUID, es: tuple[Elasticsearch, str], corruption: str
) -> None:
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import (
        DocumentChunkVerificationError,
        TenantState,
    )

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    for ordinal in inventory.ordinals:
        adapter.tombstone(inventory, ordinal)
    adapter.verify(inventory, ())
    client, name = es
    if corruption == "extra":
        client.index(
            index=name,
            id="unmanifested",
            document={
                "document_id": str(owned_file),
                "chunk_index": 444,
                "content": "unmanifested",
            },
        )
    else:
        identity = get_elasticsearch_doc_chunk_id(
            TenantState(tenant_id="public", multitenant=False), str(owned_file), 0
        )
        client.delete(index=name, id=identity)
    with pytest.raises(DocumentChunkVerificationError, match="missing/extra"):
        adapter.verify(inventory, ())


def test_full_projection_freeze_rejects_incomplete_search_source() -> None:
    projection = frozen_projection(uuid4(), 0)
    incomplete = json.loads(projection.source_json)
    del incomplete["blurb"]
    with pytest.raises(ValueError, match="required"):
        FrozenPublicationProjection.model_validate(
            {**projection.model_dump(), "source_json": json.dumps(incomplete)}
        )


def test_real_tenant_allocator_and_observation_are_separate_from_public() -> None:
    with create_owned_file("tenant_annex_local") as file_id:
        authority = store("tenant_annex_local")
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(seconds=30))
        assert authority.allocate(owner, "tenant-new-element") == 1_000_000_043
        authority.close_gate(owner)
        assert authority.unavailable(authority.observe(), (file_id,)) == {file_id}
        assert not store().unavailable(store().observe(), (file_id,))
        assert {row.user_file_id for row in authority.owned_chunks(owner)} == {file_id}
        with pytest.raises(ValueError, match="scope"):
            store().acquire(file_id, owner_id=uuid4(), ttl=timedelta(seconds=30))


@pytest.fixture(scope="module", autouse=True)
def cleanup_owned_clocks() -> Generator[None, None, None]:
    assert POSTGRES_HOST in ("localhost", "127.0.0.1") and str(POSTGRES_PORT) == "25432"
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    yield
    from onyx.db.models import RegulatoryPublicationClock

    for tenant_id in ("public", "tenant_annex_local"):
        with get_session_with_tenant(tenant_id=tenant_id) as session:
            session.execute(
                delete(RegulatoryPublicationClock).where(
                    RegulatoryPublicationClock.scope_key == store(tenant_id).scope_key
                )
            )
            session.commit()


def test_canonical_transaction_can_lock_owned_inventory_before_other_rows(
    owned_file: UUID,
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    with get_session_with_tenant(tenant_id="public") as session:
        locked = authority.lock_owned_snapshot(session, owner)
        assert locked.ordinals == (0, 1_000_000_042)
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(
                authority.allocate, owner, "concurrent-allocation"
            )
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
            finally:
                session.rollback()
            assert pending.result(timeout=5) == 1_000_000_043


@pytest.mark.parametrize(
    "mismatch",
    ["content", "vector", "context", "inputs", "config", "projection_identity"],
)
def test_metadata_actual_digest_cannot_authorize_a_different_prepared_base(
    owned_file: UUID, es: tuple[Elasticsearch, str], mismatch: str
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    stored = frozen_projection(owned_file, 0)
    adapter.upsert(inventory, stored)
    evidence = adapter.read_evidence(inventory, 0)
    assert evidence.payload_sha256 is not None
    authority.release(owner)
    newer = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    current = authority.reservations(newer)
    adapter.seal(current)
    prepared = stored.model_dump()
    source = json.loads(stored.source_json)
    metadata_adapter = adapter
    if mismatch == "content":
        source["content"] = "content that was never stored"
    elif mismatch == "vector":
        source["content_vector"] = [0.2, 0.3, 0.1]
    elif mismatch == "context":
        source["doc_summary"] = "context that was never stored"
    elif mismatch == "inputs":
        prepared["embedding_inputs"] = ("input that was never stored",)
    elif mismatch == "projection_identity":
        prepared["context_projection_id"] = "other-context-projection"
    else:
        config = {"model": "different-prepared-model"}
        prepared["embedding_config_json"] = json.dumps(config)
        metadata_adapter = FencedPublicationIndex(
            es[0],
            adapter.snapshot.model_copy(
                update={"embedding_config_sha256": publication_digest(config)}
            ),
        )
    prepared["source_json"] = json.dumps(source)
    previous = FrozenPublicationProjection.model_validate(prepared)
    source["document_sets"] = ["new-access-set"]
    updated = FrozenPublicationProjection.model_validate(
        {**prepared, "source_json": json.dumps(source)}
    )
    with pytest.raises(BadRequestError) as rejected:
        metadata_adapter.update_metadata(
            current,
            previous=previous,
            updated=updated,
            previous_payload_sha256=evidence.payload_sha256,
        )
    assert "metadata base" in json.dumps(rejected.value.body)
    after = adapter.read_evidence(current, 0)
    assert after.payload_sha256 == evidence.payload_sha256
    assert after.frozen_projection == evidence.frozen_projection
    # Rejected metadata must not consume the token's final operation slot.
    adapter.upsert(current, stored)
    adapter.tombstone(current, 1_000_000_042)
    adapter.verify(current, (stored,))
