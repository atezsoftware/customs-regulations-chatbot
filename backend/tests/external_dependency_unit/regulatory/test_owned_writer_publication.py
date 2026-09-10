"""Owned legacy publication recovery uses real PG and Elasticsearch fencing."""

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from elasticsearch import BadRequestError, Elasticsearch
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.regulatory_writer_publication import (
    canonical_scope_digest,
    pending_writer_manifest,
)
from onyx.regulatory.writer_publication import execute_writer_publication
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    LiveReview,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    live_review as live_review,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    adapter_for,
    frozen_projection,
    store,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)


def test_owned_delete_resumes_after_es_before_db_and_seals_unseen_reserved_ids(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db import search_settings as settings_repository
    from onyx.regulatory.writer_publication import complete_writer_index_inventory

    monkeypatch.setattr(
        settings_repository, "get_active_search_settings_list", lambda _: []
    )
    authority = store()
    old = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    unseen = authority.allocate(old, "abandoned-larger-job")
    authority.release(old)
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    adapter = adapter_for(es)
    with get_session_with_tenant(tenant_id="public") as session:
        before = canonical_scope_digest(session, owned_file)
    manifest = WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owned_file,
        kind="delete",
        canonical_before_sha256=before,
        indexes=[adapter.snapshot],
        previous_binding_ids=[],
        bindings=[],
    )
    manifest = complete_writer_index_inventory(owner, es[0], manifest)
    from onyx.regulatory import writer_publication

    real_finalize = writer_publication.finalize_writer_publication

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("process failed after ES verification")

    monkeypatch.setattr(writer_publication, "finalize_writer_publication", interrupted)
    with pytest.raises(RuntimeError, match="process failed"):
        execute_writer_publication(owner, es[0], manifest)
    assert authority.reservations(owner).gate_closed
    assert pending_writer_manifest(owner) == manifest
    authority.release(owner)
    recovery = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    monkeypatch.setattr(
        writer_publication, "finalize_writer_publication", real_finalize
    )
    execute_writer_publication(recovery, es[0])
    assert pending_writer_manifest(recovery) == manifest
    assert authority.reservations(recovery).gate_closed
    # A previously planned first create for an invisible abandoned ordinal cannot revive it.
    from onyx.document_index.publication_models import FileReservations

    stale = FileReservations(ownership=old, ordinals=(unseen,), gate_closed=True)
    with pytest.raises(BadRequestError) as stale_error:
        adapter.upsert(stale, frozen_projection(owned_file, unseen))
    assert "stale or unsealed publication ownership" in str(stale_error.value.body)
    assert adapter.inventory_evidence(authority.reservations(recovery)) == ()
    authority.release(recovery)


@pytest.mark.parametrize("live_review", ["verified"], indirect=True)
def test_owned_metadata_replaces_qualified_source_and_retains_approved_binding(
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        load_file_temporal_bindings,
    )
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    prepared = read_publication_preparation(draft.publication)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    for setting in settings:
        monkeypatch.setattr(
            DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            ).embedding_model,
            "encode",
            lambda *, texts, **_: [[0.25, 0.5, 0.75] for _ in texts],
        )
    assert invoke(approve(source_session, live_review)) == "approved"
    source_session.expire_all()
    before = load_file_temporal_bindings(source_session, live_review.file.id)
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.interfaces_new import MetadataUpdateRequest
    from onyx.regulatory import writer_publication

    assert hasattr(writer_publication, "publish_owned_metadata"), (
        "legacy metadata has no owned qualified representation transition"
    )
    authority = PublicationStore(prepared.scope)
    owner = authority.acquire(
        live_review.file.id, owner_id=uuid4(), ttl=timedelta(minutes=2)
    )
    writer_publication.publish_owned_metadata(
        owner,
        es[0],
        MetadataUpdateRequest(
            document_ids=[str(live_review.file.id)],
            doc_id_to_chunk_cnt={str(live_review.file.id): len(before)},
            project_ids={513},
        ),
        index_names=[index.index_name for index in prepared.indexes],
    )
    source_session.expire_all()
    after = load_file_temporal_bindings(source_session, live_review.file.id)
    assert len(after) == len(before)
    import json

    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )

    for prior in before:
        retained = source_session.get(RegulatoryTemporalProjection, prior.id)
        assert retained is not None and retained.retired_at is not None
        assert retained.payload == prior.model_dump(mode="json")
        validate_temporal_canonical_revision(source_session, retained)
        current = next(
            binding
            for binding in after
            if binding.index.index_uuid == prior.index.index_uuid
            and binding.projection.ordinal == prior.projection.ordinal
        )
        assert current.id != prior.id
        assert current.projection.embedding_inputs == prior.projection.embedding_inputs
        assert (
            current.projection.embedding_config_json
            == prior.projection.embedding_config_json
        )
        assert (
            current.effective_start == prior.effective_start
            and current.effective_end == prior.effective_end
        )
        assert json.loads(current.projection.source_json)["user_projects"] == [513]
        assert (
            json.loads(current.projection.source_json)["content_vector"]
            == json.loads(prior.projection.source_json)["content_vector"]
        )
    assert not authority.unavailable(authority.observe(), (live_review.file.id,))
    authority.release(owner)

    # Exercise the actual project/persona/document-set sync caller after publication.
    from onyx.background.celery.tasks.user_file_processing.tasks import (
        project_sync_user_file_impl,
    )
    from onyx.db import search_settings as settings_repository
    from onyx.db.enums import UserFileStatus

    live_review.file.status = UserFileStatus.COMPLETED
    live_review.file.needs_document_set_sync = True
    source_session.commit()
    monkeypatch.setattr(
        settings_repository, "get_active_search_settings_list", lambda _: settings
    )
    project_sync_user_file_impl(
        user_file_id=str(live_review.file.id),
        tenant_id=prepared.scope.tenant_id,
        redis_locking=False,
    )
    source_session.expire_all()
    assert not live_review.file.needs_document_set_sync
    synchronized = load_file_temporal_bindings(source_session, live_review.file.id)
    assert all(
        json.loads(binding.projection.source_json)["user_projects"] == []
        for binding in synchronized
    )
    assert {binding.id for binding in synchronized}.isdisjoint(
        {binding.id for binding in after}
    )
    # A failed older job does not revoke newer qualified history or its metadata.
    live_review.file.status = UserFileStatus.FAILED
    live_review.file.needs_document_set_sync = True
    source_session.commit()
    finalize = writer_publication.finalize_writer_publication

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("metadata ES completed before process termination")

    monkeypatch.setattr(writer_publication, "finalize_writer_publication", interrupted)
    with pytest.raises(RuntimeError, match="metadata ES completed"):
        project_sync_user_file_impl(
            user_file_id=str(live_review.file.id),
            tenant_id=prepared.scope.tenant_id,
            redis_locking=False,
        )
    monkeypatch.setattr(writer_publication, "finalize_writer_publication", finalize)
    project_sync_user_file_impl(
        user_file_id=str(live_review.file.id),
        tenant_id=prepared.scope.tenant_id,
        redis_locking=False,
    )
    source_session.expire_all()
    assert not live_review.file.needs_document_set_sync
    assert live_review.file.status == UserFileStatus.FAILED
    preserved = load_file_temporal_bindings(source_session, live_review.file.id)
    assert {
        (item.representation_text, item.effective_start, item.effective_end)
        for item in preserved
    } == {
        (item.representation_text, item.effective_start, item.effective_end)
        for item in synchronized
    }
    from unittest.mock import MagicMock

    from onyx.background.celery.tasks.user_file_processing import tasks as file_tasks

    live_review.file.secondary_reconcile_pending = True
    source_session.commit()
    queued: list[str] = []
    monkeypatch.setattr(file_tasks, "get_redis_client", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(
        file_tasks, "get_user_file_project_sync_queue_depth", lambda _app: 0
    )

    def enqueue(**kwargs: object) -> bool:
        queued.append(str(kwargs["user_file_id"]))
        return True

    monkeypatch.setattr(file_tasks, "enqueue_user_file_project_sync_task", enqueue)
    file_tasks.check_for_user_file_project_sync.run(tenant_id=prepared.scope.tenant_id)
    assert str(live_review.file.id) in queued, (
        "FAILED qualified history reconciliation must be delivered by the actual scanner"
    )


@pytest.mark.parametrize("live_review", ["temporary"], indirect=True)
def test_owned_correction_preserves_predecessor_successor_windows_and_approval(
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        load_file_temporal_bindings,
    )
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    prepared = read_publication_preparation(draft.publication)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    for setting in settings:
        monkeypatch.setattr(
            DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            ).embedding_model,
            "encode",
            lambda *, texts, **_: [[0.25, 0.5, 0.75] for _ in texts],
        )
    assert invoke(approve(source_session, live_review)) == "approved"
    source_session.expire_all()
    before = load_file_temporal_bindings(source_session, live_review.file.id)
    from onyx.regulatory import writer_publication

    assert hasattr(writer_publication, "correct_owned_chunk"), (
        "direct correction has no dated owned canonical publication"
    )
    from onyx.db import search_settings as settings_repository
    from onyx.db.regulatory_publication import PublicationStore

    monkeypatch.setattr(
        settings_repository, "get_active_search_settings_list", lambda _: settings
    )
    authority = PublicationStore(prepared.scope)
    owner = authority.acquire(
        live_review.file.id, owner_id=uuid4(), ttl=timedelta(minutes=2)
    )
    target = next(
        row
        for row in prepared.legal.canonical_rows
        if row.source == "amendment"
        and row.validity_start_date == prepared.legal.effective_start
    )
    writer_publication.correct_owned_chunk(
        owner,
        es[0],
        target.id,
        text="Administratively corrected temporary annex provision.",
    )
    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )

    source_session.expire_all()
    after_rows = capture_canonical_scope(source_session, live_review.file.id)
    after = load_file_temporal_bindings(source_session, live_review.file.id)
    corrected = next(row for row in after_rows if row.id == target.id)
    assert corrected.text == "Administratively corrected temporary annex provision."
    assert corrected.projection_ordinal == target.projection_ordinal
    assert corrected.validity_start_date == target.validity_start_date
    assert corrected.validity_end_date == target.validity_end_date
    assert [row for row in after_rows if row.id != target.id] == [
        row for row in prepared.legal.canonical_rows if row.id != target.id
    ]
    for previous in before:
        retained = source_session.get(RegulatoryTemporalProjection, previous.id)
        assert retained is not None and retained.payload == previous.model_dump(
            mode="json"
        )
        validate_temporal_canonical_revision(source_session, retained)
    assert read_publication_preparation(draft.publication) == prepared
    assert not authority.unavailable(authority.observe(), (live_review.file.id,))
    assert any(binding.representation_text == corrected.text for binding in after)
    authority.release(owner)
    unaffected = [
        binding
        for binding in before
        if (
            binding.effective_end is not None
            and target.validity_start_date is not None
            and binding.effective_end <= target.validity_start_date
        )
        or (
            target.validity_end_date is not None
            and binding.effective_start is not None
            and binding.effective_start >= target.validity_end_date
        )
    ]
    assert unaffected and all(binding in after for binding in unaffected)

    from onyx.db.models import User
    from onyx.server.features.regulatory import api
    from onyx.server.features.regulatory.models import RegulatoryChunkUpdateRequest

    def reject_legacy_projection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PATCH used the unfenced canonical projection")

    monkeypatch.setattr(
        api, "project_user_file_to_index", reject_legacy_projection, raising=False
    )
    user = source_session.get(User, live_review.file.user_id)
    assert user is not None
    response = api.patch_chunk(
        target.id,
        RegulatoryChunkUpdateRequest(text="Second administrative correction."),
        user=user,
        db_session=source_session,
    )
    assert (
        response.id == target.id
        and response.text == "Second administrative correction."
    )
    source_session.expire_all()
    for previous in before:
        retained = source_session.get(RegulatoryTemporalProjection, previous.id)
        assert retained is not None
        validate_temporal_canonical_revision(source_session, retained)

    from onyx.db.enums import UserFileStatus
    from onyx.regulatory import projection as legacy_projection

    live_review.file.status = UserFileStatus.COMPLETED
    source_session.commit()
    before_reindex = load_file_temporal_bindings(source_session, live_review.file.id)
    monkeypatch.setattr(
        legacy_projection,
        "_project_rows_to_search_settings",
        reject_legacy_projection,
        raising=False,
    )
    count = legacy_projection.project_user_file_to_index(
        source_session, live_review.file, prepared.scope.tenant_id
    )
    assert count == len(after_rows)
    source_session.expire_all()
    republished = load_file_temporal_bindings(source_session, live_review.file.id)
    assert {binding.id for binding in republished} == {
        binding.id for binding in before_reindex
    }
    assert any(
        binding.representation_text == "Second administrative correction."
        for binding in republished
    )
    assert read_publication_preparation(draft.publication) == prepared

    from onyx.server.features.regulatory.models import UserFileRenameRequest

    renamed = api.rename_user_file(
        live_review.file.id,
        UserFileRenameRequest(name="Renamed Regulation"),
        user=user,
        db_session=source_session,
    )
    assert renamed.name == "Renamed Regulation"
    source_session.expire_all()
    after_rename = load_file_temporal_bindings(source_session, live_review.file.id)
    assert {
        (item.effective_start, item.effective_end, item.projection.ordinal)
        for item in after_rename
    } == {
        (item.effective_start, item.effective_end, item.projection.ordinal)
        for item in republished
    }
    import json

    assert all(
        json.loads(item.projection.source_json)["title"] == "Renamed Regulation"
        for item in after_rename
    )
    assert read_publication_preparation(draft.publication) == prepared

    from onyx.background.celery.tasks.user_file_processing import tasks as file_tasks
    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection

    monkeypatch.setattr(
        file_tasks, "run_indexing_pipeline", reject_legacy_projection, raising=False
    )
    file_tasks._process_user_file_with_indexing(
        str(live_review.file.id),
        [
            Document(
                id=str(live_review.file.id),
                sections=[
                    TextSection(
                        text="Original bytes must not replace corrected canonical history.",
                        link="",
                    )
                ],
                source=DocumentSource.FILE,
                metadata={},
                semantic_identifier="Original Regulation",
            )
        ],
        prepared.scope.tenant_id,
    )
    source_session.expire_all()
    assert any(
        row.id == target.id and row.text == "Second administrative correction."
        for row in capture_canonical_scope(source_session, live_review.file.id)
    )
    assert {
        item.id
        for item in load_file_temporal_bindings(source_session, live_review.file.id)
    } == {item.id for item in after_rename}

    from onyx.db.enums import IndexModelStatus
    from onyx.db.models import SearchSettings

    future_name = es[1] + "-owned-future"
    es[0].indices.create(
        index=future_name,
        mappings=es[0].indices.get_mapping(index=es[1])[es[1]]["mappings"],
    )
    future = SearchSettings(
        id=998877,
        status=IndexModelStatus.FUTURE,
        index_name=future_name,
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    monkeypatch.setattr(
        settings_repository,
        "get_active_search_settings_list",
        lambda _: [*settings, future],
    )
    try:
        # A completed PRESENT writer still has an already planned ES request in flight.
        from onyx.document_index.elasticsearch.publication import FencedPublicationIndex

        old = authority.acquire(
            live_review.file.id, owner_id=uuid4(), ttl=timedelta(minutes=2)
        )
        authority.close_gate(old)
        stale_present = authority.reservations(old)
        present_adapter = FencedPublicationIndex(es[0], after_rename[0].index)
        present_adapter.seal(stale_present)
        live_ordinals = {binding.projection.ordinal for binding in after_rename}
        for ordinal in stale_present.ordinals:
            if ordinal not in live_ordinals:
                present_adapter.tombstone(stale_present, ordinal)
        for binding in after_rename:
            present_adapter.upsert(stale_present, binding.projection)
        proof = present_adapter.verify(
            stale_present, tuple(binding.projection for binding in after_rename)
        )
        with get_session_with_tenant(tenant_id=prepared.scope.tenant_id) as session:
            authority.finalize(session, old, proof)
            session.commit()
        authority.release(old)
        assert file_tasks._index_user_file_to_secondary(
            live_review.file.id, future, prepared.scope.tenant_id
        )
        with pytest.raises(BadRequestError):
            present_adapter.upsert(stale_present, after_rename[0].projection)
        source_session.expire_all()
        reconciled = load_file_temporal_bindings(source_session, live_review.file.id)
        current_bindings = [
            item for item in reconciled if item.index.index_name == es[1]
        ]
        future_bindings = [
            item for item in reconciled if item.index.index_name == future_name
        ]
        assert current_bindings == after_rename
        assert {
            (item.representation_text, item.effective_start, item.effective_end)
            for item in future_bindings
        } == {
            (item.representation_text, item.effective_start, item.effective_end)
            for item in current_bindings
        }
    finally:
        es[0].indices.delete(index=future_name)


def test_initial_chunking_reserves_after_abandoned_projection_identity(
    owned_file: UUID,
) -> None:
    from sqlalchemy import delete

    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.indexing import documents_to_regulatory_chunks
    from tests.external_dependency_unit.regulatory.test_durable_indexing_pipeline import (
        _CharacterTokenizer,
    )

    with get_session_with_tenant(tenant_id="public") as session:
        session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == owned_file)
        )
        session.commit()
    authority = store()
    old = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    abandoned = authority.allocate(old, "abandoned-first-create")
    authority.release(old)
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    with get_session_with_tenant(tenant_id="public") as session:
        chunks = documents_to_regulatory_chunks(
            [
                Document(
                    id=str(owned_file),
                    sections=[TextSection(text="MADDE 1\nInitial provision.", link="")],
                    source=DocumentSource.FILE,
                    metadata={},
                    semantic_identifier="Regulation",
                )
            ],
            session,
            _CharacterTokenizer(),
            enable_contextual_rag=False,
            publication_owner=owner,
        )
        session.commit()
    rows = authority.owned_chunks(owner)
    assert rows and len(chunks) == len(rows)
    assert all(row.projection_ordinal > abandoned for row in rows)
    assert set(row.projection_ordinal for row in rows).issubset(
        authority.reservations(owner).ordinals
    )
    assert all(
        chunk.chunk_id == row.projection_ordinal for chunk, row in zip(chunks, rows)
    )
    authority.release(owner)


@pytest.mark.parametrize("after_interrupt", ["resume", "delete_request"])
def test_initial_ingestion_resumes_frozen_owned_publication(
    owned_file: UUID,
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
    after_interrupt: str,
) -> None:
    from sqlalchemy import delete

    from onyx.background.celery.tasks.user_file_processing import tasks as file_tasks
    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import RegulatoryChunk, UserFile
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        load_file_temporal_bindings,
    )
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import PublicationScope
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory import writer_publication
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=settings[0]
    ).embedding_model
    encoded: list[list[str]] = []

    def encode(*, texts: list[str], **_kwargs: object) -> list[list[float]]:
        encoded.append(list(texts))
        return [[0.25, 0.5, 0.75] for _ in texts]

    monkeypatch.setattr(model, "encode", encode)
    monkeypatch.setattr(file_tasks, "store_user_file_plaintext", lambda **_kwargs: None)
    with get_session_with_tenant(tenant_id="public") as session:
        session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == owned_file)
        )
        file = session.get(UserFile, owned_file)
        assert file is not None
        file.status = UserFileStatus.PROCESSING
        session.commit()
    docs = [
        Document(
            id=str(owned_file),
            sections=[TextSection(text="MADDE 1\nInitial provision.", link="")],
            source=DocumentSource.FILE,
            metadata={},
            semantic_identifier="Regulation",
        )
    ]
    real_finalize = writer_publication.finalize_writer_publication

    def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("interrupted initial activation")

    monkeypatch.setattr(writer_publication, "finalize_writer_publication", crash)
    with pytest.raises(RuntimeError, match="interrupted initial activation"):
        file_tasks._process_user_file_with_indexing(str(owned_file), docs, "public")
    assert encoded
    first_encoded = list(encoded)
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    assert owned_file in authority.unavailable(authority.observe(), (owned_file,))
    monkeypatch.setattr(
        writer_publication, "finalize_writer_publication", real_finalize
    )
    if after_interrupt == "delete_request":
        writer_publication.request_owned_file_deletion(owned_file, "public")
        owner = authority.acquire(
            owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2)
        )
        execute_writer_publication(owner, es[0])
        authority.release(owner)
        source_session.expire_all()
        retained = source_session.get(UserFile, owned_file)
        assert retained is not None and retained.status == UserFileStatus.DELETING, (
            "recovering the earlier ingestion must not lose an accepted deletion request"
        )
        return
    file_tasks._process_user_file_with_indexing(str(owned_file), docs, "public")
    assert encoded == first_encoded
    source_session.expire_all()
    bindings = load_file_temporal_bindings(source_session, owned_file)
    assert bindings and all(
        item.representation_text and item.index.index_name == es[1] for item in bindings
    )
    assert not authority.unavailable(authority.observe(), (owned_file,))
    file = source_session.get(UserFile, owned_file)
    assert file is not None and file.status == UserFileStatus.COMPLETED

    from datetime import date

    from onyx.db.models import User
    from onyx.server.features.regulatory import api
    from onyx.server.features.regulatory.models import (
        RegulatoryFileValidityUpdateRequest,
    )

    user = source_session.get(User, file.user_id)
    assert user is not None
    previous = list(bindings)

    def reject_unfenced_validity(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validity used the unfenced metadata patch")

    monkeypatch.setattr(
        api,
        "patch_user_file_validity_in_active_indices",
        reject_unfenced_validity,
        raising=False,
    )
    result = api.patch_file_validity(
        owned_file,
        RegulatoryFileValidityUpdateRequest(
            validity_start_date=date(2020, 1, 1), validity_end_date=date(2030, 1, 1)
        ),
        user=user,
        db_session=source_session,
    )
    assert result.updated_chunk_count == len(previous)
    source_session.expire_all()
    updated = load_file_temporal_bindings(source_session, owned_file)
    assert updated and all(
        item.effective_start == date(2020, 1, 1)
        and item.effective_end == date(2030, 1, 1)
        for item in updated
    )
    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )

    for binding in previous:
        retained = source_session.get(RegulatoryTemporalProjection, binding.id)
        assert retained is not None and retained.payload == binding.model_dump(
            mode="json"
        )
        validate_temporal_canonical_revision(source_session, retained)


def test_delete_caller_retains_approved_annex_evidence(
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.tasks.user_file_processing import tasks as file_tasks
    from onyx.db.models import AnnexChangeSet, User, UserFile
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.server.features.regulatory import annex_api
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=settings[0]
    ).embedding_model
    monkeypatch.setattr(
        model, "encode", lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts]
    )
    assert invoke(approve(source_session, live_review)) == "approved"
    identifier, review_id, batch_id = (
        live_review.file.id,
        live_review.review.id,
        live_review.batch.id,
    )
    user = source_session.get(User, live_review.file.user_id)
    assert user is not None
    from onyx.auth.schemas import UserRole

    user.role = UserRole.ADMIN
    source_session.commit()
    content = {
        evidence.id: annex_api.get_evidence(
            batch_id, review_id, evidence.id, user=user, db_session=source_session
        ).body
        for evidence in draft.evidence
    }
    assert content

    def reject_legacy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("deletion used unfenced legacy ES cleanup")

    monkeypatch.setattr(file_tasks, "get_all_document_indices", reject_legacy)
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import PublicationScope
    from onyx.file_store.postgres_file_store import PostgresBackedFileStore
    from onyx.regulatory.amendments.annexes import config

    original_id = live_review.file.file_id
    real_delete = PostgresBackedFileStore.delete_file
    interrupted = False

    def delete_with_failure(
        self: PostgresBackedFileStore,
        file_id: str,
        error_on_missing: bool = True,
        db_session: Session | None = None,
    ) -> None:
        nonlocal interrupted
        if file_id == original_id and not interrupted:
            interrupted = True
            raise RuntimeError("storage cleanup interrupted")
        real_delete(
            self, file_id, error_on_missing=error_on_missing, db_session=db_session
        )

    monkeypatch.setattr(PostgresBackedFileStore, "delete_file", delete_with_failure)
    with pytest.raises(RuntimeError, match="storage cleanup interrupted"):
        file_tasks.delete_user_file_impl(
            user_file_id=str(identifier), tenant_id="public", redis_locking=False
        )
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    assert identifier in authority.unavailable(authority.observe(), (identifier,))
    file_tasks.delete_user_file_impl(
        user_file_id=str(identifier), tenant_id="public", redis_locking=False
    )
    source_session.expire_all()
    assert source_session.get(UserFile, identifier) is None
    retained = source_session.get(AnnexChangeSet, review_id)
    assert (
        retained is not None
        and retained.user_file_id == identifier
        and retained.status == "approved"
    )
    assert {
        evidence_id: annex_api.get_evidence(
            batch_id, review_id, evidence_id, user=user, db_session=source_session
        ).body
        for evidence_id in content
    } == content
    file_tasks.delete_user_file_impl(
        user_file_id=str(identifier), tenant_id="public", redis_locking=False
    )

    assert (
        es[0].count(
            index=es[1],
            query={
                "bool": {
                    "filter": [{"term": {"document_id": str(identifier)}}],
                    "must_not": [{"term": {"publication_tombstone": True}}],
                }
            },
        )["count"]
        == 0
    )


def test_ordinary_amendment_after_annex_uses_owned_full_history(
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    from onyx.background.celery.tasks.regulatory_amendments import (
        tasks as amendment_tasks,
    )
    from onyx.db.models import AmendmentProposal, RegulatoryChunk
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        load_file_temporal_bindings,
    )
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=settings[0]
    ).embedding_model
    monkeypatch.setattr(
        model, "encode", lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts]
    )
    assert invoke(approve(source_session, live_review)) == "approved"
    source_session.expire_all()
    before = load_file_temporal_bindings(source_session, live_review.file.id)
    old = source_session.get(RegulatoryChunk, live_review.outside.id)
    assert old is not None
    old_id = old.id
    proposal = AmendmentProposal(
        batch_id=live_review.batch.id,
        instruction_index=2,
        instruction_text="Madde 2 updated",
        instruction_indices=[2],
        instruction_texts=["Madde 2 updated"],
        old_chunk_id=old.id,
        old_chunk_snapshot={"id": old.id, "text": old.text},
        new_chunk_draft={
            "user_file_id": str(old.user_file_id),
            "position": old.position,
            "text": "Ordinary amended outside provision.",
            "chunk_type": old.chunk_type,
            "heading_path": old.heading_path,
            "metadata": old.chunk_metadata,
            "effective_start_date": "2026-10-01",
            "effective_end_date": None,
        },
        status="approving",
    )
    source_session.add(proposal)
    source_session.commit()
    proposal_id, identifier = proposal.id, live_review.file.id

    def reject_unfenced(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ordinary amendment used the unfenced projection")

    monkeypatch.setattr(
        amendment_tasks, "project_amendment_to_index", reject_unfenced, raising=False
    )
    monkeypatch.setattr(
        amendment_tasks,
        "validate_amendment_projection_search_settings",
        lambda *_args, **_kwargs: settings[0].id,
    )
    from onyx.regulatory import writer_publication

    real_finalize = writer_publication.finalize_writer_publication

    def interrupt_activation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("amendment interrupted before activation")

    monkeypatch.setattr(
        writer_publication, "finalize_writer_publication", interrupt_activation
    )
    with pytest.raises(RuntimeError, match="amendment interrupted before activation"):
        amendment_tasks.regulatory_amendment_approve(
            proposal_id=proposal_id, tenant_id="public"
        )
    source_session.expire_all()
    pending_proposal = source_session.get(AmendmentProposal, proposal_id)
    unchanged_old = source_session.get(RegulatoryChunk, old_id)
    assert (
        pending_proposal is not None
        and pending_proposal.status == "approving"
        and pending_proposal.applied_new_chunk_id is None
    )
    assert (
        unchanged_old is not None
        and unchanged_old.status == "active"
        and unchanged_old.validity_end_date is None
    )
    monkeypatch.setattr(
        writer_publication, "finalize_writer_publication", real_finalize
    )
    amendment_tasks.regulatory_amendment_approve(
        proposal_id=proposal_id, tenant_id="public"
    )
    source_session.expire_all()
    proposal = source_session.get(AmendmentProposal, proposal_id)
    assert proposal is not None and proposal.status == "approved"
    new = source_session.get(RegulatoryChunk, proposal.applied_new_chunk_id)
    old = source_session.get(RegulatoryChunk, old_id)
    assert new is not None and old is not None
    assert (
        new.text == "Ordinary amended outside provision."
        and new.validity_start_date == date(2026, 10, 1)
    )
    assert (
        old.validity_end_date == date(2026, 10, 1)
        and old.superseded_by_chunk_id == new.id
    )
    after = load_file_temporal_bindings(source_session, identifier)
    assert any(
        item.index.index_name == es[1] and item.representation_text == new.text
        for item in after
    )
    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )

    for binding in before:
        retained = source_session.get(RegulatoryTemporalProjection, binding.id)
        assert retained is not None and retained.payload == binding.model_dump(
            mode="json"
        )
        validate_temporal_canonical_revision(source_session, retained)


def test_durable_preparation_retains_every_approved_context_window(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from sqlalchemy import select

    from onyx.db import search_settings as settings_repository
    from onyx.db.models import (
        RegulatoryChunk,
        RegulatoryIndexingItem,
        RegulatoryIndexingJob,
    )
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        load_file_temporal_bindings,
    )
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.indexing_jobs import preparation
    from onyx.regulatory.indexing_jobs.models import RegulatoryInputHashVersion
    from onyx.regulatory.indexing_jobs.projection_identity import projection_input
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_publisher import (
        _snapshot as job_configuration,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    for setting in settings:
        monkeypatch.setattr(
            DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            ).embedding_model,
            "encode",
            lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts],
        )
    assert invoke(approve(source_session, live_review)) == "approved"
    monkeypatch.setattr(
        settings_repository, "get_active_search_settings_list", lambda _: settings
    )
    monkeypatch.setattr(
        preparation, "get_tokenizer", lambda *_args, **_kwargs: _CharacterTokenizer()
    )
    monkeypatch.setattr(
        preparation,
        "get_contextual_token_budget_tokenizer",
        lambda *_args, **_kwargs: _CharacterTokenizer(),
    )
    source_session.expire_all()
    before = load_file_temporal_bindings(source_session, live_review.file.id)
    rows = list(
        source_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == live_review.file.id
            )
        )
    )
    snapshot = job_configuration().model_copy(
        update={
            "input_hash_version": RegulatoryInputHashVersion.CHUNK_ROWS_V3,
            "input_content_hash": preparation.regulatory_chunks_content_hash(rows),
            "search_settings_id": settings[0].id,
            "index_name": settings[0].index_name,
        }
    )
    live_review.file.regulatory_chunk_generation_hash = snapshot.chunk_generation_hash
    job = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=live_review.file.id,
        content_hash=snapshot.input_content_hash,
        chunk_generation_hash=snapshot.chunk_generation_hash,
        search_settings_id=snapshot.search_settings_id,
        prompt_hash=snapshot.prompt_hash,
        config_snapshot=snapshot.model_dump(mode="json"),
        status="RUNNING",
        stage="PREPARING",
        lease_generation=1,
    )
    source_session.add(job)
    source_session.commit()
    assert (
        preparation.prepare_claimed_regulatory_indexing_job_from_chunks(
            job_id=job.id,
            expected_generation=1,
            tenant_id="public",
            db_session=source_session,
        )
        == job.id
    )
    source_session.expire_all()
    items = list(
        source_session.scalars(
            select(RegulatoryIndexingItem).where(
                RegulatoryIndexingItem.job_id == job.id
            )
        )
    )
    assert len(items) > len(rows)
    assert all(item.projection_id is not None for item in items)
    assert {item.regulatory_chunk_id for item in items} == {row.id for row in rows}
    assert {
        (item.regulatory_chunk_id, item.effective_start, item.effective_end)
        for item in items
    } == {
        (
            json.loads(binding.projection.source_json)["regulatory_chunk_id"],
            binding.effective_start,
            binding.effective_end,
        )
        for binding in before
    }
    for item in items:
        projection = projection_input(item)
        assert projection is not None and projection.canonical_revision_id is not None
    assert load_file_temporal_bindings(source_session, live_review.file.id) == before

    # Reprocessing original bytes must keep the approved canonical/history snapshot.
    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection

    documents = [
        Document(
            id=str(live_review.file.id),
            sections=[
                TextSection(link="", text="MADDE 1\nOriginal pre-amendment bytes")
            ],
            source=DocumentSource.FILE,
            metadata={},
            semantic_identifier="Regulation",
        )
    ]
    raw_hash = preparation.regulatory_documents_content_hash(
        documents, RegulatoryInputHashVersion.CANONICAL_V2
    )
    job.status = "SUCCEEDED"
    source_session.commit()
    raw_snapshot = snapshot.model_copy(
        update={
            "input_hash_version": RegulatoryInputHashVersion.CANONICAL_V2,
            "input_content_hash": raw_hash,
        }
    )
    raw_job = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=live_review.file.id,
        content_hash=raw_hash,
        chunk_generation_hash=raw_snapshot.chunk_generation_hash,
        search_settings_id=raw_snapshot.search_settings_id,
        prompt_hash=raw_snapshot.prompt_hash,
        config_snapshot=raw_snapshot.model_dump(mode="json"),
        status="RUNNING",
        stage="PREPARING",
        lease_generation=1,
    )
    source_session.add(raw_job)
    source_session.commit()

    def forbidden_rechunk(*_args, **_kwargs):
        raise AssertionError("durable raw reindex replaced approved canonical history")

    monkeypatch.setattr(
        "onyx.regulatory.indexing.documents_to_regulatory_chunks", forbidden_rechunk
    )
    assert (
        preparation.prepare_claimed_regulatory_indexing_job(
            job_id=raw_job.id,
            expected_generation=1,
            documents=documents,
            tenant_id="public",
            db_session=source_session,
        )
        == raw_job.id
    )
    raw_items = list(
        source_session.scalars(
            select(RegulatoryIndexingItem).where(
                RegulatoryIndexingItem.job_id == raw_job.id
            )
        )
    )
    assert {
        (item.regulatory_chunk_id, item.effective_start, item.effective_end)
        for item in raw_items
    } == {
        (item.regulatory_chunk_id, item.effective_start, item.effective_end)
        for item in items
    }
    assert load_file_temporal_bindings(source_session, live_review.file.id) == before


def test_failed_legacy_metadata_cleanup_adopts_sparse_index_inventory(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from onyx.background.celery.tasks.user_file_processing.tasks import (
        project_sync_user_file_impl,
    )
    from onyx.db import search_settings as settings_repository
    from onyx.db.enums import IndexModelStatus, UserFileStatus
    from onyx.db.models import SearchSettings, UserFile
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.elasticsearch.elasticsearch_document_index import (
        ElasticsearchDocumentIndex,
    )
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import TenantState
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config

    setting = SearchSettings(
        id=998877,
        status=IndexModelStatus.PRESENT,
        index_name=es[1],
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    monkeypatch.setattr(
        settings_repository, "get_active_search_settings_list", lambda _: [setting]
    )

    def reject_unfenced(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "failed metadata cleanup used unconditional whole-file deletion"
        )

    monkeypatch.setattr(ElasticsearchDocumentIndex, "delete", reject_unfenced)
    stale_ordinal = 52
    document_id = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False),
        str(owned_file),
        stale_ordinal,
    )
    es[0].index(
        index=es[1],
        id=document_id,
        document=json.loads(frozen_projection(owned_file, stale_ordinal).source_json),
        refresh=True,
    )
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None
        file.status = UserFileStatus.FAILED
        file.needs_document_set_sync = True
        session.commit()
    project_sync_user_file_impl(
        user_file_id=str(owned_file), tenant_id="public", redis_locking=False
    )
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None and file.status == UserFileStatus.FAILED
        assert not file.needs_document_set_sync
    stored = es[0].get(index=es[1], id=document_id)["_source"]
    assert stored["publication_tombstone"] and stored["hidden"]
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    assert stale_ordinal in authority.reservations(owner).ordinals
    assert not authority.reservations(owner).gate_closed
    authority.release(owner)


def test_delete_request_acquires_authority_before_file_and_job_locks(
    owned_file: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from fastapi import BackgroundTasks

    from onyx.background.celery.versioned_apps.client import app
    from onyx.db import regulatory_indexing_jobs as repository
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import User, UserFile
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import FileOwnership
    from onyx.server.features.projects import api

    real_request = repository.request_user_file_deletion_cleanup

    from datetime import datetime

    from onyx.db.regulatory_indexing_jobs import UserFileDeletionCleanupPlan

    def owned_request(
        session: Session,
        *,
        user_file_id: UUID,
        now: datetime,
        publication_owner: FileOwnership | None = None,
    ) -> UserFileDeletionCleanupPlan:
        owner = publication_owner
        assert isinstance(owner, FileOwnership), (
            "API deletion mutates file/job state before publication ownership"
        )
        PublicationStore(owner.scope).heartbeat(owner, ttl=timedelta(minutes=2))
        return real_request(
            session, user_file_id=user_file_id, now=now, publication_owner=owner
        )

    monkeypatch.setattr(repository, "request_user_file_deletion_cleanup", owned_request)
    monkeypatch.setattr(
        api, "request_user_file_deletion_cleanup", owned_request, raising=False
    )
    sent = MagicMock()
    monkeypatch.setattr(app, "send_task", sent)
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None
        file.document_sets = []
        file.status = UserFileStatus.COMPLETED
        user = session.get(User, file.user_id)
        assert user is not None
        session.commit()
        result = api.delete_user_file(owned_file, BackgroundTasks(), user, session)
        assert not result.has_associations
        session.expire_all()
        assert file.status == UserFileStatus.DELETING
        assert sent.call_args.kwargs["expires"] > 0


def test_original_ingestion_attachment_requires_retained_bytes_and_canonical_proof(
    owned_file: UUID,
    live_review: LiveReview,
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    from sqlalchemy import delete

    from onyx.background.celery.tasks.user_file_processing import tasks as file_tasks
    from onyx.chat.process_message import extract_context_files
    from onyx.configs.constants import FileOrigin
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import RegulatoryChunk, RegulatoryFilePublication, UserFile
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.db.regulatory_public_reads import protected_file_ids
    from onyx.error_handling.exceptions import OnyxError
    from onyx.file_processing.user_file_loader import load_user_file_documents
    from onyx.file_store.file_store import get_default_file_store
    from onyx.file_store.utils import (
        load_user_file,
        user_file_id_to_plaintext_file_name,
    )
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.writer_publication import republish_user_file

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=settings[0]
    ).embedding_model
    monkeypatch.setattr(
        model, "encode", lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts]
    )
    source_session.execute(
        delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == owned_file)
    )
    file = source_session.get(UserFile, owned_file)
    assert file is not None
    file.status, file.name, file.file_type = (
        UserFileStatus.PROCESSING,
        "ordinary.txt",
        "text/plain",
    )
    original_id = file.file_id
    source_session.commit()
    storage = get_default_file_store()
    original = b"MADDE 1\nThe original provision remains current."

    def save(payload: bytes) -> None:
        storage.save_file(
            BytesIO(payload),
            "ordinary.txt",
            FileOrigin.CHAT_UPLOAD,
            "text/plain",
            file_id=original_id,
        )

    save(original)
    try:
        docs, _ = load_user_file_documents(
            user_file_id=str(owned_file),
            file_id=original_id,
            file_name="ordinary.txt",
            tenant_id="public",
        )
        file_tasks._process_user_file_with_indexing(str(owned_file), docs, "public")
        source_session.expire_all()
        assert owned_file in protected_file_ids(source_session, (owned_file,))
        loaded = load_user_file(owned_file, source_session)
        assert b"original provision" in loaded.content
        file = source_session.get(UserFile, owned_file)
        assert file is not None
        context = extract_context_files([file], 10000, 100, source_session)
        assert not context.use_as_search_filter
        from onyx.chat.chat_utils import load_chat_file
        from onyx.file_store.models import ChatFileType, FileDescriptor
        from onyx.file_store.utils import load_chat_file_by_id

        descriptor: FileDescriptor = {
            "id": original_id,
            "type": ChatFileType.PLAIN_TEXT,
            "name": "ordinary.txt",
            "user_file_id": str(owned_file),
        }
        lazy = load_chat_file(descriptor, source_session)
        assert lazy.content == original
        assert load_chat_file_by_id(original_id).content == original
        publication = source_session.get(RegulatoryFilePublication, owned_file)
        assert publication is not None
        receipt = publication.original_ingestion_receipt
        assert receipt is not None
        save(b"Different bytes under the same original identity")
        with pytest.raises(OnyxError):
            load_user_file(owned_file, source_session)
        with pytest.raises(OnyxError):
            load_chat_file(descriptor, source_session)
        with pytest.raises(OnyxError):
            load_chat_file_by_id(original_id)
        save(original)
        # Administrative changes invalidate current-original proof permanently;
        # a full reindex cannot silently re-baseline the original.
        from onyx.db.regulatory_publication import PublicationStore
        from onyx.document_index.publication_models import PublicationScope
        from onyx.regulatory.amendments.annexes import config
        from onyx.regulatory.writer_publication import correct_owned_chunk

        authority = PublicationStore(
            PublicationScope(
                tenant_id="public",
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                database_identity=config.ANNEX_DATABASE_IDENTITY,
            )
        )
        owner = authority.acquire(
            owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2)
        )
        chunk = (
            source_session.query(RegulatoryChunk)
            .filter_by(user_file_id=owned_file)
            .first()
        )
        assert chunk is not None
        source_session.rollback()
        try:
            owner = correct_owned_chunk(
                owner, es[0], chunk.id, text="Corrected current provision."
            )
        finally:
            authority.release(owner)
        source_session.expire_all()
        with pytest.raises(OnyxError):
            load_user_file(owned_file, source_session)
        republish_user_file(owned_file, "public")
        source_session.expire_all()
        assert (
            source_session.get_one(
                RegulatoryFilePublication, owned_file
            ).original_ingestion_receipt
            == receipt
        )
        with pytest.raises(OnyxError):
            load_user_file(owned_file, source_session)
    finally:
        storage.delete_file(original_id, error_on_missing=False)
        storage.delete_file(
            user_file_id_to_plaintext_file_name(owned_file), error_on_missing=False
        )


@pytest.mark.parametrize("status", ["CANCELED", "DELETING"])
def test_new_writer_intent_cannot_restart_cancelled_or_deleting_file(
    owned_file: UUID,
    es: tuple[Elasticsearch, str],
    status: str,
) -> None:
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import UserFile
    from onyx.db.regulatory_writer_publication import stage_writer_publication

    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    try:
        with get_session_with_tenant(tenant_id="public") as session:
            file = session.get(UserFile, owned_file)
            assert file is not None
            file.status = UserFileStatus(status)
            before = canonical_scope_digest(session, owned_file)
            session.commit()
        manifest = WriterPublicationManifest(
            id=uuid4(),
            scope=owner.scope,
            user_file_id=owned_file,
            kind="reindex",
            canonical_before_sha256=before,
            indexes=[adapter_for(es).snapshot],
            previous_binding_ids=[],
            bindings=[],
        )
        with pytest.raises(ValueError, match="cancelled or deleting"):
            stage_writer_publication(owner, manifest)
        assert not authority.reservations(owner).gate_closed
    finally:
        authority.release(owner)
