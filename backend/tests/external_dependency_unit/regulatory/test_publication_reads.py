"""Public read races against owned local publication storage."""

from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch

from onyx.document_index.elasticsearch.client import (
    ElasticsearchDocumentMissingError,
    ElasticsearchIndexClient,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    adapter_for,
    store,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)


def test_public_tombstones_are_missing_before_deserialization(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    client, index_name = es
    client.indices.refresh(index=index_name)
    ids = [
        hit["_id"]
        for hit in client.search(index=index_name, query={"match_all": {}})["hits"][
            "hits"
        ]
    ]
    reader = ElasticsearchIndexClient(
        index_name, host="127.0.0.1", port=29200, use_ssl=False
    )
    try:
        with pytest.raises(ElasticsearchDocumentMissingError):
            reader.get_document_chunks(ids)
        with pytest.raises(RuntimeError, match="not found"):
            reader.get_document(ids[0])
        assert reader.search({"query": {"match_all": {}}}, None) == []
        assert (
            reader.search_for_document_ids(
                {"query": {"match_all": {}}, "_source": False}
            )
            == []
        )
        pit = reader.open_pit()
        try:
            chunks, _, _ = reader.fetch_chunks_for_doc_ids(pit, [str(owned_file)])
            assert chunks == []
        finally:
            reader.close_pit(pit)
    finally:
        reader.close()


def test_search_discards_a_file_closed_and_reopened_during_es_read(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes.config import (
        ANNEX_DATABASE_IDENTITY,
        REGULATORY_ANNEX_ENVIRONMENT,
    )
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    projection = frozen_projection(owned_file, 0)
    adapter.upsert(inventory, projection)
    adapter.tombstone(inventory, 1_000_000_042)
    proof = adapter.verify(inventory, (projection,))
    from onyx.db.engine.sql_engine import get_session_with_tenant

    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, owner, proof)
        session.commit()
    reader = ElasticsearchIndexClient(
        es[1], host="127.0.0.1", port=29200, use_ssl=False
    )
    reached, release = Event(), Event()
    actual_search = reader._client.search

    def paused_search(**kwargs: Any) -> object:
        response = actual_search(**kwargs)
        reached.set()
        assert release.wait(10)
        return response

    monkeypatch.setattr(reader._client, "search", paused_search)
    try:
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(reader.search, {"query": {"match_all": {}}}, None)
            assert reached.wait(10)
            authority.close_gate(owner)
            with get_session_with_tenant(tenant_id="public") as session:
                authority.finalize(session, owner, proof)
                session.commit()
            release.set()
            assert future.result(timeout=10) == []
    finally:
        release.set()
        reader.close()


@pytest.mark.parametrize("reopen", [False, True])
def test_delayed_model_flush_refuses_a_source_changed_after_retrieval(
    owned_file: UUID,
    es: tuple[Elasticsearch, str],
    reopen: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from typing import Any
    from unittest.mock import MagicMock, patch

    from onyx.chat.emitter import BufferedEmitter
    from onyx.chat.models import StreamingError
    from onyx.chat.process_message import _run_models
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.regulatory.publication_reads import public_read_store
    from onyx.server.query_and_chat.placement import Placement
    from onyx.server.query_and_chat.streaming_models import AgentResponseDelta, Packet
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )
    from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    projection = frozen_projection(owned_file, 0)
    adapter.upsert(inventory, projection)
    adapter.tombstone(inventory, 1_000_000_042)
    proof = adapter.verify(inventory, (projection,))
    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, owner, proof)
        session.commit()
    reader = ElasticsearchIndexClient(
        es[1], host="127.0.0.1", port=29200, use_ssl=False
    )
    retrieved, flush = Event(), Event()

    def model(**kwargs: Any) -> None:
        assert reader.search({"query": {"match_all": {}}}, None)
        staged = BufferedEmitter()
        staged.emit(
            Packet(
                placement=Placement(turn_index=0),
                obj=AgentResponseDelta(content="stale answer"),
            )
        )
        retrieved.set()
        assert flush.wait(10)
        staged.replay_to(kwargs["emitter"])
        kwargs["state_container"].set_answer_tokens("stale answer")

    setup = _make_setup()
    with (
        patch("onyx.chat.process_message.run_llm_loop", side_effect=model),
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_token_counter", return_value=lambda _: 0
        ),
        patch("onyx.chat.process_message.llm_loop_completion_handle") as saved,
        patch("onyx.chat.process_message.get_session_with_current_tenant"),
        patch("onyx.db.chat.invalidate_publication_chat_message"),
        patch("onyx.db.regulatory_chat_reads.mark_message_publication_generated"),
        patch("onyx.db.regulatory_chat_reads.finalize_message_publication_read"),
        ThreadPoolExecutor(1) as executor,
    ):
        future = executor.submit(lambda: list(_run_models(setup, MagicMock())))
        assert retrieved.wait(10)
        authority.close_gate(owner)
        if reopen:
            with get_session_with_tenant(tenant_id="public") as session:
                authority.finalize(session, owner, proof)
                session.commit()
        flush.set()
        packets = future.result(timeout=10)
        assert not any(
            isinstance(item, Packet) and isinstance(item.obj, AgentResponseDelta)
            for item in packets
        )
        assert any(
            isinstance(item, StreamingError)
            and item.error_code == "PUBLICATION_SOURCE_CHANGED"
            for item in packets
        )
        saved.assert_not_called()
    reader.close()


def test_authorized_file_gate_precedes_not_modified_and_file_store(
    owned_file: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request

    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import User, UserFile
    from onyx.error_handling.exceptions import OnyxError
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.publication_reads import public_read_store
    from onyx.server.query_and_chat.chat_backend import fetch_chat_file

    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", False)
    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None
        user = session.get(User, file.user_id)
        assert user is not None
        request = Request(
            {
                "type": "http",
                "headers": [(b"if-none-match", f'"{file.file_id}"'.encode())],
            }
        )
        with pytest.raises(OnyxError, match="source"):
            fetch_chat_file(
                str(owned_file), request, parsed=False, user=user, db_session=session
            )


def test_mixed_context_keeps_unversioned_text_and_searches_protected_source(
    owned_file: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.chat.process_message import extract_context_files
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import UserFile
    from onyx.file_store.models import ChatFileType, InMemoryChatFile
    from onyx.regulatory.publication_reads import public_read_store
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        create_owned_file,
    )

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    with (
        create_owned_file() as plain_id,
        get_session_with_tenant(tenant_id="public") as session,
    ):
        files = []
        for identity in (owned_file, plain_id):
            file = session.get(UserFile, identity)
            assert file is not None
            files.append(file)

        def load(
            user_file_ids: list[UUID], **_kwargs: object
        ) -> list[InMemoryChatFile]:
            return [
                InMemoryChatFile(
                    file_id=file.file_id,
                    content=(
                        b"OLD original" if file.id == owned_file else b"plain current"
                    ),
                    file_type=ChatFileType.PLAIN_TEXT,
                    filename=file.name,
                )
                for file in files
                if file is not None and file.id in user_file_ids
            ]

        monkeypatch.setattr("onyx.chat.process_message.load_in_memory_chat_files", load)
        result = extract_context_files(files, 10000, 100, session)
        assert result.file_texts == ["plain current"]
        assert result.use_as_search_filter


def test_prefetched_federated_read_without_db_session_observes_publication(
    owned_file: UUID,
) -> None:
    from unittest.mock import MagicMock

    from onyx.configs.constants import DocumentSource, FederatedConnectorSource
    from onyx.context.search.models import (
        ChunkIndexRequest,
        IndexFilters,
        InferenceChunk,
    )
    from onyx.context.search.retrieval.search_runner import search_chunks
    from onyx.federated_connectors.federated_retrieval import FederatedRetrievalInfo
    from onyx.regulatory.publication_reads import public_read_store
    from tests.unit.onyx.regulatory.test_provision_retrieval import _chunk

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    protected = _chunk(0, "protected", document_id=str(owned_file))
    unrelated = _chunk(0, None, document_id="slack-message")

    def federated(_query: ChunkIndexRequest) -> list[InferenceChunk]:
        authority.close_gate(owner)
        return [protected, unrelated]

    result = search_chunks(
        ChunkIndexRequest(
            query="source",
            filters=IndexFilters(
                access_control_list=None, source_type=[DocumentSource.SLACK]
            ),
        ),
        user_id=None,
        document_index=MagicMock(),
        db_session=None,
        prefetched_federated_retrieval_infos=[
            FederatedRetrievalInfo(
                retrieval_function=federated,
                source=FederatedConnectorSource.FEDERATED_SLACK,
            )
        ],
    )
    assert result == [unrelated]


def test_saved_inflight_message_refuses_changed_sources_but_final_history_survives(
    owned_file: UUID,
) -> None:
    from onyx.configs.constants import MessageType
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import ChatMessage, ChatSession
    from onyx.db.regulatory_chat_reads import (
        finalize_message_publication_read,
        mark_message_publication_generated,
        message_publication_available,
        stage_message_publication_read,
    )
    from onyx.regulatory.publication_reads import (
        PublicationReadEvidence,
        public_read_store,
    )

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    evidence = PublicationReadEvidence(
        observation=authority.observe(), user_file_ids=(owned_file,)
    )
    with get_session_with_tenant(tenant_id="public") as session:
        chat = ChatSession(description="5c owned source replay", persona_id=0)
        session.add(chat)
        session.flush()
        pending = ChatMessage(
            chat_session_id=chat.id,
            message="pending source answer",
            token_count=3,
            message_type=MessageType.ASSISTANT,
        )
        complete = ChatMessage(
            chat_session_id=chat.id,
            message="completed history",
            token_count=2,
            message_type=MessageType.ASSISTANT,
        )
        session.add_all([pending, complete])
        session.flush()
        stage_message_publication_read(session, pending, evidence)
        stage_message_publication_read(session, complete, evidence)
        session.commit()
        pending_id, complete_id, chat_id = pending.id, complete.id, chat.id
    try:
        assert not finalize_message_publication_read(complete_id)
        assert mark_message_publication_generated(complete_id)
        assert mark_message_publication_generated(pending_id)
        assert finalize_message_publication_read(complete_id)
        authority.close_gate(owner)
        with get_session_with_tenant(tenant_id="public") as session:
            pending = session.get(ChatMessage, pending_id)
            complete = session.get(ChatMessage, complete_id)
            assert pending is not None and complete is not None
            assert not message_publication_available(pending)
            assert message_publication_available(complete)
        finalize_message_publication_read(pending_id)
        with get_session_with_tenant(tenant_id="public") as session:
            pending = session.get(ChatMessage, pending_id)
            assert pending is not None
            assert not message_publication_available(pending)
            from onyx.db.models import User, UserFile
            from onyx.server.query_and_chat.chat_backend import get_chat_session

            file = session.get(UserFile, owned_file)
            assert file is not None
            user = session.get(User, file.user_id)
            assert user is not None
            detail = get_chat_session(chat_id, user=user, db_session=session)
            assert len(detail.packets) == 2
            assistant_messages = [
                item
                for item in detail.messages
                if item.message_type == MessageType.ASSISTANT
            ]
            pending_index = next(
                index
                for index, item in enumerate(assistant_messages)
                if item.message_id == pending_id
            )
            assert detail.packets[pending_index] == []
            assert any(item.message == "completed history" for item in detail.messages)
    finally:
        with get_session_with_tenant(tenant_id="public") as session:
            from sqlalchemy import delete

            session.execute(
                delete(ChatMessage).where(ChatMessage.chat_session_id == chat_id)
            )
            session.execute(delete(ChatSession).where(ChatSession.id == chat_id))
            session.commit()


def test_chat_history_attachment_uses_dated_search_for_versioned_original(
    owned_file: UUID,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.chat.chat_utils import load_chat_file
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import UserFile
    from onyx.file_store.models import ChatFileType
    from onyx.regulatory.publication_reads import public_read_store

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    authority.close_gate(owner)
    inventory = authority.reservations(owner)
    adapter = adapter_for(es)
    adapter.seal(inventory)
    for ordinal in inventory.ordinals:
        adapter.tombstone(inventory, ordinal)
    proof = adapter.verify(inventory, ())
    with get_session_with_tenant(tenant_id="public") as session:
        authority.finalize(session, owner, proof)
        session.commit()
        file = session.get(UserFile, owned_file)
        assert file is not None
        monkeypatch.setattr(
            "onyx.chat.chat_utils._get_or_extract_plaintext",
            lambda *_args, **_kwargs: "OLD original bytes",
        )
        loaded = load_chat_file(
            {
                "id": file.file_id,
                "user_file_id": str(file.id),
                "type": ChatFileType.PLAIN_TEXT,
                "name": file.name,
            },
            session,
        )
        assert loaded.content_text is not None
        assert "OLD original bytes" not in loaded.content_text
        assert "dated search" in loaded.content_text
        with pytest.raises(Exception, match="dated search"):
            _ = loaded.content


def test_actual_temporal_es_reads_keep_sparse_ordinals_and_own_dated_siblings(
    owned_file: UUID,
    es: tuple[Elasticsearch, str],
) -> None:
    import json
    from datetime import date, datetime, time, timezone

    from sqlalchemy import select

    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import RegulatoryChunk
    from onyx.db.regulatory_chunks import (
        get_bounded_same_provision_siblings,
        get_regulatory_provision_heading_source,
    )
    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.document_index.elasticsearch.elasticsearch_document_index import (
        convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned,
    )
    from onyx.document_index.publication_models import FrozenPublicationProjection
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from onyx.regulatory.provision_retrieval import _chunk_from_projection
    from onyx.regulatory.publication_reads import public_read_store
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=60))
    authority.close_gate(owner)
    extra = [
        authority.allocate(owner, "5c-current"),
        authority.allocate(owner, "5c-future"),
    ]
    adapter = adapter_for(es)
    with get_session_with_tenant(tenant_id="public") as session:
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == owned_file)
                .order_by(RegulatoryChunk.position)
            )
        )
        rows[1].position = 1
        for row in rows:
            row.heading_path = ["Madde 1"]
            row.chunk_metadata = {"article_no": "1"}
        session.commit()
        canonical_ids = [row.id for row in rows]
        canonical_texts = [row.text for row in rows]
    bindings = []
    for number, ordinal, label, start, end in (
        (0, 0, "past", date(2020, 1, 1), date(2026, 1, 1)),
        (0, extra[0], "current", date(2026, 1, 1), date(2027, 1, 1)),
        (0, extra[1], "future", date(2027, 1, 1), None),
        (1, 1_000_000_042, "sibling", date(2020, 1, 1), None),
    ):
        base = frozen_projection(owned_file, ordinal)
        source = json.loads(base.source_json)
        summary, context = f"{label} summary\n", f"\n{label} context"
        content = summary + canonical_texts[number] + context
        source.update(
            regulatory_chunk_id=canonical_ids[number],
            heading_path=["Madde 1"],
            content=content,
            doc_summary=summary,
            chunk_context=context,
            image_file_id=f"{label}-image",
            source_links=json.dumps({0: ""}),
            validity_start_date=int(
                datetime.combine(start, time.min, timezone.utc).timestamp()
            ),
            validity_end_date=int(
                datetime.combine(end, time.min, timezone.utc).timestamp()
            )
            if end
            else None,
        )
        identity = uuid4()
        projection = FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id=str(identity),
            source_json=json.dumps(source),
            embedding_inputs=(content,),
            embedding_config_json=base.embedding_config_json,
        )
        bindings.append(
            AnnexTemporalProjection(
                id=identity,
                index=adapter.snapshot,
                projection=projection,
                canonical_base_sha256=context_hash(canonical_texts[number]),
                derived_role="canonical",
                dependency_ids=[],
                representation_text=canonical_texts[number],
                representation_metadata={"image_file_id": f"{label}-image"},
                reference_date=start,
                effective_start=start,
                effective_end=end,
                semantic_position=number,
            )
        )
    inventory = authority.reservations(owner)
    adapter.seal(inventory)
    for binding in bindings:
        adapter.upsert(inventory, binding.projection)
    proof = adapter.verify(inventory, tuple(binding.projection for binding in bindings))
    with get_session_with_tenant(tenant_id="public") as session:
        for binding in bindings:
            activate_temporal_projection(
                session, user_file_id=owned_file, binding=binding
            )
        authority.finalize(session, owner, proof)
        session.commit()
    reader = ElasticsearchIndexClient(
        es[1], host="127.0.0.1", port=29200, use_ssl=False
    )
    try:
        for when, expected in (
            (date(2025, 1, 1), bindings[0]),
            (date(2026, 9, 10), bindings[1]),
            (date(2028, 1, 1), bindings[2]),
        ):
            hits = reader.search(
                {"query": {"match_all": {}}, "size": 20},
                None,
                as_of_date=when,
                publication_index=adapter.snapshot,
            )
            assert {hit.document_chunk.chunk_index for hit in hits} == {
                expected.projection.ordinal,
                1_000_000_042,
            }
            seed_source = next(
                hit.document_chunk
                for hit in hits
                if hit.document_chunk.regulatory_chunk_id == canonical_ids[0]
            )
            seed = convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                seed_source, 1, {}
            )
            assert seed.structural_position == 0
            with get_session_with_tenant(tenant_id="public") as session:
                siblings = get_bounded_same_provision_siblings(
                    session,
                    [canonical_ids[0]],
                    query="existing",
                    as_of_date=when,
                    query_indexes={owned_file: adapter.snapshot},
                )
                assert [item.position for item in siblings] == [0, 1]
                sibling = next(
                    item
                    for item in siblings
                    if item.regulatory_chunk_id == canonical_ids[1]
                )
                hydrated = _chunk_from_projection(sibling, seed)
                assert hydrated.chunk_id == 1_000_000_042
                assert hydrated.structural_position == 1
                assert hydrated.image_file_id == "sibling-image"
                assert hydrated.doc_summary == "sibling summary\n"
                assert hydrated.chunk_context == "\nsibling context"
                outline = get_regulatory_provision_heading_source(
                    session,
                    canonical_ids,
                    as_of_date=when,
                    query_indexes={owned_file: adapter.snapshot},
                )
                assert outline is not None
                assert [candidate.position for candidate in outline.candidates] == [
                    0,
                    1,
                ]
        assert (
            reader.search(
                {"query": {"match_all": {}}, "size": 20},
                None,
                as_of_date=date(2019, 1, 1),
                publication_index=adapter.snapshot,
            )
            == []
        )
        with pytest.raises(ValueError, match="physical index"):
            reader.search(
                {"query": {"match_all": {}}},
                None,
                publication_index=adapter.snapshot.model_copy(
                    update={"index_uuid": "recreated"}
                ),
            )
    finally:
        reader.close()
        from sqlalchemy import delete

        from onyx.db.models import RegulatoryTemporalProjection

        with get_session_with_tenant(tenant_id="public") as session:
            session.execute(
                delete(RegulatoryTemporalProjection).where(
                    RegulatoryTemporalProjection.user_file_id == owned_file
                )
            )
            session.commit()


def test_actual_http_reconnect_and_304_cannot_reuse_inflight_changed_sources(
    owned_file: UUID,
) -> None:
    from io import BytesIO

    import httpx
    from fastapi_users.password import PasswordHelper
    from sqlalchemy import delete

    from onyx.cache.factory import get_cache_backend
    from onyx.chat.chat_processing_checker import set_processing_status
    from onyx.chat.stream_buffer import StreamBufferWriter, _meta_key
    from onyx.configs.constants import FileOrigin, MessageType
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.enums import Permission
    from onyx.db.models import ChatMessage, ChatSession, User, UserFile
    from onyx.db.regulatory_chat_reads import (
        MessagePublicationRead,
        mark_message_publication_generated,
        stage_message_publication_read,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.publication_reads import (
        PublicationReadEvidence,
        public_read_store,
    )

    store = get_default_file_store()
    source_id = store.save_file(
        BytesIO(b"immutable original"),
        display_name="5c-original.txt",
        file_origin=FileOrigin.USER_FILE,
        file_type="text/plain",
    )
    unrelated_id = store.save_file(
        BytesIO(b"internal artifact"),
        display_name="5c-internal.json",
        file_origin=FileOrigin.OTHER,
        file_type="application/json",
    )
    password = PasswordHelper().generate()
    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=60))
    evidence = PublicationReadEvidence(
        observation=authority.observe(), user_file_ids=(owned_file,)
    )
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None
        file.file_id = source_id
        user = session.get(User, file.user_id)
        assert user is not None
        user.hashed_password = PasswordHelper().hash(password)
        user.effective_permissions = [
            Permission.BASIC_ACCESS.value,
            Permission.READ_CHAT.value,
        ]
        email = user.email
        chat = ChatSession(
            user_id=user.id, persona_id=0, description="5c owned reconnect"
        )
        session.add(chat)
        session.flush()
        message = ChatMessage(
            chat_session_id=chat.id,
            message="stale-saved-answer",
            token_count=2,
            message_type=MessageType.ASSISTANT,
        )
        session.add(message)
        session.flush()
        stage_message_publication_read(session, message, evidence)
        session.commit()
        chat_id, run_id = chat.id, message.id
    assert mark_message_publication_generated(run_id)
    cache = get_cache_backend()
    writer = StreamBufferWriter(cache, chat_id, run_id)
    writer.append_line(
        '{"delivered":"before-change"}\n', evidence=evidence, message_id=run_id
    )
    writer.flush()
    set_processing_status(chat_id, cache, True, run_id=run_id)
    try:
        with httpx.Client(base_url="http://localhost:23000", timeout=20) as client:
            login = client.post(
                "/api/auth/login", data={"username": email, "password": password}
            )
            assert login.status_code == 204, login.text
            first_file = client.get(f"/api/chat/file/{owned_file}")
            assert first_file.status_code == 200, first_file.text
            assert first_file.content == b"immutable original"
            assert client.get(f"/api/chat/file/{unrelated_id}").status_code == 404
            with client.stream(
                "GET", f"/api/chat/chat-session/{chat_id}/resume-stream"
            ) as response:
                assert response.status_code == 200
                assert "before-change" in next(response.iter_lines())
            with get_session_with_tenant(tenant_id="public") as session:
                pending = session.get(ChatMessage, run_id)
                assert pending is not None
                assert not MessagePublicationRead.model_validate(
                    pending.publication_read
                ).finalized
            authority.close_gate(owner)
            stale_304 = client.get(
                f"/api/chat/file/{owned_file}",
                headers={"If-None-Match": first_file.headers["etag"]},
            )
            assert stale_304.status_code == 503, stale_304.text
            assert b"immutable original" not in stale_304.content
            writer.append_line('{"stale":"after-change"}\n', evidence=evidence)
            writer.mark_done()
            replay = client.get(f"/api/chat/chat-session/{chat_id}/resume-stream")
            assert replay.status_code == 200, replay.text
            assert (
                "before-change" not in replay.text and "after-change" not in replay.text
            )
            fallback = client.get(f"/api/chat/get-chat-session/{chat_id}")
            assert fallback.status_code == 200, fallback.text
            assert "stale-saved-answer" not in fallback.text
            assert "source changed" in fallback.text.lower()
            cache.delete(_meta_key(chat_id, run_id))
            assert (
                client.get(
                    f"/api/chat/chat-session/{chat_id}/resume-stream"
                ).status_code
                == 404
            )
            missing_fallback = client.get(f"/api/chat/get-chat-session/{chat_id}")
            assert missing_fallback.status_code == 200
            assert "stale-saved-answer" not in missing_fallback.text
    finally:
        set_processing_status(chat_id, cache, False)
        cache.delete(_meta_key(chat_id, run_id))
        cache.delete(f"chatstream_{chat_id}_{run_id}:0")
        with get_session_with_tenant(tenant_id="public") as session:
            session.execute(
                delete(ChatMessage).where(ChatMessage.chat_session_id == chat_id)
            )
            session.execute(delete(ChatSession).where(ChatSession.id == chat_id))
            session.commit()
        store.delete_file(source_id)
        store.delete_file(unrelated_id)


def test_writer_drain_does_not_finalize_an_unconsumed_live_answer(
    owned_file: UUID,
) -> None:
    from threading import Event
    from unittest.mock import MagicMock, patch

    from sqlalchemy import delete

    from onyx.chat.chat_state import ChatStateContainer
    from onyx.chat.process_message import _run_models
    from onyx.configs.constants import MessageType
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import ChatMessage, ChatSession
    from onyx.db.regulatory_chat_reads import (
        MessagePublicationRead,
        stage_message_publication_read,
    )
    from onyx.regulatory.publication_reads import (
        filter_publication_read,
        observe_publication_read,
        public_read_store,
    )
    from onyx.server.query_and_chat.placement import Placement
    from onyx.server.query_and_chat.streaming_models import AgentResponseDelta, Packet
    from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup

    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=30))
    with get_session_with_tenant(tenant_id="public") as session:
        chat = ChatSession(description="5c unconsumed turn", persona_id=0)
        session.add(chat)
        session.flush()
        message = ChatMessage(
            chat_session_id=chat.id,
            message="",
            token_count=0,
            message_type=MessageType.ASSISTANT,
        )
        session.add(message)
        session.commit()
        chat_id, message_id = chat.id, message.id
        session.expunge(message)
    setup = _make_setup()
    setup.reserved_messages = [message]
    setup.chat_session.id = chat_id
    drained = Event()

    def model(**kwargs: Any) -> None:
        observation = observe_publication_read()
        assert filter_publication_read(
            observation, [str(owned_file)], lambda value: value
        )
        kwargs["state_container"].set_answer_tokens("queued stale answer")
        kwargs["emitter"].emit(
            Packet(
                placement=Placement(turn_index=0),
                obj=AgentResponseDelta(content="queued stale answer"),
            )
        )

    def save(*, state_container: ChatStateContainer, **_kwargs: Any) -> None:
        with get_session_with_tenant(tenant_id="public") as session:
            attached = session.get(ChatMessage, message_id)
            assert attached is not None
            attached.message = "queued stale answer"
            stage_message_publication_read(
                session, attached, state_container.publication_reads.evidence()
            )
            session.commit()

    def fence(*, value: bool, **_kwargs: Any) -> None:
        if not value:
            drained.set()

    try:
        with (
            patch("onyx.chat.process_message.run_llm_loop", side_effect=model),
            patch("onyx.chat.process_message.construct_tools", return_value={}),
            patch(
                "onyx.chat.process_message.get_llm_token_counter",
                return_value=lambda _: 0,
            ),
            patch(
                "onyx.chat.process_message.llm_loop_completion_handle", side_effect=save
            ),
            patch("onyx.chat.process_message.set_processing_status", side_effect=fence),
        ):
            stream = _run_models(setup, MagicMock())
            assert drained.wait(10)
            with get_session_with_tenant(tenant_id="public") as session:
                attached = session.get(ChatMessage, message_id)
                assert attached is not None
                state = MessagePublicationRead.model_validate(attached.publication_read)
                assert state.generation_done and not state.finalized
            authority.close_gate(owner)
            packets = list(stream)
            assert not any(
                isinstance(item, Packet) and isinstance(item.obj, AgentResponseDelta)
                for item in packets
            )
            with get_session_with_tenant(tenant_id="public") as session:
                attached = session.get(ChatMessage, message_id)
                assert attached is not None
                assert attached.message != "queued stale answer"
    finally:
        with get_session_with_tenant(tenant_id="public") as session:
            session.execute(
                delete(ChatMessage).where(ChatMessage.chat_session_id == chat_id)
            )
            session.execute(delete(ChatSession).where(ChatSession.id == chat_id))
            session.commit()
