from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingStage,
    UserFileStatus,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    UserFile,
)
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.schema import (
    DocumentChunk,
    get_elasticsearch_doc_chunk_id,
)
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationError,
    DocumentChunkVerificationExpectation,
    DocumentChunkVerificationRequest,
    DocumentIndex,
    TenantState,
)
from onyx.indexing.models import DocMetadataAwareIndexChunk, IndexChunk
from onyx.regulatory.indexing_jobs import publisher
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    RegulatoryInputHashVersion,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.publisher import (
    PublishVerification,
    publish_regulatory_job,
    stage_regulatory_job_in_index,
    verify_staged_regulatory_job,
)
from shared_configs.enums import EmbeddingProvider


def _snapshot() -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot(
        input_content_hash="1" * 64,
        input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
        chunk_generation_hash="2" * 64,
        search_settings_id=41,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name="openai/text-embedding-3-large",
        model_dimension=3,
        reduced_dimension=None,
        effective_dimension=3,
        index_name="regulatory-index",
        vertex=VertexBatchConfig(
            model_configuration_id=73,
            model_name="gemini-3.1-flash-lite",
            project="customs-prod",
            location="europe-west4",
            authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        ),
        prompt_version="contextual-rag-v1",
        prompt_hash="a" * 64,
    )


def _fixture() -> tuple[
    RegulatoryIndexingJob,
    UserFile,
    SearchSettings,
    list[RegulatoryChunk],
    list[RegulatoryIndexingItem],
]:
    from onyx.db.enums import IndexModelStatus
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        canonical_row,
    )

    snapshot = _snapshot()
    file = UserFile(
        id=uuid4(),
        name="Gümrük Yönetmeliği.md",
        status=UserFileStatus.INDEXING,
        chunk_count=5,
    )
    job = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=file.id,
        search_settings_id=41,
        lease_generation=11,
        config_snapshot=snapshot.model_dump(mode="json"),
    )
    settings = SearchSettings(
        id=41,
        index_name=snapshot.index_name,
        model_name=snapshot.embedding_model_name,
        provider_type=snapshot.embedding_provider,
        model_dim=3,
        reduced_dimension=None,
        status=IndexModelStatus.PRESENT,
        normalize=True,
        query_prefix="",
        passage_prefix="",
        cloud_provider=None,
    )
    rows = [
        canonical_row(file.id, i, f"MADDE {i + 1} - Yürürlük hükmü.") for i in range(2)
    ]
    items = [
        RegulatoryIndexingItem(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=row.id,
            status=RegulatoryIndexingItemStatus.EMBEDDED.value,
            context={"contextual_text": "Generated context."} if i == 0 else None,
            vector=[float(i + 1), 0.2, 0.3],
            request_hash="a" * 64,
        )
        for i, row in enumerate(rows)
    ]
    return job, file, settings, rows, items


@pytest.mark.parametrize(
    "mismatch",
    ["count", "missing", "hidden", "dimension", "non_finite"],
)
def test_elasticsearch_verification_rejects_projection_mismatch(
    mismatch: str,
) -> None:
    document_id = str(uuid4())
    tenant_state = TenantState(tenant_id="tenant-a", multitenant=False)
    expected = DocumentChunkVerificationRequest(
        document_id=document_id,
        expected_chunks=(
            DocumentChunkVerificationExpectation(
                chunk_index=0, regulatory_chunk_id="row-0"
            ),
            DocumentChunkVerificationExpectation(
                chunk_index=1, regulatory_chunk_id="row-1"
            ),
        ),
        expected_hidden=True,
        content_vector_dimension=3,
    )
    chunk_ids = [
        get_elasticsearch_doc_chunk_id(
            tenant_state=tenant_state,
            document_id=document_id,
            chunk_index=index,
        )
        for index in range(2)
    ]
    chunks: dict[str, DocumentChunk] = {
        chunk_ids[index]: cast(
            DocumentChunk,
            SimpleNamespace(
                document_id=document_id,
                chunk_index=index,
                regulatory_chunk_id=f"row-{index}",
                hidden=True,
                content_vector=[0.1, 0.2, 0.3],
            ),
        )
        for index in range(2)
    }
    count = 2
    if mismatch == "count":
        count = 3
    elif mismatch == "missing":
        chunks.pop(chunk_ids[1])
    elif mismatch == "hidden":
        chunks[chunk_ids[1]].hidden = False
    elif mismatch == "dimension":
        chunks[chunk_ids[1]].content_vector = [0.1, 0.2]
    else:
        chunks[chunk_ids[1]].content_vector = [0.1, float("nan"), 0.3]

    client = MagicMock()
    client.count_by_query.return_value = count
    client.get_document_chunks.return_value = chunks
    document_index = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    document_index._index_name = "regulatory-index"
    document_index._client = client
    document_index._tenant_state = tenant_state

    with pytest.raises(DocumentChunkVerificationError):
        document_index.verify_document_chunks(expected)


def test_elasticsearch_verification_count_is_tenant_scoped() -> None:
    document_id = str(uuid4())
    request = DocumentChunkVerificationRequest(
        document_id=document_id,
        expected_chunks=(
            DocumentChunkVerificationExpectation(
                chunk_index=0,
                regulatory_chunk_id="row-0",
            ),
        ),
        expected_hidden=True,
        content_vector_dimension=3,
    )
    client = MagicMock()
    client.count_by_query.return_value = 0
    document_index = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    document_index._index_name = "regulatory-index"
    document_index._client = client
    document_index._tenant_state = TenantState(
        tenant_id="tenant-a",
        multitenant=True,
    )

    with pytest.raises(DocumentChunkVerificationError, match="count mismatch"):
        document_index.verify_document_chunks(request)

    count_query = client.count_by_query.call_args.args[0]
    assert {"term": {"tenant_id": {"value": "tenant-a"}}} in count_query["query"][
        "bool"
    ]["filter"]


def test_elasticsearch_mget_parser_correlates_out_of_order_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.document_index.publication_models import PublicationScope, ReadObservation
    from onyx.regulatory import publication_reads

    store = MagicMock()
    store.observe.return_value = ReadObservation(
        scope=PublicationScope(
            tenant_id="public", environment="unit", database_identity="unit"
        ),
        committed_epoch=0,
    )
    store.unavailable.return_value = frozenset()
    monkeypatch.setattr(publication_reads, "public_read_store", lambda: store)
    document_id = str(uuid4())
    chunk_ids = [f"chunk-{index}" for index in range(2)]
    chunks = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=index,
            content=f"MADDE {index + 1}",
            source_type="file",
            public=True,
            access_control_list=[],
            global_boost=1,
            semantic_identifier=f"Madde {index + 1}",
            blurb=f"MADDE {index + 1}",
            doc_summary="",
            chunk_context="",
            regulatory_chunk_id=f"row-{index}",
            content_vector=[float(index), 0.2, 0.3],
        )
        for index in range(2)
    ]
    raw_client = MagicMock()
    raw_client.mget.return_value = {
        "docs": [
            {
                "_id": chunk_ids[index],
                "found": True,
                "_source": chunks[index].model_dump(mode="json"),
            }
            for index in (1, 0)
        ]
    }
    client = ElasticsearchIndexClient.__new__(ElasticsearchIndexClient)
    client._index_name = "regulatory-index"
    client._client = raw_client

    parsed = client.get_document_chunks(chunk_ids)

    assert list(parsed) == [chunk_ids[1], chunk_ids[0]]
    assert parsed[chunk_ids[0]].regulatory_chunk_id == "row-0"
    assert parsed[chunk_ids[1]].content_vector == [1.0, 0.2, 0.3]
    raw_client.mget.assert_called_once_with(
        index="regulatory-index",
        docs=[
            {
                "_id": chunk_id,
                "_source": {"includes": ["*", "content_vector", "title_vector"]},
            }
            for chunk_id in chunk_ids
        ],
    )


@pytest.mark.parametrize("stage", ["INDEX_WRITE", "VERIFY", "PUBLISH"])
def test_owned_checkpoint_uses_fresh_runtime_and_exact_caller_scope(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from onyx.db.enums import RegulatoryIndexingStage
    from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingRuntime
    from onyx.regulatory.indexing_jobs import owned_publication
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    job, file, settings, rows, items = _fixture()
    fresh_rows, fresh_items = deepcopy(rows), deepcopy(items)
    fresh_rows[0].text = "Persisted canonical update"
    runtime = RegulatoryIndexingRuntime(
        job=job,
        user_file=file,
        search_settings=settings,
        regulatory_chunks=tuple(fresh_rows),
        indexing_items=tuple(fresh_items),
    )
    session = MagicMock(spec=Session)
    captured: list[dict[str, object]] = []

    def execute(**kwargs: object) -> RegulatoryIndexingRuntime:
        session.rollback.assert_called_once_with()
        captured.append(kwargs)
        return runtime

    monkeypatch.setattr(owned_publication, "execute_owned_durable_stage", execute)
    expected_verification = publisher._expected_verification

    def verify_fresh(**kwargs: object) -> PublishVerification:
        assert kwargs["rows"] is runtime.regulatory_chunks
        assert kwargs["items"] is runtime.indexing_items
        assert runtime.regulatory_chunks[0].text == "Persisted canonical update"
        return expected_verification(
            job_id=job.id,
            user_file_id=file.id,
            rows=runtime.regulatory_chunks,
            items=runtime.indexing_items,
            snapshot=_snapshot(),
        )

    monkeypatch.setattr(publisher, "_expected_verification", verify_fresh)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("tenant-a")
    try:
        if stage == "INDEX_WRITE":
            result = stage_regulatory_job_in_index(
                job=job,
                user_file=file,
                rows=rows,
                items=items,
                search_settings=settings,
                db_session=session,
                tenant_id="tenant-a",
            )
        elif stage == "VERIFY":
            result = verify_staged_regulatory_job(
                job=job,
                user_file=file,
                rows=rows,
                items=items,
                search_settings=settings,
                db_session=session,
            )
        else:
            result = publish_regulatory_job(
                job=job,
                user_file=file,
                rows=rows,
                items=items,
                search_settings=settings,
                db_session=session,
            )
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    assert captured == [
        dict(
            job_id=job.id,
            file_id=file.id,
            generation=11,
            stage=RegulatoryIndexingStage(stage),
            tenant_id="tenant-a",
            caller_row_ids={row.id for row in rows},
            caller_item_ids={item.id for item in items},
        )
    ]
    if isinstance(result, PublishVerification):
        assert (
            result.canonical_chunk_count == 2
            and result.embedded_item_count == 2
            and result.vector_dimension == 3
        )
    else:
        assert result is publisher.PublishOutcome.COMPLETED
    session.commit.assert_not_called()


@pytest.mark.parametrize("stage", ["INDEX_WRITE", "VERIFY", "PUBLISH"])
def test_checkpoint_refuses_legacy_unfenced_transport(stage: str) -> None:
    job, file, settings, rows, items = _fixture()
    with pytest.raises(ValueError, match="configured fenced Elasticsearch"):
        publisher._owned_checkpoint(
            job=job,
            user_file=file,
            rows=rows,
            items=items,
            search_settings=settings,
            tenant_id="tenant-a",
            db_session=MagicMock(spec=Session),
            document_index=MagicMock(spec=DocumentIndex),
            stage=RegulatoryIndexingStage(stage),
        )


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("lease was lost"),
        RuntimeError("commit failed"),
        DocumentChunkVerificationError("actual index mismatch"),
    ],
)
def test_owned_failure_never_reports_completion(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    from onyx.regulatory.indexing_jobs import owned_publication

    job, file, settings, rows, items = _fixture()
    session = MagicMock(spec=Session)

    def execute(**_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(owned_publication, "execute_owned_durable_stage", execute)
    with pytest.raises(type(failure), match=str(failure)):
        publish_regulatory_job(
            job=job,
            user_file=file,
            rows=rows,
            items=items,
            search_settings=settings,
            db_session=session,
        )
    session.commit.assert_not_called()


def test_publish_rejects_stale_count_proof_before_owned_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.indexing_jobs import owned_publication

    job, file, settings, rows, items = _fixture()
    expected = publisher._expected_verification(
        job_id=job.id,
        user_file_id=file.id,
        rows=rows,
        items=items,
        snapshot=_snapshot(),
    )
    execute = MagicMock()
    monkeypatch.setattr(owned_publication, "execute_owned_durable_stage", execute)
    with pytest.raises(ValueError, match="verification identity"):
        publish_regulatory_job(
            job=job,
            user_file=file,
            rows=rows,
            items=items,
            verification=expected.model_copy(update={"embedded_item_count": 3}),
            db_session=MagicMock(spec=Session),
        )
    execute.assert_not_called()


@pytest.mark.parametrize(
    "stage",
    [
        RegulatoryIndexingStage.INDEX_WRITE,
        RegulatoryIndexingStage.VERIFY,
        RegulatoryIndexingStage.PUBLISH,
    ],
)
def test_owned_runtime_replays_staged_identity_and_stage_visibility(
    monkeypatch: pytest.MonkeyPatch, stage: RegulatoryIndexingStage
) -> None:
    from onyx.db.regulatory_durable_publication import durable_publication_input_digest
    from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingRuntime
    from onyx.document_index.publication_models import PublicationIndexSnapshot
    from onyx.regulatory.indexing_jobs import owned_publication
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
    )

    job, file, settings, rows, items = _fixture()
    runtime = RegulatoryIndexingRuntime(
        job=job,
        user_file=file,
        search_settings=settings,
        regulatory_chunks=tuple(rows),
        indexing_items=tuple(items),
    )
    authority = OwnedAuthority(file.id)
    index = PublicationIndexSnapshot(
        index_name=settings.index_name,
        index_uuid="physical-uuid",
        search_settings_id=settings.id,
        model_provider="openrouter",
        model_name=settings.model_name,
        vector_dimension=3,
        embedding_config_sha256="a" * 64,
        multitenant=False,
    )
    manifest = WriterPublicationManifest(
        id=uuid4(),
        scope=authority.scope,
        user_file_id=file.id,
        kind="durable",
        durable_job_id=job.id,
        durable_input_sha256=durable_publication_input_digest(runtime),
        canonical_before_sha256="b" * 64,
        indexes=[index],
        previous_binding_ids=[],
        bindings=[],
    )
    monkeypatch.setattr(owned_publication, "PublicationStore", authority.for_scope)
    monkeypatch.setattr(
        owned_publication, "load_owned_durable_runtime", lambda *_a, **_k: runtime
    )
    monkeypatch.setattr(
        owned_publication, "pending_writer_manifest", lambda _owner: manifest
    )
    monkeypatch.setattr(
        owned_publication,
        "prepare_durable_writer_manifest",
        lambda *_a: pytest.fail("retry must reuse staged manifest"),
    )
    transport = MagicMock()
    monkeypatch.setattr(owned_publication, "ElasticsearchClient", lambda: transport)
    execute = MagicMock()
    monkeypatch.setattr(owned_publication, "execute_writer_publication", execute)
    result = owned_publication.execute_owned_durable_stage(
        job_id=job.id,
        file_id=file.id,
        generation=11,
        stage=stage,
        tenant_id="tenant-a",
        caller_row_ids={row.id for row in rows},
        caller_item_ids={item.id for item in items},
    )
    assert result is runtime
    execute.assert_called_once_with(
        authority.owner,
        transport.__enter__().publication_client(),
        None,
        hidden=stage is not RegulatoryIndexingStage.PUBLISH,
        activate=stage is RegulatoryIndexingStage.PUBLISH,
        durable_generation=11,
    )
    assert authority.released == [authority.owner]
    # The persisted manifest commits to actual source/settings/vector/input state.
    items[0].vector = [0.9, 0.8, 0.7]
    with pytest.raises(ValueError, match="differs from staged inventory"):
        owned_publication.execute_owned_durable_stage(
            job_id=job.id,
            file_id=file.id,
            generation=11,
            stage=stage,
            tenant_id="tenant-a",
            caller_row_ids={row.id for row in rows},
            caller_item_ids={item.id for item in items},
        )
    assert execute.call_count == 1
    assert authority.released == [authority.owner, authority.owner]


def test_hidden_source_preserves_original_context_image_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, file, settings, rows, items = _fixture()
    rows[0].projection_ordinal = 1_000_000_042
    rows[0].chunk_metadata.update(
        image_file_id="old-image", source_links={"0": "https://example.gov/annex"}
    )
    monkeypatch.setattr(publisher, "get_access_for_user_files", lambda *_a: {})
    monkeypatch.setattr(
        publisher,
        "fetch_user_project_ids_for_user_files",
        lambda *_a: {str(file.id): [17]},
    )
    monkeypatch.setattr(
        publisher, "fetch_persona_ids_for_user_files", lambda *_a: {str(file.id): [23]}
    )
    monkeypatch.setattr(
        publisher,
        "fetch_document_set_names_for_user_files",
        lambda *_a: {str(file.id): ["Regulations"]},
    )
    chunks = publisher._build_hidden_chunks(
        job_id=job.id,
        user_file_id=file.id,
        user_file_name=file.name,
        rows=rows,
        items=items,
        snapshot=_snapshot(),
        tenant_id="tenant-a",
        db_session=MagicMock(spec=Session),
    )
    chunk = next(chunk for chunk in chunks if chunk.regulatory_chunk_id == rows[0].id)
    assert chunk.chunk_id == 1_000_000_042 and chunk.hidden is True
    assert chunk.image_file_id == "old-image" and chunk.source_links == {
        0: "https://example.gov/annex"
    }
    assert chunk.doc_summary == "Generated context."
    assert chunk.embeddings.full_embedding == items[0].vector
    assert (
        chunk.document_sets == {"Regulations"}
        and chunk.user_project == [17]
        and chunk.personas == [23]
    )
    base = IndexChunk.model_validate(
        {name: getattr(chunk, name) for name in IndexChunk.model_fields}
    )
    legacy = DocMetadataAwareIndexChunk.from_index_chunk(
        index_chunk=base,
        access=chunk.access,
        document_sets=chunk.document_sets,
        user_project=chunk.user_project,
        personas=chunk.personas,
        boost=chunk.boost,
        aggregated_chunk_boost_factor=chunk.aggregated_chunk_boost_factor,
        tenant_id=chunk.tenant_id,
    )
    assert legacy.hidden is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "changed-model"),
        ("model_dim", 4),
        ("index_name", "replacement-index"),
    ],
)
def test_manifest_rejects_locked_search_settings_drift_before_index_access(
    field: str, value: str | int
) -> None:
    from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingRuntime
    from onyx.regulatory.indexing_jobs.owned_publication import (
        prepare_durable_writer_manifest,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
        writer_inputs,
    )

    job, file, settings, rows, items = _fixture()
    setattr(settings, field, value)
    runtime = RegulatoryIndexingRuntime(
        job=job,
        user_file=file,
        search_settings=settings,
        regulatory_chunks=tuple(rows),
        indexing_items=tuple(items),
    )
    client = MagicMock()
    with pytest.raises(ValueError, match="no longer matches"):
        prepare_durable_writer_manifest(
            OwnedAuthority(file.id).owner, client, writer_inputs(file.id, rows), runtime
        )
    client.indices.get.assert_not_called()
