"""A real PG/ES durable job preserves dated identities through batch and ES recovery."""

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
from sqlalchemy import delete, select
from sqlalchemy.schema import DropSchema

from onyx.db import regulatory_indexing_jobs as repository
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.enums import IndexModelStatus, RegulatoryIndexingStage, UserFileStatus
from onyx.db.models import (
    CloudEmbeddingProvider,
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    UserFile,
)
from onyx.document_index.elasticsearch.schema import DocumentSchema
from onyx.regulatory.indexing_jobs import orchestrator, preparation, publisher
from onyx.regulatory.indexing_jobs.contextual import apply_contextual_results
from onyx.regulatory.indexing_jobs.models import (
    OpenRouterBatchConfig,
    RegulatoryInputHashVersion,
)
from onyx.regulatory.indexing_jobs.openrouter_batch import (
    OpenRouterBatchJobStatus,
    OpenRouterBatchState,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchResult
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    create_owned_file,
)
from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import _CharacterTokenizer
from tests.unit.onyx.regulatory.indexing_jobs.test_publisher import _snapshot


@pytest.mark.parametrize(
    "completion_mode", ["publish", "cancel_staged", "delete_staged", "legacy_repair"]
)
def test_owned_durable_batch_resumes_dated_publication(
    monkeypatch: pytest.MonkeyPatch, completion_mode: str
) -> None:
    if not MULTI_TENANT:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                str(Path(__file__).resolve()),
                "-k",
                completion_mode,
            ],
            env={**os.environ, "MULTI_TENANT": "true"},
            check=True,
        )
        return
    from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT

    assert POSTGRES_HOST in {"localhost", "127.0.0.1"} and str(POSTGRES_PORT) == "25432"
    tenant = "tenant_5d_durable_" + uuid4().hex
    index_name = "annex-5d-durable-" + uuid4().hex
    client = Elasticsearch("http://127.0.0.1:29200")
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant)
    backend = Path(__file__).resolve().parents[3]
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-x",
                f"schemas={tenant}",
                "upgrade",
                "head",
            ],
            cwd=backend,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        client.indices.create(
            index=index_name, mappings=DocumentSchema.get_document_schema(3, True)
        )
        with create_owned_file(tenant) as file_id:
            with get_session_with_tenant(tenant_id=tenant) as session:
                session.execute(delete(SearchSettings))
                provider = CloudEmbeddingProvider(
                    provider_type=EmbeddingProvider.OPENROUTER,
                    api_url=None,
                    api_key=None,
                )
                session.add(provider)
                settings = SearchSettings(
                    model_name="openai/text-embedding-3-large",
                    model_dim=3,
                    normalize=True,
                    status=IndexModelStatus.PRESENT,
                    index_name=index_name,
                    provider_type=EmbeddingProvider.OPENROUTER,
                    enable_contextual_rag=True,
                    query_prefix="",
                    passage_prefix="",
                )
                session.add(settings)
                session.flush()
                config = OpenRouterBatchConfig(
                    api_url="https://openrouter.ai/api/beta/batches",
                    model_name=settings.model_name,
                    effective_dimension=3,
                    max_inputs=1,
                )
                rows = list(
                    session.scalars(
                        select(RegulatoryChunk)
                        .where(RegulatoryChunk.user_file_id == file_id)
                        .order_by(RegulatoryChunk.position)
                    )
                )
                rows[1].validity_start_date = date(2030, 1, 1)
                snapshot = _snapshot().model_copy(
                    update={
                        "input_hash_version": RegulatoryInputHashVersion.CHUNK_ROWS_V3,
                        "input_content_hash": preparation.regulatory_chunks_content_hash(
                            rows
                        ),
                        "search_settings_id": settings.id,
                        "index_name": index_name,
                        "embedding_model_name": settings.model_name,
                        "embedding_provider": EmbeddingProvider.OPENROUTER,
                        "openrouter_batch": config,
                    }
                )
                file = session.get(UserFile, file_id)
                assert file is not None
                file.status = UserFileStatus.INDEXING
                file.regulatory_chunk_generation_hash = snapshot.chunk_generation_hash
                job = RegulatoryIndexingJob(
                    id=uuid4(),
                    user_file_id=file_id,
                    content_hash=snapshot.input_content_hash,
                    chunk_generation_hash=snapshot.chunk_generation_hash,
                    search_settings_id=snapshot.search_settings_id,
                    prompt_hash=snapshot.prompt_hash,
                    config_snapshot=snapshot.model_dump(mode="json"),
                    status="RUNNING",
                    stage="PREPARING",
                    lease_generation=1,
                )
                if completion_mode != "legacy_repair":
                    session.add(job)
                session.commit()
                job_id = job.id
                monkeypatch.setattr(
                    preparation,
                    "get_tokenizer",
                    lambda *_args, **_kwargs: _CharacterTokenizer(),
                )
                monkeypatch.setattr(
                    preparation,
                    "get_contextual_token_budget_tokenizer",
                    lambda **_kwargs: _CharacterTokenizer(),
                )
                if completion_mode == "legacy_repair":
                    actual_create = preparation.create_or_get_regulatory_indexing_job

                    def create_under_owner(*args, **kwargs):
                        from onyx.db.regulatory_publication import PublicationStore

                        owner = kwargs.get("publication_owner")
                        assert owner is not None, (
                            "durable creation locked the file without publication ownership"
                        )
                        PublicationStore(owner.scope).heartbeat(
                            owner, ttl=timedelta(minutes=2)
                        )
                        return actual_create(*args, **kwargs)

                    with monkeypatch.context() as creation:
                        creation.setattr(
                            preparation,
                            "resolve_regulatory_indexing_snapshot",
                            lambda *_args, **_kwargs: snapshot,
                        )
                        creation.setattr(
                            preparation,
                            "create_or_get_regulatory_indexing_job",
                            create_under_owner,
                        )
                        job_id = (
                            preparation.prepare_regulatory_indexing_job_from_chunks(
                                file_id, tenant, session
                            )
                        )
                    job = session.get(RegulatoryIndexingJob, job_id)
                    assert job is not None
                else:
                    preparation.prepare_claimed_regulatory_indexing_job_from_chunks(
                        job_id=job_id,
                        expected_generation=1,
                        tenant_id=tenant,
                        db_session=session,
                    )
                job.status, job.stage = "RUNNING", "CONTEXT_APPLY"
                session.commit()
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None and len(runtime.indexing_items) == 3
                results = {
                    item.request_hash: VertexBatchResult(
                        request_hash=item.request_hash, context="Retained context."
                    )
                    for item in runtime.indexing_items
                    if item.status == "PENDING"
                }
                apply_contextual_results(
                    job,
                    runtime.regulatory_chunks,
                    runtime.indexing_items,
                    results,
                    _CharacterTokenizer(),
                    session,
                )
                job.stage = "EMBEDDING"
                session.commit()
                if completion_mode == "legacy_repair":
                    from onyx.regulatory.indexing_jobs import owned_publication

                    session.expire_all()
                    contextual_items = [
                        item
                        for item in runtime.indexing_items
                        if item.status == "CONTEXT_READY"
                    ]
                    legacy, missing = contextual_items
                    retained_id, retained_context = (
                        legacy.id,
                        _context(legacy)["contextual_text"],
                    )
                    legacy.context = {
                        key: value
                        for key, value in _context(legacy).items()
                        if key != "raw_contextual_text"
                    }
                    legacy.projection_id = legacy.projection_ordinal = (
                        legacy.projection_input
                    ) = None
                    legacy.effective_start = legacy.effective_end = None
                    missing.context = {
                        key: value
                        for key, value in _context(missing).items()
                        if key != "context_input"
                    }
                    session.commit()
                    job_id, generation = job.id, job.lease_generation
                    original_contexts = {
                        item.id: item.context for item in runtime.indexing_items
                    }
                    job.vertex_submission_state = "SUBMITTING"
                    session.commit()
                    session.rollback()
                    from onyx.regulatory.indexing_jobs.models import (
                        IndexingPublicationIndeterminateError,
                    )

                    with pytest.raises(
                        IndexingPublicationIndeterminateError,
                        match="contextual submission must be reconciled",
                    ):
                        owned_publication.repair_owned_durable_items(
                            job_id=job_id,
                            user_file_id=file_id,
                            expected_generation=generation,
                            stage=RegulatoryIndexingStage.EMBEDDING,
                            tenant_id=tenant,
                        )
                    session.expire_all()
                    assert original_contexts == {
                        item.id: item.context for item in runtime.indexing_items
                    }
                    job.vertex_submission_state = "SUBMITTED"
                    job.remote_vertex_job_name = "batches/fixture-completed-context"
                    job.vertex_input_uri, job.vertex_output_uri = (
                        "files/fixture-input",
                        "files/fixture-output",
                    )
                    session.commit()
                    session.rollback()
                    with monkeypatch.context() as repair_transport:
                        repair_transport.setattr(
                            orchestrator,
                            "compute_regulatory_chunk_generation_hash",
                            lambda **_kwargs: snapshot.chunk_generation_hash,
                        )
                        repair_transport.setattr(
                            orchestrator,
                            "validate_snapshot_for_stage",
                            lambda *_args: None,
                        )

                        def refuse_embedding_before_context_repair(*_args, **_kwargs):
                            raise AssertionError(
                                "embedding started before missing context proof recovery"
                            )

                        repair_transport.setattr(
                            orchestrator,
                            "_openrouter_embedding_batch",
                            refuse_embedding_before_context_repair,
                        )
                        orchestrator._execute_claimed_step_impl(
                            runtime,
                            tenant_id=tenant,
                            db_session=session,
                            now=datetime.now(timezone.utc),
                        )
                    session.expire_all()
                    assert (
                        legacy.id == retained_id
                        and _context(legacy)["contextual_text"] == retained_context
                    )
                    assert "raw_contextual_text" not in _context(legacy)
                    assert legacy.projection_id is not None
                    assert (
                        missing.status == "PENDING"
                        and "contextual_text" not in _context(missing)
                    )
                    assert job.stage == "CONTEXT_APPLY" and job.status == "QUEUED"
                    assert job.vertex_submission_state == "RETRY_CLEANUP_REQUIRED"
                    assert repository.claim_regulatory_indexing_job(
                        session,
                        job_id=job_id,
                        expected_stage=RegulatoryIndexingStage.CONTEXT_APPLY,
                        expected_generation=job.lease_generation,
                        now=datetime.now(timezone.utc),
                    )
                    cleanup = []

                    class CompletedContextGateway:
                        def delete(self, name):
                            cleanup.append(name)

                        def cleanup(self, name):
                            cleanup.append(name)

                    with monkeypatch.context() as cleanup_transport:
                        cleanup_transport.setattr(
                            orchestrator,
                            "_build_vertex_gateway",
                            lambda *_args, **_kwargs: CompletedContextGateway(),
                        )
                        orchestrator._context_apply(
                            runtime,
                            tenant_id=tenant,
                            db_session=session,
                            now=datetime.now(timezone.utc),
                        )
                    session.expire_all()
                    assert cleanup == [
                        "batches/fixture-completed-context",
                        "files/fixture-input",
                        "files/fixture-output",
                    ]
                    assert job.stage == "CONTEXT_SUBMIT" and job.status == "QUEUED"
                    assert repository.claim_regulatory_indexing_job(
                        session,
                        job_id=job_id,
                        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
                        expected_generation=job.lease_generation,
                        now=datetime.now(timezone.utc),
                    )
                    from onyx.regulatory.indexing_jobs.contextual import (
                        build_contextual_requests,
                    )

                    requests = build_contextual_requests(
                        job,
                        runtime.regulatory_chunks,
                        runtime.indexing_items,
                        embedding_tokenizer=_CharacterTokenizer(),
                        contextual_tokenizer=_CharacterTokenizer(),
                    )
                    assert len(requests) == 1
                    job.stage = "CONTEXT_APPLY"
                    session.commit()
                    apply_contextual_results(
                        job,
                        runtime.regulatory_chunks,
                        runtime.indexing_items,
                        {
                            request.request_hash: VertexBatchResult(
                                request_hash=request.request_hash,
                                context="Rebuilt context.",
                            )
                            for request in requests
                        },
                        _CharacterTokenizer(),
                        session,
                    )
                    job.stage = "EMBEDDING"
                    session.commit()
            submitted = []

            class BatchGateway:
                def submit(self, requests, *, submission_key):
                    submitted.append((submission_key, requests))
                    return OpenRouterBatchState(
                        remote_batch_id=str(len(submitted)),
                        status=OpenRouterBatchJobStatus.PENDING,
                    )

                def get(self, remote_batch_id):
                    requests = submitted[int(remote_batch_id) - 1][1]
                    return OpenRouterBatchState(
                        remote_batch_id=remote_batch_id,
                        status=OpenRouterBatchJobStatus.SUCCEEDED,
                        results=[
                            {
                                "custom_id": request.custom_id,
                                "response": {
                                    "status_code": 200,
                                    "body": {
                                        "model": snapshot.embedding_model_name,
                                        "data": [
                                            {
                                                "index": number,
                                                "embedding": [0.25, 0.5, 0.75],
                                            }
                                            for number, _ in enumerate(request.inputs)
                                        ],
                                    },
                                },
                            }
                            for request in requests
                        ],
                    )

            monkeypatch.setattr(
                orchestrator,
                "_build_openrouter_gateway",
                lambda _runtime: BatchGateway(),
            )
            first_proven = None
            identities = None
            for delivery in range(10):
                with get_session_with_tenant(tenant_id=tenant) as session:
                    runtime = repository.get_regulatory_indexing_runtime(
                        session, job_id
                    )
                    assert runtime is not None
                    job = runtime.job
                    if job.stage == "INDEX_WRITE":
                        break
                    if job.status == "QUEUED":
                        assert repository.claim_regulatory_indexing_job(
                            session,
                            job_id=job_id,
                            expected_stage=RegulatoryIndexingStage.EMBEDDING,
                            expected_generation=job.lease_generation,
                            now=datetime.now(timezone.utc),
                        )
                        runtime = repository.get_regulatory_indexing_runtime(
                            session, job_id
                        )
                        assert runtime is not None
                    orchestrator._openrouter_embedding_batch(
                        runtime,
                        tenant_id=tenant,
                        db_session=session,
                        now=datetime.now(timezone.utc),
                    )
                    if delivery == 1:
                        session.expire_all()
                        items = list(
                            session.scalars(
                                select(RegulatoryIndexingItem).where(
                                    RegulatoryIndexingItem.job_id == job_id
                                )
                            )
                        )
                        first_proven = next(
                            item.id for item in items if item.status == "EMBEDDED"
                        )
                        legacy = next(
                            item for item in items if item.status == "CONTEXT_READY"
                        )
                        legacy.vector, legacy.status = [0.1, 0.2, 0.3], "EMBEDDED"
                        legacy.context = {
                            key: value
                            for key, value in _context(legacy).items()
                            if not key.startswith("embedding_")
                        }
                        identities = {
                            (item.id, item.projection_id, item.projection_ordinal)
                            for item in items
                        }
                        session.commit()
            assert len(submitted) == 3 and first_proven is not None
            with get_session_with_tenant(tenant_id=tenant) as session:
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None and runtime.job.stage == "INDEX_WRITE"
                assert identities == {
                    (item.id, item.projection_id, item.projection_ordinal)
                    for item in runtime.indexing_items
                }
                assert repository.claim_regulatory_indexing_job(
                    session,
                    job_id=job_id,
                    expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                    expected_generation=runtime.job.lease_generation,
                    now=datetime.now(timezone.utc),
                )
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                expected_batches = 3
                if completion_mode == "legacy_repair":
                    legacy_vector = runtime.indexing_items[0]
                    legacy_vector.context = {
                        key: value
                        for key, value in _context(legacy_vector).items()
                        if key
                        not in {"embedding_receipt", "embedding_vector_receipt_sha256"}
                    }
                    session.commit()
                    with monkeypatch.context() as repair_transport:
                        repair_transport.setattr(
                            orchestrator,
                            "compute_regulatory_chunk_generation_hash",
                            lambda **_kwargs: snapshot.chunk_generation_hash,
                        )
                        repair_transport.setattr(
                            orchestrator,
                            "validate_snapshot_for_stage",
                            lambda *_args: None,
                        )

                        def refuse_stage_before_receipt_repair(*_args, **_kwargs):
                            raise AssertionError(
                                "index staging started before legacy vector receipt recovery"
                            )

                        repair_transport.setattr(
                            orchestrator,
                            "stage_regulatory_job_in_index",
                            refuse_stage_before_receipt_repair,
                        )
                        orchestrator._execute_claimed_step_impl(
                            runtime,
                            tenant_id=tenant,
                            db_session=session,
                            now=datetime.now(timezone.utc),
                        )
                    session.expire_all()
                    assert (
                        runtime.job.stage == "EMBEDDING"
                        and runtime.job.status == "QUEUED"
                    )
                    for _ in range(4):
                        runtime = repository.get_regulatory_indexing_runtime(
                            session, job_id
                        )
                        assert runtime is not None
                        if runtime.job.stage == "INDEX_WRITE":
                            break
                        assert repository.claim_regulatory_indexing_job(
                            session,
                            job_id=job_id,
                            expected_stage=RegulatoryIndexingStage.EMBEDDING,
                            expected_generation=runtime.job.lease_generation,
                            now=datetime.now(timezone.utc),
                        )
                        runtime = repository.get_regulatory_indexing_runtime(
                            session, job_id
                        )
                        assert runtime is not None
                        orchestrator._openrouter_embedding_batch(
                            runtime,
                            tenant_id=tenant,
                            db_session=session,
                            now=datetime.now(timezone.utc),
                        )
                    assert len(submitted) == 4
                    expected_batches = 4
                    assert identities == {
                        (item.id, item.projection_id, item.projection_ordinal)
                        for item in runtime.indexing_items
                    }
                    assert repository.claim_regulatory_indexing_job(
                        session,
                        job_id=job_id,
                        expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                        expected_generation=runtime.job.lease_generation,
                        now=datetime.now(timezone.utc),
                    )
                    runtime = repository.get_regulatory_indexing_runtime(
                        session, job_id
                    )
                    assert runtime is not None
                from onyx.document_index.elasticsearch.elasticsearch_document_index import (
                    ElasticsearchDocumentIndex,
                )

                def reject_unfenced(*_args, **_kwargs):
                    raise AssertionError(
                        "durable publication bypassed shared ES ownership"
                    )

                monkeypatch.setattr(
                    ElasticsearchDocumentIndex, "index", reject_unfenced
                )
                from onyx.db.regulatory_publication import PublicationStore
                from onyx.document_index.publication_models import PublicationScope
                from onyx.regulatory.amendments.annexes import config as annex_config

                authority = PublicationStore(
                    PublicationScope(
                        tenant_id=tenant,
                        environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
                        database_identity=annex_config.ANNEX_DATABASE_IDENTITY,
                    )
                )
                assert runtime.search_settings is not None
                result = publisher.stage_regulatory_job_in_index(
                    job=runtime.job,
                    user_file=runtime.user_file,
                    rows=runtime.regulatory_chunks,
                    items=runtime.indexing_items,
                    search_settings=runtime.search_settings,
                    tenant_id=tenant,
                    db_session=session,
                )
                assert (
                    result.canonical_chunk_count == 2
                    and result.embedded_item_count == 3
                )
                from elasticsearch import BadRequestError

                from onyx.db.regulatory_writer_publication import (
                    pending_writer_manifest,
                )
                from onyx.document_index.elasticsearch.publication import (
                    FencedPublicationIndex,
                )
                from onyx.document_index.publication_models import (
                    FrozenPublicationProjection,
                )

                owner = authority.acquire(
                    file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
                )
                manifest = pending_writer_manifest(owner)
                assert (
                    manifest is not None and authority.reservations(owner).gate_closed
                )
                unseen = authority.allocate(owner, "old-larger-durable-job")
                old_reservations = authority.reservations(owner)
                authority.release(owner)
                hits = client.search(
                    index=index_name,
                    query={
                        "bool": {
                            "filter": [
                                {"term": {"document_id": str(file_id)}},
                                {"term": {"hidden": True}},
                                {"term": {"publication_tombstone": False}},
                            ]
                        }
                    },
                    size=10,
                )["hits"]["hits"]
                assert len(hits) == 3
                planned = manifest.bindings[0].projection
                abandoned_source = json.loads(planned.source_json)
                abandoned_source["chunk_index"] = unseen
                abandoned = FrozenPublicationProjection(
                    ordinal=unseen,
                    context_projection_id="abandoned-old-plan",
                    source_json=json.dumps(abandoned_source),
                    embedding_inputs=planned.embedding_inputs,
                    embedding_config_json=planned.embedding_config_json,
                )
                if completion_mode in {"cancel_staged", "delete_staged"}:
                    from onyx.db.enums import RegulatoryIndexingCancellationIntent
                    from onyx.regulatory import writer_publication

                    if completion_mode == "delete_staged":
                        from onyx.background.celery.tasks.regulatory_indexing import (
                            tasks as enqueue,
                        )

                        deliveries = []
                        monkeypatch.setattr(
                            enqueue,
                            "enqueue_regulatory_indexing_step",
                            lambda *_args, **kwargs: deliveries.append(kwargs),
                        )
                        writer_publication.delete_owned_file(file_id, tenant)
                        assert deliveries and deliveries[0]["job_id"] == job_id
                        session.expire_all()
                        assert runtime.user_file.status == UserFileStatus.DELETING
                    else:
                        assert repository.request_regulatory_indexing_cancellation(
                            session,
                            job_id=job_id,
                            expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                            expected_generation=runtime.job.lease_generation,
                            cancellation_intent=RegulatoryIndexingCancellationIntent.USER_CANCEL,
                            now=datetime.now(timezone.utc),
                        )
                    # Provider cleanup is completed before this ES compensation checkpoint.
                    runtime.job.cancellation_phase = "INDEX_DELETE"
                    session.commit()
                    finalize = writer_publication.finalize_writer_publication

                    def fail_cancel_activation(*_args, **_kwargs):
                        raise RuntimeError("cancel activation interrupted")

                    monkeypatch.setattr(
                        writer_publication,
                        "finalize_writer_publication",
                        fail_cancel_activation,
                    )
                    with pytest.raises(
                        RuntimeError, match="cancel activation interrupted"
                    ):
                        orchestrator._execute_cancellation_phase(
                            runtime,
                            tenant_id=tenant,
                            db_session=session,
                            now=datetime.now(timezone.utc),
                        )
                    assert authority.unavailable(authority.observe(), (file_id,))
                    with pytest.raises(BadRequestError):
                        FencedPublicationIndex(client, manifest.indexes[0]).upsert(
                            old_reservations, abandoned
                        )
                    session.expire_all()
                    assert runtime.job.status == "CANCELLING"
                    assert repository.claim_regulatory_indexing_job(
                        session,
                        job_id=job_id,
                        expected_stage=RegulatoryIndexingStage.INDEX_WRITE,
                        expected_generation=runtime.job.lease_generation,
                        now=datetime.now(timezone.utc),
                    )
                    monkeypatch.setattr(
                        writer_publication, "finalize_writer_publication", finalize
                    )
                    orchestrator._execute_cancellation_phase(
                        runtime,
                        tenant_id=tenant,
                        db_session=session,
                        now=datetime.now(timezone.utc),
                    )
                    session.expire_all()
                    assert runtime.job.status == "CANCELLED"
                    assert runtime.user_file.status == (
                        UserFileStatus.DELETING
                        if completion_mode == "delete_staged"
                        else UserFileStatus.CANCELED
                    )
                    assert not authority.unavailable(authority.observe(), (file_id,))
                    assert (
                        client.count(
                            index=index_name,
                            query={
                                "bool": {
                                    "filter": [
                                        {"term": {"document_id": str(file_id)}},
                                        {"term": {"publication_tombstone": False}},
                                    ]
                                }
                            },
                        )["count"]
                        == 0
                    )
                    assert len(submitted) == 3
                    return
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                orchestrator._advance(
                    runtime,
                    session,
                    next_stage=RegulatoryIndexingStage.VERIFY,
                    now=datetime.now(timezone.utc),
                )
                assert repository.claim_regulatory_indexing_job(
                    session,
                    job_id=job_id,
                    expected_stage=RegulatoryIndexingStage.VERIFY,
                    expected_generation=runtime.job.lease_generation,
                    now=datetime.now(timezone.utc),
                )
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                assert (
                    publisher.verify_staged_regulatory_job(
                        job=runtime.job,
                        user_file=runtime.user_file,
                        rows=runtime.regulatory_chunks,
                        items=runtime.indexing_items,
                        search_settings=runtime.search_settings,
                        db_session=session,
                    ).embedded_item_count
                    == 3
                )
                with pytest.raises(BadRequestError):
                    FencedPublicationIndex(client, manifest.indexes[0]).upsert(
                        old_reservations, abandoned
                    )
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                orchestrator._advance(
                    runtime,
                    session,
                    next_stage=RegulatoryIndexingStage.PUBLISH,
                    now=datetime.now(timezone.utc),
                )
                assert repository.claim_regulatory_indexing_job(
                    session,
                    job_id=job_id,
                    expected_stage=RegulatoryIndexingStage.PUBLISH,
                    expected_generation=runtime.job.lease_generation,
                    now=datetime.now(timezone.utc),
                )
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                assert runtime.search_settings is not None
                target = runtime.search_settings
                configured_model = target.model_name
                target.model_name = configured_model + "-changed"
                session.commit()
                with pytest.raises(ValueError, match="checkpoint"):
                    publisher.publish_regulatory_job(
                        job=runtime.job,
                        user_file=runtime.user_file,
                        rows=runtime.regulatory_chunks,
                        items=runtime.indexing_items,
                        search_settings=runtime.search_settings,
                        db_session=session,
                    )
                target.model_name = configured_model
                session.commit()
                from onyx.regulatory import writer_publication

                finalize = writer_publication.finalize_writer_publication

                def interrupted(*_args, **_kwargs):
                    raise RuntimeError("interrupted durable activation")

                monkeypatch.setattr(
                    writer_publication, "finalize_writer_publication", interrupted
                )
                with pytest.raises(
                    RuntimeError, match="interrupted durable activation"
                ):
                    publisher.publish_regulatory_job(
                        job=runtime.job,
                        user_file=runtime.user_file,
                        rows=runtime.regulatory_chunks,
                        items=runtime.indexing_items,
                        search_settings=runtime.search_settings,
                        db_session=session,
                    )
                session.expire_all()
                assert (
                    runtime.job.status == "RUNNING"
                    and runtime.user_file.status == UserFileStatus.INDEXING
                )
                assert file_id in authority.unavailable(authority.observe(), (file_id,))
                assert repository.schedule_regulatory_indexing_retry(
                    session,
                    job_id=job_id,
                    expected_stage=RegulatoryIndexingStage.PUBLISH,
                    expected_generation=runtime.job.lease_generation,
                    next_retry_at=datetime.now(timezone.utc),
                    error_code="interrupted",
                    error_message="fixture interruption",
                )
                assert repository.claim_regulatory_indexing_job(
                    session,
                    job_id=job_id,
                    expected_stage=RegulatoryIndexingStage.PUBLISH,
                    expected_generation=runtime.job.lease_generation,
                    now=datetime.now(timezone.utc),
                )
                runtime = repository.get_regulatory_indexing_runtime(session, job_id)
                assert runtime is not None
                monkeypatch.setattr(
                    writer_publication, "finalize_writer_publication", finalize
                )
                assert (
                    publisher.publish_regulatory_job(
                        job=runtime.job,
                        user_file=runtime.user_file,
                        rows=runtime.regulatory_chunks,
                        items=runtime.indexing_items,
                        search_settings=runtime.search_settings,
                        db_session=session,
                    )
                    == publisher.PublishOutcome.COMPLETED
                )
                session.expire_all()
                assert (
                    runtime.job.status == "SUCCEEDED"
                    and runtime.user_file.status == UserFileStatus.COMPLETED
                )
                assert (
                    runtime.user_file.chunk_count == 2
                    and len(submitted) == expected_batches
                )
                assert not authority.unavailable(authority.observe(), (file_id,))
                from onyx.db.regulatory_annex_publication import (
                    load_file_temporal_bindings,
                )

                bindings = load_file_temporal_bindings(session, file_id)
                assert {binding.id for binding in bindings} == {
                    item.projection_id for item in runtime.indexing_items
                }
                assert len(bindings) == 3
                assert all(binding.projection.embedding_inputs for binding in bindings)
                from onyx.context.search.models import IndexFilters
                from onyx.db.enums import EmbeddingPrecision
                from onyx.document_index.interfaces_new import TenantState

                index = ElasticsearchDocumentIndex(
                    TenantState(tenant_id=tenant, multitenant=True),
                    index_name,
                    3,
                    EmbeddingPrecision.FLOAT,
                )
                filters = IndexFilters(
                    access_control_list=None, as_of_date=date(2031, 1, 1)
                )
                assert len(index.keyword_retrieval("existing", filters, 10)) == 2
                assert (
                    len(index.semantic_retrieval([0.25, 0.5, 0.75], filters, 10)) == 2
                )
                for field, value in (
                    ("model_name", "changed-model"),
                    ("model_dim", 4),
                    ("passage_prefix", "passage: "),
                    ("query_prefix", "query: "),
                ):
                    original = getattr(runtime.search_settings, field)
                    setattr(runtime.search_settings, field, value)
                    session.commit()
                    with pytest.raises(ValueError, match="runtime encoder differs"):
                        index.keyword_retrieval("existing", filters, 10)
                    setattr(runtime.search_settings, field, original)
                    session.commit()
                provider = session.get(
                    CloudEmbeddingProvider, EmbeddingProvider.OPENROUTER
                )
                assert provider is not None
                provider.api_url = "https://custom.example"
                session.commit()
                with pytest.raises(ValueError, match="runtime encoder differs"):
                    index.keyword_retrieval("existing", filters, 10)
                provider.api_url = None
                session.commit()
                retained_bindings = {
                    binding.id: binding.model_dump_json() for binding in bindings
                }
                stale = RegulatoryIndexingJob(
                    id=uuid4(),
                    user_file_id=file_id,
                    content_hash="a" * 64,
                    chunk_generation_hash=snapshot.chunk_generation_hash,
                    search_settings_id=snapshot.search_settings_id,
                    prompt_hash=snapshot.prompt_hash,
                    config_snapshot=snapshot.model_dump(mode="json"),
                    status="CANCELLING",
                    stage="INDEX_WRITE",
                    lease_generation=1,
                    cancellation_phase="INDEX_DELETE",
                    cancellation_intent="USER_CANCEL",
                )
                session.add(stale)
                session.commit()
                stale_id = stale.id
                monkeypatch.setattr(
                    ElasticsearchDocumentIndex, "delete", reject_unfenced
                )
                stale_runtime = repository.get_regulatory_indexing_runtime(
                    session, stale_id
                )
                assert stale_runtime is not None
                orchestrator._execute_cancellation_phase(
                    stale_runtime,
                    tenant_id=tenant,
                    db_session=session,
                    now=datetime.now(timezone.utc),
                )
                session.expire_all()
                assert stale.status == "CANCELLED"
                assert runtime.user_file.status == UserFileStatus.COMPLETED
                assert retained_bindings == {
                    binding.id: binding.model_dump_json()
                    for binding in load_file_temporal_bindings(session, file_id)
                }
                assert len(index.keyword_retrieval("existing", filters, 10)) == 2
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.close()
        with SqlEngine.get_engine().begin() as connection:
            connection.execute(DropSchema(tenant, cascade=True, if_exists=True))


def _context(item: RegulatoryIndexingItem) -> dict[str, object]:
    context = item.context
    assert context is not None
    return context
