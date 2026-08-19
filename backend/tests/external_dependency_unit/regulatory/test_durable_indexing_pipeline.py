from __future__ import annotations

import datetime
import json
import threading
from collections.abc import Generator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as job_repository
from onyx.db.enums import (
    EmbeddingPrecision,
    IndexModelStatus,
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
    UserFileStatus,
)
from onyx.db.models import (
    CloudEmbeddingProvider,
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    User,
    UserFile,
)
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.schema import (
    DocumentSchema,
    get_elasticsearch_doc_chunk_id,
)
from onyx.document_index.interfaces_new import TenantState
from onyx.natural_language_processing import search_nlp_models
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.indexing_jobs import orchestrator, preparation, publisher
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.orchestrator import (
    OrchestrationOutcome,
    run_preclaimed_regulatory_indexing_step,
    run_regulatory_indexing_step,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
    build_vertex_jsonl,
)
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.conftest import create_test_user

_DIMENSION = 1024
_TENANT_ID = "public"


class _CharacterTokenizer(BaseTokenizer):
    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _DeterministicVertexGateway(VertexBatchGateway):
    def __init__(self) -> None:
        self.submissions: list[list[VertexBatchRequest]] = []
        self.cancelled: list[str] = []
        self.cleaned: list[str] = []

    def submit(self, requests: Sequence[VertexBatchRequest]) -> VertexBatchState:
        submission = list(requests)
        self.submissions.append(submission)
        suffix = len(self.submissions)
        return VertexBatchState(
            remote_job_name=f"jobs/deterministic-{suffix}",
            status=VertexBatchJobStatus.PENDING,
            input_uri=f"gs://disposable/input-{suffix}.jsonl",
            output_uri=f"gs://disposable/output-{suffix}",
        )

    def get(self, remote_job_name: str) -> VertexBatchState:
        return VertexBatchState(
            remote_job_name=remote_job_name,
            status=VertexBatchJobStatus.SUCCEEDED,
            output_uri=f"gs://disposable/{remote_job_name.replace('/', '-')}",
        )

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        del submission_key
        return None

    def read_results(self, output_uri: str) -> str:
        del output_uri
        current = self.submissions[-1]
        returned = current[:-1] if len(current) > 1 else current
        lines: list[str] = []
        for request in reversed(returned):
            request_payload = json.loads(build_vertex_jsonl([request]))["request"]
            lines.append(
                json.dumps(
                    {
                        "status": "",
                        "request": request_payload,
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {
                                                "text": (
                                                    "Deterministic context "
                                                    f"{request.request_hash[:8]}"
                                                )
                                            }
                                        ],
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        },
                    }
                )
            )
        return "\n".join(lines) + "\n"

    def cancel(self, remote_job_name: str) -> None:
        self.cancelled.append(remote_job_name)

    def cleanup(self, prefix: str) -> None:
        self.cleaned.append(prefix)


@pytest.fixture
def embedding_server() -> Generator[tuple[str, list[dict[str, object]]], None, None]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body_size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(body_size))
            requests.append(payload)
            texts = cast(list[str], payload["input"])
            dimension = cast(int, payload["dimensions"])
            data = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [float(index + 1)] + [0.0] * (dimension - 1),
                }
                for index in reversed(range(len(texts)))
            ]
            response = json.dumps(
                {
                    "object": "list",
                    "model": payload["model"],
                    "data": data,
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = server.server_address
        host, port = address[0], address[1]
        yield f"http://{host}:{port}/embeddings", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _markdown_document(user_file_id: UUID, marker: str) -> Document:
    repeated = " Transit rejimi kapsamında gümrük işlemleri güvenle yürütülür." * 2
    return Document(
        id=str(user_file_id),
        source=DocumentSource.USER_FILE,
        semantic_identifier=f"{marker}.md",
        title=f"{marker}.md",
        sections=[
            TextSection(
                text=(
                    f"# {marker}\n\n## MADDE 1\nBirinci hüküm.{repeated}\n\n"
                    f"## MADDE 2\nİkinci hüküm.{repeated}"
                ),
                link=None,
            )
        ],
        metadata={},
    )


def _snapshot(search_settings: SearchSettings) -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot(
        search_settings_id=search_settings.id,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name="openai/text-embedding-3-large",
        model_dimension=3072,
        reduced_dimension=_DIMENSION,
        effective_dimension=_DIMENSION,
        index_name=search_settings.index_name,
        vertex=VertexBatchConfig(
            model_configuration_id=999_999,
            model_name="gemini-3.1-flash-lite",
            project="disposable-test-project",
            location="europe-west4",
            authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
            gcs_uri="gs://disposable-regulatory-indexing",
        ),
        prompt_version="contextual-rag-v1",
        prompt_hash="a" * 64,
        poll_seconds=1,
        retry_base_seconds=1,
        retry_max_seconds=1,
        embedding_request_size=32,
    )


def _create_job(
    db_session: Session,
    user_file: UserFile,
    document: Document,
    search_settings: SearchSettings,
) -> RegulatoryIndexingJob:
    snapshot = _snapshot(search_settings)
    return job_repository.create_or_get_regulatory_indexing_job(
        db_session,
        user_file_id=user_file.id,
        content_hash=preparation._regulatory_documents_content_hash([document]),
        search_settings_id=search_settings.id,
        prompt_hash=snapshot.prompt_hash,
        config_snapshot=cast(
            job_repository.RegulatoryIndexingConfigSnapshot,
            snapshot.model_dump(mode="json"),
        ),
        now=datetime.datetime.now(datetime.timezone.utc),
    )


def _run_delivery(
    job_id: UUID,
    expected_generation: int,
) -> tuple[int, OrchestrationOutcome]:
    result = run_regulatory_indexing_step(job_id, expected_generation, _TENANT_ID)
    duplicate = run_regulatory_indexing_step(job_id, expected_generation, _TENANT_ID)
    assert duplicate.outcome is OrchestrationOutcome.SKIPPED
    return result.expected_generation or expected_generation, result.outcome


def _stage(db_session: Session, job_id: UUID) -> RegulatoryIndexingStage:
    db_session.expire_all()
    job = db_session.get(RegulatoryIndexingJob, job_id)
    assert job is not None
    return RegulatoryIndexingStage(job.stage)


def test_durable_pipeline_uses_real_postgres_elasticsearch_and_http_embedding(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_url, embedding_requests = embedding_server
    provider = CloudEmbeddingProvider(
        provider_type=EmbeddingProvider.OPENROUTER,
        api_url=embedding_url,
        api_key="disposable-test-key",
    )
    index_name = f"task8_regulatory_{uuid4().hex}"
    search_settings = SearchSettings(
        model_name="openai/text-embedding-3-large",
        model_dim=3072,
        reduced_dimension=_DIMENSION,
        normalize=False,
        query_prefix="",
        passage_prefix="",
        status=IndexModelStatus.PRESENT,
        index_name=index_name,
        provider_type=EmbeddingProvider.OPENROUTER,
        embedding_precision=EmbeddingPrecision.FLOAT,
        enable_contextual_rag=True,
    )
    db_session.add_all([provider, search_settings])
    db_session.commit()
    db_session.refresh(search_settings)

    document_index = ElasticsearchDocumentIndex(
        tenant_state=TenantState(tenant_id=_TENANT_ID, multitenant=False),
        index_name=index_name,
        embedding_dim=_DIMENSION,
        embedding_precision=EmbeddingPrecision.FLOAT,
    )
    document_index._client.create_index(
        mappings=DocumentSchema.get_document_schema(_DIMENSION, False),
        settings=DocumentSchema.get_index_settings_based_on_environment(),
    )
    gateway = _DeterministicVertexGateway()
    documents: dict[UUID, Document] = {}
    created_users: list[User] = []
    created_files: list[UserFile] = []

    def create_file(marker: str) -> tuple[UserFile, Document, RegulatoryIndexingJob]:
        user = create_test_user(db_session, f"task8-{marker}-{uuid4().hex[:8]}")
        user_file = UserFile(
            id=uuid4(),
            user_id=user.id,
            file_id=f"task8-{uuid4().hex}",
            name=f"{marker}.md",
            file_type="text/markdown",
            status=UserFileStatus.INDEXING,
        )
        db_session.add(user_file)
        db_session.commit()
        document = _markdown_document(user_file.id, marker)
        job = _create_job(db_session, user_file, document, search_settings)
        documents[user_file.id] = document
        created_users.append(user)
        created_files.append(user_file)
        return user_file, document, job

    monkeypatch.setattr(
        orchestrator,
        "validate_snapshot_for_stage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_vertex_gateway",
        lambda *_args, **_kwargs: gateway,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_claimed_markdown_documents",
        lambda runtime: [documents[runtime.user_file.id]],
    )
    monkeypatch.setattr(
        preparation, "get_tokenizer", lambda *_args, **_kwargs: _CharacterTokenizer()
    )
    monkeypatch.setattr(
        preparation,
        "get_contextual_token_budget_tokenizer",
        lambda *_args, **_kwargs: _CharacterTokenizer(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_contextual_tokenizers",
        lambda _snapshot_value: (_CharacterTokenizer(), _CharacterTokenizer()),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_tokenizer",
        lambda *_args, **_kwargs: _CharacterTokenizer(),
    )
    monkeypatch.setattr(
        publisher,
        "build_elasticsearch_document_index",
        lambda _settings: document_index,
    )
    monkeypatch.setattr(
        orchestrator,
        "build_elasticsearch_document_index",
        lambda _settings: document_index,
    )
    monkeypatch.setattr(search_nlp_models, "OPENROUTER_EMBEDDINGS_URL", embedding_url)
    try:
        user_file, _document, job = create_file("main")
        generation = 0
        generation, _ = _run_delivery(job.id, generation)
        assert _stage(db_session, job.id) is RegulatoryIndexingStage.CONTEXT_SUBMIT
        generation, _ = _run_delivery(job.id, generation)
        generation, _ = _run_delivery(job.id, generation)
        generation, _ = _run_delivery(job.id, generation)
        assert _stage(db_session, job.id) is RegulatoryIndexingStage.CONTEXT_SUBMIT

        db_session.expire_all()
        persisted_job = db_session.get(RegulatoryIndexingJob, job.id)
        assert persisted_job is not None
        persisted_job.heartbeat_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(minutes=5)
        db_session.commit()
        claims = job_repository.claim_stale_regulatory_indexing_jobs(
            db_session,
            stale_before=datetime.datetime.now(datetime.timezone.utc),
            claimed_at=datetime.datetime.now(datetime.timezone.utc),
        )
        recovery = next(claim for claim in claims if claim.job_id == job.id)
        recovered = run_preclaimed_regulatory_indexing_step(
            job.id,
            recovery.lease_generation,
            recovery.recovery_token,
            _TENANT_ID,
        )
        assert recovered.outcome is OrchestrationOutcome.NEXT_STEP
        assert (
            run_preclaimed_regulatory_indexing_step(
                job.id,
                recovery.lease_generation,
                recovery.recovery_token,
                _TENANT_ID,
            ).outcome
            is OrchestrationOutcome.SKIPPED
        )
        generation = recovery.lease_generation

        while _stage(db_session, job.id) is not RegulatoryIndexingStage.VERIFY:
            generation, outcome = _run_delivery(job.id, generation)
            if outcome is not OrchestrationOutcome.NEXT_STEP:
                db_session.expire_all()
                failed_job = db_session.get(RegulatoryIndexingJob, job.id)
                pytest.fail(
                    "pipeline terminated before VERIFY: "
                    f"status={failed_job.status if failed_job else None} "
                    f"stage={failed_job.stage if failed_job else None} "
                    f"code={failed_job.error_code if failed_job else None} "
                    f"message={failed_job.error_message if failed_job else None}"
                )

        db_session.expire_all()
        rows = list(
            db_session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == user_file.id)
                .order_by(RegulatoryChunk.position)
            ).all()
        )
        items = list(
            db_session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job.id
                )
            ).all()
        )
        assert len(rows) >= 2
        assert all(
            item.status == RegulatoryIndexingItemStatus.EMBEDDED.value for item in items
        )
        assert all(len(cast(list[float], item.vector)) == _DIMENSION for item in items)
        assert sum(item.context is not None for item in items) >= 2

        chunk_ids = [
            get_elasticsearch_doc_chunk_id(
                TenantState(tenant_id=_TENANT_ID, multitenant=False),
                str(user_file.id),
                row.position,
            )
            for row in rows
        ]
        document_index._client.refresh_index()
        hidden_chunks = document_index._client.get_document_chunks(chunk_ids)
        assert all(chunk.hidden for chunk in hidden_chunks.values())

        generation, _ = _run_delivery(job.id, generation)
        assert _stage(db_session, job.id) is RegulatoryIndexingStage.PUBLISH
        _generation, outcome = _run_delivery(job.id, generation)
        assert outcome is OrchestrationOutcome.COMPLETE
        document_index._client.refresh_index()
        visible_chunks = document_index._client.get_document_chunks(chunk_ids)
        assert all(not chunk.hidden for chunk in visible_chunks.values())

        db_session.expire_all()
        completed_job = db_session.get(RegulatoryIndexingJob, job.id)
        completed_file = db_session.get(UserFile, user_file.id)
        assert completed_job is not None
        assert completed_job.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
        assert completed_file is not None
        assert completed_file.status is UserFileStatus.COMPLETED
        assert completed_file.chunk_count == len(rows)

        first_job_submissions = gateway.submissions[:2]
        assert len(first_job_submissions[0]) == len(rows)
        assert len(first_job_submissions[1]) == 1
        assert {
            request.request_hash for request in first_job_submissions[1]
        }.isdisjoint(
            {request.request_hash for request in first_job_submissions[0][:-1]}
        )
        assert embedding_requests
        assert all(
            request["model"] == "openai/text-embedding-3-large"
            for request in embedding_requests
        )
        assert all(
            request["dimensions"] == _DIMENSION for request in embedding_requests
        )
        embedded_texts = [
            text
            for request in embedding_requests
            for text in cast(list[str], request["input"])
        ]
        contextual_texts = [
            text for text in embedded_texts if text.startswith("Deterministic context ")
        ]
        assert len(contextual_texts) >= 2
        assert all("MADDE" in text for text in embedded_texts)

        cancelled_file, _cancelled_document, cancelled_job = create_file("cancel")
        cancelled_generation = 0
        while (
            _stage(db_session, cancelled_job.id) is not RegulatoryIndexingStage.VERIFY
        ):
            cancelled_generation, _ = _run_delivery(
                cancelled_job.id, cancelled_generation
            )
        db_session.expire_all()
        persisted_cancelled_file = db_session.get(UserFile, cancelled_file.id)
        assert persisted_cancelled_file is not None
        persisted_cancelled_file.status = UserFileStatus.CANCELED
        db_session.commit()

        for _ in range(5):
            cancelled_generation, outcome = _run_delivery(
                cancelled_job.id, cancelled_generation
            )
            if outcome is OrchestrationOutcome.COMPLETE:
                break
        assert outcome is OrchestrationOutcome.COMPLETE
        db_session.expire_all()
        cancelled = db_session.get(RegulatoryIndexingJob, cancelled_job.id)
        assert cancelled is not None
        assert cancelled.status == RegulatoryIndexingJobStatus.CANCELLED.value
        cancelled_items = list(
            db_session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == cancelled_job.id
                )
            ).all()
        )
        assert all(item.vector is None for item in cancelled_items)
        assert gateway.cancelled
        assert gateway.cleaned
        document_index._client.refresh_index()
        with pytest.raises(Exception, match="missing"):
            document_index._client.get_document_chunks(
                [
                    get_elasticsearch_doc_chunk_id(
                        TenantState(tenant_id=_TENANT_ID, multitenant=False),
                        str(cancelled_file.id),
                        row.position,
                    )
                    for row in db_session.scalars(
                        select(RegulatoryChunk).where(
                            RegulatoryChunk.user_file_id == cancelled_file.id
                        )
                    ).all()
                ]
            )
    finally:
        try:
            document_index._client.delete_index()
        finally:
            db_session.rollback()
            for user_file in created_files:
                persisted = db_session.get(UserFile, user_file.id)
                if persisted is not None:
                    db_session.delete(persisted)
            db_session.commit()
            for user in created_users:
                persisted = db_session.get(User, user.id)
                if persisted is not None:
                    db_session.delete(persisted)
            db_session.delete(search_settings)
            db_session.delete(provider)
            db_session.commit()
