from __future__ import annotations

import datetime
import json
import sys
import threading
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.regulatory_indexing import tasks as regulatory_tasks
from onyx.background.celery.tasks.user_file_processing import tasks as user_file_tasks
from onyx.configs import app_configs
from onyx.configs.constants import FileOrigin
from onyx.connectors.file import connector as file_connector
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
    FileContent,
    FileRecord,
    LLMProvider,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    User,
    UserFile,
)
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.schema import (
    DocumentSchema,
    get_elasticsearch_doc_chunk_id,
)
from onyx.document_index.interfaces_new import TenantState
from onyx.file_store import postgres_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    VERTEX_LOCATION_KWARG,
    VERTEX_PROJECT_KWARG,
)
from onyx.natural_language_processing import search_nlp_models
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.indexing_jobs import (
    configuration,
    orchestrator,
    preparation,
    publisher,
)
from onyx.regulatory.indexing_jobs.models import RegulatoryInputHashVersion
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

_TEST_REDUCED_DIMENSION = 768
_TENANT_ID = "public"
_OPENROUTER_PROVIDER_ADVISORY_LOCK = 5_932_460_513_101_126_737
_LEGACY_REGULATORY_JOB_REVISION = "c8f1a6d4e2b7"
_CHUNK_IDENTITY_REVISION = "d2a9c7e4b1f6"


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


def _markdown_bytes(marker: str) -> bytes:
    repeated = " Transit rejimi kapsamında gümrük işlemleri güvenle yürütülür." * 2
    return (
        f"# {marker}\n\n## MADDE 1\nBirinci hüküm.{repeated}\n\n"
        f"## MADDE 2\nİkinci hüküm.{repeated}"
    ).encode()


def _metadata_markdown_bytes(marker: str) -> bytes:
    metadata = json.dumps(
        {
            "title": f"{marker} Kurtarma Başlığı",
            "regulation_number": "2026/7",
        },
        ensure_ascii=False,
    )
    return (
        f"<!-- ONYX_METADATA={metadata} -->\n"
        f"# {marker}\n\n## MADDE 1\nMetadata ile kurtarılan hüküm.\n\n"
        "## MADDE 2\nİkinci metadata hükmü."
    ).encode()


def _migration_config(engine: Engine) -> Config:
    backend_root = Path(__file__).resolve().parents[3]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.attributes["connection"] = engine
    config.attributes["schema_name"] = "public"
    config.attributes["configure_logger"] = False
    return config


@contextmanager
def _pre_migration_database(source_engine: Engine) -> Iterator[Engine]:
    database_name = f"regulatory_recovery_{uuid4().hex}"
    admin_engine = create_engine(
        source_engine.url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
    )
    dedicated_engine: Engine | None = None
    try:
        with admin_engine.connect() as connection:
            connection.exec_driver_sql(
                f'CREATE DATABASE "{database_name}" TEMPLATE template0'
            )
        dedicated_engine = create_engine(source_engine.url.set(database=database_name))
        command.upgrade(
            _migration_config(dedicated_engine),
            _LEGACY_REGULATORY_JOB_REVISION,
        )
        yield dedicated_engine
    finally:
        if dedicated_engine is not None:
            dedicated_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database_name}"')
        admin_engine.dispose()


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


@dataclass
class _DisposablePipelineResources:
    db_session: Session
    prefix: str
    index_name: str
    file_store: PostgresBackedFileStore = field(default_factory=PostgresBackedFileStore)
    effective_dimension: int = 0
    search_settings: SearchSettings | None = None
    document_index: ElasticsearchDocumentIndex | None = None
    database_commit_completed: bool = False
    index_mutation_started: bool = False
    embedding_provider_created: bool = False
    embedding_provider_original_state: tuple[object, ...] | None = None
    embedding_provider_test_state: tuple[object, ...] | None = None
    embedding_provider: CloudEmbeddingProvider | None = None
    embedding_provider_lock: Connection | None = None
    llm_provider_ids: list[int] = field(default_factory=list)
    model_configuration_ids: list[int] = field(default_factory=list)
    search_settings_ids: list[int] = field(default_factory=list)
    user_ids: list[UUID] = field(default_factory=list)
    user_file_ids: list[UUID] = field(default_factory=list)
    job_ids: list[UUID] = field(default_factory=list)
    store_file_ids: list[str] = field(default_factory=list)
    large_object_oids: list[int] = field(default_factory=list)


def _delete_disposable_indices(prefix: str) -> None:
    discovery_client = ElasticsearchIndexClient(f"{prefix}probe")
    try:
        response = discovery_client._client.indices.get(
            index=f"{prefix}*",
            allow_no_indices=True,
        )
    finally:
        discovery_client.close()
    for index_name in response.keys():
        if not index_name.startswith(prefix):
            raise AssertionError("Elasticsearch cleanup escaped its disposable prefix")
        client = ElasticsearchIndexClient(index_name)
        try:
            client.delete_index()
        finally:
            client.close()


def _disposable_index_names(prefix: str) -> set[str]:
    client = ElasticsearchIndexClient(f"{prefix}probe")
    try:
        response = client._client.indices.get(
            index=f"{prefix}*",
            allow_no_indices=True,
        )
        return set(response.keys())
    finally:
        client.close()


def _cleanup_disposable_pipeline(resources: _DisposablePipelineResources) -> None:
    db_session = resources.db_session
    cleanup_errors: list[tuple[str, Exception]] = []

    def attempt(stage: str, operation: Any) -> None:
        try:
            operation()
        except Exception as error:
            cleanup_errors.append((stage, error))

    def database_stage(operation: Any) -> None:
        db_session.rollback()
        try:
            operation()
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    attempt("database rollback", db_session.rollback)
    if resources.document_index is not None:
        attempt("Elasticsearch client close", resources.document_index._client.close)
    if resources.index_mutation_started:
        attempt(
            "Elasticsearch disposable-index deletion",
            lambda: _delete_disposable_indices(resources.prefix),
        )

    def delete_file_store_rows() -> None:
        file_records = list(
            db_session.scalars(
                select(FileRecord).where(
                    FileRecord.file_id.like(f"{resources.prefix}%")
                )
            ).all()
        )
        for record in file_records:
            resources.file_store.delete_file(record.file_id, error_on_missing=False)
        db_session.rollback()

    attempt("file-store rows and large objects", delete_file_store_rows)

    def delete_user_files() -> None:
        user_files = list(
            db_session.scalars(
                select(UserFile).where(UserFile.file_id.like(f"{resources.prefix}%"))
            ).all()
        )
        for user_file in user_files:
            db_session.delete(user_file)

    attempt(
        "UserFile/job/chunk state",
        lambda: database_stage(delete_user_files),
    )

    def delete_disposable_configuration() -> None:
        users = list(
            db_session.execute(
                select(User).where(User.__table__.c.email.like(f"{resources.prefix}%"))
            )
            .unique()
            .scalars()
            .all()
        )
        for user in users:
            db_session.delete(user)
        db_session.execute(
            delete(SearchSettings).where(
                SearchSettings.index_name.like(f"{resources.prefix}%")
            )
        )
        if resources.model_configuration_ids:
            db_session.execute(
                delete(ModelConfiguration).where(
                    ModelConfiguration.id.in_(resources.model_configuration_ids)
                )
            )
        db_session.execute(
            delete(LLMProvider).where(LLMProvider.name.like(f"{resources.prefix}%"))
        )

    attempt(
        "disposable users and configuration",
        lambda: database_stage(delete_disposable_configuration),
    )

    def restore_embedding_provider() -> None:
        if resources.embedding_provider_original_state is not None:
            _restore_raw_embedding_provider_state(
                db_session,
                EmbeddingProvider.OPENROUTER,
                resources.embedding_provider_original_state,
            )
            return
        if not resources.embedding_provider_created:
            return
        current_state = _raw_openrouter_provider_state(db_session)
        if current_state != resources.embedding_provider_test_state:
            raise RuntimeError(
                "disposable OpenRouter provider changed while advisory lock was held"
            )
        if resources.embedding_provider is None:
            raise RuntimeError(
                "disposable OpenRouter provider identity was not recorded"
            )
        db_session.delete(resources.embedding_provider)

    attempt(
        "OpenRouter provider restoration",
        lambda: database_stage(restore_embedding_provider),
    )

    def unlock_embedding_provider() -> None:
        lock_connection = resources.embedding_provider_lock
        if lock_connection is None:
            return
        try:
            unlocked = lock_connection.scalar(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": _OPENROUTER_PROVIDER_ADVISORY_LOCK},
            )
            if unlocked is not True:
                raise RuntimeError("OpenRouter provider advisory lock was not held")
        finally:
            lock_connection.close()
            resources.embedding_provider_lock = None

    attempt("OpenRouter provider advisory unlock", unlock_embedding_provider)
    if cleanup_errors:
        details = "; ".join(
            f"{stage}: {type(error).__name__}: {error}"
            for stage, error in cleanup_errors
        )
        raise RuntimeError(f"disposable cleanup failed: {details}")


@contextmanager
def _disposable_pipeline_scope(
    db_session: Session,
    *,
    embedding_url: str,
    prefix: str | None = None,
    after_configuration_commit: Callable[[], None] | None = None,
) -> Iterator[_DisposablePipelineResources]:
    resolved_prefix = prefix or f"task8_pipeline_{uuid4().hex}_"
    resources = _DisposablePipelineResources(
        db_session=db_session,
        prefix=resolved_prefix,
        index_name=f"{resolved_prefix}index",
    )
    try:
        engine = cast(Engine, db_session.get_bind())
        provider_lock = engine.connect()
        resources.embedding_provider_lock = provider_lock
        provider_lock.execute(
            text("SELECT pg_advisory_lock(:lock_key)"),
            {"lock_key": _OPENROUTER_PROVIDER_ADVISORY_LOCK},
        )
        resources.embedding_provider_original_state = _raw_openrouter_provider_state(
            db_session
        )

        vertex_provider = LLMProvider(
            name=f"{resolved_prefix}vertex",
            provider=LlmProviderNames.VERTEX_AI,
            custom_config={
                VERTEX_AUTH_METHOD_KWARG: VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
                VERTEX_PROJECT_KWARG: "disposable-test-project",
                VERTEX_LOCATION_KWARG: "europe-west4",
            },
        )
        vertex_model = ModelConfiguration(
            name="gemini-3.1-flash-lite",
            is_visible=True,
            llm_provider=vertex_provider,
        )
        db_session.add_all([vertex_provider, vertex_model])
        db_session.flush()
        resources.llm_provider_ids.append(vertex_provider.id)
        resources.model_configuration_ids.append(vertex_model.id)

        embedding_provider = db_session.get(
            CloudEmbeddingProvider, EmbeddingProvider.OPENROUTER
        )
        if embedding_provider is None:
            embedding_provider = CloudEmbeddingProvider(
                provider_type=EmbeddingProvider.OPENROUTER,
            )
            db_session.add(embedding_provider)
            resources.embedding_provider_created = True
        embedding_provider.api_url = embedding_url
        embedding_provider.api_key = "disposable-test-key"  # ty: ignore[invalid-assignment]
        embedding_provider.api_version = None
        embedding_provider.deployment_name = None
        resources.embedding_provider = embedding_provider
        db_session.flush()
        resources.embedding_provider_test_state = _raw_openrouter_provider_state(
            db_session
        )

        search_settings = SearchSettings(
            model_name="openai/text-embedding-3-large",
            model_dim=3072,
            reduced_dimension=_TEST_REDUCED_DIMENSION,
            normalize=False,
            query_prefix="",
            passage_prefix="",
            status=IndexModelStatus.PRESENT,
            index_name=resources.index_name,
            provider_type=EmbeddingProvider.OPENROUTER,
            embedding_precision=EmbeddingPrecision.FLOAT,
            enable_contextual_rag=True,
            contextual_rag_model_configuration_id=vertex_model.id,
        )
        db_session.add(search_settings)
        db_session.commit()
        if after_configuration_commit is not None:
            after_configuration_commit()
        resources.database_commit_completed = True
        db_session.refresh(search_settings)
        resources.search_settings = search_settings
        resources.search_settings_ids.append(search_settings.id)
        resources.effective_dimension = search_settings.final_embedding_dim
        assert resources.effective_dimension == _TEST_REDUCED_DIMENSION
        assert resources.effective_dimension != 1024

        document_index = ElasticsearchDocumentIndex(
            tenant_state=TenantState(tenant_id=_TENANT_ID, multitenant=False),
            index_name=resources.index_name,
            embedding_dim=resources.effective_dimension,
            embedding_precision=EmbeddingPrecision.FLOAT,
        )
        resources.document_index = document_index
        resources.index_mutation_started = True
        document_index._client.create_index(
            mappings=DocumentSchema.get_document_schema(
                resources.effective_dimension, False
            ),
            settings=DocumentSchema.get_index_settings_based_on_environment(),
        )
        yield resources
    except BaseException as primary_error:
        try:
            _cleanup_disposable_pipeline(resources)
        except Exception as cleanup_error:
            primary_error.add_note(str(cleanup_error))
        raise
    else:
        _cleanup_disposable_pipeline(resources)


def _create_disposable_file(
    resources: _DisposablePipelineResources,
    *,
    marker: str,
    fail_after_prepare: bool = False,
) -> tuple[UserFile, RegulatoryIndexingJob]:
    db_session = resources.db_session
    user = create_test_user(
        db_session,
        f"{resources.prefix}{marker}",
    )
    resources.user_ids.append(user.id)
    store_file_id = f"{resources.prefix}file_{uuid4().hex}"
    resources.store_file_ids.append(store_file_id)
    resources.file_store.save_file(
        content=BytesIO(_markdown_bytes(marker)),
        display_name=f"{resources.prefix}{marker}.md",
        file_origin=FileOrigin.USER_FILE,
        file_type="text/markdown",
        file_id=store_file_id,
    )
    file_content = db_session.get(FileContent, store_file_id)
    assert file_content is not None
    resources.large_object_oids.append(file_content.lobj_oid)
    user_file_id = uuid4()
    resources.user_file_ids.append(user_file_id)
    user_file = UserFile(
        id=user_file_id,
        user_id=user.id,
        file_id=store_file_id,
        name=f"{resources.prefix}{marker}.md",
        file_type="text/markdown",
        status=UserFileStatus.INDEXING,
    )
    db_session.add(user_file)
    db_session.commit()
    documents, staged_csv_ids = orchestrator._load_claimed_markdown_documents(
        cast(
            job_repository.RegulatoryIndexingRuntime,
            SimpleNamespace(user_file=user_file),
        ),
        _TENANT_ID,
    )
    assert staged_csv_ids == []
    job_id = preparation.prepare_regulatory_indexing_job(
        user_file.id,
        documents,
        _TENANT_ID,
        db_session,
    )
    resources.job_ids.append(job_id)
    job = db_session.get(RegulatoryIndexingJob, job_id)
    assert job is not None
    if fail_after_prepare:
        raise RuntimeError("forced setup failure")
    return user_file, job


def _assert_no_disposable_pipeline_state(
    db_session: Session,
    prefix: str,
    resources: _DisposablePipelineResources,
) -> None:
    db_session.expire_all()
    assert not db_session.scalars(
        select(SearchSettings.id).where(SearchSettings.index_name.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(LLMProvider.id).where(LLMProvider.name.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(FileRecord.file_id).where(FileRecord.file_id.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(FileContent.file_id).where(FileContent.file_id.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(UserFile.id).where(UserFile.file_id.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(User.__table__.c.id).where(User.__table__.c.email.like(f"{prefix}%"))
    ).all()
    if resources.job_ids:
        assert not db_session.scalars(
            select(RegulatoryIndexingJob.id).where(
                RegulatoryIndexingJob.id.in_(resources.job_ids)
            )
        ).all()
    if resources.model_configuration_ids:
        assert not db_session.scalars(
            select(ModelConfiguration.id).where(
                ModelConfiguration.id.in_(resources.model_configuration_ids)
            )
        ).all()
    if resources.large_object_oids:
        remaining_large_objects = db_session.scalar(
            text("SELECT count(*) FROM pg_largeobject_metadata WHERE oid = ANY(:oids)"),
            {"oids": resources.large_object_oids},
        )
        assert remaining_large_objects == 0
    assert not _disposable_index_names(prefix)


def test_forced_setup_failure_leaves_no_disposable_pipeline_state(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_url, _requests = embedding_server
    prefix = f"task8_cleanup_{uuid4().hex}_"

    with pytest.raises(RuntimeError, match="forced setup failure"):
        with _disposable_pipeline_scope(
            db_session,
            embedding_url=embedding_url,
            prefix=prefix,
        ) as resources:
            monkeypatch.setattr(
                app_configs,
                "REGULATORY_INDEXING_GCS_URI",
                "gs://disposable-regulatory-indexing",
            )
            monkeypatch.setattr(
                file_connector,
                "get_default_file_store",
                lambda: resources.file_store,
            )
            monkeypatch.setattr(
                preparation,
                "get_tokenizer",
                lambda *_args, **_kwargs: _CharacterTokenizer(),
            )
            monkeypatch.setattr(
                preparation,
                "get_contextual_token_budget_tokenizer",
                lambda *_args, **_kwargs: _CharacterTokenizer(),
            )
            _create_disposable_file(
                resources,
                marker="forced",
                fail_after_prepare=True,
            )

    _assert_no_disposable_pipeline_state(db_session, prefix, resources)


def test_failure_immediately_after_provider_commit_leaves_no_global_provider(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
) -> None:
    embedding_url, _requests = embedding_server
    prefix = f"task8_postcommit_failure_{uuid4().hex}_"
    assert _raw_openrouter_provider_state(db_session) is None

    def fail_immediately_after_commit() -> None:
        raise RuntimeError("forced failure immediately after provider commit")

    with pytest.raises(
        RuntimeError,
        match="forced failure immediately after provider commit",
    ):
        with _disposable_pipeline_scope(
            db_session,
            embedding_url=embedding_url,
            prefix=prefix,
            after_configuration_commit=fail_immediately_after_commit,
        ):
            pytest.fail("scope yielded after the forced post-commit failure")

    assert _raw_openrouter_provider_state(db_session) is None
    assert not db_session.scalars(
        select(SearchSettings.id).where(SearchSettings.index_name.like(f"{prefix}%"))
    ).all()
    assert not db_session.scalars(
        select(LLMProvider.id).where(LLMProvider.name.like(f"{prefix}%"))
    ).all()


def test_postcommit_cleanup_preserves_unrelated_embedding_provider(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
) -> None:
    embedding_url, _requests = embedding_server
    state_before_test = _raw_embedding_provider_state(
        db_session,
        EmbeddingProvider.OPENAI,
    )
    test_state: tuple[object, ...] | None = None

    try:
        provider = db_session.get(
            CloudEmbeddingProvider,
            EmbeddingProvider.OPENAI,
        )
        if provider is None:
            provider = CloudEmbeddingProvider(provider_type=EmbeddingProvider.OPENAI)
            db_session.add(provider)
        provider.api_url = "https://unrelated.example/embeddings"
        provider.api_key = "unrelated-secret"  # ty: ignore[invalid-assignment]
        provider.api_version = "unrelated-version"
        provider.deployment_name = "unrelated-deployment"
        db_session.flush()
        test_state = _raw_embedding_provider_state(
            db_session,
            EmbeddingProvider.OPENAI,
        )
        assert test_state is not None
        db_session.commit()

        def fail_immediately_after_commit() -> None:
            raise RuntimeError("forced failure with unrelated provider")

        with pytest.raises(
            RuntimeError,
            match="forced failure with unrelated provider",
        ):
            with _disposable_pipeline_scope(
                db_session,
                embedding_url=embedding_url,
                prefix=f"task8_unrelated_provider_{uuid4().hex}_",
                after_configuration_commit=fail_immediately_after_commit,
            ):
                pytest.fail("scope yielded after the forced post-commit failure")

        assert (
            _raw_embedding_provider_state(db_session, EmbeddingProvider.OPENAI)
            == test_state
        )
    finally:
        db_session.rollback()
        current_state = _raw_embedding_provider_state(
            db_session,
            EmbeddingProvider.OPENAI,
        )
        if current_state == test_state:
            _restore_raw_embedding_provider_state(
                db_session,
                EmbeddingProvider.OPENAI,
                state_before_test,
            )
            db_session.commit()
        elif current_state != state_before_test:
            raise RuntimeError(
                "unrelated embedding provider changed outside the test boundary"
            )


def test_cleanup_failure_preserves_primary_exception_and_runs_later_stages(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_url, _requests = embedding_server
    prefix = f"task8_cleanup_error_{uuid4().hex}_"
    original_delete_indices = _delete_disposable_indices
    caught: RuntimeError | None = None
    resources: _DisposablePipelineResources | None = None

    def delete_then_fail(cleanup_prefix: str) -> None:
        original_delete_indices(cleanup_prefix)
        raise RuntimeError("forced cleanup stage failure")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_delete_disposable_indices",
        delete_then_fail,
    )
    try:
        with _disposable_pipeline_scope(
            db_session,
            embedding_url=embedding_url,
            prefix=prefix,
        ) as resources:
            monkeypatch.setattr(
                app_configs,
                "REGULATORY_INDEXING_GCS_URI",
                "gs://disposable-regulatory-indexing",
            )
            monkeypatch.setattr(
                file_connector,
                "get_default_file_store",
                lambda: resources.file_store,
            )
            monkeypatch.setattr(
                preparation,
                "get_tokenizer",
                lambda *_args, **_kwargs: _CharacterTokenizer(),
            )
            monkeypatch.setattr(
                preparation,
                "get_contextual_token_budget_tokenizer",
                lambda *_args, **_kwargs: _CharacterTokenizer(),
            )
            _create_disposable_file(resources, marker="cleanup-error")
            raise RuntimeError("forced primary pipeline failure")
    except RuntimeError as error:
        caught = error

    assert caught is not None
    assert str(caught) == "forced primary pipeline failure"
    assert any("forced cleanup stage failure" in note for note in caught.__notes__)
    assert resources is not None
    _assert_no_disposable_pipeline_state(db_session, prefix, resources)


def _raw_embedding_provider_state(
    db_session: Session,
    provider_type: EmbeddingProvider,
) -> tuple[object, ...] | None:
    row = db_session.execute(
        text(
            "SELECT provider_type, api_url, api_key, api_version, deployment_name "
            "FROM embedding_provider WHERE provider_type = :provider_type"
        ),
        {"provider_type": provider_type.name},
    ).one_or_none()
    if row is None:
        return None
    values = list(row)
    if isinstance(values[2], memoryview):
        values[2] = bytes(values[2])
    return tuple(values)


def _raw_openrouter_provider_state(db_session: Session) -> tuple[object, ...] | None:
    return _raw_embedding_provider_state(db_session, EmbeddingProvider.OPENROUTER)


def _restore_raw_embedding_provider_state(
    db_session: Session,
    provider_type: EmbeddingProvider,
    state: tuple[object, ...] | None,
) -> None:
    if state is None:
        provider = db_session.get(
            CloudEmbeddingProvider,
            provider_type,
        )
        if provider is not None:
            db_session.delete(provider)
        return

    raw_provider_type, api_url, api_key, api_version, deployment_name = state
    if raw_provider_type != provider_type.name:
        raise RuntimeError("embedding-provider raw state has the wrong identity")
    db_session.execute(
        text(
            "INSERT INTO embedding_provider "
            "(provider_type, api_url, api_key, api_version, deployment_name) "
            "VALUES (:provider_type, :api_url, :api_key, :api_version, "
            ":deployment_name) "
            "ON CONFLICT (provider_type) DO UPDATE SET "
            "api_url = EXCLUDED.api_url, api_key = EXCLUDED.api_key, "
            "api_version = EXCLUDED.api_version, "
            "deployment_name = EXCLUDED.deployment_name"
        ),
        {
            "provider_type": raw_provider_type,
            "api_url": api_url,
            "api_key": api_key,
            "api_version": api_version,
            "deployment_name": deployment_name,
        },
    )


@contextmanager
def _preexisting_openrouter_provider_scope(
    db_session: Session,
    *,
    after_configuration_commit: Callable[[], None] | None = None,
) -> Iterator[tuple[object, ...]]:
    state_before_test = _raw_openrouter_provider_state(db_session)
    test_state: tuple[object, ...] | None = None

    def cleanup() -> None:
        db_session.rollback()
        current_state = _raw_openrouter_provider_state(db_session)
        if current_state == state_before_test:
            return
        if test_state is None or current_state != test_state:
            raise RuntimeError(
                "preexisting-provider fixture changed outside its cleanup boundary"
            )
        _restore_raw_embedding_provider_state(
            db_session,
            EmbeddingProvider.OPENROUTER,
            state_before_test,
        )
        db_session.commit()

    try:
        provider = db_session.get(
            CloudEmbeddingProvider,
            EmbeddingProvider.OPENROUTER,
        )
        if provider is None:
            provider = CloudEmbeddingProvider(
                provider_type=EmbeddingProvider.OPENROUTER,
            )
            db_session.add(provider)
        provider.api_url = "https://preexisting.example/embeddings"
        provider.api_key = "preexisting-secret"  # ty: ignore[invalid-assignment]
        provider.api_version = "preexisting-version"
        provider.deployment_name = "preexisting-deployment"
        db_session.flush()
        test_state = _raw_openrouter_provider_state(db_session)
        if test_state is None:
            raise RuntimeError("preexisting OpenRouter provider state was not recorded")
        db_session.commit()
        if after_configuration_commit is not None:
            after_configuration_commit()
        yield test_state
    except BaseException as primary_error:
        try:
            cleanup()
        except Exception as cleanup_error:
            primary_error.add_note(str(cleanup_error))
        raise
    else:
        cleanup()


def test_preexisting_provider_fixture_restores_when_setup_fails_after_commit(
    db_session: Session,
) -> None:
    state_before_test = _raw_openrouter_provider_state(db_session)

    def fail_immediately_after_commit() -> None:
        raise RuntimeError("forced preexisting-provider setup failure")

    with pytest.raises(
        RuntimeError,
        match="forced preexisting-provider setup failure",
    ):
        with _preexisting_openrouter_provider_scope(
            db_session,
            after_configuration_commit=fail_immediately_after_commit,
        ):
            pytest.fail("fixture yielded after the forced post-commit failure")

    assert _raw_openrouter_provider_state(db_session) == state_before_test


def test_preexisting_provider_fixture_restores_when_raw_snapshot_fails(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_before_test = _raw_openrouter_provider_state(db_session)
    real_state_reader = _raw_openrouter_provider_state
    calls = 0

    def fail_before_test_state_is_recorded(
        session: Session,
    ) -> tuple[object, ...] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced failure before raw provider snapshot")
        return real_state_reader(session)

    monkeypatch.setattr(
        sys.modules[__name__],
        "_raw_openrouter_provider_state",
        fail_before_test_state_is_recorded,
    )

    with pytest.raises(
        RuntimeError,
        match="forced failure before raw provider snapshot",
    ):
        with _preexisting_openrouter_provider_scope(db_session):
            pytest.fail("fixture yielded without recording its raw test state")

    assert calls == 3
    assert real_state_reader(db_session) == state_before_test


def test_preexisting_openrouter_provider_is_locked_and_restored_exactly(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
) -> None:
    embedding_url, _requests = embedding_server
    state_before_test = _raw_openrouter_provider_state(db_session)
    with _preexisting_openrouter_provider_scope(db_session) as original_state:
        with _disposable_pipeline_scope(
            db_session,
            embedding_url=embedding_url,
        ):
            db_session.expire_all()
            current_provider = db_session.get(
                CloudEmbeddingProvider, EmbeddingProvider.OPENROUTER
            )
            assert current_provider is not None
            assert current_provider.api_url == embedding_url

            engine = cast(Engine, db_session.get_bind())
            with engine.connect() as contender:
                assert (
                    contender.scalar(
                        text("SELECT pg_try_advisory_lock(:lock_key)"),
                        {"lock_key": _OPENROUTER_PROVIDER_ADVISORY_LOCK},
                    )
                    is False
                )

        db_session.expire_all()
        assert _raw_openrouter_provider_state(db_session) == original_state
    assert _raw_openrouter_provider_state(db_session) == state_before_test


@pytest.mark.parametrize(
    "migration_origin,resolved_hash_version",
    [
        ("c8-legacy", RegulatoryInputHashVersion.LEGACY_V1),
        ("d2-canonical", RegulatoryInputHashVersion.CANONICAL_V2),
        ("head-c8-head-canonical", RegulatoryInputHashVersion.CANONICAL_V2),
    ],
)
def test_preparing_compatibility_job_with_onyx_metadata_recovers_and_publishes(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    migration_origin: str,
    resolved_hash_version: RegulatoryInputHashVersion,
) -> None:
    source_engine = db_session.get_bind()
    assert isinstance(source_engine, Engine)
    embedding_url, _embedding_requests = embedding_server
    gateway = _DeterministicVertexGateway()

    with _pre_migration_database(source_engine) as dedicated_engine:
        migration_config = _migration_config(dedicated_engine)
        if migration_origin == "d2-canonical":
            command.upgrade(migration_config, _CHUNK_IDENTITY_REVISION)
        elif migration_origin == "head-c8-head-canonical":
            command.upgrade(migration_config, "head")

        @contextmanager
        def dedicated_session() -> Iterator[Session]:
            with Session(dedicated_engine) as session:
                yield session

        @contextmanager
        def dedicated_session_if_none(
            existing_session: Session | None = None,
        ) -> Iterator[Session]:
            if existing_session is not None:
                yield existing_session
                return
            with Session(dedicated_engine) as session:
                yield session

        monkeypatch.setattr(
            orchestrator,
            "get_session_with_current_tenant",
            dedicated_session,
        )
        monkeypatch.setattr(
            postgres_file_store,
            "get_session_with_current_tenant_if_none",
            dedicated_session_if_none,
        )
        monkeypatch.setattr(
            app_configs,
            "REGULATORY_INDEXING_GCS_URI",
            "gs://disposable-regulatory-indexing",
        )
        monkeypatch.setattr(
            app_configs,
            "REGULATORY_BATCH_INDEXING_ENABLED",
            True,
        )
        monkeypatch.setattr(
            orchestrator,
            "_build_vertex_gateway",
            lambda *_args, **_kwargs: gateway,
        )
        monkeypatch.setattr(
            preparation,
            "get_tokenizer",
            lambda *_args, **_kwargs: _CharacterTokenizer(),
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
            search_nlp_models,
            "OPENROUTER_EMBEDDINGS_URL",
            embedding_url,
        )

        with Session(dedicated_engine) as migration_session:
            with _disposable_pipeline_scope(
                migration_session,
                embedding_url=embedding_url,
                prefix=f"legacy_recovery_{uuid4().hex}_",
            ) as resources:
                document_index = resources.document_index
                assert document_index is not None
                monkeypatch.setattr(
                    file_connector,
                    "get_default_file_store",
                    lambda: resources.file_store,
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

                marker = "Gözetim"
                user = create_test_user(
                    migration_session,
                    f"{resources.prefix}metadata",
                )
                resources.user_ids.append(user.id)
                store_file_id = f"{resources.prefix}file_{uuid4().hex}"
                resources.store_file_ids.append(store_file_id)
                resources.file_store.save_file(
                    content=BytesIO(_metadata_markdown_bytes(marker)),
                    display_name=f"{resources.prefix}metadata.md",
                    file_origin=FileOrigin.USER_FILE,
                    file_type="text/markdown",
                    file_id=store_file_id,
                    db_session=migration_session,
                )
                file_content = migration_session.get(FileContent, store_file_id)
                assert file_content is not None
                resources.large_object_oids.append(file_content.lobj_oid)
                user_file = UserFile(
                    id=uuid4(),
                    user_id=user.id,
                    file_id=store_file_id,
                    name=f"{resources.prefix}metadata.md",
                    file_type="text/markdown",
                    status=UserFileStatus.INDEXING,
                )
                resources.user_file_ids.append(user_file.id)
                migration_session.add(user_file)
                migration_session.commit()

                initial_documents, staged_csv_ids = (
                    orchestrator._load_claimed_markdown_documents(
                        cast(
                            job_repository.RegulatoryIndexingRuntime,
                            SimpleNamespace(user_file=user_file),
                        ),
                        _TENANT_ID,
                    )
                )
                assert staged_csv_ids == []
                assert initial_documents[0].title == f"{marker} Kurtarma Başlığı"
                assert initial_documents[0].metadata["regulation_number"] == "2026/7"
                input_content_hash = preparation.regulatory_documents_content_hash(
                    initial_documents,
                    resolved_hash_version,
                )
                snapshot = preparation.resolve_regulatory_indexing_snapshot(
                    migration_session,
                    input_content_hash=input_content_hash,
                    input_hash_version=resolved_hash_version,
                )
                pre_migration_snapshot = snapshot.model_dump(mode="json")
                if migration_origin == "c8-legacy":
                    del pre_migration_snapshot["input_content_hash"]
                    del pre_migration_snapshot["input_hash_version"]
                    del pre_migration_snapshot["chunk_generation_hash"]
                elif migration_origin == "d2-canonical":
                    del pre_migration_snapshot["input_hash_version"]
                job_id = uuid4()
                resources.job_ids.append(job_id)
                insert_parameters = {
                    "id": job_id,
                    "user_file_id": user_file.id,
                    "content_hash": input_content_hash,
                    "chunk_generation_hash": snapshot.chunk_generation_hash,
                    "search_settings_id": snapshot.search_settings_id,
                    "prompt_hash": snapshot.prompt_hash,
                    "config_snapshot": json.dumps(pre_migration_snapshot),
                }
                if migration_origin == "c8-legacy":
                    migration_session.execute(
                        text(
                            """
                            INSERT INTO regulatory_indexing_job (
                                id, user_file_id, content_hash,
                                search_settings_id, prompt_hash,
                                config_snapshot, status, stage,
                                lease_generation, attempt_count, heartbeat_at,
                                created_at, updated_at
                            ) VALUES (
                                :id, :user_file_id, :content_hash,
                                :search_settings_id, :prompt_hash,
                                CAST(:config_snapshot AS jsonb), 'QUEUED',
                                'PREPARING', 0, 0, now(), now(), now()
                            )
                            """
                        ),
                        insert_parameters,
                    )
                else:
                    migration_session.execute(
                        text(
                            """
                            INSERT INTO regulatory_indexing_job (
                                id, user_file_id, content_hash,
                                chunk_generation_hash, search_settings_id,
                                prompt_hash, config_snapshot, status, stage,
                                lease_generation, attempt_count, heartbeat_at,
                                created_at, updated_at
                            ) VALUES (
                                :id, :user_file_id, :content_hash,
                                :chunk_generation_hash, :search_settings_id,
                                :prompt_hash, CAST(:config_snapshot AS jsonb),
                                'QUEUED', 'PREPARING', 0, 0, now(), now(), now()
                            )
                            """
                        ),
                        insert_parameters,
                    )
                migration_session.commit()

                if migration_origin == "head-c8-head-canonical":
                    command.downgrade(
                        migration_config,
                        _LEGACY_REGULATORY_JOB_REVISION,
                    )
                command.upgrade(migration_config, "head")
                migration_session.expire_all()
                migrated_job = migration_session.get(RegulatoryIndexingJob, job_id)
                assert migrated_job is not None
                assert (
                    migrated_job.config_snapshot["input_content_hash"]
                    == input_content_hash
                )
                assert (
                    migrated_job.config_snapshot["input_hash_version"]
                    == "legacy-or-canonical"
                )

                first = run_regulatory_indexing_step(job_id, 0, _TENANT_ID)
                assert first.outcome is OrchestrationOutcome.NEXT_STEP
                generation = first.expected_generation
                assert generation is not None
                assert (
                    _stage(migration_session, job_id)
                    is RegulatoryIndexingStage.CONTEXT_SUBMIT
                )
                migration_session.expire_all()
                resolved_job = migration_session.get(RegulatoryIndexingJob, job_id)
                assert resolved_job is not None
                assert (
                    resolved_job.config_snapshot["input_hash_version"]
                    == resolved_hash_version.value
                )

                for _ in range(30):
                    if (
                        _stage(migration_session, job_id)
                        is RegulatoryIndexingStage.VERIFY
                    ):
                        break
                    generation, outcome = _run_delivery(job_id, generation)
                    assert outcome is OrchestrationOutcome.NEXT_STEP
                else:
                    pytest.fail("legacy PREPARING recovery did not reach VERIFY")

                migration_session.expire_all()
                rows = list(
                    migration_session.scalars(
                        select(RegulatoryChunk)
                        .where(RegulatoryChunk.user_file_id == user_file.id)
                        .order_by(RegulatoryChunk.position)
                    ).all()
                )
                assert rows
                generation, outcome = _run_delivery(job_id, generation)
                assert outcome is OrchestrationOutcome.NEXT_STEP
                assert (
                    _stage(migration_session, job_id) is RegulatoryIndexingStage.PUBLISH
                )
                _generation, outcome = _run_delivery(job_id, generation)
                assert outcome is OrchestrationOutcome.COMPLETE

                migration_session.expire_all()
                completed_job = migration_session.get(RegulatoryIndexingJob, job_id)
                completed_file = migration_session.get(UserFile, user_file.id)
                assert completed_job is not None
                assert (
                    completed_job.status == RegulatoryIndexingJobStatus.SUCCEEDED.value
                )
                assert completed_file is not None
                assert completed_file.status is UserFileStatus.COMPLETED
                document_index._client.refresh_index()
                chunk_ids = [
                    get_elasticsearch_doc_chunk_id(
                        TenantState(tenant_id=_TENANT_ID, multitenant=False),
                        str(user_file.id),
                        row.position,
                    )
                    for row in rows
                ]
                visible_chunks = document_index._client.get_document_chunks(chunk_ids)
                assert all(not chunk.hidden for chunk in visible_chunks.values())


def test_durable_pipeline_uses_real_postgres_elasticsearch_and_http_embedding(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    embedding_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_url, embedding_requests = embedding_server
    gateway = _DeterministicVertexGateway()
    with _disposable_pipeline_scope(
        db_session,
        embedding_url=embedding_url,
    ) as resources:
        effective_dimension = resources.effective_dimension
        document_index = resources.document_index
        assert document_index is not None

        def create_file(marker: str) -> tuple[UserFile, RegulatoryIndexingJob]:
            return _create_disposable_file(resources, marker=marker)

        monkeypatch.setattr(
            app_configs,
            "REGULATORY_INDEXING_GCS_URI",
            "gs://disposable-regulatory-indexing",
        )
        monkeypatch.setattr(
            app_configs,
            "REGULATORY_BATCH_INDEXING_ENABLED",
            True,
        )
        monkeypatch.setattr(
            orchestrator,
            "_build_vertex_gateway",
            lambda *_args, **_kwargs: gateway,
        )
        monkeypatch.setattr(
            file_connector,
            "get_default_file_store",
            lambda: resources.file_store,
        )
        monkeypatch.setattr(
            preparation,
            "get_tokenizer",
            lambda *_args, **_kwargs: _CharacterTokenizer(),
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
        monkeypatch.setattr(
            search_nlp_models, "OPENROUTER_EMBEDDINGS_URL", embedding_url
        )
        user_file, job = create_file("main")
        generation = job.lease_generation
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
        assert all(
            len(cast(list[float], item.vector)) == effective_dimension for item in items
        )
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
            request["dimensions"] == effective_dimension
            for request in embedding_requests
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

        enqueued_jobs: list[tuple[UUID, int]] = []

        def capture_worker_delivery(
            _celery_app: object,
            *,
            job_id: UUID,
            expected_generation: int,
            **_kwargs: object,
        ) -> None:
            enqueued_jobs.append((job_id, expected_generation))

        monkeypatch.setattr(
            regulatory_tasks,
            "enqueue_regulatory_indexing_step",
            capture_worker_delivery,
        )

        def scan_processing_deliveries(
            target_user_file_id: UUID,
        ) -> list[dict[str, str]]:
            scanner_app = MagicMock()
            scanner_task = user_file_tasks.check_user_file_processing.run.__self__
            with (
                patch.object(
                    type(scanner_task),
                    "app",
                    new_callable=PropertyMock,
                    return_value=scanner_app,
                ),
                patch(
                    "onyx.background.celery.tasks.user_file_processing.tasks.celery_get_broker_client",
                    return_value=MagicMock(),
                ),
                patch(
                    "onyx.background.celery.tasks.user_file_processing.tasks.celery_get_queue_length",
                    return_value=0,
                ),
            ):
                user_file_tasks.check_user_file_processing.run(tenant_id=_TENANT_ID)

            return [
                call.kwargs["kwargs"]
                for call in scanner_app.send_task.call_args_list
                if call.kwargs["kwargs"]["user_file_id"] == str(target_user_file_id)
            ]

        def run_processing_scanner_and_worker(target_user_file_id: UUID) -> None:
            matching_deliveries = scan_processing_deliveries(target_user_file_id)
            assert matching_deliveries == [
                {
                    "user_file_id": str(target_user_file_id),
                    "tenant_id": _TENANT_ID,
                }
            ]
            user_file_tasks.process_single_user_file.run(
                user_file_id=str(target_user_file_id),
                tenant_id=_TENANT_ID,
            )

        first_job_id = job.id
        first_generation_hash = job.chunk_generation_hash
        second_generation_hash = "e" * 64
        assert second_generation_hash != first_generation_hash
        monkeypatch.setattr(
            configuration,
            "compute_regulatory_chunk_generation_hash",
            lambda *_args, **_kwargs: second_generation_hash,
        )
        monkeypatch.setattr(
            orchestrator,
            "compute_regulatory_chunk_generation_hash",
            lambda *_args, **_kwargs: second_generation_hash,
        )
        db_session.expire_all()
        reprocessed_file = db_session.get(UserFile, user_file.id)
        assert reprocessed_file is not None
        reprocessed_file.status = UserFileStatus.PROCESSING
        db_session.commit()

        run_processing_scanner_and_worker(user_file.id)
        db_session.expire_all()
        second_job = db_session.scalar(
            select(RegulatoryIndexingJob).where(
                RegulatoryIndexingJob.user_file_id == user_file.id,
                RegulatoryIndexingJob.id != first_job_id,
            )
        )
        assert second_job is not None
        assert second_job.chunk_generation_hash == second_generation_hash
        assert second_job.status == RegulatoryIndexingJobStatus.QUEUED.value
        assert second_job.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
        assert enqueued_jobs[-1] == (second_job.id, second_job.lease_generation)

        third_generation_hash = "f" * 64
        monkeypatch.setattr(
            configuration,
            "compute_regulatory_chunk_generation_hash",
            lambda *_args, **_kwargs: third_generation_hash,
        )
        monkeypatch.setattr(
            orchestrator,
            "compute_regulatory_chunk_generation_hash",
            lambda *_args, **_kwargs: third_generation_hash,
        )
        regulatory_tasks.regulatory_indexing_run_step.run(
            job_id=str(second_job.id),
            expected_generation=second_job.lease_generation,
            tenant_id=_TENANT_ID,
        )
        db_session.expire_all()
        superseded_job = db_session.get(RegulatoryIndexingJob, second_job.id)
        assert superseded_job is not None
        assert superseded_job.status == RegulatoryIndexingJobStatus.CANCELLING.value
        assert superseded_job.cancellation_intent == "SUPERSEDE"
        superseded_generation = superseded_job.lease_generation

        original_delete = document_index.delete
        remaining_delete_failures = 1

        def fail_first_index_delete(*args: object, **kwargs: object) -> int:
            nonlocal remaining_delete_failures
            if remaining_delete_failures:
                remaining_delete_failures -= 1
                raise RuntimeError("injected Elasticsearch cleanup failure")
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(document_index, "delete", fail_first_index_delete)
        superseded_generation, supersession_outcome = _run_delivery(
            superseded_job.id,
            superseded_generation,
        )
        assert supersession_outcome is OrchestrationOutcome.NEXT_STEP
        superseded_generation, supersession_outcome = _run_delivery(
            superseded_job.id,
            superseded_generation,
        )
        assert supersession_outcome is OrchestrationOutcome.NEXT_STEP
        db_session.expire_all()
        blocked_cleanup = db_session.get(RegulatoryIndexingJob, superseded_job.id)
        assert blocked_cleanup is not None
        assert blocked_cleanup.status == RegulatoryIndexingJobStatus.CANCELLING.value
        assert blocked_cleanup.cancellation_phase == "INDEX_DELETE"
        assert blocked_cleanup.attempt_count == 1
        assert blocked_cleanup.error_code == "unknown"
        assert blocked_cleanup.error_message == "RuntimeError"
        assert blocked_cleanup.next_retry_at is not None
        assert scan_processing_deliveries(user_file.id) == []
        assert (
            db_session.scalar(
                select(func.count(RegulatoryIndexingJob.id)).where(
                    RegulatoryIndexingJob.user_file_id == user_file.id
                )
            )
            == 2
        )
        document_index._client.refresh_index()
        still_visible = document_index._client.get_document_chunks(chunk_ids)
        assert all(not chunk.hidden for chunk in still_visible.values())

        blocked_cleanup.next_retry_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=1)
        db_session.commit()
        superseded_generation, supersession_outcome = _run_delivery(
            superseded_job.id,
            superseded_generation,
        )
        assert supersession_outcome is OrchestrationOutcome.NEXT_STEP
        superseded_generation, supersession_outcome = _run_delivery(
            superseded_job.id,
            superseded_generation,
        )
        assert supersession_outcome is OrchestrationOutcome.COMPLETE
        db_session.expire_all()
        resumed_file = db_session.get(UserFile, user_file.id)
        assert resumed_file is not None
        assert resumed_file.status is UserFileStatus.PROCESSING

        run_processing_scanner_and_worker(user_file.id)
        db_session.expire_all()
        newest_job = db_session.scalar(
            select(RegulatoryIndexingJob)
            .where(
                RegulatoryIndexingJob.user_file_id == user_file.id,
                RegulatoryIndexingJob.status.in_(
                    [
                        RegulatoryIndexingJobStatus.QUEUED.value,
                        RegulatoryIndexingJobStatus.RUNNING.value,
                        RegulatoryIndexingJobStatus.RETRY_WAIT.value,
                        RegulatoryIndexingJobStatus.CANCELLING.value,
                    ]
                ),
            )
            .order_by(RegulatoryIndexingJob.created_at.desc())
        )
        assert newest_job is not None
        assert newest_job.id not in {first_job_id, second_job.id}
        assert newest_job.chunk_generation_hash == third_generation_hash
        newest_generation = newest_job.lease_generation
        while _stage(db_session, newest_job.id) is not RegulatoryIndexingStage.VERIFY:
            newest_generation, newest_outcome = _run_delivery(
                newest_job.id,
                newest_generation,
            )
            if newest_outcome is not OrchestrationOutcome.NEXT_STEP:
                db_session.expire_all()
                failed_newest_job = db_session.get(RegulatoryIndexingJob, newest_job.id)
                pytest.fail(
                    "successor pipeline terminated before VERIFY: "
                    f"status={failed_newest_job.status if failed_newest_job else None} "
                    f"stage={failed_newest_job.stage if failed_newest_job else None} "
                    f"code={failed_newest_job.error_code if failed_newest_job else None} "
                    f"message={failed_newest_job.error_message if failed_newest_job else None}"
                )
        newest_generation, newest_outcome = _run_delivery(
            newest_job.id,
            newest_generation,
        )
        assert newest_outcome is OrchestrationOutcome.NEXT_STEP
        _newest_generation, newest_outcome = _run_delivery(
            newest_job.id,
            newest_generation,
        )
        assert newest_outcome is OrchestrationOutcome.COMPLETE
        db_session.expire_all()
        republished_file = db_session.get(UserFile, user_file.id)
        assert republished_file is not None
        assert republished_file.status is UserFileStatus.COMPLETED

        terminal_job_count = db_session.scalar(
            select(func.count(RegulatoryIndexingJob.id)).where(
                RegulatoryIndexingJob.user_file_id == user_file.id
            )
        )
        repeated_file = db_session.get(UserFile, user_file.id)
        assert repeated_file is not None
        repeated_file.status = UserFileStatus.PROCESSING
        db_session.commit()
        run_processing_scanner_and_worker(user_file.id)
        db_session.expire_all()
        assert (
            db_session.scalar(
                select(func.count(RegulatoryIndexingJob.id)).where(
                    RegulatoryIndexingJob.user_file_id == user_file.id
                )
            )
            == terminal_job_count
        )
        idempotent_file = db_session.get(UserFile, user_file.id)
        assert idempotent_file is not None
        assert idempotent_file.status is UserFileStatus.COMPLETED

        cancelled_file, cancelled_job = create_file("cancel")
        cancelled_generation = cancelled_job.lease_generation
        while (
            _stage(db_session, cancelled_job.id) is not RegulatoryIndexingStage.VERIFY
        ):
            cancelled_generation, _ = _run_delivery(
                cancelled_job.id, cancelled_generation
            )
        db_session.expire_all()
        persisted_cancelled_file = db_session.get(UserFile, cancelled_file.id)
        assert persisted_cancelled_file is not None
        deletion_plan = job_repository.request_user_file_deletion_cleanup(
            db_session,
            user_file_id=persisted_cancelled_file.id,
            now=datetime.datetime.now(datetime.timezone.utc),
        )
        assert deletion_plan.ready_to_delete is False
        assert len(deletion_plan.deliveries) == 1
        cancelled_generation = deletion_plan.deliveries[0].expected_generation
        assert deletion_plan.deliveries[0].job_id == cancelled_job.id

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
        persisted_cancelled_file = db_session.get(UserFile, cancelled_file.id)
        assert persisted_cancelled_file is not None
        assert persisted_cancelled_file.status is UserFileStatus.DELETING
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

    _assert_no_disposable_pipeline_state(db_session, resources.prefix, resources)
