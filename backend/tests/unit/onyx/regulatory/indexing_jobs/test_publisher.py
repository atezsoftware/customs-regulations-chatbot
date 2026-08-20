from __future__ import annotations

import datetime
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import RegulatoryIndexingItemStatus, UserFileStatus
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
    DocumentChunkVerificationResult,
    DocumentIndex,
    DocumentInsertionRecord,
    IndexingMetadata,
    MetadataUpdateRequest,
    TenantState,
)
from onyx.indexing.models import DocMetadataAwareIndexChunk, IndexChunk
from onyx.regulatory.indexing_jobs import publisher
from onyx.regulatory.indexing_jobs.models import (
    IndexingPublicationIndeterminateError,
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

_DB_SESSION = cast(Session, SimpleNamespace())


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
            gcs_uri="gs://regulatory-indexing/jobs",
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
    snapshot = _snapshot()
    user_file = cast(
        UserFile,
        SimpleNamespace(
            id=uuid4(),
            name="Gümrük Yönetmeliği.md",
            status=UserFileStatus.INDEXING,
            chunk_count=5,
        ),
    )
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=uuid4(),
            user_file_id=user_file.id,
            lease_generation=11,
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )
    settings = cast(
        SearchSettings,
        SimpleNamespace(
            id=snapshot.search_settings_id,
            index_name=snapshot.index_name,
            model_name=snapshot.embedding_model_name,
            provider_type=snapshot.embedding_provider,
            model_dim=snapshot.model_dimension,
            reduced_dimension=snapshot.reduced_dimension,
            final_embedding_dim=snapshot.effective_dimension,
        ),
    )
    rows: list[RegulatoryChunk] = []
    items: list[RegulatoryIndexingItem] = []
    for position in range(2):
        row = cast(
            RegulatoryChunk,
            SimpleNamespace(
                id=f"row-{position}",
                user_file_id=user_file.id,
                position=position,
                text=f"MADDE {position + 1} - Yürürlük hükmü.",
                heading_path=[f"MADDE {position + 1}"],
                chunk_metadata={"article_no": str(position + 1)},
                chunk_type="article",
                validity_start_date=datetime.date(2024, 1, position + 1),
                validity_end_date=(
                    datetime.date(2025, 1, 1) if position == 0 else None
                ),
            ),
        )
        rows.append(row)
        items.append(
            cast(
                RegulatoryIndexingItem,
                SimpleNamespace(
                    id=uuid4(),
                    job_id=job.id,
                    regulatory_chunk_id=row.id,
                    status=RegulatoryIndexingItemStatus.EMBEDDED.value,
                    context=(
                        {"contextual_text": "Generated context.\n"}
                        if position == 0
                        else None
                    ),
                    vector=[float(position + 1), 0.2, 0.3],
                ),
            )
        )
    return job, user_file, settings, rows, items


class _RecordingDocumentIndex:
    def __init__(
        self,
        events: list[str],
        *,
        insertion_records: list[DocumentInsertionRecord] | None = None,
        update_error: Exception | None = None,
        verification_errors: list[Exception | None] | None = None,
    ) -> None:
        self.events = events
        self.index_calls: list[
            tuple[list[DocMetadataAwareIndexChunk], IndexingMetadata]
        ] = []
        self.update_calls: list[list[MetadataUpdateRequest]] = []
        self._insertion_records = insertion_records
        self._update_error = update_error
        self._verification_errors = iter(verification_errors or [])
        self.verification_calls: list[DocumentChunkVerificationRequest] = []

    def index(
        self,
        chunks: Iterable[DocMetadataAwareIndexChunk],
        indexing_metadata: IndexingMetadata,
    ) -> list[DocumentInsertionRecord]:
        self.events.append("index")
        materialized = list(chunks)
        self.index_calls.append((materialized, indexing_metadata))
        if self._insertion_records is not None:
            return self._insertion_records
        return [
            DocumentInsertionRecord(
                document_id=materialized[0].source_document.id,
                already_existed=False,
            )
        ]

    def update(self, update_requests: list[MetadataUpdateRequest]) -> None:
        self.events.append("update")
        self.update_calls.append(update_requests)
        if self._update_error is not None:
            raise self._update_error

    def update_document_visibility(
        self, request: DocumentChunkVerificationRequest
    ) -> None:
        self.events.append("update")
        self.update_calls.append(
            [
                MetadataUpdateRequest(
                    document_ids=[request.document_id],
                    doc_id_to_chunk_cnt={
                        request.document_id: len(request.expected_chunks)
                    },
                    hidden=request.expected_hidden,
                )
            ]
        )
        if self._update_error is not None:
            raise self._update_error

    def verify_document_chunks(
        self, request: DocumentChunkVerificationRequest
    ) -> DocumentChunkVerificationResult:
        self.events.append(f"verify:{str(request.expected_hidden).lower()}")
        self.verification_calls.append(request)
        error = next(self._verification_errors, None)
        if error is not None:
            raise error
        return DocumentChunkVerificationResult(
            document_id=request.document_id,
            chunk_count=len(request.expected_chunks),
            document_chunk_ids=frozenset(
                f"chunk-{chunk.chunk_index}" for chunk in request.expected_chunks
            ),
            hidden=request.expected_hidden,
        )


class _StatefulVisibilityIndex(_RecordingDocumentIndex):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.hidden: bool | None = True

    def update_document_visibility(
        self, request: DocumentChunkVerificationRequest
    ) -> None:
        self.events.append("update")
        self.hidden = request.expected_hidden

    def verify_document_chunks(
        self, request: DocumentChunkVerificationRequest
    ) -> DocumentChunkVerificationResult:
        self.events.append(f"verify:{str(request.expected_hidden).lower()}")
        if self.hidden is not request.expected_hidden:
            raise DocumentChunkVerificationError("visibility mismatch")
        return DocumentChunkVerificationResult(
            document_id=request.document_id,
            chunk_count=len(request.expected_chunks),
            document_chunk_ids=frozenset(
                f"chunk-{chunk.chunk_index}" for chunk in request.expected_chunks
            ),
            hidden=self.hidden,
        )


def test_publish_converges_a_process_death_mixed_visibility_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    events: list[str] = []
    document_index = _StatefulVisibilityIndex(events)
    document_index.hidden = None
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        MagicMock(return_value=True),
    )

    outcome = publish_regulatory_job(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        db_session=_DB_SESSION,
        document_index=cast(DocumentIndex, document_index),
        search_settings=settings,
    )

    assert outcome is publisher.PublishOutcome.COMPLETED
    assert document_index.hidden is False
    assert events == ["update", "verify:true", "update", "verify:false"]


class _SimulatedProcessDeath(BaseException):
    pass


def _install_metadata(
    monkeypatch: pytest.MonkeyPatch,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    search_settings: SearchSettings,
    rows: list[RegulatoryChunk] | None = None,
    items: list[RegulatoryIndexingItem] | None = None,
    commit_error: Exception | None = None,
    commit_calls: list[object] | None = None,
) -> None:
    document_id = str(user_file.id)
    monkeypatch.setattr(
        publisher,
        "get_access_for_user_files",
        lambda _ids, _db: {},
    )
    monkeypatch.setattr(
        publisher,
        "fetch_user_project_ids_for_user_files",
        lambda _ids, _db: {document_id: [17]},
    )
    monkeypatch.setattr(
        publisher,
        "fetch_persona_ids_for_user_files",
        lambda _ids, _db: {document_id: [23]},
    )
    monkeypatch.setattr(
        publisher,
        "fetch_document_set_names_for_user_files",
        lambda _ids, _db: {document_id: ["Regulations"]},
    )

    @contextmanager
    def locked_lease(
        _db_session: Session,
        *,
        job_id: object,
        expected_stage: object,
        expected_generation: int,
    ) -> Iterator[object]:
        assert job_id == job.id

        def commit() -> None:
            if commit_calls is not None:
                commit_calls.append(object())
            if commit_error is not None:
                raise commit_error

        yield SimpleNamespace(
            job_id=job.id,
            user_file_id=user_file.id,
            lease_generation=expected_generation,
            stage=expected_stage,
            config_snapshot=job.config_snapshot,
            search_settings_id=job.config_snapshot["search_settings_id"],
            search_settings=search_settings,
            user_file_name=user_file.name,
            user_file_status=user_file.status,
            user_file_chunk_count=user_file.chunk_count,
            regulatory_chunks=tuple(rows or []),
            indexing_items=tuple(items or []),
            commit=commit,
        )

    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "regulatory_indexing_external_mutation_lease",
        locked_lease,
        raising=False,
    )


def _stage(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    settings: SearchSettings,
    rows: list[RegulatoryChunk],
    items: list[RegulatoryIndexingItem],
    document_index: _RecordingDocumentIndex,
) -> PublishVerification:
    return stage_regulatory_job_in_index(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        search_settings=settings,
        tenant_id="tenant-a",
        db_session=_DB_SESSION,
        document_index=cast(DocumentIndex, document_index),
    )


def test_stage_indexes_every_chunk_hidden_then_publish_completes_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    events: list[str] = []
    document_index = _RecordingDocumentIndex(events)
    completed: list[tuple[object, int, int]] = []

    def complete_file(
        _db_session: Session,
        *,
        job_id: object,
        expected_generation: int,
        chunk_count: int,
        now: datetime.datetime,
        commit: bool = True,
    ) -> bool:
        assert now.tzinfo is datetime.timezone.utc
        assert commit is False
        events.append("complete")
        completed.append((job_id, expected_generation, chunk_count))
        return True

    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        complete_file,
        raising=False,
    )

    verification = _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=list(reversed(rows)),
        items=list(reversed(items)),
        document_index=document_index,
    )

    staged_chunks, metadata = document_index.index_calls[0]
    assert events == ["index", "verify:true"]
    assert all(chunk.hidden for chunk in staged_chunks)
    assert [chunk.chunk_id for chunk in staged_chunks] == [0, 1]
    assert [chunk.regulatory_chunk_id for chunk in staged_chunks] == ["row-0", "row-1"]
    assert staged_chunks[0].validity_end_date == datetime.date(2025, 1, 1)
    assert staged_chunks[0].doc_summary == "Generated context.\n"
    assert metadata.doc_id_to_chunk_cnt_diff[str(user_file.id)].old_chunk_cnt == 5
    assert metadata.doc_id_to_chunk_cnt_diff[str(user_file.id)].new_chunk_cnt == 2
    assert verification == PublishVerification(
        job_id=job.id,
        document_id=str(user_file.id),
        canonical_chunk_count=2,
        embedded_item_count=2,
        vector_dimension=3,
        insertion_record_count=1,
    )

    publish_regulatory_job(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        verification=verification,
        db_session=_DB_SESSION,
        document_index=cast(DocumentIndex, document_index),
    )

    assert events == [
        "index",
        "verify:true",
        "update",
        "verify:true",
        "update",
        "verify:false",
        "complete",
    ]
    assert document_index.update_calls == [
        [
            MetadataUpdateRequest(
                document_ids=[str(user_file.id)],
                doc_id_to_chunk_cnt={str(user_file.id): 2},
                hidden=True,
            )
        ],
        [
            MetadataUpdateRequest(
                document_ids=[str(user_file.id)],
                doc_id_to_chunk_cnt={str(user_file.id): 2},
                hidden=False,
            )
        ],
    ]
    assert completed == [(job.id, 11, 2)]
    assert user_file.status is UserFileStatus.INDEXING


def test_publish_recovers_visible_projection_after_process_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    events: list[str] = []
    document_index = _StatefulVisibilityIndex(events)
    completions = iter([_SimulatedProcessDeath(), True])

    def complete_publication(*_args: object, **_kwargs: object) -> bool:
        events.append("complete-publication")
        outcome = next(completions)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        complete_publication,
        raising=False,
    )

    with pytest.raises(_SimulatedProcessDeath):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
            search_settings=settings,
        )

    assert document_index.hidden is False
    outcome = publish_regulatory_job(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        db_session=_DB_SESSION,
        document_index=cast(DocumentIndex, document_index),
        search_settings=settings,
    )

    assert outcome is publisher.PublishOutcome.COMPLETED
    assert document_index.hidden is False
    assert events == [
        "update",
        "verify:true",
        "update",
        "verify:false",
        "complete-publication",
        "update",
        "verify:true",
        "update",
        "verify:false",
        "complete-publication",
    ]


def test_already_visible_projection_db_failure_requires_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _StatefulVisibilityIndex([])
    document_index.hidden = False
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
            search_settings=settings,
        )

    assert document_index.hidden is True


def test_stage_retry_preserves_document_and_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])

    first = _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=rows,
        items=items,
        document_index=document_index,
    )
    second = _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=list(reversed(rows)),
        items=list(reversed(items)),
        document_index=document_index,
    )

    first_ids = [
        (chunk.source_document.id, chunk.chunk_id, chunk.regulatory_chunk_id)
        for chunk in document_index.index_calls[0][0]
    ]
    second_ids = [
        (chunk.source_document.id, chunk.chunk_id, chunk.regulatory_chunk_id)
        for chunk in document_index.index_calls[1][0]
    ]
    assert first_ids == second_ids
    assert first == second


def test_insertion_count_mismatch_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([], insertion_records=[])
    completed: list[object] = []
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        lambda *_args, **_kwargs: completed.append(object()) or True,
        raising=False,
    )

    with pytest.raises(ValueError, match="exactly one insertion record"):
        _stage(
            job=job,
            user_file=user_file,
            settings=settings,
            rows=rows,
            items=items,
            document_index=document_index,
        )

    assert document_index.update_calls == []
    assert completed == []


def test_visibility_failure_requires_reconciliation_without_db_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex(
        [], update_error=RuntimeError("Elasticsearch unavailable")
    )
    completed: list[object] = []
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        lambda *_args, **_kwargs: completed.append(object()) or True,
        raising=False,
    )
    verification = _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=rows,
        items=items,
        document_index=document_index,
    )

    with pytest.raises(IndexingPublicationIndeterminateError):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            verification=verification,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
        )

    assert completed == []


def test_publish_rejects_stale_count_verification_before_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])
    stale_verification = PublishVerification(
        job_id=job.id,
        document_id=str(user_file.id),
        canonical_chunk_count=1,
        embedded_item_count=1,
        vector_dimension=3,
        insertion_record_count=1,
    )

    with pytest.raises(ValueError, match="verification no longer matches"):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            verification=stale_verification,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
        )

    assert document_index.update_calls == []


def test_elasticsearch_maps_hidden_flag_and_legacy_enrichment_defaults_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    recorder = _RecordingDocumentIndex([])
    _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=rows,
        items=items,
        document_index=recorder,
    )
    staged_chunks, indexing_metadata = recorder.index_calls[0]

    client = MagicMock()
    client.delete_by_query.return_value = 0
    elasticsearch_index = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    elasticsearch_index._index_name = settings.index_name
    elasticsearch_index._client = client
    elasticsearch_index._tenant_state = TenantState(
        tenant_id="tenant-a", multitenant=False
    )

    elasticsearch_index.index(staged_chunks, indexing_metadata)
    hidden_documents = client.bulk_index_documents.call_args.kwargs["documents"]
    assert [document.hidden for document in hidden_documents] == [True, True]

    first = staged_chunks[0]
    base_chunk = IndexChunk.model_construct(
        **{name: getattr(first, name) for name in IndexChunk.model_fields}
    )
    legacy_chunk = DocMetadataAwareIndexChunk.from_index_chunk(
        index_chunk=base_chunk,
        access=first.access,
        document_sets=first.document_sets,
        user_project=first.user_project,
        personas=first.personas,
        boost=first.boost,
        aggregated_chunk_boost_factor=first.aggregated_chunk_boost_factor,
        tenant_id=first.tenant_id,
    )
    assert legacy_chunk.hidden is False


def test_cancelled_user_file_is_never_staged_or_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    user_file.status = UserFileStatus.CANCELED
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])
    verification = PublishVerification(
        job_id=job.id,
        document_id=str(user_file.id),
        canonical_chunk_count=2,
        embedded_item_count=2,
        vector_dimension=3,
        insertion_record_count=1,
    )

    with pytest.raises(ValueError, match="cancelled or deleting"):
        _stage(
            job=job,
            user_file=user_file,
            settings=settings,
            rows=rows,
            items=items,
            document_index=document_index,
        )
    with pytest.raises(ValueError, match="cancelled or deleting"):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            verification=verification,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
        )

    assert document_index.index_calls == []
    assert document_index.update_calls == []


def test_stage_fails_when_actual_hidden_projection_does_not_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex(
        [],
        verification_errors=[DocumentChunkVerificationError("missing chunk")],
    )

    with pytest.raises(DocumentChunkVerificationError, match="missing chunk"):
        _stage(
            job=job,
            user_file=user_file,
            settings=settings,
            rows=rows,
            items=items,
            document_index=document_index,
        )

    assert document_index.events == ["index", "verify:true"]


def test_stage_uses_fresh_locked_projection_instead_of_stale_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, fresh_rows, fresh_items = _fixture()
    stale_rows = deepcopy(fresh_rows)
    stale_items = deepcopy(fresh_items)
    stale_rows[0].text = "STALE LEGAL TEXT"
    stale_items[0].vector = [9.0, 9.0, 9.0]
    _install_metadata(
        monkeypatch,
        job,
        user_file,
        settings,
        fresh_rows,
        fresh_items,
    )
    document_index = _RecordingDocumentIndex([])

    _stage(
        job=job,
        user_file=user_file,
        settings=settings,
        rows=stale_rows,
        items=stale_items,
        document_index=document_index,
    )

    staged_chunks = document_index.index_calls[0][0]
    assert staged_chunks[0].content == "MADDE 1 - Yürürlük hükmü."
    assert staged_chunks[0].embeddings.full_embedding == [1.0, 0.2, 0.3]


def test_verify_stage_reads_actual_hidden_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])

    verification = verify_staged_regulatory_job(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        db_session=_DB_SESSION,
        document_index=cast(DocumentIndex, document_index),
    )

    assert verification.canonical_chunk_count == 2
    assert document_index.events == ["verify:true"]


def test_publish_post_update_disappearance_never_completes_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex(
        [],
        verification_errors=[
            None,
            DocumentChunkVerificationError("chunk disappeared after update"),
        ],
    )
    completed: list[object] = []
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        lambda *_args, **_kwargs: completed.append(object()) or True,
    )

    with pytest.raises(
        DocumentChunkVerificationError, match="chunk disappeared after update"
    ):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            verification=None,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
        )

    assert document_index.events == [
        "update",
        "verify:true",
        "update",
        "verify:false",
        "update",
        "verify:true",
    ]
    assert completed == []


@pytest.mark.parametrize(
    ("completion_result", "error_match"),
    [
        (False, "lease was lost"),
        (RuntimeError("completion write failed"), "completion write failed"),
    ],
)
def test_publish_completion_failure_restores_hidden_projection(
    monkeypatch: pytest.MonkeyPatch,
    completion_result: bool | Exception,
    error_match: str,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])

    def complete_file(*_args: object, **_kwargs: object) -> bool:
        if isinstance(completion_result, Exception):
            raise completion_result
        return completion_result

    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        complete_file,
    )

    with pytest.raises(RuntimeError, match=error_match):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
            search_settings=settings,
        )

    assert document_index.events == [
        "update",
        "verify:true",
        "update",
        "verify:false",
        "update",
        "verify:true",
    ]


def test_publish_commit_failure_preserves_visible_projection_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    commit_calls: list[object] = []
    _install_metadata(
        monkeypatch,
        job,
        user_file,
        settings,
        rows,
        items,
        commit_error=RuntimeError("database commit failed"),
        commit_calls=commit_calls,
    )
    document_index = _StatefulVisibilityIndex([])
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        lambda *_args, **_kwargs: True,
    )

    with pytest.raises(IndexingPublicationIndeterminateError):
        publish_regulatory_job(
            job=job,
            user_file=user_file,
            rows=rows,
            items=items,
            db_session=_DB_SESSION,
            document_index=cast(DocumentIndex, document_index),
            search_settings=settings,
        )

    assert len(commit_calls) == 1
    assert document_index.events == [
        "update",
        "verify:true",
        "update",
        "verify:false",
    ]
    assert document_index.hidden is False


@pytest.mark.parametrize("operation", ["verify", "publish"])
@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("model_name", "openai/text-embedding-3-small"),
        ("final_embedding_dim", 4),
        ("index_name", "mutated-regulatory-index"),
    ],
)
def test_verify_and_publish_reject_locked_search_settings_drift_before_es(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    field: str,
    mutated_value: object,
) -> None:
    job, user_file, settings, rows, items = _fixture()
    setattr(settings, field, mutated_value)
    _install_metadata(monkeypatch, job, user_file, settings, rows, items)
    document_index = _RecordingDocumentIndex([])
    completed: list[object] = []
    monkeypatch.setattr(
        publisher.indexing_job_repository,
        "complete_regulatory_indexing_publication",
        lambda *_args, **_kwargs: completed.append(object()) or True,
    )

    with pytest.raises(ValueError, match="no longer matches"):
        if operation == "verify":
            verify_staged_regulatory_job(
                job=job,
                user_file=user_file,
                rows=rows,
                items=items,
                db_session=_DB_SESSION,
                document_index=cast(DocumentIndex, document_index),
                search_settings=settings,
            )
        else:
            publish_regulatory_job(
                job=job,
                user_file=user_file,
                rows=rows,
                items=items,
                db_session=_DB_SESSION,
                document_index=cast(DocumentIndex, document_index),
                search_settings=settings,
            )

    assert document_index.events == []
    assert completed == []


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


def test_elasticsearch_mget_parser_correlates_out_of_order_chunks() -> None:
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
