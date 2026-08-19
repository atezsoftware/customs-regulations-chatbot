from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
)
from onyx.natural_language_processing.search_nlp_models import CloudEmbedding
from onyx.regulatory.indexing_jobs import embedding
from onyx.regulatory.indexing_jobs.embedding import (
    EmbeddingSummary,
    embed_pending_regulatory_items,
)
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from shared_configs.enums import EmbeddingProvider, EmbedTextType

_DB_SESSION = cast(Session, SimpleNamespace())


def _snapshot(*, request_size: int = 2) -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot(
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
        embedding_request_size=request_size,
    )


def _job(snapshot: RegulatoryIndexingConfigSnapshot) -> RegulatoryIndexingJob:
    return cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=uuid4(),
            user_file_id=uuid4(),
            lease_generation=9,
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )


def _settings(snapshot: RegulatoryIndexingConfigSnapshot) -> SearchSettings:
    return cast(
        SearchSettings,
        SimpleNamespace(
            id=snapshot.search_settings_id,
            model_name=snapshot.embedding_model_name,
            model_dim=snapshot.model_dimension,
            reduced_dimension=snapshot.reduced_dimension,
            final_embedding_dim=snapshot.effective_dimension,
            index_name=snapshot.index_name,
            provider_type=snapshot.embedding_provider,
            normalize=True,
            query_prefix=None,
            passage_prefix=None,
            api_key="test-openrouter-key",
            api_url=None,
            api_version=None,
            deployment_name=None,
        ),
    )


def _rows_and_items(
    job: RegulatoryIndexingJob,
    *,
    count: int,
) -> tuple[list[RegulatoryChunk], list[RegulatoryIndexingItem]]:
    rows: list[RegulatoryChunk] = []
    items: list[RegulatoryIndexingItem] = []
    for position in range(count):
        row = cast(
            RegulatoryChunk,
            SimpleNamespace(
                id=f"row-{position}",
                user_file_id=job.user_file_id,
                position=position,
                text=f"MADDE {position + 1} - legal text {position}.",
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
                    status=(
                        RegulatoryIndexingItemStatus.SKIPPED.value
                        if position == 1
                        else RegulatoryIndexingItemStatus.CONTEXT_READY.value
                    ),
                    context=(
                        None
                        if position == 1
                        else {"contextual_text": f"Context {position}.\n"}
                    ),
                    vector=None,
                ),
            )
        )
    return rows, items


class _RecordingEmbeddingModel:
    def __init__(self, responses: Sequence[list[list[float]] | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def encode(self, **kwargs: object) -> list[list[float]]:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class _RecordingEmbedder:
    constructed_with: dict[str, object]

    def __init__(
        self,
        model: _RecordingEmbeddingModel,
        constructed_with: dict[str, object],
    ) -> None:
        self.embedding_model = model
        self.constructed_with = constructed_with


def _install_embedder(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[list[list[float]] | Exception],
) -> tuple[_RecordingEmbeddingModel, dict[str, object]]:
    model = _RecordingEmbeddingModel(responses)
    constructed_with: dict[str, object] = {}

    def build_embedder(**kwargs: object) -> _RecordingEmbedder:
        constructed_with.update(kwargs)
        return _RecordingEmbedder(model, constructed_with)

    monkeypatch.setattr(embedding, "DefaultIndexingEmbedder", build_embedder)
    return model, constructed_with


def test_embedding_uses_snapshot_model_dimension_context_and_bounded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(request_size=2)
    job = _job(snapshot)
    rows, items = _rows_and_items(job, count=4)
    items[1].status = RegulatoryIndexingItemStatus.EMBEDDED.value
    items[1].vector = [0.4, 0.5, 0.6]
    model, constructed_with = _install_embedder(
        monkeypatch,
        responses=[
            [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
            [[1.0, 1.1, 1.2]],
        ],
    )
    persisted: list[tuple[list[tuple[object, list[float]]], int]] = []

    def persist_vectors(
        _db_session: Session,
        *,
        job_id: object,
        expected_generation: int,
        item_vectors: Sequence[tuple[object, list[float]]],
    ) -> bool:
        assert job_id == job.id
        persisted.append((list(item_vectors), expected_generation))
        return True

    monkeypatch.setattr(
        embedding.indexing_job_repository,
        "persist_regulatory_indexing_item_vectors",
        persist_vectors,
        raising=False,
    )

    summary = embed_pending_regulatory_items(
        job=job,
        rows=list(reversed(rows)),
        items=list(reversed(items)),
        search_settings=_settings(snapshot),
        tenant_id="tenant-a",
        db_session=_DB_SESSION,
    )

    assert constructed_with["model_name"] == "openai/text-embedding-3-large"
    assert constructed_with["provider_type"] is EmbeddingProvider.OPENROUTER
    assert constructed_with["reduced_dimension"] == 3
    assert [call["texts"] for call in model.calls] == [
        [
            "Context 0.\nMADDE 1 - legal text 0.",
            "Context 2.\nMADDE 3 - legal text 2.",
        ],
        ["Context 3.\nMADDE 4 - legal text 3."],
    ]
    assert all(call["text_type"] is EmbedTextType.PASSAGE for call in model.calls)
    assert all(call["api_embedding_batch_size"] == 2 for call in model.calls)
    assert all(call["tenant_id"] == "tenant-a" for call in model.calls)
    assert [[str(item_id) for item_id, _vector in batch] for batch, _ in persisted] == [
        [str(items[0].id), str(items[2].id)],
        [str(items[3].id)],
    ]
    assert [generation for _batch, generation in persisted] == [9, 9]
    assert summary == EmbeddingSummary(
        total_count=4,
        embedded_count=3,
        reused_count=1,
    )


def test_embedding_delivery_limits_provider_work_to_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(request_size=2)
    job = _job(snapshot)
    rows, items = _rows_and_items(job, count=5)
    model, _constructed_with = _install_embedder(
        monkeypatch,
        responses=[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
    )
    monkeypatch.setattr(
        embedding.indexing_job_repository,
        "persist_regulatory_indexing_item_vectors",
        lambda *_args, **_kwargs: True,
    )

    summary = embed_pending_regulatory_items(
        job=job,
        rows=rows,
        items=items,
        search_settings=_settings(snapshot),
        tenant_id="tenant-a",
        db_session=_DB_SESSION,
        max_batches=1,
    )

    assert len(model.calls) == 1
    assert summary == EmbeddingSummary(
        total_count=5,
        embedded_count=2,
        reused_count=0,
        remaining_count=3,
    )


@pytest.mark.parametrize(
    ("response", "error_match"),
    [
        ([[0.1, 0.2, 0.3]], "different number"),
        ([[0.1, 0.2], [0.4, 0.5, 0.6]], "dimension"),
        ([[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]], "finite"),
        ([[0.1, 0.2, 0.3], [0.4, float("inf"), 0.6]], "finite"),
    ],
)
def test_invalid_embedding_response_rejects_whole_request_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    response: list[list[float]],
    error_match: str,
) -> None:
    snapshot = _snapshot(request_size=2)
    job = _job(snapshot)
    rows, items = _rows_and_items(job, count=2)
    _install_embedder(monkeypatch, responses=[response])
    persisted: list[object] = []
    monkeypatch.setattr(
        embedding.indexing_job_repository,
        "persist_regulatory_indexing_item_vectors",
        lambda *_args, **_kwargs: persisted.append(object()) or True,
        raising=False,
    )

    with pytest.raises(ValueError, match=error_match):
        embed_pending_regulatory_items(
            job=job,
            rows=rows,
            items=items,
            search_settings=_settings(snapshot),
            tenant_id="tenant-a",
            db_session=_DB_SESSION,
        )

    assert persisted == []


def test_embedding_provider_index_rejection_precedes_vector_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(request_size=2)
    job = _job(snapshot)
    rows, items = _rows_and_items(job, count=2)
    _install_embedder(
        monkeypatch,
        responses=[ValueError("Embedding provider returned invalid vector indexes.")],
    )
    persisted: list[object] = []
    monkeypatch.setattr(
        embedding.indexing_job_repository,
        "persist_regulatory_indexing_item_vectors",
        lambda *_args, **_kwargs: persisted.append(object()) or True,
        raising=False,
    )

    with pytest.raises(ValueError, match="invalid vector indexes"):
        embed_pending_regulatory_items(
            job=job,
            rows=rows,
            items=items,
            search_settings=_settings(snapshot),
            tenant_id="tenant-a",
            db_session=_DB_SESSION,
        )

    assert persisted == []


def test_openrouter_boundary_rejects_duplicate_response_indexes() -> None:
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "object": "list",
            "model": "openai/text-embedding-3-large",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 0, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )
    cloud_embedding = CloudEmbedding(
        api_key="test-key",
        provider=EmbeddingProvider.OPENROUTER,
    )

    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=response)
    http_client.aclose = AsyncMock()
    cloud_embedding.http_client = http_client

    try:
        with pytest.raises(ValueError, match="invalid vector indexes"):
            asyncio.run(
                cloud_embedding._embed_litellm_proxy(
                    ["first", "second"],
                    "openai/text-embedding-3-large",
                    2,
                )
            )
    finally:
        asyncio.run(cloud_embedding.aclose())
