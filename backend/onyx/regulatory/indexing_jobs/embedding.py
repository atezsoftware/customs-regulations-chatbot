from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.indexing_jobs.contextual import contextualized_embedding_text
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from shared_configs.enums import EmbedTextType


class EmbeddingSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_count: int = Field(ge=0)
    embedded_count: int = Field(ge=0)
    reused_count: int = Field(ge=0)


def _is_valid_vector(vector: object, expected_dimension: int) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == expected_dimension
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in vector
        )
    )


def _validate_search_settings(
    search_settings: SearchSettings,
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> None:
    current_values = (
        search_settings.id,
        search_settings.provider_type,
        search_settings.model_name,
        search_settings.model_dim,
        search_settings.reduced_dimension,
        search_settings.final_embedding_dim,
        search_settings.index_name,
    )
    snapshot_values = (
        snapshot.search_settings_id,
        snapshot.embedding_provider,
        snapshot.embedding_model_name,
        snapshot.model_dimension,
        snapshot.reduced_dimension,
        snapshot.effective_dimension,
        snapshot.index_name,
    )
    if current_values != snapshot_values:
        raise ValueError("SearchSettings no longer matches the indexing job snapshot")
    if not search_settings.api_key:
        raise ValueError("The snapshot embedding provider has no configured credential")


def _ordered_mapping(
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
) -> list[tuple[RegulatoryChunk, RegulatoryIndexingItem]]:
    ordered_rows = sorted(rows, key=lambda row: (row.position, row.id))
    if not ordered_rows:
        raise ValueError("regulatory indexing job has no canonical chunks")
    if any(row.user_file_id != job.user_file_id for row in ordered_rows):
        raise ValueError("canonical chunk belongs to a different user file")
    row_ids = [row.id for row in ordered_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("canonical chunks contain duplicate ids")

    item_by_row_id: dict[str, RegulatoryIndexingItem] = {}
    for item in items:
        if item.job_id != job.id:
            raise ValueError("embedding item belongs to a different job")
        if item.regulatory_chunk_id in item_by_row_id:
            raise ValueError("embedding items contain a duplicate canonical chunk")
        item_by_row_id[item.regulatory_chunk_id] = item
    if set(item_by_row_id) != set(row_ids):
        raise ValueError("embedding items do not exactly cover canonical chunks")
    return [(row, item_by_row_id[row.id]) for row in ordered_rows]


def _text_for_embedding(
    row: RegulatoryChunk,
    item: RegulatoryIndexingItem,
) -> str:
    if (
        item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
        and item.context is None
    ):
        return row.text
    return contextualized_embedding_text(row, item)


def _validate_response_vectors(
    vectors: Sequence[object],
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            "Embedding provider returned a different number of vectors than inputs"
        )
    validated: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list):
            raise ValueError("Embedding provider returned an invalid vector")
        if len(vector) != expected_dimension:
            raise ValueError(
                "Embedding provider returned vector dimension "
                f"{len(vector)}; expected {expected_dimension}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Embedding provider returned a non-finite vector")
        numeric_vector = cast(list[int | float], vector)
        validated.append([float(value) for value in numeric_vector])
    return validated


def embed_pending_regulatory_items(
    *,
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    search_settings: SearchSettings,
    tenant_id: str,
    db_session: Session,
) -> EmbeddingSummary:
    """Embed only missing/invalid item vectors in bounded, atomic batches."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    _validate_search_settings(search_settings, snapshot)
    ordered = _ordered_mapping(job, rows, items)

    pending = [
        (row, item)
        for row, item in ordered
        if not (
            item.status == RegulatoryIndexingItemStatus.EMBEDDED.value
            and _is_valid_vector(item.vector, snapshot.effective_dimension)
        )
    ]
    invalid_statuses = [
        item.status
        for _row, item in pending
        if item.status
        not in {
            RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            RegulatoryIndexingItemStatus.SKIPPED.value,
            RegulatoryIndexingItemStatus.EMBEDDED.value,
        }
    ]
    if invalid_statuses:
        raise ValueError("regulatory indexing item is not ready for embedding")

    embedder = DefaultIndexingEmbedder(
        model_name=snapshot.embedding_model_name,
        normalize=search_settings.normalize,
        query_prefix=search_settings.query_prefix,
        passage_prefix=search_settings.passage_prefix,
        provider_type=snapshot.embedding_provider,
        api_key=search_settings.api_key,
        api_url=search_settings.api_url,
        api_version=search_settings.api_version,
        deployment_name=search_settings.deployment_name,
        reduced_dimension=snapshot.effective_dimension,
    )

    embedded_count = 0
    request_size = snapshot.embedding_request_size
    for start in range(0, len(pending), request_size):
        batch = pending[start : start + request_size]
        texts = [_text_for_embedding(row, item) for row, item in batch]
        raw_vectors = embedder.embedding_model.encode(
            texts=texts,
            text_type=EmbedTextType.PASSAGE,
            api_embedding_batch_size=request_size,
            tenant_id=tenant_id,
        )
        vectors = _validate_response_vectors(
            raw_vectors,
            expected_count=len(batch),
            expected_dimension=snapshot.effective_dimension,
        )
        persisted = indexing_job_repository.persist_regulatory_indexing_item_vectors(
            db_session,
            job_id=job.id,
            expected_generation=job.lease_generation,
            item_vectors=[
                (item.id, vector)
                for (_row, item), vector in zip(batch, vectors, strict=True)
            ],
        )
        if not persisted:
            raise RuntimeError("regulatory indexing lease was lost while embedding")
        embedded_count += len(batch)

    return EmbeddingSummary(
        total_count=len(ordered),
        embedded_count=embedded_count,
        reused_count=len(ordered) - embedded_count,
    )
