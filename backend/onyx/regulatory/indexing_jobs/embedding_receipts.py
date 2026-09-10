"""Exact durable encoder requests and positively evidenced vector checkpoints."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel

from pydantic import BaseModel, ConfigDict, Field

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig


class DurableEmbeddingReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    texts: list[str] = Field(min_length=1, max_length=1)
    configuration: dict[str, str | int | float | bool | None]
    source_text_sha256: str = Field(min_length=64, max_length=64)

    @property
    def sha256(self) -> str:
        return context_hash(self.model_dump(mode="json"))


def batch_embedding_configuration(
    *,
    provider: str,
    config: OpenRouterBatchConfig,
) -> dict[str, str | int | float | bool | None]:
    return {
        "transport": "openrouter_batch",
        "provider": provider,
        "model": config.model_name,
        "dimension": config.effective_dimension,
        "endpoint_sha256": context_hash(config.api_url),
        "embedding_endpoint": "/v1/embeddings",
        "formatter": "durable-context-before-text-v1",
    }


def batch_embedding_receipts(
    *,
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    config: OpenRouterBatchConfig,
) -> dict[UUID, DurableEmbeddingReceipt]:
    from onyx.regulatory.indexing_jobs.embedding import (
        _ordered_mapping,
        _text_for_embedding,
    )

    configuration = batch_embedding_configuration(
        provider=str(job.config_snapshot["embedding_provider"]),
        config=config,
    )
    return {
        item.id: DurableEmbeddingReceipt(
            texts=[text],
            configuration=configuration,
            source_text_sha256=context_hash(text),
        )
        for row, item in _ordered_mapping(job, rows, items)
        for text in [_text_for_embedding(row, item)]
    }


def item_embedding_receipt(
    item: RegulatoryIndexingItem,
) -> DurableEmbeddingReceipt | None:
    payload = (item.context or {}).get("embedding_receipt")
    return (
        DurableEmbeddingReceipt.model_validate(payload) if payload is not None else None
    )


def vector_receipt_context(
    context: dict[str, object] | None,
    vector: list[float],
) -> dict[str, object]:
    result = dict(context or {})
    payload = result.get("embedding_receipt")
    if payload is not None:
        receipt = DurableEmbeddingReceipt.model_validate(payload)
        result["embedding_vector_receipt_sha256"] = context_hash(
            [receipt.sha256, vector]
        )
    return result


def has_proven_vector(
    item: RegulatoryIndexingItem, receipt: DurableEmbeddingReceipt
) -> bool:
    return (
        item.status == "EMBEDDED"
        and isinstance(item.vector, list)
        and len(item.vector) == receipt.configuration.get("dimension")
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in item.vector
        )
        and item_embedding_receipt(item) == receipt
        and (item.context or {}).get("embedding_vector_receipt_sha256")
        == context_hash([receipt.sha256, item.vector])
    )


def synchronous_embedding_receipts(
    *,
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    model: "EmbeddingModel",
) -> dict[UUID, DurableEmbeddingReceipt]:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        freeze_encoder_inputs,
    )
    from onyx.regulatory.indexing_jobs.embedding import (
        _ordered_mapping,
        _text_for_embedding,
    )

    dimension = job.config_snapshot["effective_dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise ValueError("durable embedding dimension is unresolved")
    if (
        (model.provider_type.value if model.provider_type else None)
        != job.config_snapshot["embedding_provider"]
        or model.model_name != job.config_snapshot["embedding_model_name"]
        or model.reduced_dimension != dimension
    ):
        raise ValueError("actual durable encoder does not match the job snapshot")
    receipts = {}
    for row, item in _ordered_mapping(job, rows, items):
        text = _text_for_embedding(row, item)
        texts, configuration = freeze_encoder_inputs(
            [text],
            model,
            model_dim=dimension,
            formatter="durable-context-before-text-v1",
        )
        # Passing these already fitted strings to encode must be idempotent.
        repeated, _ = freeze_encoder_inputs(
            texts,
            model,
            model_dim=dimension,
            formatter="durable-context-before-text-v1",
        )
        if repeated != texts:
            raise ValueError("durable encoder preprocessing is not idempotent")
        configuration["transport"] = "synchronous_encoder"
        receipts[item.id] = DurableEmbeddingReceipt(
            texts=texts,
            configuration=configuration,
            source_text_sha256=context_hash(text),
        )
    return receipts
