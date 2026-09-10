"""Durable batch submissions and bounded authenticated request correlation."""

from uuid import UUID

import httpx

from onyx.db.regulatory_annex_execution import embedding_checkpoint, record_embedding
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexBatchCorrelationReceipt,
    AnnexEmbeddingCheckpoint,
    AnnexPublicationDelivery,
    AnnexPublicationNeedsReconciliation,
)
from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from onyx.regulatory.indexing_jobs.openrouter_batch import (
    HttpxOpenRouterBatchGateway,
    OpenRouterBatchJobStatus,
    OpenRouterBatchState,
    parse_openrouter_embedding_results,
)


def completed_batch_checkpoint(
    checkpoint: AnnexEmbeddingCheckpoint,
    state: OpenRouterBatchState,
    *,
    expected_remote_id: str,
) -> AnnexEmbeddingCheckpoint:
    if (
        state.remote_batch_id != expected_remote_id
        or state.status != OpenRouterBatchJobStatus.SUCCEEDED
        or state.results is None
    ):
        raise AnnexPublicationNeedsReconciliation(
            "batch is not the completed correlated provider job"
        )
    expected_model = checkpoint.request.configuration.get("model")
    if not isinstance(expected_model, str):
        raise ValueError("frozen embedding model missing")
    results = parse_openrouter_embedding_results(
        state.results,
        expected_custom_ids={checkpoint.request.custom_id},
        expected_model=expected_model,
        expected_dimension=checkpoint.request.dimension,
    )
    result = results.get(checkpoint.request.custom_id)
    if result is None or result.vectors is None or len(results) != 1:
        raise AnnexPublicationNeedsReconciliation(
            "correlated batch result missing or failed"
        )
    vectors = _validate_response_vectors(
        result.vectors,
        expected_count=len(checkpoint.request.inputs),
        expected_dimension=checkpoint.request.dimension,
    )
    receipt = AnnexBatchCorrelationReceipt(
        remote_id=expected_remote_id,
        custom_id=checkpoint.request.custom_id,
        request_sha256=publication_digest(checkpoint.request.model_dump(mode="json")),
        response_json=state.model_dump_json(),
    )
    return checkpoint.model_copy(
        update={
            "status": "complete",
            "remote_id": expected_remote_id,
            "vectors": vectors,
            "provider_receipt": receipt,
        }
    )


def correlate_completed_batch(
    draft: AnnexChangeDraft,
    checkpoint: AnnexEmbeddingCheckpoint,
    *,
    candidate_remote_id: str,
    api_key: str | None,
) -> AnnexEmbeddingCheckpoint:
    if (
        draft.indexing_configuration is None
        or checkpoint.request.configuration.get("transport") != "openrouter_batch"
    ):
        raise ValueError("frozen batch authority missing")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(
        draft.indexing_configuration
    )
    if snapshot.openrouter_batch is None:
        raise ValueError("frozen batch authority missing")
    with httpx.Client(timeout=60.0) as client:
        gateway = HttpxOpenRouterBatchGateway(
            config=snapshot.openrouter_batch,
            api_key_provider=lambda: api_key or "",
            client=client,
        )
        state = gateway.get(candidate_remote_id)
    return completed_batch_checkpoint(
        checkpoint, state, expected_remote_id=candidate_remote_id
    )


def execute_batch_embedding(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    draft: AnnexChangeDraft,
    projection_id: UUID,
    api_key: str | None,
) -> bool:
    from onyx.regulatory.indexing_jobs.openrouter_batch import (
        HttpxOpenRouterBatchGateway,
        OpenRouterBatchJobStatus,
        OpenRouterEmbeddingBatchRequest,
    )

    if not isinstance(projection_id, UUID) or draft.indexing_configuration is None:
        raise ValueError("frozen batch configuration missing")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(
        draft.indexing_configuration
    )
    config = snapshot.openrouter_batch
    if config is None:
        raise ValueError("frozen batch configuration missing")
    checkpoint = embedding_checkpoint(delivery, projection_id)
    if checkpoint.status in ("submitting", "indeterminate"):
        if checkpoint.status != "indeterminate":
            record_embedding(
                owner,
                delivery,
                checkpoint.model_copy(update={"status": "indeterminate"}),
            )
        raise AnnexPublicationNeedsReconciliation(
            "embedding submission requires manual provider reconciliation"
        )
    with httpx.Client(timeout=60.0) as http_client:
        client = HttpxOpenRouterBatchGateway(
            config=config, api_key_provider=lambda: api_key or "", client=http_client
        )
        custom_id = checkpoint.request.custom_id
        if checkpoint.status == "pending":
            request = OpenRouterEmbeddingBatchRequest(
                custom_id=custom_id, inputs=checkpoint.request.inputs
            )
            if len(request.inputs) > config.request_input_size:
                raise ValueError(
                    "frozen projection exceeds approved batch request bound"
                )
            checkpoint = checkpoint.model_copy(update={"status": "submitting"})
            record_embedding(owner, delivery, checkpoint)
            state = client.submit(
                [request], submission_key=f"annex-{delivery.change_set_id}-{custom_id}"
            )
            checkpoint = checkpoint.model_copy(
                update={"status": "submitted", "remote_id": state.remote_batch_id}
            )
            record_embedding(owner, delivery, checkpoint)
        else:
            if checkpoint.remote_id is None:
                raise ValueError("durable batch remote identity missing")
            state = client.get(checkpoint.remote_id)
        if state.status in (
            OpenRouterBatchJobStatus.PENDING,
            OpenRouterBatchJobStatus.RUNNING,
        ):
            return False
        if state.status != OpenRouterBatchJobStatus.SUCCEEDED or state.results is None:
            raise AnnexPublicationNeedsReconciliation(
                "frozen embedding batch failed; retained provider identity requires reconciliation"
            )
        assert checkpoint.remote_id is not None
        completed = completed_batch_checkpoint(
            checkpoint, state, expected_remote_id=checkpoint.remote_id
        )
        record_embedding(owner, delivery, completed)
        return True
