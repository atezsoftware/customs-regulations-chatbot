"""Execute the immutable approved inventory, with durable provider results and fencing."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context
from threading import Event, Thread
from uuid import UUID, uuid4

from onyx.db.regulatory_annex_execution import (
    embedding_checkpoint,
    freeze_operations,
    frozen_operations,
    load_delivery,
    mark_es_started,
    record_embedding,
    stage_manifest,
)
from onyx.db.regulatory_publication import PUBLICATION_LEASE_TTL, PublicationStore
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import FileOwnership
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    freeze_encoder_inputs,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexPublicationPreparation,
)
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexPublicationDelivery,
    AnnexPublicationNeedsReconciliation,
)
from onyx.regulatory.amendments.annexes.publication_operations import build_operations
from onyx.regulatory.amendments.annexes.publication_preparation import (
    _target_configuration,
    validate_frozen_publication_review,
)
from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot

LEASE_TTL = PUBLICATION_LEASE_TTL
HEARTBEAT_SECONDS = 20


@contextmanager
def publication_heartbeat(owner: FileOwnership) -> Generator[Event, None, None]:
    stop, lost = Event(), Event()

    def renew() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                PublicationStore(owner.scope).heartbeat(owner, ttl=LEASE_TTL)
            except Exception:
                lost.set()
                return

    context = copy_context()
    thread = Thread(target=lambda: context.run(renew), daemon=True)
    thread.start()
    try:
        yield lost
    finally:
        stop.set()
        thread.join()


def runtime_embedders(
    draft: AnnexChangeDraft, prepared: AnnexPublicationPreparation
) -> dict[int, DefaultIndexingEmbedder]:
    from onyx.db.regulatory_annex_execution import load_validated_runtime_settings
    from onyx.regulatory.amendments.annexes.analysis import resolve_review_context_llm

    settings = load_validated_runtime_settings(draft, prepared)
    result = {}
    for setting in settings:
        index = next(
            item for item in prepared.indexes if item.search_settings_id == setting.id
        )
        snapshot = (
            RegulatoryIndexingConfigSnapshot.model_validate(
                draft.indexing_configuration
            )
            if setting.status.is_current() and draft.indexing_configuration
            else None
        )
        embedder = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=setting
        )
        llm = resolve_review_context_llm(setting, snapshot)
        if (
            context_hash(
                [
                    _target_configuration(setting, snapshot, embedder),
                    llm.config.model_dump(mode="json") if llm else None,
                ]
            )
            != prepared.runtime_configuration[index.index_uuid]
        ):
            raise ValueError(
                "publication encoder/context runtime configuration changed"
            )
        result[setting.id] = embedder
    return result


def execute_embeddings(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    draft: AnnexChangeDraft,
    prepared: AnnexPublicationPreparation,
    embedders: dict[int, DefaultIndexingEmbedder],
) -> bool:
    from onyx.regulatory.amendments.annexes.publication_batch import (
        execute_batch_embedding,
    )
    from shared_configs.enums import EmbedTextType

    complete = True
    for plan in prepared.projections:
        if plan.reuse_from is not None:
            continue
        checkpoint = embedding_checkpoint(delivery, plan.id)
        if checkpoint.status == "complete":
            continue
        model = embedders[plan.index.search_settings_id].embedding_model
        if checkpoint.request.configuration.get("transport") == "openrouter_batch":
            complete = (
                execute_batch_embedding(owner, delivery, draft, plan.id, model.api_key)
                and complete
            )
            continue
        dimension = checkpoint.request.configuration.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ValueError("frozen encoder dimension missing")
        actual_inputs, configuration = freeze_encoder_inputs(
            checkpoint.request.inputs,
            model,
            model_dim=dimension,
            formatter=str(checkpoint.request.configuration["formatter"]),
        )
        comparable = {
            key: value
            for key, value in checkpoint.request.configuration.items()
            if key != "transport"
        }
        if actual_inputs != checkpoint.request.inputs or configuration != comparable:
            raise ValueError(
                "actual encoder no longer matches frozen complete inputs/configuration"
            )
        raw = model.encode(
            texts=actual_inputs,
            text_type=EmbedTextType.PASSAGE,
            tenant_id=delivery.tenant_id,
        )
        vectors = _validate_response_vectors(
            raw,
            expected_count=len(actual_inputs),
            expected_dimension=checkpoint.request.dimension,
        )
        record_embedding(
            owner,
            delivery,
            checkpoint.model_copy(update={"status": "complete", "vectors": vectors}),
        )
    return complete


def validate_actual_baseline(
    prepared: AnnexPublicationPreparation, owner: FileOwnership
) -> None:
    reservations = PublicationStore(owner.scope).reservations(owner)
    with ElasticsearchClient() as transport:
        for index in prepared.indexes:
            actual = FencedPublicationIndex(
                transport.publication_client(), index
            ).inventory_evidence(reservations)
            before = [
                item.evidence
                for item in prepared.indexed_baseline
                if item.evidence.index.index_uuid == index.index_uuid
            ]
            if context_hash(
                [json.loads(item.source_json) for item in actual]
            ) != context_hash([json.loads(item.source_json) for item in before]):
                raise ValueError("actual indexed publication baseline changed")


def validate_delivery_scope(delivery: AnnexPublicationDelivery) -> None:
    from onyx.regulatory.amendments.annexes import config
    from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
    from shared_configs.contextvars import get_current_tenant_id

    if (
        delivery.environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or delivery.database_identity != config.ANNEX_DATABASE_IDENTITY
        or delivery.tenant_id != get_current_tenant_id()
        or not MULTI_TENANT
        and delivery.tenant_id != POSTGRES_DEFAULT_SCHEMA
    ):
        raise ValueError("publication execution scope mismatch")


def execute_publication(delivery: AnnexPublicationDelivery) -> str:
    validate_delivery_scope(delivery)
    status, draft = load_delivery(delivery)
    if status == "approved":
        return status
    prepared = validate_frozen_publication_review(draft)
    authority = PublicationStore(prepared.scope)
    owner = authority.acquire(prepared.user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        with publication_heartbeat(owner) as lost:
            started = stage_manifest(owner, delivery, prepared)
            embedders = runtime_embedders(draft, prepared)
            if not started:
                validate_actual_baseline(prepared, owner)
            if not execute_embeddings(owner, delivery, draft, prepared, embedders):
                return "preparing"
            operations = frozen_operations(delivery)
            if operations is None:
                operations = build_operations(delivery, prepared)
                freeze_operations(owner, delivery, operations)
            if not started:
                validate_actual_baseline(prepared, owner)
            mark_es_started(owner, delivery, prepared)
            reservations = authority.reservations(owner)
            proofs = []
            with ElasticsearchClient() as transport:
                adapters = {
                    index.index_uuid: FencedPublicationIndex(
                        transport.publication_client(), index
                    )
                    for index in prepared.indexes
                }
                for adapter in adapters.values():
                    adapter.seal(reservations)
                for operation in operations.operations:
                    if lost.is_set():
                        raise ValueError("publication ownership heartbeat lost")
                    authority.reservations(owner)
                    adapter = adapters[operation.index_uuid]
                    if operation.kind == "tombstone":
                        adapter.tombstone(reservations, operation.ordinal)
                    elif operation.kind == "retain" and operation.retained is not None:
                        adapter.retain(reservations, operation.retained)
                    elif operation.binding is not None:
                        adapter.upsert(reservations, operation.binding.projection)
                    else:
                        raise ValueError("frozen live operation has no binding")
                for index in prepared.indexes:
                    projections = tuple(
                        operation.binding.projection
                        for operation in operations.operations
                        if operation.index_uuid == index.index_uuid
                        and operation.binding
                    )
                    proofs.append(
                        adapters[index.index_uuid].verify(
                            reservations,
                            projections,
                            tuple(
                                op.retained
                                for op in operations.operations
                                if op.index_uuid == index.index_uuid
                                and op.retained is not None
                            ),
                        )
                    )
            if lost.is_set():
                raise ValueError("publication ownership heartbeat lost")
            runtime_embedders(draft, prepared)
        # Stop/join the independent heartbeat BEFORE taking the activation authority lock.
        from onyx.db.regulatory_annex_activation import activate_publication

        activate_publication(owner, delivery, prepared, operations, proofs)
        return "approved"
    except Exception as error:
        from onyx.db.regulatory_annex_execution import record_publication_failure
        from onyx.regulatory.indexing_jobs.models import (
            IndexingGatewayIndeterminateSubmissionError,
        )
        from onyx.regulatory.indexing_jobs.openrouter_batch import (
            OpenRouterBatchContractError,
        )

        try:
            record_publication_failure(
                owner,
                delivery,
                requires_reconciliation=isinstance(
                    error,
                    (
                        IndexingGatewayIndeterminateSubmissionError,
                        AnnexPublicationNeedsReconciliation,
                        OpenRouterBatchContractError,
                    ),
                ),
            )
        except ValueError:
            pass
        raise
    finally:
        try:
            authority.release(owner)
        except ValueError:
            # A successor owns the durable closed gate; stale cleanup has no authority.
            pass


def reconcile_annex_batch_submission(
    delivery: AnnexPublicationDelivery, *, projection_id: UUID, candidate_remote_id: str
) -> str:
    """Operator selects one candidate; authenticated opaque request correlation authorizes resume."""
    from onyx.db.regulatory_annex_execution import (
        complete_batch_reconciliation,
        prepare_batch_reconciliation,
    )
    from onyx.regulatory.amendments.annexes.publication_batch import (
        correlate_completed_batch,
    )

    validate_delivery_scope(delivery)
    _status, draft = load_delivery(delivery, allow_reconciliation=True)
    prepared = validate_frozen_publication_review(draft)
    authority = PublicationStore(prepared.scope)
    owner = authority.acquire(prepared.user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        with publication_heartbeat(owner) as lost:
            checkpoint = prepare_batch_reconciliation(
                owner, delivery, prepared, projection_id
            )
            plan = next(
                item for item in prepared.projections if item.id == projection_id
            )
            embedders = runtime_embedders(draft, prepared)
            validate_actual_baseline(prepared, owner)
            completed = correlate_completed_batch(
                draft,
                checkpoint,
                candidate_remote_id=candidate_remote_id,
                api_key=embedders[
                    plan.index.search_settings_id
                ].embedding_model.api_key,
            )
            if lost.is_set():
                raise ValueError("publication ownership heartbeat lost")
            runtime_embedders(draft, prepared)
        complete_batch_reconciliation(owner, delivery, prepared, completed)
        return "preparing"
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass
