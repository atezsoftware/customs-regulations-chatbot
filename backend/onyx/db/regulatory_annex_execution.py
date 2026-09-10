"""Owned short transactions for durable annex publication checkpoints."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import (
    AnnexChangeSet,
    AnnexPublicationEmbedding,
    AnnexPublicationIntent,
    AnnexPublicationManifest,
    SearchSettings,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexPublicationPreparation,
)
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexEmbeddingCheckpoint,
    AnnexEmbeddingRequest,
    AnnexPublicationDelivery,
    AnnexPublicationOperations,
)


def require_delivery(
    session: Session,
    delivery: AnnexPublicationDelivery,
    *,
    locked: bool = False,
    allow_reconciliation: bool = False,
) -> tuple[AnnexChangeSet, AnnexChangeDraft]:
    from onyx.db.regulatory_annex_changes import require_current_annex_review

    intent = session.get(AnnexPublicationIntent, delivery.intent_id)
    if intent is None or any(
        getattr(intent, key) != value
        for key, value in delivery.model_dump(exclude={"intent_id"}).items()
    ):
        raise ValueError("publication intent scope mismatch")
    if locked:
        review = require_current_annex_review(
            session,
            change_set_id=delivery.change_set_id,
            expected_review_sha256=delivery.review_sha256,
            environment=delivery.environment,
        )
    else:
        review = session.get(AnnexChangeSet, delivery.change_set_id)
    if (
        review is None
        or review.publication_generation != delivery.publication_generation
        or review.review_sha256 != delivery.review_sha256
    ):
        raise ValueError("stale publication generation or review")
    if publication_digest(review.review_payload) != review.review_sha256:
        raise ValueError("immutable review payload changed")
    if (
        not allow_reconciliation
        and review.status == "failed"
        and review.error_message
        and "provider reconciliation" in review.error_message
    ):
        raise ValueError("publication requires manual provider reconciliation")
    reconciliation = (
        allow_reconciliation
        and review.status == "failed"
        and review.error_message is not None
        and "provider reconciliation" in review.error_message
    )
    if not reconciliation and review.status not in (
        "approving",
        "preparing",
        "publishing",
        "approved",
    ):
        raise ValueError("publication review state does not allow execution")
    return review, AnnexChangeDraft.model_validate(review.review_payload)


def load_delivery(
    delivery: AnnexPublicationDelivery, *, allow_reconciliation: bool = False
) -> tuple[str, AnnexChangeDraft]:
    with get_session_with_tenant(tenant_id=delivery.tenant_id) as session:
        review, draft = require_delivery(
            session, delivery, allow_reconciliation=allow_reconciliation
        )
        return review.status, draft


@contextmanager
def owned_execution(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    *,
    allow_reconciliation: bool = False,
) -> Generator[tuple[Session, AnnexChangeSet, AnnexChangeDraft], None, None]:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        review, draft = require_delivery(
            session, delivery, locked=True, allow_reconciliation=allow_reconciliation
        )
        if draft.user_file_id != owner.user_file_id:
            raise ValueError("publication owner file mismatch")
        yield session, review, draft
        session.commit()


def validate_database_baseline(
    session: Session, draft: AnnexChangeDraft, prepared: AnnexPublicationPreparation
) -> None:
    from onyx.db.regulatory_amendments import get_batch
    from onyx.db.regulatory_annex_changes import (
        lock_annex_preparation_scope,
        validate_prepared_annex_change,
    )
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        publication_input_scope_hash,
    )

    lock_annex_preparation_scope(session, draft.user_file_id)
    file, settings, access = load_annex_publication_inputs(session, draft)
    PublicationStore(prepared.scope).lock_clock(session)
    from onyx.db.regulatory_physical_indexes import validate_physical_index_snapshots

    validate_physical_index_snapshots(session, prepared.indexes)
    # Lock all concrete targets, including FUTURE, before checking the frozen set.
    from sqlalchemy import inspect

    for setting in settings:
        if inspect(setting).persistent:
            session.refresh(setting, with_for_update=True)
    if (
        publication_input_scope_hash(file, settings, access)
        != prepared.input_scope_sha256
    ):
        raise ValueError("publication file/ACL/index configuration changed")
    batch = get_batch(session, prepared.batch_id)
    if batch is None:
        raise ValueError("publication batch missing")
    validate_prepared_annex_change(
        session, batch=batch, draft=draft, environment=prepared.scope.environment
    )


def source_history_snapshot(
    session: Session, draft: AnnexChangeDraft
) -> dict[str, object]:
    import json

    from sqlalchemy import inspect

    from onyx.db.models import (
        RegulatoryAnnex,
        RegulatoryAnnexElementChunk,
        RegulatoryAnnexRevision,
        RegulatoryAnnexRevisionElement,
    )
    from onyx.db.regulatory_annexes import get_effective_annex_revision

    if (
        draft.baseline is None
        or draft.baseline.revision_id is None
        or draft.effective_date is None
    ):
        raise ValueError("annex source history baseline missing")
    baseline = session.get(RegulatoryAnnexRevision, UUID(draft.baseline.revision_id))
    if baseline is None:
        raise ValueError("annex source history baseline missing")
    annex = session.scalar(
        select(RegulatoryAnnex)
        .where(RegulatoryAnnex.id == baseline.annex_id)
        .with_for_update()
    )
    if annex is None or annex.user_file_id != draft.user_file_id:
        raise ValueError("annex source history scope mismatch")
    effective = get_effective_annex_revision(session, annex.id, draft.effective_date)
    if (
        effective is None
        or effective.id != baseline.id
        or baseline.baseline_sha256 != draft.baseline.baseline_sha256
    ):
        raise ValueError("annex source history changed")
    revisions = list(
        session.scalars(
            select(RegulatoryAnnexRevision)
            .where(RegulatoryAnnexRevision.annex_id == annex.id)
            .order_by(RegulatoryAnnexRevision.id)
            .with_for_update()
        )
    )
    ids = [row.id for row in revisions]
    elements = list(
        session.scalars(
            select(RegulatoryAnnexRevisionElement)
            .where(RegulatoryAnnexRevisionElement.revision_id.in_(ids))
            .order_by(
                RegulatoryAnnexRevisionElement.revision_id,
                RegulatoryAnnexRevisionElement.position,
            )
            .with_for_update()
        )
    )
    links = list(
        session.scalars(
            select(RegulatoryAnnexElementChunk)
            .where(RegulatoryAnnexElementChunk.revision_id.in_(ids))
            .order_by(
                RegulatoryAnnexElementChunk.revision_id,
                RegulatoryAnnexElementChunk.element_id,
                RegulatoryAnnexElementChunk.chunk_id,
            )
            .with_for_update()
        )
    )

    def payload(
        row: RegulatoryAnnex
        | RegulatoryAnnexRevision
        | RegulatoryAnnexRevisionElement
        | RegulatoryAnnexElementChunk,
    ) -> dict[str, object]:
        return {
            column.key: getattr(row, column.key)
            for column in inspect(type(row)).columns
        }

    return json.loads(
        json.dumps(
            {
                "annex": payload(annex),
                "revisions": [payload(row) for row in revisions],
                "elements": [payload(row) for row in elements],
                "links": [payload(row) for row in links],
            },
            default=str,
        )
    )


def validate_staged_history(
    session: Session,
    draft: AnnexChangeDraft,
    prepared: AnnexPublicationPreparation,
    manifest: AnnexPublicationManifest,
) -> None:
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings

    if source_history_snapshot(session, draft) != manifest.source_history:
        raise ValueError("annex source history changed")
    index_ids = {index.index_uuid for index in prepared.indexes}
    current = {
        binding.id: binding
        for binding in load_file_temporal_bindings(session, prepared.user_file_id)
        if binding.index.index_uuid in index_ids
    }
    expected = {
        item.binding.id: item.binding
        for item in prepared.indexed_baseline
        if item.binding
    }
    if current != expected:
        raise ValueError("qualified temporal baseline inventory changed")


def stage_manifest(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    prepared: AnnexPublicationPreparation,
) -> bool:
    with owned_execution(owner, delivery) as (session, review, draft):
        validate_database_baseline(session, draft, prepared)
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if list(reservations.ordinals) != prepared.reserved_ordinals:
            raise ValueError("publication reservation baseline changed")
        row = session.get(AnnexPublicationManifest, review.id)
        payload = prepared.model_dump(mode="json")
        if row is None:
            row = AnnexPublicationManifest(
                change_set_id=review.id,
                first_intent_id=delivery.intent_id,
                source_history=source_history_snapshot(session, draft),
                payload=payload,
                payload_sha256=publication_digest(payload),
                es_started=False,
            )
            session.add(row)
            session.flush()
            for plan in prepared.projections:
                if plan.reuse_from is not None:
                    continue
                request = AnnexEmbeddingRequest(
                    projection_id=plan.id,
                    custom_id=f"annex-{uuid4().hex}",
                    inputs=plan.context.embedding_texts,
                    configuration=plan.context.embedding_config,
                    dimension=plan.index.vector_dimension,
                ).model_dump(mode="json")
                session.add(
                    AnnexPublicationEmbedding(
                        change_set_id=review.id,
                        projection_id=plan.id,
                        request=request,
                        request_sha256=publication_digest(request),
                        status="pending",
                    )
                )
        elif row.payload != payload or row.payload_sha256 != publication_digest(
            payload
        ):
            raise ValueError("durable publication manifest changed")
        validate_staged_history(session, draft, prepared, row)
        review.status = "publishing" if row.es_started else "preparing"
        review.heartbeat_at = datetime.now(timezone.utc)
        return row.es_started


def embedding_checkpoint(
    delivery: AnnexPublicationDelivery,
    projection_id: UUID,
    *,
    allow_reconciliation: bool = False,
) -> AnnexEmbeddingCheckpoint:
    with get_session_with_tenant(tenant_id=delivery.tenant_id) as session:
        require_delivery(session, delivery, allow_reconciliation=allow_reconciliation)
        row = session.get(
            AnnexPublicationEmbedding, (delivery.change_set_id, projection_id)
        )
        if row is None or publication_digest(row.request) != row.request_sha256:
            raise ValueError("embedding request missing or changed")
        if (
            row.vectors is not None
            and publication_digest(row.vectors) != row.vectors_sha256
        ):
            raise ValueError("staged embedding result changed")
        return AnnexEmbeddingCheckpoint.model_validate(
            dict(
                request=row.request,
                status=row.status,
                remote_id=row.remote_id,
                vectors=row.vectors,
                provider_receipt=row.provider_receipt,
            )
        )


def record_embedding(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    checkpoint: AnnexEmbeddingCheckpoint,
) -> None:
    with owned_execution(owner, delivery) as (session, _review, _draft):
        row = session.get(
            AnnexPublicationEmbedding,
            (delivery.change_set_id, checkpoint.request.projection_id),
        )
        if row is None or row.request != checkpoint.request.model_dump(mode="json"):
            raise ValueError("embedding checkpoint request changed")
        if row.status == "complete":
            if row.vectors != checkpoint.vectors:
                raise ValueError("completed embedding result changed")
            return
        if checkpoint.vectors is not None:
            from onyx.regulatory.indexing_jobs.embedding import (
                _validate_response_vectors,
            )

            _validate_response_vectors(
                checkpoint.vectors,
                expected_count=len(checkpoint.request.inputs),
                expected_dimension=checkpoint.request.dimension,
            )
        row.status, row.remote_id, row.vectors = (
            checkpoint.status,
            checkpoint.remote_id,
            checkpoint.vectors,
        )
        row.provider_receipt = (
            checkpoint.provider_receipt.model_dump(mode="json")
            if checkpoint.provider_receipt
            else None
        )
        row.vectors_sha256 = (
            publication_digest(checkpoint.vectors)
            if checkpoint.vectors is not None
            else None
        )


def frozen_operations(
    delivery: AnnexPublicationDelivery,
) -> AnnexPublicationOperations | None:
    with get_session_with_tenant(tenant_id=delivery.tenant_id) as session:
        require_delivery(session, delivery)
        row = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if row is None:
            raise ValueError("publication manifest missing")
        if row.operations is None:
            return None
        if publication_digest(row.operations) != row.operations_sha256:
            raise ValueError("frozen publication operations changed")
        return AnnexPublicationOperations.model_validate(row.operations)


def freeze_operations(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    operations: AnnexPublicationOperations,
) -> None:
    with owned_execution(owner, delivery) as (session, _review, _draft):
        row = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if row is None:
            raise ValueError("publication manifest missing")
        payload = operations.model_dump(mode="json")
        if row.operations is not None and row.operations != payload:
            raise ValueError("publication operations cannot change on retry")
        row.operations, row.operations_sha256 = payload, publication_digest(payload)


def mark_es_started(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    prepared: AnnexPublicationPreparation,
) -> None:
    with owned_execution(owner, delivery) as (session, review, draft):
        validate_database_baseline(session, draft, prepared)
        row = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if row is None or row.operations is None:
            raise ValueError("ES publication requires frozen operations")
        validate_staged_history(session, draft, prepared, row)
        row.es_started = True
        review.status = "publishing"
        # Committed before any ES seal or write. Recovery never compares the now-partial index to OLD.
        PublicationStore(owner.scope).record_event(session, owner)


def pending_deliveries(
    *, tenant_id: str, environment: str, database_identity: str, limit: int = 100
) -> list[AnnexPublicationDelivery]:
    from onyx.db.regulatory_annex_changes import list_pending_annex_publication_intents

    with get_session_with_tenant(tenant_id=tenant_id) as session:
        return [
            AnnexPublicationDelivery(
                intent_id=row.id,
                **{
                    key: getattr(row, key)
                    for key in AnnexPublicationDelivery.model_fields
                    if key != "intent_id"
                },
            )
            for row in list_pending_annex_publication_intents(
                session,
                environment=environment,
                tenant_id=tenant_id,
                database_identity=database_identity,
                limit=limit,
            )
        ]


def record_publication_failure(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    *,
    requires_reconciliation: bool = False,
) -> None:
    with owned_execution(owner, delivery) as (_session, review, _draft):
        review.error_message = (
            "Publication interrupted; the frozen manifest will be retried."
        )
        if requires_reconciliation:
            review.status = "failed"
            review.error_message = "Publication needs manual provider reconciliation; no submission will be repeated."
            for row in _session.scalars(
                select(AnnexPublicationEmbedding).where(
                    AnnexPublicationEmbedding.change_set_id == delivery.change_set_id,
                    AnnexPublicationEmbedding.status == "submitting",
                )
            ):
                row.status = "indeterminate"
        review.heartbeat_at = datetime.now(timezone.utc)


def prepare_batch_reconciliation(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    prepared: AnnexPublicationPreparation,
    projection_id: UUID,
) -> AnnexEmbeddingCheckpoint:
    with owned_execution(owner, delivery, allow_reconciliation=True) as (
        session,
        _review,
        draft,
    ):
        validate_database_baseline(session, draft, prepared)
        manifest = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if (
            manifest is None
            or manifest.es_started
            or manifest.payload != prepared.model_dump(mode="json")
        ):
            raise ValueError(
                "batch reconciliation requires the unchanged pre-ES manifest"
            )
        validate_staged_history(session, draft, prepared, manifest)
        row = session.get(
            AnnexPublicationEmbedding, (delivery.change_set_id, projection_id)
        )
        if (
            row is None
            or row.status not in ("submitting", "indeterminate")
            or row.remote_id is not None
            or publication_digest(row.request) != row.request_sha256
        ):
            raise ValueError("batch submission is not awaiting correlation")
        return AnnexEmbeddingCheckpoint.model_validate(
            dict(
                request=row.request,
                status=row.status,
                remote_id=row.remote_id,
                vectors=row.vectors,
            )
        )


def complete_batch_reconciliation(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    prepared: AnnexPublicationPreparation,
    checkpoint: AnnexEmbeddingCheckpoint,
) -> None:
    from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors

    with owned_execution(owner, delivery, allow_reconciliation=True) as (
        session,
        review,
        draft,
    ):
        validate_database_baseline(session, draft, prepared)
        manifest = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if (
            manifest is None
            or manifest.es_started
            or manifest.payload != prepared.model_dump(mode="json")
        ):
            raise ValueError("batch reconciliation manifest changed")
        validate_staged_history(session, draft, prepared, manifest)
        row = session.get(
            AnnexPublicationEmbedding,
            (delivery.change_set_id, checkpoint.request.projection_id),
        )
        receipt = checkpoint.provider_receipt
        if (
            row is None
            or row.status not in ("submitting", "indeterminate")
            or row.remote_id is not None
            or row.request != checkpoint.request.model_dump(mode="json")
            or receipt is None
            or receipt.request_sha256 != row.request_sha256
            or receipt.custom_id != checkpoint.request.custom_id
            or checkpoint.remote_id != receipt.remote_id
            or checkpoint.status != "complete"
            or checkpoint.vectors is None
        ):
            raise ValueError("batch correlation differs from the frozen request")
        _validate_response_vectors(
            checkpoint.vectors,
            expected_count=len(checkpoint.request.inputs),
            expected_dimension=checkpoint.request.dimension,
        )
        row.status, row.remote_id, row.vectors = (
            checkpoint.status,
            checkpoint.remote_id,
            checkpoint.vectors,
        )
        row.vectors_sha256 = publication_digest(checkpoint.vectors)
        row.provider_receipt = receipt.model_dump(mode="json")
        review.status, review.error_message = "preparing", None
        review.heartbeat_at = datetime.now(timezone.utc)


def load_validated_runtime_settings(
    draft: AnnexChangeDraft, prepared: AnnexPublicationPreparation
) -> list[SearchSettings]:
    from onyx.db.regulatory_annex_publication import (
        load_annex_publication_inputs,
        publication_input_scope_hash,
    )

    with get_session_with_tenant(tenant_id=prepared.scope.tenant_id) as session:
        file, settings, access = load_annex_publication_inputs(session, draft)
        if (
            publication_input_scope_hash(file, settings, access)
            != prepared.input_scope_sha256
        ):
            raise ValueError("publication file/ACL/index configuration changed")
        return settings


def validate_publication_retry(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    tenant_id: str,
    environment: str,
    database_identity: str,
) -> bool:
    """Retain authority locks until the caller commits the current review's intent."""
    from onyx.db.regulatory_annex_changes import require_current_annex_review
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes.publication_execution import (
        runtime_embedders,
    )
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        validate_frozen_publication_review,
    )

    manifest = session.get(AnnexPublicationManifest, change_set_id)
    if manifest is None or not manifest.es_started:
        return False
    prepared = AnnexPublicationPreparation.model_validate(manifest.payload)
    scope = PublicationScope(
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
    )
    if prepared.scope != scope:
        raise ValueError("publication recovery manifest scope mismatch")
    # Authority precedes the batch/review/file/settings locks used by the producer.
    ordinals = PublicationStore(scope).lock_recovery_reservations(
        session, prepared.user_file_id
    )
    review = require_current_annex_review(
        session,
        change_set_id=change_set_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
    )
    if review.status not in ("failed", "approving", "preparing", "publishing"):
        raise ValueError("review state does not allow publication recovery")
    if publication_digest(review.review_payload) != review.review_sha256:
        raise ValueError("immutable review payload changed")
    draft = AnnexChangeDraft.model_validate(review.review_payload)
    current = session.scalar(
        select(AnnexPublicationIntent).where(
            AnnexPublicationIntent.change_set_id == review.id,
            AnnexPublicationIntent.publication_generation
            == review.publication_generation,
        )
    )
    first = session.get(AnnexPublicationIntent, manifest.first_intent_id)
    for intent in (current, first):
        if intent is None or (
            intent.change_set_id,
            intent.logical_group_id,
            intent.review_revision,
            intent.review_sha256,
            intent.tenant_id,
            intent.environment,
            intent.database_identity,
        ) != (
            review.id,
            review.logical_group_id,
            review.review_revision,
            review.review_sha256,
            tenant_id,
            environment,
            database_identity,
        ):
            raise ValueError("publication recovery intent scope mismatch")
    if (
        validate_frozen_publication_review(draft) != prepared
        or manifest.payload_sha256 != publication_digest(manifest.payload)
        or manifest.operations is None
        or manifest.operations_sha256 != publication_digest(manifest.operations)
        or manifest.approved_at is not None
        or list(ordinals) != prepared.reserved_ordinals
    ):
        raise ValueError("publication recovery manifest or reservations changed")
    validate_database_baseline(session, draft, prepared)
    validate_staged_history(session, draft, prepared, manifest)
    runtime_embedders(draft, prepared)
    return True
