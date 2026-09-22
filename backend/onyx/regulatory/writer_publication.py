"""Execute a durable legacy writer inventory with permanent actual ES fencing."""

from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch

from onyx.db.regulatory_publication import PublicationOwnershipLost, PublicationStore
from onyx.db.regulatory_writer_publication import (
    finalize_writer_publication,
    pending_writer_manifest,
    stage_writer_publication,
)
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.regulatory.amendments.annexes.publication_execution import (
    publication_heartbeat,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from onyx.utils.logger import setup_logger

logger = setup_logger()

if TYPE_CHECKING:
    from datetime import date
    from typing import Literal
    from uuid import UUID

    from pydantic import JsonValue

    from onyx.connectors.models import Document
    from onyx.db.regulatory_chunks import RegulatoryFileValidityUpdateResult
    from onyx.db.regulatory_writer_publication import OwnedWriterInputs
    from onyx.document_index.interfaces_new import MetadataUpdateRequest


def execute_writer_publication(
    owner: FileOwnership,
    client: Elasticsearch,
    manifest: WriterPublicationManifest | None = None,
    *,
    hidden: bool = False,
    activate: bool = True,
    durable_generation: int | None = None,
) -> None:
    started = monotonic()
    authority = PublicationStore(owner.scope)
    if manifest is not None:
        pending = pending_writer_manifest(owner)
        if pending is None or (
            manifest.kind == "cancellation" and pending.kind == "durable"
        ):
            manifest = complete_writer_index_inventory(owner, client, manifest)
        stage_writer_publication(owner, manifest, durable_generation=durable_generation)
    manifest = pending_writer_manifest(owner)
    if manifest is None:
        raise ValueError("owned writer has no durable publication to execute")
    proofs = []
    with publication_heartbeat(owner) as lost:
        reservations = authority.reservations(owner)
        for index in manifest.indexes:
            adapter = FencedPublicationIndex(client, index)
            projections = {
                binding.projection.ordinal: binding.projection
                for binding in manifest.bindings
                if binding.index.index_uuid == index.index_uuid
            }
            if hidden:
                import json

                projections = {
                    ordinal: projection.model_copy(
                        update={
                            "source_json": json.dumps(
                                {**json.loads(projection.source_json), "hidden": True}
                            )
                        }
                    )
                    for ordinal, projection in projections.items()
                }

            def check_owner() -> None:
                if lost.is_set():
                    raise PublicationOwnershipLost("writer publication heartbeat lost")
                authority.reservations(owner)

            if manifest.kind in {"amendment", "baseline"}:
                adapter.publish_inventory(
                    reservations, tuple(projections.values()), before_batch=check_owner
                )
            else:
                adapter.seal(reservations)
                for ordinal in reservations.ordinals:
                    check_owner()
                    if ordinal in projections:
                        adapter.upsert(reservations, projections[ordinal])
                    else:
                        adapter.tombstone(reservations, ordinal)
            proofs.append(adapter.verify(reservations, tuple(projections.values())))
        if lost.is_set():
            raise PublicationOwnershipLost("writer publication heartbeat lost")
    # The heartbeat has joined before authority/canonical activation locks.
    indexed = monotonic()
    if activate:
        if manifest.kind in {"durable", "cancellation"}:
            finalize_writer_publication(
                owner, manifest, proofs, durable_generation=durable_generation
            )
        else:
            finalize_writer_publication(owner, manifest, proofs)
    logger.info(
        "writer_publication_finished kind=%s file_id=%s proposal_id=%s index_seconds=%.3f activation_seconds=%.3f bindings=%d",
        manifest.kind,
        owner.user_file_id,
        manifest.amendment_proposal_id,
        indexed - started,
        monotonic() - indexed,
        len(manifest.bindings),
    )


def recover_owned_writer(owner: FileOwnership) -> bool:
    """Resume committed frozen work before accepting another ordinary writer request."""
    from onyx.document_index.elasticsearch.client import ElasticsearchClient

    pending = pending_writer_manifest(owner)
    if pending is None:
        return False
    if pending.kind in {"durable", "cancellation"}:
        raise ValueError("durable publication must resume through its indexing job")
    with ElasticsearchClient() as transport:
        execute_writer_publication(owner, transport.publication_client())
    return True


def recover_owned_writer_before_next(owner: FileOwnership) -> FileOwnership:
    if recover_owned_writer(owner) and pending_writer_manifest(owner) is None:
        return PublicationStore(owner.scope).advance_after_publication(owner)
    return owner


def prepare_owned_metadata(
    owner: FileOwnership,
    client: Elasticsearch,
    request: "MetadataUpdateRequest",
    *,
    index_names: list[str],
) -> WriterPublicationManifest | None:
    """The caller retains its original lease through qualified metadata publication."""
    import json
    from uuid import uuid4

    from onyx.db.regulatory_writer_publication import owned_metadata_baseline
    from onyx.document_index.elasticsearch.elasticsearch_document_index import (
        generate_elasticsearch_filtered_access_control_list,
    )
    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        ObservedPublicationProjection,
        matches_indexed_evidence,
    )

    if request.document_ids != [str(owner.user_file_id)]:
        raise ValueError("metadata request differs from owned file")
    before_digest, before, revisions, index_state = owned_metadata_baseline(
        owner, index_names
    )
    changes: dict[str, object] = {}
    if request.access is not None:
        changes["access_control_list"] = sorted(
            generate_elasticsearch_filtered_access_control_list(request.access)
        )
        changes["public"] = request.access.is_public
    for attribute, field in (
        ("document_sets", "document_sets"),
        ("project_ids", "user_projects"),
        ("persona_ids", "personas"),
    ):
        value = getattr(request, attribute)
        if value is not None:
            changes[field] = sorted(value)
    if request.boost is not None:
        changes["global_boost"] = int(request.boost)
    if request.hidden is not None:
        changes["hidden"] = request.hidden
    if request.created_at is not None:
        changes["created_at"] = int(request.created_at.timestamp())
    if not changes:
        return
    authority = PublicationStore(owner.scope)
    reservations = authority.reservations(owner)
    from onyx.document_index.publication_models import merge_publication_indexes

    indexes = {
        uuid: merge_publication_indexes(
            [binding.index for binding in before if binding.index.index_uuid == uuid]
        )
        for uuid in {binding.index.index_uuid for binding in before}
    }
    for index in indexes.values():
        actuals = FencedPublicationIndex(client, index).inventory_evidence(reservations)
        expected = {
            binding.projection.ordinal: binding
            for binding in before
            if binding.index.index_uuid == index.index_uuid
        }
        if {json.loads(actual.source_json)["chunk_index"] for actual in actuals} != set(
            expected
        ):
            raise ValueError(
                "metadata indexed inventory differs from qualified baseline"
            )
        for actual in actuals:
            previous = expected[json.loads(actual.source_json)["chunk_index"]]
            if not matches_indexed_evidence(previous.projection, actual):
                raise ValueError(
                    "metadata actual source differs from qualified evidence"
                )
    updated = []
    revision_ids = {}
    for previous in before:
        identifier = uuid4()
        source = {**json.loads(previous.projection.source_json), **changes}
        if isinstance(previous.projection, ObservedPublicationProjection):
            projection = ObservedPublicationProjection.model_validate(
                {
                    **previous.projection.model_dump(),
                    "context_projection_id": str(identifier),
                    "source_json": json.dumps(source),
                }
            )
        else:
            projection = FrozenPublicationProjection(
                ordinal=previous.projection.ordinal,
                context_projection_id=str(identifier),
                source_json=json.dumps(source),
                embedding_inputs=previous.projection.embedding_inputs,
                embedding_config_json=previous.projection.embedding_config_json,
            )
        updated.append(
            previous.model_copy(update={"id": identifier, "projection": projection})
        )
        revision_ids[identifier] = revisions[previous.id]
    return WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="metadata",
        index_state_sha256=index_state,
        canonical_before_sha256=before_digest,
        indexes=list(indexes.values()),
        previous_binding_ids=[binding.id for binding in before],
        bindings=updated,
        canonical_revisions=revision_ids,
    )


def publish_owned_metadata(
    owner: FileOwnership,
    client: Elasticsearch,
    request: "MetadataUpdateRequest",
    *,
    index_names: list[str],
) -> None:
    with publication_heartbeat(owner) as lost:
        manifest = prepare_owned_metadata(
            owner, client, request, index_names=index_names
        )
        if lost.is_set():
            raise ValueError("metadata publication ownership heartbeat lost")
    if manifest is not None:
        execute_writer_publication(owner, client, manifest)


def sync_qualified_file_metadata(user_file_id: str, tenant_id: str) -> bool:
    """Synchronize a versioned file from fresh DB metadata under its original lease."""
    from uuid import UUID, uuid4

    from onyx.db.regulatory_writer_publication import (
        finish_owned_metadata_sync,
        owned_metadata_sync_request,
        writer_file_exists,
    )
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    identifier = UUID(user_file_id)
    from onyx.configs.app_configs import DISABLE_VECTOR_DB

    if DISABLE_VECTOR_DB:
        return False
    if not writer_file_exists(identifier, tenant_id):
        return True
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(identifier, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        owner = recover_owned_writer_before_next(owner)
        current = owned_metadata_sync_request(owner)
        if current is None:
            return True
        request, indexes = current.request, current.index_names
        with ElasticsearchClient() as transport:
            if current.cleanup_failed:
                from onyx.db.regulatory_writer_publication import (
                    load_owned_writer_inputs,
                )
                from onyx.document_index.publication_models import (
                    PublicationIndexSnapshot,
                    publication_digest,
                )
                from shared_configs.configs import MULTI_TENANT

                inputs = load_owned_writer_inputs(owner)
                setting = inputs.settings[0]
                physical = transport.publication_client().indices.get(
                    index=setting.index_name
                )
                if set(physical) != {setting.index_name}:
                    raise ValueError("metadata cleanup requires a concrete index")
                index = PublicationIndexSnapshot(
                    index_name=setting.index_name,
                    index_uuid=physical[setting.index_name]["settings"]["index"][
                        "uuid"
                    ],
                    search_settings_id=setting.id,
                    model_provider=setting.provider_type.value
                    if setting.provider_type
                    else "",
                    model_name=setting.model_name,
                    vector_dimension=setting.final_embedding_dim,
                    embedding_config_sha256=publication_digest({"cleanup": setting.id}),
                    multitenant=MULTI_TENANT,
                )
                manifest = WriterPublicationManifest(
                    id=uuid4(),
                    scope=owner.scope,
                    user_file_id=identifier,
                    kind="metadata",
                    index_state_sha256=inputs.index_state_sha256,
                    canonical_before_sha256=publication_digest(
                        [row.model_dump(mode="json") for row in inputs.canonical]
                    ),
                    indexes=[index],
                    previous_binding_ids=[],
                    bindings=[],
                )
                execute_writer_publication(
                    owner, transport.publication_client(), manifest
                )
            elif current.requires_content:
                from onyx.db.regulatory_writer_publication import (
                    load_owned_writer_inputs,
                )
                from onyx.regulatory.writer_projection import prepare_owned_correction

                with publication_heartbeat(owner) as lost:
                    inputs = load_owned_writer_inputs(owner)
                    adopted = not inputs.canonical
                    if adopted:
                        inputs = adopt_owned_original_canonical(owner, inputs)
                        request = request.model_copy(
                            update={
                                "doc_id_to_chunk_cnt": {
                                    str(identifier): len(inputs.canonical)
                                }
                            }
                        )
                    manifest = prepare_owned_correction(
                        owner,
                        transport.publication_client(),
                        inputs,
                        inputs.canonical,
                        changed_id=None,
                    )
                    if adopted:
                        manifest = manifest.model_copy(
                            update={
                                "complete_file": True,
                                "chunk_count_after": len(inputs.canonical),
                            }
                        )
                    if lost.is_set():
                        raise ValueError("metadata reconciliation ownership lost")
                execute_writer_publication(
                    owner, transport.publication_client(), manifest
                )
            else:
                publish_owned_metadata(
                    owner, transport.publication_client(), request, index_names=indexes
                )
        finish_owned_metadata_sync(owner, request)
        return True
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def correct_owned_chunk(
    owner: FileOwnership,
    client: Elasticsearch,
    chunk_id: str,
    *,
    text: str | None = None,
    heading_path: list[str] | None = None,
    chunk_metadata: dict[str, "JsonValue"] | None = None,
    validity_start_date: "date | None | Literal['unset']" = "unset",
    validity_end_date: "date | None | Literal['unset']" = "unset",
) -> FileOwnership:
    from datetime import date

    from onyx.db.regulatory_chunks import update_chunk
    from onyx.db.regulatory_writer_publication import (
        apply_owned_deferred_edit,
        load_owned_writer_inputs,
    )
    from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
    from onyx.regulatory.writer_projection import prepare_owned_correction

    owner = recover_owned_writer_before_next(owner)
    with publication_heartbeat(owner) as lost:
        inputs = load_owned_writer_inputs(owner)
        rows = canonical_snapshot_rows(inputs.canonical)
        target = next((row for row in rows if row.id == chunk_id), None)
        if target is None:
            raise ValueError("correction target is outside the owned canonical file")
        if target.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate":
            raise ValueError("derived aggregates cannot be edited directly")
        if text is not None and not text.strip():
            raise ValueError("chunk text cannot be emptied")
        update_chunk(
            target,
            text=text,
            heading_path=heading_path,
            chunk_metadata=chunk_metadata,
            validity_start_date=validity_start_date,
            validity_end_date=validity_end_date,
        )
        if (target.validity_start_date or date.min) >= (
            target.validity_end_date or date.max
        ):
            raise ValueError("correction validity window is empty")
        after = [_snapshot(row) for row in rows]
        if apply_owned_deferred_edit(
            owner,
            canonical_before_sha256=publication_digest(
                [row.model_dump(mode="json") for row in inputs.canonical]
            ),
            canonical_after=after,
        ):
            return owner
        manifest = prepare_owned_correction(
            owner, client, inputs, after, changed_id=chunk_id
        )
        if lost.is_set():
            raise ValueError("correction publication ownership heartbeat lost")
    execute_writer_publication(owner, client, manifest)

    return owner


def republish_user_file(
    user_file_id: "UUID",
    tenant_id: str,
    *,
    include_chunked: bool = False,
    include_failed: bool = False,
    current_search_settings_id: int | None = None,
    target_search_settings_id: int | None = None,
    adopt_original: bool = False,
    before_stage: Callable[[], bool] | None = None,
    index_request_attempt_id: "UUID | None" = None,
) -> int:
    from uuid import uuid4

    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.writer_projection import prepare_owned_correction

    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        if index_request_attempt_id is not None:
            from onyx.db.user_file import validate_user_file_index_request

            validate_user_file_index_request(owner, index_request_attempt_id)
        owner = recover_owned_writer_before_next(owner)
        with publication_heartbeat(owner) as lost:
            inputs = load_owned_writer_inputs(owner)
            allowed = {UserFileStatus.COMPLETED}
            if include_chunked:
                allowed.add(UserFileStatus.CHUNKED)
            if include_failed:
                allowed.add(UserFileStatus.FAILED)
            if inputs.file.status not in allowed:
                return 0
            if adopt_original and not inputs.canonical:
                inputs = adopt_owned_original_canonical(owner, inputs)
            if not inputs.canonical:
                return 0
            future_pending = False
            targets = (
                {target_search_settings_id}
                if target_search_settings_id is not None
                else None
            )
            if targets is not None and not targets.issubset(
                {item.id for item in inputs.settings}
            ):
                raise ValueError("target search settings is no longer active")
            if targets is not None:
                for setting in inputs.settings:
                    if setting.status.is_current() and not any(
                        binding.index.index_name == setting.index_name
                        for binding in inputs.bindings
                    ):
                        targets.add(setting.id)
            if current_search_settings_id is not None:
                current = [
                    settings
                    for settings in inputs.settings
                    if settings.status.is_current()
                ]
                if len(current) != 1 or current[0].id != current_search_settings_id:
                    raise ValueError("current search settings changed after validation")
                future_pending = any(
                    settings.status.is_future() for settings in inputs.settings
                )
                targets = {current_search_settings_id}
            with ElasticsearchClient() as transport:
                manifest = prepare_owned_correction(
                    owner,
                    transport.publication_client(),
                    inputs,
                    inputs.canonical,
                    changed_id=None,
                    target_settings_ids=targets,
                )
            manifest = manifest.model_copy(
                update={
                    "complete_file": True,
                    "chunk_count_after": len(inputs.canonical),
                    "secondary_reconcile_pending": future_pending,
                }
            )
            if lost.is_set():
                raise PublicationOwnershipLost(
                    "reindex publication ownership heartbeat lost"
                )
            if before_stage is not None and not before_stage():
                return 0
        with ElasticsearchClient() as transport:
            execute_writer_publication(owner, transport.publication_client(), manifest)
        return len(inputs.canonical)
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def rename_owned_file(user_file_id: "UUID", tenant_id: str, name: str) -> None:
    from uuid import uuid4

    from onyx.db.regulatory_writer_publication import (
        apply_owned_deferred_edit,
        load_owned_writer_inputs,
        rename_owned_unprojected_file,
    )
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.writer_projection import prepare_owned_correction

    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        owner = recover_owned_writer_before_next(owner)
        inputs = load_owned_writer_inputs(owner)
        if not inputs.canonical:
            rename_owned_unprojected_file(owner, name)
            return
        if apply_owned_deferred_edit(
            owner,
            canonical_before_sha256=publication_digest(
                [row.model_dump(mode="json") for row in inputs.canonical]
            ),
            name=name,
        ):
            return
        with publication_heartbeat(owner) as lost:
            inputs.file.name = name
            with ElasticsearchClient() as transport:
                manifest = prepare_owned_correction(
                    owner,
                    transport.publication_client(),
                    inputs,
                    inputs.canonical,
                    changed_id=None,
                ).model_copy(update={"name_after": name})
            if lost.is_set():
                raise ValueError("rename publication ownership heartbeat lost")
        with ElasticsearchClient() as transport:
            execute_writer_publication(owner, transport.publication_client(), manifest)
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def chunk_owned_file(
    user_file_id: "UUID", tenant_id: str, documents: list["Document"]
) -> bool:
    """Write first canonical rows; an existing revision always remains authoritative."""
    from uuid import uuid4

    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_writer_publication import (
        load_owned_writer_inputs,
        persist_owned_initial_chunks,
    )
    from onyx.document_index.publication_models import PublicationScope
    from onyx.indexing.contextual_settings import effective_contextual_rag_enabled
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.indexing_jobs.configuration import (
        compute_regulatory_chunk_generation_hash,
    )

    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        owner = recover_owned_writer_before_next(owner)
        inputs = load_owned_writer_inputs(owner)
        if inputs.file.status == UserFileStatus.DELETING:
            return False
        if inputs.canonical:
            return False
        current = next(
            (item for item in inputs.settings if item.status.is_current()), None
        )
        if current is None:
            raise ValueError("initial chunking requires current search settings")
        embedder = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=current
        )
        contextual = effective_contextual_rag_enabled(current)
        generation_hash = compute_regulatory_chunk_generation_hash(
            embedding_provider=current.provider_type,
            embedding_model_name=current.model_name,
            enable_contextual_rag=contextual,
        )
        persist_owned_initial_chunks(
            owner,
            documents,
            embedder.embedding_model.tokenizer,
            contextual=contextual,
            generation_hash=generation_hash,
        )
        return True
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def delete_owned_file(user_file_id: "UUID", tenant_id: str) -> None:
    from uuid import uuid4

    from onyx.background.celery.apps.client import celery_app
    from onyx.background.celery.tasks.regulatory_indexing.tasks import (
        enqueue_regulatory_indexing_step,
    )
    from onyx.configs.app_configs import DISABLE_VECTOR_DB
    from onyx.db.regulatory_writer_publication import (
        begin_owned_deletion,
        finish_owned_deletion,
        load_owned_writer_inputs,
        owned_deletion_file_id,
        owned_unindexed_deletion_file_id,
        writer_file_exists,
    )
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import (
        PublicationIndexSnapshot,
        PublicationScope,
        publication_digest,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.file_store.utils import user_file_id_to_plaintext_file_name
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.indexing_jobs.orchestrator import OrchestrationDeliveryKind
    from shared_configs.configs import MULTI_TENANT

    if not writer_file_exists(user_file_id, tenant_id):
        return
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        pending = pending_writer_manifest(owner)
        if pending is None or pending.kind not in {"durable", "cancellation"}:
            owner = recover_owned_writer_before_next(owner)
        plan = begin_owned_deletion(owner)
        if not plan.ready_to_delete:
            for delivery in plan.deliveries:
                enqueue_regulatory_indexing_step(
                    celery_app,
                    job_id=delivery.job_id,
                    expected_generation=delivery.expected_generation,
                    tenant_id=tenant_id,
                    delivery_kind=OrchestrationDeliveryKind.NORMAL,
                )
            return
        unindexed_file_id = (
            owned_unindexed_deletion_file_id(owner) if DISABLE_VECTOR_DB else None
        )
        if unindexed_file_id is None and pending_writer_manifest(owner) is None:
            inputs = load_owned_writer_inputs(owner)
            indexes = {
                binding.index.index_uuid: binding.index for binding in inputs.bindings
            }
            with ElasticsearchClient() as transport:
                client = transport.publication_client()
                for settings in inputs.settings:
                    info = client.indices.get(index=settings.index_name)
                    if set(info) != {settings.index_name}:
                        raise ValueError("deletion requires a concrete index")
                    index_uuid = info[settings.index_name]["settings"]["index"]["uuid"]
                    indexes.setdefault(
                        index_uuid,
                        PublicationIndexSnapshot(
                            index_name=settings.index_name,
                            index_uuid=index_uuid,
                            search_settings_id=settings.id,
                            model_provider=settings.provider_type.value
                            if settings.provider_type
                            else "",
                            model_name=settings.model_name,
                            vector_dimension=settings.final_embedding_dim,
                            embedding_config_sha256=publication_digest(
                                {"deletion": settings.id}
                            ),
                            multitenant=MULTI_TENANT,
                        ),
                    )
                manifest = WriterPublicationManifest(
                    id=uuid4(),
                    scope=owner.scope,
                    user_file_id=user_file_id,
                    kind="delete",
                    index_state_sha256=inputs.index_state_sha256,
                    canonical_before_sha256=publication_digest(
                        [item.model_dump(mode="json") for item in inputs.canonical]
                    ),
                    indexes=list(indexes.values()),
                    previous_binding_ids=[item.id for item in inputs.bindings],
                    bindings=[],
                )
                execute_writer_publication(owner, client, manifest)
        file_id = (
            unindexed_file_id
            if unindexed_file_id is not None
            else owned_deletion_file_id(owner)
        )
        with publication_heartbeat(owner) as lost:
            file_store = get_default_file_store()
            file_store.delete_file(file_id, error_on_missing=False)
            file_store.delete_file(
                user_file_id_to_plaintext_file_name(user_file_id),
                error_on_missing=False,
            )
            if lost.is_set():
                raise ValueError("file deletion ownership heartbeat lost")
        finish_owned_deletion(
            owner, without_index_authority=unindexed_file_id is not None
        )
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


class FileValidityPublicationConflict(ValueError):
    pass


def update_owned_file_validity(
    user_file_id: "UUID",
    tenant_id: str,
    *,
    validity_start_date: "date | None | Literal['unset']" = "unset",
    validity_end_date: "date | None | Literal['unset']" = "unset",
) -> "RegulatoryFileValidityUpdateResult":
    from uuid import uuid4

    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_chunks import apply_file_validity_window
    from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
    from onyx.regulatory.writer_projection import prepare_owned_correction

    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        owner = recover_owned_writer_before_next(owner)
        inputs = load_owned_writer_inputs(owner)
        if inputs.file.status != UserFileStatus.COMPLETED:
            raise ValueError("Validity can only be updated for a completed file.")
        rows = canonical_snapshot_rows(inputs.canonical)
        result = apply_file_validity_window(
            rows,
            validity_start_date=validity_start_date,
            validity_end_date=validity_end_date,
        )
        if not result.updated_chunk_count:
            raise ValueError("The file has no unversioned indexed chunks to update.")
        if (
            result.skipped_versioned_chunk_count
            or result.previous_window is None
            or result.updated_window is None
        ):
            raise FileValidityPublicationConflict(
                "Versioned files require explicit per-version validity corrections."
            )
        with publication_heartbeat(owner) as lost:
            with ElasticsearchClient() as transport:
                manifest = prepare_owned_correction(
                    owner,
                    transport.publication_client(),
                    inputs,
                    [_snapshot(row) for row in rows],
                    changed_id=None,
                ).model_copy(update={"kind": "validity"})
            if lost.is_set():
                raise ValueError("validity publication ownership heartbeat lost")
        with ElasticsearchClient() as transport:
            execute_writer_publication(owner, transport.publication_client(), manifest)
        return result
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def approve_owned_amendment(
    proposal_id: int, tenant_id: str, current_search_settings_id: int
) -> int:
    from uuid import uuid4

    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_amendment_impact import resolve_context_impact_audit
    from onyx.db.regulatory_writer_publication import (
        amendment_writer_target,
        load_owned_writer_inputs,
        preview_owned_amendment,
        record_amendment_execution_stage,
        record_owned_amendment_failure,
    )
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
    from onyx.regulatory.writer_projection import prepare_owned_correction

    started = monotonic()
    target = amendment_writer_target(proposal_id, tenant_id)
    if target is None:
        return 0
    file_id, canonical_ids = target
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        owner = recover_owned_writer_before_next(owner)
        if amendment_writer_target(proposal_id, tenant_id) is None:
            return 0
        record_amendment_execution_stage(owner, proposal_id, "baseline")
        from onyx.regulatory.publication_baseline import ensure_owned_baseline

        with ElasticsearchClient() as transport:
            owner = ensure_owned_baseline(
                owner, transport.publication_client(), origin_proposal_id=proposal_id
            )
        inputs = load_owned_writer_inputs(owner)
        if inputs.file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}:
            raise ValueError("amendment file is canceled or deleting")
        if [item.id for item in inputs.settings if item.status.is_current()] != [
            current_search_settings_id
        ]:
            raise ValueError("current amendment search settings changed")
        for canonical_id in canonical_ids:
            authority.allocate(owner, "canonical:" + canonical_id)
        record_amendment_execution_stage(owner, proposal_id, "review")
        after, reviewed = preview_owned_amendment(owner, proposal_id)
        existing_ids = {row.id for row in inputs.canonical}
        # Preview rolls back derived reservations; keep stable identities durably.
        after = [
            row.model_copy(
                update={
                    "projection_ordinal": authority.allocate(
                        owner, "canonical:" + row.id
                    )
                }
            )
            if row.id not in existing_ids
            else row
            for row in after
        ]
        record_amendment_execution_stage(owner, proposal_id, "context")
        with publication_heartbeat(owner) as lost:
            with ElasticsearchClient() as transport:
                manifest = prepare_owned_correction(
                    owner,
                    transport.publication_client(),
                    inputs,
                    after,
                    changed_id=None,
                    target_settings_ids={current_search_settings_id},
                    selective_amendment=True,
                    audit_cache=lambda key, generate: resolve_context_impact_audit(
                        owner, key, generate
                    ),
                ).model_copy(
                    update={
                        "kind": "amendment",
                        "amendment_proposal_id": proposal_id,
                        "amendment_review_sha256": reviewed,
                        "complete_file": True,
                        "chunk_count_after": len(after),
                        "secondary_reconcile_pending": any(
                            item.status.is_future() for item in inputs.settings
                        ),
                    }
                )
            if lost.is_set():
                raise ValueError("amendment publication ownership heartbeat lost")
        previous_ids = {binding.id for binding in inputs.bindings}
        logger.info(
            "amendment_prepared proposal_id=%s seconds=%.3f retained_bindings=%d new_bindings=%d generated_contexts=%d",
            proposal_id,
            monotonic() - started,
            sum(binding.id in previous_ids for binding in manifest.bindings),
            sum(binding.id not in previous_ids for binding in manifest.bindings),
            sum(len(view.projections) for view in manifest.views),
        )
        record_amendment_execution_stage(owner, proposal_id, "publication")
        with ElasticsearchClient() as transport:
            execute_writer_publication(owner, transport.publication_client(), manifest)
        logger.info(
            "amendment_approved proposal_id=%s total_seconds=%.3f",
            proposal_id,
            monotonic() - started,
        )
        return len(after)
    except Exception as error:
        record_owned_amendment_failure(proposal_id, tenant_id, error, owner=owner)
        raise
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def complete_writer_index_inventory(
    owner: FileOwnership,
    client: Elasticsearch,
    manifest: WriterPublicationManifest,
) -> WriterPublicationManifest:
    """Every takeover seals the file in every active concrete physical index."""
    import json

    from onyx.db.regulatory_writer_publication import owned_writer_index_inventory

    inventory = owned_writer_index_inventory(
        owner, [index.search_settings_id for index in manifest.indexes]
    )
    if (
        manifest.index_state_sha256 is not None
        and manifest.index_state_sha256 != inventory.index_state_sha256
    ):
        raise ValueError("writer index settings changed before staging")
    indexes = {index.index_uuid: index for index in manifest.indexes}
    target_uuids = set(indexes)
    active_names = {setting.index_name for setting in inventory.settings}
    bindings = list(manifest.bindings)
    previous_ids = set(manifest.previous_binding_ids)
    revisions = dict(manifest.canonical_revisions)
    for binding in inventory.bindings:
        if (
            binding.index.index_uuid in target_uuids
            or binding.index.index_name not in active_names
        ):
            continue
        indexes.setdefault(binding.index.index_uuid, binding.index)
        previous_ids.add(binding.id)
        if manifest.kind != "delete":
            bindings.append(binding)
            revisions[binding.id] = inventory.revisions[binding.id]
    for setting in inventory.settings:
        if not client.indices.exists(index=setting.index_name):
            if any(
                index.index_name == setting.index_name for index in indexes.values()
            ):
                raise ValueError("writer's active physical index disappeared")
            continue
        physical = client.indices.get(index=setting.index_name)
        if set(physical) != {setting.index_name}:
            raise ValueError("writer requires a concrete active physical index")
        index_uuid = physical[setting.index_name]["settings"]["index"]["uuid"]
        if any(
            index.index_name == setting.index_name and index.index_uuid != index_uuid
            for index in indexes.values()
        ):
            raise ValueError("writer's active physical index was replaced")
        from onyx.regulatory.publication_baseline import observed_index_snapshot

        indexes.setdefault(index_uuid, observed_index_snapshot(setting, index_uuid))
    authority = PublicationStore(owner.scope)
    for index_uuid, index in list(indexes.items()):
        existing = FencedPublicationIndex(client, index).existing_ordinals(
            authority.reservations(owner)
        )
        authority.reserve_existing_ordinals(owner, existing)
        if index_uuid not in target_uuids and manifest.kind != "delete":
            from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
            from onyx.regulatory.publication_baseline import observed_baseline_binding

            expected_ordinals = {
                binding.projection.ordinal
                for binding in bindings
                if binding.index.index_uuid == index_uuid
            }
            actuals = FencedPublicationIndex(client, index).inventory_evidence(
                authority.reservations(owner)
            )
            missing = [
                item
                for item in actuals
                if json.loads(item.source_json)["chunk_index"] not in expected_ordinals
            ]
            if missing:
                inputs = load_owned_writer_inputs(owner)
                for actual in missing:
                    if (
                        actual.frozen_projection is not None
                        or actual.observed_projection is not None
                    ):
                        raise ValueError("secondary index lost its temporal binding")
                    binding = observed_baseline_binding(actual, inputs.canonical)
                    bindings.append(binding)
                    canonical_id = str(
                        json.loads(actual.source_json)["regulatory_chunk_id"]
                    )
                    revisions[binding.id] = inputs.canonical_revisions[canonical_id]
        from onyx.document_index.publication_models import merge_publication_indexes

        indexes[index_uuid] = merge_publication_indexes(
            [
                index,
                *(
                    binding.index
                    for binding in bindings
                    if binding.index.index_uuid == index_uuid
                ),
            ]
        )
    return WriterPublicationManifest.model_validate(
        manifest.model_copy(
            update={
                "indexes": list(indexes.values()),
                "bindings": bindings,
                "previous_binding_ids": sorted(previous_ids, key=str),
                "canonical_revisions": revisions,
                "index_state_sha256": inventory.index_state_sha256,
            }
        ).model_dump(mode="json")
    )


def adopt_owned_original_canonical(
    owner: FileOwnership,
    inputs: "OwnedWriterInputs",
) -> "OwnedWriterInputs":
    """Use the existing complete original-file loader only when canonical rows are absent."""
    from onyx.db.regulatory_writer_publication import (
        load_owned_writer_inputs,
        persist_owned_initial_chunks,
    )
    from onyx.file_processing.user_file_loader import load_user_file_documents
    from onyx.file_store.staging import delete_files_best_effort
    from onyx.indexing.contextual_settings import effective_contextual_rag_enabled
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.indexing_jobs.configuration import (
        compute_regulatory_chunk_generation_hash,
    )

    if inputs.canonical:
        return inputs
    setting = next(item for item in inputs.settings if item.status.is_current())
    embedder = DefaultIndexingEmbedder.from_db_search_settings(search_settings=setting)
    contextual = effective_contextual_rag_enabled(setting)
    documents, staged = load_user_file_documents(
        user_file_id=str(owner.user_file_id),
        file_id=inputs.file.file_id,
        file_name=inputs.file.name,
        tenant_id=owner.scope.tenant_id,
    )
    try:
        persist_owned_initial_chunks(
            owner,
            documents,
            embedder.embedding_model.tokenizer,
            contextual=contextual,
            generation_hash=compute_regulatory_chunk_generation_hash(
                embedding_provider=setting.provider_type,
                embedding_model_name=setting.model_name,
                enable_contextual_rag=contextual,
            ),
        )
    finally:
        delete_files_best_effort(staged, context="owned legacy user-file supply")
    return load_owned_writer_inputs(owner)


def request_owned_file_deletion(user_file_id: "UUID", tenant_id: str) -> None:
    from uuid import uuid4

    from onyx.db.regulatory_writer_publication import begin_owned_deletion
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        begin_owned_deletion(owner)
    finally:
        authority.release(owner)
