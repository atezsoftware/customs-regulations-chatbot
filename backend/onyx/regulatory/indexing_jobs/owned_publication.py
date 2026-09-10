"""Fenced durable index checkpoints consume only persisted, proven batch results."""

import json
from datetime import timedelta
from typing import cast
from uuid import UUID, uuid4, uuid5

from elasticsearch import Elasticsearch

from onyx.db.enums import RegulatoryIndexingStage
from onyx.db.regulatory_durable_publication import (
    durable_publication_input_digest,
    load_owned_durable_runtime,
)
from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingRuntime
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import (
    OwnedWriterInputs,
    load_owned_writer_inputs,
    pending_writer_manifest,
)
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
    PublicationScope,
    publication_digest,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt
from onyx.regulatory.amendments.annexes.publication_representations import (
    _epoch,
    _source_template,
)
from onyx.regulatory.indexing_jobs.embedding import (
    _ordered_mapping,
    _validate_search_settings,
)
from onyx.regulatory.indexing_jobs.embedding_receipts import (
    batch_embedding_configuration,
    has_proven_vector,
    item_embedding_receipt,
)
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from onyx.regulatory.indexing_jobs.projection_identity import (
    projection_input,
    projection_ordinal,
)
from onyx.regulatory.indexing_jobs.projection_preparation import (
    freeze_durable_item_context,
)
from onyx.regulatory.writer_publication import execute_writer_publication
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from shared_configs.configs import MULTI_TENANT


def prepare_durable_writer_manifest(
    owner: FileOwnership,
    client: Elasticsearch,
    inputs: OwnedWriterInputs,
    runtime: RegulatoryIndexingRuntime,
) -> WriterPublicationManifest:
    from onyx.llm.constants import LlmProviderNames
    from onyx.regulatory.indexing_jobs.preparation import (
        get_contextual_token_budget_tokenizer,
        get_tokenizer,
    )

    job = runtime.job
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    settings = runtime.search_settings
    if settings is None:
        raise ValueError("durable publication index configuration disappeared")
    # Batch transport has its own explicit native encoder receipt.
    from onyx.regulatory.indexing_jobs.publisher import (
        _validate_search_settings as validate_target,
    )

    validate_target(settings, snapshot)
    if settings.id not in {value.id for value in inputs.settings}:
        raise ValueError("durable publication target is no longer an active index")
    embedding_tokenizer = get_tokenizer(
        snapshot.embedding_model_name, snapshot.embedding_provider
    )
    contextual_tokenizer = get_contextual_token_budget_tokenizer(
        model_provider=LlmProviderNames.VERTEX_AI, model_name=snapshot.vertex.model_name
    )
    if snapshot.openrouter_batch is not None:
        configuration = batch_embedding_configuration(
            provider=snapshot.embedding_provider.value, config=snapshot.openrouter_batch
        )
    else:
        from onyx.indexing.embedder import DefaultIndexingEmbedder
        from onyx.regulatory.indexing_jobs.embedding_receipts import (
            synchronous_embedding_receipts,
        )

        _validate_search_settings(settings, snapshot)
        embedder = DefaultIndexingEmbedder(
            model_name=snapshot.embedding_model_name,
            normalize=settings.normalize,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
            provider_type=snapshot.embedding_provider,
            api_key=settings.api_key,
            api_url=settings.api_url,
            api_version=settings.api_version,
            deployment_name=settings.deployment_name,
            reduced_dimension=snapshot.effective_dimension,
        )
        embedding_tokenizer = embedder.embedding_model.tokenizer
        expected_receipts = synchronous_embedding_receipts(
            job=job,
            rows=runtime.regulatory_chunks,
            items=runtime.indexing_items,
            model=embedder.embedding_model,
        )
        configuration = next(iter(expected_receipts.values())).configuration
    receipt = _encoder_receipt(configuration, resolution=context_hash(configuration))
    physical = client.indices.get(index=settings.index_name)
    if set(physical) != {settings.index_name}:
        raise ValueError("durable publication requires a concrete physical index")
    index_uuid = physical[settings.index_name]["settings"]["index"]["uuid"]
    prior = [
        binding for binding in inputs.bindings if binding.index.index_uuid == index_uuid
    ]
    if any(
        binding.index.index_name == settings.index_name
        and binding.index.index_uuid != index_uuid
        for binding in inputs.bindings
    ):
        raise ValueError("durable publication physical index was replaced")
    accepted = {context_hash(configuration): receipt}
    for binding in prior:
        for previous in binding.index.encoder_receipts:
            if previous.effective_authority() != receipt.effective_authority():
                raise ValueError("durable encoder change requires a new physical index")
            accepted[context_hash(json.loads(previous.configuration_json))] = previous
    index = PublicationIndexSnapshot(
        index_name=settings.index_name,
        index_uuid=index_uuid,
        search_settings_id=settings.id,
        model_provider=receipt.authority.provider or "",
        model_name=receipt.authority.model,
        vector_dimension=snapshot.effective_dimension,
        embedding_config_sha256=context_hash(configuration),
        multitenant=MULTI_TENANT,
        encoder_authority=receipt.authority,
        encoder_receipts=tuple(accepted.values()),
    )
    bindings, views, revisions = [], [], {}
    for offset, (row, item) in enumerate(
        _ordered_mapping(job, runtime.regulatory_chunks, runtime.indexing_items)
    ):
        frozen = projection_input(item)
        encoder = item_embedding_receipt(item)
        if (
            encoder is None
            or not has_proven_vector(item, encoder)
            or encoder.configuration != configuration
        ):
            raise ValueError("durable vector requires an evidenced encoder request")
        view = freeze_durable_item_context(
            job=job,
            rows=list(runtime.regulatory_chunks),
            row=row,
            item=item,
            embedding_tokenizer=embedding_tokenizer,
            contextual_tokenizer=contextual_tokenizer,
            receipt=encoder,
        )
        views.append(view)
        context = view.projections[0]
        ordinal = projection_ordinal(row, item, offset)
        identity = item.projection_id or uuid5(job.id, str(item.id))
        lower = item.effective_start if frozen else row.validity_start_date
        upper = item.effective_end if frozen else row.validity_end_date
        position, text, metadata = row.position, row.text, dict(row.chunk_metadata)
        row.projection_ordinal, row.validity_start_date, row.validity_end_date = (
            ordinal,
            lower,
            upper,
        )
        source = json.loads(
            _source_template(
                row,
                context,
                file_name=inputs.file.name,
                access=inputs.access,
                tenant_id=owner.scope.tenant_id,
                dimension=index.vector_dimension,
            )
        )
        source.update(
            content_vector=item.vector,
            validity_start_date=_epoch(lower),
            validity_end_date=_epoch(upper),
        )
        projection = FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id=str(identity),
            source_json=json.dumps(source),
            embedding_inputs=tuple(encoder.texts),
            embedding_config_json=json.dumps(configuration),
        )
        role = (
            "hierarchical_aggregate"
            if metadata.get("chunk_variant") == "hierarchical_aggregate"
            else "image_companion"
            if metadata.get("bound_to_regulatory_chunk_id")
            else "canonical"
        )
        binding = AnnexTemporalProjection(
            id=identity,
            index=index,
            projection=projection,
            canonical_base_sha256=frozen.canonical_base_sha256
            if frozen and frozen.canonical_base_sha256
            else context_hash(text),
            derived_role=role,
            dependency_ids=context.canonical_dependency_ids,
            representation_text=text,
            representation_metadata=metadata,
            context=context,
            reference_date=frozen.reference_date if frozen else None,
            effective_start=lower,
            effective_end=upper,
            semantic_position=position,
        )
        bindings.append(binding)
        revisions[identity] = (
            frozen.canonical_revision_id
            if frozen and frozen.canonical_revision_id
            else inputs.canonical_revisions[row.id]
        )
    return WriterPublicationManifest(
        id=uuid5(job.id, "owned-durable-publication"),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="durable",
        index_state_sha256=inputs.index_state_sha256,
        canonical_before_sha256=publication_digest(
            [row.model_dump(mode="json") for row in inputs.canonical]
        ),
        durable_job_id=job.id,
        durable_input_sha256=durable_publication_input_digest(runtime),
        indexes=[index],
        previous_binding_ids=[binding.id for binding in prior],
        bindings=bindings,
        canonical_revisions=revisions,
        views=views,
        complete_file=True,
        chunk_count_after=len(runtime.regulatory_chunks),
        secondary_reconcile_pending=True,
    )


def execute_owned_durable_stage(
    *,
    job_id: UUID,
    file_id: UUID,
    generation: int,
    stage: RegulatoryIndexingStage,
    tenant_id: str,
    caller_row_ids: set[str],
    caller_item_ids: set[UUID],
) -> RegulatoryIndexingRuntime:
    scope = PublicationScope(
        tenant_id=tenant_id,
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=2))
    try:
        runtime = load_owned_durable_runtime(
            owner, job_id=job_id, expected_stage=stage, expected_generation=generation
        )
        if caller_row_ids != {
            row.id for row in runtime.regulatory_chunks
        } or caller_item_ids != {item.id for item in runtime.indexing_items}:
            raise ValueError(
                "durable publication caller items differ from the current job"
            )
        pending = pending_writer_manifest(owner)
        with ElasticsearchClient() as transport:
            client = transport.publication_client()
            manifest = pending
            if manifest is None:
                if stage != RegulatoryIndexingStage.INDEX_WRITE:
                    raise ValueError("durable publication has no staged inventory")
                inputs = load_owned_writer_inputs(owner)
                manifest = prepare_durable_writer_manifest(
                    owner, client, inputs, runtime
                )
            if (
                manifest.durable_job_id != job_id
                or manifest.durable_input_sha256
                != durable_publication_input_digest(runtime)
            ):
                raise ValueError(
                    "durable publication checkpoint differs from staged inventory"
                )
            execute_writer_publication(
                owner,
                client,
                manifest if pending is None else None,
                hidden=stage != RegulatoryIndexingStage.PUBLISH,
                activate=stage == RegulatoryIndexingStage.PUBLISH,
                durable_generation=generation,
            )
        return runtime
    finally:
        authority.release(owner)


def execute_owned_cancellation(
    *, job_id: UUID, user_file_id: UUID, expected_generation: int, tenant_id: str
) -> None:
    from uuid import uuid4

    from onyx.db.regulatory_durable_publication import prepare_cancellation_manifest
    from onyx.document_index.publication_models import publication_digest
    from onyx.regulatory.amendments.annexes.publication_execution import (
        LEASE_TTL,
        publication_heartbeat,
    )
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

    scope = PublicationScope(
        tenant_id=tenant_id,
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        with publication_heartbeat(owner) as lost:
            inputs = prepare_cancellation_manifest(
                owner, job_id=job_id, expected_generation=expected_generation
            )
            indexes = {
                binding.index.index_uuid: binding.index for binding in inputs.bindings
            }
            if inputs.pending is not None:
                for index in inputs.pending.indexes:
                    indexes.setdefault(index.index_uuid, index)
            with ElasticsearchClient() as transport:
                client = transport.publication_client()
                for setting in inputs.settings:
                    if not client.indices.exists(index=setting.index_name):
                        if any(
                            index.index_name == setting.index_name
                            for index in indexes.values()
                        ):
                            raise ValueError(
                                "cancellation historical physical index disappeared"
                            )
                        continue
                    info = client.indices.get(index=setting.index_name)
                    if set(info) != {setting.index_name}:
                        raise ValueError(
                            "cancellation requires a concrete physical index"
                        )
                    index_uuid = info[setting.index_name]["settings"]["index"]["uuid"]
                    if any(
                        index.index_name == setting.index_name
                        and index.index_uuid != index_uuid
                        for index in indexes.values()
                    ):
                        raise ValueError("cancellation physical index was replaced")
                    indexes.setdefault(
                        index_uuid,
                        PublicationIndexSnapshot(
                            index_name=setting.index_name,
                            index_uuid=index_uuid,
                            search_settings_id=setting.id,
                            model_provider=setting.provider_type.value
                            if setting.provider_type
                            else "",
                            model_name=setting.model_name,
                            vector_dimension=setting.final_embedding_dim,
                            embedding_config_sha256=publication_digest(
                                {"cancellation": setting.id}
                            ),
                            multitenant=MULTI_TENANT,
                        ),
                    )
            manifest = (
                inputs.pending
                if inputs.pending is not None and inputs.pending.kind == "cancellation"
                else WriterPublicationManifest(
                    id=uuid4(),
                    scope=owner.scope,
                    user_file_id=user_file_id,
                    kind="cancellation",
                    index_state_sha256=inputs.index_state_sha256,
                    cancellation_job_id=job_id,
                    cancelled_manifest_sha256=publication_digest(
                        inputs.pending.model_dump(mode="json")
                    )
                    if inputs.pending
                    else None,
                    canonical_before_sha256=inputs.canonical_before_sha256,
                    indexes=list(indexes.values()),
                    previous_binding_ids=[binding.id for binding in inputs.bindings],
                    bindings=inputs.bindings,
                    canonical_revisions=inputs.canonical_revisions,
                )
            )
            if lost.is_set():
                raise ValueError("cancellation publication heartbeat lost")
        with ElasticsearchClient() as transport:
            execute_writer_publication(
                owner,
                transport.publication_client(),
                manifest,
                durable_generation=expected_generation,
            )
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def repair_owned_durable_items(
    *,
    job_id: UUID,
    user_file_id: UUID,
    expected_generation: int,
    stage: RegulatoryIndexingStage,
    tenant_id: str,
) -> bool:
    """Upgrade legacy checkpoints and requeue only requests lacking retained proof."""
    from onyx.db.regulatory_durable_publication import (
        durable_context_reference_dates,
        repair_durable_item_checkpoints,
    )
    from onyx.llm.constants import LlmProviderNames
    from onyx.regulatory.amendments.annexes.publication_execution import (
        LEASE_TTL,
        publication_heartbeat,
    )
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
    from onyx.regulatory.indexing_jobs import preparation
    from onyx.regulatory.indexing_jobs.models import (
        IndexingPublicationIndeterminateError,
        RegulatoryInputHashVersion,
    )
    from onyx.regulatory.indexing_jobs.projection_preparation import (
        prepare_owned_durable_items,
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
        pending = pending_writer_manifest(owner)
        if pending is not None:
            if pending.kind == "durable" and pending.durable_job_id == job_id:
                return False
            raise IndexingPublicationIndeterminateError(
                "durable recovery must wait for the other owned publication"
            )
        if authority.reservations(owner).gate_closed:
            raise IndexingPublicationIndeterminateError(
                "durable recovery must wait for pending annex publication"
            )
        runtime = load_owned_durable_runtime(
            owner,
            job_id=job_id,
            expected_stage=stage,
            expected_generation=expected_generation,
        )
        if runtime.job.openrouter_submission_state != "NONE":
            if stage is RegulatoryIndexingStage.EMBEDDING:
                return False
            raise IndexingPublicationIndeterminateError(
                "durable embedding submission must be reconciled before projection recovery"
            )
        if runtime.job.vertex_submission_state not in {
            "NONE",
            "SUBMITTED",
            "RETRY_CLEANUP_REQUIRED",
        }:
            raise IndexingPublicationIndeterminateError(
                "durable contextual submission must be reconciled before projection recovery"
            )
        snapshot = RegulatoryIndexingConfigSnapshot.model_validate(
            runtime.job.config_snapshot
        )
        with publication_heartbeat(owner) as lost:
            inputs = load_owned_writer_inputs(owner)
            if (
                snapshot.input_hash_version is RegulatoryInputHashVersion.CHUNK_ROWS_V3
                and preparation.regulatory_chunks_content_hash(
                    canonical_snapshot_rows(inputs.canonical)
                )
                != runtime.job.content_hash
            ):
                raise IndexingPublicationIndeterminateError(
                    "durable canonical input changed; restart from current canonical authority"
                )
            embedding_tokenizer = preparation.get_tokenizer(
                snapshot.embedding_model_name, snapshot.embedding_provider
            )
            contextual_tokenizer = preparation.get_contextual_token_budget_tokenizer(
                model_provider=LlmProviderNames.VERTEX_AI,
                model_name=snapshot.vertex.model_name,
            )
            prepared = prepare_owned_durable_items(
                owner=owner,
                job=runtime.job,
                inputs=inputs,
                embedding_tokenizer=embedding_tokenizer,
                contextual_tokenizer=contextual_tokenizer,
            )
            retained_dates = durable_context_reference_dates(owner)
            references = {}
            for item in runtime.indexing_items:
                frozen = projection_input(item)
                if frozen is not None and frozen.reference_date is not None:
                    references[
                        (
                            item.regulatory_chunk_id,
                            item.effective_start,
                            item.effective_end,
                        )
                    ] = frozen.reference_date
                    continue
                context_input = (item.context or {}).get("context_input")
                source_hash = (
                    cast(dict[str, object], context_input).get("source_snapshot_sha256")
                    if isinstance(context_input, dict)
                    else None
                )
                reference = (
                    retained_dates.get(source_hash)
                    if isinstance(source_hash, str)
                    else None
                )
                if reference is not None:
                    for desired in prepared:
                        if (
                            desired.regulatory_chunk_id == item.regulatory_chunk_id
                            and (
                                desired.effective_start is None
                                or desired.effective_start <= reference
                            )
                            and (
                                desired.effective_end is None
                                or reference < desired.effective_end
                            )
                        ):
                            references[
                                (
                                    item.regulatory_chunk_id,
                                    desired.effective_start,
                                    desired.effective_end,
                                )
                            ] = reference
            if references:
                prepared = prepare_owned_durable_items(
                    owner=owner,
                    job=runtime.job,
                    inputs=inputs,
                    embedding_tokenizer=embedding_tokenizer,
                    contextual_tokenizer=contextual_tokenizer,
                    reference_dates=references,
                )
            if lost.is_set():
                raise IndexingPublicationIndeterminateError(
                    "durable preparation ownership heartbeat lost"
                )
        return repair_durable_item_checkpoints(
            owner,
            job_id=job_id,
            expected_generation=expected_generation,
            stage=stage,
            expected_input_sha256=durable_publication_input_digest(runtime),
            prepared=prepared,
        )
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass
