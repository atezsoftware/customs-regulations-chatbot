"""Final preapproval search evidence and temporal plans; never submits embeddings."""

import json
from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid4, uuid5

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_annex_publication import (
    load_annex_publication_inputs,
    load_file_temporal_bindings,
    publication_input_scope_hash,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationScope,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexIndexedBaseline,
    AnnexPublicationImpactCounts,
    AnnexPublicationPreparation,
    AnnexPublicationProjectionPlan,
    AnnexPublicationReview,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.publication import (
    exact_reusable_projection,
    prepare_legal_publication_timeline,
)
from onyx.regulatory.amendments.annexes.publication_evidence import (
    _encoder_receipt,
    _historical_plan,
    _uncovered_history,
    _validate_physical_encoder_authority,
    read_publication_preparation,
)
from onyx.regulatory.amendments.annexes.publication_representations import (
    _epoch,
    _snapshot,
    _source_template,
    temporal_candidate_rows,
)
from onyx.regulatory.amendments.annexes.staging import (
    staged_canonical_predecessors,
)

if TYPE_CHECKING:
    from onyx.db.models import SearchSettings
    from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot


def _target_configuration(
    settings: "SearchSettings",
    snapshot: "RegulatoryIndexingConfigSnapshot | None",
    embedder: DefaultIndexingEmbedder,
) -> dict[str, str | int | float | bool | None]:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        encoder_model_fingerprint,
    )

    configuration: dict[str, str | int | float | bool | None] = (
        encoder_model_fingerprint(
            embedder.embedding_model,
            model_dim=settings.final_embedding_dim,
            formatter="normal-v1",
        )
    )
    if snapshot:
        if snapshot.openrouter_batch:
            configuration = {
                "transport": "openrouter_batch",
                "provider": snapshot.embedding_provider.value,
                "model": snapshot.openrouter_batch.model_name,
                "dimension": snapshot.effective_dimension,
                "endpoint_sha256": context_hash(snapshot.openrouter_batch.api_url),
                "embedding_endpoint": "/v1/embeddings",
                "formatter": "durable-context-before-text-v1",
            }
        else:
            configuration.update(
                formatter="durable-context-before-text-v1",
                transport="synchronous_encoder",
            )
    return configuration


def prepare_publication_review(draft: AnnexChangeDraft) -> AnnexChangeDraft:
    """Acquire/freeze the complete real indexed delta before a human can approve."""
    from onyx.configs.constants import FileOrigin
    from onyx.regulatory.amendments.annexes.analysis import (
        prepare_publication_context_view,
        resolve_review_context_llm,
    )
    from onyx.regulatory.amendments.annexes.publication import (
        resolve_after_window_authority,
    )
    from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
    from shared_configs.configs import MULTI_TENANT
    from shared_configs.contextvars import get_current_tenant_id

    if (
        draft.user_file_id is None
        or draft.effective_date is None
        or draft.impact is None
    ):
        raise ValueError("logical annex context must precede publication preparation")
    draft = resolve_after_window_authority(draft)
    assert (
        draft.user_file_id is not None
        and draft.effective_date is not None
        and draft.impact is not None
    )
    legal = prepare_legal_publication_timeline(draft)
    scope = PublicationScope(
        tenant_id=get_current_tenant_id(),
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(
        draft.user_file_id, owner_id=uuid4(), ttl=timedelta(hours=1)
    )
    try:
        with get_session_with_current_tenant() as session:
            file, settings_list, access = load_annex_publication_inputs(session, draft)
            bindings = load_file_temporal_bindings(session, file.id)
        old_ids = {row.id for row in draft.baseline_scope}
        allocated = []
        for row in legal.canonical_rows:
            ordinal = authority.allocate(owner, f"canonical:{row.id}")
            if row.id in old_ids and row.projection_ordinal != ordinal:
                raise ValueError("existing canonical ordinal changed")
            allocated.append(row.model_copy(update={"projection_ordinal": ordinal}))
        legal = legal.model_copy(update={"canonical_rows": allocated})
        baseline = {row.id: row for row in draft.baseline_scope}
        canonical = {row.id: row for row in legal.canonical_rows}
        dates = sorted(
            {
                draft.effective_date,
                *(
                    value
                    for row in legal.canonical_rows
                    for value in (row.validity_start_date, row.validity_end_date)
                    if value is not None
                ),
                *([legal.effective_end] if legal.effective_end else []),
            }
        )
        all_plans: list[AnnexPublicationProjectionPlan] = []
        views: dict[str, PreparedContextView] = {}
        all_evidence: list[AnnexIndexedBaseline] = []
        indexes: list[PublicationIndexSnapshot] = []
        retired: dict[str, list[int]] = {}
        runtime_configuration: dict[str, str] = {}
        with ElasticsearchClient() as transport:
            client = transport.publication_client()
            present_actuals: list[AnnexIndexedBaseline] = []
            for settings in sorted(
                settings_list, key=lambda item: not item.status.is_current()
            ):
                embedder = DefaultIndexingEmbedder.from_db_search_settings(
                    search_settings=settings
                )
                snapshot = (
                    RegulatoryIndexingConfigSnapshot.model_validate(
                        draft.indexing_configuration
                    )
                    if settings.status.is_current() and draft.indexing_configuration
                    else None
                )
                llm = resolve_review_context_llm(settings, snapshot)
                configuration = _target_configuration(settings, snapshot, embedder)
                receipt = _encoder_receipt(
                    configuration,
                    resolution=context_hash(draft.preparation_configuration),
                )
                info = client.indices.get(index=settings.index_name)
                if set(info) != {settings.index_name}:
                    raise ValueError("publication requires a concrete index")
                index = PublicationIndexSnapshot(
                    index_name=settings.index_name,
                    index_uuid=info[settings.index_name]["settings"]["index"]["uuid"],
                    search_settings_id=settings.id,
                    model_provider=receipt.authority.provider or "",
                    model_name=receipt.authority.model,
                    vector_dimension=settings.final_embedding_dim,
                    embedding_config_sha256=context_hash(configuration),
                    multitenant=MULTI_TENANT,
                    encoder_authority=receipt.authority,
                    encoder_receipts=(receipt,),
                )
                runtime_configuration[index.index_uuid] = context_hash(
                    [configuration, llm.config.model_dump(mode="json") if llm else None]
                )
                adapter = FencedPublicationIndex(client, index)
                actuals = [
                    AnnexIndexedBaseline(
                        evidence=item,
                        reference_date=next(
                            (
                                binding.reference_date
                                for binding in bindings
                                if binding.index.index_uuid == index.index_uuid
                                and binding.projection.ordinal
                                == json.loads(item.source_json)["chunk_index"]
                            ),
                            None,
                        ),
                        binding=next(
                            (
                                binding
                                for binding in bindings
                                if binding.index.index_uuid == index.index_uuid
                                and binding.projection.ordinal
                                == json.loads(item.source_json)["chunk_index"]
                            ),
                            None,
                        ),
                    )
                    for item in adapter.inventory_evidence(
                        authority.reservations(owner)
                    )
                ]
                for actual in actuals:
                    _validate_physical_encoder_authority(actual, index)
                all_evidence.extend(actuals)
                historical_actuals = actuals
                if settings.status.is_current():
                    present_actuals = actuals
                else:
                    historical_actuals = [*actuals, *present_actuals]
                plans: list[AnnexPublicationProjectionPlan] = []
                for actual in historical_actuals:
                    source = json.loads(actual.evidence.source_json)
                    row = baseline.get(source.get("regulatory_chunk_id"))
                    if row is None:
                        raise ValueError(
                            "indexed historical canonical identity is outside the complete legal baseline"
                        )
                    historical = _historical_plan(
                        actual,
                        row,
                        index,
                        configuration,
                        draft.effective_date,
                        embedder.embedding_model,
                    )
                    if historical:
                        views[historical.view_sha256] = PreparedContextView(
                            projections=[historical.context]
                        )
                        plans.extend(_uncovered_history(historical, plans))

                # Missing historical source cannot be reconstructed from fresh OLD generation.
                if settings.status.is_current():
                    indexed_ids = {
                        json.loads(item.evidence.source_json).get("regulatory_chunk_id")
                        for item in actuals
                    }
                    if any(
                        row.id not in indexed_ids
                        and (
                            row.validity_start_date is None
                            or row.validity_start_date < draft.effective_date
                        )
                        for row in draft.baseline_scope
                    ):
                        raise ValueError("historical indexed source is missing")
                cached = draft.impact.prepared if settings.status.is_current() else None
                predecessors = staged_canonical_predecessors(draft.items)
                predecessors.update(legal.restoration_predecessors)
                index_dates = sorted(
                    {
                        *dates,
                        *(
                            value
                            for actual in historical_actuals
                            if actual.binding is not None
                            for value in (
                                actual.binding.effective_start,
                                actual.binding.effective_end,
                            )
                            if value is not None
                        ),
                    }
                )
                intervals: list[tuple[date | None, date | None, date]] = [
                    (
                        when,
                        index_dates[offset + 1]
                        if offset + 1 < len(index_dates)
                        else None,
                        when,
                    )
                    for offset, when in enumerate(index_dates)
                ]
                if any(row.validity_start_date is None for row in draft.baseline_scope):
                    if index_dates[0] == date.min:
                        raise ValueError(
                            "open historical interval has no representable reconstruction reference"
                        )
                    intervals.insert(
                        0, (None, index_dates[0], index_dates[0] - timedelta(days=1))
                    )
                for start, end, when in intervals:
                    owner = authority.heartbeat(owner, ttl=timedelta(hours=1))
                    rows = temporal_candidate_rows(
                        draft,
                        legal,
                        when,
                        [
                            actual.binding
                            for actual in present_actuals
                            if actual.binding is not None
                        ],
                    )
                    if len({row.position for row in rows}) != len(rows):
                        raise ValueError(
                            "historical/context source snapshot has ambiguous canonical positions"
                        )
                    if when < draft.effective_date and all(
                        any(
                            plan.row.id == row.id
                            and (
                                plan.effective_start is None
                                or plan.effective_start <= when
                            )
                            and (
                                plan.effective_end is None or when < plan.effective_end
                            )
                            for plan in plans
                        )
                        for row in rows
                    ):
                        continue
                    view = prepare_publication_context_view(
                        rows=rows,
                        file=file,
                        settings=settings,
                        snapshot=snapshot,
                        embedder=embedder,
                        context_llm=llm,
                        reference_date=when,
                        cached=cached,
                    )
                    from onyx.regulatory.amendments.annexes.context_dependencies import (
                        validate_complete_context_view,
                    )

                    validate_complete_context_view(
                        rows=rows, view=view, as_of_date=when
                    )
                    view_sha256 = context_hash(view.model_dump(mode="json"))
                    views[view_sha256] = view
                    cached = view
                    by_id = {row.id: row for row in rows}
                    for context in view.projections:
                        row = by_id[context.canonical_chunk_id]
                        if when < draft.effective_date and any(
                            item.row.id == row.id
                            and (
                                item.effective_start is None
                                or item.effective_start <= when
                            )
                            and (
                                item.effective_end is None or when < item.effective_end
                            )
                            for item in plans
                        ):
                            continue
                        limit = (
                            min(
                                value
                                for value in (end, row.validity_end_date)
                                if value is not None
                            )
                            if end or row.validity_end_date
                            else None
                        )
                        template = json.loads(
                            _source_template(
                                row,
                                context,
                                file_name=file.name,
                                access=access,
                                tenant_id=scope.tenant_id,
                                dimension=index.vector_dimension,
                            )
                        )
                        (
                            template["validity_start_date"],
                            template["validity_end_date"],
                        ) = _epoch(start), _epoch(limit)
                        matches = [
                            actual
                            for actual in actuals
                            if (
                                when >= draft.effective_date
                                or actual.reference_date is not None
                            )
                            and exact_reusable_projection(
                                context,
                                actual.evidence,
                                predecessor_id=predecessors.get(row.id),
                            )
                            is not None
                        ]
                        reuse = (
                            matches[0].evidence.frozen_projection if matches else None
                        )
                        identity = uuid5(
                            NAMESPACE_URL,
                            context_hash(
                                [
                                    index.index_uuid,
                                    row.id,
                                    start,
                                    limit,
                                    template,
                                    context.model_dump(mode="json"),
                                ]
                            ),
                        )
                        plan = AnnexPublicationProjectionPlan(
                            id=identity,
                            index=index,
                            ordinal=-1,
                            row=_snapshot(row),
                            canonical_base_sha256=context_hash(canonical[row.id].text),
                            context=context,
                            view_sha256=view_sha256,
                            effective_start=start,
                            effective_end=limit,
                            reference_date=when,
                            source_template_json=json.dumps(template),
                            reuse_from=reuse,
                            reason="historical_context_rebuilt_from_dated_canonical_source"
                            if when < draft.effective_date
                            else "exact_index_input_reuse"
                            if reuse
                            else "full_input_or_configuration_requires_encoding",
                        )
                        plans.append(plan)
                receipts = {receipt.configuration_json: receipt}
                finalized: list[AnnexPublicationProjectionPlan] = []
                for plan in plans:
                    cfg = plan.context.embedding_config
                    additional = _encoder_receipt(
                        cfg, resolution=context_hash(draft.preparation_configuration)
                    )
                    if additional.authority != receipt.authority:
                        raise ValueError(
                            "historical encoder authority is incompatible with the target index"
                        )
                    receipts[additional.configuration_json] = additional
                index = index.model_copy(
                    update={"encoder_receipts": tuple(receipts.values())}
                )
                index = PublicationIndexSnapshot.model_validate(
                    index.model_dump(mode="json")
                )
                indexes.append(index)
                for plan in plans:
                    ordinal = (
                        plan.ordinal
                        if plan.ordinal >= 0
                        else authority.allocate(owner, f"context:{plan.id}")
                    )
                    source = json.loads(plan.source_template_json)
                    source["chunk_index"] = ordinal
                    finalized.append(
                        plan.model_copy(
                            update={
                                "ordinal": ordinal,
                                "index": index,
                                "source_template_json": json.dumps(source),
                            }
                        )
                    )
                all_plans.extend(finalized)
                retired[index.index_uuid] = sorted(
                    {
                        json.loads(actual.evidence.source_json)["chunk_index"]
                        for actual in actuals
                    }
                    - {plan.ordinal for plan in finalized}
                )
        from onyx.regulatory.amendments.annexes.context_dependencies import (
            compare_context_views,
            verify_existing_index_evidence,
        )
        from onyx.regulatory.amendments.annexes.models import (
            ExistingIndexEmbeddingEvidence,
        )

        if draft.baseline_context is not None and draft.patch_plan is not None:
            proven = []
            for context in draft.baseline_context.projections:
                context = context.model_copy(
                    update={
                        "vector_reuse_verified": False,
                        "existing_index_evidence": {},
                    }
                )
                match = next(
                    (
                        actual
                        for actual in present_actuals
                        if exact_reusable_projection(
                            context, actual.evidence, predecessor_id=None
                        )
                    ),
                    None,
                )
                if match is not None:
                    source = json.loads(match.evidence.source_json)
                    context = verify_existing_index_evidence(
                        context,
                        ExistingIndexEmbeddingEvidence(
                            index_name=match.evidence.index.index_name,
                            document_id=str(file.id),
                            canonical_chunk_id=context.canonical_chunk_id,
                            projection_ordinal=source["chunk_index"],
                            embedding_input_sha256=context.embedding_input_sha256,
                            embedding_config_sha256=context.embedding_config_sha256,
                            canonical_text_sha256=context.canonical_text_sha256,
                            vector_dimension=len(source["content_vector"]),
                            expected_dimension=match.evidence.index.vector_dimension,
                        ),
                    )
                proven.append(context)
            before = draft.baseline_context.model_copy(update={"projections": proven})
            impact = compare_context_views(
                old=before,
                new=draft.impact.prepared,
                direct_canonical_changes=[
                    row.id for item in draft.items for row in item.new_chunks
                ],
                metadata_only=draft.patch_plan.metadata_only,
                canonical_predecessors=staged_canonical_predecessors(draft.items),
            )
            draft = draft.model_copy(
                update={"baseline_context": before, "impact": impact}
            )
        reservations = list(authority.reservations(owner).ordinals)
        retired = {
            index.index_uuid: sorted(
                set(reservations)
                - {plan.ordinal for plan in all_plans if plan.index == index}
            )
            for index in indexes
        }
        counts = AnnexPublicationImpactCounts(
            canonical_changes=sum(
                row.id not in baseline or row != baseline[row.id]
                for row in legal.canonical_rows
            ),
            context_consumers=len({plan.row.id for plan in all_plans}),
            embeddings=sum(plan.reuse_from is None for plan in all_plans),
            exact_vector_reuses=sum(plan.reuse_from is not None for plan in all_plans),
            historical_projections=sum(
                plan.effective_end is not None
                and plan.effective_end <= legal.effective_start
                for plan in all_plans
            ),
            retired_projections=sum(len(values) for values in retired.values()),
            total_projections=len(all_plans),
        )
        assert draft.batch_id is not None
        preparation = AnnexPublicationPreparation(
            user_file_id=file.id,
            batch_id=draft.batch_id,
            review_input_sha256=context_hash(
                draft.model_dump(mode="json", exclude={"publication"})
            ),
            input_scope_sha256=publication_input_scope_hash(
                file, settings_list, access
            ),
            counts=counts,
            runtime_configuration=runtime_configuration,
            scope=scope,
            indexes=indexes,
            legal=legal,
            indexed_baseline=all_evidence,
            projections=all_plans,
            views=views,
            reserved_ordinals=list(authority.reservations(owner).ordinals),
            retired_ordinals=retired,
        )
        payload = preparation.model_dump_json().encode()
        file_id = get_default_file_store().save_file(
            BytesIO(payload),
            f"annex-publication-{uuid4()}.json",
            FileOrigin.OTHER,
            "application/json",
        )
        review = AnnexPublicationReview(
            artifact_file_id=file_id,
            artifact_sha256=sha256(payload).hexdigest(),
            artifact_byte_count=len(payload),
            scope=scope,
            indexes=indexes,
            counts=counts,
            effective_dates=sorted(
                {
                    *dates,
                    *(
                        value
                        for plan in all_plans
                        for value in (plan.effective_start, plan.effective_end)
                        if value is not None
                    ),
                }
            ),
            projection_ids=[plan.id for plan in all_plans],
        )
        return draft.model_copy(update={"publication": review})
    finally:
        authority.release(owner)


def validate_frozen_publication_review(
    draft: AnnexChangeDraft,
) -> AnnexPublicationPreparation:
    if draft.publication is None or draft.user_file_id is None:
        raise ValueError("final indexed publication preparation requires revalidation")
    from shared_configs.contextvars import get_current_tenant_id

    expected_scope = PublicationScope(
        tenant_id=get_current_tenant_id(),
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    frozen = read_publication_preparation(draft.publication)
    if (
        frozen.scope != expected_scope
        or frozen.user_file_id != draft.user_file_id
        or frozen.batch_id != draft.batch_id
        or frozen.review_input_sha256
        != context_hash(draft.model_dump(mode="json", exclude={"publication"}))
    ):
        raise ValueError(
            "final publication proof belongs to a different review; revalidation required"
        )
    return frozen


def validate_indexed_publication_baseline(draft: AnnexChangeDraft) -> None:
    frozen = validate_frozen_publication_review(draft)
    authority = PublicationStore(frozen.scope)
    owner = authority.acquire(
        frozen.user_file_id, owner_id=uuid4(), ttl=timedelta(minutes=10)
    )
    try:
        reservations = authority.reservations(owner)
        if list(reservations.ordinals) != frozen.reserved_ordinals:
            raise ValueError("publication reservation baseline changed")
        with get_session_with_current_tenant() as session:
            file, settings, access = load_annex_publication_inputs(session, draft)
            if (
                publication_input_scope_hash(file, settings, access)
                != frozen.input_scope_sha256
            ):
                raise ValueError("publication file/ACL/index configuration changed")
        if {
            (item.id, item.index_name, item.final_embedding_dim) for item in settings
        } != {
            (item.search_settings_id, item.index_name, item.vector_dimension)
            for item in frozen.indexes
        }:
            raise ValueError("active publication indexes changed")
        from onyx.regulatory.amendments.annexes.analysis import (
            resolve_review_context_llm,
        )
        from onyx.regulatory.indexing_jobs.models import (
            RegulatoryIndexingConfigSnapshot,
        )

        for setting in settings:
            index = next(
                item for item in frozen.indexes if item.search_settings_id == setting.id
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
            configuration = _target_configuration(setting, snapshot, embedder)
            if (
                context_hash(
                    [configuration, llm.config.model_dump(mode="json") if llm else None]
                )
                != frozen.runtime_configuration[index.index_uuid]
            ):
                raise ValueError(
                    "publication encoder/context runtime configuration changed"
                )
        with ElasticsearchClient() as transport:
            for index in frozen.indexes:
                actual = FencedPublicationIndex(
                    transport.publication_client(), index
                ).inventory_evidence(reservations)
                before = [
                    item.evidence
                    for item in frozen.indexed_baseline
                    if item.evidence.index.index_uuid == index.index_uuid
                ]

                def normalized(items: object) -> str:
                    return context_hash(items)

                if normalized(
                    [json.loads(item.source_json) for item in actual]
                ) != normalized([json.loads(item.source_json) for item in before]):
                    raise ValueError("actual indexed publication baseline changed")
    finally:
        authority.release(owner)
