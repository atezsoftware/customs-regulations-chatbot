"""Prepare only dependency-proven deltas and preserve actual historical vectors."""

import json
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
from uuid import NAMESPACE_URL, uuid4, uuid5

from onyx.configs.constants import FileOrigin
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_annex_publication import (
    effective_positions,
    load_annex_publication_inputs,
    load_binding_context_sources,
    load_file_position_views,
    load_file_temporal_bindings,
    publication_input_scope_hash,
)
from onyx.db.regulatory_context_projections import load_context_view
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationScope,
    RetainedPublicationProjection,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexContextImpact,
    AnnexIndexedBaseline,
    AnnexPositionView,
    AnnexPublicationImpactCounts,
    AnnexPublicationPreparation,
    AnnexPublicationProjectionPlan,
    AnnexPublicationReview,
    AnnexRetainedProjection,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.publication import (
    prepare_legal_publication_timeline,
    resolve_after_window_authority,
)
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt
from onyx.regulatory.amendments.annexes.publication_representations import (
    _epoch,
    _snapshot,
    _source_template,
    temporal_candidate_rows,
)
from onyx.regulatory.amendments.annexes.selective_impact import (
    dependency_impact,
    local_source_rows,
    review_units,
)
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import get_current_tenant_id


def prepare_selective_publication(draft: AnnexChangeDraft) -> AnnexChangeDraft:
    from onyx.regulatory.amendments.annexes.analysis import (
        prepare_publication_context_view,
        resolve_review_context_llm,
    )
    from onyx.regulatory.amendments.annexes.preparation_progress import (
        report_preparation_progress,
    )
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        _target_configuration,
    )

    if (
        draft.user_file_id is None
        or draft.effective_date is None
        or draft.batch_id is None
    ):
        raise ValueError("selective publication scope missing")
    draft = resolve_after_window_authority(draft)
    assert draft.user_file_id is not None and draft.batch_id is not None
    selected_file_id, selected_batch_id = draft.user_file_id, draft.batch_id
    legal = prepare_legal_publication_timeline(draft)
    scope = PublicationScope(
        tenant_id=get_current_tenant_id(),
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(
        selected_file_id, owner_id=uuid4(), ttl=timedelta(hours=1)
    )
    try:
        with get_session_with_current_tenant() as session:
            file, settings_list, access = load_annex_publication_inputs(session, draft)
            bindings = load_file_temporal_bindings(session, file.id)
            previous_positions = load_file_position_views(session, file.id)
            existing_context = load_context_view(
                session, user_file_id=file.id, as_of_date=legal.effective_start
            )
            bound_context = load_binding_context_sources(
                session,
                file.id,
                [
                    b
                    for b in bindings
                    if (
                        b.effective_start is None
                        or b.effective_start <= legal.effective_start
                    )
                    and (
                        b.effective_end is None
                        or b.effective_end > legal.effective_start
                    )
                ],
            )
            existing_context = PreparedContextView(
                projections=[*existing_context.projections, *bound_context.projections],
                snapshots=list(
                    {
                        s.sha256: s
                        for s in [*existing_context.snapshots, *bound_context.snapshots]
                    }.values()
                ),
                calls=existing_context.calls,
            )
        rows = temporal_candidate_rows(
            draft,
            legal,
            legal.effective_start,
            bindings,
            effective_positions(previous_positions, legal.effective_start),
        )
        snapshots = [_snapshot(row) for row in rows]
        unchanged_draft = draft.model_copy(
            update={
                "items": [],
                "after_window_authority": None,
                "date_resolution": None,
                "source_only_canonical_ids": [],
            }
        )
        current_legal = prepare_legal_publication_timeline(unchanged_draft)
        current_snapshots = [
            _snapshot(row)
            for row in temporal_candidate_rows(
                unchanged_draft,
                current_legal,
                legal.effective_start,
                bindings,
                effective_positions(previous_positions, legal.effective_start),
            )
        ]
        indexes = []
        baselines: list[AnnexIndexedBaseline] = []
        runtimes: dict[str, str] = {}
        contextual_ids: set[str] = set()
        with ElasticsearchClient() as transport:
            client = transport.publication_client()
            for setting in settings_list:
                embedder = DefaultIndexingEmbedder.from_db_search_settings(
                    search_settings=setting
                )
                snapshot = (
                    RegulatoryIndexingConfigSnapshot.model_validate(
                        draft.indexing_configuration
                    )
                    if setting.status.is_current() and draft.indexing_configuration
                    else None
                )
                llm = resolve_review_context_llm(setting, snapshot)
                configuration = _target_configuration(setting, snapshot, embedder)
                receipt = _encoder_receipt(
                    configuration,
                    resolution=context_hash(draft.preparation_configuration),
                )
                info = client.indices.get(index=setting.index_name)
                if set(info) != {setting.index_name}:
                    raise ValueError("publication requires a concrete index")
                index = PublicationIndexSnapshot(
                    index_name=setting.index_name,
                    index_uuid=info[setting.index_name]["settings"]["index"]["uuid"],
                    search_settings_id=setting.id,
                    model_provider=receipt.authority.provider or "",
                    model_name=receipt.authority.model,
                    vector_dimension=setting.final_embedding_dim,
                    embedding_config_sha256=context_hash(configuration),
                    multitenant=MULTI_TENANT,
                    encoder_authority=receipt.authority,
                    encoder_receipts=(receipt,),
                )
                indexes.append(index)
                runtimes[index.index_uuid] = context_hash(
                    [configuration, llm.config.model_dump(mode="json") if llm else None]
                )
                for evidence in FencedPublicationIndex(
                    client, index
                ).inventory_evidence(authority.reservations(owner)):
                    source = json.loads(evidence.source_json)
                    bound = next(
                        (
                            b
                            for b in bindings
                            if b.index.matches_temporal_index(index)
                            and b.projection.ordinal == source["chunk_index"]
                        ),
                        None,
                    )
                    baselines.append(
                        AnnexIndexedBaseline(
                            evidence=evidence,
                            reference_date=bound.reference_date if bound else None,
                            binding=bound,
                        )
                    )
                    boundary = _epoch(legal.effective_start)
                    if (
                        (source.get("chunk_context") or source.get("doc_summary"))
                        and (
                            source.get("validity_start_date") is None
                            or source["validity_start_date"] <= boundary
                        )
                        and (
                            source.get("validity_end_date") is None
                            or source["validity_end_date"] > boundary
                        )
                    ):
                        contextual_ids.add(source["regulatory_chunk_id"])
        report_preparation_progress("dependency_impact", total=len(snapshots))
        direct_old = {
            identifier for item in draft.items for identifier in item.old_chunk_ids
        }
        impact = dependency_impact(
            before=current_snapshots,
            after=snapshots,
            items=draft.items,
            contexts=existing_context,
            contextual_ids=contextual_ids - direct_old,
        )
        # A global-to-local policy migration is separate work, never hidden in this update.
        unresolved = dict(impact.unresolved)
        source_snapshots = {s.sha256: s for s in existing_context.snapshots}
        for actual in baselines:
            source = json.loads(actual.evidence.source_json)
            identifier = source["regulatory_chunk_id"]
            if identifier not in contextual_ids - direct_old:
                continue
            boundary = _epoch(legal.effective_start)
            if (
                source.get("validity_start_date") is not None
                and source["validity_start_date"] > boundary
            ) or (
                source.get("validity_end_date") is not None
                and source["validity_end_date"] <= boundary
            ):
                continue
            matching = [
                p
                for p in existing_context.projections
                if p.canonical_chunk_id == identifier
                and p.doc_summary == (source.get("doc_summary") or "")
                and p.chunk_context == (source.get("chunk_context") or "")
            ]
            if not matching:
                unresolved[identifier] = [
                    "indexed context does not match recorded source proof"
                ]
        for identifier in (set(impact.affected_ids) & contextual_ids) - direct_old:
            if identifier not in {r.id for r in snapshots}:
                continue
            local_ids = {r.id for r in local_source_rows(snapshots, identifier)}
            for context in existing_context.projections:
                if context.canonical_chunk_id != identifier:
                    continue
                source_snapshot = source_snapshots.get(context.source_snapshot_sha256)
                if (
                    source_snapshot
                    and {r.canonical_chunk_id for r in source_snapshot.ordered_ranges}
                    - local_ids
                ):
                    unresolved[identifier] = [
                        "existing broad context requires an explicit scope migration"
                    ]
        impact = impact.model_copy(update={"unresolved": unresolved})
        draft = draft.model_copy(
            update={
                "dependency_impact": impact,
                "baseline_context": existing_context,
                "publication": None,
            }
        )
        affected = set(impact.affected_ids)
        if unresolved:
            return draft.model_copy(
                update={
                    "impact": AnnexContextImpact(
                        direct_canonical_changes=impact.changed_new_ids,
                        contextual_candidates=sorted(affected),
                        embedding_changes=[],
                        context_only=[],
                        metadata_only=[],
                        retire_history=[],
                        unchanged=impact.unchanged_ids,
                        reasons={**impact.reasons, **unresolved},
                        prepared=PreparedContextView(),
                        ready=False,
                    )
                }
            )
        baseline_ids = {row.id for row in draft.baseline_scope}
        legal = legal.model_copy(
            update={
                "canonical_rows": [
                    row
                    if row.id in baseline_ids
                    else row.model_copy(
                        update={
                            "projection_ordinal": authority.allocate(
                                owner, f"canonical:{row.id}"
                            )
                        }
                    )
                    for row in legal.canonical_rows
                ]
            }
        )
        retained: list[AnnexRetainedProjection] = []
        for actual in baselines:
            source = json.loads(actual.evidence.source_json)
            identifier = source["regulatory_chunk_id"]
            end = source.get("validity_end_date")
            start = source.get("validity_start_date")
            boundary = _epoch(legal.effective_start)
            if identifier in affected and (end is None or end > boundary):
                if start == boundary and identifier not in direct_old:
                    # A revised same-day derived view supersedes an earlier publication,
                    # while its exact source/vector remains in the frozen prior manifest.
                    continue
                if start is not None and start >= boundary:
                    raise ValueError(
                        "scheduled affected projection requires a separately dated review"
                    )
                retained.append(
                    AnnexRetainedProjection(
                        evidence=actual.evidence,
                        close_interval=True,
                        validity_end=legal.effective_start,
                    )
                )
            else:
                retained.append(AnnexRetainedProjection(evidence=actual.evidence))
        for item in retained:
            source = json.loads(item.evidence.source_json)
            if item.close_interval:
                source["validity_end_date"] = _epoch(item.validity_end)
            RetainedPublicationProjection(
                evidence=item.evidence, source_json=json.dumps(source)
            )
        lower_bounds = [
            row.validity_start_date
            for row in current_snapshots
            if row.validity_start_date is not None
        ]
        lower_bounds.extend(
            view.effective_start
            for view in previous_positions
            if view.effective_start is not None
            and view.effective_start < legal.effective_start
        )
        lower = max(lower_bounds) if lower_bounds else None
        position_views = (
            [
                AnnexPositionView(
                    effective_start=lower,
                    effective_end=legal.effective_start,
                    positions={row.id: row.position for row in current_snapshots},
                )
            ]
            if lower is None or lower < legal.effective_start
            else []
        )
        plans: list[AnnexPublicationProjectionPlan] = []
        views: dict[str, PreparedContextView] = {}
        dates = sorted(
            {
                legal.effective_start,
                *([legal.effective_end] if legal.effective_end else []),
            }
        )
        for setting, index in zip(settings_list, indexes, strict=True):
            embedder = DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            )
            snapshot = (
                RegulatoryIndexingConfigSnapshot.model_validate(
                    draft.indexing_configuration
                )
                if setting.status.is_current() and draft.indexing_configuration
                else None
            )
            llm = resolve_review_context_llm(setting, snapshot)
            for when in dates:
                candidates = temporal_candidate_rows(
                    draft,
                    legal,
                    when,
                    bindings,
                    effective_positions(previous_positions, when),
                )
                if setting == settings_list[0]:
                    position_views.append(
                        AnnexPositionView(
                            effective_start=when,
                            effective_end=legal.effective_end
                            if when == legal.effective_start
                            else None,
                            positions={row.id: row.position for row in candidates},
                        )
                    )
                candidates_by_id = {row.id: row for row in candidates}
                candidate_snapshots = [_snapshot(row) for row in candidates]
                for row in candidates:
                    if (
                        row.id not in affected
                        and row.id not in legal.restoration_predecessors
                    ):
                        continue
                    owner = authority.heartbeat(owner, ttl=timedelta(hours=1))
                    sources = canonical_snapshot_rows(
                        local_source_rows(candidate_snapshots, row.id)
                    )
                    view = prepare_publication_context_view(
                        rows=sources,
                        target_ids={row.id},
                        file=file,
                        settings=setting,
                        snapshot=snapshot,
                        embedder=embedder,
                        context_llm=llm,
                        reference_date=when,
                        cached=draft.impact.prepared if draft.impact else None,
                    )
                    if (
                        view.issues
                        or len(view.projections) != 1
                        or view.projections[0].canonical_chunk_id != row.id
                    ):
                        raise ValueError("selective context coverage mismatch")
                    context = view.projections[0]
                    digest = context_hash(view.model_dump(mode="json"))
                    views[digest] = view
                    limit = (
                        min(
                            v
                            for v in (
                                row.validity_end_date,
                                legal.effective_end
                                if when == legal.effective_start
                                else None,
                            )
                            if v is not None
                        )
                        if row.validity_end_date
                        or when == legal.effective_start
                        and legal.effective_end
                        else None
                    )
                    template = json.loads(
                        _source_template(
                            candidates_by_id[row.id],
                            context,
                            file_name=file.name,
                            access=access,
                            tenant_id=scope.tenant_id,
                            dimension=index.vector_dimension,
                        )
                    )
                    template.update(
                        validity_start_date=_epoch(when),
                        validity_end_date=_epoch(limit),
                    )
                    identity = uuid5(
                        NAMESPACE_URL,
                        context_hash([index.index_uuid, row.id, when, limit, digest]),
                    )
                    ordinal = authority.allocate(owner, f"context:{identity}")
                    template["chunk_index"] = ordinal
                    from onyx.regulatory.amendments.annexes.publication import (
                        exact_reusable_projection,
                    )
                    from onyx.regulatory.amendments.annexes.staging import (
                        staged_canonical_predecessors,
                    )

                    predecessor = staged_canonical_predecessors(draft.items).get(row.id)
                    reuse = None
                    for actual in baselines:
                        if actual.evidence.index.matches_temporal_index(index):
                            reuse = exact_reusable_projection(
                                context, actual.evidence, predecessor_id=predecessor
                            )
                            if reuse is not None:
                                break
                    plans.append(
                        AnnexPublicationProjectionPlan(
                            id=identity,
                            index=index,
                            ordinal=ordinal,
                            row=_snapshot(row),
                            canonical_base_sha256=context_hash(
                                next(
                                    r.text
                                    for r in legal.canonical_rows
                                    if r.id == row.id
                                )
                            ),
                            context=context,
                            view_sha256=digest,
                            effective_start=when,
                            effective_end=limit,
                            reference_date=when,
                            source_template_json=json.dumps(template),
                            reuse_from=reuse,
                            reason="verified_source_dependency_changed",
                        )
                    )
        prepared_view = PreparedContextView(
            projections=[p.context for p in plans if p.index == indexes[0]],
            snapshots=list(
                {s.sha256: s for v in views.values() for s in v.snapshots}.values()
            ),
            calls=list(
                {c.request_sha256: c for v in views.values() for c in v.calls}.values()
            ),
        )
        draft = draft.model_copy(
            update={
                "impact": AnnexContextImpact(
                    direct_canonical_changes=impact.changed_new_ids,
                    contextual_candidates=sorted(affected),
                    embedding_changes=sorted(
                        {p.row.id for p in plans if p.reuse_from is None}
                    ),
                    context_only=sorted(
                        affected
                        - set(impact.changed_old_ids)
                        - set(impact.changed_new_ids)
                    ),
                    metadata_only=[],
                    retire_history=impact.changed_old_ids,
                    unchanged=impact.unchanged_ids,
                    reasons=impact.reasons,
                    prepared=prepared_view,
                    ready=True,
                )
            }
        )
        reserved = list(authority.reservations(owner).ordinals)
        retired = {
            index.index_uuid: sorted(
                set(reserved)
                - {p.ordinal for p in plans if p.index == index}
                - {
                    json.loads(r.evidence.source_json)["chunk_index"]
                    for r in retained
                    if r.evidence.index.matches_temporal_index(index)
                }
            )
            for index in indexes
        }
        counts = AnnexPublicationImpactCounts(
            canonical_changes=len(review_units(draft.items)),
            context_consumers=len({p.row.id for p in plans}),
            embeddings=sum(
                len(p.context.embedding_texts) for p in plans if p.reuse_from is None
            ),
            exact_vector_reuses=sum(p.reuse_from is not None for p in plans),
            historical_projections=sum(r.close_interval for r in retained),
            retired_projections=sum(len(v) for v in retired.values()),
            total_projections=len(plans),
            preserved_vectors=len(retained),
            metadata_updates=sum(r.close_interval for r in retained),
        )
        prepared = AnnexPublicationPreparation(
            user_file_id=file.id,
            batch_id=selected_batch_id,
            review_input_sha256=context_hash(
                draft.model_dump(mode="json", exclude={"publication"})
            ),
            input_scope_sha256=publication_input_scope_hash(
                file, settings_list, access
            ),
            runtime_configuration=runtimes,
            counts=counts,
            scope=scope,
            indexes=indexes,
            legal=legal,
            indexed_baseline=baselines,
            projections=plans,
            views=views,
            reserved_ordinals=reserved,
            retired_ordinals=retired,
            retained=retained,
            position_views=position_views,
        )
        payload = prepared.model_dump_json().encode()
        file_id = get_default_file_store().save_file(
            BytesIO(payload),
            f"annex-delta-{uuid4()}.json",
            FileOrigin.OTHER,
            "application/json",
        )
        return draft.model_copy(
            update={
                "publication": AnnexPublicationReview(
                    artifact_file_id=file_id,
                    artifact_sha256=sha256(payload).hexdigest(),
                    artifact_byte_count=len(payload),
                    scope=scope,
                    indexes=indexes,
                    counts=counts,
                    effective_dates=dates,
                    projection_ids=[p.id for p in plans],
                )
            }
        )
    finally:
        authority.release(owner)
