"""Date-qualified normal writer preparation using the complete contextual policy."""

import json
from datetime import date, timedelta
from uuid import UUID, uuid4

from elasticsearch import Elasticsearch

from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import OwnedWriterInputs
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
    ObservedPublicationProjection,
    PublicationIndexSnapshot,
    PublicationProjection,
    matches_indexed_evidence,
    publication_digest,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendment_projection_impact import (
    ContextAuditResolver,
    analyze_amendment_impact,
    merge_windows,
)
from onyx.regulatory.amendments.annexes.analysis import resolve_review_context_llm
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    effective_context_rows,
    freeze_encoder_inputs,
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt
from onyx.regulatory.amendments.annexes.publication_preparation import (
    _target_configuration,
)
from onyx.regulatory.amendments.annexes.publication_representations import (
    _epoch,
    _source_template,
)
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors
from onyx.regulatory.projection import prepare_normal_context_view
from onyx.regulatory.writer_publication_models import (
    AmendmentImpactReport,
    WriterPublicationManifest,
)
from shared_configs.configs import MULTI_TENANT
from shared_configs.enums import EmbedTextType


def _contains(binding: AnnexTemporalProjection, when: date) -> bool:
    return (binding.effective_start is None or binding.effective_start <= when) and (
        binding.effective_end is None or when < binding.effective_end
    )


def prepare_owned_correction(
    owner: FileOwnership,
    client: Elasticsearch,
    inputs: OwnedWriterInputs,
    after: list[AnnexCanonicalSnapshot],
    *,
    changed_id: str | None,
    target_settings_ids: set[int] | None = None,
    selective_amendment: bool = False,
    audit_cache: ContextAuditResolver | None = None,
) -> WriterPublicationManifest:
    """Correct one canonical version, preserving all other legal time windows."""
    authority = PublicationStore(owner.scope)
    identifier = uuid4()
    desired = {row.id: row for row in after}
    previous_canonical = {row.id: row for row in inputs.canonical}
    changed_ids = {row.id for row in after if previous_canonical.get(row.id) != row}
    target = desired[changed_id] if changed_id is not None else None
    affected = [(date.min, date.max)]
    if target is not None:
        previous_target = previous_canonical[target.id]
        windows = sorted(
            (row.validity_start_date or date.min, row.validity_end_date or date.max)
            for row in (previous_target, target)
        )
        if any(start >= end for start, end in windows):
            raise ValueError("correction validity window is empty")
        affected = [windows[0]]
        for start, end in windows[1:]:
            if start <= affected[-1][1]:
                affected[-1] = (affected[-1][0], max(end, affected[-1][1]))
            else:
                affected.append((start, end))
    lower, upper = affected[0][0], affected[-1][1]
    bindings: list[AnnexTemporalProjection] = []
    indexes: list[PublicationIndexSnapshot] = []
    views: list[PreparedContextView] = []
    impacts: list[AmendmentImpactReport] = []
    revisions: dict[UUID, UUID] = {}
    for settings in sorted(
        [
            item
            for item in inputs.settings
            if target_settings_ids is None or item.id in target_settings_ids
        ],
        key=lambda item: not item.status.is_current(),
    ):
        embedder = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=settings
        )
        configuration = _target_configuration(settings, None, embedder)
        receipt = _encoder_receipt(
            configuration, resolution=context_hash(configuration)
        )
        info = client.indices.get(index=settings.index_name)
        if set(info) != {settings.index_name}:
            raise ValueError("writer requires a concrete physical index")
        index_uuid = info[settings.index_name]["settings"]["index"]["uuid"]
        prior = [
            binding
            for binding in inputs.bindings
            if binding.index.index_uuid == index_uuid
        ]
        if any(
            binding.index.index_name == settings.index_name
            and binding.index.index_uuid != index_uuid
            for binding in inputs.bindings
        ):
            raise ValueError("writer physical index identity changed")
        context_prior = (
            prior
            if settings.status.is_current()
            else [
                binding
                for binding in inputs.bindings
                if binding.index.index_name
                in {
                    item.index_name
                    for item in inputs.settings
                    if item.status.is_current()
                }
            ]
        )
        receipts = {context_hash(configuration): receipt}
        for binding in prior:
            for retained in binding.index.encoder_receipts:
                if retained.effective_authority() != receipt.effective_authority():
                    raise ValueError(
                        "changed encoder authority requires a new physical index"
                    )
                receipts[context_hash(json.loads(retained.configuration_json))] = (
                    retained
                )
        index = PublicationIndexSnapshot(
            index_name=settings.index_name,
            index_uuid=index_uuid,
            search_settings_id=settings.id,
            model_provider=receipt.authority.provider or "",
            model_name=receipt.authority.model,
            vector_dimension=settings.final_embedding_dim,
            embedding_config_sha256=context_hash(configuration),
            multitenant=MULTI_TENANT,
            encoder_authority=receipt.authority,
            encoder_receipts=tuple(receipts.values()),
        )
        indexes.append(index)
        llm = resolve_review_context_llm(settings, None)
        impact_windows = None
        if selective_amendment:
            if not prior:
                raise ValueError(
                    "Selective amendment requires an existing qualified index baseline"
                )
            impact = analyze_amendment_impact(
                before=inputs.canonical,
                after=after,
                bindings=prior,
                llm=llm,
                audit_cache=audit_cache,
            ).model_copy(update={"index_uuid": index_uuid})
            if impact.unresolved:
                raise ValueError(
                    "amendment impact unresolved: " + "; ".join(impact.unresolved)
                )
            impacts.append(impact)
            impact_windows = impact.affected_windows
            affected = merge_windows(
                [w for ranges in impact_windows.values() for w in ranges]
            )
            if not affected:
                raise ValueError("Amendment has no effective canonical change")
            lower, upper = affected[0][0], affected[-1][1]
        used: set[int] = set()
        canonical_by_binding = {
            binding.id: json.loads(binding.projection.source_json)[
                "regulatory_chunk_id"
            ]
            for binding in context_prior + prior
        }
        candidates_for_reuse = [
            binding.projection
            for binding in [*prior, *context_prior]
            if isinstance(binding.projection, FrozenPublicationProjection)
        ]
        if prior:
            from onyx.document_index.elasticsearch.publication import (
                FencedPublicationIndex,
            )

            actual = FencedPublicationIndex(client, index).inventory_evidence(
                authority.reservations(owner)
            )
            actual_by_ordinal = {
                json.loads(item.source_json)["chunk_index"]: item for item in actual
            }
            for previous in prior:
                evidence = actual_by_ordinal.get(previous.projection.ordinal)
                if evidence is None or not matches_indexed_evidence(
                    previous.projection, evidence
                ):
                    raise ValueError(
                        "writer qualified baseline no longer matches actual ES evidence"
                    )

        def ordinal_for(
            canonical_id: str,
            start: date | None,
            end: date | None,
            previous: AnnexTemporalProjection | None,
        ) -> int:
            ordinal = (
                previous.projection.ordinal
                if previous
                else desired[canonical_id].projection_ordinal
            )
            if ordinal in used:
                ordinal = authority.allocate(
                    owner,
                    f"writer:{identifier}:{index_uuid}:{canonical_id}:{start}:{end}",
                )
            used.add(ordinal)
            return ordinal

        # Retain untouched intervals verbatim; split only a binding crossing the correction.
        for previous in prior:
            start, end = (
                previous.effective_start or date.min,
                previous.effective_end or date.max,
            )
            untouched = [(start, end)]
            previous_windows = (
                impact_windows.get(canonical_by_binding[previous.id], [])
                if impact_windows is not None
                else affected
            )
            for affected_start, affected_end in previous_windows:
                untouched = [
                    (part_start, part_end)
                    for original_start, original_end in untouched
                    for part_start, part_end in (
                        (original_start, min(original_end, affected_start)),
                        (max(original_start, affected_end), original_end),
                    )
                    if part_start < part_end
                ]
            if untouched == [(start, end)]:
                bindings.append(previous)
                used.add(previous.projection.ordinal)
                continue
            for part_start, part_end in untouched:
                if part_start >= part_end:
                    continue
                new_id = uuid4()
                start_date = None if part_start == date.min else part_start
                end_date = None if part_end == date.max else part_end
                source = json.loads(previous.projection.source_json)
                ordinal = ordinal_for(
                    source["regulatory_chunk_id"], start_date, end_date, previous
                )
                source.update(
                    chunk_index=ordinal,
                    validity_start_date=_epoch(start_date),
                    validity_end_date=_epoch(end_date),
                )
                projection: PublicationProjection
                if isinstance(previous.projection, ObservedPublicationProjection):
                    projection = ObservedPublicationProjection.model_validate(
                        {
                            **previous.projection.model_dump(),
                            "ordinal": ordinal,
                            "context_projection_id": str(new_id),
                            "source_json": json.dumps(source),
                        }
                    )
                else:
                    projection = FrozenPublicationProjection(
                        ordinal=ordinal,
                        context_projection_id=str(new_id),
                        source_json=json.dumps(source),
                        embedding_inputs=previous.projection.embedding_inputs,
                        embedding_config_json=previous.projection.embedding_config_json,
                    )
                bindings.append(
                    previous.model_copy(
                        update={
                            "id": new_id,
                            "projection": projection,
                            "effective_start": start_date,
                            "effective_end": end_date,
                        }
                    )
                )
                revisions[new_id] = inputs.revisions[previous.id]

        window_lower, window_upper = (lower, upper) if prior else (date.min, date.max)
        boundaries = sorted(
            {
                window_lower,
                window_upper,
                *(
                    value
                    for window in affected
                    for value in window
                    if window_lower < value < window_upper
                ),
                *(
                    value
                    for row in after
                    for value in (row.validity_start_date, row.validity_end_date)
                    if value is not None and window_lower < value < window_upper
                ),
                *(
                    value
                    for binding in context_prior
                    for value in (binding.effective_start, binding.effective_end)
                    if value is not None and window_lower < value < window_upper
                ),
            }
        )
        for start, end in zip(boundaries, boundaries[1:]):
            if prior and not any(
                lower <= start and end <= upper for lower, upper in affected
            ):
                continue
            when = (
                start
                if start != date.min
                else (end - timedelta(days=1) if end != date.max else date(2000, 1, 1))
            )
            start_date, end_date = (
                (None if start == date.min else start),
                (None if end == date.max else end),
            )
            rows = canonical_snapshot_rows(after)
            by_id = {row.id: row for row in rows}
            matching: dict[str, AnnexTemporalProjection] = {}
            for previous in context_prior:
                if _contains(previous, when):
                    canonical_id = canonical_by_binding[previous.id]
                    matching[canonical_id] = previous
                    row = by_id[canonical_id]
                    if row.id not in changed_ids:
                        row.text = previous.representation_text
                        row.chunk_metadata = dict(previous.representation_metadata)
                        row.position = previous.semantic_position
            rows = rebuild_context_aggregates(rows, changed_ids=sorted(changed_ids))
            active = effective_context_rows(rows, when)
            if not active:
                continue
            by_id = {row.id: row for row in active}
            view = prepare_normal_context_view(
                rows=rows,
                user_file=inputs.file,
                search_settings=settings,
                embedder=embedder,
                llm=llm,
                cached=inputs.cached,
                as_of_date=when,
                target_ids=(
                    {
                        identifier
                        for identifier, ranges in impact_windows.items()
                        if any(left <= start and end <= right for left, right in ranges)
                    }
                    if impact_windows is not None
                    else None
                ),
            )
            views.append(view)
            for context in view.projections:
                row = by_id[context.canonical_chunk_id]
                semantic_previous = matching.get(row.id)
                previous = next(
                    (
                        binding
                        for binding in prior
                        if (
                            canonical_by_binding[binding.id] == row.id
                            and _contains(binding, when)
                        )
                    ),
                    None,
                )
                ordinal = ordinal_for(row.id, start_date, end_date, previous)
                new_id = uuid4()
                row.projection_ordinal = ordinal
                row.validity_start_date, row.validity_end_date = start_date, end_date
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
                    validity_start_date=_epoch(start_date),
                    validity_end_date=_epoch(end_date),
                )
                reusable = next(
                    (
                        item
                        for item in candidates_for_reuse
                        if item.embedding_inputs == tuple(context.embedding_texts)
                        and publication_digest(json.loads(item.embedding_config_json))
                        == publication_digest(context.embedding_config)
                    ),
                    None,
                )
                if reusable is not None:
                    actual_source = json.loads(reusable.source_json)
                    source["content_vector"] = actual_source["content_vector"]
                    if "title_vector" in actual_source:
                        source["title_vector"] = actual_source["title_vector"]
                else:
                    texts, actual_configuration = freeze_encoder_inputs(
                        context.embedding_texts,
                        embedder.embedding_model,
                        model_dim=settings.model_dim,
                        formatter=str(context.embedding_config["formatter"]),
                    )
                    if (
                        texts != context.embedding_texts
                        or actual_configuration != context.embedding_config
                    ):
                        raise ValueError(
                            "writer encoder differs from complete frozen inputs"
                        )
                    vectors = _validate_response_vectors(
                        embedder.embedding_model.encode(
                            texts=texts,
                            text_type=EmbedTextType.PASSAGE,
                            tenant_id=owner.scope.tenant_id,
                        ),
                        expected_count=len(texts),
                        expected_dimension=index.vector_dimension,
                    )
                    source["content_vector"] = vectors[0]
                    if source.get("title"):
                        source["title_vector"] = vectors[-1]
                projection = FrozenPublicationProjection(
                    ordinal=ordinal,
                    context_projection_id=str(new_id),
                    source_json=json.dumps(source),
                    embedding_inputs=tuple(context.embedding_texts),
                    embedding_config_json=json.dumps(context.embedding_config),
                )
                candidates_for_reuse.append(projection)
                base = desired[row.id]
                if semantic_previous is not None and row.id not in changed_ids:
                    canonical_base = semantic_previous.canonical_base_sha256
                    revisions[new_id] = inputs.revisions[semantic_previous.id]
                else:
                    canonical_base = context_hash(base.text)
                role = (
                    "hierarchical_aggregate"
                    if row.chunk_metadata.get("chunk_variant")
                    == "hierarchical_aggregate"
                    else "image_companion"
                    if row.chunk_metadata.get("bound_to_regulatory_chunk_id")
                    else "canonical"
                )
                rebuilt = AnnexTemporalProjection(
                    id=new_id,
                    index=index,
                    projection=projection,
                    canonical_base_sha256=canonical_base,
                    derived_role=role,
                    dependency_ids=context.canonical_dependency_ids,
                    representation_text=row.text,
                    representation_metadata=row.chunk_metadata,
                    context=context,
                    reference_date=when,
                    effective_start=start_date,
                    effective_end=end_date,
                    semantic_position=row.position,
                )

                if (
                    previous is not None
                    and isinstance(previous.projection, FrozenPublicationProjection)
                    and (
                        previous.index.index_uuid == index.index_uuid
                        and previous.projection.ordinal == ordinal
                        and previous.effective_start == start_date
                        and previous.effective_end == end_date
                        and previous.canonical_base_sha256 == canonical_base
                        and previous.representation_text == rebuilt.representation_text
                        and previous.representation_metadata
                        == rebuilt.representation_metadata
                        and previous.dependency_ids == rebuilt.dependency_ids
                        and publication_digest(
                            json.loads(previous.projection.source_json)
                        )
                        == publication_digest(source)
                        and previous.projection.embedding_inputs
                        == projection.embedding_inputs
                        and publication_digest(
                            json.loads(previous.projection.embedding_config_json)
                        )
                        == publication_digest(context.embedding_config)
                    )
                ):
                    bindings.append(previous)
                else:
                    bindings.append(rebuilt)

    return WriterPublicationManifest(
        id=identifier,
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="correction" if changed_id is not None else "reindex",
        index_state_sha256=inputs.index_state_sha256,
        canonical_before_sha256=publication_digest(
            [row.model_dump(mode="json") for row in inputs.canonical]
        ),
        canonical_after=after,
        indexes=indexes,
        previous_binding_ids=[
            binding.id
            for binding in inputs.bindings
            if binding.index.index_uuid in {index.index_uuid for index in indexes}
        ],
        bindings=bindings,
        views=views,
        amendment_impacts=impacts,
        canonical_revisions=revisions,
    )
