"""Uncitable parent-family candidates; finalists need the public original reader."""

import json
from collections.abc import Sequence
from datetime import date
from time import perf_counter
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    Integer,
    Text,
    and_,
    cast,
    column,
    func,
    literal_column,
    or_,
    select,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from onyx.db.asv3_candidate_inventory import current_asv3_source_inventory_scope
from onyx.db.models import RegulatoryChunk, RegulatoryTemporalProjection
from onyx.db.regulatory_chunks import RegulatoryChunkProjection
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory.publication_reads import (
    observe_publication_read,
    require_publication_files,
)
from onyx.tracing.answer_graph import graph_step


def get_asv3_initial_rerank_context(
    session: Session,
    seed_ids: Sequence[str],
    *,
    indexes: dict[UUID, PublicationIndexSnapshot],
    as_of_date: date,
) -> dict[str, tuple[RegulatoryChunkProjection, ...]]:
    """Select ALL frozen immediate-parent siblings as scoring/navigation leads.

    Read only compact fields, not full revision/vector payloads. This stage grants
    neither citations nor legal completeness. ASv3 hydrates selected search centers
    through its ACL/date/index/immutable-publication-fenced original reader before
    adding them to the evidence ledger.
    """
    scope = current_asv3_source_inventory_scope()
    if scope is None:
        raise ValueError("Initial context requires an explicit ASv3 scope")
    scope.check_active()
    with scope.lock:
        for file_id, index in indexes.items():
            previous = scope.query_indexes.get(file_id)
            if previous is not None and not previous.matches_temporal_index(index):
                raise ValueError(
                    "ASv3 source physical index changed during one invocation"
                )
            scope.query_indexes[file_id] = index
    unique_ids = tuple(dict.fromkeys(seed_ids))
    if not unique_ids:
        return {}
    if not indexes:
        raise ValueError("ASv3 parent selection requires pinned physical indexes")
    observation = observe_publication_read()
    require_publication_files(observation, tuple(indexes))
    projection, canonical = RegulatoryTemporalProjection, RegulatoryChunk
    # This expression matches the frozen-parent index; never use mutable headings.
    headings = cast(
        projection.payload.op("#>>")(literal_column("'{projection,source_json}'")),
        JSONB,
    ).op("->", return_type=JSONB)(literal_column("'heading_path'"))
    parent = headings.op("-", return_type=JSONB)(literal_column("-1"))
    parent_hash = func.md5(cast(parent, Text))
    qualified = (
        or_(
            *[
                and_(
                    projection.user_file_id == file_id,
                    projection.index_uuid == index.index_uuid,
                )
                for file_id, index in indexes.items()
            ]
        ),
        projection.retired_at.is_(None),
        or_(
            projection.effective_start.is_(None),
            projection.effective_start <= as_of_date,
        ),
        or_(projection.effective_end.is_(None), projection.effective_end > as_of_date),
        or_(
            canonical.validity_start_date.is_(None),
            canonical.validity_start_date <= as_of_date,
        ),
        or_(
            canonical.validity_end_date.is_(None),
            canonical.validity_end_date > as_of_date,
        ),
    )
    canonical_join = and_(
        canonical.id == projection.canonical_chunk_id,
        canonical.user_file_id == projection.user_file_id,
    )
    seed_query = (
        select(
            projection.id.label("binding_id"),
            projection.canonical_chunk_id.label("chunk_id"),
            projection.user_file_id.label("file_id"),
            projection.index_uuid.label("index_uuid"),
            parent.label("parent"),
            parent_hash.label("parent_hash"),
            (func.jsonb_array_length(headings) > 0).label("parent_known"),
        )
        .join(canonical, canonical_join)
        .where(*qualified, projection.canonical_chunk_id.in_(unique_ids))
        .execution_options(stream_results=True, yield_per=128)
    )
    with graph_step(
        "asv3.initial_packet_context",
        {
            "seed_count": len(unique_ids),
            "context_policy": "all_immediate_parent_siblings",
            "citable": False,
            "complete_operative_unit": False,
        },
    ) as span:
        leads: dict[str, list[UUID]] = {}
        started = perf_counter()
        # Retain only selected parent locators, not an inventory of the source.
        families: dict[tuple[UUID, str, str], list[tuple[str, JsonValue]]] = {}
        result = session.execute(seed_query)
        try:
            for row in result:
                scope.check_active()
                leads.setdefault(row.chunk_id, [])
                if not row.parent_known:
                    leads[row.chunk_id].append(row.binding_id)
                    continue
                key = (row.file_id, row.index_uuid, row.parent_hash)
                families.setdefault(key, []).append((row.chunk_id, row.parent))
        finally:
            result.close()
        if set(leads) != set(unique_ids):
            raise ValueError("ASv3 initial context lost a selected seed")
        if families:
            family_query = (
                select(
                    projection.id.label("binding_id"),
                    projection.user_file_id.label("file_id"),
                    projection.index_uuid,
                    parent.label("parent"),
                    parent_hash.label("parent_hash"),
                    (func.jsonb_array_length(headings) > 0).label("parent_known"),
                )
                .join(canonical, canonical_join)
                .where(
                    *qualified,
                    or_(
                        *[
                            and_(
                                projection.user_file_id == file_id,
                                projection.index_uuid == index_uuid,
                                parent_hash == digest,
                            )
                            for file_id, index_uuid, digest in families
                        ]
                    ),
                )
                .order_by(projection.projection_ordinal, projection.canonical_chunk_id)
                .execution_options(stream_results=True, yield_per=128)
            )
            result = session.execute(family_query)
            try:
                for row in result:
                    scope.check_active()
                    key = (row.file_id, row.index_uuid, row.parent_hash)
                    # A hash only locates candidates; exact equality defines membership.
                    for seed_id, seed_parent in families.get(key, []):
                        if row.parent_known and row.parent == seed_parent:
                            leads[seed_id].append(row.binding_id)
            finally:
                result.close()
        leads = {key: list(dict.fromkeys(ids)) for key, ids in leads.items()}
        if any(not ids for ids in leads.values()):
            raise ValueError("ASv3 initial context lost a selected parent family")
        lead_seconds = perf_counter() - started
        selected_ids = tuple(
            dict.fromkeys(member for ids in leads.values() for member in ids)
        )
        binding_fields = (
            func.jsonb_to_record(projection.payload)
            .table_valued(
                column("semantic_position", Integer),
                column("representation_text", Text),
                column("index", JSONB),
                column("projection", JSONB),
            )
            .render_derived(with_types=True)
            .lateral("scoring_binding")
        )
        projection_fields = (
            func.jsonb_to_record(binding_fields.c.projection)
            .table_valued(
                column("source_json", Text),
                column("embedding_config_json", Text),
                column("observed_index", JSONB),
            )
            .render_derived(with_types=True)
            .lateral("scoring_projection")
        )
        scoring_fields = (
            func.json_to_record(cast(projection_fields.c.source_json, JSON))
            .table_valued(
                column("regulatory_chunk_id", Text),
                column("document_id", Text),
                column("heading_path", JSON),
                column("image_file_id", Text),
                column("article_no", Text),
            )
            .render_derived(with_types=True)
            .lateral("scoring_source")
        )
        scoring_query = (
            select(
                projection.id,
                projection.canonical_chunk_id,
                projection.user_file_id,
                projection.projection_ordinal,
                projection.effective_start,
                projection.effective_end,
                binding_fields.c.semantic_position,
                binding_fields.c.representation_text,
                binding_fields.c.index.label("binding_index"),
                projection_fields.c.embedding_config_json,
                projection_fields.c.observed_index,
                scoring_fields.c.regulatory_chunk_id,
                scoring_fields.c.document_id,
                scoring_fields.c.heading_path,
                scoring_fields.c.image_file_id,
                scoring_fields.c.article_no,
            )
            .join(canonical, canonical_join)
            .join(binding_fields, true())
            .join(projection_fields, true())
            .join(scoring_fields, true())
            .where(*qualified, projection.id.in_(selected_ids))
            .execution_options(stream_results=True, yield_per=128)
        )
        candidates: dict[UUID, RegulatoryChunkProjection] = {}
        snapshots: dict[str, PublicationIndexSnapshot] = {}
        accepted_receipts: set[tuple[UUID, str]] = set()
        started = perf_counter()
        result = session.execute(scoring_query)
        try:
            for row in result:
                scope.check_active()
                index = indexes[row.user_file_id]
                key = json.dumps(row.binding_index, sort_keys=True)
                if key not in snapshots:
                    snapshots[key] = PublicationIndexSnapshot.model_validate(
                        row.binding_index
                    )
                if not snapshots[key].matches_temporal_index(index):
                    raise ValueError("ASv3 planning binding differs from pinned index")
                receipt = row.embedding_config_json
                receipt_key = (
                    row.user_file_id,
                    receipt or json.dumps(row.observed_index, sort_keys=True),
                )
                if receipt_key not in accepted_receipts:
                    if receipt is not None:
                        if not index.accepts_encoder_configuration(json.loads(receipt)):
                            raise ValueError(
                                "temporal binding encoder receipt is not accepted"
                            )
                    elif not PublicationIndexSnapshot.model_validate(
                        row.observed_index
                    ).matches_temporal_index(index):
                        raise ValueError("temporal observation index is not accepted")
                    accepted_receipts.add(receipt_key)
                if (
                    row.regulatory_chunk_id != row.canonical_chunk_id
                    or row.document_id != str(row.user_file_id)
                ):
                    raise ValueError("ASv3 planning source identity mismatch")
                candidates[row.id] = RegulatoryChunkProjection(
                    regulatory_chunk_id=row.canonical_chunk_id,
                    user_file_id=row.user_file_id,
                    projection_index=row.projection_ordinal,
                    position=row.semantic_position,
                    text=row.representation_text,
                    heading_path=tuple(row.heading_path or ()),
                    article_no=row.article_no,
                    status="planning",
                    validity_start_date=row.effective_start,
                    validity_end_date=row.effective_end,
                    image_file_id=row.image_file_id,
                )
        finally:
            result.close()
        selected_seconds = perf_counter() - started
        groups: dict[str, tuple[RegulatoryChunkProjection, ...]] = {}
        for seed_id, ids in leads.items():
            family = [
                candidates[binding_id] for binding_id in ids if binding_id in candidates
            ]
            seed = next(
                (member for member in family if member.regulatory_chunk_id == seed_id),
                None,
            )
            if seed is None or len(family) != len(ids):
                raise ValueError("ASv3 parent selection changed during scoring read")
            if any(
                member.regulatory_chunk_id != seed_id
                and (
                    not seed.heading_path
                    or member.heading_path[:-1] != seed.heading_path[:-1]
                )
                for member in family
            ):
                raise ValueError("ASv3 sibling metadata differs from selected parent")
            groups[seed_id] = tuple(
                sorted(
                    family,
                    key=lambda item: (
                        item.position,
                        item.projection_index,
                        item.regulatory_chunk_id,
                    ),
                )
            )
        require_publication_files(observation, tuple(indexes))
        span.output_value = {
            "seed_count": len(groups),
            "lead_rows": sum(map(len, leads.values())),
            "selected_bindings": len(candidates),
            "validated_bindings": 0,
            "source_wide_inventory": False,
            "lead_query_seconds": lead_seconds,
            "selected_scoring_seconds": selected_seconds,
            "selected_text_chars": sum(len(item.text) for item in candidates.values()),
            "selected_source_json_bytes": 0,
            "citable": False,
            "original_validation": "selected_finalists_before_evidence_delivery",
        }
        return groups
