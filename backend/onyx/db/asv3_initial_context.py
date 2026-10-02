"""Same-parent navigation with freshly verified, independently citable originals."""

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from time import perf_counter
from uuid import UUID

from sqlalchemy import and_, cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from onyx.db.asv3_candidate_inventory import current_asv3_source_inventory_scope
from onyx.db.models import RegulatoryChunk, RegulatoryTemporalProjection
from onyx.db.regulatory_chunks import (
    RegulatoryChunkProjection,
)
from onyx.db.regulatory_public_reads import iter_public_temporal_bindings
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
    """Use frozen parent metadata, never infer whole-provision completeness.

    Every sibling in the same immediate heading parent is selected from frozen
    metadata. Only that family's originals are hydrated and verified. No full-source
    inventory or positional window is used; wider reading remains a harness decision.
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
    observation = observe_publication_read()
    projection = RegulatoryTemporalProjection
    canonical = RegulatoryChunk
    headings = cast(projection.payload["projection"]["source_json"].astext, JSONB)[
        "heading_path"
    ]
    pinned_index = or_(
        *[
            and_(
                projection.user_file_id == file_id,
                projection.index_uuid == index.index_uuid,
            )
            for file_id, index in indexes.items()
        ]
    )
    # Materialize only parent locators once per binding, not once per seed join.
    # Frozen source_json may be large; no source body or vector is selected here.
    locators = (
        select(
            projection.id.label("binding_id"),
            projection.canonical_chunk_id.label("chunk_id"),
            projection.user_file_id.label("file_id"),
            projection.index_uuid.label("index_uuid"),
            projection.projection_ordinal.label("ordinal"),
            headings.op("-", return_type=JSONB)(-1).label("parent"),
            (func.jsonb_array_length(headings) > 0).label("parent_known"),
        )
        .join(
            canonical,
            and_(
                canonical.id == projection.canonical_chunk_id,
                canonical.user_file_id == projection.user_file_id,
            ),
        )
        .where(
            pinned_index,
            projection.retired_at.is_(None),
            or_(
                projection.effective_start.is_(None),
                projection.effective_start <= as_of_date,
            ),
            or_(
                projection.effective_end.is_(None),
                projection.effective_end > as_of_date,
            ),
            or_(
                canonical.validity_start_date.is_(None),
                canonical.validity_start_date <= as_of_date,
            ),
            or_(
                canonical.validity_end_date.is_(None),
                canonical.validity_end_date > as_of_date,
            ),
        )
        .cte("asv3_parent_locators")
        .prefix_with("MATERIALIZED")
    )
    seeds, members = (
        locators.alias("asv3_parent_seed"),
        locators.alias("asv3_parent_sibling"),
    )
    query = (
        select(seeds.c.chunk_id, seeds.c.file_id, members.c.chunk_id)
        .select_from(seeds)
        .join(
            members,
            and_(
                members.c.file_id == seeds.c.file_id,
                members.c.index_uuid == seeds.c.index_uuid,
                or_(
                    and_(
                        seeds.c.parent_known,
                        members.c.parent_known,
                        members.c.parent == seeds.c.parent,
                    ),
                    members.c.binding_id == seeds.c.binding_id,
                ),
            ),
        )
        .where(seeds.c.chunk_id.in_(unique_ids))
        .order_by(seeds.c.chunk_id, members.c.ordinal, members.c.chunk_id)
        .execution_options(yield_per=128)
    )
    with graph_step(
        "asv3.initial_packet_context",
        {
            "seed_count": len(unique_ids),
            "context_policy": "all_immediate_parent_siblings",
            "complete_operative_unit": False,
        },
    ) as span:
        leads: dict[str, list[str]] = {}
        files: dict[str, UUID] = {}
        started = perf_counter()
        for seed_id, file_id, member_id in session.execute(query):
            files[seed_id] = file_id
            leads.setdefault(seed_id, []).append(member_id)
        lead_seconds = perf_counter() - started
        require_publication_files(observation, tuple(set(files.values())))
        selected: dict[str, RegulatoryChunkProjection] = {}
        validated_bindings = 0
        original_bytes = 0
        started = perf_counter()
        for file_id in dict.fromkeys(files.values()):
            scope.check_active()
            members = tuple(
                dict.fromkeys(
                    member
                    for seed, ids in leads.items()
                    if files[seed] == file_id
                    for member in ids
                )
            )
            for binding in iter_public_temporal_bindings(
                session,
                file_id,
                index=indexes[file_id],
                as_of_date=as_of_date,
                canonical_chunk_ids=members,
            ):
                validated_bindings += 1
                source_json = binding.projection.source_json
                original_bytes += len(source_json.encode())
                source = json.loads(source_json)
                chunk_id = source["regulatory_chunk_id"]
                if source["document_id"] != str(file_id) or chunk_id not in members:
                    raise ValueError("ASv3 initial context source identity mismatch")
                previous = selected.get(chunk_id)
                if previous is not None and (
                    previous.position,
                    previous.projection_index,
                ) > (binding.semantic_position, binding.projection.ordinal):
                    continue
                selected[chunk_id] = RegulatoryChunkProjection(
                    regulatory_chunk_id=chunk_id,
                    user_file_id=file_id,
                    projection_index=binding.projection.ordinal,
                    position=binding.semantic_position,
                    text=binding.representation_text,
                    heading_path=tuple(source.get("heading_path") or ()),
                    article_no=source.get("article_no"),
                    status="approved",
                    validity_start_date=binding.effective_start,
                    validity_end_date=binding.effective_end,
                    source_json=source_json,
                    image_file_id=source.get("image_file_id"),
                    publication_source_sha256=hashlib.sha256(
                        source_json.encode()
                    ).hexdigest(),
                )
        validation_seconds = perf_counter() - started
        result: dict[str, tuple[RegulatoryChunkProjection, ...]] = {}
        for seed_id, ids in leads.items():
            seed = selected.get(seed_id)
            if seed is None:
                raise ValueError("ASv3 initial packet seed is no longer available")
            members = []
            for member_id in dict.fromkeys(ids):
                member = selected.get(member_id)
                if member is None:
                    continue
                if member.regulatory_chunk_id != seed.regulatory_chunk_id and (
                    not seed.heading_path
                    or member.heading_path[:-1] != seed.heading_path[:-1]
                ):
                    raise ValueError(
                        "ASv3 sibling metadata differs from its verified parent"
                    )
                members.append(member)
            result[seed_id] = tuple(
                sorted(
                    members,
                    key=lambda item: (
                        item.position,
                        item.projection_index,
                        item.regulatory_chunk_id,
                    ),
                )
            )
        if set(leads) != set(unique_ids):
            raise ValueError("ASv3 initial context lost a selected seed")
        require_publication_files(observation, tuple(set(files.values())))
        scope.check_active()
        span.output_value = {
            "seed_count": len(result),
            "lead_rows": sum(map(len, leads.values())),
            "validated_bindings": validated_bindings,
            "source_wide_inventory": False,
            "lead_query_seconds": lead_seconds,
            "selected_validation_seconds": validation_seconds,
            "selected_source_json_bytes": original_bytes,
        }
        return result
