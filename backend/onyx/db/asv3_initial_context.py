"""Bounded navigation leads with freshly verified, independently citable originals."""

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from time import perf_counter
from uuid import UUID

from sqlalchemy import and_, or_, select, true, union_all
from sqlalchemy.orm import Session, aliased

from onyx.db.asv3_candidate_inventory import current_asv3_source_inventory_scope
from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_chunks import (
    RegulatoryChunkProjection,
    _fold_text,
    _last_explicit_article_anchor,
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
    """Use current positions only as leads, never as historical completeness proof.

    At most two neighboring canonical rows on either side are inspected per seed.
    Immutable selected bindings decide dates, index identity, text and parent scope.
    Wider operative-unit reading remains a separate harness action.
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
    seeds = aliased(RegulatoryChunk, name="asv3_context_seed")
    neighbor = aliased(RegulatoryChunk, name="asv3_context_neighbor")
    seed_filter = and_(seeds.id.in_(unique_ids), seeds.user_file_id.in_(indexes))
    visible_neighbor = and_(
        neighbor.user_file_id == seeds.user_file_id,
        or_(
            neighbor.validity_start_date.is_(None),
            neighbor.validity_start_date <= as_of_date,
        ),
        or_(
            neighbor.validity_end_date.is_(None),
            neighbor.validity_end_date > as_of_date,
        ),
    )
    before = (
        select(neighbor.id.label("id"))
        .where(
            visible_neighbor,
            or_(
                neighbor.position < seeds.position,
                and_(neighbor.position == seeds.position, neighbor.id < seeds.id),
            ),
        )
        .order_by(neighbor.position.desc(), neighbor.id.desc())
        .limit(2)
        .correlate(seeds)
        .lateral("asv3_context_before")
    )
    after = (
        select(neighbor.id.label("id"))
        .where(
            visible_neighbor,
            or_(
                neighbor.position > seeds.position,
                and_(neighbor.position == seeds.position, neighbor.id > seeds.id),
            ),
        )
        .order_by(neighbor.position, neighbor.id)
        .limit(2)
        .correlate(seeds)
        .lateral("asv3_context_after")
    )
    query = union_all(
        select(
            seeds.id.label("seed_id"), seeds.user_file_id, seeds.id.label("member_id")
        ).where(seed_filter),
        select(seeds.id, seeds.user_file_id, before.c.id)
        .select_from(seeds)
        .join(before, true())
        .where(seed_filter),
        select(seeds.id, seeds.user_file_id, after.c.id)
        .select_from(seeds)
        .join(after, true())
        .where(seed_filter),
    )
    with graph_step(
        "asv3.initial_packet_context",
        {
            "seed_count": len(unique_ids),
            "neighbors_per_side": 2,
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
            seed_parent = _last_explicit_article_anchor(seed.heading_path)
            seed_parent_key = (
                seed_parent.key
                if seed_parent is not None
                else tuple(_fold_text(part) for part in seed.heading_path[:-1])
                if len(seed.heading_path) >= 2
                else None
            )
            members = []
            for member_id in dict.fromkeys(ids):
                member = selected.get(member_id)
                if member is None:
                    continue
                member_parent = _last_explicit_article_anchor(member.heading_path)
                member_parent_key = (
                    member_parent.key
                    if member_parent is not None
                    else tuple(_fold_text(part) for part in member.heading_path[:-1])
                    if len(member.heading_path) >= 2
                    else None
                )
                if (
                    seed_parent_key is not None
                    and member_parent_key is not None
                    and seed_parent_key != member_parent_key
                ):
                    continue
                # Missing parent identity is a local lead, not an asserted same-article match.
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
