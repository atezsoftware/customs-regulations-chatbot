"""Allocate prospective canonical identities without mutating live legal history."""

import hashlib
from datetime import date
from typing import TYPE_CHECKING
from uuid import uuid4

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeItemDraft,
    AnnexComparison,
    AnnexPatchPlan,
)

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk


def stage_canonical_items(
    *,
    plan: AnnexPatchPlan,
    baseline_scope: list[AnnexCanonicalSnapshot],
    insertion_after_chunk_id: str | None = None,
    comparison: AnnexComparison | None = None,
    prospective_ids: list[str] | None = None,
) -> list[AnnexChangeItemDraft]:
    if (
        comparison is not None
        and plan.comparison_sha256
        != hashlib.sha256(comparison.model_dump_json().encode()).hexdigest()
    ):
        raise ValueError("comparison identity changed")
    if not plan.ready or plan.effective_date is None:
        raise ValueError("patch plan is not ready")
    if not baseline_scope or len({row.user_file_id for row in baseline_scope}) != 1:
        raise ValueError("canonical scope is missing or ambiguous")
    rows = {row.id: row for row in baseline_scope}
    if insertion_after_chunk_id is not None and insertion_after_chunk_id not in rows:
        raise ValueError("insertion anchor outside canonical scope")
    next_ordinal = max(row.projection_ordinal for row in baseline_scope) + 1
    items: list[AnnexChangeItemDraft] = []
    expected_count = sum(patch.operation != "remove" for patch in plan.patches)
    if prospective_ids is not None and (
        len(prospective_ids) != expected_count
        or len(set(prospective_ids)) != expected_count
    ):
        raise ValueError("prospective identity count mismatch")
    identifiers = iter(prospective_ids) if prospective_ids is not None else None
    for patch in plan.patches:
        old = rows.get(patch.old_chunk_id) if patch.old_chunk_id else None
        if patch.old_chunk_id and (old is None or old.text != patch.old_text):
            raise ValueError("canonical patch scope changed")
        new_chunks: list[AnnexCanonicalSnapshot] = []
        if patch.operation != "remove":
            if not patch.new_text or not patch.new_text.strip():
                raise ValueError("empty canonical replacement")
            if (
                old is None
                and insertion_after_chunk_id is None
                and len(baseline_scope) != 1
            ):
                raise ValueError("insertion anchor is required")
            template = (
                old or rows[insertion_after_chunk_id]
                if insertion_after_chunk_id is not None
                else old or baseline_scope[0]
            )
            new_chunks.append(
                template.model_copy(
                    update={
                        "id": next(identifiers)
                        if identifiers is not None
                        else str(uuid4()),
                        "text": patch.new_text,
                        "source": "amendment",
                        "status": "active",
                        "projection_ordinal": next_ordinal,
                        "supersedes_chunk_id": old.id if old else None,
                        "superseded_by_chunk_id": None,
                        "validity_start_date": plan.effective_date,
                        "validity_end_date": old.validity_end_date if old else None,
                    }
                )
            )
            next_ordinal += 1
        items.append(
            AnnexChangeItemDraft(
                insertion_after_chunk_id=(
                    insertion_after_chunk_id or baseline_scope[0].id
                )
                if old is None
                else None,
                operation=patch.operation,
                old_chunk_ids=[old.id] if old else [],
                new_chunks=new_chunks,
                old_positions=patch.old_positions,
                new_positions=patch.new_positions,
            )
        )
    if comparison is None:
        return items
    groups: list[set[int]] = [{index} for index in range(len(items))]
    for change in comparison.changes:
        if change.operation not in ("split", "merge"):
            continue
        old_positions = {reference.position for reference in change.old}
        new_positions = {reference.position for reference in change.new}
        touched = {
            index
            for index, item in enumerate(items)
            if old_positions.intersection(item.old_positions)
            or new_positions.intersection(item.new_positions)
        }
        joined: set[int] = set()
        for group in groups:
            if group.intersection(touched):
                joined.update(group)
        groups = [group for group in groups if not group.intersection(touched)]
        if joined:
            groups.append(joined)
    grouped: list[AnnexChangeItemDraft] = []
    for group in sorted(groups, key=min):
        members = [items[index] for index in sorted(group)]
        old_positions = sorted(
            {position for item in members for position in item.old_positions}
        )
        new_positions = sorted(
            {position for item in members for position in item.new_positions}
        )
        kinds = {
            change.operation
            for change in comparison.changes
            if change.operation in ("split", "merge")
            and (
                {reference.position for reference in change.old}.intersection(
                    old_positions
                )
                or {reference.position for reference in change.new}.intersection(
                    new_positions
                )
            )
        }
        operation = members[0].operation
        if kinds == {"merge"}:
            operation = "merge"
        elif kinds == {"split"}:
            operation = "split"
        grouped.append(
            AnnexChangeItemDraft(
                operation=operation,
                old_chunk_ids=[
                    chunk_id for item in members for chunk_id in item.old_chunk_ids
                ],
                new_chunks=[chunk for item in members for chunk in item.new_chunks],
                old_positions=old_positions,
                new_positions=new_positions,
                insertion_after_chunk_id=members[0].insertion_after_chunk_id,
            )
        )
    return grouped


def validate_staged_items(
    *,
    plan: AnnexPatchPlan,
    baseline_scope: list[AnnexCanonicalSnapshot],
    items: list[AnnexChangeItemDraft],
    comparison: AnnexComparison | None = None,
    insertion_after_chunk_id: str | None = None,
) -> None:
    """Reconstruct each reviewed operation while retaining allocated identities."""
    chunks = sorted(
        (chunk for item in items for chunk in item.new_chunks),
        key=lambda chunk: chunk.projection_ordinal,
    )
    expected = stage_canonical_items(
        plan=plan,
        baseline_scope=baseline_scope,
        comparison=comparison,
        insertion_after_chunk_id=insertion_after_chunk_id,
        prospective_ids=[chunk.id for chunk in chunks],
    )
    if expected != items:
        raise ValueError("staged operation differs from reviewed canonical patch")


def canonical_snapshot_rows(
    snapshots: list[AnnexCanonicalSnapshot],
) -> list["RegulatoryChunk"]:
    from uuid import UUID

    from onyx.db.models import RegulatoryChunk

    return [
        RegulatoryChunk(
            id=row.id,
            user_file_id=UUID(row.user_file_id),
            chunk_type=row.chunk_type,
            status=row.status,
            projection_ordinal=row.projection_ordinal,
            supersedes_chunk_id=row.supersedes_chunk_id,
            superseded_by_chunk_id=row.superseded_by_chunk_id,
            position=row.position,
            text=row.text,
            heading_path=list(row.heading_path),
            chunk_metadata=dict(row.metadata),
            source=row.source,
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
        )
        for row in snapshots
    ]


def prepare_staged_candidate_rows(
    *,
    baseline_scope: list[AnnexCanonicalSnapshot],
    items: list[AnnexChangeItemDraft],
    effective_date: "date",
) -> list["RegulatoryChunk"]:
    """Build the complete candidate, preserving history and explicit insertion order."""
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        rebuild_context_aggregates,
    )
    from onyx.regulatory.contextual import validity_window_contains

    replacements = {
        old_id: [chunk.id for chunk in item.new_chunks]
        for item in items
        for old_id in item.old_chunk_ids
    }
    snapshots = [
        row.model_copy(
            update={
                "validity_end_date": min(row.validity_end_date, effective_date)
                if row.validity_end_date
                else effective_date,
                "status": "superseded",
            }
        )
        if row.id in replacements
        else row
        for row in baseline_scope
    ]
    snapshots.extend(chunk for item in items for chunk in item.new_chunks)
    rows = canonical_snapshot_rows(snapshots)
    by_id = {row.id: row for row in rows}
    insertions = [item for item in items if not item.old_chunk_ids]
    for item in reversed(insertions):
        anchor = by_id.get(item.insertion_after_chunk_id or "")
        if anchor is None:
            raise ValueError("candidate insertion anchor missing")
        for row in rows:
            if row.position > anchor.position:
                row.position += len(item.new_chunks)
        for offset, chunk in enumerate(item.new_chunks, 1):
            by_id[chunk.id].position = anchor.position + offset
    retired = {
        identifier
        for identifier, targets in replacements.items()
        if not targets and identifier in by_id
    }
    pending_aggregates = [
        row
        for row in rows
        if row.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate"
        and validity_window_contains(
            row.validity_start_date, row.validity_end_date, effective_date
        )
    ]
    while pending_aggregates:
        newly_retired: set[str] = set()
        for row in pending_aggregates:
            sources = row.chunk_metadata.get("source_regulatory_chunk_ids")
            if (
                not isinstance(sources, list)
                or not sources
                or not all(
                    isinstance(source, str) and source in retired for source in sources
                )
            ):
                continue
            root = row.chunk_metadata.get("hierarchy_root_path")
            if not isinstance(root, list) or not root or not isinstance(root[-1], str):
                raise ValueError("aggregate_provenance_unavailable")
            # Keep the historical text and source bindings before any rewriting.
            row.validity_end_date = effective_date
            row.status = "superseded"
            replacements[row.id] = []
            newly_retired.add(row.id)
        if not newly_retired:
            break
        retired.update(newly_retired)
        pending_aggregates = [
            row for row in pending_aggregates if row.id not in newly_retired
        ]
    for row in rows:
        if not validity_window_contains(
            row.validity_start_date, row.validity_end_date, effective_date
        ):
            continue
        metadata = dict(row.chunk_metadata)
        sources = metadata.get("source_regulatory_chunk_ids")
        if isinstance(sources, list):
            if not all(isinstance(source, str) for source in sources):
                raise ValueError("invalid candidate source references")
            metadata["source_regulatory_chunk_ids"] = list(
                dict.fromkeys(
                    new_id
                    for source in sources
                    for new_id in replacements.get(source, [source])
                )
            )
        binding = metadata.get("bound_to_regulatory_chunk_id")
        if isinstance(binding, str) and binding in replacements:
            targets = replacements[binding]
            if len(targets) != 1:
                raise ValueError("candidate image binding is ambiguous")
            metadata["bound_to_regulatory_chunk_id"] = targets[0]
        row.chunk_metadata = metadata
    changed = [chunk.id for item in items for chunk in item.new_chunks]
    effective = [
        row
        for row in rows
        if validity_window_contains(
            row.validity_start_date, row.validity_end_date, effective_date
        )
    ]
    aggregate_roots = [
        row.id
        for row in effective
        if row.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate"
    ]
    rebuilt = {
        row.id: row
        for row in rebuild_context_aggregates(
            effective, changed_ids=[*changed, *aggregate_roots]
        )
    }
    return sorted(
        [rebuilt.get(row.id, row) for row in rows],
        key=lambda row: (row.position, row.id),
    )
