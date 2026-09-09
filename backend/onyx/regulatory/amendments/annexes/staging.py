"""Allocate prospective canonical identities without mutating live legal history."""

import hashlib
from uuid import uuid4

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeItemDraft,
    AnnexComparison,
    AnnexPatchPlan,
)


def stage_canonical_items(
    *,
    plan: AnnexPatchPlan,
    baseline_scope: list[AnnexCanonicalSnapshot],
    insertion_after_chunk_id: str | None = None,
    comparison: AnnexComparison | None = None,
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
                        "id": str(uuid4()),
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
