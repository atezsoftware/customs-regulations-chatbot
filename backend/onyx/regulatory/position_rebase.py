"""Recognize numbering shifts that preserve all existing relative order."""

from uuid import uuid4

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)


def monotonic_position_map(
    before: list[AnnexCanonicalSnapshot], after: list[AnnexCanonicalSnapshot]
) -> dict[int, int]:
    desired = {row.id: row for row in after}
    if not {row.id for row in before}.issubset(desired):
        return {}
    mapping: dict[int, int] = {}
    for row in before:
        target = desired[row.id].position
        if row.position in mapping and mapping[row.position] != target:
            return {}
        mapping[row.position] = target
    positions = [mapping[position] for position in sorted(mapping)]
    if any(left >= right for left, right in zip(positions, positions[1:])):
        return {}
    return mapping


def position_only_changes(
    before: list[AnnexCanonicalSnapshot], after: list[AnnexCanonicalSnapshot]
) -> set[str]:
    mapping = monotonic_position_map(before, after)
    desired = {row.id: row for row in after}
    return {
        row.id
        for row in before
        if row.position in mapping
        and mapping[row.position] != row.position
        and row.model_copy(update={"position": mapping[row.position]})
        == desired[row.id]
    }


def rebase_binding_position(
    binding: AnnexTemporalProjection, *, previous_position: int, position: int
) -> AnnexTemporalProjection:
    if binding.semantic_position == position:
        return binding
    if binding.semantic_position != previous_position:
        raise ValueError(
            "Historical position cannot be rebased without matching source evidence"
        )
    identifier = uuid4()
    projection = binding.projection.model_copy(
        update={"context_projection_id": str(identifier)}
    )
    return binding.model_copy(
        update={
            "id": identifier,
            "projection": projection,
            "semantic_position": position,
        }
    )
