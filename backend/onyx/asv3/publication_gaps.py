"""Combine acquisition defects without another model assessment."""

from pydantic import JsonValue

from onyx.asv3.models import OutcomeStatus, ToolOutcome


def combine_source_publication_gaps(gaps: list[ToolOutcome]) -> ToolOutcome | None:
    """Return all computed source defects without losing acquisition-routing fields."""
    if not gaps:
        return None
    if len(gaps) == 1:
        return gaps[0]
    data: dict[str, JsonValue] = {}
    instructions: list[JsonValue] = []
    for gap in gaps:
        instructions.append(
            {"summary": gap.summary, "instruction": gap.data.get("instruction")}
        )
        for key, value in gap.data.items():
            if key == "instruction":
                continue
            if key in data and data[key] != value:
                raise ValueError("Conflicting publication check fields")
            data[key] = value
    data["publication_check_instructions"] = instructions
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Address every reported source defect together in the retained candidate; reuse delivered originals and preserve supported detail.",
        data=data,
    )
