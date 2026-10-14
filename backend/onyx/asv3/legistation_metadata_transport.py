"""Conditional guidance for optional file validity annotations."""

from typing import cast

from onyx.regulatory.legistation_metadata import parse_legistation_metadata

LEGISTATION_METADATA_USAGE_RULE = (
    "When legistation_metadata is present, validity_status and repeal_note are "
    "file-level validity annotations, separate from the original legal passage. "
    "Do not call a repealed (mülga) instrument currently in force. Account for "
    "the repeal date and the requested date; repeal does not automatically "
    "invalidate earlier applicability. This annotation is not the original "
    "repealing instrument or proof of its other provisions."
)


def has_legistation_annotations(value: object) -> bool:
    """Inspect structured fields without parsing or changing source passage text."""
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            mapping = cast(dict[object, object], current)
            annotation = parse_legistation_metadata(mapping.get("legistation_metadata"))
            if annotation is not None and (
                (annotation.validity_status or "").strip()
                or (annotation.repeal_note or "").strip()
            ):
                return True
            pending.extend(
                item for item in mapping.values() if isinstance(item, (dict, list))
            )
        elif isinstance(current, list):
            pending.extend(item for item in current if isinstance(item, (dict, list)))
    return False
