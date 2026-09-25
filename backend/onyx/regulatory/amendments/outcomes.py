"""Conservative recognition of an unchanged, previously applied legal version."""

from typing import Any


def identical_applied_result(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if (
        before.get("source") != "amendment"
        or before.get("status") != "active"
        or not before.get("supersedes_chunk_id")
        or not before.get("validity_start_date")
        or before["validity_start_date"] != after.get("effective_start_date")
        or before.get("validity_end_date") != after.get("effective_end_date")
    ):
        return False
    if any(
        before.get(key) != after.get(key)
        for key in (
            "user_file_id",
            "position",
            "text",
            "chunk_type",
            "heading_path",
        )
    ):
        return False
    original = {
        key: value
        for key, value in (before.get("metadata") or {}).items()
        if value is not None
    }
    proposed = {
        key: value
        for key, value in (after.get("metadata") or {}).items()
        if value is not None
    }
    original.setdefault("heading_path", before.get("heading_path"))
    proposed.setdefault("heading_path", after.get("heading_path"))
    return original == proposed
