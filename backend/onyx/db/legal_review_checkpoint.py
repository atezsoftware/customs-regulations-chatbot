"""Lossless storage dictionary for Legal Review's repeated source and audit values."""

from pydantic import JsonValue

from onyx.db.checkpoint_dictionary import (
    compact_checkpoint_dictionary,
    restore_checkpoint_dictionary,
)

STORAGE_KEY = "legal_review_checkpoint_storage"
MAX_EXPANDED_BYTES = 64_000_000
MAX_DEPTH = 64


def compact_legal_review_checkpoint(
    snapshot: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    if STORAGE_KEY in snapshot:
        raise ValueError("Legal Review checkpoint is already compacted")
    if snapshot.get("asv3_workflow_variant") != "legal_review":
        return snapshot
    return compact_checkpoint_dictionary(
        snapshot,
        storage_key=STORAGE_KEY,
        variant="legal_review",
        label="Legal Review",
        max_expanded_bytes=MAX_EXPANDED_BYTES,
        max_depth=MAX_DEPTH,
    )


def restore_legal_review_checkpoint(
    snapshot: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return restore_checkpoint_dictionary(
        snapshot,
        storage_key=STORAGE_KEY,
        variant="legal_review",
        label="Legal Review",
        max_expanded_bytes=MAX_EXPANDED_BYTES,
        max_depth=MAX_DEPTH,
    )
