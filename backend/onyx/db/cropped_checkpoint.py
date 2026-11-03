"""Lossless pooling of Cropped originals, receipts, and native audit history."""

from pydantic import JsonValue

from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
)
from onyx.db.checkpoint_dictionary import (
    compact_checkpoint_dictionary,
    restore_checkpoint_dictionary,
)

CROPPED_CHECKPOINT_STORAGE_KEY = "asv3_cropped_checkpoint_storage"
MAX_CROPPED_EXPANDED_BYTES = 64_000_000
MAX_CROPPED_DEPTH = 64
_CROPPED_VARIANTS = frozenset({ASV3_EXP_CROPPED_VARIANT, ASV3_FAST_CROPPED_VARIANT})


def _variant(snapshot: dict[str, JsonValue]) -> str:
    variant = snapshot.get("asv3_workflow_variant")
    if not isinstance(variant, str) or variant not in _CROPPED_VARIANTS:
        raise ValueError("Cropped checkpoint requires its own workflow identity")
    return variant


def compact_cropped_checkpoint(snapshot: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return compact_checkpoint_dictionary(
        snapshot,
        storage_key=CROPPED_CHECKPOINT_STORAGE_KEY,
        variant=_variant(snapshot),
        label="Cropped",
        max_expanded_bytes=MAX_CROPPED_EXPANDED_BYTES,
        max_depth=MAX_CROPPED_DEPTH,
        use_if_smaller=True,
    )


def restore_cropped_checkpoint(snapshot: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if CROPPED_CHECKPOINT_STORAGE_KEY not in snapshot:
        return snapshot
    return restore_checkpoint_dictionary(
        snapshot,
        storage_key=CROPPED_CHECKPOINT_STORAGE_KEY,
        variant=_variant(snapshot),
        label="Cropped",
        max_expanded_bytes=MAX_CROPPED_EXPANDED_BYTES,
        max_depth=MAX_CROPPED_DEPTH,
    )
