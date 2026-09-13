from types import SimpleNamespace
from typing import cast

from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_labeling import _is_atomic


def test_legacy_chunk_without_variant_or_dependencies_is_atomic() -> None:
    row = cast(
        RegulatoryChunk,
        SimpleNamespace(chunk_type="article", chunk_metadata={}),
    )

    assert _is_atomic(row)


def test_legacy_companion_with_binding_is_not_atomic() -> None:
    row = cast(
        RegulatoryChunk,
        SimpleNamespace(
            chunk_type="image",
            chunk_metadata={"bound_to_regulatory_chunk_id": "canonical"},
        ),
    )

    assert not _is_atomic(row)
