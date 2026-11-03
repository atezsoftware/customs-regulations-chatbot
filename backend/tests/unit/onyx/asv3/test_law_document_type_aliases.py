"""Existing statute type aliases keep identical original-source identity guards."""

from dataclasses import replace

import pytest

from onyx.db.asv3_corpus import _source_has_own_law_number
from tests.unit.onyx.asv3.test_numbered_law_identity_213 import _VUK, candidates
from tests.unit.onyx.regulatory.labeling.test_native_word_self_repeal import (
    proof as word_proof,
)
from tests.unit.onyx.regulatory.labeling.test_native_word_self_repeal import (
    rows as word_rows,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import proof, rows


@pytest.mark.parametrize(
    "kind", ["kanun", "Kanun", "law", "Law", "act", "ACT", "statute", "Statute"]
)
def test_existing_law_aliases_preserve_exact_own_original_proofs(kind: str) -> None:
    sources, heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    original = [
        replace(row, metadata={**row.metadata, "document_type": kind})
        for row in heads[_VUK]
    ]
    assert _source_has_own_law_number(source, original, "213")
    scope = rows()
    for row in scope:
        row.metadata["document_type"] = kind
    assert proof(scope=scope).document_number == "6001"
    native = word_rows()
    for row in native:
        row.metadata["document_type"] = kind
    assert word_proof(scope=native).document_number == "5393"


@pytest.mark.parametrize(
    "kind",
    [
        "court_decision",
        "judgment",
        "decree",
        "KHK",
        "regulation",
        "law?",
        "kanun / karar",
    ],
)
def test_nonlaw_types_never_gain_law_identity_or_source_only_disposition(
    kind: str,
) -> None:
    sources, heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    original = [
        replace(row, metadata={**row.metadata, "document_type": kind})
        for row in heads[_VUK]
    ]
    assert not _source_has_own_law_number(source, original, "213")
    scope = rows()
    for row in scope:
        row.metadata["document_type"] = kind
    with pytest.raises(ValueError, match="complete unconditional"):
        proof(scope=scope)
