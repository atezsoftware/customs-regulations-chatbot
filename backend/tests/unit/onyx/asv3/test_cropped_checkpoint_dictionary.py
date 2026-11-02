"""Cropped checkpoint pooling preserves originals and every native audit field."""

import copy

import pytest
from pydantic import JsonValue

from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
)
from onyx.db import asv3_runs as codec
from onyx.db import cropped_checkpoint as storage
from tests.unit.onyx.asv3.test_cropped_checkpoint_limits import compressed, raw


def checkpoint(variant: str = ASV3_EXP_CROPPED_VARIANT) -> dict[str, JsonValue]:
    original = "Şart, istisna ve özgün hüküm aynen korunur. " * 160
    records: list[JsonValue] = [
        {
            "citation": index,
            "item": {
                "source_id": f"source-{index % 20}",
                "chunk_id": f"chunk-{index}",
                "text": original,
                "metadata": {"publication_id": "published-revision", "article": "17"},
            },
        }
        for index in range(1, 901)
    ]
    return {
        "version": 1,
        "asv3_workflow_variant": variant,
        "research_profile": "normal",
        "parallel_research": False,
        "execution_mode": "native",
        "run_id": "owned-run",
        "sequence": 12,
        "scope": {"tenant_id": "public", "asv3_document_set_id": 15},
        "evidence": {"records": records, "included": [900]},
        "receipts": [{"call": {"call_id": "read-1"}, "outcome": {"data": records}}],
        "turns": [{"native": {"source_passages": records, "call_id": "native-1"}}],
        "last_draft": "Kaynaklı koşullu değerlendirme [900].",
        "literal_json": {"o": [[0, "a"]], "a": [3], "s": 1},
    }


@pytest.mark.parametrize(
    "variant", [ASV3_EXP_CROPPED_VARIANT, ASV3_FAST_CROPPED_VARIANT]
)
def test_over_sixteen_mb_native_checkpoint_roundtrips_every_field(variant: str) -> None:
    saved = checkpoint(variant)
    before = copy.deepcopy(saved)
    assert len(raw(saved)) > codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES

    packed = storage.compact_cropped_checkpoint(saved)
    assert storage.CROPPED_CHECKPOINT_STORAGE_KEY in packed
    assert len(raw(packed)) < codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES
    encoded = codec.encode_asv3_checkpoint(saved)
    assert len(encoded.encode()) < codec.MAX_CROPPED_CHECKPOINT_BYTES
    restored = codec.decode_asv3_checkpoint(encoded)
    assert restored == saved == before
    assert restored["turns"] == saved["turns"]
    assert restored["receipts"] == saved["receipts"]


def test_irreducible_over_capacity_checkpoint_is_still_rejected() -> None:
    saved: dict[str, JsonValue] = {
        "version": 1,
        "asv3_workflow_variant": ASV3_EXP_CROPPED_VARIANT,
        "literal": "x" * codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES,
    }
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.encode_asv3_checkpoint(saved)


def test_small_and_legacy_native_checkpoints_keep_their_representation() -> None:
    saved: dict[str, JsonValue] = {
        "version": 1,
        "asv3_workflow_variant": ASV3_EXP_CROPPED_VARIANT,
        "sequence": 2,
    }
    assert storage.compact_cropped_checkpoint(saved) is saved
    assert codec.decode_asv3_checkpoint(raw(saved).decode()) == saved
    assert codec.decode_asv3_checkpoint(compressed(raw(saved))) == saved


@pytest.mark.parametrize(
    "tampering",
    [
        "reference",
        "dictionary",
        "unused",
        "size",
        "digest",
        "variant",
        "unknown",
        "envelope",
        "version",
    ],
)
def test_tampered_cropped_storage_cannot_restore_as_original(tampering: str) -> None:
    packed = storage.compact_cropped_checkpoint(checkpoint())
    body = packed[storage.CROPPED_CHECKPOINT_STORAGE_KEY]
    assert isinstance(body, dict)
    if tampering == "reference":
        body["value"] = [-1]
    elif tampering == "dictionary":
        texts = body["strings"]
        assert isinstance(texts, list)
        texts[0] = "changed source"
    elif tampering == "unused":
        texts = body["strings"]
        assert isinstance(texts, list)
        texts.append("unused source data")
    elif tampering == "size":
        body["expanded_bytes"] = 1
    elif tampering == "digest":
        body["sha256"] = "0" * 64
    elif tampering == "variant":
        packed["asv3_workflow_variant"] = ASV3_FAST_CROPPED_VARIANT
    elif tampering == "unknown":
        packed["asv3_workflow_variant"] = "standard"
    elif tampering == "version":
        body["version"] = True
    else:
        packed["extra"] = "unverified outside the dictionary"
    with pytest.raises(ValueError):
        codec.decode_asv3_checkpoint(compressed(raw(packed)))


def test_cropped_expansion_is_bounded_before_constructing_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = storage.compact_cropped_checkpoint(checkpoint())
    body = packed[storage.CROPPED_CHECKPOINT_STORAGE_KEY]
    assert isinstance(body, dict)
    body["expanded_bytes"] = 100
    monkeypatch.setattr(storage, "MAX_CROPPED_EXPANDED_BYTES", 1000)
    with pytest.raises(ValueError, match="expanded storage budget"):
        storage.restore_cropped_checkpoint(packed)


def test_dictionary_storage_remains_workflow_scoped() -> None:
    saved = checkpoint()
    saved["asv3_workflow_variant"] = "standard"
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.encode_asv3_checkpoint(saved)
    with pytest.raises(ValueError, match="workflow identity"):
        storage.compact_cropped_checkpoint(saved)


def test_cropped_raw_expansion_and_depth_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = checkpoint()
    monkeypatch.setattr(storage, "MAX_CROPPED_EXPANDED_BYTES", 1000)
    with pytest.raises(ValueError, match="expanded storage budget"):
        storage.compact_cropped_checkpoint(saved)
    monkeypatch.setattr(storage, "MAX_CROPPED_EXPANDED_BYTES", 64_000_000)
    monkeypatch.setattr(storage, "MAX_CROPPED_DEPTH", 2)
    with pytest.raises(ValueError, match="storage depth"):
        storage.compact_cropped_checkpoint(saved)
