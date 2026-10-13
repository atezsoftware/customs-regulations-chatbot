"""Cropped storage headroom preserves originals and other workflow codec budgets."""

import base64
import json
import random
import zlib

import pytest
from pydantic import JsonValue

from onyx.asv3.workflow_variant import ASV3_EXP_CROPPED_VARIANT
from onyx.db import asv3_runs as codec


def snapshot(variant: str | None = ASV3_EXP_CROPPED_VARIANT) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"version": 1, "run_id": "run", "sequence": 34}
    if variant is not None:
        result["asv3_workflow_variant"] = variant
    return result


def compressed(raw: bytes, *, outer_variant: str | None = None) -> str:
    envelope: dict[str, JsonValue] = {
        "version": 1,
        "encoding": "zlib-base64",
        "data": base64.b64encode(zlib.compress(raw, level=3)).decode("ascii"),
    }
    if outer_variant is not None:
        envelope["asv3_workflow_variant"] = outer_variant
    return json.dumps(envelope)


def raw(snapshot: dict[str, JsonValue]) -> bytes:
    return json.dumps(snapshot, ensure_ascii=False).encode()


def test_cropped_limits_double_without_changing_existing_constants() -> None:
    assert codec.MAX_CHECKPOINT_BYTES == 2_500_000
    assert codec.MAX_DECODED_CHECKPOINT_BYTES == 8_000_000
    assert codec.MAX_CROPPED_CHECKPOINT_BYTES == 5_000_000
    assert codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES == 16_000_000


def test_actual_over_eight_mb_cropped_originals_roundtrip_exactly() -> None:
    saved = snapshot()
    saved["original_text"] = "Özgün hüküm; şart ve istisna. " * 270_000
    assert (
        codec.MAX_DECODED_CHECKPOINT_BYTES
        < len(raw(saved))
        < codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES
    )
    encoded = codec.encode_asv3_checkpoint(saved)
    assert json.loads(encoded)["version"] == 1
    assert codec.decode_asv3_checkpoint(encoded) == saved
    assert codec.decode_asv3_checkpoint(raw(saved).decode()) == saved


@pytest.mark.parametrize(
    "variant", [None, "standard", "asv3_tuned", "unknown", "ASV3_EXP_CROPPED"]
)
def test_ordinary_and_unknown_inner_variants_keep_eight_mb_limit(
    variant: str | None,
) -> None:
    saved = snapshot(variant)
    saved["literal"] = "x" * codec.MAX_DECODED_CHECKPOINT_BYTES
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.encode_asv3_checkpoint(saved)
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.decode_asv3_checkpoint(
            compressed(raw(saved), outer_variant=ASV3_EXP_CROPPED_VARIANT)
        )
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.decode_asv3_checkpoint(raw(saved).decode())


def test_cropped_durable_capacity_accepts_large_compressed_originals() -> None:
    saved = snapshot()
    saved["unique_original_bytes"] = random.Random(17).randbytes(2_000_000).hex()
    encoded = codec.encode_asv3_checkpoint(saved)
    assert (
        codec.MAX_CHECKPOINT_BYTES
        < len(encoded.encode())
        < codec.MAX_CROPPED_CHECKPOINT_BYTES
    )
    assert codec.decode_asv3_checkpoint(encoded) == saved
    ordinary = {**saved, "asv3_workflow_variant": "asv3_tuned"}
    with pytest.raises(ValueError, match="durable payload budget"):
        codec.encode_asv3_checkpoint(ordinary)
    # Historical non-Cropped reads did not enforce the encoder's durable cap.
    assert codec.decode_asv3_checkpoint(compressed(raw(ordinary))) == ordinary


@pytest.mark.parametrize("cropped", [False, True])
def test_exact_decoded_byte_boundary(
    cropped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codec, "MAX_DECODED_CHECKPOINT_BYTES", 1000)
    monkeypatch.setattr(codec, "MAX_CROPPED_DECODED_CHECKPOINT_BYTES", 2000)
    saved = snapshot(ASV3_EXP_CROPPED_VARIANT if cropped else "asv3_tuned")
    saved["literal"] = ""
    limit = 2000 if cropped else 1000
    saved["literal"] = "x" * (limit - len(raw(saved)))
    assert len(raw(saved)) == limit
    assert codec.decode_asv3_checkpoint(codec.encode_asv3_checkpoint(saved)) == saved
    assert codec.decode_asv3_checkpoint(raw(saved).decode()) == saved
    saved["literal"] += "x"
    with pytest.raises(ValueError, match="decoded payload budget"):
        codec.encode_asv3_checkpoint(saved)
    with pytest.raises(ValueError, match="oversized|decoded payload budget"):
        codec.decode_asv3_checkpoint(compressed(raw(saved)))


def test_cropped_durable_boundary_and_plain_legacy_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = snapshot()
    saved["literal"] = "x" * 3000
    encoded = codec.encode_asv3_checkpoint(saved)
    monkeypatch.setattr(codec, "MAX_CROPPED_CHECKPOINT_BYTES", len(encoded.encode()))
    assert codec.encode_asv3_checkpoint(saved) == encoded
    assert codec.decode_asv3_checkpoint(encoded) == saved
    monkeypatch.setattr(
        codec, "MAX_CROPPED_CHECKPOINT_BYTES", len(encoded.encode()) - 1
    )
    with pytest.raises(ValueError, match="durable payload budget"):
        codec.encode_asv3_checkpoint(saved)
    with pytest.raises(ValueError, match="durable payload budget"):
        codec.decode_asv3_checkpoint(encoded)
    assert codec.decode_asv3_checkpoint(raw(saved).decode()) == saved


def test_global_decompression_limit_remains_bounded_before_json() -> None:
    with pytest.raises(ValueError, match="oversized"):
        codec.decode_asv3_checkpoint(
            compressed(b"x" * (codec.MAX_CROPPED_DECODED_CHECKPOINT_BYTES + 1))
        )


@pytest.mark.parametrize(
    "mode", ["truncated", "trailing", "concatenated", "base64", "json", "array"]
)
def test_malformed_compressed_streams_cannot_enter_restore(mode: str) -> None:
    packed = zlib.compress(raw(snapshot()))
    if mode == "truncated":
        packed = packed[:-1]
    elif mode == "trailing":
        packed += b"trailing bytes"
    elif mode == "concatenated":
        packed += zlib.compress(raw(snapshot()))
    elif mode == "json":
        packed = zlib.compress(b"not json")
    elif mode == "array":
        packed = zlib.compress(b"[]")
    encoded = "bad base64!" if mode == "base64" else base64.b64encode(packed).decode()
    with pytest.raises(ValueError):
        codec.decode_asv3_checkpoint(
            json.dumps({"version": 1, "encoding": "zlib-base64", "data": encoded})
        )


def test_plain_non_object_is_rejected_as_invalid_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        codec.decode_asv3_checkpoint("[]")
