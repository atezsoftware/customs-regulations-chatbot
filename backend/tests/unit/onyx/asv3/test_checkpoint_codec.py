import base64
import json
import zlib

import pytest
from pydantic import JsonValue

from onyx.db.asv3_runs import (
    MAX_DECODED_CHECKPOINT_BYTES,
    decode_asv3_checkpoint,
    encode_asv3_checkpoint,
)


def test_large_original_evidence_roundtrip_without_replicating_binary_payload() -> None:
    snapshot: dict[str, JsonValue] = {
        "version": 1,
        "run_id": "run",
        "sequence": 9,
        "evidence": "Özgün madde metni. " * 100000,
    }
    encoded = encode_asv3_checkpoint(snapshot)
    assert len(encoded) < 100000
    assert decode_asv3_checkpoint(encoded) == snapshot


def test_decoder_rejects_oversized_compressed_evidence() -> None:
    encoded = base64.b64encode(
        zlib.compress(b"x" * (MAX_DECODED_CHECKPOINT_BYTES + 1))
    ).decode()
    with pytest.raises(ValueError, match="oversized"):
        decode_asv3_checkpoint(
            json.dumps({"version": 1, "encoding": "zlib-base64", "data": encoded})
        )
