import copy
import json

import pytest
from pydantic import JsonValue

from onyx.db.asv3_runs import (
    MAX_CHECKPOINT_BYTES,
    MAX_DECODED_CHECKPOINT_BYTES,
    decode_asv3_checkpoint,
    encode_asv3_checkpoint,
)
from onyx.db.legal_review_checkpoint import (
    STORAGE_KEY,
    compact_legal_review_checkpoint,
    restore_legal_review_checkpoint,
)


def large_checkpoint() -> dict[str, JsonValue]:
    original = "Özgün hüküm; koşul ve istisna aynen korunur. " * 45
    return {
        "version": 1,
        "asv3_workflow_variant": "legal_review",
        "run_id": "owned-run",
        "sequence": 2,
        "legal_review": {
            "answer": "Koşullu sonuç [659].",
            "editor_verification_passed": True,
        },
        "evidence": {
            "records": [
                {"citation": index, "item": {"text": original, "chunk_id": str(index)}}
                for index in range(1, 660)
            ],
            "included": [659],
        },
        "candidate_audit": {
            "records": [
                {
                    "candidate_id": f"candidate-{index}",
                    "source_id": f"source-{index % 20}",
                    "hydration_locator": {
                        "semantic_identifier": original,
                        "document_id": f"source-{index % 20}",
                    },
                    "status": "excluded" if index % 2 else "delivered",
                    "normalized_score": 0.82,
                }
                for index in range(4000)
            ]
        },
        "progress": [{"sequence": 4, "phase": "completed"}],
        "literal_json": {"o": [[0, "a"]], "a": [3], "s": 1},
    }


def test_large_legal_review_preserves_every_source_audit_and_current_answer() -> None:
    snapshot = large_checkpoint()
    before = copy.deepcopy(snapshot)
    assert (
        len(json.dumps(snapshot, ensure_ascii=False).encode())
        > MAX_DECODED_CHECKPOINT_BYTES
    )
    payload = encode_asv3_checkpoint(snapshot)
    assert len(payload.encode()) < MAX_CHECKPOINT_BYTES
    restored = decode_asv3_checkpoint(payload)
    assert restored == before == snapshot
    evidence = restored["evidence"]
    review = restored["legal_review"]
    assert isinstance(evidence, dict) and isinstance(review, dict)
    records = evidence["records"]
    assert isinstance(records, list) and isinstance(records[-1], dict)
    assert records[-1]["citation"] == 659
    assert review["editor_verification_passed"] is True


def test_other_workflow_keeps_its_existing_checkpoint_size_guard() -> None:
    snapshot = large_checkpoint()
    snapshot["asv3_workflow_variant"] = "standard"
    with pytest.raises(ValueError, match="decoded payload budget"):
        encode_asv3_checkpoint(snapshot)


@pytest.mark.parametrize("tampering", ["reference", "dictionary", "size", "identity"])
def test_changed_storage_cannot_restore_as_original_evidence(tampering: str) -> None:
    packed = compact_legal_review_checkpoint(large_checkpoint())
    storage = packed[STORAGE_KEY]
    assert isinstance(storage, dict)
    if tampering == "reference":
        storage["value"] = [-1]
    elif tampering == "dictionary":
        strings = storage["strings"]
        assert isinstance(strings, list)
        strings[0] = "changed source"
    elif tampering == "size":
        storage["expanded_bytes"] = 1
    else:
        packed["asv3_workflow_variant"] = "standard"
    with pytest.raises(ValueError):
        restore_legal_review_checkpoint(packed)


def test_expansion_is_bounded_before_constructing_a_dictionary_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = compact_legal_review_checkpoint(large_checkpoint())
    storage = packed[STORAGE_KEY]
    assert isinstance(storage, dict)
    storage["expanded_bytes"] = 100
    monkeypatch.setattr("onyx.db.legal_review_checkpoint.MAX_EXPANDED_BYTES", 1000)
    with pytest.raises(ValueError, match="expanded storage budget"):
        restore_legal_review_checkpoint(packed)
