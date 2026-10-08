"""Exact receipt transport retains all flags, catalogues and JSON identities."""

import json
from copy import deepcopy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.supersearch.receipts import (
    RECEIPT_DATA_DECODER,
    RECEIPT_DATA_POOL_KEY,
    compact_receipt_data_payload,
    expand_receipt_data_payload,
)


def serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def receipts_payload() -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "sources": [
            {
                "source_id": f"canonical-pc-source-{number}",
                "name": "Kanun, karar, tebliğ ve koşullar " * 8,
            }
            for number in range(12)
        ],
        "has_more": True,
        "scan_truncated": False,
        "evidence_truncated": True,
        "subunit_verified": False,
        "unavailable_source_ids": ["not-readable"],
        "unhydrated_centers": [{"canonical_chunk_id": "unread-original"}],
        "next_position": 3,
        "next_offset": None,
        "future_quality_flag": {"nested": [True, 1, 1.0, False, 0, None, []]},
        "literal_reference_fields": {"data_ref": 9, "values": [0, 1]},
    }
    return {
        "request": "Özgün metinlerden bütün koşulları incele.",
        "original_evidence": [{"citation": 1, "text": "Tam özgün hüküm."}],
        "authority_dependencies": [{"discovery_gaps": ["Unresolved own law"]}],
        "receipts": [
            {
                "tool": "read_named_provision",
                "status": "ambiguous",
                "need_ids": ["governing", f"need-{number}"],
                "data": deepcopy(data),
                "citations": [1, number + 2],
                "host_arguments": {"source_name": "Kanun", "article": "143"},
            }
            for number in range(4)
        ],
    }


def test_pool_round_trip_retains_every_receipt_field_flag_and_order() -> None:
    payload = receipts_payload()
    before = serialized(payload)
    projection, pooled = compact_receipt_data_payload(payload)
    assert pooled
    assert serialized(expand_receipt_data_payload(projection)) == before
    assert serialized(payload) == before
    assert len(serialized(projection).encode()) + len(
        ("\n" + RECEIPT_DATA_DECODER).encode()
    ) < len(before.encode())
    assert projection["original_evidence"] == payload["original_evidence"]
    assert projection["authority_dependencies"] == payload["authority_dependencies"]
    pool = cast(dict[str, JsonValue], projection[RECEIPT_DATA_POOL_KEY])
    values = cast(list[JsonValue], pool["values"])
    assert len(values) == 1
    originals = cast(list[dict[str, JsonValue]], payload["receipts"])
    encoded = cast(list[dict[str, JsonValue]], projection["receipts"])
    assert values[0] == originals[0]["data"]
    assert all(row["data_ref"] == 0 for row in encoded)
    for original, row in zip(originals, encoded, strict=True):
        assert {key: value for key, value in original.items() if key != "data"} == {
            key: value for key, value in row.items() if key != "data_ref"
        }


def test_json_types_missing_fields_and_object_order_remain_distinct() -> None:
    payload = receipts_payload()
    rows = cast(list[JsonValue], payload["receipts"])
    values: list[JsonValue] = [
        True,
        1,
        1.0,
        False,
        0,
        0.0,
        None,
        "",
        [],
        {},
        {"first": 1, "second": 2},
        {"second": 2, "first": 1},
    ]
    for value in values:
        rows.extend({"data": deepcopy(value)} for _ in range(2))
    rows.extend([{"status": "not_found"}, {"data": "unique value"}])
    projection, pooled = compact_receipt_data_payload(payload)
    assert pooled
    assert serialized(expand_receipt_data_payload(projection)) == serialized(payload)
    encoded = cast(list[dict[str, JsonValue]], projection["receipts"])
    references = [row["data_ref"] for row in encoded[4:-2:2]]
    assert len(set(cast(list[int], references))) == len(values)
    assert encoded[-2] == {"status": "not_found"}
    assert encoded[-1] == {"data": "unique value"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"receipts": []},
        {"receipts": [{"data": {"unique": True}}]},
        {"receipts": [{"data": True}, {"data": True}]},
        {"receipts": [None]},
        {"receipts": [{"data_ref": 0}, {"data_ref": 0}]},
        {RECEIPT_DATA_POOL_KEY: {"existing": True}, **receipts_payload()},
    ],
)
def test_small_unique_or_reserved_payloads_keep_the_original_transport(
    payload: dict[str, JsonValue],
) -> None:
    projection, pooled = compact_receipt_data_payload(payload)
    assert not pooled
    assert projection is payload
    assert expand_receipt_data_payload(payload) is payload


@pytest.mark.parametrize("reference", [-1, True, 1, "0", None])
def test_invalid_references_cannot_decode_another_data_value(
    reference: JsonValue,
) -> None:
    projection, pooled = compact_receipt_data_payload(receipts_payload())
    assert pooled
    rows = cast(list[dict[str, JsonValue]], projection["receipts"])
    rows[0]["data_ref"] = reference
    with pytest.raises(ValueError, match="reference is out of range"):
        expand_receipt_data_payload(projection)


def test_malformed_encoded_receipts_fail_without_silent_field_loss() -> None:
    projection, pooled = compact_receipt_data_payload(receipts_payload())
    assert pooled
    rows = cast(list[dict[str, JsonValue]], projection["receipts"])
    rows[0]["data"] = "conflict"
    with pytest.raises(ValueError, match="cannot coexist"):
        expand_receipt_data_payload(projection)
    projection["receipts"] = [False]
    with pytest.raises(ValueError, match="object rows"):
        expand_receipt_data_payload(projection)
    projection["receipts"] = None
    with pytest.raises(ValueError, match="receipt rows"):
        expand_receipt_data_payload(projection)
