"""Lossless transport of repeated complete acquisition receipt data."""

from __future__ import annotations

import json
from collections import Counter
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

RECEIPT_DATA_POOL_KEY = "receipt_data"
RECEIPT_DATA_CODEC = "supersearch_receipts_v1"
RECEIPT_DATA_DECODER = """
Receipt data encoding: when receipt_data.codec is supersearch_receipts_v1, a
receipts row's data_ref indexes receipt_data.values and replaces its data field.
Each entry is the exact complete data value, including all source catalogues,
quality flags and continuation cursors. Decode it before assessing source gaps.
Rows with data keep it directly; missing data remains missing. Other receipt
fields remain explicit. References are table indexes, not citation/source IDs.
""".strip()


class _ReceiptDataPool(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    codec: Literal["supersearch_receipts_v1"]
    values: list[JsonValue]


def _serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_receipt_data_payload(
    payload: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], bool]:
    """Pool only exact repeated data when its projection plus decoder is smaller."""
    raw_receipts = payload.get("receipts")
    if (
        not isinstance(raw_receipts, list)
        or not raw_receipts
        or RECEIPT_DATA_POOL_KEY in payload
        or any(not isinstance(row, dict) or "data_ref" in row for row in raw_receipts)
    ):
        return payload, False
    receipts = cast(list[dict[str, JsonValue]], raw_receipts)
    signatures = Counter(_serialized(row["data"]) for row in receipts if "data" in row)
    if not any(count > 1 for count in signatures.values()):
        return payload, False
    values: list[JsonValue] = []
    references: dict[str, int] = {}
    projected_receipts: list[JsonValue] = []
    for receipt in receipts:
        signature = _serialized(receipt["data"]) if "data" in receipt else None
        if signature is None or signatures[signature] < 2:
            projected_receipts.append(dict(receipt))
            continue
        if signature not in references:
            references[signature] = len(values)
            # JSON identities distinguish bool/int/float, null and object order.
            values.append(json.loads(signature))
        projected_receipts.append(
            {
                "data_ref" if key == "data" else key: (
                    references[signature] if key == "data" else value
                )
                for key, value in receipt.items()
            }
        )
    projection: dict[str, JsonValue] = {
        **payload,
        "receipts": projected_receipts,
        RECEIPT_DATA_POOL_KEY: {"codec": RECEIPT_DATA_CODEC, "values": values},
    }
    if len(_serialized(projection).encode("utf-8")) + len(
        ("\n" + RECEIPT_DATA_DECODER).encode("utf-8")
    ) >= len(_serialized(payload).encode("utf-8")):
        return payload, False
    return projection, True


def expand_receipt_data_payload(
    projection: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Restore complete receipt data and field order for exact transport audits."""
    raw_pool = projection.get(RECEIPT_DATA_POOL_KEY)
    if not isinstance(raw_pool, dict) or raw_pool.get("codec") != RECEIPT_DATA_CODEC:
        return projection
    pool = _ReceiptDataPool.model_validate(raw_pool, strict=True)
    raw_receipts = projection.get("receipts")
    if not isinstance(raw_receipts, list):
        raise ValueError("Encoded receipt data must contain receipt rows")
    decoded_receipts: list[JsonValue] = []
    for receipt in raw_receipts:
        if not isinstance(receipt, dict):
            raise ValueError("Encoded receipt data must contain object rows")
        if "data_ref" not in receipt:
            decoded_receipts.append(dict(receipt))
            continue
        if "data" in receipt:
            raise ValueError("Receipt data and its reference cannot coexist")
        reference = receipt["data_ref"]
        if type(reference) is not int or not 0 <= reference < len(pool.values):
            raise ValueError("Receipt data reference is out of range")
        decoded_receipts.append(
            {
                "data" if key == "data_ref" else key: (
                    pool.values[reference] if key == "data_ref" else value
                )
                for key, value in receipt.items()
            }
        )
    return {
        key: decoded_receipts if key == "receipts" else value
        for key, value in projection.items()
        if key != RECEIPT_DATA_POOL_KEY
    }
