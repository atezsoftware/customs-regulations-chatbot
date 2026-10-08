"""Lossless, flat metadata tables for selected-model provider payloads."""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

METADATA_POOL_KEY = "original_metadata"
METADATA_CODEC = "supersearch_metadata_v1"
METADATA_DECODER = """
Original metadata encoding: when original_metadata.codec is supersearch_metadata_v1,
each original_evidence row's metadata_ref indexes original_metadata.metadata.
That entry contains field_set_ref and value_refs. The field names, in order, are
original_metadata.field_sets[field_set_ref]. For each position i, its exact metadata
value is original_metadata.values[value_refs[i]]. Decode every metadata field this
way, including legal dates, validity, heading paths, labels, provenance and ACLs.
Values are complete JSON values; nested objects/arrays remain directly readable.
Missing fields remain missing; null, false, numeric values and empty values are
separate entries. References are table indexes, not evidence or citation numbers.
Treat decoded metadata exactly as ordinary original metadata. Text, citations,
source/chunk identities, hashes, truncation and citable flags remain explicit.
""".strip()


class _MetadataReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    field_set_ref: int = Field(ge=0)
    value_refs: list[int]


class _MetadataPool(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    codec: Literal["supersearch_metadata_v1"]
    field_sets: list[list[str]]
    values: list[JsonValue]
    metadata: list[_MetadataReference]


def _serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_original_evidence_payload(
    payload: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], bool]:
    """Pool metadata only when the provider projection plus its decoder is smaller."""
    raw_records = payload.get("original_evidence")
    if (
        not isinstance(raw_records, list)
        or not raw_records
        or METADATA_POOL_KEY in payload
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("metadata"), dict)
            or "metadata_ref" in row
            for row in raw_records
        )
    ):
        return payload, False

    values: list[JsonValue] = []
    value_indices: dict[str, int] = {}
    field_sets: list[list[str]] = []
    field_set_indices: dict[str, int] = {}
    metadata: list[JsonValue] = []
    metadata_indices: dict[str, int] = {}
    projected_records: list[JsonValue] = []
    for raw_record in raw_records:
        record = cast(dict[str, JsonValue], raw_record)
        record_metadata = cast(dict[str, JsonValue], record["metadata"])
        fields = list(record_metadata)
        fields_signature = _serialized(cast(JsonValue, fields))
        if fields_signature not in field_set_indices:
            field_set_indices[fields_signature] = len(field_sets)
            field_sets.append(fields)
        references: list[JsonValue] = []
        for value in record_metadata.values():
            # JSON signatures retain bool/int/float, null/empty and object order.
            signature = _serialized(value)
            if signature not in value_indices:
                value_indices[signature] = len(values)
                values.append(json.loads(signature))
            references.append(value_indices[signature])
        metadata_record: dict[str, JsonValue] = {
            "field_set_ref": field_set_indices[fields_signature],
            "value_refs": references,
        }
        metadata_signature = _serialized(metadata_record)
        if metadata_signature not in metadata_indices:
            metadata_indices[metadata_signature] = len(metadata)
            metadata.append(metadata_record)
        projected_records.append(
            {
                "metadata_ref" if key == "metadata" else key: (
                    metadata_indices[metadata_signature] if key == "metadata" else value
                )
                for key, value in record.items()
            }
        )
    projection: dict[str, JsonValue] = {
        **payload,
        "original_evidence": projected_records,
        METADATA_POOL_KEY: {
            "codec": METADATA_CODEC,
            "field_sets": cast(JsonValue, field_sets),
            "values": values,
            "metadata": metadata,
        },
    }
    if len(_serialized(projection).encode("utf-8")) + len(
        METADATA_DECODER.encode("utf-8")
    ) >= len(_serialized(payload).encode("utf-8")):
        return payload, False
    return projection, True


def expand_original_evidence_payload(
    projection: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Restore all metadata values and field order for auditing the exact projection."""
    raw_pool = projection.get(METADATA_POOL_KEY)
    if not isinstance(raw_pool, dict) or raw_pool.get("codec") != METADATA_CODEC:
        return projection
    pool = _MetadataPool.model_validate(raw_pool, strict=True)
    raw_records = projection.get("original_evidence")
    if not isinstance(raw_records, list):
        raise ValueError("Encoded original evidence must contain records")
    decoded_records: list[JsonValue] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or "metadata" in raw_record:
            raise ValueError("Encoded evidence must contain a metadata reference")
        metadata_ref = raw_record.get("metadata_ref")
        if (
            not isinstance(metadata_ref, int)
            or isinstance(metadata_ref, bool)
            or metadata_ref < 0
            or metadata_ref >= len(pool.metadata)
        ):
            raise ValueError("Original metadata reference is out of range")
        reference = pool.metadata[metadata_ref]
        if reference.field_set_ref >= len(pool.field_sets):
            raise ValueError("Original metadata field set is out of range")
        fields = pool.field_sets[reference.field_set_ref]
        if len(fields) != len(reference.value_refs) or len(set(fields)) != len(fields):
            raise ValueError("Original metadata field/value alignment is invalid")
        if any(
            index < 0 or index >= len(pool.values) for index in reference.value_refs
        ):
            raise ValueError("Original metadata value reference is out of range")
        decoded_metadata = {
            field: pool.values[index]
            for field, index in zip(fields, reference.value_refs, strict=True)
        }
        decoded_records.append(
            {
                "metadata" if key == "metadata_ref" else key: (
                    decoded_metadata if key == "metadata_ref" else value
                )
                for key, value in raw_record.items()
            }
        )
    return {
        key: decoded_records if key == "original_evidence" else value
        for key, value in projection.items()
        if key != METADATA_POOL_KEY
    }
