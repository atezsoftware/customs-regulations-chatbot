"""Cropped-only exact handoff references to the same-message source catalogue."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import cast

from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import expand_cheap_payload
from onyx.asv3.cropped_catalogue_aliases import expand_cropped_catalogue_aliases
from onyx.asv3.source_metadata_transport import expand_source_metadata

_HEADER = "cropped_canonical_fields_transport"
_REFERENCE = "canonical_fields_ref"
_FIELDS = ("research_tasks", "repair_research_observations")
_KEYS = ("citation", "source_id", "chunk_id", "text_hash", "metadata")
_INSTRUCTION = (
    "canonical_fields_ref:[citation,layout_index] replaces ONLY the layout.fields keys "
    "from the complete original_metadata_catalogue row in this same message. "
    "Restore exact fields in layout.order, preserving the entire object's key order "
    "so recorded JSON strings reconstruct byte for byte; every other key is verbatim. "
    "The original_evidence bodies remain literal, complete and unchanged. "
    "Resolve every reference before reading the full worker object and checking "
    "its unchanged hashes. These references do not approve findings or create "
    "model-delivery receipts. Reject an unavailable citation, invalid layout, "
    "field collision or unused layout. No external catalogue lookup is permitted."
)


def _copy(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)),
    )


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _ordered_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _field_order_hash(payload: dict[str, JsonValue], fields: list[str]) -> str:
    return hashlib.sha256(
        _ordered_json({field: payload[field] for field in fields}).encode()
    ).hexdigest()


def canonical_cropped_handoff_payload(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Resolve the existing source table before comparing complete JSON values."""
    copied = _copy(expand_cheap_payload(payload))
    if "original_metadata_catalogue" in copied:
        copied["original_metadata_catalogue"] = cast(
            list[JsonValue], expand_source_metadata(copied)
        )
        copied.pop("original_source_metadata", None)
    return copied


def _catalogue(payload: dict[str, JsonValue]) -> dict[int, dict[str, JsonValue]]:
    rows = payload.get("original_metadata_catalogue")
    if not isinstance(rows, list):
        raise ValueError("Cropped handoff references require their actual catalogue")
    catalogue: dict[int, dict[str, JsonValue]] = {}
    identities: set[tuple[str, str | None, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_KEYS):
            raise ValueError("Cropped handoff catalogue has invalid fields")
        number = row["citation"]
        if type(number) is not int or number <= 0 or number in catalogue:
            raise ValueError("Cropped handoff catalogue has an invalid citation")
        values = [row[key] for key in ("source_id", "chunk_id", "text_hash")]
        if any(
            not isinstance(values[index], str) or not values[index] for index in (0, 2)
        ) or (
            values[1] is not None and (not isinstance(values[1], str) or not values[1])
        ):
            raise ValueError("Cropped handoff catalogue identity is unavailable")
        if not isinstance(row["metadata"], dict):
            raise ValueError("Cropped handoff catalogue metadata is not an object")
        identity = cast(tuple[str, str | None, str], tuple(values))
        if identity in identities:
            raise ValueError("Cropped handoff catalogue identity is ambiguous")
        identities.add(identity)
        catalogue[number] = row
    return catalogue


def _reject_reference(node: JsonValue) -> None:
    if isinstance(node, list):
        for value in node:
            _reject_reference(value)
    elif isinstance(node, dict):
        if _REFERENCE in node:
            raise ValueError("Cropped handoff reference is outside its declared field")
        for value in node.values():
            _reject_reference(value)


def encode_cropped_handoff_transport(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Factor only byte-equivalent source fields; keep source text untouched."""
    copied = canonical_cropped_handoff_payload(payload)
    original_hash = _hash(copied)
    if _HEADER in copied:
        raise ValueError("Cropped handoff transport is already encoded")
    for value in copied.values():
        _reject_reference(value)
    fields = [field for field in _FIELDS if field in copied]
    if not fields or "original_metadata_catalogue" not in copied:
        return copied
    catalogue = _catalogue(copied)
    identities = {
        (row["source_id"], row["chunk_id"], row["text_hash"]): row
        for row in catalogue.values()
    }
    order_hash = _field_order_hash(copied, fields)
    layouts: list[dict[str, list[str]]] = []

    def encode(node: JsonValue) -> JsonValue:
        if isinstance(node, list):
            return [encode(value) for value in node]
        if not isinstance(node, dict):
            return node
        canonical = None
        if "citation" in node:
            number = node["citation"]
            if type(number) is int:
                canonical = catalogue.get(number)
        elif all(isinstance(node.get(key), str) for key in _KEYS[1:4]):
            canonical = identities.get(
                (node["source_id"], node["chunk_id"], node["text_hash"])
            )
        keys = (
            [
                key
                for key in _KEYS
                if key in node
                and _ordered_json(node[key]) == _ordered_json(canonical[key])
            ]
            if canonical is not None
            else []
        )
        if len(keys) < 2:
            keys = []
        encoded = {key: encode(value) for key, value in node.items() if key not in keys}
        if keys and canonical is not None:
            layout = {"fields": keys, "order": list(node)}
            if layout not in layouts:
                layouts.append(layout)
            encoded[_REFERENCE] = [canonical["citation"], layouts.index(layout)]
        return encoded

    for field in fields:
        copied[field] = encode(copied[field])
    if layouts:
        copied[_HEADER] = {
            "version": 1,
            "fields": fields,
            "layouts": cast(list[JsonValue], layouts),
            "instruction": _INSTRUCTION,
            "payload_hash": original_hash,
            "field_order_hash": order_hash,
        }
    return copied


def expand_cropped_handoff_transport(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Restore exact fields solely from the actual transmitted catalogue."""
    copied = canonical_cropped_handoff_payload(payload)
    if _HEADER not in copied:
        for value in copied.values():
            _reject_reference(value)
        return expand_cropped_catalogue_aliases(copied)
    header = copied.pop(_HEADER)
    if (
        not isinstance(header, dict)
        or set(header)
        != {
            "version",
            "fields",
            "layouts",
            "instruction",
            "payload_hash",
            "field_order_hash",
        }
        or type(header["version"]) is not int
        or header["version"] != 1
        or header["instruction"] != _INSTRUCTION
        or not isinstance(header["payload_hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", header["payload_hash"]) is None
        or not isinstance(header["field_order_hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", header["field_order_hash"]) is None
    ):
        raise ValueError("Cropped handoff transport header is invalid")
    fields, raw_layouts = header["fields"], header["layouts"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) for field in fields)
        or fields != [field for field in _FIELDS if field in fields]
        or any(field not in copied for field in fields)
        or not isinstance(raw_layouts, list)
        or not raw_layouts
    ):
        raise ValueError("Cropped handoff transport fields/layouts are invalid")
    layouts: list[dict[str, list[str]]] = []
    for raw in raw_layouts:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"fields", "order"}
            or not isinstance(raw["fields"], list)
            or len(raw["fields"]) < 2
            or raw["fields"] != [key for key in _KEYS if key in raw["fields"]]
            or not isinstance(raw["order"], list)
            or any(not isinstance(key, str) for key in raw["order"])
            or len(set(raw["order"])) != len(raw["order"])
            or _REFERENCE in raw["order"]
            or not set(raw["fields"]).issubset(raw["order"])
            or raw in layouts
        ):
            raise ValueError("Cropped handoff transport layout is invalid")
        layouts.append(cast(dict[str, list[str]], raw))
    catalogue = _catalogue(copied)
    used: set[int] = set()

    def decode(node: JsonValue) -> JsonValue:
        if isinstance(node, list):
            return [decode(value) for value in node]
        if not isinstance(node, dict):
            return node
        decoded = {
            key: decode(value) for key, value in node.items() if key != _REFERENCE
        }
        if _REFERENCE not in node:
            return decoded
        reference = node[_REFERENCE]
        if (
            not isinstance(reference, list)
            or len(reference) != 2
            or type(reference[0]) is not int
            or type(reference[1]) is not int
            or reference[0] not in catalogue
            or not 0 <= reference[1] < len(layouts)
        ):
            raise ValueError("Cropped handoff source reference is invalid")
        canonical = catalogue[reference[0]]
        layout = layouts[reference[1]]
        keys, order = layout["fields"], layout["order"]
        if any(key in decoded for key in keys):
            raise ValueError("Cropped handoff source reference collides with a field")
        for key in keys:
            decoded[key] = copy.deepcopy(canonical[key])
        if set(decoded) != set(order):
            raise ValueError(
                "Cropped handoff source reference has an invalid key order"
            )
        used.add(reference[1])
        return {key: decoded[key] for key in order}

    for key, value in copied.items():
        if key not in fields:
            _reject_reference(value)
    for field in fields:
        copied[field] = decode(copied[field])
    if used != set(range(len(layouts))):
        raise ValueError("Cropped handoff transport contains an unused layout")
    if _hash(copied) != header["payload_hash"]:
        raise ValueError("Cropped handoff transport changed its original payload")
    if _field_order_hash(copied, cast(list[str], fields)) != header["field_order_hash"]:
        raise ValueError("Cropped handoff transport changed its original field order")
    return expand_cropped_catalogue_aliases(copied)
