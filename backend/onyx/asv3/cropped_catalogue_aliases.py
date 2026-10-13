"""Exact worker metadata aliases to a complete same-message source catalogue."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import cast

from pydantic import JsonValue

CATALOGUE_ALIAS_HEADER = "cropped_catalogue_alias_transport"
_SUBTREE = "canonical_metadata_subtree_ref"
_BINDING = "canonical_full_original_ref"
_FIELDS = ("research_tasks", "repair_research_observations")
_CATALOGUE_KEYS = {"citation", "source_id", "chunk_id", "text_hash", "metadata"}
_BINDING_KEYS = {
    "citation",
    "source_id",
    "chunk_id",
    "text_hash",
    "metadata_hash",
    "start_char",
    "end_char",
    "range_hash",
}
_INSTRUCTION = (
    "Version1 canonical_metadata_subtree_ref:[citation,path_index,order_index] "
    "replaces only the literal JSON value at paths[path_index] in the complete "
    "original_metadata_catalogue row in this same message. orders[order_index] "
    "restores every original dictionary key order; scalar types and list order "
    "are unchanged. canonical_full_original_ref:[citation,opaque_metadata_hash,"
    "order_index] restores a verified full-range binding: source_id/chunk_id/"
    "text_hash from that row, start_char=0, end_char=len(the complete inline "
    "original_evidence text), range_hash=text_hash. The metadata_hash is the "
    "transmitted opaque literal, never inferred from projected metadata. "
    "Expand before reading worker observations and checking their unchanged "
    "hashes. Alias citations address literal data, not new legal citations, "
    "approval, model-delivery receipts or corpus-absence findings. No external "
    "catalogue or source-text lookup is allowed. All original bodies stay inline."
)
_MAX_EXPANDED_BYTES = 64 * 1024 * 1024
_MAX_EXPANDED_NODES = 2_000_000


def _json(value: JsonValue, *, ordered: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=not ordered,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash(value: JsonValue, *, ordered: bool = False) -> str:
    return hashlib.sha256(_json(value, ordered=ordered).encode()).hexdigest()


def _field_order_hash(payload: dict[str, JsonValue], fields: list[str]) -> str:
    return _hash({field: payload[field] for field in fields}, ordered=True)


def _catalogue(payload: dict[str, JsonValue]) -> dict[int, dict[str, JsonValue]]:
    raw = payload.get("original_metadata_catalogue")
    if not isinstance(raw, list):
        raise ValueError("Catalogue aliases require their transmitted catalogue")
    rows: dict[int, dict[str, JsonValue]] = {}
    identities: set[str] = set()
    for row in raw:
        if not isinstance(row, dict) or set(row) != _CATALOGUE_KEYS:
            raise ValueError("Catalogue alias identity fields are invalid")
        number = row["citation"]
        if type(number) is not int or number <= 0 or number in rows:
            raise ValueError("Catalogue alias citation is invalid")
        if (
            not isinstance(row["source_id"], str)
            or not row["source_id"]
            or (
                row["chunk_id"] is not None
                and (not isinstance(row["chunk_id"], str) or not row["chunk_id"])
            )
            or not isinstance(row["text_hash"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["text_hash"]) is None
            or not isinstance(row["metadata"], dict)
        ):
            raise ValueError("Catalogue alias source identity is unavailable")
        identity = _json([row[key] for key in ("source_id", "chunk_id", "text_hash")])
        if identity in identities:
            raise ValueError("Catalogue alias source identity is ambiguous")
        identities.add(identity)
        rows[number] = row
    return rows


def _full_bodies(
    payload: dict[str, JsonValue],
    rows: dict[int, dict[str, JsonValue]],
) -> dict[int, str]:
    raw = payload.get("original_evidence")
    if not isinstance(raw, list):
        raise ValueError("Full-range aliases require complete inline originals")
    bodies: dict[int, str] = {}
    for body in raw:
        if not isinstance(body, dict):
            raise ValueError("Full-range alias original is not an object")
        number, text = body.get("citation"), body.get("text")
        if type(number) is not int or number not in rows or number in bodies:
            raise ValueError("Full-range alias original citation is invalid")
        if (
            not isinstance(text, str)
            or not text
            or type(body.get("start_char")) is not int
            or body["start_char"] != 0
            or type(body.get("end_char")) is not int
            or body["end_char"] != len(text)
            or body.get("citable") is not True
            or body.get("truncated") is not False
            or hashlib.sha256(text.encode()).hexdigest() != rows[number]["text_hash"]
        ):
            raise ValueError("Full-range alias original text or range is invalid")
        if "identity_ref" in body:
            if type(body["identity_ref"]) is not int or body["identity_ref"] != number:
                raise ValueError("Full-range alias original identity changed")
            if any(
                key in body and _json(body[key]) != _json(rows[number][key])
                for key in ("source_id", "chunk_id", "text_hash")
            ):
                raise ValueError("Full-range alias original identity conflicts")
        elif any(
            key not in body or _json(body[key]) != _json(rows[number][key])
            for key in ("source_id", "chunk_id", "text_hash")
        ):
            raise ValueError("Full-range alias original identity is missing")
        bodies[number] = text
    if set(bodies) != set(rows):
        raise ValueError("Full-range aliases require equal body/catalogue citations")
    return bodies


def _order(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        children = {
            key: order
            for key, child in value.items()
            if (order := _order(child)) is not None
        }
        return {"keys": list(value), "children": children}
    if isinstance(value, list):
        children = {
            str(index): order
            for index, child in enumerate(value)
            if (order := _order(child)) is not None
        }
        return {"list_children": children} if children else None
    return None


def _restore_order(value: JsonValue, order: JsonValue) -> JsonValue:
    if order is None:
        return copy.deepcopy(value)
    if not isinstance(order, dict):
        raise ValueError("Catalogue alias order descriptor is invalid")
    if set(order) == {"keys", "children"}:
        keys, children = order["keys"], order["children"]
        if (
            not isinstance(value, dict)
            or not isinstance(keys, list)
            or any(not isinstance(key, str) for key in keys)
            or len(set(cast(list[str], keys))) != len(keys)
            or set(keys) != set(value)
            or not isinstance(children, dict)
            or not set(children).issubset(value)
        ):
            raise ValueError("Catalogue alias object order is invalid")
        return {
            key: _restore_order(value[key], children.get(key))
            for key in cast(list[str], keys)
        }
    if set(order) == {"list_children"}:
        children = order["list_children"]
        if (
            not isinstance(value, list)
            or not isinstance(children, dict)
            or any(
                not key.isdecimal() or str(int(key)) != key or int(key) >= len(value)
                for key in children
            )
        ):
            raise ValueError("Catalogue alias list order is invalid")
        return [
            _restore_order(child, children.get(str(index)))
            for index, child in enumerate(value)
        ]
    raise ValueError("Catalogue alias order descriptor has unexpected fields")


def _reject_references(value: JsonValue) -> None:
    if isinstance(value, dict):
        if {_SUBTREE, _BINDING, CATALOGUE_ALIAS_HEADER}.intersection(value):
            raise ValueError("Catalogue alias reference is outside its declared fields")
        for child in value.values():
            _reject_references(child)
    elif isinstance(value, list):
        for child in value:
            _reject_references(child)


def encode_cropped_catalogue_aliases(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Reference equal JSON values while preserving every old field and key order."""
    copied = copy.deepcopy(payload)
    _reject_references(copied)
    fields = [field for field in _FIELDS if field in copied]
    if not fields or "original_metadata_catalogue" not in copied:
        return copied
    rows = _catalogue(copied)
    try:
        bodies = _full_bodies(copied, rows)
    except ValueError:
        # Metadata references still work; incomplete bodies cannot supply bindings.
        bodies = {}
    lookup: dict[str, tuple[int, list[JsonValue]]] = {}
    paths: list[JsonValue] = []
    orders: list[JsonValue] = []

    def index(value: JsonValue, number: int, path: list[JsonValue]) -> None:
        if isinstance(value, (dict, list)):
            encoded = _json(value)
            if len(encoded) > 100:
                lookup.setdefault(encoded, (number, path))
        if isinstance(value, dict):
            for key, child in value.items():
                index(child, number, [*path, key])
        elif isinstance(value, list):
            for offset, child in enumerate(value):
                index(child, number, [*path, offset])

    for number, row in rows.items():
        index(row, number, [])

    def store(table: list[JsonValue], value: JsonValue) -> int:
        encoded = _json(value, ordered=True)
        for offset, existing in enumerate(table):
            if _json(existing, ordered=True) == encoded:
                return offset
        table.append(value)
        return len(table) - 1

    def encode(value: JsonValue) -> JsonValue:
        if not isinstance(value, (dict, list)):
            return value
        if isinstance(value, dict) and set(value) == _BINDING_KEYS:
            number = value["citation"]
            row = rows.get(number) if type(number) is int else None
            if (
                row is not None
                and number in bodies
                and type(value["start_char"]) is int
                and value["start_char"] == 0
                and type(value["end_char"]) is int
                and value["end_char"] == len(bodies[number])
                and all(
                    _json(value[key]) == _json(row[key])
                    for key in ("source_id", "chunk_id", "text_hash")
                )
                and _json(value["range_hash"]) == _json(row["text_hash"])
                and isinstance(value["metadata_hash"], str)
                and re.fullmatch(r"[0-9a-f]{64}", value["metadata_hash"]) is not None
            ):
                return {
                    _BINDING: [
                        number,
                        value["metadata_hash"],
                        store(orders, _order(value)),
                    ]
                }
        if (match := lookup.get(_json(value))) is not None:
            number, path = match
            return {
                _SUBTREE: [number, store(paths, path), store(orders, _order(value))]
            }
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()}
        return [encode(child) for child in value]

    for field in fields:
        copied[field] = encode(copied[field])
    if orders:
        copied[CATALOGUE_ALIAS_HEADER] = {
            "version": 1,
            "fields": fields,
            "paths": paths,
            "orders": orders,
            "instruction": _INSTRUCTION,
            "payload_hash": _hash(payload),
            "field_order_hash": _field_order_hash(payload, fields),
        }
    return copied


def expand_cropped_catalogue_aliases(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Resolve only transmitted values and verify the complete prior payload."""
    copied = copy.deepcopy(payload)
    if CATALOGUE_ALIAS_HEADER not in copied:
        _reject_references(copied)
        return copied
    header = copied.pop(CATALOGUE_ALIAS_HEADER)
    if (
        not isinstance(header, dict)
        or set(header)
        != {
            "version",
            "fields",
            "paths",
            "orders",
            "instruction",
            "payload_hash",
            "field_order_hash",
        }
        or type(header["version"]) is not int
        or header["version"] != 1
        or header["instruction"] != _INSTRUCTION
        or any(
            not isinstance(header[key], str)
            or re.fullmatch(r"[0-9a-f]{64}", cast(str, header[key])) is None
            for key in ("payload_hash", "field_order_hash")
        )
    ):
        raise ValueError("Catalogue alias header is invalid")
    fields, paths, orders = header["fields"], header["paths"], header["orders"]
    if (
        not isinstance(fields, list)
        or not fields
        or fields != [field for field in _FIELDS if field in copied]
        or not isinstance(paths, list)
        or not isinstance(orders, list)
        or not orders
        or len({_json(path, ordered=True) for path in paths}) != len(paths)
        or len({_json(order, ordered=True) for order in orders}) != len(orders)
    ):
        raise ValueError("Catalogue alias fields/tables are invalid")
    rows = _catalogue(copied)
    bodies: dict[int, str] | None = None
    used_paths: set[int] = set()
    used_orders: set[int] = set()
    expanded_bytes = len(_json(copied, ordered=True).encode())
    expanded_nodes = 0

    def charge(value: JsonValue) -> None:
        nonlocal expanded_bytes, expanded_nodes
        expanded_bytes += len(_json(value, ordered=True).encode())
        pending = [value]
        while pending:
            child = pending.pop()
            expanded_nodes += 1
            if (
                expanded_bytes > _MAX_EXPANDED_BYTES
                or expanded_nodes > _MAX_EXPANDED_NODES
            ):
                raise ValueError(
                    "Catalogue alias expansion exceeds its bounded envelope"
                )
            if isinstance(child, dict):
                pending.extend(child.values())
            elif isinstance(child, list):
                pending.extend(child)

    def restore(value: JsonValue, reference: JsonValue) -> JsonValue:
        if type(reference) is not int or not 0 <= reference < len(orders):
            raise ValueError("Catalogue alias order reference is invalid")
        charge(value)
        used_orders.add(reference)
        return _restore_order(value, orders[reference])

    def decode(value: JsonValue) -> JsonValue:
        nonlocal bodies
        if isinstance(value, list):
            return [decode(child) for child in value]
        if not isinstance(value, dict):
            return value
        if _SUBTREE in value:
            raw = value[_SUBTREE]
            if (
                set(value) != {_SUBTREE}
                or not isinstance(raw, list)
                or len(raw) != 3
                or type(raw[0]) is not int
                or raw[0] not in rows
                or type(raw[1]) is not int
                or not 0 <= raw[1] < len(paths)
                or not isinstance(paths[raw[1]], list)
            ):
                raise ValueError("Catalogue alias subtree reference is invalid")
            actual: JsonValue = rows[raw[0]]
            for key in cast(list[JsonValue], paths[raw[1]]):
                if isinstance(actual, dict) and isinstance(key, str) and key in actual:
                    actual = actual[key]
                elif (
                    isinstance(actual, list)
                    and type(key) is int
                    and 0 <= key < len(actual)
                ):
                    actual = actual[key]
                else:
                    raise ValueError("Catalogue alias path is unavailable")
            if not isinstance(actual, (dict, list)):
                raise ValueError("Catalogue alias subtree is not structured metadata")
            used_paths.add(raw[1])
            return restore(actual, raw[2])
        if _BINDING in value:
            raw = value[_BINDING]
            if (
                set(value) != {_BINDING}
                or not isinstance(raw, list)
                or len(raw) != 3
                or type(raw[0]) is not int
                or raw[0] not in rows
                or not isinstance(raw[1], str)
                or re.fullmatch(r"[0-9a-f]{64}", raw[1]) is None
            ):
                raise ValueError("Catalogue alias full binding is invalid")
            if bodies is None:
                bodies = _full_bodies(copied, rows)
            row = rows[raw[0]]
            binding: dict[str, JsonValue] = {
                "citation": raw[0],
                "source_id": row["source_id"],
                "chunk_id": row["chunk_id"],
                "text_hash": row["text_hash"],
                "metadata_hash": raw[1],
                "start_char": 0,
                "end_char": len(bodies[raw[0]]),
                "range_hash": row["text_hash"],
            }
            return restore(binding, raw[2])
        return {key: decode(child) for key, child in value.items()}

    for key, value in copied.items():
        if key not in fields:
            _reject_references(value)
    for field in cast(list[str], fields):
        copied[field] = decode(copied[field])
    if used_paths != set(range(len(paths))) or used_orders != set(range(len(orders))):
        raise ValueError("Catalogue alias table contains an unused entry")
    if _hash(copied) != header["payload_hash"]:
        raise ValueError("Catalogue alias payload changed")
    if _field_order_hash(copied, cast(list[str], fields)) != header["field_order_hash"]:
        raise ValueError("Catalogue alias worker key order changed")
    return copied
