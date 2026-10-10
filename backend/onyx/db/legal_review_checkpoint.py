"""Lossless storage dictionary for Legal Review's repeated source and audit values."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from pydantic import JsonValue

STORAGE_KEY = "legal_review_checkpoint_storage"
MAX_EXPANDED_BYTES = 64_000_000
MAX_DEPTH = 64


def _serialized(value: JsonValue) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode()


def compact_legal_review_checkpoint(
    snapshot: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    if STORAGE_KEY in snapshot:
        raise ValueError("Legal Review checkpoint is already compacted")
    if snapshot.get("asv3_workflow_variant") != "legal_review":
        return snapshot
    raw = _serialized(snapshot)
    if len(raw) > MAX_EXPANDED_BYTES:
        raise ValueError("Legal Review checkpoint exceeds its expanded storage budget")
    counts: Counter[str] = Counter()
    keys: set[str] = set()

    def scan(value: JsonValue, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            raise ValueError("Legal Review checkpoint exceeds its storage depth")
        if isinstance(value, str):
            if len(value) >= 32:
                counts[value] += 1
        elif isinstance(value, dict):
            keys.update(value)
            for part in value.values():
                scan(part, depth + 1)
        elif isinstance(value, list):
            for part in value:
                scan(part, depth + 1)

    scan(snapshot)
    strings = sorted(keys | {value for value, count in counts.items() if count > 1})
    identities = {value: index for index, value in enumerate(strings)}

    def pack(value: JsonValue) -> JsonValue:
        if isinstance(value, str) and value in identities:
            return [identities[value]]
        if isinstance(value, dict):
            return {"o": [[identities[key], pack(part)] for key, part in value.items()]}
        if isinstance(value, list):
            return {"a": [pack(part) for part in value]}
        return value

    return {
        "version": snapshot.get("version"),
        "asv3_workflow_variant": "legal_review",
        STORAGE_KEY: {
            "version": 1,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "expanded_bytes": len(raw),
            "strings": strings,
            "value": pack(snapshot),
        },
    }


def restore_legal_review_checkpoint(
    snapshot: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    if STORAGE_KEY not in snapshot:
        return snapshot
    storage = snapshot[STORAGE_KEY]
    if (
        snapshot.get("asv3_workflow_variant") != "legal_review"
        or not isinstance(storage, dict)
        or storage.get("version") != 1
        or not isinstance(storage.get("sha256"), str)
        or type(storage.get("expanded_bytes")) is not int
        or not 0 < storage["expanded_bytes"] <= MAX_EXPANDED_BYTES
        or not isinstance(storage.get("strings"), list)
    ):
        raise ValueError("Invalid Legal Review checkpoint storage")
    strings = storage["strings"]
    if any(not isinstance(value, str) for value in strings):
        raise ValueError("Invalid Legal Review checkpoint dictionary")
    string_sizes = [len(_serialized(value)) for value in strings]
    remaining = MAX_EXPANDED_BYTES

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError(
                "Legal Review checkpoint exceeds its expanded storage budget"
            )

    def reference(value: JsonValue) -> str:
        if type(value) is not int or not 0 <= value < len(strings):
            raise ValueError("Invalid Legal Review checkpoint reference")
        text = strings[value]
        assert isinstance(text, str)
        charge(string_sizes[value])
        return text

    def unpack(value: JsonValue, depth: int = 0) -> JsonValue:
        if depth > MAX_DEPTH:
            raise ValueError("Legal Review checkpoint exceeds its storage depth")
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Invalid Legal Review checkpoint reference")
            return reference(value[0])
        if isinstance(value, dict):
            if set(value) == {"a"} and isinstance(value["a"], list):
                parts = value["a"]
                charge(2 + max(0, len(parts) - 1) * 2)
                return [unpack(part, depth + 1) for part in parts]
            if set(value) == {"o"} and isinstance(value["o"], list):
                fields = value["o"]
                charge(2 + max(0, len(fields) - 1) * 2 + len(fields) * 2)
                result: dict[str, JsonValue] = {}
                for field in fields:
                    if not isinstance(field, list) or len(field) != 2:
                        raise ValueError("Invalid Legal Review checkpoint field")
                    key = reference(field[0])
                    if key in result:
                        raise ValueError("Duplicate Legal Review checkpoint field")
                    result[key] = unpack(field[1], depth + 1)
                return result
            raise ValueError("Invalid Legal Review checkpoint value")
        charge(len(_serialized(value)))
        return value

    restored = unpack(storage.get("value"))
    if (
        not isinstance(restored, dict)
        or restored.get("asv3_workflow_variant") != "legal_review"
        or restored.get("version") != snapshot.get("version")
    ):
        raise ValueError("Invalid Legal Review checkpoint identity")
    raw = _serialized(restored)
    if (
        len(raw) != storage["expanded_bytes"]
        or hashlib.sha256(raw).hexdigest() != storage["sha256"]
    ):
        raise ValueError("Changed Legal Review checkpoint contents")
    return restored
