"""Bounded, lossless dictionary storage for repeated checkpoint values."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import cast

from pydantic import JsonValue


def _serialized(value: JsonValue) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode()


def compact_checkpoint_dictionary(
    snapshot: dict[str, JsonValue],
    *,
    storage_key: str,
    variant: str,
    label: str,
    max_expanded_bytes: int,
    max_depth: int,
    use_if_smaller: bool = False,
) -> dict[str, JsonValue]:
    if storage_key in snapshot:
        raise ValueError(f"{label} checkpoint is already compacted")
    if snapshot.get("asv3_workflow_variant") != variant:
        raise ValueError(f"Invalid {label} checkpoint identity")
    raw = _serialized(snapshot)
    if len(raw) > max_expanded_bytes:
        raise ValueError(f"{label} checkpoint exceeds its expanded storage budget")
    counts: Counter[str] = Counter()
    keys: set[str] = set()

    def scan(value: JsonValue, depth: int = 0) -> None:
        if depth > max_depth:
            raise ValueError(f"{label} checkpoint exceeds its storage depth")
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

    packed: dict[str, JsonValue] = {
        "version": snapshot.get("version"),
        "asv3_workflow_variant": variant,
        storage_key: {
            "version": 1,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "expanded_bytes": len(raw),
            "strings": list(strings),
            "value": pack(snapshot),
        },
    }

    return (
        snapshot if use_if_smaller and len(_serialized(packed)) >= len(raw) else packed
    )


def restore_checkpoint_dictionary(
    snapshot: dict[str, JsonValue],
    *,
    storage_key: str,
    variant: str,
    label: str,
    max_expanded_bytes: int,
    max_depth: int,
) -> dict[str, JsonValue]:
    if storage_key not in snapshot:
        return snapshot
    storage = snapshot[storage_key]
    if (
        set(snapshot) != {"version", "asv3_workflow_variant", storage_key}
        or snapshot.get("asv3_workflow_variant") != variant
        or not isinstance(storage, dict)
        or set(storage) != {"version", "sha256", "expanded_bytes", "strings", "value"}
        or type(storage.get("version")) is not int
        or storage.get("version") != 1
        or not isinstance(storage.get("sha256"), str)
        or type(storage.get("expanded_bytes")) is not int
        or not 0 < storage["expanded_bytes"] <= max_expanded_bytes
        or not isinstance(storage.get("strings"), list)
    ):
        raise ValueError(f"Invalid {label} checkpoint storage")
    strings = storage["strings"]
    if any(not isinstance(value, str) for value in strings):
        raise ValueError(f"Invalid {label} checkpoint dictionary")
    texts = cast(list[str], strings)
    if texts != sorted(set(texts)):
        raise ValueError(f"Noncanonical {label} checkpoint dictionary")
    string_sizes = [len(_serialized(value)) for value in texts]
    remaining = cast(int, storage["expanded_bytes"])
    used: set[int] = set()

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError(f"{label} checkpoint exceeds its expanded storage budget")

    def reference(value: JsonValue) -> str:
        if type(value) is not int or not 0 <= value < len(texts):
            raise ValueError(f"Invalid {label} checkpoint reference")
        used.add(value)
        charge(string_sizes[value])
        return texts[value]

    def unpack(value: JsonValue, depth: int = 0) -> JsonValue:
        if depth > max_depth:
            raise ValueError(f"{label} checkpoint exceeds its storage depth")
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Invalid {label} checkpoint reference")
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
                        raise ValueError(f"Invalid {label} checkpoint field")
                    key = reference(field[0])
                    if key in result:
                        raise ValueError(f"Duplicate {label} checkpoint field")
                    result[key] = unpack(field[1], depth + 1)
                return result
            raise ValueError(f"Invalid {label} checkpoint value")
        charge(len(_serialized(value)))
        return value

    restored = unpack(storage.get("value"))
    if (
        not isinstance(restored, dict)
        or restored.get("asv3_workflow_variant") != variant
        or restored.get("version") != snapshot.get("version")
        or used != set(range(len(texts)))
    ):
        raise ValueError(f"Invalid {label} checkpoint identity")
    raw = _serialized(restored)
    if (
        remaining != 0
        or len(raw) != storage["expanded_bytes"]
        or hashlib.sha256(raw).hexdigest() != storage["sha256"]
    ):
        raise ValueError(f"Changed {label} checkpoint contents")
    return restored
