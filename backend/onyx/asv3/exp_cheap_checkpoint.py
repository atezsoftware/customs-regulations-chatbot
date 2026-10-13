"""Exact storage pooling for Cheap checkpoints, separate from research transport."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import cast

from pydantic import JsonValue

from onyx.asv3.workflow_variant import ASV3_EXP_CHEAP_POLICY, ASV3_EXP_CHEAP_VARIANT

EXP_CHEAP_CHECKPOINT_STORAGE_KEY = "asv3_exp_cheap_checkpoint_storage"
EXP_CHEAP_CHECKPOINT_STORAGE_VERSION = 1
# Match the existing Legal Review expansion ceiling; packed/durable limits stay put.
MAX_EXP_CHEAP_EXPANDED_BYTES = 64_000_000
MAX_EXP_CHEAP_DEPTH = 64
_MIN_POOLED_STRING_CHARACTERS = 32


def _require_cheap(snapshot: dict[str, JsonValue]) -> None:
    if (
        snapshot.get("asv3_workflow_variant") != ASV3_EXP_CHEAP_VARIANT
        or snapshot.get("asv3_workflow_policy") != ASV3_EXP_CHEAP_POLICY
        or snapshot.get("research_profile") != "normal"
        or snapshot.get("parallel_research") is not False
    ):
        raise ValueError("Cheap checkpoint requires its own normal-profile policy")


def _encoded(value: JsonValue) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid Cheap checkpoint JSON value") from error


def _measure(value: JsonValue, maximum: int) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False)
    try:
        for part in encoder.iterencode(value):
            raw = part.encode("utf-8")
            size += len(raw)
            if size > maximum:
                raise ValueError("Cheap checkpoint exceeds its expanded storage budget")
            digest.update(raw)
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError("Invalid Cheap checkpoint JSON value") from error
    return size, digest.hexdigest()


def compact_exp_cheap_checkpoint(
    snapshot: dict[str, JsonValue], *, max_packed_bytes: int = 8_000_000
) -> dict[str, JsonValue]:
    """Pool exact repeated strings and keys without changing any stored field."""
    _require_cheap(snapshot)
    if EXP_CHEAP_CHECKPOINT_STORAGE_KEY in snapshot:
        raise ValueError("Cheap checkpoint is already compacted")
    if type(max_packed_bytes) is not int or max_packed_bytes < 1:
        raise ValueError("Invalid Cheap checkpoint storage capacity")
    counts: Counter[str] = Counter()
    keys: set[str] = set()
    active: set[int] = set()
    remaining = MAX_EXP_CHEAP_EXPANDED_BYTES

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("Cheap checkpoint exceeds its expanded storage budget")

    def scan(value: JsonValue, depth: int = 0) -> None:
        if depth > MAX_EXP_CHEAP_DEPTH:
            raise ValueError("Cheap checkpoint exceeds its storage depth")
        if isinstance(value, str):
            charge(len(_encoded(value).encode("utf-8")))
            if len(value) >= _MIN_POOLED_STRING_CHARACTERS:
                counts[value] += 1
        elif isinstance(value, (dict, list)):
            if id(value) in active:
                raise ValueError("Cyclic Cheap checkpoint value")
            active.add(id(value))
            charge(2 + max(0, len(value) - 1) * 2)
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise ValueError("Invalid Cheap checkpoint JSON key")
                keys.update(value)
                for key in value:
                    charge(len(_encoded(key).encode("utf-8")) + 2)
                members = value.values()
            else:
                members = value
            for part in members:
                scan(part, depth + 1)
            active.remove(id(value))
        else:
            charge(len(_encoded(value).encode("utf-8")))

    scan(snapshot)
    expanded_bytes, digest = _measure(snapshot, MAX_EXP_CHEAP_EXPANDED_BYTES)
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
        "asv3_workflow_variant": ASV3_EXP_CHEAP_VARIANT,
        "asv3_workflow_policy": ASV3_EXP_CHEAP_POLICY,
        EXP_CHEAP_CHECKPOINT_STORAGE_KEY: {
            "version": EXP_CHEAP_CHECKPOINT_STORAGE_VERSION,
            "sha256": digest,
            "expanded_bytes": expanded_bytes,
            "strings": cast(list[JsonValue], strings),
            "value": pack(snapshot),
        },
    }
    packed_bytes, _ = _measure(packed, MAX_EXP_CHEAP_EXPANDED_BYTES)
    # Small snapshots need no dictionary overhead or new storage representation.
    result = snapshot if expanded_bytes <= packed_bytes else packed
    _measure(result, max_packed_bytes)
    return result


def restore_exp_cheap_checkpoint(
    snapshot: dict[str, JsonValue], *, max_packed_bytes: int = 8_000_000
) -> dict[str, JsonValue]:
    """Restore independent containers; shared immutable strings cannot cross-mutate."""
    if EXP_CHEAP_CHECKPOINT_STORAGE_KEY not in snapshot:
        return snapshot
    if type(max_packed_bytes) is not int or max_packed_bytes < 1:
        raise ValueError("Invalid Cheap checkpoint storage capacity")
    _measure(snapshot, max_packed_bytes)
    storage = snapshot[EXP_CHEAP_CHECKPOINT_STORAGE_KEY]
    if (
        set(snapshot)
        != {
            "version",
            "asv3_workflow_variant",
            "asv3_workflow_policy",
            EXP_CHEAP_CHECKPOINT_STORAGE_KEY,
        }
        or snapshot.get("asv3_workflow_variant") != ASV3_EXP_CHEAP_VARIANT
        or snapshot.get("asv3_workflow_policy") != ASV3_EXP_CHEAP_POLICY
        or not isinstance(storage, dict)
        or set(storage) != {"version", "sha256", "expanded_bytes", "strings", "value"}
        or type(storage.get("version")) is not int
        or storage["version"] != EXP_CHEAP_CHECKPOINT_STORAGE_VERSION
        or not isinstance(storage.get("sha256"), str)
        or type(storage.get("expanded_bytes")) is not int
        or not 0 < storage["expanded_bytes"] <= MAX_EXP_CHEAP_EXPANDED_BYTES
        or not isinstance(storage.get("strings"), list)
    ):
        raise ValueError("Invalid Cheap checkpoint storage")
    strings = storage["strings"]
    if any(not isinstance(value, str) for value in strings):
        raise ValueError("Invalid Cheap checkpoint dictionary")
    texts = cast(list[str], strings)
    if texts != sorted(set(texts)):
        raise ValueError("Noncanonical Cheap checkpoint dictionary")
    sizes = [len(_encoded(value).encode("utf-8")) for value in texts]
    remaining = cast(int, storage["expanded_bytes"])
    used: set[int] = set()

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("Cheap checkpoint exceeds its expanded storage budget")

    def reference(value: JsonValue) -> str:
        if type(value) is not int or not 0 <= value < len(texts):
            raise ValueError("Invalid Cheap checkpoint reference")
        used.add(value)
        charge(sizes[value])
        return texts[value]

    def unpack(value: JsonValue, depth: int = 0) -> JsonValue:
        if depth > MAX_EXP_CHEAP_DEPTH:
            raise ValueError("Cheap checkpoint exceeds its storage depth")
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Invalid Cheap checkpoint reference")
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
                        raise ValueError("Invalid Cheap checkpoint field")
                    key = reference(field[0])
                    if key in result:
                        raise ValueError("Duplicate Cheap checkpoint field")
                    result[key] = unpack(field[1], depth + 1)
                return result
            raise ValueError("Invalid Cheap checkpoint value")
        charge(len(_encoded(value).encode("utf-8")))
        return value

    restored = unpack(storage["value"])
    if (
        not isinstance(restored, dict)
        or restored.get("version") != snapshot.get("version")
        or used != set(range(len(texts)))
    ):
        raise ValueError("Invalid Cheap checkpoint identity or unused pool object")
    _require_cheap(restored)
    expanded_bytes, digest = _measure(restored, cast(int, storage["expanded_bytes"]))
    if remaining != 0 or expanded_bytes != storage["expanded_bytes"]:
        raise ValueError("Changed Cheap checkpoint expanded size")
    if digest != storage["sha256"]:
        raise ValueError("Changed Cheap checkpoint contents")
    return restored
