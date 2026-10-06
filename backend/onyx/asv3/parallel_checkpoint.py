"""Lossless storage references for the shared evidence in parallel checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Never, cast

from pydantic import JsonValue

_STORAGE_KEY = "parallel_checkpoint_storage"
_HASH = re.compile(r"[a-f0-9]{64}")
_DEFAULT_UNIT_BYTES = 8_000_000


class _ImmutableDict(dict[str, JsonValue]):
    def _reject(self, *_args: object, **_kwargs: object) -> Never:
        raise TypeError("Shared checkpoint evidence is immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    clear = _reject
    pop = _reject
    popitem = _reject
    setdefault = _reject
    update = _reject
    __ior__ = _reject

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableDict:
        return self


class _ImmutableList(list[JsonValue]):
    def _reject(self, *_args: object, **_kwargs: object) -> Never:
        raise TypeError("Shared checkpoint evidence is immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    append = _reject
    clear = _reject
    extend = _reject
    insert = _reject
    pop = _reject
    remove = _reject
    reverse = _reject
    sort = _reject
    __iadd__ = _reject
    __imul__ = _reject

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableList:
        return self


def _freeze(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _ImmutableDict((key, _freeze(part)) for key, part in value.items())
    if isinstance(value, list):
        return _ImmutableList(_freeze(part) for part in value)
    return value


def _digest(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _parallel(snapshot: dict[str, JsonValue]) -> bool:
    return (
        snapshot.get("research_profile") == "experimental"
        and snapshot.get("parallel_research") is True
    )


def _transform_ledgers(
    snapshot: dict[str, JsonValue],
    transform: Callable[[dict[str, JsonValue]], dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    result = dict(snapshot)
    ledger = result.get("evidence")
    if not isinstance(ledger, dict):
        raise ValueError("Parallel checkpoint evidence is missing")
    result["evidence"] = transform(ledger)
    workers = result.get("workers")
    if not isinstance(workers, dict):
        return result
    raw_tasks = workers.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError("Invalid parallel checkpoint workers")
    tasks: list[JsonValue] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise ValueError("Invalid parallel checkpoint task")
        task = dict(raw_task)
        wrapper = task.get("child_checkpoint")
        if isinstance(wrapper, dict):
            child = wrapper.get("snapshot")
            if not isinstance(child, dict):
                raise ValueError("Invalid parallel child checkpoint")
            evidence = child.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError("Parallel child checkpoint evidence is missing")
            task["child_checkpoint"] = {
                **wrapper,
                "snapshot": {**child, "evidence": transform(evidence)},
            }
        tasks.append(task)
    result["workers"] = {**workers, "tasks": tasks}
    return result


def _transform_ledger(
    ledger: dict[str, JsonValue],
    record: Callable[[JsonValue], JsonValue],
    row: Callable[[JsonValue], JsonValue],
) -> dict[str, JsonValue]:
    records = ledger.get("records")
    deliveries = ledger.get("deliveries", [])
    if (
        ledger.get("version") != 1
        or not isinstance(records, list)
        or not isinstance(deliveries, list)
    ):
        raise ValueError("Invalid parallel evidence checkpoint")
    replaced: list[JsonValue] = []
    for delivery in deliveries:
        if not isinstance(delivery, dict):
            raise ValueError("Invalid parallel evidence delivery")
        ranges = delivery.get("records")
        if not isinstance(ranges, list):
            raise ValueError("Invalid parallel evidence delivery")
        replaced.append({**delivery, "records": [row(part) for part in ranges]})
    return {
        **ledger,
        "records": [record(part) for part in records],
        **({"deliveries": replaced} if "deliveries" in ledger else {}),
    }


def _check_units(snapshot: dict[str, JsonValue], max_unit_bytes: int) -> None:
    # Bound each validator's allocation without counting shared historical copies.
    if type(max_unit_bytes) is not int or max_unit_bytes < 1:
        raise ValueError("Invalid checkpoint storage capacity")

    def check(unit: dict[str, JsonValue]) -> None:
        size = 0
        for part in json.JSONEncoder(ensure_ascii=False).iterencode(unit):
            size += len(part.encode("utf-8"))
            if size > max_unit_bytes:
                raise ValueError(
                    "Parallel checkpoint unit exceeds its storage capacity"
                )

    workers = snapshot.get("workers")
    if not isinstance(workers, dict):
        check(snapshot)
        return
    raw_tasks = workers.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError("Invalid parallel checkpoint workers")
    tasks: list[JsonValue] = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise ValueError("Invalid parallel checkpoint task")
        wrapper = task.get("child_checkpoint")
        if isinstance(wrapper, dict):
            child = wrapper.get("snapshot")
            if not isinstance(child, dict):
                raise ValueError("Invalid parallel child checkpoint")
            check(child)
        tasks.append({**task, "child_checkpoint": None})
    check({**snapshot, "workers": {**workers, "tasks": tasks}})


def _check_record(record: dict[str, JsonValue]) -> None:
    if (
        set(record) != {"citation", "item"}
        or type(record.get("citation")) is not int
        or cast(int, record["citation"]) < 1
        or not isinstance(record.get("item"), dict)
    ):
        raise ValueError("Invalid parallel checkpoint evidence record")


def _check_row(row: dict[str, JsonValue]) -> None:
    if (
        set(row)
        != {
            "citation",
            "source_id",
            "chunk_id",
            "text_hash",
            "start_char",
            "end_char",
            "passage_hash",
            "complete",
        }
        or type(row.get("citation")) is not int
        or cast(int, row["citation"]) < 1
        or not isinstance(row.get("source_id"), str)
        or not (row.get("chunk_id") is None or isinstance(row.get("chunk_id"), str))
        or type(row.get("start_char")) is not int
        or type(row.get("end_char")) is not int
        or cast(int, row["start_char"]) < 0
        or cast(int, row["end_char"]) < cast(int, row["start_char"])
        or type(row.get("complete")) is not bool
        or any(
            not isinstance(value := row.get(key), str) or not _HASH.fullmatch(value)
            for key in ("text_hash", "passage_hash")
        )
    ):
        raise ValueError("Invalid parallel checkpoint delivery range")


def compact_parallel_checkpoint(
    snapshot: dict[str, JsonValue],
    *,
    max_unit_bytes: int = _DEFAULT_UNIT_BYTES,
) -> dict[str, JsonValue]:
    """Pool exact historical records; the native snapshot digests stay unchanged."""
    if _STORAGE_KEY in snapshot:
        raise ValueError("Parallel checkpoint is already compacted")
    if not _parallel(snapshot) or "evidence" not in snapshot:
        return snapshot
    _check_units(snapshot, max_unit_bytes)
    records: dict[str, JsonValue] = {}
    rows: dict[str, JsonValue] = {}

    def retain(
        value: JsonValue,
        pool: dict[str, JsonValue],
        validate: Callable[[dict[str, JsonValue]], None],
    ) -> str:
        if not isinstance(value, dict):
            raise ValueError("Invalid parallel checkpoint pool object")
        validate(value)
        digest = _digest(value)
        if digest in pool and pool[digest] != value:
            raise ValueError("Conflicting parallel checkpoint pool object")
        pool[digest] = value
        return digest

    result = _transform_ledgers(
        snapshot,
        lambda ledger: _transform_ledger(
            ledger,
            lambda value: retain(value, records, _check_record),
            lambda value: retain(value, rows, _check_row),
        ),
    )
    result[_STORAGE_KEY] = {
        "version": 1,
        "records": records,
        "delivery_rows": rows,
    }
    return result


def restore_parallel_checkpoint(
    snapshot: dict[str, JsonValue],
    *,
    max_unit_bytes: int = _DEFAULT_UNIT_BYTES,
) -> dict[str, JsonValue]:
    """Restore exact values using immutable shared objects, never recursive refs."""
    storage = snapshot.get(_STORAGE_KEY)
    if storage is None and _STORAGE_KEY not in snapshot:
        return snapshot
    if (
        not _parallel(snapshot)
        or not isinstance(storage, dict)
        or set(storage) != {"version", "records", "delivery_rows"}
        or type(storage.get("version")) is not int
        or storage["version"] != 1
    ):
        raise ValueError("Invalid parallel checkpoint storage format")

    def pool(
        name: str, validate: Callable[[dict[str, JsonValue]], None]
    ) -> dict[str, dict[str, JsonValue]]:
        values = storage.get(name)
        if not isinstance(values, dict):
            raise ValueError("Invalid parallel checkpoint pool")
        result: dict[str, dict[str, JsonValue]] = {}
        for key, value in values.items():
            if (
                not _HASH.fullmatch(key)
                or not isinstance(value, dict)
                or _digest(value) != key
            ):
                raise ValueError("Parallel checkpoint pool integrity changed")
            validate(value)
            result[key] = cast(dict[str, JsonValue], _freeze(value))
        return result

    records, rows = pool("records", _check_record), pool("delivery_rows", _check_row)
    used_records: set[str] = set()
    used_rows: set[str] = set()

    def resolve(
        reference: JsonValue,
        values: dict[str, dict[str, JsonValue]],
        used: set[str],
    ) -> dict[str, JsonValue]:
        if not isinstance(reference, str) or reference not in values:
            raise ValueError("Missing or invalid parallel checkpoint reference")
        used.add(reference)
        return values[reference]

    result = _transform_ledgers(
        snapshot,
        lambda ledger: _transform_ledger(
            ledger,
            lambda value: resolve(value, records, used_records),
            lambda value: resolve(value, rows, used_rows),
        ),
    )
    if used_records != set(records) or used_rows != set(rows):
        raise ValueError("Unreferenced parallel checkpoint pool object")
    del result[_STORAGE_KEY]

    def validate_ledger(ledger: dict[str, JsonValue]) -> dict[str, JsonValue]:
        members = cast(list[dict[str, JsonValue]], ledger["records"])
        if any(record["citation"] != index for index, record in enumerate(members, 1)):
            raise ValueError("Invalid parallel checkpoint citation sequence")
        return ledger

    _transform_ledgers(result, validate_ledger)
    _check_units(result, max_unit_bytes)
    return result
