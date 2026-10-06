"""Lossless storage references for the shared evidence in parallel checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Callable, Iterator, Never, TypeGuard, cast

from pydantic import JsonValue

_STORAGE_KEY = "parallel_checkpoint_storage"
_HASH = re.compile(r"[a-f0-9]{64}")
_DEFAULT_UNIT_BYTES = 8_000_000


class _ImmutableDict(dict[str, JsonValue]):
    def __init__(
        self,
        values: Mapping[str, JsonValue] | Iterable[tuple[str, JsonValue]] = (),
    ) -> None:
        members = values.items() if isinstance(values, Mapping) else values
        checked: list[tuple[str, JsonValue]] = []
        for key, part in members:
            if type(key) is not str or not _immutable_member(part):
                raise ValueError("Invalid immutable checkpoint JSON value")
            checked.append((key, part))
        super().__init__(checked)

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
    def __init__(self, values: Iterable[JsonValue] = ()) -> None:
        members: list[JsonValue] = []
        for part in values:
            if not _immutable_member(part):
                raise ValueError("Invalid immutable checkpoint JSON value")
            members.append(part)
        super().__init__(members)

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


def _immutable_member(value: object) -> TypeGuard[JsonValue]:
    return (
        value is None
        or type(value) in (str, int, float, bool)
        or isinstance(value, (_ImmutableDict, _ImmutableList))
    )


def _freeze(value: JsonValue) -> JsonValue:
    if isinstance(value, (_ImmutableDict, _ImmutableList)):
        return value
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


def _restore_v1(
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


class _PoolObject(_ImmutableDict):
    """A concrete, hash-checked record or scalar range row."""

    serialized_bytes: int = 0


class _SharedSnapshot(_ImmutableDict):
    """A complete JSON snapshot validated within the compact storage capacity."""

    compact_bytes: int = _DEFAULT_UNIT_BYTES


def _checked_freeze(
    value: JsonValue,
    memo: dict[int, JsonValue] | None = None,
    active: set[int] | None = None,
) -> JsonValue:
    memo = {} if memo is None else memo
    active = set() if active is None else active
    if isinstance(value, (_ImmutableDict, _ImmutableList)):
        return value
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, (dict, list)):
        raise ValueError("Invalid checkpoint JSON value")
    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic checkpoint JSON value")
    if identity in memo:
        return memo[identity]
    active.add(identity)
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("Invalid checkpoint JSON key")
        result: JsonValue = _ImmutableDict(
            (key, _checked_freeze(part, memo, active)) for key, part in value.items()
        )
    else:
        result = _ImmutableList(_checked_freeze(part, memo, active) for part in value)
    active.remove(identity)
    memo[identity] = result
    return result


def _capacity(value: dict[str, JsonValue], maximum: int) -> int:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid checkpoint storage capacity")
    size = 0
    try:
        for token in json.JSONEncoder(ensure_ascii=False).iterencode(value):
            size += len(token.encode("utf-8"))
            if size > maximum:
                raise ValueError(
                    "Parallel checkpoint unit exceeds its storage capacity"
                )
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError("Invalid checkpoint JSON value") from error
    return size


def _all_ledgers(
    snapshot: dict[str, JsonValue],
    transform: Callable[[dict[str, JsonValue]], dict[str, JsonValue]],
    active: set[int] | None = None,
) -> dict[str, JsonValue]:
    active = set() if active is None else active
    if id(snapshot) in active:
        raise ValueError("Cyclic parallel child checkpoint")
    active.add(id(snapshot))
    result = dict(snapshot)
    ledger = result.get("evidence")
    if not isinstance(ledger, dict):
        raise ValueError("Parallel checkpoint evidence is missing")
    result["evidence"] = transform(ledger)
    workers = result.get("workers")
    if isinstance(workers, dict):
        raw_tasks = workers.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise ValueError("Invalid parallel checkpoint workers")
        tasks: list[JsonValue] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                raise ValueError("Invalid parallel checkpoint task")
            task = dict(raw_task)
            wrapper = task.get("child_checkpoint")
            if wrapper is not None:
                if not isinstance(wrapper, dict):
                    raise ValueError("Invalid parallel child checkpoint")
                child = wrapper.get("snapshot")
                if not isinstance(child, dict):
                    raise ValueError("Invalid parallel child checkpoint")
                task["child_checkpoint"] = {
                    **wrapper,
                    "snapshot": _all_ledgers(child, transform, active),
                }
            tasks.append(task)
        result["workers"] = {**workers, "tasks": tasks}
    active.remove(id(snapshot))
    return result


def _compact_v2(snapshot: dict[str, JsonValue], maximum: int) -> dict[str, JsonValue]:
    pools: dict[str, dict[str, JsonValue]] = {
        name: {} for name in ("records", "delivery_rows", "readsets", "deliveries")
    }
    identities: dict[int, tuple[JsonValue, str]] = {}

    def retain(name: str, value: JsonValue) -> str:
        if isinstance(value, _PoolObject) and id(value) in identities:
            return identities[id(value)][1]
        key = hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        existing = pools[name].get(key)
        if existing is not None and existing != value:
            raise ValueError("Conflicting parallel checkpoint pool object")
        pools[name][key] = value
        if isinstance(value, _PoolObject):
            identities[id(value)] = (value, key)
        return key

    def transform(ledger: dict[str, JsonValue]) -> dict[str, JsonValue]:
        records, deliveries = ledger.get("records"), ledger.get("deliveries", [])
        if (
            ledger.get("version") != 1
            or not isinstance(records, list)
            or not isinstance(deliveries, list)
        ):
            raise ValueError("Invalid parallel evidence checkpoint")
        refs: list[JsonValue] = []
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError("Invalid parallel checkpoint evidence record")
            _check_record(record)
            if record["citation"] != index:
                raise ValueError("Invalid parallel checkpoint citation sequence")
            refs.append(retain("records", record))
        delivery_refs: list[JsonValue] = []
        for delivery in deliveries:
            if not isinstance(delivery, dict):
                raise ValueError("Invalid parallel evidence delivery")
            ranges = delivery.get("records")
            if not isinstance(ranges, list):
                raise ValueError("Invalid parallel evidence delivery")
            range_refs: list[JsonValue] = []
            for row in ranges:
                if not isinstance(row, dict):
                    raise ValueError("Invalid parallel checkpoint delivery range")
                _check_row(row)
                range_refs.append(retain("delivery_rows", row))
            readset = retain("readsets", range_refs)
            delivery_refs.append(retain("deliveries", {**delivery, "records": readset}))
        return {
            **ledger,
            "records": refs,
            **({"deliveries": delivery_refs} if "deliveries" in ledger else {}),
        }

    result = _all_ledgers(snapshot, transform)
    result[_STORAGE_KEY] = {"version": 2, **pools}
    _capacity(result, maximum)
    return result


def _restore_v2(snapshot: dict[str, JsonValue], maximum: int) -> dict[str, JsonValue]:
    compact_bytes = _capacity(snapshot, maximum)
    storage = snapshot.get(_STORAGE_KEY)
    names = ("records", "delivery_rows", "readsets", "deliveries")
    if not isinstance(storage, dict) or set(storage) != {"version", *names}:
        raise ValueError("Invalid parallel checkpoint storage format")
    pools: dict[str, dict[str, JsonValue]] = {}
    used: dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        raw = storage.get(name)
        if not isinstance(raw, dict):
            raise ValueError("Invalid parallel checkpoint pool")
        pools[name] = {}
        for key, value in raw.items():
            encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
            if not _HASH.fullmatch(key) or hashlib.sha256(encoded).hexdigest() != key:
                raise ValueError("Parallel checkpoint pool integrity changed")
            if name in {"records", "delivery_rows"}:
                if not isinstance(value, dict):
                    raise ValueError("Invalid parallel checkpoint pool object")
                (_check_record if name == "records" else _check_row)(value)
                frozen = _checked_freeze(value)
                assert isinstance(frozen, dict)
                pooled = _PoolObject(frozen)
                pooled.serialized_bytes = len(encoded)
                pools[name][key] = pooled
            elif name == "readsets":
                if not isinstance(value, list) or any(
                    not isinstance(part, str) for part in value
                ):
                    raise ValueError("Invalid parallel checkpoint readset")
                pools[name][key] = value
            else:
                if not isinstance(value, dict) or not isinstance(
                    value.get("records"), str
                ):
                    raise ValueError("Invalid parallel checkpoint delivery")
                pools[name][key] = value

    def resolve(name: str, key: JsonValue) -> JsonValue:
        if not isinstance(key, str) or key not in pools[name]:
            raise ValueError("Missing or invalid parallel checkpoint reference")
        used[name].add(key)
        return pools[name][key]

    # Fixed typed edges prevent reference cycles and nested amplification.
    for key, raw in pools["readsets"].items():
        assert isinstance(raw, list)
        pools["readsets"][key] = _ImmutableList(
            resolve("delivery_rows", reference) for reference in raw
        )
    for key, raw in pools["deliveries"].items():
        assert isinstance(raw, dict)
        pools["deliveries"][key] = _checked_freeze(
            {**raw, "records": resolve("readsets", raw["records"])}
        )

    checked_ranges: set[tuple[int, int]] = set()

    def transform(ledger: dict[str, JsonValue]) -> dict[str, JsonValue]:
        refs, delivery_refs = ledger.get("records"), ledger.get("deliveries", [])
        if (
            ledger.get("version") != 1
            or not isinstance(refs, list)
            or not isinstance(delivery_refs, list)
        ):
            raise ValueError("Invalid parallel evidence checkpoint")
        records = [resolve("records", ref) for ref in refs]
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict) or record["citation"] != index:
                raise ValueError("Invalid parallel checkpoint citation sequence")
        deliveries = [resolve("deliveries", ref) for ref in delivery_refs]
        for delivery in deliveries:
            assert isinstance(delivery, dict)
            rows = delivery["records"]
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, dict)
                citation = cast(int, row["citation"])
                if citation > len(records):
                    raise ValueError("Unknown delivered evidence")
                item = cast(
                    dict[str, JsonValue],
                    cast(dict[str, JsonValue], records[citation - 1])["item"],
                )
                pair = (id(row), id(item))
                if pair in checked_ranges:
                    continue
                if any(
                    row[field] != item.get(field)
                    for field in ("source_id", "chunk_id", "text_hash")
                ):
                    raise ValueError("Changed delivered evidence identity")
                text = item.get("text")
                start, end = cast(int, row["start_char"]), cast(int, row["end_char"])
                if (
                    not isinstance(text, str)
                    or end > len(text)
                    or hashlib.sha256(text[start:end].encode()).hexdigest()
                    != row["passage_hash"]
                    or (row["complete"] is True and (start != 0 or end != len(text)))
                ):
                    raise ValueError("Changed delivered evidence range")
                checked_ranges.add(pair)
        return {
            **ledger,
            "records": records,
            **({"deliveries": deliveries} if "deliveries" in ledger else {}),
        }

    result = _all_ledgers(snapshot, transform)
    if any(used[name] != set(pools[name]) for name in names):
        raise ValueError("Unreferenced parallel checkpoint pool object")
    del result[_STORAGE_KEY]
    # Check/freeze the entire arbitrary native snapshot, not just evidence fields.
    frozen = _checked_freeze(result)
    assert isinstance(frozen, dict)
    shared = _SharedSnapshot(frozen)
    shared.compact_bytes = compact_bytes
    return shared


def compact_parallel_checkpoint(
    snapshot: dict[str, JsonValue], *, max_unit_bytes: int = _DEFAULT_UNIT_BYTES
) -> dict[str, JsonValue]:
    """Store exact historical values once, including ordered delivery containers."""
    if _STORAGE_KEY in snapshot:
        raise ValueError("Parallel checkpoint is already compacted")
    if not _parallel(snapshot) or "evidence" not in snapshot:
        return snapshot
    return _compact_v2(snapshot, max_unit_bytes)


def restore_parallel_checkpoint(
    snapshot: dict[str, JsonValue], *, max_unit_bytes: int = _DEFAULT_UNIT_BYTES
) -> dict[str, JsonValue]:
    storage = snapshot.get(_STORAGE_KEY)
    if storage is None and _STORAGE_KEY not in snapshot:
        return snapshot
    if (
        not _parallel(snapshot)
        or not isinstance(storage, dict)
        or type(storage.get("version")) is not int
    ):
        raise ValueError("Invalid parallel checkpoint storage format")
    if storage["version"] == 1:
        return _restore_v1(snapshot, max_unit_bytes=max_unit_bytes)
    if storage["version"] == 2:
        return _restore_v2(snapshot, max_unit_bytes)
    raise ValueError("Invalid parallel checkpoint storage format")


def share_parallel_snapshot(
    snapshot: dict[str, JsonValue], *, max_unit_bytes: int = _DEFAULT_UNIT_BYTES
) -> dict[str, JsonValue]:
    """Validate and intern a complete child snapshot before recursive copies."""
    return _restore_v2(_compact_v2(snapshot, max_unit_bytes), max_unit_bytes)


def parallel_checkpoint_digest(value: dict[str, JsonValue]) -> str:
    """Hash the original JSON byte sequence without allocating expanded JSON."""
    cache: dict[int, bytes] = {}
    cache_bytes = 0
    snapshot = value.get("snapshot")
    cache_limit = max(
        0,
        _DEFAULT_UNIT_BYTES
        - (snapshot.compact_bytes if isinstance(snapshot, _SharedSnapshot) else 0),
    )

    def tokens(part: JsonValue) -> Iterator[bytes]:
        nonlocal cache_bytes
        if isinstance(part, _PoolObject) and (
            id(part) in cache or cache_bytes + part.serialized_bytes <= cache_limit
        ):
            encoded = cache.get(id(part))
            if encoded is None:
                encoded = json.dumps(part, sort_keys=True, ensure_ascii=False).encode()
                cache_bytes += len(encoded)
                if cache_bytes > cache_limit:
                    raise ValueError(
                        "Parallel checkpoint digest cache exceeds its storage capacity"
                    )
                cache[id(part)] = encoded
            yield encoded
        elif isinstance(part, dict):
            yield b"{"
            for index, key in enumerate(sorted(part)):
                if index:
                    yield b", "
                yield json.dumps(key, ensure_ascii=False).encode()
                yield b": "
                yield from tokens(part[key])
            yield b"}"
        elif isinstance(part, list):
            yield b"["
            for index, member in enumerate(part):
                if index:
                    yield b", "
                yield from tokens(member)
            yield b"]"
        else:
            yield json.dumps(part, ensure_ascii=False).encode()

    digest = hashlib.sha256()
    for token in tokens(value):
        digest.update(token)
    return digest.hexdigest()
