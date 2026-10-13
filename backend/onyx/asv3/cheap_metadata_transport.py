"""Lossless, message-local metadata tables for the explicit Cheap workflow."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

from pydantic import JsonValue

from onyx.asv3.source_metadata_transport import (
    expand_source_metadata,
    share_source_metadata,
)

_CATALOGUE_FIELDS = ["citation", "source_ref", "chunk_id", "text_hash", "metadata"]
_SOURCE_FIELDS = ["source_id", "metadata"]
_PAYLOAD_FIELDS = (
    "original_evidence_ranges",
    "delivered_provisions",
    "original_evidence",
    "retained_source_conditions",
    "publication_gap",
)
_EXTRA_PAYLOAD_FIELDS = (
    "research_tasks",
    "repair_research_observations",
    "cropped_completion",
    "session_research",
)
_CODEC = {
    "version": 1,
    "catalogue_fields": _CATALOGUE_FIELDS,
    "source_fields": _SOURCE_FIELDS,
}
_TRANSPORT_FIELDS = frozenset(
    {
        "original_metadata_codec",
        "original_metadata_sources",
        "original_metadata_layouts",
        "original_metadata_values",
        "original_metadata_transport_note",
    }
)
_NOTE = (
    "Version 1 catalogue rows follow original_metadata_codec.catalogue_fields. "
    "source_ref indexes original_metadata_sources, whose rows follow source_fields. "
    "A metadata object {layout_ref:[index,values]} maps the ordered keys in "
    "original_metadata_layouts[index] to those values; absent keys remain absent "
    "and null remains null. Ordinary lists and scalars retain their literal types. "
    "{value_ref:index} inserts the complete literal JSON value from "
    "original_metadata_values[index]; pool entries are literal data, never transport "
    "references. Fields listed in payload_fields use the same reversible node "
    "grammar. In original_evidence, every string including text stays inline "
    "and literal; no source text is recovered from the ledger or value pool. "
    "Source metadata and row metadata have distinct keys and combine "
    "without overrides. Resolve these tables within this same message to recover "
    "the exact source, chunk, text hash and metadata. This representation replaces "
    "original_source_metadata/source_metadata_ref for this catalogue. It conveys "
    "identity, not source text, applicability or legal approval."
)
_IDENTITY_NOTE = _NOTE + (
    " Version 2 also permits {value_ref:index} in catalogue chunk_id/text_hash "
    "and source-table source_id cells. Resolve those cells to their complete "
    "literal string in this message's original_metadata_values; a null chunk_id "
    "remains null. Source IDs, chunk IDs and hashes are never shortened or "
    "recovered from host state. Citation and source_ref cells remain integers."
)


def _text(value: object, *, compact: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":") if compact else None,
    )


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: JsonValue) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Metadata object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif isinstance(value, list):
        for child in value:
            _json_value(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("Metadata value is not JSON data")
    _text(value)


def _index(value: JsonValue, length: int) -> int:
    if type(value) is not int or not 0 <= value < length:
        raise ValueError("Metadata table reference is invalid")
    return value


def expand_cheap_payload(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Recover metadata and declared fields only from this actual host message."""
    if "original_metadata_codec" not in payload:
        if _TRANSPORT_FIELDS.intersection(payload):
            raise ValueError("Metadata tables require their versioned codec")
        expand_source_metadata(payload)
        return copy.deepcopy(payload)
    header = payload["original_metadata_codec"]
    payload_fields = header.get("payload_fields") if isinstance(header, dict) else None
    version = header.get("version") if isinstance(header, dict) else None
    if (
        not isinstance(header, dict)
        or set(header) != {*_CODEC, "payload_fields"}
        or type(version) is not int
        or version not in (1, 2)
        or {key: value for key, value in header.items() if key != "payload_fields"}
        != {**_CODEC, "version": version}
        or not isinstance(payload_fields, list)
        or any(
            not isinstance(field, str)
            or field not in (*_PAYLOAD_FIELDS, *_EXTRA_PAYLOAD_FIELDS)
            for field in payload_fields
        )
        or len(set(cast(list[str], payload_fields))) != len(payload_fields)
        or payload.get("original_metadata_transport_note")
        != (_NOTE if version == 1 else _IDENTITY_NOTE)
        or (
            version == 2
            and not any(field in _EXTRA_PAYLOAD_FIELDS for field in payload_fields)
        )
        or "original_source_metadata" in payload
    ):
        raise ValueError("Metadata codec header or grammar is invalid")
    assert isinstance(payload_fields, list)
    rows = payload.get("original_metadata_catalogue")
    sources = payload.get("original_metadata_sources")
    layouts = payload.get("original_metadata_layouts")
    values = payload.get("original_metadata_values")
    if not all(isinstance(table, list) for table in (rows, sources, layouts, values)):
        raise ValueError("Metadata codec tables must be lists")
    assert isinstance(rows, list) and isinstance(sources, list)
    assert isinstance(layouts, list) and isinstance(values, list)
    shapes: list[list[str]] = []
    for layout in layouts:
        if (
            not isinstance(layout, list)
            or not layout
            or any(not isinstance(key, str) for key in layout)
            or len(set(cast(list[str], layout))) != len(layout)
        ):
            raise ValueError("Metadata layout has invalid or duplicate keys")
        shapes.append(cast(list[str], layout))
    if len({tuple(shape) for shape in shapes}) != len(shapes):
        raise ValueError("Metadata layouts are duplicated")
    for value in values:
        _json_value(value)
    if len({_text(value, compact=True) for value in values}) != len(values):
        raise ValueError("Metadata value table has duplicate entries")
    used_layouts: set[int] = set()
    used_values: set[int] = set()

    def decode(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            if not value:
                return {}
            if set(value) == {"value_ref"}:
                reference = _index(value["value_ref"], len(values))
                used_values.add(reference)
                return copy.deepcopy(values[reference])
            if set(value) != {"layout_ref"}:
                raise ValueError("Metadata object has no valid layout")
            node = value["layout_ref"]
            if not isinstance(node, list) or len(node) != 2:
                raise ValueError("Metadata layout node must contain index and values")
            reference = _index(node[0], len(shapes))
            children = node[1]
            if not isinstance(children, list) or len(children) != len(
                shapes[reference]
            ):
                raise ValueError("Metadata layout and values have different arity")
            used_layouts.add(reference)
            return {
                key: decode(child)
                for key, child in zip(shapes[reference], children, strict=True)
            }
        if isinstance(value, list):
            return [decode(child) for child in value]
        _json_value(value)
        return copy.deepcopy(value)

    def identity(value: JsonValue) -> JsonValue:
        if version == 2 and isinstance(value, dict):
            if set(value) != {"value_ref"}:
                raise ValueError("Metadata identity needs a literal value reference")
            return decode(value)
        return value

    canonical_sources: list[tuple[str, dict[str, JsonValue]]] = []
    for source in sources:
        if not isinstance(source, list) or len(source) != len(_SOURCE_FIELDS):
            raise ValueError("Metadata source record is invalid")
        source_id = identity(source[0])
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Metadata source identity must be a nonempty string")
        metadata = decode(source[1])
        if not isinstance(metadata, dict):
            raise ValueError("Source metadata must remain an object")
        canonical_sources.append((source_id, metadata))
    if len({source[0] for source in canonical_sources}) != len(canonical_sources):
        raise ValueError("Metadata source identities are duplicated")
    used_sources: set[int] = set()
    citations: set[int] = set()
    expanded: list[dict[str, JsonValue]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(_CATALOGUE_FIELDS):
            raise ValueError("Metadata catalogue row has invalid arity")
        number, source_ref, chunk, digest, encoded_metadata = row
        chunk, digest = identity(chunk), identity(digest)
        if (
            type(number) is not int
            or number <= 0
            or number in citations
            or (chunk is not None and not isinstance(chunk, str))
            or not isinstance(digest, str)
        ):
            raise ValueError("Metadata catalogue has an invalid canonical identity")
        reference = _index(source_ref, len(canonical_sources))
        used_sources.add(reference)
        citations.add(number)
        source_id, base = canonical_sources[reference]
        metadata = decode(encoded_metadata)
        if not isinstance(metadata, dict) or set(base).intersection(metadata):
            raise ValueError("Metadata source and row fields overlap")
        expanded.append(
            {
                "citation": number,
                "source_id": source_id,
                "chunk_id": chunk,
                "text_hash": digest,
                "metadata": copy.deepcopy({**base, **metadata}),
            }
        )
    restored_fields: dict[str, JsonValue] = {}
    for field in payload_fields:
        assert isinstance(field, str)
        if field not in payload:
            raise ValueError("Declared metadata payload field is missing")
        restored_fields[field] = decode(payload[field])
    if (
        used_sources != set(range(len(sources)))
        or used_layouts != set(range(len(layouts)))
        or used_values != set(range(len(values)))
    ):
        raise ValueError("Metadata codec contains unused table entries")
    restored = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in _TRANSPORT_FIELDS
    }
    restored["original_metadata_catalogue"] = cast(JsonValue, expanded)
    restored.update(restored_fields)
    return restored


def expand_cheap_metadata(
    payload: dict[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    """Return the same canonical five-field catalogue for the native ledger guard."""
    if "original_metadata_codec" not in payload:
        if _TRANSPORT_FIELDS.intersection(payload):
            raise ValueError("Metadata tables require their versioned codec")
        return expand_source_metadata(payload)
    restored = expand_cheap_payload(payload)
    return cast(list[dict[str, JsonValue]], restored["original_metadata_catalogue"])


def encode_cheap_metadata(
    payload: dict[str, JsonValue],
    token_counter: Callable[[str], int],
    *,
    extra_fields: Sequence[str] = (),
) -> dict[str, JsonValue]:
    """Keep a separate encoded payload only when its full text estimates smaller."""
    if (
        isinstance(extra_fields, (str, bytes))
        or any(field not in _EXTRA_PAYLOAD_FIELDS for field in extra_fields)
        or len(set(extra_fields)) != len(extra_fields)
    ):
        raise ValueError("Metadata extra payload fields are invalid")
    if not extra_fields:
        return _encode_cheap_metadata(payload, token_counter)
    if "original_metadata_codec" in payload:
        # A completed v2 encoding is stable; v1 can opt into the newer grammar.
        expand_cheap_payload(payload)
        header = cast(dict[str, JsonValue], payload["original_metadata_codec"])
        declared = cast(list[str], header["payload_fields"])
        if header["version"] == 2 and set(extra_fields).issubset(declared):
            return copy.deepcopy(payload)
        extra_fields = tuple(
            dict.fromkeys(
                [
                    *(field for field in declared if field in _EXTRA_PAYLOAD_FIELDS),
                    *extra_fields,
                ]
            )
        )
        payload = expand_cheap_payload(payload)
    if not any(field in payload for field in extra_fields):
        return _encode_cheap_metadata(payload, token_counter, extra_fields=extra_fields)
    candidates = [
        _encode_cheap_metadata(payload, token_counter, extra_fields=extra_fields),
        _encode_cheap_metadata(
            payload, token_counter, extra_fields=extra_fields, identity_refs=True
        ),
        _encode_cheap_metadata(
            payload,
            token_counter,
            extra_fields=extra_fields,
            identity_refs=True,
            small_containers=True,
        ),
    ]
    return min(
        candidates,
        key=lambda candidate: min(
            token_counter(_text(candidate)),
            token_counter(_text(candidate, compact=True)),
        ),
    )


def _encode_cheap_metadata(
    payload: dict[str, JsonValue],
    token_counter: Callable[[str], int],
    *,
    extra_fields: Sequence[str] = (),
    identity_refs: bool = False,
    small_containers: bool = False,
) -> dict[str, JsonValue]:
    canonical = expand_cheap_metadata(payload)
    if "original_metadata_codec" in payload:
        header = cast(dict[str, JsonValue], payload["original_metadata_codec"])
        declared = cast(list[str], header["payload_fields"])
        current_extra = [field for field in declared if field in _EXTRA_PAYLOAD_FIELDS]
        if set(extra_fields).issubset(current_extra):
            return copy.deepcopy(payload)
        upgraded = encode_cheap_metadata(
            expand_cheap_payload(payload),
            token_counter,
            extra_fields=tuple(dict.fromkeys([*current_extra, *extra_fields])),
        )
        return (
            upgraded
            if token_counter(_text(upgraded)) < token_counter(_text(payload))
            else copy.deepcopy(payload)
        )
    if not canonical:
        return copy.deepcopy(payload)
    if any(
        set(row) != {"citation", "source_id", "chunk_id", "text_hash", "metadata"}
        for row in canonical
    ):
        raise ValueError("Metadata catalogue must have exact canonical identity fields")
    original_rows, common = share_source_metadata(canonical)
    source_ids = list(dict.fromkeys(cast(str, row["source_id"]) for row in canonical))
    source_indexes = {source: index for index, source in enumerate(source_ids)}
    occurrences: Counter[str] = Counter()
    literal_values: dict[str, JsonValue] = {}
    costs: dict[str, int] = {}

    def scan(value: JsonValue) -> None:
        _json_value(value)
        if isinstance(value, (str, list, dict)):
            signature = _text(value, compact=True)
            if signature not in costs:
                costs[signature] = token_counter(_text(value))
            if costs[signature] >= 8:
                occurrences[signature] += 1
                literal_values.setdefault(signature, value)
        if isinstance(value, dict):
            for child in value.values():
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    for row in original_rows:
        assert isinstance(row, dict)
        scan(row["metadata"])
    for source in source_ids:
        scan(common.get(source, {}))
    if identity_refs:
        for row in original_rows:
            scan(row["chunk_id"])
            scan(row["text_hash"])
        for source in source_ids:
            scan(source)
    payload_fields = [
        field for field in (*_PAYLOAD_FIELDS, *extra_fields) if field in payload
    ]
    for field in payload_fields:
        if field != "original_evidence":
            scan(payload[field])
    reference_cost = token_counter(_text({"value_ref": 999}))
    selected = {
        signature
        for signature, count in occurrences.items()
        if count > 1
        and (
            not small_containers
            or isinstance(literal_values[signature], str)
            or costs[signature] < 64
        )
        and costs[signature] * count
        > reference_cost * count + token_counter(_text([literal_values[signature]])) + 8
    }
    values: list[JsonValue] = []
    value_indexes: dict[str, int] = {}
    layouts: list[JsonValue] = []
    layout_indexes: dict[tuple[str, ...], int] = {}

    def encode(value: JsonValue, *, allow_pool: bool = True) -> JsonValue:
        signature = _text(value, compact=True)
        if allow_pool and signature in selected:
            if signature not in value_indexes:
                value_indexes[signature] = len(values)
                values.append(copy.deepcopy(value))
            return {"value_ref": value_indexes[signature]}
        if isinstance(value, dict):
            if not value:
                return {}
            shape = tuple(value)
            if shape not in layout_indexes:
                layout_indexes[shape] = len(layouts)
                layouts.append(list(shape))
            return {
                "layout_ref": [
                    layout_indexes[shape],
                    [encode(child, allow_pool=allow_pool) for child in value.values()],
                ]
            }
        if isinstance(value, list):
            return [encode(child, allow_pool=allow_pool) for child in value]
        return copy.deepcopy(value)

    rows: list[JsonValue] = []
    for row in original_rows:
        assert isinstance(row, dict)
        rows.append(
            [
                row["citation"],
                source_indexes[cast(str, row["source_id"])],
                encode(row["chunk_id"]) if identity_refs else row["chunk_id"],
                encode(row["text_hash"]) if identity_refs else row["text_hash"],
                encode(row["metadata"]),
            ]
        )
    sources: list[JsonValue] = [
        [encode(source) if identity_refs else source, encode(common.get(source, {}))]
        for source in source_ids
    ]
    candidate = copy.deepcopy(payload)
    candidate.pop("original_source_metadata", None)
    candidate.update(
        {
            "original_metadata_codec": {
                **copy.deepcopy(_CODEC),
                "version": 2 if identity_refs else 1,
                "payload_fields": payload_fields,
            },
            "original_metadata_catalogue": rows,
            "original_metadata_sources": sources,
            "original_metadata_layouts": layouts,
            "original_metadata_values": values,
            "original_metadata_transport_note": _IDENTITY_NOTE
            if identity_refs
            else _NOTE,
        }
    )
    for field in payload_fields:
        candidate[field] = encode(
            payload[field], allow_pool=field != "original_evidence"
        )
    expected = copy.deepcopy(payload)
    expected.pop("original_source_metadata", None)
    expected["original_metadata_catalogue"] = cast(JsonValue, canonical)
    if _canonical_text(expand_cheap_payload(candidate)) != _canonical_text(expected):
        raise ValueError("Metadata codec did not preserve its complete payload")
    if token_counter(_text(candidate)) >= token_counter(_text(payload)):
        return copy.deepcopy(payload)
    return candidate


def serialize_cheap_payload(
    payload: dict[str, JsonValue],
    token_counter: Callable[[str], int],
    *,
    extra_fields: Sequence[str] = (),
) -> str:
    """Remove JSON separator whitespace only when the actual estimator improves."""
    encoded = encode_cheap_metadata(payload, token_counter, extra_fields=extra_fields)
    ordinary = _text(encoded)
    compact = _text(encoded, compact=True)
    if json.loads(compact) != encoded:
        raise ValueError("Cheap JSON serialization did not preserve its payload")
    return compact if token_counter(compact) < token_counter(ordinary) else ordinary
