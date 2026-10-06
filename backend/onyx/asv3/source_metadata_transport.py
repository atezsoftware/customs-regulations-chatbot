"""Factor identical source metadata without changing citation-specific locators."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import Sequence

from pydantic import JsonValue


def _encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def share_source_metadata(
    catalogue: Sequence[dict[str, JsonValue]],
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue]]:
    """Keep only fields literally identical in every selected row of one source."""
    groups: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for row in catalogue:
        source = row.get("source_id")
        if not isinstance(source, str) or not isinstance(row.get("metadata"), dict):
            raise ValueError("Source metadata needs canonical catalogue records")
        if "source_metadata_ref" in row:
            raise ValueError("Source metadata catalogue is already encoded")
        groups[source].append(row)
    shared: dict[str, JsonValue] = {}
    encoded = copy.deepcopy(list(catalogue))
    for source, rows in groups.items():
        if len(rows) < 2:
            continue
        first = rows[0]["metadata"]
        assert isinstance(first, dict)
        common = {
            key: copy.deepcopy(value)
            for key, value in first.items()
            if all(
                isinstance(metadata := row["metadata"], dict)
                and key in metadata
                and _encoded(metadata[key]) == _encoded(value)
                for row in rows[1:]
            )
        }
        if not common:
            continue
        replacements: list[dict[str, JsonValue]] = []
        for row in encoded:
            if row["source_id"] != source:
                continue
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            replacements.append(
                {
                    **row,
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key not in common
                    },
                    "source_metadata_ref": source,
                }
            )
        if len(_encoded(replacements)) + len(_encoded({source: common})) >= len(
            _encoded(rows)
        ):
            continue
        shared[source] = common
        replacement_iterator = iter(replacements)
        encoded = [
            next(replacement_iterator) if row["source_id"] == source else row
            for row in encoded
        ]
    return encoded, shared


def expand_source_metadata(payload: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Decode only metadata actually present in the same host model message."""
    rows = payload.get("original_metadata_catalogue", [])
    sources = payload.get("original_source_metadata", {})
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Native original catalogue must contain identity records")
    if not isinstance(sources, dict) or any(
        not isinstance(value, dict) for value in sources.values()
    ):
        raise ValueError("Native source metadata must contain source records")
    expanded: list[dict[str, JsonValue]] = []
    used: set[str] = set()
    for row in rows:
        assert isinstance(row, dict)
        if "source_metadata_ref" not in row:
            expanded.append(copy.deepcopy(row))
            continue
        reference, metadata = row["source_metadata_ref"], row.get("metadata")
        base = sources.get(reference) if isinstance(reference, str) else None
        if (
            not isinstance(reference, str)
            or reference != row.get("source_id")
            or not isinstance(metadata, dict)
            or not isinstance(base, dict)
            or set(base).intersection(metadata)
        ):
            raise ValueError("Native source metadata has an invalid source binding")
        expanded.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "source_metadata_ref"
                },
                "metadata": copy.deepcopy({**base, **metadata}),
            }
        )
        used.add(reference)
    if used != set(sources):
        raise ValueError("Native source metadata contains unbound source records")
    return expanded
