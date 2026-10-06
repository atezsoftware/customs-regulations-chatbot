"""Lossless spelling and nesting normalization for exposed terminal metadata."""

from __future__ import annotations

import json

from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.asv3.parallel_execution import parallel_execution_enabled

_TERMINALS = {"submit_answer", "submit_partial_answer"}
_METADATA = {
    "_outcomes",
    "_coverage",
    "_related_source_reviews",
    "_language",
    "_public_update",
    "_notifications",
    "_need_id",
    "_external_requested",
}
_ALIASES = {
    "outcomes": "_outcomes",
    "coverage": "_coverage",
    "related_source_reviews": "_related_source_reviews",
}


def _same_json(left: JsonValue, right: JsonValue) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def normalize_terminal_metadata(
    name: str,
    arguments: dict[str, JsonValue],
    context: RunContext,
    parameters: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Preserve all values and unknown fields; conflicts remain schema-invalid."""
    properties = parameters.get("properties")
    if (
        name not in _TERMINALS
        or not parallel_execution_enabled(context)
        or not isinstance(properties, dict)
    ):
        return arguments
    coverage_schema = properties.get("_coverage")
    coverage_properties = (
        coverage_schema.get("properties") if isinstance(coverage_schema, dict) else None
    )
    relocatable = (
        {key for key in ("conditions", "resolutions") if key in coverage_properties}
        if isinstance(coverage_properties, dict)
        else set()
    )
    normalized: dict[str, JsonValue] = {}
    relocated: dict[str, JsonValue] = {}
    for key, value in arguments.items():
        target = key
        trimmed = key.strip()
        if trimmed in _METADATA and trimmed in properties:
            target = trimmed
        elif trimmed in _ALIASES and trimmed not in properties:
            alias = _ALIASES[trimmed]
            if alias in properties:
                target = alias
        elif trimmed in relocatable and trimmed not in properties:
            if trimmed in relocated and not _same_json(relocated[trimmed], value):
                return arguments
            relocated[trimmed] = value
            continue
        if target in normalized and not _same_json(normalized[target], value):
            return arguments
        normalized[target] = value
    coverage = normalized.get("_coverage")
    if isinstance(coverage, dict) and isinstance(coverage_properties, dict):
        fixed_coverage: dict[str, JsonValue] = {}
        for key, value in coverage.items():
            target = key.strip() if key.strip() in coverage_properties else key
            if target in fixed_coverage and not _same_json(
                fixed_coverage[target], value
            ):
                return arguments
            fixed_coverage[target] = value
        coverage = fixed_coverage
        normalized["_coverage"] = coverage
    if relocated:
        if "_coverage" in normalized and not isinstance(coverage, dict):
            return arguments
        coverage = dict(coverage) if isinstance(coverage, dict) else {}
        for key, value in relocated.items():
            if key in coverage and not _same_json(coverage[key], value):
                return arguments
            coverage[key] = value
        normalized["_coverage"] = coverage
    return arguments if _same_json(normalized, arguments) else normalized
