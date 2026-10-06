"""Provider-only constrained terminal arguments; host validation stays authoritative."""

from __future__ import annotations

import copy

import jsonschema
from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.asv3.parallel_execution import parallel_execution_enabled
from onyx.llm.model_capabilities import is_true_openai_model

TERMINALS = frozenset(
    {
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
    }
)
_UNSUPPORTED = frozenset(
    {"oneOf", "allOf", "not", "if", "then", "else", "patternProperties", "$ref"}
)


class _UnsupportedSchema(ValueError):
    pass


def _nullable(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"anyOf": [schema, {"type": "null"}]}


def _compile(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if set(schema) & _UNSUPPORTED:
        raise _UnsupportedSchema("Terminal schema is not a supported closed schema")
    result = copy.deepcopy(schema)
    result.pop("default", None)
    if "const" in result:
        result["enum"] = [result.pop("const")]
    alternatives = result.get("anyOf")
    if isinstance(alternatives, list):
        if any(not isinstance(item, dict) for item in alternatives):
            raise _UnsupportedSchema("Alternative schemas must be objects")
        result["anyOf"] = [
            _compile(item) if isinstance(item, dict) else item for item in alternatives
        ]
    if result.get("type") == "object":
        properties = result.get("properties")
        if (
            not isinstance(properties, dict)
            or result.get("additionalProperties") is not False
        ):
            raise _UnsupportedSchema(
                "Strict terminal objects must have closed named fields"
            )
        required = result.get("required", [])
        if not isinstance(required, list):
            raise _UnsupportedSchema("Required fields must be a list")
        compiled: dict[str, JsonValue] = {}
        for key, value in properties.items():
            if not isinstance(value, dict):
                raise _UnsupportedSchema("Field schema must be an object")
            field = _compile(value)
            compiled[key] = field if key in required else _nullable(field)
        result["properties"] = compiled
        result["required"] = list(compiled)
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = _compile(items)
    return result


def _bind_resolution_shapes(properties: dict[str, JsonValue]) -> None:
    coverage = properties.get("_coverage")
    fields = coverage.get("properties") if isinstance(coverage, dict) else None
    resolutions = fields.get("resolutions") if isinstance(fields, dict) else None
    items = resolutions.get("items") if isinstance(resolutions, dict) else None
    parts = items.get("properties") if isinstance(items, dict) else None
    if not isinstance(parts, dict) or not isinstance(items, dict):
        return
    status, numbers, gap = (
        parts.get(key) for key in ("status", "evidence_numbers", "gap")
    )
    if not (
        isinstance(status, dict)
        and status.get("enum") == ["supported", "conditional", "unresolved"]
        and isinstance(numbers, dict)
        and numbers.get("type") == "array"
        and isinstance(gap, dict)
        and gap.get("type") == "string"
    ):
        return
    branches: list[JsonValue] = []
    for states in (["supported", "conditional"], ["unresolved"]):
        branch = copy.deepcopy(items)
        branch_fields = branch["properties"]
        assert isinstance(branch_fields, dict)
        branch_fields["status"] = {**status, "enum": states}
        required = branch.get("required", [])
        assert isinstance(required, list)
        if states == ["unresolved"]:
            branch_fields["gap"] = {**gap, "minLength": 1}
            branch["required"] = list(dict.fromkeys([*required, "gap"]))
        else:
            branch_fields["evidence_numbers"] = {**numbers, "minItems": 1}
            branch_fields["gap"] = {**gap, "enum": [""]}
            branch["required"] = list(
                dict.fromkeys([*required, "evidence_numbers", "gap"])
            )
        branches.append(branch)
    assert isinstance(resolutions, dict)
    resolutions["items"] = {"anyOf": branches}


def _bind_review_shapes(properties: dict[str, JsonValue]) -> None:
    reviews = properties.get("_related_source_reviews")
    items = reviews.get("items") if isinstance(reviews, dict) else None
    fields = items.get("properties") if isinstance(items, dict) else None
    if not isinstance(fields, dict) or not isinstance(items, dict):
        return
    status, role, witnesses, gap = (
        fields.get(key) for key in ("status", "source_role", "witnesses", "gap")
    )
    if not (
        isinstance(status, dict)
        and status.get("enum") == ["examined", "not_material", "unresolved"]
        and isinstance(role, dict)
        and isinstance(witnesses, dict)
        and witnesses.get("type") == "array"
        and isinstance(gap, dict)
        and gap.get("type") == "string"
    ):
        return
    branches: list[JsonValue] = []
    for state in ("examined", "not_material", "unresolved"):
        branch = copy.deepcopy(items)
        parts = branch["properties"]
        assert isinstance(parts, dict)
        parts["status"] = {**status, "enum": [state]}
        required = branch.get("required", [])
        assert isinstance(required, list)
        if state == "unresolved":
            parts["gap"] = {**gap, "minLength": 1}
            branch["required"] = list(dict.fromkeys([*required, "gap"]))
        else:
            parts["witnesses"] = {**witnesses, "minItems": 1}
            parts["gap"] = {**gap, "enum": [""]}
            branch["required"] = list(dict.fromkeys([*required, "witnesses", "gap"]))
            if state == "examined":
                parts["source_role"] = {**role, "enum": ["operative_text"]}
        branches.append(branch)
    assert isinstance(reviews, dict)
    reviews["items"] = {"anyOf": branches}


def _wire_parameters(
    name: str, host: dict[str, JsonValue]
) -> dict[str, JsonValue] | None:
    parameters = copy.deepcopy(host)
    properties = parameters.get("properties")
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return None
    # Language notification maps are established on the first useful research action.
    if "_notifications" not in required:
        properties.pop("_notifications", None)
    if name in {"submit_answer", "submit_partial_answer"}:
        # Retained-body actions make the alternative explicit in the tool identity.
        if "answer" not in properties:
            return None
        properties.pop("retained_answer_id", None)
        properties.pop("retained_answer_edits", None)
        parameters["required"] = list(dict.fromkeys([*required, "answer"]))
    _bind_resolution_shapes(properties)
    _bind_review_shapes(properties)
    try:
        return _compile(parameters)
    except _UnsupportedSchema:
        return None


def strict_terminal_tools(
    tools: list[dict[str, JsonValue]],
    context: RunContext,
    provider: str,
    model_name: str | None = None,
) -> list[dict[str, JsonValue]]:
    """Constrain supported OpenAI terminal calls without mutating canonical schemas."""
    if (
        provider != "openai"
        or not parallel_execution_enabled(context)
        or (model_name is not None and not is_true_openai_model(provider, model_name))
    ):
        return tools
    wire = copy.deepcopy(tools)
    for tool in wire:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(function, dict)
            or not isinstance(name, str)
            or name not in TERMINALS
        ):
            continue
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        constrained = _wire_parameters(name, parameters)
        if constrained is not None:
            function["parameters"] = constrained
            function["strict"] = True
    return wire


def decode_optional_nulls(
    name: str,
    arguments: dict[str, JsonValue],
    parameters: dict[str, JsonValue],
    context: RunContext,
    provider: str,
    model_name: str | None = None,
) -> dict[str, JsonValue]:
    """Decode optional null sentinels only; preserve all substantive supplied values."""
    if (
        name not in TERMINALS
        or provider != "openai"
        or not parallel_execution_enabled(context)
        or (model_name is not None and not is_true_openai_model(provider, model_name))
        or _wire_parameters(name, parameters) is None
    ):
        return arguments

    def decode(value: JsonValue, schema: dict[str, JsonValue]) -> JsonValue:
        alternatives = schema.get("anyOf")
        if isinstance(alternatives, list):
            for alternative in alternatives:
                if not isinstance(alternative, dict):
                    continue
                decoded = decode(value, alternative)
                if jsonschema.Draft202012Validator(alternative).is_valid(decoded):
                    return decoded
            return value
        if isinstance(value, dict):
            properties = schema.get("properties")
            required = schema.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                return value
            result: dict[str, JsonValue] = {}
            for key, part in value.items():
                field = properties.get(key)
                if not isinstance(field, dict):
                    result[key] = part
                    continue
                if part is None and key not in required:
                    # A field whose host schema admits null has a semantic null value.
                    if not jsonschema.Draft202012Validator(field).is_valid(None):
                        continue
                result[key] = decode(part, field)
            return result
        items = schema.get("items")
        if isinstance(value, list) and isinstance(items, dict):
            return [decode(part, items) for part in value]
        return value

    decoded = decode(arguments, parameters)
    assert isinstance(decoded, dict)
    return decoded
