"""Constrained provider arguments keep exact bodies and authoritative host validation."""

import copy
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.legal_source_reviews import RelatedSourceReview
from onyx.asv3.models import RunContext
from onyx.asv3.registry import _inline_metadata_schema, _outcome_metadata_properties
from onyx.asv3.terminal_wire_schema import decode_optional_nulls, strict_terminal_tools


def context(parallel: bool = True) -> RunContext:
    return RunContext(
        services={"research_profile": "experimental", "experimental_parallel": parallel}
    )


def tools(name: str = "submit_answer") -> list[dict[str, JsonValue]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {"type": "string", "enum": ["originals"]},
                        "_language": {"type": "string"},
                        "_notifications": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        **_outcome_metadata_properties(),
                    },
                    "required": ["answer", "basis"],
                },
            },
        }
    ]


def parameters(definitions: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    function = definitions[0]["function"]
    assert isinstance(function, dict)
    return cast(dict[str, JsonValue], function["parameters"])


def assert_closed(schema: JsonValue) -> None:
    if isinstance(schema, list):
        for child in schema:
            assert_closed(child)
    elif isinstance(schema, dict):
        if schema.get("type") == "object":
            fields = schema["properties"]
            assert isinstance(fields, dict)
            assert schema["additionalProperties"] is False
            assert schema["required"] == list(fields)
        for child in schema.values():
            assert_closed(child)


def test_wire_is_closed_nullable_and_does_not_mutate_host_schema() -> None:
    host = tools()
    before = copy.deepcopy(host)
    wire = strict_terminal_tools(host, context(), "openai")
    assert host == before
    assert wire != host
    function = wire[0]["function"]
    assert isinstance(function, dict) and function["strict"] is True
    schema = parameters(wire)
    assert_closed(schema)
    jsonschema.Draft202012Validator.check_schema(schema)
    fields = schema["properties"]
    assert isinstance(fields, dict) and "_notifications" not in fields
    args: dict[str, JsonValue] = {
        "answer": "  Exact body [1].\n\nAll details and whitespace.\n",
        "basis": "originals",
        "_language": None,
        "_outcomes": None,
        "_coverage": None,
    }
    jsonschema.Draft202012Validator(schema).validate(args)
    decoded = decode_optional_nulls(
        "submit_answer", args, parameters(host), context(), "openai"
    )
    assert decoded == {"answer": args["answer"], "basis": "originals"}
    jsonschema.Draft202012Validator(parameters(host)).validate(decoded)
    assert args["_outcomes"] is None


@pytest.mark.parametrize(
    "provider,parallel", [("vertex_ai", True), ("openai", False), ("anthropic", True)]
)
def test_other_provider_and_workflow_payloads_remain_exact(
    provider: str, parallel: bool
) -> None:
    host = tools()
    assert strict_terminal_tools(host, context(parallel), provider) is host
    args: dict[str, JsonValue] = {"answer": "body", "_outcomes": None}
    assert (
        decode_optional_nulls(
            "submit_answer", args, parameters(host), context(parallel), provider
        )
        is args
    )


@pytest.mark.parametrize("native", [True, False])
def test_actual_model_identity_gates_openai_compatible_servers(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    model = "gpt-6-luna" if native else "compatible-model"
    identities: list[tuple[str, str]] = []

    def is_native(provider: str, model_name: str) -> bool:
        identities.append((provider, model_name))
        return native

    monkeypatch.setattr(
        "onyx.asv3.terminal_wire_schema.is_true_openai_model", is_native
    )
    host = tools()
    wire = strict_terminal_tools(host, context(), "openai", model)
    args: dict[str, JsonValue] = {
        "answer": "Exact original body [1].\n",
        "basis": "originals",
        "_outcomes": None,
    }
    decoded = decode_optional_nulls(
        "submit_answer", args, parameters(host), context(), "openai", model
    )
    assert identities == [("openai", model), ("openai", model)]
    if native:
        assert wire is not host
        assert decoded == {"answer": args["answer"], "basis": "originals"}
    else:
        assert wire is host
        assert decoded is args


@pytest.mark.parametrize("status", ["supported", "conditional"])
@pytest.mark.parametrize(
    "numbers,gap,valid", [([1], "", True), ([], "", False), ([1], "unread rule", False)]
)
def test_positive_resolution_shape_cannot_claim_support_with_open_gap(
    status: str, numbers: list[int], gap: str, valid: bool
) -> None:
    wire = strict_terminal_tools(tools(), context(), "openai")
    args: dict[str, JsonValue] = {
        "answer": "body [1]",
        "basis": "originals",
        "_language": None,
        "_outcomes": None,
        "_coverage": {
            "conditions": None,
            "resolutions": [
                {
                    "outcome_id": "result",
                    "status": status,
                    "condition_ids": None,
                    "evidence_numbers": cast(list[JsonValue], numbers),
                    "gap": gap,
                }
            ],
        },
    }
    assert jsonschema.Draft202012Validator(parameters(wire)).is_valid(args) is valid


def test_decoder_preserves_required_null_unknown_fields_and_semantic_null() -> None:
    host = parameters(tools())
    fields = host["properties"]
    assert isinstance(fields, dict)
    fields["semantic"] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    args: dict[str, JsonValue] = {"answer": None, "unknown": None, "semantic": None}
    decoded = decode_optional_nulls("submit_answer", args, host, context(), "openai")
    assert decoded == args
    assert not jsonschema.Draft202012Validator(host).is_valid(decoded)


def test_recursive_decoder_keeps_source_witness_and_all_nonnull_values() -> None:
    args: dict[str, JsonValue] = {
        "answer": " Exact body [7].\n",
        "basis": "originals",
        "_outcomes": [
            {
                "outcome_id": "result",
                "question_ids": ["q0"],
                "detail": "condition",
                "decisive_facts": None,
            }
        ],
        "_coverage": {
            "conditions": [
                {
                    "condition_id": "proof",
                    "outcome_ids": ["result"],
                    "detail": "actual condition",
                    "witnesses": [{"citation": 7, "start_char": None, "end_char": 123}],
                }
            ],
            "resolutions": [
                {
                    "outcome_id": "result",
                    "status": "supported",
                    "condition_ids": ["proof"],
                    "evidence_numbers": [7],
                    "gap": "",
                }
            ],
        },
    }
    decoded = decode_optional_nulls(
        "submit_answer", args, parameters(tools()), context(), "openai"
    )
    jsonschema.Draft202012Validator(parameters(tools())).validate(decoded)
    assert decoded["answer"] == args["answer"]
    coverage = decoded["_coverage"]
    assert isinstance(coverage, dict)
    conditions = coverage["conditions"]
    assert isinstance(conditions, list)
    condition = conditions[0]
    assert isinstance(condition, dict)
    assert condition["witnesses"] == [{"citation": 7, "end_char": 123}]
    assert (
        coverage["resolutions"]
        == cast(dict[str, JsonValue], args["_coverage"])["resolutions"]
    )


def test_unsupported_required_free_map_stays_on_existing_wire_path() -> None:
    host = tools()
    schema = parameters(host)
    required = schema["required"]
    assert isinstance(required, list)
    required.append("_notifications")
    wire = strict_terminal_tools(host, context(), "openai")
    assert wire == host
    function = wire[0]["function"]
    assert isinstance(function, dict) and "strict" not in function


def test_full_body_action_cannot_be_metadata_only_after_retained_binding() -> None:
    host = tools()
    schema = parameters(host)
    fields = schema["properties"]
    assert isinstance(fields, dict)
    fields["retained_answer_id"] = {"type": "string", "enum": ["owned"]}
    schema["required"] = ["basis"]
    wire = strict_terminal_tools(host, context(), "openai")
    wire_fields = parameters(wire)["properties"]
    assert isinstance(wire_fields, dict) and "retained_answer_id" not in wire_fields
    assert "answer" in cast(list[JsonValue], parameters(wire)["required"])
    assert schema["required"] == ["basis"]


@pytest.mark.parametrize(
    "status,role,witnesses,gap,valid",
    [
        (
            "examined",
            "operative_text",
            [{"citation": 1, "start_char": 0, "end_char": 10}],
            "",
            True,
        ),
        (
            "examined",
            "unknown",
            [{"citation": 1, "start_char": 0, "end_char": 10}],
            "",
            False,
        ),
        ("not_material", "unknown", [], "", False),
        (
            "not_material",
            "unknown",
            [{"citation": 1, "start_char": 0, "end_char": 10}],
            "",
            True,
        ),
        (
            "examined",
            "operative_text",
            [{"citation": 1, "start_char": 0, "end_char": 10}],
            "unexamined effect",
            False,
        ),
        ("unresolved", "unknown", [], "exact unresolved interaction", True),
        ("unresolved", "unknown", [], "", False),
    ],
)
def test_review_wire_shape_mirrors_existing_closed_and_open_review_guards(
    status: str, role: str, witnesses: list[dict[str, JsonValue]], gap: str, valid: bool
) -> None:
    host = tools()
    fields = parameters(host)["properties"]
    assert isinstance(fields, dict)
    fields["_related_source_reviews"] = {
        "type": "array",
        "items": _inline_metadata_schema(RelatedSourceReview.model_json_schema()),
    }
    before = copy.deepcopy(host)
    wire = strict_terminal_tools(host, context(), "openai")
    args: dict[str, JsonValue] = {
        "answer": "Actual complete body [1]",
        "basis": "originals",
        "_language": None,
        "_outcomes": None,
        "_coverage": None,
        "_related_source_reviews": [
            {
                "lead_id": "lead_" + "a" * 64,
                "status": status,
                "source_role": role,
                "effect": "actual effect",
                "limitations": "actual scope",
                "witnesses": cast(list[JsonValue], witnesses),
                "gap": gap,
            }
        ],
    }
    assert host == before
    assert_closed(parameters(wire))
    assert jsonschema.Draft202012Validator(parameters(wire)).is_valid(args) is valid
    if valid:
        decoded = decode_optional_nulls(
            "submit_answer", args, parameters(host), context(), "openai"
        )
        assert decoded["_related_source_reviews"] == args["_related_source_reviews"]
