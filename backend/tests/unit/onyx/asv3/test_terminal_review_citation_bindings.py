"""Selected terminal reviews can use only their own invocation-delivered originals."""

import copy
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.legal_source_reviews import RelatedSourceReview
from onyx.asv3.models import RunContext
from onyx.asv3.registry import _inline_metadata_schema
from onyx.asv3.terminal_wire_schema import (
    bind_terminal_review_citations,
    decode_optional_nulls,
    strict_terminal_tools,
)
from tests.unit.onyx.asv3.test_terminal_wire_schema import (
    assert_closed,
    context,
    parameters,
    tools,
)

A, B, C, D = ("lead_" + letter * 64 for letter in "abcd")


def selected() -> list[dict[str, JsonValue]]:
    definitions = tools()
    properties = parameters(definitions)["properties"]
    assert isinstance(properties, dict)
    item = _inline_metadata_schema(RelatedSourceReview.model_json_schema())
    fields = item["properties"]
    assert isinstance(fields, dict)
    lead = fields["lead_id"]
    assert isinstance(lead, dict)
    lead["enum"] = [A, B, C, D]
    properties["_related_source_reviews"] = {"type": "array", "items": item}
    return definitions


def arguments(lead: str, status: str, citation: int | None) -> dict[str, JsonValue]:
    return {
        "answer": "Exact retained body, effect and limitations [1] [2].\n",
        "basis": "originals",
        "_language": None,
        "_outcomes": None,
        "_coverage": None,
        "_related_source_reviews": [
            {
                "lead_id": lead,
                "status": status,
                "source_role": "operative_text" if status == "examined" else "unknown",
                "effect": "Own source operative effect",
                "limitations": "Own source scope limitation",
                "witnesses": None
                if citation is None
                else [{"citation": citation, "start_char": None, "end_char": 10}],
                "gap": "Precise unresolved interaction"
                if status == "unresolved"
                else "",
            }
        ],
    }


@pytest.mark.parametrize(
    "lead,status,citation,valid",
    [
        (A, "examined", 1, True),
        (A, "examined", 4, True),
        (A, "examined", 2, False),
        (B, "not_material", 2, True),
        (B, "not_material", 1, False),
        (A, "unresolved", 1, True),
        (A, "unresolved", 2, False),
        (A, "examined", None, False),
        (C, "examined", 1, False),
        (C, "not_material", None, False),
        (C, "unresolved", None, True),
        (C, "unresolved", 1, False),
        (D, "unresolved", None, True),
        (D, "examined", 2, False),
        ("lead_" + "e" * 64, "unresolved", None, False),
    ],
)
def test_selected_and_strict_wire_bind_the_same_lead_to_own_originals(
    lead: str, status: str, citation: int | None, valid: bool
) -> None:
    canonical = selected()
    before = copy.deepcopy(canonical)
    bound = bind_terminal_review_citations(
        canonical, context(), {A: (1, 4), B: (2,), C: ()}
    )
    wire = strict_terminal_tools(bound, context(), "openai")
    schema = parameters(wire)
    assert_closed(schema)
    jsonschema.Draft202012Validator.check_schema(schema)
    args = arguments(lead, status, citation)
    assert jsonschema.Draft202012Validator(schema).is_valid(args) is valid
    if valid:
        decoded = decode_optional_nulls(
            "submit_answer", args, parameters(bound), context(), "openai"
        )
        jsonschema.Draft202012Validator(parameters(bound)).validate(decoded)
        assert decoded["answer"] == args["answer"]
        reviews = cast(list[dict[str, JsonValue]], decoded["_related_source_reviews"])
        assert reviews[0]["lead_id"] == lead
        if citation is not None:
            assert reviews[0]["witnesses"] == [{"citation": citation, "end_char": 10}]
    assert canonical == before


def test_own_citation_binding_never_approves_role_gap_or_missing_witness() -> None:
    wire = strict_terminal_tools(
        bind_terminal_review_citations(selected(), context(), {A: (1,)}),
        context(),
        "openai",
    )
    base = arguments(A, "examined", 1)
    for key, value in [
        ("source_role", "argument_only"),
        ("gap", "Open effect"),
        ("witnesses", []),
    ]:
        args = copy.deepcopy(base)
        reviews = cast(list[dict[str, JsonValue]], args["_related_source_reviews"])
        reviews[0][key] = value
        assert not jsonschema.Draft202012Validator(parameters(wire)).is_valid(args)


def test_equal_citation_sets_share_a_branch_without_source_scope_cuts() -> None:
    binding = {A: (4, 1), B: (1, 4), C: (), "lead_" + "f" * 64: (999,)}
    bound = bind_terminal_review_citations(selected(), context(), binding)
    fields = parameters(bound)["properties"]
    assert isinstance(fields, dict)
    reviews = fields["_related_source_reviews"]
    assert isinstance(reviews, dict)
    item = reviews["items"]
    assert isinstance(item, dict)
    branches = item["anyOf"]
    assert isinstance(branches, list) and len(branches) == 2
    binding[A] = (999,)
    decoded = decode_optional_nulls(
        "submit_answer",
        arguments(A, "examined", 1),
        parameters(bound),
        context(),
        "openai",
    )
    assert jsonschema.Draft202012Validator(parameters(bound)).is_valid(decoded)
    wrong = decode_optional_nulls(
        "submit_answer",
        arguments(A, "examined", 999),
        parameters(bound),
        context(),
        "openai",
    )
    assert not jsonschema.Draft202012Validator(parameters(bound)).is_valid(wrong)


def test_existing_host_citation_enum_is_never_widened() -> None:
    host = selected()
    fields = cast(dict[str, JsonValue], parameters(host)["properties"])
    reviews = cast(dict[str, JsonValue], fields["_related_source_reviews"])
    item = cast(dict[str, JsonValue], reviews["items"])
    review_fields = cast(dict[str, JsonValue], item["properties"])
    witness_array = cast(dict[str, JsonValue], review_fields["witnesses"])
    witness = cast(dict[str, JsonValue], witness_array["items"])
    witness_fields = cast(dict[str, JsonValue], witness["properties"])
    citation = cast(dict[str, JsonValue], witness_fields["citation"])
    citation["enum"] = [4]
    wire = strict_terminal_tools(
        bind_terminal_review_citations(host, context(), {A: (1, 4)}),
        context(),
        "openai",
    )
    assert jsonschema.Draft202012Validator(parameters(wire)).is_valid(
        arguments(A, "examined", 4)
    )
    assert not jsonschema.Draft202012Validator(parameters(wire)).is_valid(
        arguments(A, "examined", 1)
    )


def test_each_shrinking_fit_uses_only_its_own_frozen_citation_map() -> None:
    unbound = selected()
    before = copy.deepcopy(unbound)
    full = strict_terminal_tools(
        bind_terminal_review_citations(unbound, context(), {A: (1, 4)}),
        context(),
        "openai",
    )
    shrunk = strict_terminal_tools(
        bind_terminal_review_citations(unbound, context(), {A: (1,)}),
        context(),
        "openai",
    )
    empty = strict_terminal_tools(
        bind_terminal_review_citations(unbound, context(), {}), context(), "openai"
    )
    assert jsonschema.Draft202012Validator(parameters(full)).is_valid(
        arguments(A, "examined", 4)
    )
    assert not jsonschema.Draft202012Validator(parameters(shrunk)).is_valid(
        arguments(A, "examined", 4)
    )
    assert jsonschema.Draft202012Validator(parameters(shrunk)).is_valid(
        arguments(A, "examined", 1)
    )
    assert not jsonschema.Draft202012Validator(parameters(empty)).is_valid(
        arguments(A, "examined", 1)
    )
    assert jsonschema.Draft202012Validator(parameters(empty)).is_valid(
        arguments(A, "unresolved", None)
    )
    assert jsonschema.Draft202012Validator(parameters(full)).is_valid(
        arguments(A, "examined", 4)
    )
    assert unbound == before


def test_trusted_hosted_scope_is_bound_and_plain_serial_is_exact() -> None:
    host = selected()
    assert bind_terminal_review_citations(host, context(False), {A: (1,)}) is host
    hosted = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": False,
            "serial_session_diagnostics": True,
            "lean_native_mode": True,
            "task_id": "owned-child",
        }
    )
    assert bind_terminal_review_citations(host, hosted, {A: (1,)}) != host


def test_unsupported_custom_and_zero_lead_schema_remain_unchanged() -> None:
    host = selected()
    fields = cast(dict[str, JsonValue], parameters(host)["properties"])
    fields["_related_source_reviews"] = {"type": "array", "items": {"type": "string"}}
    assert bind_terminal_review_citations(host, context(), {A: (1,)}) == host
    host = selected()
    fields = cast(dict[str, JsonValue], parameters(host)["properties"])
    reviews = cast(dict[str, JsonValue], fields["_related_source_reviews"])
    reviews["maxItems"] = 0
    assert bind_terminal_review_citations(host, context(), {}) == host
