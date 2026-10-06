"""Structural terminal metadata fixes preserve values, provenance and legal gates."""

import copy
import hashlib
import json
import re
from typing import Any

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import COORDINATOR_SESSION_ACTIONS, ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.registry import _outcome_metadata_properties
from onyx.asv3.terminal_metadata import normalize_terminal_metadata
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_COORDINATOR_PROMPT,
    EXPERIMENTAL_PARALLEL_COORDINATOR,
    SOURCE_NAVIGATION,
    parallel_metadata_instructions,
)
from tests.unit.onyx.asv3.test_model_adapter import (
    scripted_model,
    text_response,
    tool_response,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_native_model_adapter import view
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    declaration,
    outer,
    serial_run,
    session,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def _context(scope: str = "parallel") -> RunContext:
    return RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": scope == "parallel",
            "serial_session_diagnostics": scope == "hosted",
            "lean_native_mode": True,
            "task_id": "owned-task",
            "scenario_request": "Explain the material condition.",
        }
    )


def _parameters() -> dict[str, JsonValue]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "basis": {"type": "string", "enum": ["originals"]},
            "_language": {"type": "string", "enum": ["tr"]},
            "_related_source_reviews": {"type": "array", "maxItems": 0},
            **_outcome_metadata_properties(),
        },
        "required": ["answer", "basis", "_language"],
    }


@pytest.mark.parametrize("scope", ["parallel", "hosted"])
def test_aliases_and_nesting_preserve_all_semantic_values(scope: str) -> None:
    body = "Exact multiline body [1].\n  Keep whitespace and alternatives."
    declaration_row = declaration()
    resolution: dict[str, JsonValue] = {
        "outcome_id": "local-result",
        "status": "unresolved",
        "gap": "Exact missing original.",
    }
    args: dict[str, JsonValue] = {
        "answer": body,
        "basis": "originals",
        " _language ": "tr",
        "outcomes": [declaration_row],
        "related_source_reviews": [],
        " _coverage": {"conditions": []},
        "resolutions": [resolution],
    }
    before = copy.deepcopy(args)
    normalized = normalize_terminal_metadata(
        "submit_answer", args, _context(scope), _parameters()
    )
    jsonschema.Draft202012Validator(_parameters()).validate(normalized)
    assert normalized == {
        "answer": body,
        "basis": "originals",
        "_language": "tr",
        "_outcomes": [declaration_row],
        "_related_source_reviews": [],
        "_coverage": {"conditions": [], "resolutions": [resolution]},
    }
    assert args == before


@pytest.mark.parametrize(
    "conflict", ["outcomes", "coverage", "nested", "bool-int", "float-int", "null"]
)
def test_conflicts_leave_original_arguments_invalid(conflict: str) -> None:
    args: dict[str, JsonValue] = {"outcomes": [declaration()]}
    if conflict == "outcomes":
        args["_outcomes"] = []
    elif conflict == "coverage":
        args.update(_coverage={"resolutions": []}, coverage={"conditions": []})
    elif conflict == "nested":
        args["_coverage"] = {
            "resolutions": [],
            " resolutions": [{"outcome_id": "foreign"}],
        }
    elif conflict in {"bool-int", "float-int"}:
        args["_outcomes"] = [True if conflict == "bool-int" else 1.0]
        args["outcomes"] = [1]
    else:
        args.update(_coverage=None, resolutions=[])
    assert (
        normalize_terminal_metadata("submit_answer", args, _context(), _parameters())
        is args
    )
    assert not jsonschema.Draft202012Validator(_parameters()).is_valid(args)


def test_exact_duplicates_collapse_unknowns_and_invalid_ids_are_retained() -> None:
    row = {**declaration(), "outcome_id": "q0:d0"}
    args: dict[str, JsonValue] = {
        "answer": "Original body.",
        "basis": "originals",
        "_language": "tr",
        "outcomes": [row],
        "_outcomes": [copy.deepcopy(row)],
        "resolutions": [],
        "_coverage": {" resolutions": []},
        "foreign_metadata": {"condition": "Never discard this."},
    }
    normalized = normalize_terminal_metadata(
        "submit_answer", args, _context(), _parameters()
    )
    assert normalized["_outcomes"] == [row]
    assert normalized["_coverage"] == {"resolutions": []}
    assert normalized["foreign_metadata"] == args["foreign_metadata"]
    assert not jsonschema.Draft202012Validator(_parameters()).is_valid(normalized)
    del normalized["foreign_metadata"]
    errors = list(
        jsonschema.Draft202012Validator(_parameters()).iter_errors(normalized)
    )
    assert any(error.validator == "pattern" for error in errors)


@pytest.mark.parametrize(
    "case",
    ["plain", "normal", "other_tool", "unexposed", "actual_alias", "unexposed_basis"],
)
def test_normalization_uses_only_current_exposed_terminal_contract(case: str) -> None:
    context = _context("plain" if case == "plain" else "parallel")
    name = "submit_answer"
    parameters: dict[str, JsonValue] = _parameters()
    args: dict[str, JsonValue] = {"outcomes": [declaration()]}
    if case == "normal":
        context.services["research_profile"] = "normal"
    elif case == "other_tool":
        name = "update_research"
    elif case == "unexposed":
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    elif case == "actual_alias":
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        properties["outcomes"] = {"type": "array"}
    elif case == "unexposed_basis":
        name = "submit_partial_answer"
        parameters = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        }
        args = {"answer": "Exact", "basis": "originals"}
    assert normalize_terminal_metadata(name, args, context, parameters) is args


@pytest.mark.parametrize("missing", [False, True])
def test_hosted_one_decision_normalizes_wire_without_approving_missing_delivery(
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.selected.reset_mock()
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    own = original("own", "The actual operative original.")
    ledger.add(
        [
            own,
            original(
                "unseen",
                "Not delivered here.",
                source="other",
                headings=["Other instrument"],
            ),
        ],
        child.context,
    )
    child.harness.evidence_working_set.remember(1, 0, len(own.text))
    body = f"The precise original condition applies [{2 if missing else 1}]."
    raw = json.dumps(
        {
            "answer": body,
            "basis": "originals",
            "_language": "tr",
            "outcomes": [declaration()],
            "related_source_reviews": [],
            "resolutions": [
                {
                    "outcome_id": "local-result",
                    "status": "unresolved",
                    "gap": "Exact missing continuation.",
                }
            ],
        }
    )

    def invoke(**arguments: Any) -> Any:
        assert {row["citation"] for row in actual_originals(arguments["prompt"])} == {1}
        return tool_response(raw, "submit_answer")

    run.selected.invoke.side_effect = invoke
    decision = child.model.decide(child.harness.view())
    assert run.selected.invoke.call_count == 1
    call = decision.calls[0]
    assert call.argument_error is None and call.arguments["answer"] == body
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert decision.assistant_message.tool_calls[0].function.arguments == raw
    principal = child.model.last_call_id or ""
    assert child.context.services["last_model_call_id"] == principal
    assert ledger.completely_delivered(principal) == {1}
    child._on_decision(decision)
    receipts = child.harness._dispatch(decision.calls)
    if missing:
        assert receipts[0].outcome.status == OutcomeStatus.PARTIAL
        assert receipts[0].outcome.data["undelivered_citations"] == [2]
        assert child.context.services.get("submitted_answer") is None
    else:
        assert receipts[0].outcome.status == OutcomeStatus.FOUND
        assert child.context.services["submitted_answer"] == body


def test_real_small_metadata_patch_keeps_source_bearing_origin_and_body() -> None:
    context, ledger = _context(), EvidenceLedger()
    context.services["evidence"] = ledger
    own = original("own", "The actual operative original.")
    ledger.add([own], context)
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    body = "The exact original condition applies [1].\n  Exact end."
    definitions: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {"name": "submit_answer", "parameters": _parameters()},
        }
    ]
    current = view(original_evidence=[full_record(ledger, 1)])
    current.tools = definitions
    semantic_id = ""

    def invoke(**arguments: Any) -> Any:
        nonlocal semantic_id
        if selected.invoke.call_count == 1:
            assert {
                row["citation"] for row in actual_originals(arguments["prompt"])
            } == {1}
            return tool_response(
                json.dumps(
                    {
                        "answer": body,
                        "basis": "originals",
                        "_language": "de",
                        "outcomes": [declaration()],
                    }
                ),
                "submit_answer",
            )
        semantic_id = adapter.last_call_id or ""
        assert actual_originals(arguments["prompt"]) == []
        payload = json.loads(arguments["prompt"][-1].content)
        identity = payload["invalid_actions"][0]["call_id"]
        return text_response(
            {
                "entries": [
                    {
                        "call_id": identity,
                        "arguments_json": json.dumps(
                            {
                                "basis": "originals",
                                "_language": "tr",
                                "outcomes": [declaration()],
                            }
                        ),
                    }
                ],
            }
        )

    selected.invoke.side_effect = invoke
    decision = adapter.decide(current)
    assert selected.invoke.call_count == 2
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments["answer"] == body
    assert decision.calls[0].arguments["_outcomes"] == [declaration()]
    assert adapter.last_call_id == context.services["last_model_call_id"] == semantic_id
    assert ledger.completely_delivered(semantic_id) == {1}


@pytest.mark.parametrize("scope", ["parallel", "hosted", "plain"])
def test_ignored_patch_keeps_unexpected_field_diagnostic_and_original_wire(
    scope: str,
) -> None:
    context, ledger = _context(scope), EvidenceLedger()
    context.services["evidence"] = ledger
    ledger.add([original("own", "The actual operative original.")], context)
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    body = "Exact original answer [1]."
    raw = json.dumps(
        {
            "answer": body,
            "basis": "originals",
            "_language": "tr",
            "unrecognized_metadata": True,
        }
    )
    current = view(original_evidence=[full_record(ledger, 1)])
    current.tools = [
        {
            "type": "function",
            "function": {"name": "submit_answer", "parameters": _parameters()},
        }
    ]
    semantic_id = ""

    def invoke(**arguments: Any) -> Any:
        nonlocal semantic_id
        if selected.invoke.call_count == 1:
            assert {
                row["citation"] for row in actual_originals(arguments["prompt"])
            } == {1}
            return tool_response(raw, "submit_answer")
        semantic_id = adapter.last_call_id or ""
        assert actual_originals(arguments["prompt"]) == []
        return text_response({"entries": []})

    selected.invoke.side_effect = invoke
    decision = adapter.decide(current)
    assert selected.invoke.call_count == 2
    call = decision.calls[0]
    assert (
        call.arguments["answer"] == body
        and call.arguments["unrecognized_metadata"] is True
    )
    assert call.argument_error and ("unrecognized_metadata" in call.argument_error) == (
        scope != "plain"
    )
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert decision.assistant_message.tool_calls[0].function.arguments == raw
    assert ledger.completely_delivered(semantic_id) == {1}
    if scope != "plain":
        assert (
            adapter.last_call_id
            == context.services["last_model_call_id"]
            == semantic_id
        )


@pytest.mark.parametrize("scope", ["parallel", "hosted", "plain", "normal"])
def test_only_scoped_metadata_paragraph_changes_without_legal_instruction_delta(
    scope: str,
) -> None:
    context = _context(scope)
    if scope == "normal":
        context.services["research_profile"] = "normal"
    adapter = ResearchModel(scripted_model(), context, lean_native_mode=True)
    baseline = EXPERIMENTAL_COORDINATOR_PROMPT + "\n\n" + COORDINATOR_SESSION_ACTIONS
    assert (
        hashlib.sha256(baseline.encode()).hexdigest()
        == "bf8b166e5daca156751835eac0f02d50b2e76602f50971f62916e20aea2fea83"
    )
    changed = parallel_metadata_instructions(baseline)
    start, end = (
        "OUTCOME COMPLETENESS IN THE SAME DECISION\n",
        "Before submission compare every actual requested outcome",
    )
    before, _, rest = baseline.partition(start)
    old, _, after = rest.partition(end)
    new = changed[len(before + start) : -len(end + after)]
    assert changed.startswith(before + start) and changed.endswith(end + after)
    assert len(new) <= len(old) and not re.search(r"\d", new)
    assert SOURCE_NAVIGATION in baseline and SOURCE_NAVIGATION in changed
    if scope in {"parallel", "hosted"}:
        expected = changed + (
            "\n\n" + EXPERIMENTAL_PARALLEL_COORDINATOR if scope == "parallel" else ""
        )
        assert adapter._research_instruction() == expected
    elif scope == "plain":
        assert adapter._research_instruction() == baseline
    else:
        reference = ResearchModel(
            scripted_model(),
            RunContext(services={"research_profile": "normal"}),
            lean_native_mode=True,
        )
        assert adapter._research_instruction() == reference._research_instruction()
