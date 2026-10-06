"""Invocation-owned terminal commits retain bodies and require fresh legal checks."""

import copy
import json
from typing import Any, cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import _outcome_metadata_properties
from onyx.asv3.retained_answer import bind_retained_answer, resolve_retained_answer
from onyx.asv3.terminal_fact_references import (
    bind_terminal_fact_references,
    normalize_terminal_fact_references,
    terminal_fact_catalogue,
)
from tests.unit.onyx.asv3.test_model_adapter import (
    argument_patch_response,
    tool_response,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, payloads
from tests.unit.onyx.asv3.test_retained_answer import REQUEST, context, decision, tools
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    declaration,
    outer,
    serial_run,
    session,
)
from tests.unit.onyx.asv3.test_shared_originals import original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize(
    "name,canonical",
    [
        ("submit_retained_answer", "submit_answer"),
        ("submit_retained_partial_answer", "submit_partial_answer"),
    ],
)
def test_owned_commit_has_no_body_identifier_or_basis_on_wire(
    hosted: bool, name: str, canonical: str
) -> None:
    ctx = context(hosted=hosted)
    draft = "  First material rule [1].\n\nSecond condition [2].  "
    supplied = tools()
    original_tools = copy.deepcopy(supplied)
    bound, reference = bind_retained_answer(supplied, ctx, draft, request=REQUEST)
    assert supplied == original_tools and reference
    definition = next(
        cast(dict[str, JsonValue], row["function"])
        for row in bound
        if cast(dict[str, JsonValue], row["function"])["name"] == name
    )
    schema = cast(dict[str, JsonValue], definition["parameters"])
    assert not {"answer", "retained_answer_id", "basis"} & set(
        cast(dict[str, JsonValue], schema["properties"])
    )
    jsonschema.validate({}, schema)
    emitted = decision({}, name=name)
    wire = (
        emitted.assistant_message.model_dump_json() if emitted.assistant_message else ""
    )
    resolved = resolve_retained_answer(emitted, ctx, draft, request=REQUEST)
    assert resolved.calls[0].name == canonical
    assert resolved.calls[0].arguments == {"answer": draft, "basis": "originals"}
    assert resolved.assistant_message is emitted.assistant_message
    assert (
        resolved.assistant_message
        and resolved.assistant_message.model_dump_json() == wire
    )
    units = cast(list[dict[str, JsonValue]], reference["units"])
    edited = decision(
        {
            "retained_answer_edits": [
                {
                    "unit_id": units[1]["unit_id"],
                    "replacement": "Corrected condition [2].",
                }
            ]
        },
        name=name,
    )
    result = resolve_retained_answer(edited, ctx, draft, request=REQUEST)
    assert (
        result.calls[0].arguments["answer"]
        == "  First material rule [1].\n\nCorrected condition [2].  "
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"answer": "foreign"},
        {"retained_answer_id": "foreign"},
        {"basis": "conversation"},
        {"retained_answer_edits": []},
        {"retained_answer_edits": [{"unit_id": "foreign", "replacement": "New"}]},
    ],
)
def test_owned_commit_rejects_explicit_or_foreign_body_and_invalid_edits(
    invalid: dict[str, JsonValue],
) -> None:
    rejected = resolve_retained_answer(
        decision(invalid, name="submit_retained_answer"),
        context(),
        "Owned [1].",
        request=REQUEST,
    )
    assert rejected.calls[0].argument_error


def test_no_draft_and_plain_profiles_do_not_expose_owned_commit() -> None:
    ctx = context()
    bound, descriptor = bind_retained_answer(tools(), ctx, None, request=REQUEST)
    assert bound == tools() and descriptor is None
    assert (
        resolve_retained_answer(
            decision({}, name="submit_retained_answer"), ctx, None, request=REQUEST
        )
        .calls[0]
        .argument_error
    )
    ctx.services["experimental_parallel"] = False
    assert bind_retained_answer(tools(), ctx, "Owned [1].", request=REQUEST) == (
        tools(),
        None,
    )


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("patch", [False, True])
def test_actual_hosted_commit_needs_current_invocation_originals(
    monkeypatch: pytest.MonkeyPatch, missing: bool, patch: bool
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.selected.reset_mock()
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    own = original("own", "The genuine operative condition.")
    ledger.add(
        [
            own,
            original(
                "other",
                "An unseen condition.",
                source="other",
                headings=["Other instrument"],
            ),
        ],
        child.context,
    )
    child.harness.evidence_working_set.remember(1, 0, len(own.text))
    body = f"The condition applies [{2 if missing else 1}]."
    wire = json.dumps({"_language": "tr", "_outcomes": [declaration()]})

    count = 0
    semantic_call = ""

    def invoke(**kwargs: Any) -> Any:
        nonlocal count, semantic_call
        count += 1
        if count == 2:
            assert patch and actual_originals(kwargs["prompt"]) == []
            semantic_call = child.model.last_call_id or ""
            return argument_patch_response(wire)
        assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {1}
        if not patch:
            supplied = payloads(kwargs["prompt"])[-1]["decisive_fact_catalogue"]
            metadata = json.loads(wire)
            metadata["_outcomes"][0]["decisive_facts"] = None
            metadata["_outcomes"][0]["decisive_fact_refs"] = [supplied[0]["fact_id"]]
            metadata["_coverage"] = None
            metadata["retained_answer_edits"] = None
            emitted_wire = json.dumps(metadata)
        else:
            emitted_wire = json.dumps({**json.loads(wire), "unknown": True})
        names = [tool["function"]["name"] for tool in kwargs["tools"]]
        assert "submit_retained_answer" in names
        return tool_response(emitted_wire, "submit_retained_answer")

    run.selected.invoke.side_effect = invoke
    view = child.harness.view()
    view.draft_to_repair = body
    view.publication_gap = {"repair": True}
    result = child.model.decide(view)
    assert run.selected.invoke.call_count == (2 if patch else 1)
    if patch:
        assert (
            child.model.last_call_id
            == child.context.services["last_model_call_id"]
            == semantic_call
        )
        assert ledger.completely_delivered(semantic_call) == {1}
    assert (
        result.calls[0].name == "submit_answer"
        and result.calls[0].argument_error is None
    )
    assert result.calls[0].arguments["answer"] == body
    assert result.assistant_message and result.assistant_message.tool_calls
    assert (
        result.assistant_message.tool_calls[0].function.name == "submit_retained_answer"
    )
    if patch:
        assert json.loads(
            result.assistant_message.tool_calls[0].function.arguments
        ) == json.loads(wire)
    else:
        raw_arguments = json.loads(
            result.assistant_message.tool_calls[0].function.arguments
        )
        assert raw_arguments["_coverage"] is None
        assert raw_arguments["_outcomes"][0]["decisive_facts"] is None
        assert "decisive_fact_refs" in raw_arguments["_outcomes"][0]
        normalized_outcome = cast(
            list[dict[str, JsonValue]], result.calls[0].arguments["_outcomes"]
        )[0]
        assert "decisive_fact_refs" not in normalized_outcome
        assert normalized_outcome["decisive_facts"] == [
            terminal_fact_catalogue(child.context)[0]["text"]
        ]
    child._on_decision(result)
    receipt = child.harness._dispatch(result.calls)[0]
    if missing:
        assert receipt.outcome.status == OutcomeStatus.PARTIAL
        assert receipt.outcome.data["undelivered_citations"] == [2]
    else:
        assert receipt.outcome.status == OutcomeStatus.FOUND
        assert child.context.services["submitted_answer"] == body


def fact_tools() -> list[dict[str, JsonValue]]:
    result = tools()
    for row in result:
        function = cast(dict[str, JsonValue], row["function"])
        schema = cast(dict[str, JsonValue], function["parameters"])
        cast(dict[str, JsonValue], schema["properties"]).update(
            _outcome_metadata_properties()
        )
    return result


def test_owned_fact_refs_decode_literal_context_without_changing_body_or_ids() -> None:
    ctx = context()
    raw = (
        "The first import occurred four months ago.\nNo initial reduction was allowed."
    )
    outcomes = OutcomeMap(["Explain"], ctx, factual_context=raw)
    ctx.services["outcome_map"] = outcomes
    bound, catalogue = bind_terminal_fact_references(fact_tools(), ctx)
    assert catalogue and all(cast(str, row["text"]) in raw for row in catalogue)
    args: dict[str, JsonValue] = {
        "answer": "Exact body [1].",
        "basis": "originals",
        "_outcomes": [
            {
                "outcome_id": "owned",
                "question_ids": ["q0"],
                "detail": "Assess",
                "decisive_fact_refs": [catalogue[0]["fact_id"]],
            }
        ],
    }
    original_args = copy.deepcopy(args)
    normalized = normalize_terminal_fact_references("submit_answer", args, ctx)
    assert args == original_args and normalized["answer"] == args["answer"]
    row = cast(list[dict[str, JsonValue]], normalized["_outcomes"])[0]
    assert row["decisive_facts"] == [catalogue[0]["text"]]
    assert row["outcome_id"] == "owned" and "decisive_fact_refs" not in row
    outcomes.update(OutcomeUpdate.model_validate({"outcomes": [row]}), EvidenceLedger())
    schema = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], bound[0]["function"])["parameters"],
    )
    jsonschema.validate(normalized, schema)
    for bad in (
        "foreign",
        terminal_fact_catalogue(context())[0]["fact_id"]
        if terminal_fact_catalogue(context())
        else "missing",
    ):
        invalid = copy.deepcopy(args)
        cast(list[dict[str, JsonValue]], invalid["_outcomes"])[0][
            "decisive_fact_refs"
        ] = [bad]
        assert (
            normalize_terminal_fact_references("submit_answer", invalid, ctx) is invalid
        )
    conflict = copy.deepcopy(args)
    cast(list[dict[str, JsonValue]], conflict["_outcomes"])[0]["decisive_facts"] = [
        "paraphrase"
    ]
    assert (
        normalize_terminal_fact_references("submit_answer", conflict, ctx) is conflict
    )
    ctx.services["experimental_parallel"] = False
    assert normalize_terminal_fact_references("submit_answer", args, ctx) is args
