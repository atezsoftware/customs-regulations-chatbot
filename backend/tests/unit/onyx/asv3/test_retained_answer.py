"""Retained drafts reduce wire text while every publication check remains fresh."""

import copy
import json
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.models import CapabilityCall, Decision, ResearchTurn, RunContext
from onyx.asv3.retained_answer import (
    bind_retained_answer,
    project_failed_terminal_turns,
    resolve_retained_answer,
)
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage

DRAFT = "## Operative conclusion\n\n" + (
    "Every condition and exception remains literal [1].\n" * 210
)
REQUEST = "Explain this exact scenario and its alternative."


def context(*, hosted: bool = False) -> RunContext:
    return RunContext(
        run_id="owned-run",
        scope={"tenant": "A", "document_sets": [15]},
        services={
            "research_profile": "experimental",
            "experimental_parallel": not hosted,
            "task_id": "owned-session",
            "lean_native_mode": True,
            "serial_session_diagnostics": hosted,
        },
    )


def tools() -> list[dict[str, JsonValue]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Publish only after its normal checks.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {"type": "string", "enum": ["originals"]},
                        "_related_source_reviews": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["answer", "basis"],
                },
            },
        }
        for name in ("submit_answer", "submit_partial_answer")
    ]


def reference(ctx: RunContext, draft: str = DRAFT, request: str = REQUEST) -> str:
    _, binding = bind_retained_answer(tools(), ctx, draft, request=request)
    assert binding is not None
    result = binding["retained_answer_id"]
    assert isinstance(result, str)
    return result


def decision(
    arguments: dict[str, JsonValue], *, name: str = "submit_answer"
) -> Decision:
    return Decision(
        calls=[
            CapabilityCall(
                name=name, call_id="actual-provider-call", arguments=arguments
            )
        ],
        assistant_message=AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="actual-provider-call",
                    function=FunctionCall(name=name, arguments=json.dumps(arguments)),
                )
            ]
        ),
    )


@pytest.mark.parametrize("hosted", [False, True])
def test_schema_copy_and_reference_expand_only_host_call_with_exact_body(
    hosted: bool,
) -> None:
    ctx, definitions = context(hosted=hosted), tools()
    original = copy.deepcopy(definitions)
    bound, descriptor = bind_retained_answer(definitions, ctx, DRAFT, request=REQUEST)
    assert definitions == original and bound != original and descriptor is not None
    units = descriptor["units"]
    assert isinstance(units, list)
    assert (
        "".join(str(unit["text"]) for unit in units if isinstance(unit, dict)) == DRAFT
    )
    assert DRAFT not in json.dumps(
        {key: value for key, value in descriptor.items() if key != "units"}
    )
    parameters = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], bound[0]["function"])["parameters"],
    )
    arguments: dict[str, JsonValue] = {
        "retained_answer_id": descriptor["retained_answer_id"],
        "basis": "originals",
        "_related_source_reviews": [{"lead_id": "literal-kept", "status": "examined"}],
    }
    jsonschema.validate(arguments, parameters)
    assert parameters["required"] == ["basis"]
    emitted = decision(arguments)
    raw_native = (
        emitted.assistant_message.model_dump_json() if emitted.assistant_message else ""
    )
    resolved = resolve_retained_answer(emitted, ctx, DRAFT, request=REQUEST)
    assert resolved.calls[0].arguments == {
        "answer": DRAFT,
        "basis": "originals",
        "_related_source_reviews": arguments["_related_source_reviews"],
    }
    assert "answer" not in emitted.calls[0].arguments
    assert resolved.assistant_message is emitted.assistant_message
    assert resolved.assistant_message is not None
    assert resolved.assistant_message.model_dump_json() == raw_native
    assert len(raw_native) < len(DRAFT) // 5


@pytest.mark.parametrize("change", ["run", "owner", "scope", "request", "body"])
def test_reference_is_bound_to_exact_owner_run_scope_request_and_body(
    change: str,
) -> None:
    ctx = context()
    emitted = decision({"retained_answer_id": reference(ctx), "basis": "originals"})
    draft, request = DRAFT, REQUEST
    if change == "run":
        ctx.run_id = "another-run"
    elif change == "owner":
        ctx.services["task_id"] = "another-owner"
    elif change == "scope":
        ctx.scope["tenant"] = "another-tenant"
    elif change == "request":
        request += "New decisive fact."
    else:
        draft += "Changed punctuation."
    rejected = resolve_retained_answer(emitted, ctx, draft, request=request)
    assert (
        rejected.calls[0].argument_error and "answer" not in rejected.calls[0].arguments
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"retained_answer_id": "foreign"},
        {"retained_answer_id": None},
        {"retained_answer_id": 1},
        {"answer": DRAFT, "retained_answer_id": "foreign"},
    ],
)
def test_invalid_or_mixed_body_representations_are_not_dispatched(
    arguments: dict[str, JsonValue],
) -> None:
    ctx = context()
    rejected = resolve_retained_answer(decision(arguments), ctx, DRAFT, request=REQUEST)
    assert rejected.calls[0].argument_error is not None


def test_a_reference_without_current_draft_and_new_body_requests_are_distinct() -> None:
    ctx = context()
    emitted = decision({"retained_answer_id": reference(ctx), "basis": "originals"})
    assert (
        resolve_retained_answer(emitted, ctx, None, request=REQUEST)
        .calls[0]
        .argument_error
    )
    changed = decision(
        {"answer": "A corrected source-bound result [1].", "basis": "originals"}
    )
    assert (
        resolve_retained_answer(changed, ctx, DRAFT, request=REQUEST).calls
        == changed.calls
    )


def test_resolved_body_still_needs_its_actual_current_model_delivery() -> None:
    from onyx.asv3.authority_requirements import AuthorityRequirements
    from onyx.asv3.legal_source_reviews import LegalSourceReviews
    from onyx.asv3.serial_experimental_session import _publication_gap
    from tests.unit.onyx.asv3.test_shared_originals import (
        full_record,
        original,
        recorded,
    )

    ctx = context(hosted=True)
    ledger = recorded([original("operative", "The actual operative requirement.")])
    ledger.record_delivery(
        "older-model-call", "asv3_coordinator", [full_record(ledger, 1)]
    )
    before = ledger.export()
    body = "The operative requirement applies [1]."
    emitted = decision(
        {"retained_answer_id": reference(ctx, body), "basis": "originals"}
    )
    resolved = resolve_retained_answer(emitted, ctx, body, request=REQUEST)
    assert resolved.calls[0].arguments["answer"] == body and ledger.export() == before
    gap = _publication_gap(
        body,
        "current-model-call",
        ctx,
        ledger,
        AuthorityRequirements(ctx, REQUEST),
        LegalSourceReviews(ctx, REQUEST),
        requires_sources=True,
    )
    assert gap is not None and gap.data["undelivered_citations"] == [1]
    assert ledger.completely_delivered("current-model-call") == set()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_exact_unit_partition_and_one_edit_keep_all_other_bytes(newline: str) -> None:
    ctx = context(hosted=True)
    body = (
        f"  ## Başlık{newline}{newline}"
        f"Aynı olgu [1].{newline}Devam satırı.{newline} \t{newline}"
        f"Aynı olgu [1].{newline}{newline}\tSonuç 東京 [2].  {newline}"
    )
    bound, descriptor = bind_retained_answer(tools(), ctx, body, request=REQUEST)
    assert descriptor is not None
    units = cast(list[dict[str, JsonValue]], descriptor["units"])
    assert len(units) == 4
    assert "".join(str(unit["text"]) for unit in units) == body
    assert units[0]["start_char"] == 0 and units[-1]["end_char"] == len(body)
    assert all(
        left["end_char"] == right["start_char"] for left, right in zip(units, units[1:])
    )
    assert len({unit["unit_id"] for unit in units}) == len(units)
    arguments: dict[str, JsonValue] = {
        "retained_answer_id": descriptor["retained_answer_id"],
        "retained_answer_edits": [
            {
                "unit_id": units[2]["unit_id"],
                "replacement": "  Düzeltilen sonuç [1]. \n",
            }
        ],
        "basis": "originals",
    }
    parameters = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], bound[0]["function"])["parameters"],
    )
    jsonschema.validate(arguments, parameters)
    emitted = decision(arguments)
    assert emitted.assistant_message is not None
    wire = emitted.assistant_message.model_dump_json()
    resolved = resolve_retained_answer(emitted, ctx, body, request=REQUEST)
    start, end = units[2]["start_char"], units[2]["end_char"]
    assert isinstance(start, int) and isinstance(end, int)
    assert (
        resolved.calls[0].arguments["answer"]
        == body[:start] + f"Düzeltilen sonuç [1].{newline}{newline}" + body[end:]
    )
    assert resolved.calls[0].argument_error is None
    assert "retained_answer_edits" not in resolved.calls[0].arguments
    assert resolved.assistant_message is emitted.assistant_message
    assert resolved.assistant_message.model_dump_json() == wire
    assert "answer" not in emitted.calls[0].arguments


def test_two_simultaneous_edits_preserve_separator_and_position_identity() -> None:
    ctx, body = context(), "Same paragraph [1].\n\nSame paragraph [1].\n\nLast [2].\n"
    _bound, descriptor = bind_retained_answer(tools(), ctx, body, request=REQUEST)
    assert descriptor is not None
    units = cast(list[dict[str, JsonValue]], descriptor["units"])
    assert units[0]["unit_id"] != units[1]["unit_id"]
    resolved = resolve_retained_answer(
        decision(
            {
                "retained_answer_id": descriptor["retained_answer_id"],
                "retained_answer_edits": [
                    {"unit_id": units[2]["unit_id"], "replacement": "New last [2]."},
                    {
                        "unit_id": units[0]["unit_id"],
                        "replacement": "First changed [1].",
                    },
                ],
                "basis": "originals",
            }
        ),
        ctx,
        body,
        request=REQUEST,
    )
    assert (
        resolved.calls[0].arguments["answer"]
        == "First changed [1].\n\nSame paragraph [1].\n\nNew last [2].\n"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "foreign",
        "duplicate",
        "empty",
        "blank",
        "extra",
        "missing_ref",
        "mixed",
        "stale_unit",
        "not_list",
    ],
)
def test_malformed_or_unowned_unit_edits_cannot_dispatch(defect: str) -> None:
    ctx, body = context(), "First [1].\n\nSecond [2].\n"
    _bound, descriptor = bind_retained_answer(tools(), ctx, body, request=REQUEST)
    assert descriptor is not None
    units = cast(list[dict[str, JsonValue]], descriptor["units"])
    edit: dict[str, JsonValue] = {
        "unit_id": units[0]["unit_id"],
        "replacement": "Fixed [1].",
    }
    args: dict[str, JsonValue] = {
        "retained_answer_id": descriptor["retained_answer_id"],
        "retained_answer_edits": [edit],
        "basis": "originals",
    }
    if defect == "foreign":
        edit["unit_id"] = "retained_unit_foreign"
    elif defect == "duplicate":
        args["retained_answer_edits"] = [edit, dict(edit)]
    elif defect == "empty":
        args["retained_answer_edits"] = []
    elif defect == "blank":
        edit["replacement"] = "  \n\t"
    elif defect == "extra":
        edit["start_char"] = 0
    elif defect == "missing_ref":
        args.pop("retained_answer_id")
    elif defect == "mixed":
        args["answer"] = body
    elif defect == "not_list":
        args["retained_answer_edits"] = "JSON string"
    else:
        body += "Changed source detail."
        args["retained_answer_id"] = reference(ctx, body)
    rejected = resolve_retained_answer(decision(args), ctx, body, request=REQUEST)
    assert rejected.calls[0].argument_error is not None


def test_an_edited_answer_does_not_supply_its_new_citations_or_approval() -> None:
    from onyx.asv3.authority_requirements import AuthorityRequirements
    from onyx.asv3.legal_source_reviews import LegalSourceReviews
    from onyx.asv3.serial_experimental_session import _publication_gap
    from tests.unit.onyx.asv3.test_shared_originals import (
        full_record,
        original,
        recorded,
    )

    ctx, body = context(hosted=True), "First supported [1].\n\nSecond supported [1]."
    ledger = recorded(
        [
            original("first", "An original rule."),
            original("second", "A different original."),
        ]
    )
    ledger.record_delivery("current", "asv3_coordinator", [full_record(ledger, 1)])
    _bound, descriptor = bind_retained_answer(tools(), ctx, body, request=REQUEST)
    assert descriptor is not None
    units = cast(list[dict[str, JsonValue]], descriptor["units"])
    resolved = resolve_retained_answer(
        decision(
            {
                "retained_answer_id": descriptor["retained_answer_id"],
                "retained_answer_edits": [
                    {
                        "unit_id": units[1]["unit_id"],
                        "replacement": "A changed legal result [2].",
                    }
                ],
                "basis": "originals",
            }
        ),
        ctx,
        body,
        request=REQUEST,
    )
    answer = resolved.calls[0].arguments["answer"]
    assert isinstance(answer, str)
    gap = _publication_gap(
        answer,
        "current",
        ctx,
        ledger,
        AuthorityRequirements(ctx, REQUEST),
        LegalSourceReviews(ctx, REQUEST),
        requires_sources=True,
    )
    assert gap is not None and gap.data["undelivered_citations"] == [2]


@pytest.mark.parametrize(
    "services,depth",
    [
        ({}, 0),
        ({"research_profile": "experimental", "experimental_parallel": False}, 0),
        ({"research_profile": "experimental", "experimental_parallel": "true"}, 0),
        (
            {
                "research_profile": "experimental",
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
            },
            0,
        ),
        (
            {
                "research_profile": "experimental",
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
                "task_id": "owned",
            },
            1,
        ),
    ],
)
def test_ordinary_profiles_and_untrusted_flags_remain_byte_equal(
    services: dict[str, JsonValue], depth: int
) -> None:
    ctx = RunContext(services=dict(services), depth=depth)
    definitions = tools()
    bound, descriptor = bind_retained_answer(definitions, ctx, DRAFT, request=REQUEST)
    assert json.dumps(bound) == json.dumps(definitions) and descriptor is None
    emitted = decision({"answer": DRAFT, "basis": "originals"})
    assert resolve_retained_answer(emitted, ctx, DRAFT, request=REQUEST) is emitted


def turn(
    status: str = "invalid", *, data: dict[str, JsonValue] | None = None
) -> ResearchTurn:
    payload: dict[str, JsonValue] = {
        "outcome": {
            "status": status,
            "summary": "Publication assessment",
            "data": {"invalid_related_source_review": {"lead_id": "owned"}}
            if data is None
            else data,
            "evidence": [],
            "original_reads": [],
            "artifacts": [],
        },
        "evidence_ids": [],
        "original_evidence": [],
    }
    native = decision({"answer": DRAFT, "basis": "originals"}).assistant_message
    assert native is not None
    return ResearchTurn(
        assistant=native,
        results=[
            ToolMessage(
                tool_call_id="actual-provider-call",
                content=json.dumps(payload),
            )
        ],
    )


def test_failed_terminal_projection_drops_only_model_view_without_mutating_journal() -> (
    None
):
    ctx = context()
    rejected = [turn(), turn("partial"), turn("denied")]
    accepted = turn("found")
    journal = [*rejected, accepted]
    before = [item.model_dump_json() for item in journal]
    projected = project_failed_terminal_turns(
        journal, ctx, DRAFT, {"missing": "exact review"}
    )
    assert projected == [accepted]
    assert [item.model_dump_json() for item in journal] == before
    assert (
        len(json.dumps([item.model_dump() for item in projected]))
        < len(json.dumps([item.model_dump() for item in journal])) // 3
    )


def test_tuned_known_duplicate_rejection_is_obsolete_but_unknown_empty_result_is_retained() -> (
    None
):
    ctx = context()
    ctx.services.update(asv3_workflow_variant="asv3_tuned", research_profile="normal")
    item = turn(data={})
    payload = json.loads(item.results[0].content)
    payload["outcome"]["summary"] = (
        "Repeated failed call: change the arguments or method"
    )
    item.results[0].content = json.dumps(payload)
    before = item.model_dump_json()
    gap: dict[str, JsonValue] = {"missing": "Actual current gap"}
    assert project_failed_terminal_turns([item], ctx, DRAFT, gap) == []
    assert item.model_dump_json() == before
    assert project_failed_terminal_turns([item], ctx, DRAFT, None) == [item]
    ctx.services["asv3_workflow_variant"] = "standard"
    assert project_failed_terminal_turns([item], ctx, DRAFT, gap) == [item]
    ctx.services["asv3_workflow_variant"] = "asv3_tuned"
    payload["outcome"]["summary"] = "Unknown empty assessment"
    item.results[0].content = json.dumps(payload)
    assert project_failed_terminal_turns([item], ctx, DRAFT, gap) == [item]


@pytest.mark.parametrize(
    "keep",
    [
        "accepted_partial",
        "original_record",
        "read_range",
        "artifact",
        "evidence_id",
        "mixed_source_action",
        "pending_pair",
        "malformed_result",
        "substantive_prose",
        "accepted_receipt",
        "nonterminal",
        "empty_gap",
        "empty_draft",
        "ordinary_profile",
    ],
)
def test_projection_preserves_accepted_source_bearing_or_ambiguous_turns(
    keep: str,
) -> None:
    ctx, item, draft = context(), turn(), DRAFT
    gap: dict[str, JsonValue] = {"missing": "exact review"}
    assert item.assistant.tool_calls is not None
    payload = json.loads(item.results[0].content)
    if keep == "accepted_partial":
        payload["outcome"].update(status="partial", data={})
    elif keep == "original_record":
        payload["original_evidence"] = [{"citation": 1, "text": "literal"}]
    elif keep == "read_range":
        payload["outcome"]["original_reads"] = [
            {"citation": 1, "start_char": 0, "end_char": 9}
        ]
    elif keep == "artifact":
        payload["outcome"]["artifacts"] = [{"artifact_id": "retained"}]
    elif keep == "evidence_id":
        payload["evidence_ids"] = [1]
    elif keep == "accepted_receipt":
        payload["outcome"]["data"]["parallel_answer_receipt"] = "accepted"
    elif keep == "mixed_source_action":
        item.assistant.tool_calls.append(
            ToolCall(
                id="read", function=FunctionCall(name="read_provision", arguments="{}")
            )
        )
        item.results.append(ToolMessage(tool_call_id="read", content="{}"))
    elif keep == "pending_pair":
        item.results.clear()
    elif keep == "substantive_prose":
        item.assistant.content = "An additional condition must be preserved [1]."
    elif keep == "nonterminal":
        item.assistant.tool_calls[0].function.name = "read_provision"
    elif keep == "empty_gap":
        gap = {}
    elif keep == "empty_draft":
        draft = ""
    elif keep == "ordinary_profile":
        ctx.services["experimental_parallel"] = False
    if keep == "malformed_result":
        item.results[0].content = "Incomplete JSON"
    elif item.results:
        item.results[0].content = json.dumps(payload)
    assert project_failed_terminal_turns([item], ctx, draft, gap) == [item]
