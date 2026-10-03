"""Recover argument errors without dropping parallel actions or changing their intent."""

import json

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    HarnessView,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_model_adapter import (
    argument_patch_response,
    original_state,
    scripted_model,
    text_response,
    tool_response,
)


def model_view(tools: list[dict[str, JsonValue]]) -> HarnessView:
    return HarnessView(
        request="Find the independent information needs without changing the scenario.",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools,
    )


@pytest.mark.parametrize("envelope", ["json_tool_call", "ToolArgumentPatch"])
def test_structured_patch_uses_host_owned_native_call_identity(envelope: str) -> None:
    context, _ledger, tools = original_state()
    llm = scripted_model()
    patch = argument_patch_response('{"citation":1}')
    assert patch.choice.message.content
    native_patch = tool_response(patch.choice.message.content, envelope)
    assert native_patch.choice.message.tool_calls
    native_patch.choice.message.tool_calls[0].id = "new-provider-envelope-id"
    llm.invoke.side_effect = [tool_response('{"citation":"1"}'), native_patch]
    decision = ResearchModel(llm, context).decide(model_view(tools))
    assert decision.calls[0].call_id == "call-1"
    assert decision.calls[0].arguments == {"citation": 1}
    assert decision.calls[0].argument_error is None
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert decision.assistant_message.tool_calls[0].id == "call-1"
    assert context.budget.snapshot()["decisions"] == 1
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("ids", [[], ["unknown"], ["broken", "broken"], ["call-1"]])
def test_missing_or_changed_patch_ids_leave_valid_parallel_action_intact(
    ids: list[str],
) -> None:
    context, _ledger, tools = original_state()
    first = tool_response('{"citation":1}')
    broken = tool_response('{"citation":"1"}')
    assert first.choice.message.tool_calls and broken.choice.message.tool_calls
    broken.choice.message.tool_calls[0].id = "broken"
    first.choice.message.tool_calls += broken.choice.message.tool_calls
    patch = text_response(
        {
            "entries": [
                {
                    "call_id": call_id,
                    "arguments_json": '{"citation":2}',
                    "explanation": "Change one invalid value.",
                }
                for call_id in ids
            ]
        }
    )
    llm = scripted_model()
    llm.invoke.side_effect = [first, patch]
    decision = ResearchModel(llm, context).decide(model_view(tools))
    assert [call.call_id for call in decision.calls] == ["call-1", "broken"]
    assert decision.calls[0].arguments == {"citation": 1}
    assert decision.calls[0].argument_error is None
    assert (
        decision.calls[1].argument_error
        and "patch rejected" in decision.calls[1].argument_error
    )
    assert decision.calls[1].arguments == {"citation": "1"}
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert (
        decision.assistant_message.tool_calls[1].function.arguments
        == '{"citation":"1"}'
    )
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("change_query", [False, True])
def test_enum_repair_receives_full_arguments_and_preserves_valid_query(
    change_query: bool,
) -> None:
    query = "Distinct scenario detail. " * 200
    original = {"query": query, "mode": "invalid-method"}
    corrected = {"query": "different" if change_query else query, "mode": "keyword"}
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "query_corpus",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {"type": "string", "enum": ["keyword", "semantic"]},
                    },
                    "required": ["query", "mode"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response(json.dumps(original), "query_corpus"),
        argument_patch_response(json.dumps(corrected)),
    ]
    decision = ResearchModel(llm, RunContext()).decide(model_view(tools))
    assert decision.calls[0].arguments["query"] == query
    assert bool(decision.calls[0].argument_error) is change_query
    payload = json.loads(llm.invoke.call_args_list[1].kwargs["prompt"][1].content)
    assert json.loads(payload["invalid_actions"][0]["arguments_json"]) == original
    assert payload["request"] == model_view(tools).request
    assert llm.invoke.call_args_list[1].kwargs["tools"] is None


def test_invalid_patch_does_not_abort_harness_or_execute_bad_enum() -> None:
    executed: list[str] = []

    def action(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        mode = arguments["mode"]
        assert isinstance(mode, str)
        executed.append(mode)
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original information read"
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_test_source",
                description="Read a source with a declared mode",
                parameters={
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["keyword"]}},
                    "required": ["mode"],
                },
                handler=action,
            )
        ]
    )
    first = tool_response('{"mode":"keyword"}', "read_test_source")
    broken = tool_response('{"mode":"invalid"}', "read_test_source")
    assert first.choice.message.tool_calls and broken.choice.message.tool_calls
    broken.choice.message.tool_calls[0].id = "broken"
    first.choice.message.tool_calls += broken.choice.message.tool_calls
    llm = scripted_model()
    llm.invoke.side_effect = [
        first,
        text_response({"entries": []}),
        text_response({"finished": True}),
    ]
    context = RunContext()
    result = Harness(
        request="Read the independent sources",
        context=context,
        registry=registry,
        decide=ResearchModel(llm, context).decide,
    ).run()
    assert result.status == OutcomeStatus.FOUND
    assert executed == ["keyword"]
    assert [receipt.outcome.status for receipt in result.receipts] == [
        OutcomeStatus.FOUND,
        OutcomeStatus.INVALID,
    ]
    assert context.budget.snapshot()["tools"] == 1
    assert llm.invoke.call_count == 3


def test_valid_actions_add_no_argument_repair_call() -> None:
    context, _ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.return_value = tool_response('{"citation":1}')
    assert (
        ResearchModel(llm, context).decide(model_view(tools)).calls[0].argument_error
        is None
    )
    assert llm.invoke.call_count == 1
    assert context.budget.snapshot()["decisions"] == 0


def test_one_unrepairable_entry_keeps_an_independent_valid_patch() -> None:
    context, _ledger, tools = original_state()
    first = tool_response('{"citation":"1"}')
    second = tool_response('{"citation":"2"}')
    assert first.choice.message.tool_calls and second.choice.message.tool_calls
    second.choice.message.tool_calls[0].id = "second"
    first.choice.message.tool_calls += second.choice.message.tool_calls
    patch = text_response(
        {
            "entries": [
                {
                    "call_id": "call-1",
                    "arguments_json": '{"citation":1}',
                    "explanation": "Integer value",
                },
                {
                    "call_id": "second",
                    "arguments_json": "{",
                    "explanation": "Incomplete JSON",
                },
            ]
        }
    )
    llm = scripted_model()
    llm.invoke.side_effect = [first, patch]
    decision = ResearchModel(llm, context).decide(model_view(tools))
    assert decision.calls[0].arguments == {"citation": 1}
    assert decision.calls[0].argument_error is None
    assert decision.calls[1].arguments == {"citation": "2"}
    assert decision.calls[1].argument_error is not None
