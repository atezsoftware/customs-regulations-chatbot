import json
from typing import Any

import pytest

from onyx.asv3.llm_adapter import ResearchModel, strict_json_decoder
from onyx.asv3.models import HarnessView, OutcomeStatus, RunContext, ToolSpec
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.llm.models import UserMessage
from onyx.llm.tool_argument_stream import (
    TOOL_ARGUMENT_STREAM_ABORT,
    TOOL_ARGUMENT_WHITESPACE_RUN,
    complete_object_prefix_fields,
)
from tests.unit.onyx.asv3.test_model_adapter import (
    argument_patch_response,
    original_state,
    scripted_model,
    tool_response,
)


@pytest.mark.parametrize("changed_field", [False, True])
def test_aborted_argument_repair_retains_completed_calls_and_original_task(
    changed_field: bool,
) -> None:
    context, ledger, tools = original_state()
    task = "Read original transport liability only; retain version uncertainty."
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "spawn_researcher",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                    "additionalProperties": False,
                },
            },
        }
    )
    initial = tool_response('{"citation":1}')
    interrupted = tool_response(
        json.dumps({"task": task})[:-1] + "," + "\t" * TOOL_ARGUMENT_WHITESPACE_RUN,
        "spawn_researcher",
    )
    assert interrupted.choice.message.tool_calls and initial.choice.message.tool_calls
    interrupted.choice.message.tool_calls[0].id = "call-2"
    initial.choice.message.tool_calls += interrupted.choice.message.tool_calls
    initial.choice.finish_reason = TOOL_ARGUMENT_STREAM_ABORT
    llm = scripted_model()
    llm.invoke.side_effect = [
        initial,
        argument_patch_response(
            json.dumps(
                {"task": "Change the research scope" if changed_field else task}
            ),
            "call-2",
        ),
    ]
    model = ResearchModel(llm, context, lean_native_mode=True)
    before = ledger.export()
    current = HarnessView(
        request="Investigate original transport provisions",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools,
        original_evidence=[
            {"citation": 1, "text": ledger.get(1).text, "source_id": "law"}
        ],
    )
    result = model._invoke_decision(
        current,
        [
            UserMessage(
                content=json.dumps({"original_evidence": current.original_evidence})
            )
        ],
        tools,
        6000,
    )
    assert llm.invoke.call_count == 2
    assert [call.call_id for call in result.calls] == ["call-1", "call-2"]
    assert result.calls[0].arguments == {"citation": 1}
    assert result.calls[1].arguments == {"task": task}
    assert bool(result.calls[1].argument_error) is changed_field
    assert result.assistant_message and result.assistant_message.tool_calls
    assert result.assistant_message.tool_calls[0].function.arguments == '{"citation":1}'
    correction = llm.invoke.call_args_list[1].kwargs["prompt"][1].content
    payload = json.loads(
        correction if isinstance(correction, str) else correction[0].text
    )
    assert [action["call_id"] for action in payload["invalid_actions"]] == ["call-2"]
    after = ledger.export()
    assert after["records"] == before["records"]
    assert len(after["deliveries"]) == 1
    delivery = after["deliveries"][0]
    assert delivery["flow"] == "asv3_coordinator"
    assert ledger.completely_delivered(delivery["call_id"]) == {1}
    assert delivery["call_id"] != model.last_call_id
    if changed_field:
        registry = CapabilityRegistry()
        for spec in build_core_specs(registry, ledger, lambda: {}):
            registry.register(spec)
        registry.register(
            ToolSpec(
                name="spawn_researcher",
                description="Delegate original source research",
                parameters=tools[-1]["function"]["parameters"],
                handler=lambda _call, _context: pytest.fail("Invalid call executed"),
            )
        )
        # The invalid partial field projection never grants execution permission.
        outcome = registry.dispatch(result.calls[1], RunContext())
        assert outcome.status == OutcomeStatus.INVALID
        assert "argument_error" in outcome.data


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            '{"task":"complete","nested":{"a":1},"pending":',
            {"task": "complete", "nested": {"a": 1}},
        ),
        ('{"task":"complete","task":"duplicate",', {}),
        ('{"task":"complete","value":NaN,', {"task": "complete"}),
        ('{"task":"complete","value":1invalid,', {"task": "complete"}),
        ('{"task":"unfinished', {}),
    ],
)
def test_only_strict_complete_prefix_fields_are_repair_inputs(
    arguments: str, expected: dict[str, Any]
) -> None:
    assert (
        complete_object_prefix_fields(arguments, decoder=strict_json_decoder())
        == expected
    )
