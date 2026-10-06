"""Terminal argument repair retains a valid legal draft without regenerating it."""

import copy
import json

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_model_adapter import (
    argument_patch_response,
    scripted_model,
    tool_response,
)

ANSWER = 'IMMUTABLE_LEGAL_DRAFT “koşul ve istisna” [1]\r\n"literal" \\ path\n' * 200


def tools(name: str = "submit_partial_answer") -> list[dict[str, JsonValue]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "_language": {"type": "string", "enum": ["tr"]},
                    },
                    "required": ["answer", "_language"],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.mark.parametrize("name", ["submit_answer", "submit_partial_answer"])
def test_valid_terminal_answer_is_absent_from_repair_and_restored_exactly(
    name: str,
) -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    llm.invoke.return_value = argument_patch_response('{"_language":"tr"}')
    definitions = tools(name)
    preserved_definitions = copy.deepcopy(definitions)
    original = {"answer": ANSWER, "_language": "tr", "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), name),
        definitions,
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    assert llm.invoke.call_count == 1
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    action = payload["invalid_actions"][0]
    assert json.loads(action["arguments_json"]) == {
        "_language": "tr",
        "basis": "originals",
    }
    assert action["host_retained_argument_keys"] == ["answer"]
    assert "answer" not in action["parameters"]["properties"]
    assert "answer" not in action["parameters"]["required"]
    assert "IMMUTABLE_LEGAL_DRAFT" not in json.dumps(payload)
    assert definitions == preserved_definitions
    call = decision.calls[0]
    assert call.argument_error is None
    assert call.arguments == {"answer": ANSWER, "_language": "tr"}
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert (
        json.loads(decision.assistant_message.tool_calls[0].function.arguments)[
            "answer"
        ].encode()
        == ANSWER.encode()
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"_language": "tr", "answer": ANSWER},
        {"_language": "tr", "answer": "Changed legal result"},
        {"_language": "tr", "basis": "originals"},
        {"_language": "en"},
        {},
    ],
)
def test_reintroduced_answer_or_invalid_remaining_fields_cannot_execute(
    patch: dict[str, JsonValue],
) -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    llm.invoke.return_value = argument_patch_response(json.dumps(patch))
    original = {"answer": ANSWER, "_language": "tr", "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), "submit_partial_answer"),
        tools(),
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    assert decision.calls[0].argument_error is not None
    assert decision.calls[0].arguments == original
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_legacy_profiles_keep_the_existing_full_argument_repair(profile: str) -> None:
    context = RunContext()
    context.services["research_profile"] = profile
    llm = scripted_model()
    corrected = {"answer": ANSWER, "_language": "tr"}
    llm.invoke.return_value = argument_patch_response(json.dumps(corrected))
    original = {**corrected, "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), "submit_partial_answer"),
        tools(),
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    action = payload["invalid_actions"][0]
    assert json.loads(action["arguments_json"]) == original
    assert "answer" in action["parameters"]["properties"]
    assert "host_retained_argument_keys" not in action
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments == corrected


@pytest.mark.parametrize("invalid_answer", [None, "", 17])
def test_invalid_answer_uses_existing_complete_argument_repair(
    invalid_answer: JsonValue,
) -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    corrected = {"answer": "Actual corrected response", "_language": "tr"}
    llm.invoke.return_value = argument_patch_response(json.dumps(corrected))
    original = {"answer": invalid_answer, "_language": "tr", "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), "submit_partial_answer"),
        tools(),
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert "host_retained_argument_keys" not in payload["invalid_actions"][0]
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments == corrected


def test_missing_answer_uses_existing_complete_argument_repair() -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    corrected = {"answer": "Actual corrected response", "_language": "tr"}
    llm.invoke.return_value = argument_patch_response(json.dumps(corrected))
    original = {"_language": "tr", "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), "submit_partial_answer"),
        tools(),
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert "host_retained_argument_keys" not in payload["invalid_actions"][0]
    assert "answer" in payload["invalid_actions"][0]["parameters"]["properties"]
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments == corrected


@pytest.mark.parametrize("nested", [False, True])
def test_answer_reference_keeps_full_schema_repair_and_valid_answer(
    nested: bool,
) -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    definitions = tools()
    function = definitions[0]["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {"Answer": {"type": "string", "minLength": 1}}
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    reference: JsonValue = {"$ref": "#/$defs/Answer"}
    properties["answer"] = {"allOf": [reference]} if nested else reference
    corrected = {"answer": ANSWER, "_language": "tr"}
    llm.invoke.return_value = argument_patch_response(json.dumps(corrected))
    original = {**corrected, "basis": "originals"}
    decision = ResearchModel(llm, context)._repair_action_arguments(
        tool_response(json.dumps(original), "submit_partial_answer"),
        definitions,
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    action = payload["invalid_actions"][0]
    assert "host_retained_argument_keys" not in action
    assert json.loads(action["arguments_json"]) == original
    assert action["parameters"] == parameters
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments == corrected
    assert decision.calls[0].arguments["answer"] == ANSWER
    assert llm.invoke.call_count == 1


def test_truncated_response_does_not_freeze_its_answer() -> None:
    context = RunContext()
    context.services["research_profile"] = "experimental"
    llm = scripted_model()
    original = {"answer": ANSWER, "_language": "tr", "basis": "originals"}
    llm.invoke.return_value = argument_patch_response(
        json.dumps({"answer": ANSWER, "_language": "tr"})
    )
    response = tool_response(json.dumps(original), "submit_partial_answer")
    response.choice.finish_reason = "length"
    decision = ResearchModel(llm, context)._repair_action_arguments(
        response, tools(), "Which rule applies?", LLMFlow.ASV3_COORDINATOR
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert "host_retained_argument_keys" not in payload["invalid_actions"][0]
    assert json.loads(payload["invalid_actions"][0]["arguments_json"]) == original
    assert decision.calls[0].argument_error is None
