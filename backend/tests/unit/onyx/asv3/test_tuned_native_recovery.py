"""Check tuned recovery boundaries without altering legal evidence or wire identities."""

import json
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.error_handling.exceptions import OnyxError
from onyx.server.query_and_chat.streaming_models import AgentResponseDelta, ASv3Progress
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_experimental_argument_repair import ANSWER, tools
from tests.unit.onyx.asv3.test_experimental_review_handoff import content_response
from tests.unit.onyx.asv3.test_experimental_workflow import terminal_registry
from tests.unit.onyx.asv3.test_legal_source_reviews import seen
from tests.unit.onyx.asv3.test_model_adapter import (
    scripted_model,
    text_response,
    tool_response,
)
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_runtime import setup_run
from tests.unit.onyx.asv3.test_shared_originals import full_record
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


@pytest.mark.parametrize("foreign_id", [False, True])
def test_repair_retains_answer_and_wire_identity_without_echoing_opaque_signature(
    foreign_id: bool,
) -> None:
    context = RunContext(services={"asv3_workflow_variant": ASV3_TUNED_VARIANT})
    context.services["last_model_call_id"] = "semantic-call"
    selected = scripted_model()
    opaque_id = "call-1__thought__" + "opaque" * 2000
    original = tool_response(
        json.dumps({"answer": ANSWER, "_language": "tr", "extra": 1}), "submit_answer"
    )
    assert original.choice.message.tool_calls
    original.choice.message.tool_calls[0].id = opaque_id

    def repair(**arguments: Any) -> Any:
        payload = json.loads(arguments["prompt"][1].content)
        action = payload["invalid_actions"][0]
        assert action["call_id"] == "repair:0"
        assert "opaque" not in json.dumps(payload)
        assert "IMMUTABLE_LEGAL_DRAFT" not in json.dumps(payload)
        assert action["host_retained_argument_keys"] == ["answer"]
        return text_response(
            {
                "entries": [
                    {
                        "call_id": "foreign" if foreign_id else "repair:0",
                        "arguments_json": '{"_language":"tr"}',
                        "explanation": "Remove the undeclared field.",
                    }
                ]
            }
        )

    selected.invoke.side_effect = repair
    adapter = ResearchModel(selected, context)
    adapter.last_call_id = "semantic-call"
    decision = adapter._repair_action_arguments(
        original,
        tools("submit_answer"),
        "Which rule applies?",
        LLMFlow.ASV3_COORDINATOR,
    )
    call = decision.calls[0]
    assert call.call_id == opaque_id
    assert call.arguments["answer"] == ANSWER
    assert bool(call.argument_error) is foreign_id
    assert selected.invoke.call_count == 1
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert decision.assistant_message.tool_calls[0].id == opaque_id
    if not foreign_id:
        assert call.arguments == {"_language": "tr", "answer": ANSWER}
        assert adapter.last_call_id == "semantic-call"
        assert context.services["last_model_call_id"] == "semantic-call"


def test_pending_own_originals_receive_acquisition_before_terminal_metadata() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    selected = model()
    selected.invoke.return_value = native_action(
        "read_provision", {"source_id": "decision"}
    )
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={"tools": terminal_registry([]).definitions(context)}
    )
    ResearchModel(selected, context, lean_native_mode=True).decide(current)
    payload = last_payload(selected)
    actions = cast(list[dict[str, JsonValue]], payload["related_source_acquisition"])
    assert actions[0]["source_id"] == "decision"
    assert actions[0]["suggested_acquisition"] == {
        "name": "read_evidence",
        "arguments": {"citation": 2},
    }
    assert "before assessing" in str(payload["related_source_terminal_transport"])
    assert "return only one strict JSON object" not in str(
        payload["related_source_terminal_transport"]
    )
    assert selected.invoke.call_count == 1


@pytest.mark.parametrize("extra", [False, True])
def test_content_control_envelope_uses_normal_action_validation(extra: bool) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    selected = model()
    arguments = {"answer": "Rule [1].", "basis": "originals"}
    if extra:
        arguments["extra"] = "invalid"
    selected.invoke.side_effect = [
        content_response(json.dumps({"name": "submit_answer", "arguments": arguments})),
        text_response({"entries": []}),
    ]
    registry = terminal_registry([])
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter.decide(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
        ).model_copy(update={"tools": registry.definitions(context)})
    )
    assert decision.answer is None
    assert decision.calls[0].name == "submit_answer"
    assert bool(decision.calls[0].argument_error) is extra
    outcome = registry.dispatch(decision.calls[0], context)
    assert outcome.status == (OutcomeStatus.INVALID if extra else OutcomeStatus.FOUND)
    assert reviews.view(context, ledger, {1})["pending_lead_ids"]
    assert (
        reviews.publication_gap(
            "Rule [1].", adapter.last_call_id or "", context, ledger
        )
        is not None
    )


def test_unexposed_content_action_is_rejected_before_a_new_valid_decision() -> None:
    context, ledger, _reviews = tuned_context()
    selected = model()
    selected.invoke.side_effect = [
        content_response('{"name":"invented_tool","arguments":{}}'),
        native_action("read_provision", {"source_id": "decision"}),
    ]
    decision = ResearchModel(selected, context, lean_native_mode=True).decide(
        adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    )
    assert [call.name for call in decision.calls] == ["read_provision"]
    assert selected.invoke.call_count == 2


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_control_envelope_conversion_is_not_enabled_for_protected_modes(
    profile: str,
) -> None:
    selected = model()
    adapter = ResearchModel(
        selected, RunContext(services={"research_profile": profile})
    )
    response = content_response('{"name":"submit_answer","arguments":{}}')
    assert adapter._tuned_content_action_response(response) is response


def test_stopped_tuned_research_never_publishes_failure_placeholder_as_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, selected, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language", None)
    kwargs["workflow_variant"] = ASV3_TUNED_VARIANT
    kwargs["research_profile"] = "normal"
    selected.invoke.side_effect = RunStopped(
        "Invalid provider action could not be recovered"
    )
    with pytest.raises(OnyxError):
        runtime.run_asv3_loop(**kwargs)
    assert (
        checkpoints[-1]["publication_stop_reason"]
        == "Invalid provider action could not be recovered"
    )
    packets = [packet for _tag, packet in queue.queue]
    assert not any(isinstance(packet.obj, AgentResponseDelta) for packet in packets)
    assert not any(
        isinstance(packet.obj, ASv3Progress) and packet.obj.phase == "completed"
        for packet in packets
    )
