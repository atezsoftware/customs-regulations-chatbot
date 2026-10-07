import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import BadRequestError, InternalServerError
from pydantic import JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import (
    ChatCompletionMessageToolCall,
    Choice,
    Message,
    ModelResponse,
)
from onyx.llm.model_response import FunctionCall as ResponseFunctionCall
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    FunctionCall,
    ToolCall,
    ToolChoiceOptions,
    ToolMessage,
    UserMessage,
)
from tests.unit.onyx.asv3.test_shared_originals import (
    full_record,
    recorded,
)
from tests.unit.onyx.asv3.test_shared_originals import (
    original as provision_original,
)


def model(limit: int = 100000) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=limit,
    )
    llm.invoke.return_value = ModelResponse(
        id="answer", created="0", choice=Choice(message=Message(content="Rule [1]."))
    )
    return llm


def source_answer_adapter() -> tuple[ResearchModel, RunContext, MagicMock, MagicMock]:
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    context.services.update(
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        explicit_research_temperature=True,
        evidence=EvidenceLedger(),
    )
    research, writer = model(), model()
    research.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0.1,
        max_input_tokens=1000000,
    )
    writer.config = LLMConfig(
        model_provider="anthropic",
        model_name="claude-sonnet-5-5",
        temperature=1,
        max_input_tokens=1000000,
    )
    return (
        ResearchModel(research, context, answer_llm=writer, lean_native_mode=True),
        context,
        research,
        writer,
    )


def test_source_answer_handoff_preserves_sources_without_research_draft_or_provider_history() -> (
    None
):
    adapter, context, research, writer = source_answer_adapter()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    general = original(ledger, context, "General obligation and application.")
    relief = original(
        ledger, context, "Notice before detection reduces the consequence."
    )
    current = adaptive_tool_view(
        original_evidence=[relief],
        turns=[turn("read-general", [general, relief])],
    )
    research.invoke.return_value = ModelResponse(
        id="candidate", created="0", choice=Choice(message=Message(content="Duty [1]."))
    )
    writer.invoke.return_value = ModelResponse(
        id="answer",
        created="0",
        choice=Choice(message=Message(content="Duty [1]; conditional relief [2].")),
    )
    result = adapter.decide(current)
    assert result.answer == "Duty [1]; conditional relief [2]."
    assert research.invoke.call_count == writer.invoke.call_count == 1
    payload = last_payload(writer)
    assert "draft_to_repair" not in payload
    assert "research_candidate" not in payload
    assert "candidate_outcome_coverage" not in payload
    sources = payload["original_evidence"]
    assert {(row["citation"], row["text"]) for row in sources} == {
        (general["citation"], general["text"]),
        (relief["citation"], relief["text"]),
    }
    messages = writer.invoke.call_args.kwargs["prompt"]
    assert not any(
        isinstance(message, (AssistantMessage, ToolMessage)) for message in messages
    )
    assert "EVERY legal assertion" in messages[0].content
    assert writer.invoke.call_args.kwargs["tool_choice"].value == "auto"
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    research.with_seed.assert_not_called()
    writer.with_seed.assert_not_called()


def test_source_answer_publication_repair_does_not_repeat_research_model() -> None:
    adapter, context, research, writer = source_answer_adapter()
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    current = view(
        original_evidence=[record],
        draft_to_repair="Conditional rule [1].",
        publication_gap={"uncited_application": True},
    )
    result = adapter.decide(current)
    assert result.answer == "Rule [1]."
    research.invoke.assert_not_called()
    assert writer.invoke.call_count == 1
    assert "Conditional rule" in json.dumps(last_payload(writer), ensure_ascii=False)


def test_research_and_writer_get_distinct_communication_defaults() -> None:
    from onyx.prompts.asv3.research import DEFAULT_RESPONSE_PREFERENCES
    from onyx.prompts.asv3.tuned import TUNED_RESEARCH_HANDOFF_PREFERENCES

    adapter, context, research, writer = source_answer_adapter()
    context.services["assistant_instructions"] = "Preserve relevant legal exceptions."
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    adapter.decide(adaptive_tool_view(original_evidence=[record]))
    research_question = research.invoke.call_args.kwargs["prompt"][1].content
    writer_question = writer.invoke.call_args.kwargs["prompt"][1].content
    assert TUNED_RESEARCH_HANDOFF_PREFERENCES in research_question
    assert DEFAULT_RESPONSE_PREFERENCES not in research_question
    assert DEFAULT_RESPONSE_PREFERENCES in writer_question
    assert "Preserve relevant legal exceptions." in research_question
    assert "Preserve relevant legal exceptions." in writer_question

    ordinary = model()
    ResearchModel(ordinary, RunContext(), lean_native_mode=True).decide(view())
    ordinary_question = ordinary.invoke.call_args.kwargs["prompt"][1].content
    assert DEFAULT_RESPONSE_PREFERENCES in ordinary_question
    assert TUNED_RESEARCH_HANDOFF_PREFERENCES not in ordinary_question


def test_source_answer_can_request_missing_evidence_then_return_to_research() -> None:
    adapter, context, research, writer = source_answer_adapter()
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    current = adaptive_tool_view(original_evidence=[record])
    writer.invoke.side_effect = [
        native_response("read_provision", '{"source_id":"existing"}'),
        ModelResponse(
            id="complete",
            created="0",
            choice=Choice(message=Message(content="Rule [1].")),
        ),
    ]
    first = adapter.decide(current)
    assert first.calls[0].name == "read_provision"
    assert adapter._answer_researching is True
    result = adapter.decide(
        current.model_copy(
            update={
                "draft_to_repair": "Old candidate [1].",
                "publication_gap": {"unread_parameter": True},
            }
        )
    )
    assert result.answer == "Rule [1]."
    assert research.invoke.call_count == writer.invoke.call_count == 2
    assert adapter._answer_researching is False


def test_source_answer_snapshot_fences_model_policy_and_sampling() -> None:
    adapter, _, _, _ = source_answer_adapter()
    snapshot = adapter.native_sampling_snapshot()
    policy = snapshot["settings"]
    assert isinstance(policy, dict)
    assert policy["source_answer_model"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "research_provider": "vertex_ai",
        "research_model": "gemini-3.8-flash",
        "research_temperature": 0.1,
    }
    adapter.restore_native_sampling({"native_coordinator_sampling": snapshot})
    with pytest.raises(ValueError, match="same coordinator sampling"):
        adapter.restore_native_sampling(
            {
                "native_coordinator_sampling": {
                    **snapshot,
                    "settings": {"mode": "unchanged"},
                }
            }
        )


@pytest.mark.parametrize(
    "profile,variant,depth",
    [
        ("normal", "standard", 0),
        ("experimental", ASV3_TUNED_VARIANT, 0),
        ("normal", ASV3_TUNED_VARIANT, 1),
    ],
)
def test_source_answer_model_cannot_change_protected_modes(
    profile: str, variant: str, depth: int
) -> None:
    context = RunContext(depth=depth)
    context.services.update(research_profile=profile, asv3_workflow_variant=variant)
    with pytest.raises(ValueError, match="isolated root experiment"):
        ResearchModel(model(), context, answer_llm=model(), lean_native_mode=True)


def test_source_answer_does_not_add_writer_to_conversation_or_clarification() -> None:
    adapter, _, research, writer = source_answer_adapter()
    research.invoke.return_value = native_response(
        "ask_user", '{"question":"Which date?"}'
    )
    current = view().model_copy(
        update={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "parameters": {
                            "type": "object",
                            "properties": {"question": {"type": "string"}},
                            "required": ["question"],
                        },
                    },
                }
            ]
        }
    )
    assert adapter.decide(current).calls[0].name == "ask_user"
    writer.invoke.assert_not_called()


def test_source_answer_does_not_replace_research_calls_with_a_prose_handoff() -> None:
    adapter, context, research, writer = source_answer_adapter()
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    response = native_response("read_provision", '{"source_id":"existing"}')
    response.choice.message.content = "I will read the unresolved effect."
    research.invoke.return_value = response
    decision = adapter.decide(adaptive_tool_view(original_evidence=[record]))
    assert decision.calls[0].name == "read_provision"
    writer.invoke.assert_not_called()


def test_research_handoff_changes_role_without_changing_terminal_validation() -> None:
    from onyx.asv3.source_answer_transport import source_research_handoff_tools

    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Publish a complete answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {"type": "string", "enum": ["originals"]},
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
            },
        },
        {"type": "function", "function": {"name": "read_provision"}},
    ]
    before = json.dumps(tools)
    projected = source_research_handoff_tools(tools)
    assert json.dumps(tools) == before
    assert projected[1] == tools[1]
    projected_function = projected[0]["function"]
    assert isinstance(projected_function, dict)
    assert "separate answer writer" in str(projected_function["description"])
    parameters = projected_function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict) and isinstance(properties["answer"], dict)
    properties["answer"].pop("description")
    original_function = tools[0]["function"]
    assert isinstance(original_function, dict)
    assert parameters == original_function["parameters"]


def test_invalid_source_handoff_cannot_trigger_the_answer_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, context, _research, writer = source_answer_adapter()
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    invalid = Decision(
        calls=[
            CapabilityCall(
                name="submit_answer",
                arguments={"answer": "Rule [1].", "basis": "originals"},
                argument_error="Tool arguments violate the exposed schema",
            )
        ]
    )
    monkeypatch.setattr(adapter, "_invoke_decision", lambda *_args, **_kwargs: invalid)
    result = adapter.decide(adaptive_tool_view(original_evidence=[record]))
    assert result.calls[0].argument_error
    writer.invoke.assert_not_called()


def test_source_writer_gets_exact_foreign_reference_leads_from_uncited_passages() -> (
    None
):
    from tests.unit.onyx.asv3.test_native_authority import (
        original as authority_original,
    )

    adapter, context, _research, writer = source_answer_adapter()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    ledger.add(
        [
            authority_original("8917 sayılı Faaliyet Kanunu", "27"),
            authority_original(
                "İşlem Tebliği",
                "9",
                kind="tebliğ",
                text="Bu sonuç Faaliyet Kanununun 38 inci maddesine göre belirlenir.",
            ),
        ],
        context,
    )
    adapter.decide(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
        )
    )
    payload = last_payload(writer)
    catalogue = payload["source_contained_references"]
    assert catalogue["references"][0]["citation"] == 2
    assert catalogue["references"][0]["article"] == "38"
    assert catalogue["references"][0]["instrument_number"] == "8917"
    assert catalogue["references"][0]["role"] == "source_contained_reference_navigation"
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert writer.invoke.call_args.kwargs["tool_choice"].value == "auto"


@pytest.mark.parametrize("terminal", ["submit_answer", "submit_partial_answer"])
def test_source_answer_binds_same_response_text_before_terminal_validation(
    terminal: str,
) -> None:
    from tests.unit.onyx.asv3.test_runtime import response

    adapter, context, _research, writer = source_answer_adapter()
    record = original(
        cast(EvidenceLedger, context.services["evidence"]), context, "Rule."
    )
    body = "Duty [1].\n\nIts material condition and supported application [1]."
    arguments = {"basis": "originals"} if terminal == "submit_answer" else {}
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            **({"basis": {"type": "string"}} if arguments else {}),
        },
        "required": ["answer", *arguments],
        "additionalProperties": False,
    }
    writer.invoke.return_value = response(body, calls=[(terminal, arguments)])
    current = view(original_evidence=[record]).model_copy(
        update={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": terminal, "parameters": schema},
                }
            ]
        }
    )
    result = adapter.decide(current)
    assert result.calls[0].arguments == {"answer": body, **arguments}
    assert result.calls[0].argument_error is None
    emitted = writer.invoke.call_args.kwargs["tools"][0]["function"]["parameters"]
    assert "answer" not in emitted["properties"]
    assert "answer" not in emitted["required"]
    assert schema["properties"]["answer"] == {"type": "string"}


def test_source_answer_reads_a_bound_pending_lead_before_calling_writer() -> None:
    from tests.unit.onyx.asv3.test_legal_source_reviews import seen
    from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context

    context, ledger, reviews = tuned_context()
    law = ledger.get(1)
    assert law is not None
    ledger = EvidenceLedger()
    ledger.add([law], context)
    context.services["evidence"] = ledger
    ledger.record_delivery("law-call", "asv3_coordinator", [full_record(ledger, 1)])
    seen(context, ledger, reviews)
    research, writer = model(), model()
    adapter = ResearchModel(research, context, answer_llm=writer, lean_native_mode=True)
    research.invoke.return_value = ModelResponse(
        id="candidate",
        created="0",
        choice=Choice(message=Message(content="Current result [1].")),
    )
    current = view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_source_range",
                        "parameters": {"type": "object"},
                    },
                },
            ]
        }
    )
    result = adapter.decide(current)
    assert [(call.name, call.arguments) for call in result.calls] == [
        ("read_source_range", {"source_id": "decision", "start": 0})
    ]
    writer.invoke.assert_not_called()


def original(
    ledger: EvidenceLedger, context: RunContext, text: str
) -> dict[str, JsonValue]:
    number = ledger.add(
        [EvidenceItem(source_id="law", chunk_id=str(len(text)), text=text)], context
    )[0]
    return cast(dict[str, JsonValue], json.loads(ledger.serialize_records([number]))[0])


def turn(
    identity: str, records: list[dict[str, JsonValue]], *, extra: str = ""
) -> ResearchTurn:
    return ResearchTurn(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=identity,
                    function=FunctionCall(
                        name="read_provision", arguments='{"article":"168"}'
                    ),
                )
            ]
        ),
        results=[
            ToolMessage(
                tool_call_id=identity,
                content=json.dumps(
                    {
                        "outcome": {"status": "found", "data": {"provision": "168"}},
                        "evidence_ids": [record["citation"] for record in records],
                        "original_evidence": records,
                        "navigation": extra,
                    },
                    ensure_ascii=False,
                ),
            )
        ],
    )


def view(**kwargs: Any) -> HarnessView:
    return HarnessView(
        request="Explain both outcomes without dropping conditions.",
        questions=["Explain both outcomes without dropping conditions."],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
        **kwargs,
    )


def last_payload(llm: MagicMock) -> dict[str, Any]:
    content = llm.invoke.call_args.kwargs["prompt"][-1].content
    return cast(dict[str, Any], json.loads(content[0].text))


def native_response(
    name: str | None,
    arguments: str | None = "{}",
    *,
    call_id: str = "native-call",
) -> ModelResponse:
    return ModelResponse(
        id="native-response",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=call_id,
                        function=ResponseFunctionCall(name=name, arguments=arguments),
                    )
                ]
            )
        ),
    )


@pytest.mark.parametrize(
    "name,arguments,call_id",
    [
        ("report_message", '{"title":"Status","message":"Work continues"}', "live-id"),
        (
            "repair_question_answer",
            '{"expected_answer_hash":"recorded","question_id":"q1","edits":[]}',
            "live-repair-id",
        ),
        ("unavailable_capability", "{}", "unknown-name"),
        ("read_provision", "{}", ""),
        (None, "{}", "missing-name"),
        ("read_provision", None, "null-arguments"),
    ],
)
def test_malformed_native_batch_gets_a_fresh_decision_from_actual_exposed_tools(
    name: str | None, arguments: str | None, call_id: str
) -> None:
    context, llm = RunContext(), model()
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["task_outcome_ids"] = ["retained-outcome"]
    record = original(ledger, context, "Applicable original text")
    current = adaptive_tool_view(original_evidence=[record])
    rejected = native_response(name, arguments, call_id=call_id)
    rejected_copy = rejected.model_dump()
    llm.invoke.side_effect = [
        rejected,
        native_response("read_provision", '{"source_id":"existing"}', call_id="fresh"),
    ]
    decision = ResearchModel(llm, context, lean_native_mode=True).decide(current)
    assert [call.name for call in decision.calls] == ["read_provision"]
    assert decision.calls[0].call_id == "fresh"
    first, recovery = llm.invoke.call_args_list
    assert recovery.kwargs["prompt"][:-1] == first.kwargs["prompt"]
    assert recovery.kwargs["tools"] == first.kwargs["tools"]
    assert "rejected before execution" in recovery.kwargs["prompt"][-1].content
    assert "report_message" not in {
        tool["function"]["name"] for tool in recovery.kwargs["tools"]
    }
    assert context.services["task_outcome_ids"] == ["retained-outcome"]
    retained = ledger.get(1)
    assert retained is not None
    assert retained.text == record["text"]
    assert rejected.model_dump() == rejected_copy
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["tools"] == 0


@pytest.mark.parametrize("provider_error", [BadRequestError, InternalServerError])
def test_provider_native_rejection_gets_fresh_decision_without_transport_retry(
    provider_error: type[BadRequestError] | type[InternalServerError],
) -> None:
    context, llm = RunContext(), model()
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["task_outcome_ids"] = ["retained"]
    record = original(ledger, context, "Retained governing text")
    current = adaptive_tool_view(original_evidence=[record])
    llm.invoke.side_effect = [
        provider_error(
            message="MALFORMED_FUNCTION_CALL: private-provider-detail",
            model="mock",
            llm_provider="vertex_ai",
        ),
        native_response("read_provision", '{"source_id":"existing"}'),
    ]
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    decision = adapter.decide(current)
    assert [call.name for call in decision.calls] == ["read_provision"]
    first, corrected = llm.invoke.call_args_list
    assert corrected.kwargs["prompt"][:-1] == first.kwargs["prompt"]
    assert corrected.kwargs["tools"] == first.kwargs["tools"]
    diagnostic = corrected.kwargs["prompt"][-1].content
    assert "MALFORMED_FUNCTION_CALL" in diagnostic
    assert "private-provider-detail" not in diagnostic
    assert ledger.get(1) is not None
    assert context.services["task_outcome_ids"] == ["retained"]
    assert context.budget.snapshot()["tools"] == 0
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("has_calls", [True, False])
def test_malformed_provider_finish_rejects_content_and_calls(has_calls: bool) -> None:
    rejected = native_response("read_provision", '{"source_id":"rejected"}')
    rejected.choice.finish_reason = "MALFORMED_FUNCTION_CALL"
    rejected.choice.message.content = "A seemingly complete answer."
    if not has_calls:
        rejected.choice.message.tool_calls = None
    llm = model()
    llm.invoke.side_effect = [
        rejected,
        native_response("read_provision", '{"source_id":"replacement"}'),
    ]
    decision = ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
        adaptive_tool_view()
    )
    assert decision.answer is None
    assert [call.arguments for call in decision.calls] == [{"source_id": "replacement"}]
    assert (
        "rejected before execution"
        in llm.invoke.call_args_list[1].kwargs["prompt"][-1].content
    )
    assert llm.invoke.call_count == 2


def test_repeated_provider_native_rejection_preserves_research_and_stops_protocol_loop() -> (
    None
):
    context, llm = RunContext(), model()
    context.services["task_outcome_ids"] = ["retained"]
    llm.invoke.side_effect = [
        BadRequestError(
            message=f"MALFORMED_FUNCTION_CALL: opaque-request-{index}",
            model="mock",
            llm_provider="vertex_ai",
        )
        for index in range(2)
    ]
    with pytest.raises(RunStopped, match="unchanged invalid native action batch"):
        ResearchModel(llm, context, lean_native_mode=True).decide(adaptive_tool_view())
    assert context.services["task_outcome_ids"] == ["retained"]
    assert context.budget.snapshot()["tools"] == 0
    assert llm.invoke.call_count == 2
    assert "opaque-request" not in llm.invoke.call_args.kwargs["prompt"][-1].content


@pytest.mark.parametrize("message", ["Invalid schema", "NOT_MALFORMED_FUNCTION_CALL"])
def test_unrelated_provider_bad_request_is_not_native_recovery(message: str) -> None:
    llm = model()
    llm.invoke.side_effect = BadRequestError(
        message=message, model="mock", llm_provider="vertex_ai"
    )
    with pytest.raises(BadRequestError):
        ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
            adaptive_tool_view()
        )
    assert llm.invoke.call_count == 1


def test_no_choices_without_provider_marker_is_not_native_recovery() -> None:
    llm = model()
    llm.invoke.side_effect = ValueError(
        "LiteLLM response must include at least one choice."
    )
    with pytest.raises(ValueError, match="at least one choice"):
        ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
            adaptive_tool_view()
        )
    assert llm.invoke.call_count == 1


def test_provider_rejection_during_truncated_completion_restarts_native_decision() -> (
    None
):
    llm = model()
    llm.invoke.side_effect = [
        ModelResponse(
            id="incomplete",
            created="0",
            choice=Choice(
                finish_reason="length", message=Message(content="Incomplete ")
            ),
        ),
        BadRequestError(
            message="MALFORMED_FUNCTION_CALL", model="mock", llm_provider="vertex_ai"
        ),
        native_response("read_provision"),
    ]
    decision = ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
        adaptive_tool_view()
    )
    assert [call.name for call in decision.calls] == ["read_provision"]
    first, _, corrected = llm.invoke.call_args_list
    assert corrected.kwargs["prompt"][:-1] == first.kwargs["prompt"]
    assert corrected.kwargs["tools"] == first.kwargs["tools"]
    assert llm.invoke.call_count == 3


def test_native_recovery_reuses_base_context_after_repeated_envelope_defects() -> None:
    llm = model()
    llm.invoke.side_effect = [
        native_response("report_message"),
        native_response("read_provision", call_id=""),
        native_response("read_provision", call_id="complete"),
    ]
    context = RunContext()
    decision = ResearchModel(llm, context, lean_native_mode=True).decide(
        adaptive_tool_view()
    )
    assert decision.calls[0].call_id == "complete"
    first, second, third = llm.invoke.call_args_list
    assert second.kwargs["prompt"][:-1] == first.kwargs["prompt"]
    assert third.kwargs["prompt"][:-1] == first.kwargs["prompt"]
    assert llm.invoke.call_count == 3


def test_identical_invalid_protocol_stops_without_echoing_opaque_provider_ids() -> None:
    context, llm = RunContext(), model()
    opaque = "__thought__" + "signature" * 1000
    llm.invoke.side_effect = [
        native_response(
            "report_message", '{"title":"Status","message":"Continues"}', call_id=opaque
        ),
        native_response(
            "report_message",
            '{ "message": "Continues", "title": "Status" }',
            call_id="different-provider-id-" + opaque,
        ),
    ]
    with pytest.raises(RunStopped, match="unchanged invalid native action batch"):
        ResearchModel(llm, context, lean_native_mode=True).decide(adaptive_tool_view())
    diagnostic = llm.invoke.call_args_list[1].kwargs["prompt"][-1].content
    assert "signature" not in diagnostic
    assert "__thought__" not in diagnostic
    assert '"call_index": 0' in diagnostic
    assert '"call_id_present": true' in diagnostic
    assert len(diagnostic) < 2000
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["tools"] == 0


def test_envelope_preflight_precedes_every_argument_patch() -> None:
    context, llm = RunContext(), model()
    current = adaptive_tool_view()
    current.tools[2]["function"] = {
        "name": "read_provision",
        "parameters": {
            "type": "object",
            "properties": {"article": {"type": "string"}},
            "required": ["article"],
        },
    }
    rejected = native_response("read_provision", '{"article":17}')
    assert rejected.choice.message.tool_calls is not None
    rejected.choice.message.tool_calls.append(
        ChatCompletionMessageToolCall(
            id="unexposed-peer",
            function=ResponseFunctionCall(name="report_message", arguments="{}"),
        )
    )
    llm.invoke.side_effect = [
        rejected,
        native_response("read_provision", '{"article":"17"}'),
    ]
    decision = ResearchModel(llm, context, lean_native_mode=True).decide(current)
    assert [call.arguments for call in decision.calls] == [{"article": "17"}]
    recovery = llm.invoke.call_args_list[1].kwargs
    assert recovery["tools"] == llm.invoke.call_args_list[0].kwargs["tools"]
    assert recovery["structured_response_format"] is None
    assert llm.invoke.call_count == 2


def test_duplicate_native_call_id_requires_replacement_of_the_whole_batch() -> None:
    rejected = native_response("read_provision", call_id="duplicate")
    assert rejected.choice.message.tool_calls is not None
    rejected.choice.message.tool_calls.append(
        ChatCompletionMessageToolCall(
            id="duplicate",
            function=ResponseFunctionCall(name="read_provision", arguments="{}"),
        )
    )
    llm = model()
    llm.invoke.side_effect = [
        rejected,
        native_response("read_provision", call_id="unique"),
    ]
    decision = ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
        adaptive_tool_view()
    )
    assert [call.call_id for call in decision.calls] == ["unique"]
    assert llm.invoke.call_count == 2


def test_native_envelope_recovery_respects_run_cancellation() -> None:
    context, llm = RunContext(), model()

    def invoke(**_arguments: Any) -> ModelResponse:
        if llm.invoke.call_count == 2:
            context.cancel()
        return native_response("report_message")

    llm.invoke.side_effect = invoke
    with pytest.raises(RunStopped, match="cancelled"):
        ResearchModel(llm, context, lean_native_mode=True).decide(adaptive_tool_view())
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["tools"] == 0


def test_a_malformed_peer_prevents_execution_of_the_entire_native_batch() -> None:
    executed: list[str] = []

    def read(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executed.append(str(arguments["label"]))
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Source read")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_test_source",
                description="Read the requested original",
                parameters={
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                },
                handler=read,
            )
        ]
    )
    rejected = native_response("read_test_source", '{"label":"rejected-batch"}')
    assert rejected.choice.message.tool_calls is not None
    malformed = native_response("report_message", call_id="unknown")
    assert malformed.choice.message.tool_calls is not None
    rejected.choice.message.tool_calls.extend(malformed.choice.message.tool_calls)
    llm, context = model(), RunContext()

    def invoke(**_arguments: Any) -> ModelResponse:
        if llm.invoke.call_count == 1:
            return rejected
        if llm.invoke.call_count == 2:
            assert executed == []
            return native_response("read_test_source", '{"label":"replacement"}')
        return ModelResponse(
            id="done", created="0", choice=Choice(message=Message(content="Complete"))
        )

    llm.invoke.side_effect = invoke
    result = Harness(
        request="Read the requested original",
        context=context,
        registry=registry,
        decide=ResearchModel(llm, context, lean_native_mode=True).decide,
    ).run()
    assert result.status is OutcomeStatus.FOUND
    assert executed == ["replacement"]
    assert len(result.receipts) == 1
    assert llm.invoke.call_count == 3


def independent_tool_view(**kwargs: Any) -> HarnessView:
    return view(**kwargs).model_copy(
        update={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": name, "parameters": {"type": "object"}},
                }
                for name in ("research_questions", "assemble_answers", "read_provision")
            ]
        }
    )


def adaptive_tool_view(**kwargs: Any) -> HarnessView:
    current = independent_tool_view(**kwargs)
    for name in ("search_corpus", "ask_user"):
        current.tools.append(
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
        )
    current.tools.append(
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {
                            "type": "string",
                            "enum": ["conversation", "scenario", "originals"],
                        },
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return current


@pytest.mark.parametrize(
    "action,arguments",
    [
        (
            "submit_answer",
            {"answer": "Merhaba, nasıl yardımcı olabilirim?", "basis": "conversation"},
        ),
        ("ask_user", {"question": "Belirleyici işlem hangi tarihte yapıldı?"}),
        ("read_provision", {"source_id": "known-source", "article": "7"}),
        (
            "search_corpus",
            {
                "query": "İşleme uygulanabilir koşul",
                "mode": "hybrid",
                "coverage_item": "İşlemin uygulanabilirliği",
                "evidence_target": "Uygulanabilir özgün koşul",
            },
        ),
        (
            "research_questions",
            {"questions": [{"question_id": "new", "question": "Yeni etki?"}]},
        ),
        (None, {}),
    ],
)
def test_first_native_decision_preserves_all_adaptive_choices_in_one_call(
    action: str | None, arguments: dict[str, Any]
) -> None:
    context, llm = RunContext(), model()
    context.services["independent_question_mode"] = True
    current = adaptive_tool_view()
    canonical_tools = current.model_dump(mode="json")["tools"]
    llm.invoke.return_value = ModelResponse(
        id="first-decision",
        created="0",
        choice=Choice(
            message=Message(
                content="Doğrudan yerel cevap." if action is None else None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="chosen-action",
                        function=ResponseFunctionCall(
                            name=action, arguments=json.dumps(arguments)
                        ),
                    )
                ]
                if action is not None
                else None,
            )
        ),
    )
    decision = ResearchModel(llm, context, lean_native_mode=True).decide(current)
    assert llm.invoke.call_count == 1
    assert llm.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert llm.invoke.call_args.kwargs["tools"] == canonical_tools
    assert current.model_dump(mode="json")["tools"] == canonical_tools
    if action is None:
        assert decision.answer == "Doğrudan yerel cevap." and decision.calls == []
    else:
        assert [call.name for call in decision.calls] == [action]
        assert decision.calls[0].arguments == arguments


def test_first_native_decision_can_submit_a_source_memory_answer_without_new_research() -> (
    None
):
    context, ledger, llm = RunContext(), EvidenceLedger(), model()
    source_text = "The applicable exception needs both supplied conditions. " * 500
    source = original(ledger, context, source_text)
    memory = {"status": "revalidated", "reused_evidence_numbers": [1]}
    context.services.update(
        independent_question_mode=True, session_research=memory, evidence=ledger
    )
    current = adaptive_tool_view(original_evidence=[source])
    answer = "Both supplied conditions meet the applicable exception [1]."
    llm.invoke.return_value = ModelResponse(
        id="source-memory-answer",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="finish",
                        function=ResponseFunctionCall(
                            name="submit_answer",
                            arguments=json.dumps(
                                {"answer": answer, "basis": "originals"}
                            ),
                        ),
                    )
                ]
            )
        ),
    )
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    decision = adapter.decide(current)
    assert llm.invoke.call_count == 1
    assert llm.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert [call.name for call in decision.calls] == ["submit_answer"]
    assert decision.calls[0].arguments == {"answer": answer, "basis": "originals"}
    assert last_payload(llm)["session_research"] == memory
    assert last_payload(llm)["original_evidence"] == [source]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}
    assert source["text"] == source_text
    assert context.services.get("question_research_started") is not True


@pytest.mark.parametrize("started", [False, True])
def test_independent_root_keeps_first_choice_adaptive_and_forces_completed_assembly(
    started: bool,
) -> None:
    context, llm = RunContext(), model()
    context.services.update(
        independent_question_mode=True,
        question_research_started=started,
        independent_answers=[{"question_id": "q1", "answer": "Complete answer."}]
        if started
        else [],
    )
    current = adaptive_tool_view()
    original_tools = current.model_dump(mode="json")["tools"]
    ResearchModel(llm, context, lean_native_mode=True).decide(current)
    assert llm.invoke.call_args.kwargs["tool_choice"] is (
        ToolChoiceOptions.REQUIRED if started else ToolChoiceOptions.AUTO
    )
    assert [t["function"]["name"] for t in llm.invoke.call_args.kwargs["tools"]] == (
        ["assemble_answers", "read_provision", "search_corpus"]
        if started
        else [
            "research_questions",
            "assemble_answers",
            "read_provision",
            "search_corpus",
            "ask_user",
            "submit_answer",
        ]
    )
    assert current.model_dump(mode="json")["tools"] == original_tools
    assert llm.invoke.call_count == 1


def test_independent_first_decision_can_ask_a_decisive_user_fact_with_auto_tools() -> (
    None
):
    context, llm = RunContext(), model()
    context.services["independent_question_mode"] = True
    current = independent_tool_view()
    current.tools.append(
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
    )
    clarification = "Bu işlemde eşyanın sahibi hangi taraftır?"
    llm.invoke.return_value = ModelResponse(
        id="clarification",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="ask-fact",
                        function=ResponseFunctionCall(
                            name="ask_user",
                            arguments=json.dumps({"question": clarification}),
                        ),
                    )
                ]
            )
        ),
    )
    canonical_tools = current.model_dump(mode="json")["tools"]
    decision = ResearchModel(llm, context, lean_native_mode=True).decide(current)
    assert llm.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert [
        tool["function"]["name"] for tool in llm.invoke.call_args.kwargs["tools"]
    ] == ["research_questions", "assemble_answers", "read_provision", "ask_user"]
    assert [call.name for call in decision.calls] == ["ask_user"]
    assert decision.calls[0].arguments == {"question": clarification}
    assert current.model_dump(mode="json")["tools"] == canonical_tools
    assert llm.invoke.call_count == 1


def test_independent_child_preserves_user_preferences_and_full_scenario_without_sibling_answers() -> (
    None
):
    context, llm = RunContext(), model()
    preferences = "Her alternatifin kendi dayanağını ve somut işlem adımlarını açıkla."
    context.services.update(
        assistant_instructions=preferences,
        independent_answers=[
            {"question_id": "sibling", "answer": "Sibling draft must not leak."}
        ],
        independent_question_mode=True,
    )
    child = context.independent_child()
    scenario = (
        "Tam kullanıcı senaryosu: A aktörü, B rejimi, iki ayrı alternatif ve tarih C."
    )
    current = independent_tool_view().model_copy(
        update={
            "request": "Yalnız ikinci alternatifin sonucunu araştır.",
            "questions": ["Yalnız ikinci alternatifin sonucunu araştır."],
        }
    )
    adapter = ResearchModel(llm, child, lean_native_mode=True, history=scenario)
    adapter.decide(current)
    payload, _ = first_user_payload(llm.invoke.call_args.kwargs["prompt"][1])
    assert payload == {
        "request": current.request,
        "conversation": scenario,
        "assistant_instructions": preferences,
    }
    assert "independent_answers" not in last_payload(llm)
    assert context.services["assistant_instructions"] == preferences
    assert child.services["assistant_instructions"] == preferences
    assert current.request == "Yalnız ikinci alternatifin sonucunu araştır."
    assert len(llm.invoke.call_args.kwargs["tools"]) == 3
    assert llm.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert llm.invoke.call_count == 1


def test_independent_child_keeps_native_source_tools_and_originals_without_timeout() -> (
    None
):
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    child, ledger, llm = context.independent_child(), EvidenceLedger(), model()
    child.services["evidence"] = ledger
    source = original(
        ledger,
        child,
        "The actor must satisfy both conditions, present the specified proof and complete later settlement.",
    )
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ("read_provision", "search_corpus", "submit_partial_answer")
    ]
    current = view(
        original_evidence=[source], required_evidence_numbers=[1]
    ).model_copy(update={"tools": tools})
    llm.invoke.side_effect = [
        ModelResponse(
            id="continuing",
            created="0",
            choice=Choice(
                message=Message(
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="follow-source",
                            function=ResponseFunctionCall(
                                name="read_provision", arguments="{}"
                            ),
                        )
                    ]
                )
            ),
        ),
        ModelResponse(
            id="complete",
            created="0",
            choice=Choice(
                message=Message(
                    content="Both conditions, the specified proof and later settlement apply [1]."
                )
            ),
        ),
    ]
    adapter = ResearchModel(llm, child, lean_native_mode=True)
    assert adapter.decide(current).calls[0].name == "read_provision"
    assert (
        adapter.decide(current).answer
        == "Both conditions, the specified proof and later settlement apply [1]."
    )
    assert llm.invoke.call_count == 2
    for call in llm.invoke.call_args_list:
        assert call.kwargs["timeout_override"] is None
        assert call.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
        assert call.kwargs["tools"] == tools
    assert last_payload(llm)["original_evidence"] == [source]
    assert "independent_finalization" not in last_payload(llm)
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}


@pytest.mark.parametrize("is_child", [False, True])
def test_session_research_and_revalidated_original_reach_root_and_child(
    is_child: bool,
) -> None:
    context, ledger, llm = RunContext(), EvidenceLedger(), model()
    memory = {
        "requests": ["Earlier question with an unresolved alternative."],
        "status": "revalidated",
        "reused_evidence_numbers": [1],
        "source_gaps": [],
    }
    context.services.update(
        independent_question_mode=True,
        session_research=memory,
        evidence=ledger,
    )
    source = original(
        ledger,
        context,
        "The alternative applies only when the actor meets both stated conditions.",
    )
    selected = context.independent_child() if is_child else context
    current = independent_tool_view(
        original_evidence=[source], required_evidence_numbers=[1]
    )
    history = (
        "Prior assistant prose is conversation context; the current facts changed."
    )
    adapter = ResearchModel(llm, selected, lean_native_mode=True, history=history)
    adapter.decide(current)

    initial, _ = first_user_payload(llm.invoke.call_args.kwargs["prompt"][1])
    assert initial["request"] == current.request
    assert initial["conversation"] == history
    assert last_payload(llm)["session_research"] == memory
    assert last_payload(llm)["original_evidence"] == [source]
    assert memory["requests"] == ["Earlier question with an unresolved alternative."]
    tool_names = [
        tool["function"]["name"] for tool in llm.invoke.call_args.kwargs["tools"]
    ]
    assert tool_names == ["research_questions", "assemble_answers", "read_provision"]
    assert llm.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert llm.invoke.call_count == 1
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}


def test_source_free_gap_assembly_rejects_added_legal_connections() -> None:
    from onyx.asv3.question_research import QuestionResearch
    from onyx.asv3.workers import WorkerPool

    context = RunContext()
    research = QuestionResearch(
        context, MagicMock(spec=WorkerPool), ["Unknown outcome?"]
    )
    gap = "Bu sonucun belirleyici özgün hükmü henüz doğrulanamadı."
    research.restore(
        {
            "answers": [
                {
                    "question_id": "only",
                    "question": "Unknown outcome?",
                    "answer": gap,
                    "status": "partial",
                    "evidence_numbers": [],
                }
            ]
        }
    )
    guard = MagicMock(return_value=None)
    research.publish_guard = guard
    with pytest.raises(ValueError, match="(?i)connections"):
        research.assemble_answers(
            {
                "order": ["only"],
                "connections": "Bu nedenle vergi otomatik olarak ortadan kalkar.",
            },
            context,
        )
    guard.assert_not_called()
    assert "assembled_answer" not in context.services
    research.assemble_answers({"order": ["only"]}, context)
    assert context.services["assembled_answer"] == gap
    assert context.services["independent_partial"] is True


def test_independent_answers_and_body_citations_survive_capacity_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_adapter, "COORDINATOR_PROMPT", "Keep complete answers and originals."
    )
    context, ledger, llm = RunContext(), EvidenceLedger(), model()
    context.services["evidence"] = ledger
    required = original(ledger, context, "Full cumulative condition. " * 24)
    optional = original(ledger, context, "Uncited supplementary original. " * 300)
    answers = [
        {
            "question_id": "q1",
            "question": "All conditions?",
            "answer": "Precise condition [1].\n" * 12,
            "status": "found",
            "evidence_numbers": [],
        }
    ]
    context.services.update(
        independent_question_mode=True,
        question_research_started=True,
        independent_answers=answers,
    )
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    current = independent_tool_view(original_evidence=[required, optional])
    complete, tools, _ = adapter._fit_native_decision(current)
    payload = json.loads(cast(str, complete[-1].content))
    payload["original_evidence"] = [required]
    payload["original_evidence_omitted"] = [
        {
            "citation": 2,
            "source_id": "law",
            "start_char": 0,
            "end_char": len(str(optional["text"])),
            "reason": "physical_model_context",
        }
    ]
    minimal = [
        *complete[:-1],
        UserMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    cost = adapter._input_cost(minimal, tools)
    llm.config.max_input_tokens = (cost * 4 + 2) // 3
    adapter.decide(current)
    assert last_payload(llm)["independent_answers"] == answers
    assert last_payload(llm)["original_evidence"] == [required]
    assert last_payload(llm)["original_evidence_omitted"][0]["citation"] == 2
    assert ledger.completely_delivered(cast(str, adapter.last_call_id)) == {1}
    assert llm.invoke.call_count == 1
    llm.config.max_input_tokens = 300
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(current)
    assert llm.invoke.call_count == 1
    assert context.services["independent_answers"] == answers


def test_independent_assembly_recovery_uses_global_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, llm = RunContext(), model()
    context.services.update(
        independent_question_mode=True,
        question_research_started=True,
        independent_answers=[{"question_id": "q1", "answer": "Complete answer."}],
    )
    research_check = MagicMock(side_effect=RunStopped("Research ended"))
    monkeypatch.setattr(context, "check_research_active", research_check)
    llm.invoke.side_effect = [
        ModelResponse(id="empty", created="0", choice=Choice(message=Message())),
        ModelResponse(
            id="partial",
            created="0",
            choice=Choice(
                finish_reason="length", message=Message(content="Exact first part. ")
            ),
        ),
        ModelResponse(
            id="complete",
            created="0",
            choice=Choice(message=Message(content="Exact remaining part.")),
        ),
    ]
    result = ResearchModel(llm, context, lean_native_mode=True).decide(
        independent_tool_view()
    )
    assert result.answer == "Exact first part. Exact remaining part."
    research_check.assert_not_called()
    assert llm.invoke.call_count == 3
    assert (
        llm.invoke.call_args_list[0].kwargs["tool_choice"] is ToolChoiceOptions.REQUIRED
    )
    assert llm.invoke.call_args_list[-1].kwargs["tool_choice"] is ToolChoiceOptions.NONE


def test_independent_researcher_and_publication_repair_keep_source_tools() -> None:
    context = RunContext()
    context.services.update(
        independent_question_mode=True,
        question_research_started=True,
        independent_answers=[
            {"question_id": "q1", "answer": "Sibling body must not leak."}
        ],
    )
    for selected_context, gap in [
        (context.child(), None),
        (context, {"missing": "original"}),
    ]:
        llm = model()
        ResearchModel(llm, selected_context, lean_native_mode=True).decide(
            independent_tool_view(publication_gap=gap)
        )
        assert len(llm.invoke.call_args.kwargs["tools"]) == 3
        assert llm.invoke.call_args.kwargs["tool_choice"] is (
            ToolChoiceOptions.AUTO
            if selected_context.depth
            else ToolChoiceOptions.REQUIRED
        )
        if selected_context.depth:
            assert "independent_answers" not in last_payload(llm)


def first_user_payload(message: ChatCompletionMessage) -> tuple[dict[str, Any], str]:
    assert isinstance(message, UserMessage)
    assert isinstance(message.content, str)
    question, offset = json.JSONDecoder().raw_decode(message.content)
    assert isinstance(question, dict)
    return cast(dict[str, Any], question), message.content[offset:]


def seed_forks(llm: MagicMock) -> list[MagicMock]:
    forks: list[MagicMock] = []

    def fork(seed: int) -> MagicMock:
        selected = model(llm.config.max_input_tokens)
        selected.config = llm.config.model_copy(update={"seed": seed})
        selected.invoke.side_effect = llm.invoke
        forks.append(selected)
        return selected

    llm.with_seed.side_effect = fork
    return forks


def test_native_coordinator_pins_first_and_adaptive_decisions_without_extra_calls() -> (
    None
):
    llm = model(500000)
    llm.config = llm.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-3.8-flash",
            "temperature": 1,
        }
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    adapter._native_output_capacity = 65536
    adapter.decide(view())
    first_state = adapter.native_sampling_snapshot()
    adapter.decide(view())
    assert [fork.config.seed for fork in forks] == [31, 1424088823]
    assert [fork.invoke.call_count for fork in forks] == [1, 1]
    assert llm.invoke.call_count == 2
    assert llm.config.seed is None
    assert all(fork.config.temperature == 1 for fork in forks)
    assert all(fork.invoke.call_args.kwargs["max_tokens"] == 65536 for fork in forks)
    assert first_state["first_decision_started"] is True
    assert first_state["first_decision_completed"] is True
    assert first_state["settings"] == {
        "mode": "native_coordinator",
        "version": 1,
        "first_seed": 31,
        "continuation_seed": 1424088823,
    }


def test_explicit_research_sampling_does_not_insert_automatic_seed() -> None:
    llm = model(500000)
    llm.config = llm.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-3.8-flash",
            "temperature": 0.1,
        }
    )
    context = RunContext(services={"explicit_research_temperature": True})
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter._native_output_capacity = 65536
    adapter.decide(view())
    adapter.decide(view())
    assert llm.invoke.call_count == 2
    llm.with_seed.assert_not_called()
    assert adapter.native_sampling_snapshot()["settings"] == {"mode": "unchanged"}
    assert llm.config.seed is None


@pytest.mark.parametrize(
    ("provider", "name", "seed", "depth", "native"),
    [
        ("vertex_ai", "gemini-3.8-flash", 0, 0, True),
        ("vertex_ai", "gemini-3.8-flash", 42, 0, True),
        ("openai", "gemini-3.8-flash", None, 0, True),
        ("vertex_ai", "another-model", None, 0, True),
        ("vertex_ai", "gemini-3.8-flash", None, 1, True),
        ("vertex_ai", "gemini-3.8-flash", None, 0, False),
    ],
)
def test_native_default_seed_profile_preserves_explicit_seed_and_other_llms(
    provider: str,
    name: str,
    seed: int | None,
    depth: int,
    native: bool,
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": provider, "model_name": name, "seed": seed}
    )
    adapter = ResearchModel(llm, RunContext(depth=depth), lean_native_mode=native)
    assert adapter._native_decision_llm(view()) is llm
    llm.with_seed.assert_not_called()
    assert llm.config.seed == seed


def test_native_provider_retries_keep_the_same_first_seed_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    successful = llm.invoke.return_value
    llm.invoke.side_effect = [
        RuntimeError("retryable provider failure"),
        successful,
        successful,
    ]
    monkeypatch.setattr(llm_adapter, "is_retryable_provider_error", lambda _error: True)
    monkeypatch.setattr(llm_adapter, "provider_retry_delay", lambda _error, _attempt: 0)
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    adapter._native_output_capacity = 65536
    adapter.decide(view())
    adapter.decide(view())
    assert [fork.config.seed for fork in forks] == [31, 1424088823]
    assert [fork.invoke.call_count for fork in forks] == [2, 1]
    assert llm.config.seed is None


@pytest.mark.parametrize(
    "progress",
    [
        {"pending_calls": [{"name": "read_provision"}]},
        {"turns": [{}]},
        {"receipts": [{}]},
        {"evidence": {"records": [{}]}},
        {"last_draft": "Supported outcome [1]."},
    ],
)
def test_native_resume_uses_continuation_for_recorded_decisions(
    progress: dict[str, JsonValue],
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    interrupted = adapter.native_sampling_snapshot()
    interrupted["first_decision_started"] = True
    for saved in (progress, {**progress, "native_coordinator_sampling": interrupted}):
        adapter.restore_native_sampling(saved)
        adapter._native_decision_llm(view())
    assert [fork.config.seed for fork in forks] == [1424088823, 1424088823]


def test_native_interrupted_first_resume_replays_first_and_rejects_changed_profile() -> (
    None
):
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    interrupted = adapter.native_sampling_snapshot()
    interrupted["first_decision_started"] = True
    adapter.restore_native_sampling({"native_coordinator_sampling": interrupted})
    adapter._native_decision_llm(view())
    assert forks[0].config.seed == 31
    llm.config = llm.config.model_copy(update={"seed": 0})
    with pytest.raises(ValueError, match="same coordinator sampling profile"):
        adapter.restore_native_sampling({"native_coordinator_sampling": interrupted})
    llm.invoke.assert_not_called()


def test_native_history_preserves_each_complete_batch_and_delivers_originals_once() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["assistant_instructions"] = "Answer briefly in Turkish."
    first = original(ledger, context, "The applicant must request the relief.")
    second = original(
        ledger, context, "Release requires completion of institution checks."
    )
    llm = model()
    history = "User: Preserve the two requested alternatives."
    adapter = ResearchModel(llm, context, lean_native_mode=True, history=history)
    batch1, batch2 = turn("read-law", [first]), turn("read-condition", [second])
    original_turns = [batch1.model_dump(mode="json"), batch2.model_dump(mode="json")]
    current = view(
        turns=[batch1, batch2],
        original_evidence=[first, second],
        research_state={"needs": [{"description": "UNNEEDED_BOARD" * 10000}]},
    )
    current = current.model_copy(
        update={
            "questions": ["First outcome?", "Second outcome?"],
            "facts": ["The user requested both outcomes."],
            "draft_to_repair": "Explain the request and later control [1] [2].",
            "publication_gap": {"summary": "Communicate the later control."},
        }
    )
    canonical_request = current.request
    canonical_questions = list(current.questions)
    adapter.decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    first_user, preference_paragraph = first_user_payload(prompt[1])
    assert first_user == {
        "request": canonical_request,
        "questions": canonical_questions,
        "conversation": history,
        "assistant_instructions": "Answer briefly in Turkish.",
    }
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    assert current.request == canonical_request
    assert current.questions == canonical_questions
    assert context.services["assistant_instructions"] == "Answer briefly in Turkish."
    tool_results = [message for message in prompt if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_results] == [
        "read-law",
        "read-condition",
    ]
    assert last_payload(llm)["original_evidence"] == [first, second]
    assert last_payload(llm)["request"] == current.request
    assert last_payload(llm)["questions"] == current.questions
    assert last_payload(llm)["recorded_facts"] == current.facts
    assert last_payload(llm)["draft_to_repair"] == current.draft_to_repair
    assert last_payload(llm)["publication_gap"] == current.publication_gap
    for result, record in zip(tool_results, [first, second], strict=True):
        payload = json.loads(result.content)
        assert "original_evidence" not in payload
        assert payload["original_evidence_refs"] == [
            {key: value for key, value in record.items() if key != "text"}
        ]
        assert payload["evidence_ids"] == [record["citation"]]
        assert payload["outcome"] == {"status": "found", "data": {"provision": "168"}}
    assert [
        batch1.model_dump(mode="json"),
        batch2.model_dump(mode="json"),
    ] == original_turns
    assert "UNNEEDED_BOARD" not in json.dumps(
        [message.model_dump(mode="json") for message in prompt]
    )
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    assert llm.invoke.call_count == 1
    first_prefix = prompt[:2]
    adapter.decide(current)
    assert llm.invoke.call_args.kwargs["prompt"][:2] == first_prefix
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("researcher", [False, True], ids=["coordinator", "researcher"])
def test_core_role_instructions_survive_response_preference_eviction(
    researcher: bool,
) -> None:
    context, ledger = RunContext(depth=int(researcher)), EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["assistant_instructions"] = "Keep the requested units in Turkish."
    first = original(ledger, context, "The first condition must hold. " * 16)
    second = original(ledger, context, "The second condition also applies. " * 16)
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    current = view(original_evidence=[first, second]).model_copy(
        update={
            "request": "İki sonucu ve belge şartlarını aynen açıkla.\nSonra süreyi belirt."
        }
    )
    canonical_request = current.request
    complete_prompt, tools, _ = adapter._fit_native_decision(current)
    question_content = complete_prompt[1].content
    assert isinstance(question_content, str)
    question, preference_paragraph = first_user_payload(complete_prompt[1])
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    expected_question = {"request": canonical_request}
    if not researcher:
        expected_question["assistant_instructions"] = (
            "Keep the requested units in Turkish."
        )
    assert question == expected_question
    _, offset = json.JSONDecoder().raw_decode(question_content)
    json_only = question_content[:offset]
    prompt_without_preferences = [
        complete_prompt[0],
        UserMessage(content=json_only),
        *complete_prompt[2:],
    ]
    required_input = adapter._input_cost(prompt_without_preferences, tools)
    llm.config.max_input_tokens = (required_input * 4 + 2) // 3
    ceiling, _ = adapter._limits(adapter._native_output_limit())
    assert required_input <= ceiling
    assert adapter._input_cost(complete_prompt, tools) > ceiling

    adapter.decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert prompt[0].content == (
        llm_adapter.RESEARCHER_PROMPT if researcher else llm_adapter.COORDINATOR_PROMPT
    )
    first_content = prompt[1].content
    assert isinstance(first_content, str)
    delivered_question, preference_paragraph = first_user_payload(prompt[1])
    assert first_content == json_only
    assert delivered_question == expected_question
    assert preference_paragraph == ""
    assert current.request == canonical_request
    assert (
        context.services["assistant_instructions"]
        == "Keep the requested units in Turkish."
    )
    assert last_payload(llm)["original_evidence"] == [first, second]
    assert "original_evidence_omitted" not in last_payload(llm)
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    assert llm.invoke.call_count == 1


def test_context_fitting_drops_transcript_atomically_but_restores_required_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use original evidence.")
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    first = original(ledger, context, "Only if institutional checks are complete.")
    second = original(ledger, context, "The security must also be acceptable.")
    llm = model(4000)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    adapter.decide(
        view(
            turns=[
                turn("old", [first], extra="OLD_NAVIGATION" * 1000),
                turn("latest", [second]),
            ],
            original_evidence=[first, second],
            required_evidence_numbers=[1, 2],
        )
    )
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["latest"]
    restored = last_payload(llm)["original_evidence"]
    assert restored == [first, second]
    assert "OLD_NAVIGATION" not in json.dumps(
        [message.model_dump(mode="json") for message in prompt]
    )
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    request = llm.invoke.call_args.kwargs
    assert (
        adapter._input_cost(prompt, []) + request["max_tokens"]
        <= llm.config.max_input_tokens
    )


@pytest.mark.parametrize("full_first", [False, True])
def test_final_originals_replace_only_literally_covered_partial_ranges(
    full_first: bool,
) -> None:
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger, context, "Permission requires the notice and subsequent control."
    )
    text = cast(str, full["text"])
    partial = {
        **full,
        "text": text[11:34],
        "start_char": 11,
        "end_char": 34,
        "total_chars": len(text),
        "truncated": True,
    }
    records = [full, partial] if full_first else [partial, full]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(
        view(turns=[turn("read-parts", records)], original_evidence=[partial, full])
    )
    assert last_payload(llm)["original_evidence"] == [full]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert llm.invoke.call_count == 1


def test_final_originals_keep_overlapping_ranges_without_reading_the_full_ledger_text() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger,
        context,
        "First permission. Later notice. Final control. Separate exception.",
    )
    text = cast(str, full["text"])
    parts = [
        {
            **full,
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
            "total_chars": len(text),
            "truncated": True,
        }
        for start, end in [(0, 25), (15, 45), (18, 23)]
    ]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=[turn("read-parts", parts)], original_evidence=parts))
    assert last_payload(llm)["original_evidence"] == parts[:2]
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    assert llm.invoke.call_count == 1


def test_final_originals_clear_only_current_ledger_verified_omissions() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger, context, "A request requires notice followed by authority control."
    )
    unknown = {"citation": 2, "reason": "serialized_evidence_limit"}
    changed = {"citation": 1, "text_hash": "another-version", "reason": "old_context"}
    beyond = {"citation": 1, "start_char": 0, "end_char": 1000, "reason": "old_context"}
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(
        view(
            turns=[turn("read-full", [full])],
            original_evidence_omitted=[{"citation": 1}, unknown, changed, beyond],
        )
    )
    assert last_payload(llm)["original_evidence"] == [full]
    assert last_payload(llm)["original_evidence_omitted"] == [unknown, changed, beyond]
    assert llm.invoke.call_count == 1


def test_physical_capacity_omission_is_explicit_and_never_changes_ledger_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use originals.")
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    required = original(ledger, context, "Required qualifying clause.")
    optional = original(ledger, context, "Older supplementary source. " * 150)
    llm = model(2500)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    adapter.decide(
        view(original_evidence=[required, optional], required_evidence_numbers=[1])
    )
    payload = last_payload(llm)
    assert payload["original_evidence"] == [required]
    assert payload["original_evidence_omitted"][0]["citation"] == 2
    assert payload["original_evidence_omitted"][0]["reason"] == "physical_model_context"
    stored = ledger.get(2)
    assert stored is not None and stored.text == optional["text"]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}


def test_required_original_cannot_be_clipped_to_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use originals.")
    context, ledger = RunContext(), EvidenceLedger()
    required = original(ledger, context, "Complete required condition. " * 150)
    llm = model(1500)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(
            view(original_evidence=[required], required_evidence_numbers=[1])
        )
    assert llm.invoke.call_count == 0


@pytest.mark.parametrize("lean", [False, True])
def test_lean_unlimited_context_defers_provider_timeout(lean: bool) -> None:
    context = RunContext(timeout_seconds=float("inf"))
    llm = model()
    ResearchModel(llm, context, lean_native_mode=lean).decide(view())
    assert llm.invoke.call_args.kwargs["timeout_override"] == (None if lean else 120)


@pytest.mark.parametrize(
    "arguments", ['{"article":"168","article":"169"}', '{"article":NaN}']
)
def test_invalid_historical_json_is_not_replayed(arguments: str) -> None:
    historical = turn("invalid", [])
    assert historical.assistant.tool_calls is not None
    historical.assistant.tool_calls[0].function.arguments = arguments
    llm = model()
    ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
        view(turns=[historical])
    )
    assert not any(
        isinstance(message, ToolMessage)
        for message in llm.invoke.call_args.kwargs["prompt"]
    )


def test_orphan_turn_is_not_sent_and_failed_arguments_remain_actionable() -> None:
    orphan = turn("broken", [])
    orphan.results = []
    receipt = ToolReceipt(
        call=CapabilityCall(name="read_provision", call_id="broken", arguments={}),
        outcome=ToolOutcome(
            status=OutcomeStatus.INVALID, summary="Specify the exact article."
        ),
        elapsed_seconds=0,
    )
    llm = model()
    current = view(turns=[turn("valid", [])])
    current.turns.insert(0, orphan)
    current.receipts = [receipt]
    ResearchModel(llm, RunContext(), lean_native_mode=True).decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["valid"]
    assert (
        last_payload(llm)["failed_calls"][0]["summary"] == "Specify the exact article."
    )
    assert llm.invoke.call_count == 1


def test_native_mode_inherits_into_worker_without_changing_targeted_structured_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_adapter, "RESEARCHER_PROMPT", "Research the assigned original."
    )
    context = RunContext()
    context.services["lean_native_mode"] = True
    context.services["assistant_instructions"] = "Coordinator-only instruction."
    llm = model()
    adapter = ResearchModel(llm, context.child())
    assert adapter.lean_native_mode
    current = view()
    canonical_request = current.request
    adapter.decide(current)
    assert (
        llm.invoke.call_args.kwargs["prompt"][0].content
        == "Research the assigned original."
    )
    assert llm.invoke.call_args.kwargs["structured_response_format"] is None
    first_user, preference_paragraph = first_user_payload(
        llm.invoke.call_args.kwargs["prompt"][1]
    )
    assert first_user == {"request": canonical_request}
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    assert current.request == canonical_request
    assert context.services["assistant_instructions"] == "Coordinator-only instruction."
    llm.invoke.return_value = ModelResponse(
        id="review",
        created="0",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "status": "supported",
                        "explanation": "The exact claim is supported.",
                        "required_conditions": [],
                        "missing_conditions": [],
                        "evidence_numbers": [1],
                        "safe_to_publish": True,
                    }
                )
            )
        ),
    )
    result = adapter.invoke_verification(
        "Check this exact claim.", '{"claim":"Rule [1]."}'
    )
    assert result.status == "supported"
    assert llm.invoke.call_args.kwargs["structured_response_format"] is not None
    assert llm.invoke.call_count == 2
    assert adapter.last_call_id is not None
    assert context.services.get("evidence") is None


def test_native_completed_answer_uses_provider_output_capacity_without_an_extra_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 16000)
    llm = model()
    llm.invoke.return_value.choice.finish_reason = "stop"
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    result = adapter.decide(view())
    assert result.answer == "Rule [1]."
    assert llm.invoke.call_count == 1
    assert llm.invoke.call_args.kwargs["max_tokens"] == 16000
    assert adapter.last_finish_reason == "stop" and not adapter.last_response_truncated


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "MAX_OUTPUT_TOKENS"])
def test_native_truncated_text_continues_from_same_originals_and_preserves_exact_prefix(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    record = original(ledger, context, "The quoted condition and later outcome.")
    llm = model()
    prefix = "## 1\nThe original states “The quoted condition"
    suffix = " and later outcome.” [1]\n\n## 2\nThe second requested outcome [1]."
    llm.invoke.side_effect = [
        ModelResponse(
            id="cut",
            created="0",
            choice=Choice(
                finish_reason=finish_reason,
                message=Message(content=prefix),
            ),
        ),
        ModelResponse(
            id="complete",
            created="0",
            choice=Choice(
                finish_reason="stop",
                message=Message(content=suffix),
            ),
        ),
    ]
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    result = adapter.decide(view(original_evidence=[record]))
    assert result.answer == prefix + suffix
    assert llm.invoke.call_count == 2
    continuation = llm.invoke.call_args.kwargs
    assert continuation["tools"] is None
    assert any(
        isinstance(message, AssistantMessage) and message.content == prefix
        for message in continuation["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert not adapter.last_response_truncated


def test_native_truncated_action_never_executes_partial_arguments_when_no_capacity_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    llm = model()
    llm.invoke.return_value = ModelResponse(
        id="cut",
        created="0",
        choice=Choice(
            finish_reason="length",
            message=Message.model_validate(
                {
                    "tool_calls": [
                        {
                            "id": "unfinished",
                            "function": {
                                "name": "read_provision",
                                "arguments": '{"source_id":',
                            },
                        }
                    ]
                }
            ),
        ),
    )
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    with pytest.raises(RunStopped, match="truncated action arguments"):
        adapter.decide(view())
    assert llm.invoke.call_count == 1 and adapter.last_response_truncated


def test_native_failed_continuation_never_returns_a_complete_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    llm = model()
    llm.invoke.side_effect = [
        ModelResponse(
            id="cut",
            created="0",
            choice=Choice(
                finish_reason="length", message=Message(content="Unfinished claim")
            ),
        ),
        ModelResponse(
            id="empty",
            created="0",
            choice=Choice(finish_reason="stop", message=Message(content="")),
        ),
    ]
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    with pytest.raises(RunStopped, match="did not complete"):
        adapter.decide(view())
    assert llm.invoke.call_count == 2 and adapter.last_response_truncated


def native_action(name: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        id=name,
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=name,
                        function=ResponseFunctionCall(
                            name=name,
                            arguments=json.dumps(arguments),
                        ),
                    ),
                ]
            )
        ),
    )


@pytest.mark.parametrize("depth", [0, 1])
def test_intermediate_model_research_hands_full_originals_to_answer_model(
    depth: int,
) -> None:
    selected, cheap = model(), model()
    context = RunContext(depth=depth)
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    text = "Permission requires both authenticated proof and a later approval."
    record = original(ledger, context, text)
    current = adaptive_tool_view(
        original_evidence=[record],
        research_state={
            "needs": [
                {
                    "need_id": "later_stage",
                    "status": "open",
                    "gap": "The later stage remains unresolved.",
                }
            ]
        },
    )
    selected.invoke.return_value = ModelResponse(
        id="full-answer",
        created="0",
        choice=Choice(
            message=Message(
                content="Permission requires authenticated proof and later approval [1].",
            )
        ),
    )
    cheap.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "7"}),
        ModelResponse(
            id="candidate",
            created="0",
            choice=Choice(
                message=Message(
                    content="Permission requires authenticated proof and later approval [1].",
                )
            ),
        ),
    ]
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    if depth == 0:
        selected.invoke.return_value = native_action(
            "read_provision", {"source_id": "law", "article": "7"}
        )
        assert adapter.decide(current).calls[0].name == "read_provision"
        selected.invoke.return_value = ModelResponse(
            id="full-answer",
            created="0",
            choice=Choice(message=Message(content="Complete rule [1].")),
        )
    assert adapter.decide(current).calls[0].name == "read_provision"
    decision = adapter.decide(current)
    assert decision.answer and "[1]" in decision.answer
    assert cheap.invoke.call_count == 2
    assert selected.invoke.call_count == (2 if depth == 0 else 1)
    payload = last_payload(selected)
    assert (
        payload["draft_to_repair"]
        == "Permission requires authenticated proof and later approval [1]."
    )
    assert payload["original_evidence"][0]["text"] == text
    assert (
        payload["research_gap_signals"] == last_payload(cheap)["research_gap_signals"]
    )
    assert payload["research_gap_signals"][0]["need_id"] == "later_stage"
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}


@pytest.mark.parametrize("profile,depth", [("normal", 0), ("deep", 0), ("deep", 1)])
def test_native_gap_leads_reuse_complete_originals_without_an_extra_model_call(
    profile: str, depth: int
) -> None:
    context, ledger, llm = RunContext(depth=depth), EvidenceLedger(), model()
    context.services.update(evidence=ledger, research_profile=profile)
    record = original(
        ledger, context, "Both cumulative conditions and their exception."
    )
    receipt = ToolReceipt(
        call=CapabilityCall(name="search_corpus", call_id="source-lookup"),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Two centers were not mapped",
            data={
                "unhydrated_centers": [
                    {"source_id": "law", "canonical_chunk_id": record["chunk_id"]},
                    {"source_id": "law", "canonical_chunk_id": "unread"},
                ]
            },
        ),
        elapsed_seconds=1,
    )
    current = adaptive_tool_view(original_evidence=[record]).model_copy(
        update={"receipts": [receipt]}
    )
    before = current.model_dump(mode="json")
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(current)
    assert llm.invoke.call_count == 1
    assert llm.invoke.call_args.kwargs["tools"] == current.tools
    payload = last_payload(llm)
    assert payload["research_gap_signals"] == [
        {
            "kind": "retrieved_original_not_delivered",
            "source_id": "law",
            "canonical_chunk_id": "unread",
            "call_id": "source-lookup",
        }
    ]
    assert payload["original_evidence"] == [record]
    assert current.model_dump(mode="json") == before


@pytest.mark.parametrize("incomplete", ["partial_range", "different_hash"])
def test_incomplete_or_unverified_text_cannot_close_a_source_delivery_gap(
    incomplete: str,
) -> None:
    context, ledger, llm = RunContext(), EvidenceLedger(), model()
    context.services["evidence"] = ledger
    record = original(
        ledger, context, "A complete original with its restrictive conditions."
    )
    receipt = ToolReceipt(
        call=CapabilityCall(name="search_corpus", call_id="lookup"),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Original missing",
            data={
                "unhydrated_centers": [
                    {"source_id": "law", "canonical_chunk_id": record["chunk_id"]}
                ]
            },
        ),
        elapsed_seconds=1,
    )
    if incomplete == "partial_range":
        record["text"] = str(record["text"])[:10]
    else:
        record["text_hash"] = "0" * 64
    current = view(original_evidence=[record]).model_copy(
        update={"receipts": [receipt]}
    )
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    assert (
        payload["research_gap_signals"][0]["canonical_chunk_id"] == record["chunk_id"]
    )
    llm.invoke.assert_not_called()


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("ask_user", {"question": "İşlemin tarihi nedir?"}),
        ("submit_answer", {"answer": "Merhaba!", "basis": "conversation"}),
        ("submit_answer", {"answer": "2 + 2 = 4", "basis": "scenario"}),
    ],
)
def test_intermediate_model_can_clarify_or_answer_conversation_without_handoff(
    name: str,
    arguments: dict[str, Any],
) -> None:
    selected, cheap = model(), model()
    cheap.invoke.return_value = native_action(name, arguments)
    adapter = ResearchModel(
        selected, RunContext(depth=1), research_llm=cheap, lean_native_mode=True
    )
    assert adapter.decide(adaptive_tool_view()).calls[0].name == name
    assert cheap.invoke.call_count == 1
    selected.invoke.assert_not_called()


def test_intermediate_legal_submission_runs_selected_answer_model() -> None:
    selected, cheap = model(), model()
    cheap.invoke.return_value = native_action(
        "submit_answer", {"answer": "Rule [1].", "basis": "originals"}
    )
    selected.invoke.return_value = native_action(
        "submit_answer", {"answer": "Rule and exception [1].", "basis": "originals"}
    )
    adapter = ResearchModel(
        selected, RunContext(depth=1), research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(adaptive_tool_view())
    assert decision.calls[0].arguments["answer"] == "Rule and exception [1]."
    assert last_payload(selected)["draft_to_repair"] == "Rule [1]."
    assert cheap.invoke.call_count == selected.invoke.call_count == 1


@pytest.mark.parametrize("profile,depth", [("normal", 0), ("deep", 0), ("deep", 1)])
def test_existing_decision_receives_read_siblings_once_without_a_new_source_call(
    profile: str,
    depth: int,
) -> None:
    ledger = recorded(
        [
            provision_original("permission", "An application may be approved."),
            provision_original(
                "proof",
                "Approval requires an authenticated certificate and an authority's consent.",
                headings=["Statute", "MADDE 17", "(2) Conditions"],
            ),
            provision_original(
                "unrelated", "Another instrument's rule.", source="another-source"
            ),
            provision_original(
                "old", "An older rule.", metadata={"read_as_of_date": "2025-01-01"}
            ),
        ]
    )
    context = RunContext(
        depth=depth, services={"evidence": ledger, "research_profile": profile}
    )
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    decision = adapter.decide(current)
    assert decision.answer == "Rule [1]." and decision.calls == []
    assert llm.invoke.call_count == 1
    originals = last_payload(llm)["original_evidence"]
    assert [record["citation"] for record in originals] == [1, 2]
    assert originals[1]["text"] == full_record(ledger, 2)["text"]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}


def test_root_review_uses_child_citations_without_changing_child_body_or_using_lite() -> (
    None
):
    ledger = recorded(
        [
            provision_original("permission", "The transaction is permitted."),
            provision_original(
                "exception", "The permission excludes the specified category."
            ),
        ]
    )
    child_answer = "Detailed result [1].\n\nThe supported procedure remains intact [1]."
    answers = [{"question_id": "q0", "answer": child_answer, "evidence_numbers": [1]}]
    context = RunContext(
        services={
            "evidence": ledger,
            "independent_question_mode": True,
            "question_research_started": True,
            "independent_answers": answers,
        }
    )
    selected, cheap = model(), model()
    selected.invoke.return_value = native_action("assemble_answers", {})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(independent_tool_view())
    assert [call.name for call in decision.calls] == ["assemble_answers"]
    assert selected.invoke.call_count == 1 and cheap.invoke.call_count == 0
    payload = last_payload(selected)
    assert [record["citation"] for record in payload["original_evidence"]] == [1, 2]
    assert payload["independent_answers"][0]["answer"] == child_answer
    assert context.services["independent_answers"] == answers


def test_assigned_outcome_carries_a_cross_question_condition_and_its_read_sibling() -> (
    None
):
    ledger = recorded(
        [
            provision_original("permission", "The transaction needs a permission."),
            provision_original(
                "condition",
                "Settlement needs an authority's consent.",
                source="procedure",
                headings=["Procedure", "MADDE 23"],
            ),
            provision_original(
                "exception",
                "Consent may be withheld for the specified exception.",
                source="procedure",
                headings=["Procedure", "MADDE 23", "(2) Exception"],
            ),
            provision_original(
                "other", "An unrelated question's requirement.", source="other"
            ),
        ]
    )
    context = RunContext(
        depth=1, services={"evidence": ledger, "task_outcome_ids": ["settlement"]}
    )
    outcomes = OutcomeMap(["Permission?", "Settlement?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "permission",
                        "question_ids": ["q0"],
                        "detail": "Permission",
                    },
                    {
                        "outcome_id": "settlement",
                        "question_ids": ["q1"],
                        "detail": "Settlement",
                    },
                    {
                        "outcome_id": "other",
                        "question_ids": ["q0"],
                        "detail": "Another independent issue",
                    },
                ],
                "conditions": [
                    {
                        "condition_id": "shared-consent",
                        "outcome_ids": ["permission", "settlement"],
                        "detail": "Authority's consent",
                        "witnesses": [{"citation": 2, "end_char": 39}],
                    },
                    {
                        "condition_id": "other-condition",
                        "outcome_ids": ["other"],
                        "detail": "Separate condition",
                        "witnesses": [
                            {
                                "citation": 4,
                                "end_char": len("An unrelated question's requirement."),
                            }
                        ],
                    },
                ],
            }
        ),
        ledger,
    )
    context.services["outcome_map"] = outcomes
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(adaptive_tool_view(original_evidence=[full_record(ledger, 1)]))
    payload = last_payload(llm)
    assert [record["citation"] for record in payload["original_evidence"]] == [1, 2, 3]
    assert payload["outcome_map"]["undelivered_evidence_numbers"] == []
    assert [item["outcome_id"] for item in payload["outcome_map"]["outcomes"]] == [
        "settlement"
    ]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2, 3}
    assert llm.invoke.call_count == 1


def test_native_bundle_preserves_selected_partial_range_and_non_citable_original() -> (
    None
):
    ledger = recorded(
        [
            provision_original("rule", "Start. Selected clause. Original tail."),
            provision_original(
                "exception", "A material exception in another read original."
            ),
            provision_original(
                "nonlegal",
                "A user's self-contained calculation.",
                source="nonlegal",
                citable=False,
            ),
        ]
    )
    own = full_record(ledger, 1)
    own.update(
        {"text": "Selected clause.", "start_char": 7, "end_char": 23, "truncated": True}
    )
    context = RunContext(services={"evidence": ledger})
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(adaptive_tool_view(original_evidence=[own, full_record(ledger, 3)]))
    originals = last_payload(llm)["original_evidence"]
    assert originals[0] == own
    assert [record["citation"] for record in originals] == [1, 3, 2]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {2, 3}
    assert llm.invoke.call_count == 1


def test_required_related_original_cannot_be_silently_evicted_for_physical_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = recorded(
        [
            provision_original("permission", "A permitted transaction needs proof."),
            provision_original("proof", "A signed certificate is required. " * 1000),
        ]
    )
    context = RunContext(services={"evidence": ledger})
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1)], required_evidence_numbers=[1]
    )
    prompt, tools, output = adapter._fit_native_decision(current)
    full_cost = adapter._input_cost(prompt, tools)
    required_original_cost = len(json.dumps(full_record(ledger, 2)))

    def physical_limits(max_tokens: int) -> tuple[int, int]:
        del max_tokens
        return full_cost - required_original_cost, output

    monkeypatch.setattr(adapter, "_limits", physical_limits)
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(current)
    assert llm.invoke.call_count == 0
