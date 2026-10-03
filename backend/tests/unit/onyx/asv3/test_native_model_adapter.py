import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage


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
                    {"original_evidence": records, "navigation": extra},
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


def test_native_history_preserves_each_complete_batch_and_delivers_originals_once() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    first = original(ledger, context, "The applicant must request the relief.")
    second = original(
        ledger, context, "Release requires completion of institution checks."
    )
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    batch1, batch2 = turn("read-law", [first]), turn("read-condition", [second])
    current = view(
        turns=[batch1, batch2],
        original_evidence=[first, second],
        research_state={"needs": [{"description": "UNNEEDED_BOARD" * 10000}]},
    )
    adapter.decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    tool_results = [message for message in prompt if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_results] == [
        "read-law",
        "read-condition",
    ]
    assert "original_evidence" not in last_payload(llm)
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
    assert restored == [first]
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
    llm = model()
    adapter = ResearchModel(llm, context.child())
    assert adapter.lean_native_mode
    adapter.decide(view())
    assert (
        llm.invoke.call_args.kwargs["prompt"][0].content
        == "Research the assigned original."
    )
    assert llm.invoke.call_args.kwargs["structured_response_format"] is None
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
