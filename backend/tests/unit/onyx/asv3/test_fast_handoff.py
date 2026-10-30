"""Fast handoff changes writing order, never original delivery or publication approval."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.fast_policy import (
    FAST_HANDOFF,
    fast_research_decision_gap,
    fast_research_phase,
    finish_research_spec,
    restore_fast_handoff,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchState
from tests.unit.onyx.asv3.test_native_model_adapter import (
    last_payload,
    model,
    native_response,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded


def context(ledger: EvidenceLedger, **services: object) -> RunContext:
    captured = RunContext(budget=SharedBudget(unlimited_execution=True))
    captured.services.update(asv3_fast=True, research_profile="normal", evidence=ledger)
    captured.services.update(services)
    captured.services["research_state"] = ResearchState(
        ["Rule and exception?"], captured
    )
    return captured


def notes(number: int = 1) -> dict[str, JsonValue]:
    return {
        "findings": [
            {
                "question_ids": ["q0"],
                "statement": "Both conditions and the later exception need application.",
                "evidence_numbers": [number],
            }
        ],
        "open_needs": ["The applicable date is not supplied."],
    }


def registry(ledger: EvidenceLedger, owner: RunContext) -> CapabilityRegistry:
    result = CapabilityRegistry([finish_research_spec(ledger, owner)])
    for name in ("submit_answer", "submit_partial_answer"):
        result.register(
            ToolSpec(
                name=name,
                description="Publish through the ordinary guard.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="The publication guard still has a source gap",
                ),
            )
        )
    return result


def test_handoff_is_notes_only_and_originals_are_delivered_to_selected_writer() -> None:
    text = "Rule applies only if A AND B; exception C and later procedure D. " * 100
    ledger = recorded([original("clause", text)])
    captured = context(ledger)
    tools = registry(ledger, captured)
    captured.services["registry"] = tools
    current = HarnessView(
        request="Rule and exception?",
        questions=["Rule and exception?"],
        facts=[],
        receipts=[],
        evidence=[],
        original_evidence=[full_record(ledger, 1)],
        tools=tools.definitions(captured),
    )
    selected, helper = model(), model()
    helper.invoke.return_value = native_response("finish_research", json.dumps(notes()))
    selected.invoke.return_value = native_response(
        "submit_answer", json.dumps({"answer": "A and B, subject to C and D [1]."})
    )
    adapter = ResearchModel(
        selected, captured, research_llm=helper, lean_native_mode=True
    )
    adapter._native_first_decision_completed = True
    snapshot = copy.deepcopy(current)
    research = adapter.decide(current)
    assert [call.name for call in research.calls] == ["finish_research"]
    assert selected.invoke.call_count == 0
    research_tools = helper.invoke.call_args.kwargs["tools"]
    assert {tool["function"]["name"] for tool in research_tools} == {"finish_research"}
    result = tools.dispatch(research.calls[0], captured)
    assert result.status is OutcomeStatus.FOUND
    assert result.data["legal_approval"] is False
    assert "submitted_answer" not in captured.services
    assert FAST_HANDOFF in captured.services

    written = adapter.decide(current)
    assert helper.invoke.call_count == selected.invoke.call_count == 1
    assert [call.name for call in written.calls] == ["submit_answer"]
    writer_tools = selected.invoke.call_args.kwargs["tools"]
    assert {tool["function"]["name"] for tool in writer_tools} == {
        "submit_answer",
        "submit_partial_answer",
    }
    assert isinstance(adapter.last_call_id, str)
    assert ledger.completely_delivered(adapter.last_call_id) == {1}
    item = ledger.get(1)
    assert item is not None
    assert item.text == text
    assert current == snapshot
    prompt = selected.invoke.call_args.kwargs["prompt"]
    assert text in json.dumps([message.model_dump() for message in prompt])
    assert tools.dispatch(written.calls[0], captured).status is OutcomeStatus.PARTIAL
    assert "submitted_answer" not in captured.services


@pytest.mark.parametrize("fast,depth", [(False, 0), (True, 1)])
def test_legacy_and_worker_cannot_use_handoff(fast: bool, depth: int) -> None:
    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger, asv3_fast=fast)
    captured.depth = depth
    result = finish_research_spec(ledger, captured).handler(notes(), captured)
    assert result.status is OutcomeStatus.DENIED
    assert FAST_HANDOFF not in captured.services


@pytest.mark.parametrize("change", ["citation", "question", "extra", "bool"])
def test_invalid_handoff_cannot_change_phase(change: str) -> None:
    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger)
    payload = notes()
    findings = cast(list[dict[str, JsonValue]], payload["findings"])
    if change == "citation":
        findings[0]["evidence_numbers"] = [42]
    elif change == "question":
        findings[0]["question_ids"] = ["q7"]
    elif change == "bool":
        findings[0]["evidence_numbers"] = [True]
    else:
        payload["approved"] = True
    outcome = finish_research_spec(ledger, captured).handler(payload, captured)
    assert outcome.status is OutcomeStatus.INVALID
    assert fast_research_phase(captured)


def test_resumed_handoff_retains_same_evidence_and_unresolved_effects() -> None:
    ledger = recorded([original("c", "Original condition and exception.")])
    saved: dict[str, JsonValue] = {FAST_HANDOFF: notes()}
    captured = context(ledger)
    restore_fast_handoff(saved, ledger, captured)
    assert captured.services[FAST_HANDOFF] == notes()
    assert not fast_research_phase(captured)
    assert "submitted_answer" not in captured.services
    with pytest.raises(ValueError, match="recorded original"):
        restore_fast_handoff(saved, EvidenceLedger(), context(EvidenceLedger()))


def test_handoff_is_not_replayed_or_copied_into_legacy_context() -> None:
    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger)
    spec = finish_research_spec(ledger, captured)
    assert spec.handler(notes(), captured).status is OutcomeStatus.FOUND
    assert spec.handler(notes(), captured).status is OutcomeStatus.INVALID
    legacy = context(ledger, asv3_fast=False)
    restore_fast_handoff({FAST_HANDOFF: notes()}, ledger, legacy)
    assert FAST_HANDOFF not in legacy.services


def test_cropped_handoff_uses_existing_full_original_writer_without_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger, asv3_exp_cropped=True)
    captured.services[FAST_HANDOFF] = notes()
    selected = model()
    adapter = ResearchModel(selected, captured, lean_native_mode=True)
    expected = Decision()
    existing_writer = Mock(return_value=expected)
    monkeypatch.setattr(adapter, "_decide_cropped_source_answer", existing_writer)
    current = HarnessView(
        request="Rule?",
        questions=["Rule?"],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
    )
    assert adapter.decide(current) is expected
    existing_writer.assert_called_once_with(current)
    assert selected.invoke.call_count == 0


@pytest.mark.parametrize(
    "decision",
    [
        Decision(answer="Premature long answer."),
        Decision(
            calls=[
                CapabilityCall(name="finish_research"),
                CapabilityCall(name="read_provision"),
            ]
        ),
        Decision(
            calls=[
                CapabilityCall(
                    name="compose_tool_calls",
                    arguments={
                        "steps": [
                            {"id": "finish", "tool": "finish_research", "arguments": {}}
                        ]
                    },
                )
            ]
        ),
        Decision(
            calls=[
                CapabilityCall(
                    name="compose_tool_calls",
                    arguments={
                        "steps": [
                            {"id": "publish", "tool": "submit_answer", "arguments": {}}
                        ]
                    },
                )
            ]
        ),
    ],
)
def test_research_phase_rejects_prose_and_nested_or_batched_handoff(
    decision: Decision,
) -> None:
    assert fast_research_decision_gap(decision) is not None


def test_plain_prose_is_recovered_without_becoming_an_answer_candidate() -> None:
    from onyx.llm.model_response import Choice, Message, ModelResponse

    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger)
    tools = registry(ledger, captured)
    selected = model()
    selected.invoke.side_effect = [
        ModelResponse(
            id="prose",
            created="0",
            choice=Choice(message=Message(content="Premature draft.")),
        ),
        native_response("finish_research", json.dumps(notes())),
    ]
    adapter = ResearchModel(selected, captured, lean_native_mode=True)
    current = HarnessView(
        request="Rule?",
        questions=["Rule?"],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools.definitions(captured),
    )
    result = adapter.decide(current)
    assert selected.invoke.call_count == 2
    assert result.answer is None
    assert [call.name for call in result.calls] == ["finish_research"]
    assert FAST_HANDOFF not in captured.services


def test_actual_harness_child_dispatch_commits_handoff_to_root_owner() -> None:
    from onyx.asv3.harness import Harness

    ledger = recorded([original("c", "Original condition.")])
    captured = context(ledger)
    tools = registry(ledger, captured)
    harness = Harness(
        request="Rule?",
        context=captured,
        registry=tools,
        evidence=ledger,
        decide=lambda _view: Decision(),
    )
    results = harness._dispatch(
        [CapabilityCall(name="finish_research", arguments=notes())]
    )
    assert results[0].outcome.status is OutcomeStatus.FOUND
    assert captured.services[FAST_HANDOFF] == notes()
    assert not fast_research_phase(captured)
    assert "submitted_answer" not in captured.services


@pytest.mark.parametrize("named_original_fits", [True, False])
def test_handoff_cited_original_is_required_in_a_tighter_main_context(
    monkeypatch: pytest.MonkeyPatch, named_original_fits: bool
) -> None:
    from onyx.asv3.models import RunStopped
    from onyx.llm.models import UserMessage

    named_text = (
        "Operative condition A AND B, exception C, and later procedure D. " * 250
    )
    other_text = "A different provision on a separately researched issue. " * 250
    ledger = recorded(
        [
            original("governing", named_text, source="named-statute"),
            original("other", other_text, source="other-statute"),
        ]
    )
    captured = context(ledger)
    captured.services[FAST_HANDOFF] = notes(1)
    tools = registry(ledger, captured)
    selected, helper = model(100_000), model(1_000_000)
    selected.invoke.return_value = native_response(
        "submit_answer", json.dumps({"answer": "A and B, subject to C and D [1]."})
    )
    adapter = ResearchModel(
        selected,
        captured,
        research_llm=helper,
        lean_native_mode=True,
        token_counter=len,
    )
    current = HarnessView(
        request="Rule and exception?",
        questions=["Rule and exception?"],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools.definitions(captured),
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
    )
    complete, definitions, output = adapter._fit_native_decision(current)
    payload = json.loads(cast(str, complete[-1].content))
    payload["original_evidence"] = [full_record(ledger, 1)]
    payload["original_evidence_omitted"] = [
        {
            "citation": 2,
            "source_id": "other-statute",
            "start_char": 0,
            "end_char": len(other_text),
            "reason": "physical_model_context",
        }
    ]
    minimal = [
        *complete[:-1],
        UserMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    named_cost = adapter._input_cost(minimal, definitions)
    ceiling = (
        named_cost + 20 if named_original_fits else named_cost - len(named_text) - 1024
    )
    assert 0 < ceiling < adapter._input_cost(complete, definitions)
    monkeypatch.setattr(adapter, "_limits", lambda _requested: (ceiling, output))

    if not named_original_fits:
        with pytest.raises(RunStopped, match="required originals"):
            adapter.decide(current)
        selected.invoke.assert_not_called()
    else:
        result = adapter.decide(current)
        assert [call.name for call in result.calls] == ["submit_answer"]
        assert isinstance(adapter.last_call_id, str)
        assert ledger.completely_delivered(adapter.last_call_id) == {1}
        actual = last_payload(selected)
        assert actual["original_evidence"] == [full_record(ledger, 1)]
        assert actual["original_evidence_omitted"][0]["citation"] == 2
    helper.invoke.assert_not_called()
    retained = ledger.get(1)
    assert retained is not None and retained.text == named_text
    assert captured.services[FAST_HANDOFF] == notes(1)
