"""Cropped checks isolate receipts; integration preserves the owned original draft."""

import json
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.cropped_cost_budget import CroppedCostBudget
from onyx.asv3.cropped_repair_model import fresh_model, integrate_repair
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import HarnessView, RunContext, RunStopped, SharedBudget
from onyx.asv3.performance import (
    FAST_TIMING_SERVICE,
    fast_timing_step,
    install_fast_timing,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import ChatCompletionMessage, TextContentPart, ToolMessage
from onyx.prompts.asv3_cropped_repair import CROPPED_REPAIR_INTEGRATION
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cheap_repair_packet import definitions
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response, turn
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def fixture(
    metadata: dict[str, JsonValue] | None = None,
) -> tuple[ResearchModel, EvidenceLedger, MagicMock]:
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                "rule",
                "Rule applies only with the signed certificate.",
                metadata=metadata,
            ),
            original("exception", "Exception requires authority consent."),
            original("later", "After approval the declaration must be submitted."),
        ],
        context,
    )
    context.services.update(
        evidence=ledger,
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        asv3_exp_cropped=True,
        assistant_instructions="Preserve scenario facts.",
    )
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter._native_output_capacity = 8192
    return adapter, ledger, selected


def current(ledger: EvidenceLedger) -> HarnessView:
    passage = full_record(ledger, 1)
    passage.update(text="Rule applies", end_char=len("Rule applies"), truncated=True)
    return HarnessView(
        request="Explain the rule, exception and later step.",
        questions=["Explain the rule, exception and later step."],
        facts=["A certificate has not yet been signed."],
        receipts=[],
        evidence=[],
        tools=definitions(),
        turns=[turn("old-original-writer", [full_record(ledger, 1)])],
        original_evidence=[passage, full_record(ledger, 2)],
        required_evidence_numbers=[3],
        draft_to_repair="The rule always applies [1].\n\nA later declaration is due [3].",
        publication_gap={"missing_condition": True},
        research_state={"certificate_signed": False},
    )


def checks() -> list[dict[str, JsonValue]]:
    return [
        {
            "key": "rule-condition",
            "status": "found",
            "evidence_numbers": [1, 2],
            "detail": "Both certificate and consent conditions are present.",
            "absence_proven": False,
        }
    ]


def payload(prompt: list[ChatCompletionMessage]) -> dict[str, JsonValue]:
    content = prompt[-1].content
    assert isinstance(content, list) and isinstance(content[0], TextContentPart)
    return cast(dict[str, JsonValue], json.loads(content[0].text))


def test_fresh_same_model_shares_fences_but_isolates_writer_and_callbacks() -> None:
    adapter, ledger, _selected = fixture()
    adapter.history = "Prior conversation."
    adapter.last_call_id = "original-writer"
    adapter.context.services["last_model_call_id"] = "original-writer"
    adapter.context.services["search_history"] = ["prior search"]
    budget = CroppedCostBudget(adapter.context, "Actual question")
    controller = CroppedResearchLimit(adapter.context, "Actual question")
    adapter.context.services["cropped_cost_budget"] = budget
    adapter.context.services["cropped_research_controller"] = controller
    fresh = fresh_model(adapter)
    assert fresh.llm is adapter.llm
    assert fresh.user_identity is adapter.user_identity
    assert fresh.reasoning_effort is adapter.reasoning_effort
    assert fresh.token_counter is adapter.token_counter
    assert not fresh.history and fresh.updates() == [] and fresh.pending_tasks() == []
    assert fresh.last_call_id is None
    assert fresh.context.services is not adapter.context.services
    assert fresh.context.services["evidence"] is ledger
    assert fresh.context.services["cropped_cost_budget"] is budget
    assert fresh.context.services["cropped_research_controller"] is controller
    assert "last_model_call_id" not in fresh.context.services
    assert "search_history" not in fresh.context.services
    assert "asv3_exp_cheap" not in fresh.context.services
    assert fresh.context.services["provider_max_attempts"] == 1
    assert fresh.context.services["provider_compatibility_attempts"] == 1
    assert fresh.context.budget is adapter.context.budget
    assert fresh.context.scope is adapter.context.scope
    assert fresh.context.run_id == adapter.context.run_id
    assert fresh.context.deadline == adapter.context.deadline
    assert fresh.context.research_deadline == adapter.context.research_deadline
    assert fresh._native_output_capacity == 8192
    adapter.context.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        fresh.context.check_active()


def test_fresh_requires_cropped_and_rejects_cheap() -> None:
    adapter, _ledger, _selected = fixture()
    adapter.context.services.pop("asv3_exp_cropped")
    with pytest.raises(ValueError, match="Cropped"):
        fresh_model(adapter)
    adapter.context.services.update(asv3_exp_cropped=True, asv3_exp_cheap=True)
    with pytest.raises(ValueError, match="Cropped"):
        fresh_model(adapter)


def test_fresh_shares_timing_and_response_policy_without_fast_flags() -> None:
    adapter, _ledger, _selected = fixture()
    adapter.context.services.update(
        asv3_fast=True,
        asv3_fast_retrieval=True,
        asv3_fast_response_policy="material-outcomes-v1",
    )
    collector = install_fast_timing(adapter.context)
    assert collector is not None
    fresh = fresh_model(adapter)
    assert fresh.context.services[FAST_TIMING_SERVICE] is collector
    assert "asv3_fast" not in fresh.context.services
    assert "asv3_fast_retrieval" not in fresh.context.services
    assert fresh.context.services["asv3_fast_response_policy"] == "material-outcomes-v1"
    with fast_timing_step(fresh.context, "asv3.model_request") as step:
        assert step is not None
    assert collector._totals["asv3.model_request"].count == 1


class CheckResult(BaseModel):
    status: Literal["not_found"]


def test_checker_isolated_actual_receipt_does_not_replace_parent_writer() -> None:
    adapter, ledger, selected = fixture()
    adapter.last_call_id = "owned-writer"
    adapter.context.services["last_model_call_id"] = "owned-writer"
    fresh = fresh_model(adapter)
    selected.invoke.return_value.choice.message.content = '{"status":"not_found"}'
    fresh.invoke_text(
        "Locate only the supplied clause.",
        json.dumps({"evidence": [full_record(ledger, 2)]}),
        LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK,
        response_model_override=CheckResult,
    )
    assert selected.invoke.call_count == 1
    assert adapter.last_call_id == "owned-writer"
    assert adapter.context.services["last_model_call_id"] == "owned-writer"
    assert fresh.last_call_id is not None
    assert ledger.completely_delivered(fresh.last_call_id) == {2}


def test_invalid_checker_json_cannot_trigger_a_format_retry() -> None:
    adapter, _ledger, selected = fixture()
    fresh = fresh_model(adapter)
    with pytest.raises(RunStopped, match="one model call"):
        fresh.invoke_text(
            "Locate only the supplied clause.",
            "{}",
            LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK,
            response_model_override=CheckResult,
        )
    assert selected.invoke.call_count == 1
    assert adapter.last_call_id is None


def test_exact_retained_edit_delivers_full_union_once_and_updates_writer() -> None:
    adapter, ledger, selected = fixture()
    old = current(ledger)
    saved = old.model_dump(mode="json")
    replacement = (
        "The rule requires a signed certificate [1] and authority consent [2]."
    )

    def response(**kwargs: object) -> ModelResponse:
        data = payload(cast(list[ChatCompletionMessage], kwargs["prompt"]))
        retained = data["draft_to_repair"]
        assert isinstance(retained, dict) and isinstance(retained["units"], list)
        unit = retained["units"][0]
        assert isinstance(unit, dict)
        return native_response(
            "submit_retained_answer",
            json.dumps(
                {
                    "retained_answer_edits": [
                        {"unit_id": unit["unit_id"], "replacement": replacement}
                    ]
                }
            ),
        )

    selected.invoke.side_effect = response
    result = integrate_repair(adapter, old, checks())
    assert selected.invoke.call_count == 1
    assert result.calls[0].name == "submit_answer"
    assert (
        result.calls[0].arguments["answer"]
        == replacement + "\n\nA later declaration is due [3]."
    )
    assert old.model_dump(mode="json") == saved
    kwargs = selected.invoke.call_args.kwargs
    prompt = kwargs["prompt"]
    assert not any(isinstance(message, ToolMessage) for message in prompt)
    data = payload(prompt)
    originals = data["original_evidence"]
    assert isinstance(originals, list) and len(originals) == 3
    assert {row["citation"] for row in originals if isinstance(row, dict)} == {1, 2, 3}
    expected_texts = set()
    for number in (1, 2, 3):
        item = ledger.get(number)
        assert item is not None
        expected_texts.add(item.text)
    assert {row["text"] for row in originals if isinstance(row, dict)} == expected_texts
    assert data["recorded_facts"] == old.facts
    assert data["research_state"] == old.research_state
    gap = data["publication_gap"]
    assert isinstance(gap, dict) and gap["cropped_evidence_checks"] == checks()
    assert gap["absence_proven"] is False
    question_content = prompt[1].content
    assert isinstance(question_content, str)
    question = cast(
        dict[str, JsonValue], json.JSONDecoder().raw_decode(question_content)[0]
    )
    instructions = question["assistant_instructions"]
    assert isinstance(instructions, str)
    assert CROPPED_REPAIR_INTEGRATION in instructions
    assert {tool["function"]["name"] for tool in kwargs["tools"]} == {
        "submit_retained_answer",
        "submit_retained_partial_answer",
    }
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2, 3}
    assert (
        ledger.delivery_flow(adapter.last_call_id) == LLMFlow.ASV3_ANSWER_REPAIR.value
    )
    assert adapter.context.services["last_model_call_id"] == adapter.last_call_id


@pytest.mark.parametrize(
    "field,value", [("text", "Invented text"), ("text_hash", "changed")]
)
def test_changed_original_fails_before_provider(field: str, value: str) -> None:
    adapter, ledger, selected = fixture()
    old = current(ledger)
    old.original_evidence[0][field] = value
    with pytest.raises(RunStopped, match="identity or text"):
        integrate_repair(adapter, old, checks())
    selected.invoke.assert_not_called()


@pytest.mark.parametrize(
    "metadata", [{"truncated": True}, {"canonical_metadata": {"truncated": True}}]
)
def test_partial_ledger_original_is_not_invented_complete(
    metadata: dict[str, JsonValue],
) -> None:
    adapter, ledger, selected = fixture(metadata)
    with pytest.raises(RunStopped, match="incomplete canonical"):
        integrate_repair(adapter, current(ledger), checks())
    selected.invoke.assert_not_called()


def test_nonterminal_native_action_cannot_trigger_another_generation() -> None:
    adapter, ledger, selected = fixture()
    selected.invoke.return_value = native_response("read_provision", "{}")
    with pytest.raises(RunStopped, match="one model call"):
        integrate_repair(adapter, current(ledger), checks())
    assert selected.invoke.call_count == 1


def test_incomplete_check_preserves_prior_user_facts_without_old_tool_history() -> None:
    adapter, ledger, selected = fixture()
    scenario = (
        "USER 1: The certificate has not been signed.\n"
        "USER 2: Authority consent has also not been obtained."
    )
    adapter.context.services["cropped_user_context"] = scenario
    adapter.history = "Old assistant conclusion and tool results must not be replayed."
    incomplete = checks()
    incomplete[0].update(
        status="incomplete", detail="Consent evidence remains incomplete."
    )
    selected.invoke.return_value = native_response(
        "submit_retained_partial_answer", "{}"
    )
    result = integrate_repair(adapter, current(ledger), incomplete)
    assert result.calls[0].name == "submit_partial_answer"
    prompt = selected.invoke.call_args.kwargs["prompt"]
    assert not any(isinstance(message, ToolMessage) for message in prompt)
    gap = payload(prompt)["publication_gap"]
    assert isinstance(gap, dict)
    assert gap["user_supplied_scenario_context"] == scenario
    assert gap["cropped_evidence_checks"] == incomplete
    assert gap["absence_proven"] is False
    assert adapter.history not in json.dumps(
        [message.model_dump() for message in prompt]
    )
    assert selected.invoke.call_count == 1


def test_no_actual_original_receipt_rejects_even_a_valid_retained_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected = fixture()
    selected.invoke.return_value = native_response("submit_retained_answer", "{}")
    monkeypatch.setattr(ledger, "completely_delivered", lambda _call_id: set())
    with pytest.raises(RunStopped, match="deliver every required"):
        integrate_repair(adapter, current(ledger), checks())
    assert selected.invoke.call_count == 1
