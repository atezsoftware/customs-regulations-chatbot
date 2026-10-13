"""Fresh calls must retain original delivery and the ordinary publication boundary."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter, runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    RunContext,
    RunStopped,
    SharedBudget,
)
from onyx.asv3.workflow_variant import ASV3_EXP_CHEAP_VARIANT, ASV3_TUNED_VARIANT
from onyx.llm.models import ToolMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_native_model_adapter import model, turn, view
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original
from tests.unit.onyx.llm.test_asv3_exp_cheap_cache import complete, request
from tests.unit.onyx.llm.test_asv3_exp_cheap_cache import model as native_sol_model


def adapter_fixture(
    cheap: bool = True,
) -> tuple[ResearchModel, EvidenceLedger, MagicMock]:
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    ledger = EvidenceLedger()
    ledger.add(
        [
            original("rule", "Rule."),
            original("condition", "Uncited condition AND exception."),
        ],
        context,
    )
    context.services.update(
        evidence=ledger,
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        asv3_exp_cheap=cheap,
    )
    selected = model()
    return ResearchModel(selected, context, lean_native_mode=True), ledger, selected


def repair_view(ledger: EvidenceLedger) -> HarnessView:
    return view(
        original_evidence=[full_record(ledger, 1)],
        turns=[turn("read-all", [full_record(ledger, 1), full_record(ledger, 2)])],
        draft_to_repair="Rule [1].",
        publication_gap={"missing_condition": True},
    )


def test_fresh_call_keeps_uncited_original_and_records_new_actual_delivery() -> None:
    adapter, ledger, selected = adapter_fixture()
    old = repair_view(ledger)
    ledger.record_delivery("old-writer", "asv3_coordinator", [full_record(ledger, 1)])
    decision = adapter.decide(old)
    prompt = selected.invoke.call_args.kwargs["prompt"]
    assert len(prompt) == 3
    assert not any(isinstance(message, ToolMessage) for message in prompt)
    assert decision.answer == "Rule [1]."
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert ledger.completely_delivered("old-writer") == {1}
    assert (
        ledger.delivery_flow(adapter.last_call_id or "")
        == LLMFlow.ASV3_ANSWER_REPAIR.value
    )
    assert len(old.turns) == 1
    rows = delivered_originals(selected.invoke.call_args.kwargs)
    assert {row["text"] for row in rows} == {
        "Rule.",
        "Uncited condition AND exception.",
    }


def test_experimental_native_history_and_flow_stay_unchanged() -> None:
    adapter, ledger, selected = adapter_fixture(cheap=False)
    adapter.decide(repair_view(ledger))
    assert any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert (
        ledger.delivery_flow(adapter.last_call_id or "")
        == LLMFlow.ASV3_COORDINATOR.value
    )
    assert "cheap_repair" not in adapter.native_sampling_snapshot()


def test_repeated_gap_broadens_context_and_new_original_reenables_fresh() -> None:
    adapter, ledger, selected = adapter_fixture()
    current = repair_view(ledger)
    for _ in range(3):
        adapter.decide(current)
    prompts = [call.kwargs["prompt"] for call in selected.invoke.call_args_list]
    assert all(
        not any(isinstance(message, ToolMessage) for message in prompt)
        for prompt in prompts[:2]
    )
    assert any(isinstance(message, ToolMessage) for message in prompts[2])
    ledger.add(
        [original("new", "New independently supported prerequisite.")], adapter.context
    )
    adapter.decide(current)
    assert not any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}


def test_repeat_state_roundtrip_does_not_restart_identical_fresh_attempts() -> None:
    adapter, ledger, _selected = adapter_fixture()
    adapter.decide(repair_view(ledger))
    snapshot: dict[str, JsonValue] = {
        "native_coordinator_sampling": adapter.native_sampling_snapshot()
    }
    restored = ResearchModel(model(), adapter.context, lean_native_mode=True)
    restored.restore_native_sampling(snapshot)
    assert (
        restored.native_sampling_snapshot() == snapshot["native_coordinator_sampling"]
    )


def test_larger_fresh_state_falls_back_before_invoking_model() -> None:
    adapter, ledger, selected = adapter_fixture()
    current = repair_view(ledger).model_copy(
        update={"research_state": {"needed_finding": "Kept finding. " * 2000}}
    )
    adapter.decide(current)
    assert selected.invoke.call_count == 1
    assert any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert adapter._cheap_repair_attempts == 0


def test_recorded_native_history_cache_boundary_keeps_ordinary_prompt_and_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, mock_selected = adapter_fixture()
    selected = native_sol_model().with_asv3_exp_cheap_cache()
    policy = selected._asv3_exp_cheap_cache
    assert policy is not None
    complete(policy, policy.prepare_request(request(1)))
    invoke = MagicMock(return_value=mock_selected.invoke.return_value)
    monkeypatch.setattr(selected, "invoke", invoke)
    adapter.llm = selected
    fit = MagicMock(wraps=adapter._fit_native_decision)
    monkeypatch.setattr(adapter, "_fit_native_decision", fit)
    preparation = MagicMock()
    recorded_step = MagicMock()
    recorded_step.return_value.__enter__.return_value = preparation
    monkeypatch.setattr(llm_adapter, "graph_step", recorded_step)

    adapter.decide(repair_view(ledger))

    assert invoke.call_count == 1
    assert fit.call_count == 1
    assert "repair_observations" not in fit.call_args.kwargs
    assert any(
        isinstance(message, ToolMessage)
        for message in invoke.call_args.kwargs["prompt"]
    )
    assert adapter._cheap_repair_attempts == 0
    assert preparation.output_value == {
        "fallback": "recorded_history_cache_boundary",
        "billable_advantage": "unverified",
    }
    assert (
        ledger.delivery_flow(adapter.last_call_id or "")
        == LLMFlow.ASV3_COORDINATOR.value
    )


def test_only_physical_fresh_capacity_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected = adapter_fixture()
    fit = adapter._fit_native_decision

    def limited(view: HarnessView, **options):
        if options.get("repair_observations") is not None:
            raise RunStopped(
                "The complete question, required originals and native tools exceed the selected model context"
            )
        return fit(view, **options)

    monkeypatch.setattr(adapter, "_fit_native_decision", limited)
    adapter.decide(repair_view(ledger))
    assert selected.invoke.call_count == 1
    assert adapter._cheap_repair_attempts == 0

    def cancelled(_view, **_options):
        raise RunStopped("Run cancelled")

    monkeypatch.setattr(adapter, "_fit_native_decision", cancelled)
    with pytest.raises(RunStopped, match="cancelled"):
        adapter.decide(repair_view(ledger))
    assert selected.invoke.call_count == 1


def test_identical_host_acquisition_without_new_evidence_is_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected = adapter_fixture()
    acquisition = Decision(
        calls=[CapabilityCall(name="read_provision", arguments={"article": "1"})]
    )
    monkeypatch.setattr(
        adapter, "_publication_source_acquisition", lambda _view: acquisition
    )
    current = repair_view(ledger)
    assert adapter.decide(current).calls == acquisition.calls
    selected.invoke.assert_not_called()
    assert adapter.decide(current).answer == "Rule [1]."
    assert selected.invoke.call_count == 1


def test_real_runtime_retained_edit_rechecks_full_answer_after_cost_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CHEAP_VARIANT)
    initial = selected.invoke.side_effect
    count = 0
    independent_detail = "Bağımsız destekli işlem ayrıntısı [1]. " * 100
    rejected = "Koşulsuz sonuç [1] ve [2].\n\n" + independent_detail
    repaired = "Koşula bağlı sonuç [1] ve [2].\n\n" + independent_detail

    def invoke(**arguments):
        nonlocal count
        count += 1
        if count == 1:
            return initial(**arguments)
        if count == 2:
            return response(
                calls=[("submit_answer", {"answer": rejected, "basis": "originals"})]
            )
        if count == 3:
            payload = user_payload(arguments["prompt"][-1])
            assert "answer_units" not in payload and "draft_to_repair" not in payload
            assert payload["required_evidence_numbers"] == [1, 2]
            return response(
                text=json.dumps(
                    {
                        "source_scope_id": payload["source_scope_id"],
                        "whole_scope_completed": True,
                        "requirements": [],
                    }
                )
            )
        if count == 5:
            payload = user_payload(arguments["prompt"][-1])
            assert payload["required_evidence_numbers"] == [1, 2]
            assert "answer_units" in payload
            assert len(delivered_originals(arguments)) == 2
            return response(
                text=json.dumps(
                    {
                        "source_scope_id": payload["source_scope_id"],
                        "whole_scope_completed": True,
                        "conditions": [],
                        "resolutions": [],
                    }
                )
            )
        assert count == 4
        assert any(isinstance(message, ToolMessage) for message in arguments["prompt"])
        payload = user_payload(arguments["prompt"][-1])
        assert len(delivered_originals(arguments)) == 2
        unit = payload["draft_to_repair"]["units"][0]
        return response(
            calls=[
                (
                    "submit_retained_answer",
                    {
                        "retained_answer_edits": [
                            {
                                "unit_id": unit["unit_id"],
                                "replacement": "Koşula bağlı sonuç [1] ve [2].",
                            }
                        ],
                    },
                )
            ]
        )

    selected.invoke.side_effect = invoke
    monkeypatch.setattr(
        runtime,
        "cited_lower_statute_gap",
        lambda answer, _ledger, _delivered: (
            {"missing_condition": True} if "Koşulsuz" in answer else None
        ),
    )
    runtime.run_asv3_loop(**kwargs)
    assert selected.invoke.call_count == 5
    assert kwargs["state_container"].answer_tokens == repaired.strip().replace(
        "[1]", "[[1]](https://example.test/law-0)"
    ).replace("[2]", "[[2]](https://example.test/law-1)")
    assert all(
        item["asv3_workflow_variant"] == ASV3_EXP_CHEAP_VARIANT for item in checkpoints
    )
    final = checkpoints[-1]
    assert final["last_draft"] == repaired.strip()
    assert (
        cast(dict, final["native_coordinator_sampling"])["cheap_repair"]["attempts"]
        == 0
    )
