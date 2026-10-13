"""Emergency phase limits reach the actual model and preserve the first writer."""

from unittest.mock import MagicMock

import pytest

from onyx.asv3 import cropped_cost_budget
from onyx.asv3.cropped_cost_budget import CroppedCostBudget, CroppedRepairBudgetReached
from onyx.asv3.cropped_focused_repair import CroppedFocusedRepair
from onyx.asv3.cropped_repair_model import fresh_model
from onyx.asv3.cropped_repair_state import CroppedRepairState
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_task_packets import CroppedTaskPacketStore
from onyx.llm.model_response import Usage
from onyx.llm.models import UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cropped_repair_model import current, fixture
from tests.unit.onyx.asv3.test_native_model_adapter import native_response


def test_ten_dollars_transitions_to_one_writer_with_all_actual_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected = fixture()
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    monkeypatch.setattr(
        cropped_cost_budget, "_rates", lambda *_args: (1e-6, 1e-6, 0.1e-6, 2e-6)
    )
    budget = CroppedCostBudget(adapter.context, view.request)
    budget.record_external_charge(
        "prior-paid-operations",
        known_cost_usd=10.0,
        uncertain_upper_usd=0.0,
        context=adapter.context,
    )
    controller = CroppedResearchLimit(adapter.context, view.request)
    packets = CroppedTaskPacketStore(adapter.context, view.request)
    adapter.context.services.update(
        cropped_cost_budget=budget,
        cropped_research_controller=controller,
        cropped_task_packet_store=packets,
    )
    result = native_response("submit_partial_answer", "{}")
    result.choice.message.content = "The certificate remains necessary [1]."
    result.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.return_value = result
    decision = adapter.decide(view)
    assert decision.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}
    state = budget.export()["state"]
    assert isinstance(state, dict)
    calls = state["calls"]
    assert isinstance(calls, list) and len(calls) == 1
    assert calls[0]["phase"] == "finalization"
    assert not budget.research_allowed and not budget.repair_allowed
    assert packets.export()["input_tokens"] == {"root": 100}


@pytest.mark.parametrize(
    ("repair_limit", "spent"), [(None, 10.0), (None, 10.01), (5.0, 5.0)]
)
def test_repair_limit_prevents_focused_source_work_and_any_repair_model_call(
    monkeypatch: pytest.MonkeyPatch, repair_limit: float | None, spent: float
) -> None:
    adapter, ledger, selected = fixture()
    view = current(ledger)
    monkeypatch.setattr(
        cropped_cost_budget, "_rates", lambda *_args: (1e-6, 1e-6, 0.1e-6, 2e-6)
    )
    budget = (
        CroppedCostBudget(adapter.context, view.request)
        if repair_limit is None
        else CroppedCostBudget(adapter.context, view.request, repair_usd=repair_limit)
    )
    assert budget.repair_usd == (10.0 if repair_limit is None else repair_limit)
    budget.record_external_charge(
        "prior", known_cost_usd=spent, uncertain_upper_usd=0.0, context=adapter.context
    )
    adapter.context.services["cropped_cost_budget"] = budget
    harness = MagicMock()
    harness.context = adapter.context
    harness.view.return_value = view
    state = CroppedRepairState(adapter.context, view.request)
    with pytest.raises(CroppedRepairBudgetReached):
        CroppedFocusedRepair(state).decide(harness, adapter)
    harness.view.assert_not_called()
    harness.save_repair_checkpoint.assert_not_called()
    selected.invoke.assert_not_called()
    assert "cropped_cost_phase" not in adapter.context.services
    isolated = fresh_model(adapter)
    with pytest.raises(CroppedRepairBudgetReached):
        isolated._invoke_once(
            [UserMessage(content="Check the requested defect")],
            [],
            LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK,
            max_tokens=1024,
            research=False,
            structured=False,
        )
    selected.invoke.assert_not_called()
    assert budget.research_allowed is (spent < budget.research_usd)
