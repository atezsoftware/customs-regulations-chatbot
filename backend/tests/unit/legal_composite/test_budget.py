from concurrent.futures import ThreadPoolExecutor

import pytest

from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.models import WorkflowPolicy


def test_research_retains_draft_and_review_tokens_and_cost() -> None:
    budget = WorkflowBudget(
        WorkflowPolicy(
            max_input_tokens=1_000, max_output_tokens=600, max_cost_usd=0.015
        )
    )
    budget.configure_finalization(300, 100, 0.004)
    budget.request(400, 100, 10, 10)
    with pytest.raises(RunStopped, match="finalization allocation"):
        budget.request(1, 1, 10, 10)
    budget.request(300, 100, 10, 10, finalizing=True)
    budget.request(300, 100, 10, 10, finalizing=True)
    assert budget.snapshot()["input_tokens"] == 1_000
    assert budget.snapshot()["estimated_cost_usd"] == 0.013


def test_parallel_requests_cannot_spend_final_generation_slots() -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    budget.configure_finalization(100, 100, 0.0001)

    def allocate() -> bool:
        try:
            budget.request(1, 1, 0.1, 0.5)
        except RunStopped:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        successes = list(executor.map(lambda _: allocate(), range(30)))
    assert sum(successes) == 6
    budget.request(100, 100, 0.1, 0.5, finalizing=True)
    budget.request(100, 100, 0.1, 0.5, finalizing=True)
    assert budget.snapshot()["model_calls"] == 8


def test_usage_refund_and_failed_reservation_are_accounted_separately() -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    budget.configure_finalization(100, 100, 0.0001)
    completed = budget.request(1_000, 2_000, 1, 5)
    budget.request(1_000, 2_000, 1, 5)
    budget.settle(completed, 200, 50)
    snapshot = budget.snapshot()
    assert snapshot["input_tokens"] == 1_200
    assert snapshot["output_tokens"] == 2_050
    assert snapshot["estimated_cost_usd"] == 0.01145
    assert snapshot["unsettled_calls"] == 1
    with pytest.raises(ValueError, match="already settled"):
        budget.settle(completed, 200, 50)


def test_usage_above_estimate_blocks_further_spend() -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    budget.configure_finalization(100, 100, 0.0001)
    reservation = budget.request(100, 100, 1, 5)
    budget.settle(reservation, 101, 100)
    assert budget.snapshot()["usage_overrun"] is True
    with pytest.raises(RunStopped, match="usage exceeded"):
        budget.request(1, 1, 1, 5, finalizing=True)


def test_research_deadline_leaves_finalization_time() -> None:
    now = [0.0]
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: now[0])
    budget.configure_finalization(100, 100, 0.0001)
    now[0] = 80
    assert not budget.research_available()
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(1, 1, 1, 5)
    final = budget.request(100, 100, 1, 5, finalizing=True)
    assert final.timeout_seconds == 30
    now[0] = 118
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(1, 1, 1, 5, finalizing=True)


def test_expensive_model_reserve_is_rejected_before_generation() -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    with pytest.raises(RunStopped, match="cannot fit"):
        budget.configure_finalization(32_000, 4_096, 0.2)
