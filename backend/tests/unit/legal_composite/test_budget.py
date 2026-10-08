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
    assert final.timeout_seconds == 40
    now[0] = 118
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(1, 1, 1, 5, finalizing=True)


@pytest.mark.parametrize("finalizing,deadline", [(False, 80), (True, 120)])
def test_returned_response_uses_exact_deadline_without_admitting_another_call(
    finalizing: bool, deadline: int
) -> None:
    now = [0.0]
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: now[0])
    budget.configure_finalization(100, 100, 0.0001)
    now[0] = deadline - 1.6686
    budget.check_response_active(finalizing)
    retained = budget.snapshot()
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(100, 100, 0.1, 0.5, finalizing)
    assert budget.snapshot() == retained
    now[0] = deadline
    with pytest.raises(RunStopped, match="deadline"):
        budget.check_response_active(finalizing)


def test_returned_response_cannot_ignore_persistent_provider_stop() -> None:
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: 0.0)
    budget.stop("Provider outcome unknown; no further spend")
    with pytest.raises(RunStopped, match="Provider outcome unknown"):
        budget.check_response_active(finalizing=True)


def test_delayed_budget_setup_keeps_the_runtime_absolute_deadline() -> None:
    now = [10.0]
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: now[0], deadline=120.0)
    budget.configure_finalization(100, 100, 0.0001)
    assert budget.snapshot()["elapsed_seconds"] == 10
    assert budget.deadline == 120
    now[0] = 110
    call = budget.request(100, 100, 0.1, 0.5, finalizing=True)
    assert call.timeout_seconds == 10
    now[0] = 120
    with pytest.raises(RunStopped, match="deadline"):
        budget.check_response_active(finalizing=True)


def test_longer_writer_window_preserves_total_deadline_and_other_limits() -> None:
    now = [0.0]
    policy = WorkflowPolicy()
    budget = WorkflowBudget(policy, clock=lambda: now[0])
    budget.configure_finalization(100, 100, 0.0001)
    assert (
        policy.timeout_seconds,
        policy.finalization_reserve_seconds,
        policy.max_model_calls,
        policy.max_context_tokens,
        policy.max_input_tokens,
        policy.max_output_tokens,
        policy.max_cost_usd,
    ) == (120, 40, 8, 32_000, 120_000, 24_000, 0.10)
    now[0] = 46
    writer = budget.request(100, 100, 0.1, 0.5, finalizing=True)
    assert writer.timeout_seconds == 45
    now[0] += writer.timeout_seconds
    review = budget.request(100, 100, 0.1, 0.5, finalizing=True)
    assert review.timeout_seconds == 29
    assert now[0] + review.timeout_seconds == budget.deadline == 120
    now[0] = 120
    retained = budget.snapshot()
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(1, 1, 0.1, 0.5, finalizing=True)
    assert budget.snapshot() == retained


def test_expensive_model_reserve_is_rejected_before_generation() -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    with pytest.raises(RunStopped, match="cannot fit"):
        budget.configure_finalization(32_000, 4_096, 0.2)
