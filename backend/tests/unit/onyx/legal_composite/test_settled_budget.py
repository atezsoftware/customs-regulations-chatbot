import pytest

from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.models import WorkflowPolicy
from onyx.legal_composite.settled_budget import SettledUsageBudget


def policy(**limits: float | int) -> WorkflowPolicy:
    values = {
        "max_input_tokens": 1000,
        "max_output_tokens": 1000,
        "max_model_calls": 10,
        "max_cost_usd": 0.1,
    }
    return WorkflowPolicy.model_validate({**values, **limits})


@pytest.mark.parametrize("actual_input,actual_output", [(101, 10), (100, 11)])
def test_settled_variance_keeps_exact_charges_and_remaining_admission(
    actual_input: int, actual_output: int
) -> None:
    budget = SettledUsageBudget(policy())
    first = budget.request(100, 10, 1, 1)
    budget.settle(first, actual_input, actual_output)
    assert budget.snapshot()["input_tokens"] == actual_input
    assert budget.snapshot()["output_tokens"] == actual_output
    assert budget.snapshot()["estimated_cost_usd"] == pytest.approx(
        (actual_input + actual_output) / 1_000_000
    )
    assert budget.research_available()
    next_call = budget.request(100, 10, 1, 1)
    assert next_call.input_tokens == 100


@pytest.mark.parametrize(
    "limits,actual_input,actual_output",
    [
        ({"max_input_tokens": 100}, 101, 10),
        ({"max_output_tokens": 10}, 100, 11),
        ({"max_cost_usd": 0.0001105}, 101, 10),
    ],
)
def test_total_usage_caps_still_stop_every_new_generation(
    limits: dict[str, int | float], actual_input: int, actual_output: int
) -> None:
    budget = SettledUsageBudget(policy(**limits))
    first = budget.request(100, 10, 1, 1)
    budget.settle(first, actual_input, actual_output)
    assert not budget.research_available()
    with pytest.raises(RunStopped, match="capacity"):
        budget.request(1, 1, 1, 1, finalizing=True)


def test_failed_calls_keep_their_full_allocation_and_finalization_reserve() -> None:
    budget = SettledUsageBudget(policy(max_input_tokens=400))
    budget.configure_finalization(100, 10, 0.00011)
    budget.request(100, 10, 1, 1)
    assert budget.affordable_input_tokens(10, 1, 1) == 100
    with pytest.raises(RunStopped, match="allocation"):
        budget.request(101, 10, 1, 1)
    budget.stop("Provider call exceeded its deadline; no further spend authorized")
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(1, 1, 1, 1, finalizing=True)


def test_shared_legacy_budget_retains_original_variance_guard() -> None:
    budget = WorkflowBudget(policy())
    first = budget.request(100, 10, 1, 1)
    budget.settle(first, 101, 10)
    with pytest.raises(RunStopped, match="estimate"):
        budget.request(100, 10, 1, 1)
