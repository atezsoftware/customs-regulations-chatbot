"""Research timeout can retain finalization without waiving any spending fence."""

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any
from unittest.mock import Mock

import pytest
from litellm.exceptions import Timeout as LiteLLMTimeout

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import ResearchPhaseClosed, WorkflowBudget
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import DraftAnswer, IssueResearchStep, WorkflowPolicy
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.tracing.flows import LLMFlow
from tests.unit.legal_composite.test_timeout_transport import model as model
from tests.unit.legal_composite.test_timeout_transport import response


def gateway(model: LitellmLLM, ledger: Mock, now: list[float]) -> BudgetedGateway:
    return BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy(), lambda: now[0]),
        ledger=ledger,
        preserve_research_finalization_on_timeout=True,
    )


def test_research_closure_retains_failed_allocation_and_exact_final_reserves() -> None:
    now = [0.0]
    budget = WorkflowBudget(WorkflowPolicy(), lambda: now[0])
    budget.configure_finalization(100, 10, 0.00011)
    failed = budget.request(120, 20, 1, 1)
    before = budget.snapshot()
    budget.close_research("host_research_deadline")
    closed = budget.snapshot()
    assert {
        key: value for key, value in closed.items() if key != "research_stop_reason"
    } == before
    assert not budget.research_available()
    with pytest.raises(ResearchPhaseClosed) as stopped:
        budget.request(10, 1, 1, 1)
    assert stopped.value.reason == "host_research_deadline"
    assert budget.snapshot() == closed
    budget.close_research("provider_timeout")
    assert budget.snapshot() == closed
    for pending in (1, 0):
        final = budget.request(100, 10, 1, 1, finalizing=True)
        budget.settle(final, 90, 9)
        assert budget.snapshot()["pending_final_calls"] == pending
    assert budget.snapshot()["unsettled_calls"] == 1
    assert budget.snapshot()["stop_reason"] is None
    assert budget.snapshot()["input_tokens"] == failed.input_tokens + 180
    now[0] = budget.deadline
    with pytest.raises(RunStopped, match="deadline"):
        budget.request(10, 1, 1, 1, finalizing=True)


@pytest.mark.parametrize("fence", ["input", "output", "cost", "global_stop"])
def test_research_closure_cannot_override_capacity_or_global_stop(fence: str) -> None:
    policy = WorkflowPolicy()
    budget = WorkflowBudget(policy, lambda: 0.0)
    reservation = budget.request(10, 1, 1, 1)
    budget.close_research("provider_timeout")
    if fence == "global_stop":
        budget.stop("Explicit global stop")
    elif fence == "input":
        budget.settle(reservation, policy.max_input_tokens + 1, 1)
    elif fence == "output":
        budget.settle(reservation, 10, policy.max_output_tokens + 1)
    else:
        budget.settle(reservation, 110_000, 1)
    before = budget.snapshot()
    with pytest.raises(RunStopped):
        budget.request(1, 1, 1, 1, finalizing=True)
    assert budget.snapshot() == before


def test_research_closure_does_not_authorize_unaffordable_finalization() -> None:
    budget = WorkflowBudget(WorkflowPolicy(), lambda: 0.0)
    budget.close_research("provider_timeout")
    before = budget.snapshot()
    with pytest.raises(RunStopped, match="budget exhausted"):
        budget.request(100_000, 1, 2, 1, finalizing=True)
    assert budget.snapshot() == before


def test_actual_provider_reported_research_timeout_allows_one_reserved_writer(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = Mock(
        side_effect=[
            LiteLLMTimeout("timeout", "gemini-3.8-flash", "vertex_ai"),
            response('{"answer":"supported","unresolved_need_ids":[]}'),
        ]
    )
    monkeypatch.setattr("litellm.completion", completion)
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = set()
    workflow = gateway(model, ledger, [0.0])
    with pytest.raises(ResearchPhaseClosed) as stopped:
        workflow.complete(
            "Read originals", {}, IssueResearchStep, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert stopped.value.reason == "provider_timeout"
    retained = workflow.budget.snapshot()
    assert retained["stop_reason"] is None
    assert retained["unsettled_calls"] == 1 and retained["pending_final_calls"] == 2
    assert not workflow.budget.research_available()
    with pytest.raises(ResearchPhaseClosed):
        workflow.research_proxy().invoke(UserMessage(content="No new research"))
    completion.assert_called_once()
    assert workflow.budget.snapshot() == retained
    ledger.record_delivery.assert_not_called()
    final = workflow.complete(
        "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
    )
    assert final.answer == "supported" and completion.call_count == 2
    assert workflow.budget.snapshot()["unsettled_calls"] == 1
    assert workflow.budget.snapshot()["pending_final_calls"] == 1
    ledger.record_delivery.assert_called_once()


@pytest.mark.parametrize(
    "phase_limit", [False, True], ids=["host-call-window", "host-research-boundary"]
)
def test_host_timeout_origin_is_distinct_and_late_result_never_becomes_evidence(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch, phase_limit: bool
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = set()
    workflow = gateway(model, ledger, now)
    if phase_limit:
        now[0] = 76.0

    class TimedFuture(Future[object]):
        def result(self, timeout: float | None = None) -> object:
            assert timeout is not None
            now[0] += timeout
            raise FutureTimeout()

    future = TimedFuture()
    future.set_running_or_notify_cancel()
    shutdown = Mock()

    class Executor:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 1

        def submit(self, *_args: Any, **_kwargs: Any) -> TimedFuture:
            return future

        def shutdown(self, **kwargs: Any) -> None:
            shutdown(**kwargs)

    monkeypatch.setattr("onyx.legal_composite.gateway.ThreadPoolExecutor", Executor)
    with pytest.raises(ResearchPhaseClosed) as stopped:
        workflow.complete(
            "Read originals", {}, IssueResearchStep, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert stopped.value.reason == (
        "host_research_deadline" if phase_limit else "host_call_timeout"
    )
    assert workflow.budget.snapshot()["stop_reason"] is None
    assert workflow.budget.snapshot()["unsettled_calls"] == 1
    shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    future.set_result(response("Late unread source cannot be delivered"))
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None and workflow.last_delivered_citations == set()
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.ThreadPoolExecutor", ThreadPoolExecutor
    )
    completion = Mock(
        return_value=response('{"answer":"supported","unresolved_need_ids":[]}')
    )
    monkeypatch.setattr("litellm.completion", completion)
    assert (
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        ).answer
        == "supported"
    )
    assert workflow.budget.snapshot()["unsettled_calls"] == 1


def test_finalizing_timeout_remains_global_even_with_research_optin(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = Mock(
        side_effect=LiteLLMTimeout("timeout", "gemini-3.8-flash", "vertex_ai")
    )
    monkeypatch.setattr("litellm.completion", completion)
    workflow = gateway(model, Mock(spec=EvidenceLedger), [0.0])
    with pytest.raises(RunStopped, match="timed out") as stopped:
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert not isinstance(stopped.value, ResearchPhaseClosed)
    assert (
        workflow.budget.snapshot()["stop_reason"]
        == "Provider call exceeded its deadline; no further spend authorized"
    )
    with pytest.raises(RunStopped, match="no further spend"):
        workflow.complete(
            "Review", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_REVIEW, True
        )
    completion.assert_called_once()


def test_default_research_timeout_keeps_legacy_global_stop_behavior(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = Mock(
        side_effect=LiteLLMTimeout("timeout", "gemini-3.8-flash", "vertex_ai")
    )
    monkeypatch.setattr("litellm.completion", completion)
    workflow = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy(), lambda: 0.0),
        ledger=Mock(spec=EvidenceLedger),
    )
    with pytest.raises(RunStopped, match="timed out") as stopped:
        workflow.complete(
            "Read originals", {}, IssueResearchStep, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert not isinstance(stopped.value, ResearchPhaseClosed)
    assert workflow.budget.snapshot()["stop_reason"] is not None
    assert "research_stop_reason" not in workflow.budget.snapshot()
    completion.assert_called_once()


def test_research_closure_never_waives_cancellation_before_finalization(
    model: LitellmLLM,
) -> None:
    workflow = gateway(model, Mock(spec=EvidenceLedger), [0.0])
    workflow.budget.close_research("host_research_deadline")
    before = workflow.budget.snapshot()
    workflow.check_active = Mock(side_effect=RunStopped("Research cancelled"))
    with pytest.raises(RunStopped, match="cancelled"):
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert workflow.budget.snapshot() == before
