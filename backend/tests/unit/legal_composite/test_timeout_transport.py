"""Exercise the canonical non-streaming adapter without provider network calls."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import litellm
import pytest
from litellm.exceptions import Timeout as LiteLLMTimeout

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine, SourceAcquirer
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import DraftAnswer, WorkflowPolicy
from onyx.llm.cost import ModelPrice
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM, LLMTimeoutError
from onyx.tracing.flows import LLMFlow


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> LitellmLLM:
    monkeypatch.setattr("onyx.llm.multi_llm._env_injection_enabled", lambda: False)
    monkeypatch.setattr("onyx.llm.multi_llm.get_llm_mock_response", lambda: None)
    monkeypatch.setattr("litellm.HTTPHandler", Mock())
    monkeypatch.setattr("litellm.get_model_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.get_model_price_per_million",
        lambda *_args: ModelPrice(
            model="gemini-3.8-flash",
            provider="vertex_ai",
            input_per_mtok=0.1,
            output_per_mtok=0.5,
            cache_per_mtok=None,
        ),
    )
    return LitellmLLM(
        api_key="dummy",
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        max_input_tokens=32_000,
        timeout=30,
    )


def response(content: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="canonical-provider-response",
        created=1,
        model="gemini-3.8-flash",
        choices=[
            litellm.Choices(
                message=litellm.Message(role="assistant", content=content),
                finish_reason="stop",
                index=0,
            )
        ],
        usage=litellm.Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )


def gateway(
    model: LitellmLLM,
    ledger: EvidenceLedger | Mock,
    clock: Callable[[], float] | None = None,
) -> BudgetedGateway:
    policy = WorkflowPolicy()
    budget = WorkflowBudget(policy) if clock is None else WorkflowBudget(policy, clock)
    return BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )


def test_full_transport_window_allows_answer_rejected_by_ten_second_timeout(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def complete(**kwargs: Any) -> litellm.ModelResponse:
        if kwargs["timeout"] < 20:
            raise LiteLLMTimeout(
                "inference needs 20 seconds", "gemini-3.8-flash", "vertex_ai"
            )
        return response('{"answer":"supported","unresolved_need_ids":[]}')

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    workflow = gateway(model, ledger)
    result = workflow.complete(
        "Use full original law",
        {"original_evidence": [{"citation": 1, "text": "complete original"}]},
        DraftAnswer,
        LLMFlow.LEGAL_COMPOSITE_ANSWER,
        True,
    )
    assert result.answer == "supported"
    completion.assert_called_once()
    assert completion.call_args.kwargs["timeout"] == 30
    assert completion.call_args.kwargs["client"] is not None
    ledger.record_delivery.assert_called_once()
    assert workflow.budget.snapshot()["unsettled_calls"] == 0


def test_canonical_timeout_has_no_compatibility_retry_or_followup_spend(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = LiteLLMTimeout("deadline", "gemini-3.8-flash", "vertex_ai")
    completion = Mock(side_effect=failure)
    monkeypatch.setattr("litellm.completion", completion)
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger)
    with pytest.raises(RunStopped, match="timed out") as stopped:
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert isinstance(stopped.value.__cause__, LLMTimeoutError)
    assert stopped.value.__cause__.args == (failure,)
    retained = workflow.budget.snapshot()
    assert retained["model_calls"] == 1 and retained["unsettled_calls"] == 1
    retained_cost = retained["estimated_cost_usd"]
    assert isinstance(retained_cost, (int, float)) and retained_cost > 0
    assert retained["stop_reason"] == (
        "Provider call exceeded its deadline; no further spend authorized"
    )
    with pytest.raises(RunStopped, match="no further spend"):
        workflow.research_proxy().invoke(
            UserMessage(content="Retry the failed provider")
        )
    completion.assert_called_once()
    assert completion.call_args.kwargs["stream"] is False
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None
    assert (
        workflow.budget.snapshot()["estimated_cost_usd"]
        == retained["estimated_cost_usd"]
    )


def test_remaining_workflow_deadline_rejects_late_original_delivery(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger, lambda: now[0])
    now[0] = 116.0

    def late(**_kwargs: Any) -> litellm.ModelResponse:
        now[0] = 120.1
        return response('{"answer":"late","unresolved_need_ids":[]}')

    completion = Mock(side_effect=late)
    monkeypatch.setattr("litellm.completion", completion)
    with pytest.raises(RunStopped, match="deadline reached"):
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert completion.call_args.kwargs["timeout"] == 4
    with pytest.raises(RunStopped, match="deadline reached"):
        workflow.complete(
            "Review", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_REVIEW, True
        )
    completion.assert_called_once()
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None


def test_writer_timeout_ends_engine_without_review_or_repair(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = (
        '{"language":"tr","requires_sources":false,"needs":'
        '[{"need_id":"greeting","question":"Greet user","governing_source":"",'
        '"conditions_to_check":[]}],"initial_actions":[],"missing_user_facts":[]}'
    )
    completion = Mock(
        side_effect=[
            response(plan),
            LiteLLMTimeout("deadline", "gemini-3.8-flash", "vertex_ai"),
        ]
    )
    monkeypatch.setattr("litellm.completion", completion)
    ledger = EvidenceLedger()
    workflow = gateway(model, ledger)
    acquirer = Mock(spec=SourceAcquirer)
    acquirer.definitions.return_value = []
    engine = LegalCompositeEngine(
        gateway=workflow,
        acquirer=acquirer,
        ledger=ledger,
        policy=workflow.budget.policy,
        check_active=lambda: None,
        research_available=workflow.budget.research_available,
    )
    result = engine.run("merhaba", "", None)
    assert result.status == "unavailable" and result.answer is None
    assert result.review is None
    assert result.gaps == ["The bounded model invocation timed out"]
    assert completion.call_count == 2
    acquirer.acquire.assert_not_called()
    assert workflow.budget.snapshot()["model_calls"] == 2
    assert workflow.budget.snapshot()["unsettled_calls"] == 1
