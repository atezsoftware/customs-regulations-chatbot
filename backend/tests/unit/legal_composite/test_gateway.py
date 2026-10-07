from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import DraftAnswer, WorkflowPolicy
from onyx.llm.cost import ModelPrice
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import (
    ImageContentPart,
    ImageUrlDetail,
    ReasoningEffort,
    UserMessage,
)
from onyx.tracing.flows import LLMFlow


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Mock:
    llm = Mock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="price-test",
        model_name="price-test-model",
        temperature=0,
        max_input_tokens=32_000,
    )
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.get_model_price_per_million",
        lambda *_args: ModelPrice(
            model="price-test-model",
            provider="price-test",
            input_per_mtok=0.1,
            output_per_mtok=0.5,
            cache_per_mtok=None,
        ),
    )
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.check_number_of_tokens",
        lambda text: len(text) // 4,
    )
    llm.invoke.return_value = ModelResponse(
        id="test",
        created="0",
        choice=Choice(
            message=Message(content='{"answer":"supported","unresolved_need_ids":[]}')
        ),
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    return llm


def test_gateway_records_exact_delivered_originals_and_bounds_invocation(
    model: Mock,
) -> None:
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    budget = WorkflowBudget(WorkflowPolicy())
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    records: list[JsonValue] = [{"citation": 1, "text": "complete original"}]
    answer = gateway.complete(
        "Answer using original law",
        {"original_evidence": records},
        DraftAnswer,
        LLMFlow.LEGAL_COMPOSITE_ANSWER,
        finalizing=True,
    )
    assert answer.answer == "supported"
    ledger.record_delivery.assert_called_once_with(
        gateway.last_call_id, LLMFlow.LEGAL_COMPOSITE_ANSWER.value, records
    )
    kwargs = model.invoke.call_args.kwargs
    assert kwargs["timeout_override"] == 10
    assert kwargs["max_tokens"] == 4_096
    assert kwargs["use_streaming"] is False
    assert budget.snapshot()["model_calls"] == 1
    assert gateway.last_delivered_citations == {1}


def test_context_fit_preserves_required_originals_and_marks_omitted_ids(
    model: Mock,
) -> None:
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy(max_context_tokens=2_000)),
        ledger=ledger,
    )
    payload: dict[str, JsonValue] = {
        "original_evidence": [
            {"citation": 1, "text": "required complete original"},
            {"citation": 2, "text": "x" * 20_000},
        ],
        "required_evidence_numbers": [1],
    }
    gateway.complete(
        "Answer", payload, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
    )
    sent = model.invoke.call_args.args[0][1].content
    assert '"omitted_original_ids": [2]' in sent
    ledger.record_delivery.assert_called_once_with(
        gateway.last_call_id,
        LLMFlow.LEGAL_COMPOSITE_ANSWER.value,
        [{"citation": 1, "text": "required complete original"}],
    )
    assert len(payload["original_evidence"]) == 2


def test_oversized_required_original_cannot_be_clipped_or_dropped(model: Mock) -> None:
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy(max_context_tokens=1_000)),
        ledger=EvidenceLedger(),
    )
    with pytest.raises(RunStopped, match="Required originals"):
        gateway.complete(
            "Answer",
            {
                "original_evidence": [{"citation": 1, "text": "x" * 20_000}],
                "required_evidence_numbers": [1],
            },
            DraftAnswer,
            LLMFlow.LEGAL_COMPOSITE_ANSWER,
            True,
        )
    model.invoke.assert_not_called()


@pytest.mark.parametrize("context_cap, expected_rate", [(32_000, 0.75), (240_000, 1.5)])
def test_pricing_reserves_only_reachable_default_online_tiers(
    model: Mock, monkeypatch: pytest.MonkeyPatch, context_cap: int, expected_rate: float
) -> None:
    from onyx.legal_composite.gateway import _priced_model

    monkeypatch.setattr(
        "litellm.get_model_info",
        lambda **_kwargs: {
            "input_cost_per_token": 0.00000075,
            "output_cost_per_token": 0.00000375,
            "input_cost_per_token_above_200k_tokens": 0.0000015,
            "output_cost_per_token_above_200k_tokens": 0.0000075,
            "input_cost_per_token_priority": 0.000003,
        },
    )
    price = _priced_model(model, None, context_cap)
    assert price.input_per_mtok == expected_rate


def test_unknown_pricing_rejects_before_any_invocation(
    model: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.get_model_price_per_million",
        lambda *_args: ModelPrice(
            model="unknown",
            provider="price-test",
            input_per_mtok=None,
            output_per_mtok=None,
            cache_per_mtok=None,
        ),
    )
    with pytest.raises(RunStopped, match="verified token price"):
        BudgetedGateway(
            selected_llm=model,
            research_llm=model,
            budget=WorkflowBudget(WorkflowPolicy()),
            ledger=EvidenceLedger(),
        )
    model.invoke.assert_not_called()


def test_cancelled_request_cannot_start_a_paid_invocation(model: Mock) -> None:
    cancelled = [False]

    def check_active() -> None:
        if cancelled[0]:
            raise RunStopped("Research cancelled")

    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
        check_active=check_active,
    )
    cancelled[0] = True
    with pytest.raises(RunStopped, match="cancelled"):
        gateway.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    model.invoke.assert_not_called()


def test_invalid_schema_is_not_retried(model: Mock) -> None:
    model.invoke.return_value.choice.message.content = (
        '{"answer":123,"unresolved_need_ids":[]}'
    )
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
    )
    with pytest.raises(RunStopped, match="workflow schema"):
        gateway.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert model.invoke.call_count == 1


def test_late_provider_result_cannot_publish_evidence_or_start_more_calls(
    model: Mock,
) -> None:
    released = Event()
    finished = Event()
    response = model.invoke.return_value

    def delayed(*_args: object, **_kwargs: object) -> ModelResponse:
        released.wait(timeout=10)
        finished.set()
        return response

    model.invoke.side_effect = delayed
    ledger = Mock(spec=EvidenceLedger)
    budget = WorkflowBudget(WorkflowPolicy(max_call_seconds=3))
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    try:
        with pytest.raises(RunStopped, match="timed out"):
            gateway.complete(
                "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
            )
    finally:
        released.set()
        assert finished.wait(timeout=2)
    ledger.record_delivery.assert_not_called()
    assert gateway.last_call_id is None
    assert not gateway.last_delivered_citations
    with pytest.raises(RunStopped, match="no further spend"):
        gateway.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    assert model.invoke.call_count == 1


def test_final_generation_preserves_the_selected_reasoning_effort(model: Mock) -> None:
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
        reasoning_effort=ReasoningEffort.HIGH,
    )
    gateway.complete("Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True)
    assert model.invoke.call_args.kwargs["reasoning_effort"] is ReasoningEffort.HIGH


def test_search_helpers_share_generation_budget_and_preserve_two_final_calls(
    model: Mock,
) -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    gateway.complete("Plan", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_RESEARCH)
    delivered_call = gateway.last_call_id
    proxy = gateway.research_proxy()
    for _ in range(5):
        proxy.invoke(UserMessage(content="Select a relevant legal section"))
    assert gateway.last_call_id == delivered_call
    assert gateway.last_delivered_citations == {1}
    assert ledger.record_delivery.call_count == 1
    assert budget.snapshot()["model_calls"] == 6
    assert budget.snapshot()["unsettled_calls"] == 0
    assert budget.snapshot()["input_tokens"] == 600
    with pytest.raises(RunStopped, match="finalization allocation retained"):
        proxy.invoke(UserMessage(content="One more selector"))
    assert model.invoke.call_count == 6
    gateway.complete("Draft", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True)
    gateway.complete("Review", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_REVIEW, True)
    assert budget.snapshot()["model_calls"] == 8


def test_search_helper_clamps_output_and_forces_low_collected_generation(
    model: Mock,
) -> None:
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
    )
    proxy = gateway.research_proxy()
    proxy.invoke(
        UserMessage(content="Rank originals"),
        max_tokens=12_000,
        reasoning_effort=ReasoningEffort.HIGH,
        structured_response_format={"type": "json_object"},
        timeout_override=6,
    )
    kwargs = model.invoke.call_args.kwargs
    assert kwargs["max_tokens"] == 2_048
    assert kwargs["reasoning_effort"] is ReasoningEffort.LOW
    assert kwargs["timeout_override"] == 2
    assert kwargs["use_streaming"] is False
    assert kwargs["structured_response_format"] == {"type": "json_object"}


def test_unsupported_auxiliary_calls_fail_before_spend(model: Mock) -> None:
    budget = WorkflowBudget(WorkflowPolicy(max_context_tokens=1_000))
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=EvidenceLedger()
    )
    proxy = gateway.research_proxy()
    prompt = UserMessage(content="Select a legal section")
    with pytest.raises(RunStopped, match="cannot execute tools"):
        proxy.invoke(prompt, tools=[{"type": "function"}])
    with pytest.raises(RunStopped, match="streaming is unavailable"):
        list(proxy.stream(prompt))
    with pytest.raises(RunStopped, match="context budget"):
        proxy.invoke(UserMessage(content="x" * 20_000))
    with pytest.raises(ValueError, match="timeout"):
        proxy.invoke(prompt, timeout_override=1)
    with pytest.raises(ValueError, match="output limit"):
        proxy.invoke(prompt, max_tokens=0)
    with pytest.raises(RunStopped, match="Multimodal"):
        proxy.invoke(
            UserMessage(
                content=[ImageContentPart(image_url=ImageUrlDetail(url="test-image"))]
            )
        )
    model.invoke.assert_not_called()
    assert budget.snapshot()["model_calls"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"service_tier": "priority"},
        {"extra_body": '{"service_tier":"priority"}'},
        {"extra_body": '{"provider":{"order":["unpriced-provider"]}}'},
        {"extra_body": '{"options":{"max_retries":2}}'},
        {"extra_body": '[{"nested":{"num_retries":2}}]'},
        {"extra_body": '[[{"nested":{"num_retries":2}}]]'},
        {"extra_body": '{"custom_llm_provider":"unpriced-provider"}'},
    ],
)
def test_provider_cost_or_retry_overrides_fail_before_spend(
    model: Mock, settings: dict[str, str]
) -> None:
    model.config.custom_config = settings
    with pytest.raises(RunStopped, match="retry, routing or service-tier"):
        BudgetedGateway(
            selected_llm=model,
            research_llm=model,
            budget=WorkflowBudget(WorkflowPolicy()),
            ledger=EvidenceLedger(),
        )
    model.invoke.assert_not_called()


def test_concurrent_search_helpers_serialize_before_reserving_paid_calls(
    model: Mock,
) -> None:
    first_started, second_queued, release_first = Event(), Event(), Event()
    response = model.invoke.return_value

    def check_active() -> None:
        if current_thread().name == "helper_1":
            second_queued.set()

    def generation(*_args: object, **_kwargs: object) -> ModelResponse:
        if not first_started.is_set():
            first_started.set()
            assert release_first.wait(timeout=2)
        return response

    model.invoke.side_effect = generation
    budget = WorkflowBudget(WorkflowPolicy())
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=budget,
        ledger=EvidenceLedger(),
        check_active=check_active,
    )
    proxy = gateway.research_proxy()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="helper") as executor:
        first = executor.submit(proxy.invoke, UserMessage(content="First selector"))
        assert first_started.wait(timeout=2)
        second = executor.submit(proxy.invoke, UserMessage(content="Second selector"))
        try:
            assert second_queued.wait(timeout=2)
            assert model.invoke.call_count == 1
            assert budget.snapshot()["model_calls"] == 1
        finally:
            release_first.set()
        assert first.result(timeout=2) is response
        assert second.result(timeout=2) is response
    assert budget.snapshot()["model_calls"] == 2
