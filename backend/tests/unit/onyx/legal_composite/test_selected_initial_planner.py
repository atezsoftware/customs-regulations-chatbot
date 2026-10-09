"""The LC planner opt-in changes its model without changing research fences."""

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import TypedDict, cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.legal_composite.budget import ResearchPhaseClosed, WorkflowBudget
from onyx.legal_composite.gateway import (
    BudgetedGateway,
    MissingRequiredOriginal,
    ModelContextLimit,
)
from onyx.legal_composite.models import (
    DraftAnswer,
    IssueResearchPlan,
    IssueResearchStep,
    ResearchPlan,
    ResearchStep,
    WorkflowPolicy,
)
from onyx.legal_composite.query_repair import DiscoveryQuery
from onyx.legal_composite.reading_evidence import (
    IssueReadingResponse,
    number_reading_witnesses,
)
from onyx.llm.cost import ModelPrice
from onyx.llm.interfaces import LLM, LLMConfig, LLMUserIdentity
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.tracing.flows import LLMFlow

PLAN = {
    "language": "tr",
    "requires_sources": True,
    "needs": [
        {
            "need_id": "conditions",
            "question": "Which conditions apply to the requested transaction?",
            "governing_source": "Not yet established",
            "conditions_to_check": ["Scope and exceptions"],
        },
        {
            "need_id": "procedure",
            "question": "Which procedure applies?",
            "governing_source": "Not yet established",
            "conditions_to_check": ["Required documents"],
        },
    ],
    "initial_actions": [],
    "missing_user_facts": [],
}
STEP = {"actions": [], "ready_to_answer": False, "remaining_gaps": []}


def response(content: str, *, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        id="fake-priced-provider-response",
        created="0",
        choice=Choice(message=Message(content=content), finish_reason=finish_reason),
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=12,
            total_tokens=112,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@dataclass
class Providers:
    selected: Mock
    research: Mock


class _PlannerSettings(TypedDict, total=False):
    use_selected_llm_for_initial_plan: bool


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> Providers:
    def fake_model(provider: str) -> Mock:
        model = Mock(spec=LLM)
        model.config = LLMConfig(
            model_provider=provider,
            model_name=f"{provider}-model",
            max_input_tokens=32_000,
            temperature=0,
        )
        model.invoke.return_value = response(json.dumps(PLAN))
        return model

    def price(model: str, provider: str, _session: object) -> ModelPrice:
        rates = {"selected-provider": (2.0, 3.0), "research-provider": (0.1, 0.5)}
        input_rate, output_rate = rates[provider]
        return ModelPrice(
            model=model,
            provider=provider,
            input_per_mtok=input_rate,
            output_per_mtok=output_rate,
            cache_per_mtok=None,
        )

    monkeypatch.setattr(
        "onyx.legal_composite.gateway.get_model_price_per_million", price
    )
    monkeypatch.setattr("litellm.get_model_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.check_number_of_tokens",
        lambda text: len(text) // 4,
    )
    return Providers(fake_model("selected-provider"), fake_model("research-provider"))


def gateway(
    providers: Providers,
    *,
    enabled: bool | None = True,
    policy: WorkflowPolicy | None = None,
    ledger: EvidenceLedger | None = None,
    check_active: Callable[[], None] = lambda: None,
    preserve_timeout: bool = False,
) -> BudgetedGateway:
    settings: _PlannerSettings = (
        {} if enabled is None else {"use_selected_llm_for_initial_plan": enabled}
    )
    return BudgetedGateway(
        selected_llm=providers.selected,
        research_llm=providers.research,
        budget=WorkflowBudget(policy or WorkflowPolicy(max_cost_usd=10), lambda: 0.0),
        ledger=ledger or EvidenceLedger(),
        check_active=check_active,
        user_identity=LLMUserIdentity(user_id="fenced-test-actor"),
        preserve_research_finalization_on_timeout=preserve_timeout,
        **settings,
    )


def original_payload(
    text: str = "Complete original. Condition, exception and procedure remain intact.",
    *,
    numbered: bool = False,
) -> tuple[EvidenceLedger, dict[str, JsonValue]]:
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="canonical-source",
                chunk_id="canonical-chunk",
                text=text,
                question_ids=["conditions"],
                metadata={"title": "Canonical original", "document_type": "LAW"},
            )
        ],
        RunContext(),
    )
    records = cast(
        list[JsonValue],
        json.loads(ledger.serialize_records([1], include_witness_spans=numbered)),
    )
    return ledger, {
        "request": "Explain conditions and procedure",
        "original_evidence": number_reading_witnesses(records) if numbered else records,
        "required_evidence_numbers": [1],
    }


@pytest.mark.parametrize("enabled", [None, False, True], ids=["default", "off", "on"])
def test_planner_routes_and_accounts_one_research_call_with_whole_original(
    providers: Providers, monkeypatch: pytest.MonkeyPatch, enabled: bool | None
) -> None:
    ledger, payload = original_payload()
    before = ledger.export()["records"]
    workflow = gateway(providers, enabled=enabled, ledger=ledger)
    allocation = Mock(wraps=workflow.budget.request)
    generation = Mock(wraps=workflow._generate)
    monkeypatch.setattr(workflow.budget, "request", allocation)
    monkeypatch.setattr(workflow, "_generate", generation)

    plan = workflow.complete(
        "Enumerate requested outcomes without guessing legal authority",
        payload,
        IssueResearchPlan,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    called = providers.selected if enabled is True else providers.research
    unused = providers.research if enabled is True else providers.selected
    called.invoke.assert_called_once()
    unused.invoke.assert_not_called()
    assert plan == IssueResearchPlan.model_validate(PLAN)
    assert generation.call_args.args[0] is called
    assert generation.call_args.args[2] is LLMFlow.LEGAL_COMPOSITE_RESEARCH
    assert generation.call_args.args[5] is False
    assert allocation.call_args.args[1] == 6_144
    assert allocation.call_args.args[2:5] == (
        (2.0, 3.0, False) if enabled is True else (0.1, 0.5, False)
    )
    kwargs = called.invoke.call_args.kwargs
    assert kwargs["max_tokens"] == 6_144
    assert kwargs["timeout_override"] == 45
    assert kwargs["reasoning_effort"] is ReasoningEffort.LOW
    assert kwargs["use_streaming"] is False
    assert kwargs["user_identity"].user_id == "fenced-test-actor"
    assert kwargs["structured_response_format"]["json_schema"]["name"] == (
        "IssueResearchPlan"
    )
    sent = json.loads(called.invoke.call_args.args[0][1].content)
    assert sent["original_evidence"] == payload["original_evidence"]
    assert ledger.export()["records"] == before
    assert workflow.last_call_id is not None
    assert ledger.completely_delivered(workflow.last_call_id) == {1}
    assert (
        ledger.delivery_flow(workflow.last_call_id) == LLMFlow.LEGAL_COMPOSITE_RESEARCH
    )
    assert workflow.last_reading_manifest is None
    retained = workflow.budget.snapshot()
    assert retained["model_calls"] == 1 and retained["pending_final_calls"] == 2
    assert retained["unsettled_calls"] == 0
    assert retained["estimated_cost_usd"] == pytest.approx(
        (100 * 2 + 12 * 3) / 1_000_000
        if enabled is True
        else (100 * 0.1 + 12 * 0.5) / 1_000_000
    )


@pytest.mark.parametrize(
    "response_type,content,capacity",
    [
        (ResearchPlan, json.dumps(PLAN), 2_048),
        (ResearchStep, json.dumps(STEP), 2_048),
        (IssueResearchStep, json.dumps(STEP), 6_144),
        (IssueReadingResponse, json.dumps(STEP), 6_144),
        (DiscoveryQuery, '{"query":"distinctive request terms"}', 2_048),
    ],
)
def test_other_research_contracts_keep_research_model_and_exact_delivery(
    providers: Providers,
    response_type: type[BaseModel],
    content: str,
    capacity: int,
) -> None:
    ledger, payload = original_payload(numbered=response_type is IssueReadingResponse)
    workflow = gateway(providers, ledger=ledger)
    providers.research.invoke.return_value = response(content)
    result = workflow.complete(
        "Continue source investigation",
        payload,
        response_type,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    assert result == response_type.model_validate_json(content, strict=True)
    providers.research.invoke.assert_called_once()
    providers.selected.invoke.assert_not_called()
    assert providers.research.invoke.call_args.kwargs["max_tokens"] == capacity
    sent = json.loads(providers.research.invoke.call_args.args[0][1].content)
    assert sent["original_evidence"] == payload["original_evidence"]
    assert workflow.last_delivered_citations == {1}
    if response_type is IssueReadingResponse:
        assert workflow.last_reading_manifest is not None
        assert workflow.last_reading_manifest.call_id == workflow.last_call_id
        assert {row.citation for row in workflow.last_reading_manifest.witnesses} == {1}
    else:
        assert workflow.last_reading_manifest is None


def test_helper_proxy_is_research_model_and_writer_is_selected(
    providers: Providers,
) -> None:
    workflow = gateway(providers)
    helper = workflow.research_proxy()
    assert helper.config == providers.research.config
    providers.research.invoke.return_value = response("Navigation result")
    helper.invoke(
        UserMessage(content="Select relevant source passages"), max_tokens=120
    )
    providers.research.invoke.assert_called_once()
    providers.selected.invoke.assert_not_called()
    assert providers.research.invoke.call_args.kwargs["max_tokens"] == 120
    providers.selected.invoke.return_value = response(
        '{"answer":"Supported answer","unresolved_need_ids":[]}'
    )
    draft = workflow.complete(
        "Write the answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
    )
    assert draft.answer == "Supported answer"
    providers.selected.invoke.assert_called_once()
    assert providers.selected.invoke.call_args.kwargs["max_tokens"] == 4_096
    assert workflow.budget.snapshot()["pending_final_calls"] == 1


def test_plan_subclass_does_not_opt_in_through_inheritance(
    providers: Providers,
) -> None:
    class AuxiliaryPlan(IssueResearchPlan):
        pass

    workflow = gateway(providers)
    workflow.complete(
        "Auxiliary plan", {}, AuxiliaryPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
    )
    providers.research.invoke.assert_called_once()
    providers.selected.invoke.assert_not_called()


@pytest.mark.parametrize("selected_has_capacity", [False, True])
def test_initial_plan_uses_chosen_models_context_and_never_clips_required_original(
    providers: Providers, selected_has_capacity: bool
) -> None:
    providers.selected.config.max_input_tokens = (
        32_000 if selected_has_capacity else 5_000
    )
    providers.research.config.max_input_tokens = (
        5_000 if selected_has_capacity else 32_000
    )
    ledger, payload = original_payload(
        "Exact condition, exception and procedure.\n" * 500
    )
    retained = json.dumps(payload, ensure_ascii=False)
    for enabled in (False, True):
        workflow = gateway(providers, enabled=enabled, ledger=ledger)
        should_fit = selected_has_capacity is enabled
        if should_fit:
            workflow.complete(
                "Plan", payload, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
            )
            called = providers.selected if enabled else providers.research
            called.invoke.assert_called_once()
            sent = json.loads(called.invoke.call_args.args[0][1].content)
            assert sent["original_evidence"] == payload["original_evidence"]
            assert workflow.last_delivered_citations == {1}
            text = cast(dict[str, JsonValue], sent["original_evidence"][0])["text"]
            assert isinstance(text, str)
            canonical = ledger.get(1)
            assert canonical is not None
            assert hashlib.sha256(text.encode()).hexdigest() == canonical.text_hash
        else:
            with pytest.raises(ModelContextLimit):
                workflow.complete(
                    "Plan", payload, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
                )
            assert workflow.budget.snapshot()["model_calls"] == 0
            assert workflow.last_call_id is None
        assert json.dumps(payload, ensure_ascii=False) == retained
        providers.selected.invoke.reset_mock()
        providers.research.invoke.reset_mock()


def test_selected_planner_price_cannot_spend_reserved_writer_and_review_budget(
    providers: Providers,
) -> None:
    policy = WorkflowPolicy(max_cost_usd=0.16)
    selected = gateway(providers, policy=policy)
    with pytest.raises(RunStopped, match="budget exhausted"):
        selected.complete(
            "Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    providers.selected.invoke.assert_not_called()
    providers.research.invoke.assert_not_called()
    assert selected.budget.snapshot()["pending_final_calls"] == 2
    assert selected.budget.snapshot()["model_calls"] == 0
    legacy = gateway(providers, enabled=False, policy=policy)
    legacy.complete("Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH)
    providers.research.invoke.assert_called_once()


def test_selected_plan_does_not_consume_one_of_two_final_generation_slots(
    providers: Providers,
) -> None:
    workflow = gateway(
        providers, policy=WorkflowPolicy(max_cost_usd=10, max_model_calls=4)
    )
    workflow.complete("Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH)
    providers.research.invoke.return_value = response(json.dumps(STEP))
    workflow.complete("Read", {}, IssueResearchStep, LLMFlow.LEGAL_COMPOSITE_RESEARCH)
    with pytest.raises(RunStopped, match="budget exhausted"):
        workflow.complete(
            "Read again", {}, IssueResearchStep, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert workflow.budget.snapshot()["pending_final_calls"] == 2
    providers.selected.invoke.return_value = response(
        '{"answer":"Retained final allocation","unresolved_need_ids":[]}'
    )
    for flow, remaining in (
        (LLMFlow.LEGAL_COMPOSITE_ANSWER, 1),
        (LLMFlow.LEGAL_COMPOSITE_REVIEW, 0),
    ):
        workflow.complete("Reserved final generation", {}, DraftAnswer, flow, True)
        assert workflow.budget.snapshot()["pending_final_calls"] == remaining
    assert providers.selected.invoke.call_count == 3
    providers.research.invoke.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_required_original_and_cancellation_still_block_before_spend(
    providers: Providers, enabled: bool
) -> None:
    cancelled = False

    def check_active() -> None:
        if cancelled:
            raise RunStopped("Explicit cancellation")

    workflow = gateway(providers, enabled=enabled, check_active=check_active)
    with pytest.raises(MissingRequiredOriginal):
        workflow.complete(
            "Plan",
            {"required_evidence_numbers": [1]},
            IssueResearchPlan,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
    cancelled = True
    with pytest.raises(RunStopped, match="Explicit cancellation"):
        workflow.complete(
            "Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    providers.selected.invoke.assert_not_called()
    providers.research.invoke.assert_not_called()
    assert workflow.budget.snapshot()["model_calls"] == 0


@pytest.mark.parametrize("finalizing", [False, True])
def test_model_selection_does_not_bypass_flow_phase_consistency(
    providers: Providers, finalizing: bool
) -> None:
    workflow = gateway(providers)
    wrong_flow = (
        LLMFlow.LEGAL_COMPOSITE_RESEARCH
        if finalizing
        else LLMFlow.LEGAL_COMPOSITE_ANSWER
    )
    with pytest.raises(ValueError, match="flow and phase"):
        workflow.complete("Plan", {}, IssueResearchPlan, wrong_flow, finalizing)
    providers.selected.invoke.assert_not_called()
    providers.research.invoke.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["timeout", "schema", "truncated", "provider_error"]
)
def test_failed_selected_plan_has_no_retry_or_provider_fallback(
    providers: Providers, failure: str
) -> None:
    ledger, payload = original_payload()
    workflow = gateway(providers, ledger=ledger, preserve_timeout=True)
    if failure == "timeout":
        providers.selected.invoke.side_effect = LLMTimeoutError(
            RuntimeError("fake timeout")
        )
        expected: type[RunStopped] = ResearchPhaseClosed
    elif failure == "provider_error":
        providers.selected.invoke.side_effect = RuntimeError("fake provider failure")
        expected = RunStopped
    else:
        providers.selected.invoke.return_value = response(
            "{}" if failure == "schema" else json.dumps(PLAN),
            finish_reason="length" if failure == "truncated" else "stop",
        )
        expected = RunStopped
    with pytest.raises(expected) as stopped:
        workflow.complete(
            "Plan", payload, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    providers.selected.invoke.assert_called_once()
    providers.research.invoke.assert_not_called()
    if failure == "timeout":
        assert isinstance(stopped.value, ResearchPhaseClosed)
        assert stopped.value.reason == "provider_timeout"
        assert workflow.budget.snapshot()["research_stop_reason"] == "provider_timeout"
        assert workflow.budget.snapshot()["pending_final_calls"] == 2
        with pytest.raises(ResearchPhaseClosed):
            workflow.complete(
                "Do not start a reading",
                {},
                IssueReadingResponse,
                LLMFlow.LEGAL_COMPOSITE_RESEARCH,
            )
        with pytest.raises(ResearchPhaseClosed):
            workflow.research_proxy().invoke(
                UserMessage(content="Do not switch provider")
            )
        assert workflow.last_call_id is None and not workflow.last_delivered_citations
    if failure == "schema":
        assert workflow.last_delivered_citations == {1}
    else:
        assert ledger.export()["deliveries"] == []
    assert workflow.budget.snapshot()["model_calls"] == 1


@pytest.mark.parametrize("research_boundary", [False, True])
def test_selected_planner_host_timeout_closes_research_without_late_delivery(
    providers: Providers,
    monkeypatch: pytest.MonkeyPatch,
    research_boundary: bool,
) -> None:
    now = [0.0]
    ledger, payload = original_payload()
    workflow = BudgetedGateway(
        selected_llm=providers.selected,
        research_llm=providers.research,
        budget=WorkflowBudget(WorkflowPolicy(max_cost_usd=10), lambda: now[0]),
        ledger=ledger,
        use_selected_llm_for_initial_plan=True,
        preserve_research_finalization_on_timeout=True,
    )
    if research_boundary:
        now[0] = 70.0

    class TimedFuture(Future[ModelResponse]):
        def result(self, timeout: float | None = None) -> ModelResponse:
            assert timeout is not None
            now[0] += timeout
            raise FutureTimeout()

    shutdown = Mock()

    class Executor:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 1

        def submit(
            self, operation: Callable[..., ModelResponse], *args: object
        ) -> TimedFuture:
            future = TimedFuture()
            future.set_result(operation(*args))
            return future

        def shutdown(self, **kwargs: bool) -> None:
            shutdown(**kwargs)

    monkeypatch.setattr("onyx.legal_composite.gateway.ThreadPoolExecutor", Executor)
    with pytest.raises(ResearchPhaseClosed) as stopped:
        workflow.complete(
            "Plan", payload, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    expected = "host_research_deadline" if research_boundary else "host_call_timeout"
    assert stopped.value.reason == expected
    providers.selected.invoke.assert_called_once()
    providers.research.invoke.assert_not_called()
    assert providers.selected.invoke.call_args.kwargs["timeout_override"] == (
        10 if research_boundary else 45
    )
    assert ledger.export()["deliveries"] == []
    assert workflow.last_call_id is None and not workflow.last_delivered_citations
    retained = workflow.budget.snapshot()
    assert retained["research_stop_reason"] == expected
    assert retained["pending_final_calls"] == 2 and retained["unsettled_calls"] == 1
    assert retained["stop_reason"] is None
    shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    with pytest.raises(ResearchPhaseClosed):
        workflow.complete(
            "No second provider",
            {},
            IssueReadingResponse,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
    assert workflow.budget.snapshot() == retained
    providers.selected.invoke.assert_called_once()
    providers.research.invoke.assert_not_called()


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true", [], {}])
def test_initial_planner_selection_requires_literal_boolean(
    providers: Providers, invalid: object
) -> None:
    with pytest.raises(ValueError, match="explicit boolean"):
        BudgetedGateway(
            selected_llm=providers.selected,
            research_llm=providers.research,
            budget=WorkflowBudget(WorkflowPolicy(max_cost_usd=10)),
            ledger=EvidenceLedger(),
            use_selected_llm_for_initial_plan=cast(bool, invalid),
        )
    providers.selected.invoke.assert_not_called()
    providers.research.invoke.assert_not_called()
