import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine, SourceAcquirer
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import (
    DraftAnswer,
    IssueResearchPlan,
    IssueResearchStep,
    ResearchPlan,
    ResearchStep,
    WorkflowPolicy,
)
from onyx.llm.cost import ModelPrice
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import (
    ImageContentPart,
    ImageUrlDetail,
    ReasoningEffort,
    UserMessage,
)
from onyx.llm.utils import check_number_of_tokens
from onyx.regulatory.structured_llm import _portable_structured_output_schema
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.test_review_assessment import original


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
    assert kwargs["timeout_override"] == 45
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


def test_final_phase_navigation_cannot_crowd_out_whole_originals_under_same_cap(
    model: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.asv3.models import RunContext

    monkeypatch.setattr(
        "onyx.legal_composite.gateway.check_number_of_tokens", check_number_of_tokens
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                f"Complete operative provision {index}. "
                "Its condition, exception and legal scope remain unchanged. " * 4,
                f"canonical-chunk-{index}",
                metadata={
                    "title": "Source navigation title " + "jurisdiction " * 40,
                    "heading_path": [
                        f"control-{part}-jurisdiction-navigational-reference"
                        for part in range(20)
                    ],
                },
            )
            for index in range(63)
        ],
        RunContext(),
    )
    policy = WorkflowPolicy()
    budget = WorkflowBudget(policy)
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    acquirer = Mock(spec=SourceAcquirer)
    acquirer.definitions.return_value = [
        {"name": "read_evidence", "description": "Navigation control " * 2_000}
    ]
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=policy,
        check_active=lambda: None,
        research_available=lambda: False,
    )
    receipt: dict[str, JsonValue] = {
        "need_ids": ["general-need"],
        "status": "completed",
        "citations": cast(JsonValue, ledger.citation_numbers()),
        "data": {"navigation": "Source lookup control " * 2_000},
        "summary": "Navigation is not legal authority",
    }
    engine.receipts = [receipt]
    research = engine._payload("Apply complete original provisions", "")
    final = engine._payload(
        "Apply complete original provisions", "", source_phase=False
    )
    catalogue = research["original_catalogue"]
    if isinstance(catalogue, dict):
        assert catalogue["codec"] == "shared_metadata_v1"
        rows = catalogue["rows"]
        assert isinstance(rows, list) and len(rows) == 63
    else:
        assert isinstance(catalogue, list) and len(catalogue) == 63
    assert research["original_evidence"] == final["original_evidence"]
    assert "original_catalogue" not in final and "tools" not in final
    assert policy.max_context_tokens == 32_000
    try:
        gateway.complete(
            "Answer", research, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
        baseline = json.loads(model.invoke.call_args.args[0][1].content)[
            "original_evidence"
        ]
    except RunStopped as error:
        assert "protocol exceed" in str(error)
        baseline = []
    gateway.complete("Answer", final, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True)
    sent = json.loads(model.invoke.call_args.args[0][1].content)
    records = sent["original_evidence"]
    assert len(records) >= max(1, len(baseline))
    assert records == final["original_evidence"]
    final_records = cast(list[dict[str, JsonValue]], final["original_evidence"])
    expected = {cast(int, row["citation"]): row for row in final_records}
    for row in records:
        assert row == expected[row["citation"]]
        canonical = ledger.get(row["citation"])
        assert canonical is not None
        assert (
            hashlib.sha256(row["text"].encode()).hexdigest()
            == hashlib.sha256(canonical.text.encode()).hexdigest()
        )
        assert "Navigation" not in row["text"]
    assert set(sent["omitted_original_ids"]) == set(ledger.citation_numbers()) - {
        row["citation"] for row in records
    }
    assert gateway.last_delivered_citations == {row["citation"] for row in records}


@pytest.fixture
def constrained_research_model(model: Mock, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.get_model_price_per_million",
        lambda *_args: ModelPrice(
            model="price-test-model",
            provider="price-test",
            input_per_mtok=1,
            output_per_mtok=1,
            cache_per_mtok=None,
        ),
    )
    monkeypatch.setattr("litellm.get_model_info", lambda **_kwargs: {})
    model.invoke.return_value.choice.message.content = (
        '{"actions":[],"ready_to_answer":false,"remaining_gaps":["Unread original"]}'
    )
    return model


def test_research_fits_remaining_cost_with_whole_originals_and_complete_navigation(
    constrained_research_model: Mock,
) -> None:
    from onyx.asv3.models import RunContext

    model = constrained_research_model
    ledger = EvidenceLedger()
    ledger.add(
        [
            original("Complete operative condition. " * 300, f"chunk-{index}")
            for index in range(10)
        ],
        RunContext(),
    )
    records = json.loads(ledger.serialize_records(list(ledger.citation_numbers())))
    catalogue = [
        {key: row[key] for key in ("citation", "source_id", "chunk_id", "metadata")}
        for row in records
    ]
    payload: dict[str, JsonValue] = {
        "original_evidence": records,
        "required_evidence_numbers": [1],
        "original_catalogue": catalogue,
    }
    budget = WorkflowBudget(WorkflowPolicy())
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    full_schema = ResearchStep.model_json_schema()
    response_format: dict[str, JsonValue] = {
        "type": "json_schema",
        "json_schema": {
            "name": "ResearchStep",
            "schema": _portable_structured_output_schema(full_schema),
            "strict": False,
        },
    }
    _, context_only_tokens, _ = gateway._fit_messages(
        "Select a missing original",
        payload,
        json.dumps(full_schema, ensure_ascii=False),
        model,
        response_format,
        finalizing=True,
    )
    affordable = budget.affordable_input_tokens(2_048, 1, 1)
    assert affordable < context_only_tokens <= budget.policy.max_context_tokens
    with pytest.raises(RunStopped, match="finalization allocation"):
        budget.request(context_only_tokens, 2_048, 1, 1)
    gateway.complete(
        "Select a missing original",
        payload,
        ResearchStep,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    sent = json.loads(model.invoke.call_args.args[0][1].content)
    delivered = sent["original_evidence"]
    assert 0 < len(delivered) < len(records)
    assert 1 in {row["citation"] for row in delivered}
    expected = {row["citation"]: row for row in records}
    assert all(row == expected[row["citation"]] for row in delivered)
    assert sent["original_catalogue"] == catalogue
    assert set(sent["omitted_original_ids"]) == set(expected) - {
        row["citation"] for row in delivered
    }
    assert payload["original_evidence"] == records
    assert ledger.citation_numbers() == tuple(range(1, 11))
    assert gateway.last_delivered_citations == {row["citation"] for row in delivered}
    assert budget.snapshot()["pending_final_calls"] == 2
    model.invoke.assert_called_once()


@pytest.mark.parametrize("required", [False, True])
def test_unaffordable_research_protocol_or_required_original_fails_without_spend(
    constrained_research_model: Mock, required: bool
) -> None:
    model = constrained_research_model
    budget = WorkflowBudget(WorkflowPolicy())
    ledger = Mock(spec=EvidenceLedger)
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    payload: dict[str, JsonValue] = (
        {
            "original_evidence": [
                {"citation": 1, "text": "Original condition. " * 8_000}
            ],
            "required_evidence_numbers": [1],
        }
        if required
        else {"original_catalogue": {"navigation": "Source identity. " * 8_000}}
    )
    before = budget.snapshot()
    with pytest.raises(RunStopped, match="remaining research budget"):
        gateway.complete(
            "Select a missing original",
            payload,
            ResearchStep,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
    assert budget.snapshot()["model_calls"] == before["model_calls"] == 0
    ledger.record_delivery.assert_not_called()
    model.invoke.assert_not_called()


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


def test_raw_provider_output_limit_cannot_deliver_a_valid_json_prefix(
    model: Mock,
) -> None:
    model.invoke.return_value.choice.finish_reason = "MAX_OUTPUT_TOKENS"
    ledger = Mock(spec=EvidenceLedger)
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
    )
    with pytest.raises(RunStopped, match="truncated"):
        gateway.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    model.invoke.assert_called_once()
    ledger.record_delivery.assert_not_called()
    assert gateway.last_call_id is None


def _long_issue_plan() -> str:
    return json.dumps(
        {
            "language": "tr",
            "requires_sources": True,
            "needs": [
                {
                    "need_id": f"requested-outcome-{index}",
                    "question": f"What rules govern requested outcome {index} under the supplied circumstances?",
                    "governing_source": "To be established from applicable originals",
                    "conditions_to_check": [
                        "Which eligibility conditions affect this requested outcome?",
                        "Which supplied dates matter to the applicable rule?",
                        "Which procedure and supporting documents are required?",
                        "Which exceptions change the result for these facts?",
                    ],
                    "required_outcome": "Explain availability, material conditions, the procedure, the deadline and how the supplied facts affect the conclusion.",
                    "research_dimensions": [
                        "legal basis",
                        "eligibility",
                        "procedure",
                        "deadline",
                        "proof",
                        "exceptions",
                    ],
                    "relevant_facts": [
                        f"The user asks about requested outcome {index} separately from the other alternatives and requires its own conclusion.",
                        "The question supplies an earlier event date and a later discovery date; neither user date establishes a legal deadline.",
                        "Some supporting documents exist while the remaining documents have not been identified; the legal proof requirement remains unknown.",
                    ],
                    "evidence_gaps": [],
                    "evidence_gap_resolutions": [],
                }
                for index in range(8)
            ],
            "discovery_query": "Requested outcomes eligibility procedure deadline proof exceptions",
            "initial_actions": [
                {
                    "need_ids": [f"requested-outcome-{index}"],
                    "tool": "search",
                    "arguments": {
                        "query": f"requested outcome {index} applicable rules and procedure"
                    },
                    "source_kind": None,
                }
                for index in range(8)
            ],
            "missing_user_facts": [],
        },
        ensure_ascii=False,
    )


def test_lc_long_issue_plan_receives_capacity_without_losing_issues_or_actions(
    model: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _long_issue_plan()
    actual_output_tokens = check_number_of_tokens(content)
    assert 2_048 < actual_output_tokens < 6_144
    expected = IssueResearchPlan.model_validate_json(content, strict=True)
    returned = model.invoke.return_value.model_copy(deep=True)
    returned.choice.message.content = content
    returned.usage.completion_tokens = actual_output_tokens
    returned.usage.total_tokens = returned.usage.prompt_tokens + actual_output_tokens
    observed_caps: list[int] = []

    def capacity_limited_provider(*_args: Any, **kwargs: Any) -> ModelResponse:
        cap = kwargs["max_tokens"]
        observed_caps.append(cap)
        response = returned.model_copy(deep=True)
        if cap < actual_output_tokens:
            response.choice.message.content = content[
                : len(content) * cap // actual_output_tokens
            ]
            response.choice.finish_reason = "length"
            response.usage.completion_tokens = cap
            response.usage.total_tokens = response.usage.prompt_tokens + cap
        return response

    model.invoke.side_effect = capacity_limited_provider
    budget = WorkflowBudget(WorkflowPolicy())
    reserve = Mock(wraps=budget.request)
    monkeypatch.setattr(budget, "request", reserve)
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=EvidenceLedger()
    )
    result = gateway.complete(
        "Enumerate the independent issues without guessing law",
        {"request": "Investigate the supplied alternatives"},
        IssueResearchPlan,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    assert result == expected
    assert len(result.needs) == len(result.initial_actions) == 8
    assert observed_caps == [6_144]
    assert reserve.call_args.args[1] == 6_144
    assert model.invoke.call_args.kwargs["timeout_override"] <= 45
    model.invoke.assert_called_once()


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "MAX_OUTPUT_TOKENS"])
def test_lc_issue_plan_rejects_truncation_before_delivery_even_for_valid_json(
    finish_reason: str, model: Mock
) -> None:
    model.invoke.return_value.choice.message.content = _long_issue_plan()
    model.invoke.return_value.choice.finish_reason = finish_reason
    ledger = Mock(spec=EvidenceLedger)
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
    )
    with pytest.raises(RunStopped, match="truncated"):
        gateway.complete(
            "Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    model.invoke.assert_called_once()
    assert model.invoke.call_args.kwargs["max_tokens"] == 6_144
    ledger.record_delivery.assert_not_called()
    assert gateway.last_call_id is None and not gateway.last_delivered_citations


def test_lc_issue_plan_rejects_incomplete_json_without_retry(model: Mock) -> None:
    model.invoke.return_value.choice.message.content = _long_issue_plan()[:-20]
    model.invoke.return_value.choice.finish_reason = "stop"
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
    )
    with pytest.raises(RunStopped, match="workflow schema"):
        gateway.complete(
            "Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    model.invoke.assert_called_once()


@pytest.mark.parametrize("limited_resource", ["output_tokens", "cost"])
def test_lc_issue_plan_capacity_cannot_consume_finalization_reserves(
    limited_resource: str, model: Mock
) -> None:
    policy = (
        WorkflowPolicy(max_output_tokens=10_240)
        if limited_resource == "output_tokens"
        else WorkflowPolicy(max_cost_usd=0.012)
    )
    budget = WorkflowBudget(policy)
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=EvidenceLedger()
    )
    with pytest.raises(RunStopped, match="budget exhausted"):
        gateway.complete(
            "Plan", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    model.invoke.assert_not_called()
    assert budget.snapshot()["model_calls"] == 0


@pytest.mark.parametrize(
    "response_type,content,capacity",
    [
        (
            ResearchPlan,
            '{"language":"tr","requires_sources":true,"needs":[{"need_id":"one","question":"What applies?","governing_source":"Unknown","conditions_to_check":[]}],"initial_actions":[],"missing_user_facts":[]}',
            2_048,
        ),
        (
            ResearchStep,
            '{"actions":[],"ready_to_answer":false,"remaining_gaps":[]}',
            2_048,
        ),
        (
            IssueResearchStep,
            '{"actions":[],"ready_to_answer":false,"remaining_gaps":[]}',
            6_144,
        ),
    ],
)
def test_lc_planning_allowance_preserves_existing_research_class_capacities(
    response_type: type[ResearchPlan] | type[ResearchStep],
    content: str,
    capacity: int,
    model: Mock,
) -> None:
    model.invoke.return_value.choice.message.content = content
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
    )
    result = gateway.complete(
        "Research", {}, response_type, LLMFlow.LEGAL_COMPOSITE_RESEARCH
    )
    assert isinstance(result, response_type)
    assert model.invoke.call_args.kwargs["max_tokens"] == capacity


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


@pytest.mark.parametrize("setup_seconds", [7, 8])
def test_executor_setup_consumes_reserved_time_before_any_provider_admission(
    model: Mock, monkeypatch: pytest.MonkeyPatch, setup_seconds: int
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: now[0], deadline=120)
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=ledger
    )
    response = model.invoke.return_value

    def prepare(*, max_workers: int) -> ThreadPoolExecutor:
        now[0] += setup_seconds
        return ThreadPoolExecutor(max_workers=max_workers)

    def finish(*_args: object, **_kwargs: object) -> ModelResponse:
        now[0] = 119
        return response

    monkeypatch.setattr("onyx.legal_composite.gateway.ThreadPoolExecutor", prepare)
    model.invoke.side_effect = finish
    now[0] = 110
    if setup_seconds == 7:
        result = gateway.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
        assert result.answer == "supported"
        assert model.invoke.call_args.kwargs["timeout_override"] == 3
        ledger.record_delivery.assert_called_once()
    else:
        with pytest.raises(RunStopped, match="deadline"):
            gateway.complete(
                "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
            )
        model.invoke.assert_not_called()
        ledger.record_delivery.assert_not_called()
    assert budget.deadline == 120


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
    assert kwargs["timeout_override"] == 6
    assert kwargs["use_streaming"] is False
    assert kwargs["structured_response_format"] == {"type": "json_object"}


def test_auxiliary_one_second_transport_is_admitted_as_one_call(model: Mock) -> None:
    budget = WorkflowBudget(WorkflowPolicy())
    gateway = BudgetedGateway(
        selected_llm=model, research_llm=model, budget=budget, ledger=EvidenceLedger()
    )
    gateway.research_proxy().invoke(
        UserMessage(content="Select section"), timeout_override=1
    )
    model.invoke.assert_called_once()
    assert model.invoke.call_args.kwargs["timeout_override"] == 1
    assert budget.snapshot()["model_calls"] == 1


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
        proxy.invoke(prompt, timeout_override=0)
    with pytest.raises(ValueError, match="timeout"):
        proxy.invoke(prompt, timeout_override=True)
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
