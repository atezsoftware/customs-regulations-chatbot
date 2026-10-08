"""Exercise the canonical non-streaming adapter without provider network calls."""

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from copy import deepcopy
from threading import Event
from typing import Any
from unittest.mock import Mock

import litellm
import pytest
from litellm.exceptions import BadRequestError
from litellm.exceptions import Timeout as LiteLLMTimeout
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine, SourceAcquirer
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import (
    AnswerReview,
    DraftAnswer,
    ResearchPlan,
    ResearchStep,
    WorkflowPolicy,
)
from onyx.llm.cost import ModelPrice
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM, LLMTimeoutError
from onyx.tracing.answer_graph import _span_contents
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import trace
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.provider import DefaultTraceProvider
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span


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
    *,
    policy: WorkflowPolicy | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
) -> BudgetedGateway:
    policy = policy or WorkflowPolicy()
    budget = WorkflowBudget(policy) if clock is None else WorkflowBudget(policy, clock)
    return BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=budget,
        ledger=ledger,
        reasoning_effort=reasoning_effort,
    )


@pytest.mark.parametrize("compatibility_attempts", [None, 1])
def test_scoped_search_proxy_honors_canonical_compatibility_attempt_bound(
    model: LitellmLLM,
    monkeypatch: pytest.MonkeyPatch,
    compatibility_attempts: int | None,
) -> None:
    def complete(**kwargs: Any) -> litellm.ModelResponse:
        if "reasoning_effort" in kwargs:
            raise BadRequestError(
                "reasoning_effort unsupported", "gemini-3.8-flash", "vertex_ai"
            )
        return response("Selected relevant section")

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    workflow = gateway(model, EvidenceLedger())
    helper = ScopedSearchLLM(
        workflow.research_proxy(), RunContext(timeout_seconds=float("inf")), None
    )
    if compatibility_attempts == 1:
        with pytest.raises(RunStopped, match="invocation failed"):
            helper.invoke(
                UserMessage(content="Select a relevant original"),
                provider_compatibility_attempts=compatibility_attempts,
            )
        completion.assert_called_once()
        assert workflow.budget.snapshot()["unsettled_calls"] == 1
    else:
        result = helper.invoke(
            UserMessage(content="Select a relevant original"),
            provider_compatibility_attempts=compatibility_attempts,
        )
        assert result.choice.message.content == "Selected relevant section"
        assert completion.call_count == 2
        assert workflow.budget.snapshot()["unsettled_calls"] == 0
    assert workflow.budget.snapshot()["model_calls"] == 1


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
    workflow = gateway(model, ledger, policy=WorkflowPolicy(max_call_seconds=30))
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


def test_admitted_window_accepts_thirty_seven_second_canonical_transport(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    workflow = gateway(model, ledger, lambda: now[0])
    now[0] = 46.0

    def complete(**kwargs: Any) -> litellm.ModelResponse:
        if kwargs["timeout"] < 37:
            raise LiteLLMTimeout(
                "inference needs 37 seconds", "gemini-3.8-flash", "vertex_ai"
            )
        now[0] += 37
        return response('{"answer":"supported","unresolved_need_ids":[]}')

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    result = workflow.complete(
        "Use full original law",
        {"original_evidence": [{"citation": 1, "text": "complete original"}]},
        DraftAnswer,
        LLMFlow.LEGAL_COMPOSITE_ANSWER,
        True,
    )
    assert result.answer == "supported"
    completion.assert_called_once()
    assert completion.call_args.kwargs["timeout"] == 45
    assert completion.call_args.kwargs["max_tokens"] == 4_096
    assert workflow.budget.remaining_seconds(finalizing=True) == 37
    assert workflow.budget.deadline == 120
    assert workflow.budget.snapshot()["unsettled_calls"] == 0
    ledger.record_delivery.assert_called_once()


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
    assert completion.call_args.kwargs["timeout"] == 45
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None
    assert (
        workflow.budget.snapshot()["estimated_cost_usd"]
        == retained["estimated_cost_usd"]
    )


def test_returned_canonical_review_keeps_the_last_seconds_for_validation(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    workflow = gateway(model, ledger, lambda: now[0])
    now[0] = 93.15252
    content = (
        '{"request_coverage_complete":true,"material_claims_supported":true,'
        '"counter_authority_checked":true,"needs":[],"defects":[],"repair_actions":[]}'
    )

    def complete(**_kwargs: Any) -> litellm.ModelResponse:
        now[0] += 25.1789
        return response(content)

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    result = workflow.complete(
        "Review exact originals",
        {"original_evidence": [{"citation": 1, "text": "complete original"}]},
        AnswerReview,
        LLMFlow.LEGAL_COMPOSITE_REVIEW,
        True,
    )
    assert result == AnswerReview.model_validate_json(content, strict=True)
    assert workflow.budget.remaining_seconds(finalizing=True) == pytest.approx(1.66858)
    ledger.record_delivery.assert_called_once()
    retained = workflow.budget.snapshot()
    with pytest.raises(RunStopped, match="deadline"):
        workflow.complete(
            "New answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    completion.assert_called_once()
    assert workflow.budget.snapshot() == retained


@pytest.mark.parametrize("returned_at", [120.0, 120.1])
def test_remaining_workflow_deadline_rejects_late_original_delivery(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch, returned_at: float
) -> None:
    now = [0.0]
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger, lambda: now[0])
    now[0] = 116.0

    def late(**_kwargs: Any) -> litellm.ModelResponse:
        now[0] = returned_at
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


def test_late_canonical_compatibility_retry_cannot_republish_closed_capture(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, released, finished = Event(), Event(), Event()
    futures: list[Future[Any]] = []
    captures: list[Any] = []
    closed: list[Span[GenerationSpanData]] = []
    processor = Mock(spec=TracingProcessor)

    def capture(span: Span[Any]) -> None:
        if isinstance(span.span_data, GenerationSpanData):
            captures.append(deepcopy(_span_contents(span)))
            closed.append(span)

    processor.on_span_end.side_effect = capture
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    monkeypatch.setattr("onyx.tracing.framework.setup.GLOBAL_TRACE_PROVIDER", provider)
    attempts: list[dict[str, Any]] = []

    def complete(**kwargs: Any) -> litellm.ModelResponse:
        attempts.append(kwargs)
        if len(attempts) == 1:
            started.set()
            assert released.wait(timeout=5)
            raise BadRequestError(
                "reasoning_effort unsupported", "gemini-3.8-flash", "vertex_ai"
            )
        assert "reasoning_effort" not in kwargs
        return response('{"answer":"late","unresolved_need_ids":[]}')

    def host_timeout(future: Future[Any], timeout: float | None = None) -> Any:
        assert timeout is not None and 0 < timeout <= 45
        assert started.wait(timeout=2)
        futures.append(future)
        raise FutureTimeout()

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr(Future, "result", host_timeout)
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger)
    try:
        with trace("late-canonical-compatibility"):
            with pytest.raises(RunStopped, match="timed out"):
                workflow.complete(
                    "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
                )
            assert len(captures) == 1 and closed[0].ended_at is not None
        recorded = deepcopy(captures[0])
        assert recorded[0]["request_params"]["sent_kwargs"]["reasoning_effort"] == (
            "medium"
        )
        assert recorded[1] is None and recorded[3]["usage"] is None
        retained = workflow.budget.snapshot()
        error = deepcopy(closed[0].error)
        futures[0].add_done_callback(lambda _future: finished.set())
    finally:
        released.set()
        if futures:
            assert finished.wait(timeout=2)
    completion.assert_called()
    assert completion.call_count == 2
    assert captures == [recorded]
    processor.on_span_end.assert_called_once()
    # Canonical request-parameter writes may mutate the live object after capture.
    assert (
        "reasoning_effort"
        not in (closed[0].span_data.request_params or {})["sent_kwargs"]
    )
    assert closed[0].span_data.output is None and closed[0].span_data.usage is None
    assert closed[0].error == error
    assert "legal_composite_response_id" not in (closed[0].span_data.model_config or {})
    assert (
        workflow.budget.snapshot()["estimated_cost_usd"]
        == retained["estimated_cost_usd"]
    )
    assert workflow.budget.snapshot()["unsettled_calls"] == 1
    ledger.record_delivery.assert_not_called()


@pytest.mark.parametrize(
    "selected_effort", [ReasoningEffort.AUTO, ReasoningEffort.HIGH]
)
@pytest.mark.parametrize(
    "response_type,content,required",
    [
        (
            DraftAnswer,
            '{"answer":"supported","unresolved_need_ids":[]}',
            ["answer", "unresolved_need_ids"],
        ),
        (
            ResearchPlan,
            '{"language":"tr","requires_sources":true,"needs":[{"need_id":"generic",'
            '"question":"Which rule applies?","governing_source":"Applicable law",'
            '"conditions_to_check":[]}],"initial_actions":[],"missing_user_facts":[]}',
            [
                "language",
                "requires_sources",
                "needs",
                "initial_actions",
                "missing_user_facts",
            ],
        ),
        (
            ResearchStep,
            '{"actions":[{"need_ids":["generic"],"tool":"read_evidence",'
            '"arguments":{"citations":[1],"options":{"value":null}}}],'
            '"ready_to_answer":false,"remaining_gaps":[]}',
            ["actions", "ready_to_answer", "remaining_gaps"],
        ),
        (
            AnswerReview,
            '{"request_coverage_complete":false,"material_claims_supported":false,'
            '"counter_authority_checked":false,"needs":[],"defects":["Original missing"],'
            '"repair_actions":[{"need_ids":["generic"],"tool":"read_evidence",'
            '"arguments":{"citations":[1]}}]}',
            [
                "request_coverage_complete",
                "material_claims_supported",
                "counter_authority_checked",
                "needs",
                "defects",
                "repair_actions",
            ],
        ),
    ],
)
def test_canonical_typed_schema_and_phase_reasoning_map_to_native_vertex(
    model: LitellmLLM,
    monkeypatch: pytest.MonkeyPatch,
    response_type: type[BaseModel],
    content: str,
    required: list[str],
    selected_effort: ReasoningEffort,
) -> None:
    def complete(**kwargs: Any) -> litellm.ModelResponse:
        native: dict[str, Any] = {}
        VertexGeminiConfig().apply_response_schema_transformation(
            kwargs["response_format"], native, "gemini-3.8-flash"
        )
        assert native["response_mime_type"] == "application/json"
        grammar = native["response_json_schema"]
        assert grammar["required"] == required
        assert grammar["additionalProperties"] is False
        if response_type is DraftAnswer:
            assert grammar["properties"]["answer"]["type"] == "string"
            assert grammar["properties"]["unresolved_need_ids"]["type"] == "array"
        else:
            assert "SourceAction" in grammar["$defs"]
            assert "JsonValue" in grammar["$defs"]
        assert kwargs["tools"] is None
        expected_effort = (
            ("medium" if selected_effort is ReasoningEffort.AUTO else "high")
            if response_type is DraftAnswer
            else "low"
        )
        assert kwargs["reasoning_effort"] == expected_effort
        assert kwargs["max_tokens"] == (
            2_048 if response_type in {ResearchPlan, ResearchStep} else 4_096
        )
        return response(content)

    completion = Mock(side_effect=complete)
    monkeypatch.setattr("litellm.completion", completion)
    workflow = gateway(model, EvidenceLedger(), reasoning_effort=selected_effort)
    research = response_type in {ResearchPlan, ResearchStep}
    result = workflow.complete(
        "Return the typed result",
        {},
        response_type,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH
        if research
        else LLMFlow.LEGAL_COMPOSITE_REVIEW
        if response_type is AnswerReview
        else LLMFlow.LEGAL_COMPOSITE_ANSWER,
        not research,
    )
    assert isinstance(result, response_type)
    assert result == response_type.model_validate_json(content, strict=True)
    completion.assert_called_once()


@pytest.mark.parametrize(
    "content", [None, '{"answer":"supported","unresolved_need_ids":[]}']
)
def test_canonical_undeclared_tool_result_never_becomes_a_typed_answer(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    returned = litellm.ModelResponse(
        id="undeclared-tool-result",
        created=1,
        model="gemini-3.8-flash",
        choices=[
            litellm.Choices(
                message=litellm.Message(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="undeclared",
                            type="function",
                            function=Function(
                                name="read_evidence", arguments='{"citation":1}'
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                index=0,
            )
        ],
    )
    completion = Mock(return_value=returned)
    monkeypatch.setattr("litellm.completion", completion)
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger)
    with pytest.raises(RunStopped, match="undeclared tool call"):
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    completion.assert_called_once()
    assert completion.call_args.kwargs["tools"] is None
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None and not workflow.last_delivered_citations
    assert workflow.budget.snapshot()["model_calls"] == 1


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_valid_json_from_truncated_canonical_response_cannot_publish(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    returned = litellm.ModelResponse(
        id="truncated-result",
        created=1,
        model="gemini-3.8-flash",
        choices=[
            litellm.Choices(
                message=litellm.Message(
                    role="assistant",
                    content='{"answer":"valid JSON but incomplete generation","unresolved_need_ids":[]}',
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
    )
    completion = Mock(return_value=returned)
    monkeypatch.setattr("litellm.completion", completion)
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger)
    with pytest.raises(RunStopped, match="truncated"):
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    completion.assert_called_once()
    ledger.record_delivery.assert_not_called()
    assert workflow.last_call_id is None


def test_portable_native_schema_keeps_strict_host_validation(
    model: LitellmLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = Mock(return_value=response('{"answer":"","unresolved_need_ids":[]}'))
    monkeypatch.setattr("litellm.completion", completion)
    workflow = gateway(model, EvidenceLedger())
    with pytest.raises(RunStopped, match="workflow schema"):
        workflow.complete(
            "Answer", {}, DraftAnswer, LLMFlow.LEGAL_COMPOSITE_ANSWER, True
        )
    completion.assert_called_once()
    portable = completion.call_args.kwargs["response_format"]["json_schema"]["schema"]
    assert "minLength" not in portable["properties"]["answer"]
