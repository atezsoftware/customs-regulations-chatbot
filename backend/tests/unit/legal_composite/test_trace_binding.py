"""One paid auxiliary call has one priced span and an explicit proof binding."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.db.response_usage import get_response_usage
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import DraftAnswer, WorkflowPolicy
from onyx.llm.cost import ModelPrice
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.llm.usage_cost import price_generation
from onyx.tracing.answer_graph import _span_contents
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import trace
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.provider import DefaultTraceProvider
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> list[Span[GenerationSpanData]]:
    completed: list[Span[GenerationSpanData]] = []
    processor = Mock(spec=TracingProcessor)

    def collect(span: Span[Any]) -> None:
        if isinstance(span.span_data, GenerationSpanData):
            completed.append(span)

    processor.on_span_end.side_effect = collect
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    monkeypatch.setattr("onyx.tracing.framework.setup.GLOBAL_TRACE_PROVIDER", provider)
    return completed


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
    llm.invoke.return_value = ModelResponse(
        id="actual-provider-response",
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


def gateway(
    model: Mock, ledger: EvidenceLedger | Mock | None = None
) -> BudgetedGateway:
    return BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger if ledger is not None else EvidenceLedger(),
        run_id="canonical-run-id",
        scope={"forced_document_set": ["Türk hukuku"], "access_control_list": ["u:1"]},
    )


def test_auxiliary_fallback_is_one_priced_span_with_scope_and_response_binding(
    model: Mock, spans: list[Span[GenerationSpanData]], monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = gateway(model)
    with trace("auxiliary-fallback"):
        workflow.research_proxy().invoke(UserMessage(content="Select a legal section"))
    assert model.invoke.call_count == 1
    assert workflow.budget.snapshot()["model_calls"] == 1
    assert len(spans) == 1
    assert sum((span.span_data.usage or {})["input_tokens"] for span in spans) == 100
    captured, _, _, _, operation = _span_contents(spans[0])
    assert operation == LLMFlow.LEGAL_COMPOSITE_RESEARCH.value
    config = captured["model_config"]
    assert config["legal_composite_helper_flow"] == LLMFlow.UNTAGGED_INVOKE.value
    assert config["legal_composite_call_id"]
    assert config["legal_composite_run_id"] == "canonical-run-id"
    assert config["legal_composite_response_id"] == "actual-provider-response"
    scope_json = config["legal_composite_scope"]
    assert json.loads(scope_json) == {
        "forced_document_set": ["Türk hukuku"],
        "access_control_list": ["u:1"],
    }
    assert (
        config["legal_composite_scope_sha256"]
        == hashlib.sha256(scope_json.encode("utf-8")).hexdigest()
    )
    monkeypatch.setattr(
        "onyx.llm.cost_overrides.get_override",
        lambda *_args: Mock(
            input_cost_per_mtok=0.1,
            output_cost_per_mtok=0.5,
            cache_read_cost_per_mtok=None,
        ),
    )
    session = Mock()
    run_id, now = uuid4(), datetime.now(timezone.utc)
    generations = []
    for span in spans:
        _, _, _, attributes, operation = _span_contents(span)
        attributes["usage_cost"] = price_generation(
            attributes["model"], attributes["provider"], attributes["usage"], session
        ).model_dump(mode="json")
        generations.append((run_id, attributes, "COMPLETE", operation))
    session.execute.side_effect = [
        [(12, 0, run_id, "COMPLETE", "COMPLETE", now, now)],
        generations,
    ]
    summary = get_response_usage(session, [12], admin=True)[12]
    assert summary.calls == 1
    assert summary.total_cost_usd == pytest.approx(0.000015)
    assert summary.models[0].input_tokens == 100


@pytest.mark.parametrize(
    "flow",
    [
        LLMFlow.SEMANTIC_QUERY_REPHRASE,
        LLMFlow.KEYWORD_QUERY_EXPANSION,
        LLMFlow.SOURCE_FILTER_EXTRACTION,
        LLMFlow.TIME_FILTER_EXTRACTION,
        LLMFlow.CLASSIFY_SECTION_RELEVANCE,
        LLMFlow.SELECT_SECTIONS_FOR_EXPANSION,
    ],
)
def test_matching_search_helper_reuses_explicit_span_once(
    model: Mock, spans: list[Span[GenerationSpanData]], flow: LLMFlow
) -> None:
    workflow = gateway(model)
    proxy = workflow.research_proxy()
    helper = ScopedSearchLLM(proxy, RunContext(timeout_seconds=120), None)
    prompt = UserMessage(content="Select a legal section")
    with trace("explicit-helper"):
        with llm_generation_span(helper, flow, input_messages=[prompt]) as outer:
            response = helper.invoke(prompt)
            record_llm_response(outer, response)
    assert model.invoke.call_count == 1
    assert len(spans) == 1
    config = spans[0].span_data.model_config or {}
    assert config["flow"] == LLMFlow.LEGAL_COMPOSITE_RESEARCH.value
    assert config["legal_composite_helper_flow"] == flow.value
    assert spans[0].span_data.usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


@pytest.mark.parametrize(
    "mismatch", ["model", "provider", "input", "flow", "usage", "output"]
)
def test_unrelated_or_used_parent_is_never_retagged(
    model: Mock, spans: list[Span[GenerationSpanData]], mismatch: str
) -> None:
    workflow = gateway(model)
    parent_model = Mock(spec=LLM)
    parent_config = model.config.model_copy()
    if mismatch == "model":
        parent_config.model_name = "foreign-model"
    if mismatch == "provider":
        parent_config.model_provider = "foreign-provider"
    parent_model.config = parent_config
    prompt = UserMessage(content="Select a legal section")
    parent_prompt = (
        UserMessage(content="Genuine parent task") if mismatch == "input" else prompt
    )
    parent_flow = (
        LLMFlow.CHAT_RESPONSE if mismatch == "flow" else LLMFlow.UNTAGGED_INVOKE
    )
    with trace("unrelated-parent"):
        with llm_generation_span(
            parent_model, parent_flow, input_messages=[parent_prompt]
        ) as parent:
            if mismatch == "usage":
                parent.span_data.usage = {"input_tokens": 7, "output_tokens": 2}
            if mismatch == "output":
                parent.span_data.output = [{"role": "assistant", "content": "parent"}]
            original_config = dict(parent.span_data.model_config or {})
            workflow.research_proxy().invoke(prompt)
            assert parent.span_data.model_config == original_config
    assert len(spans) == 2
    child = next(span for span in spans if span.span_id != parent.span_id)
    assert child.parent_id == parent.span_id
    assert (child.span_data.model_config or {})[
        "flow"
    ] == LLMFlow.LEGAL_COMPOSITE_RESEARCH.value
    assert "legal_composite_call_id" not in (parent.span_data.model_config or {})


def test_deciding_span_call_id_matches_ledger_delivery(
    model: Mock, spans: list[Span[GenerationSpanData]]
) -> None:
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    workflow = gateway(model, ledger)
    records = [{"citation": 1, "text": "complete canonical original"}]
    with trace("deciding-call"):
        workflow.complete(
            "Use original authority",
            {"original_evidence": records},
            DraftAnswer,
            LLMFlow.LEGAL_COMPOSITE_ANSWER,
            finalizing=True,
        )
    assert len(spans) == 1
    captured, _, _, _, operation = _span_contents(spans[0])
    assert operation == LLMFlow.LEGAL_COMPOSITE_ANSWER.value
    call_id = captured["model_config"]["legal_composite_call_id"]
    assert call_id == workflow.last_call_id
    ledger.record_delivery.assert_called_once_with(
        call_id, LLMFlow.LEGAL_COMPOSITE_ANSWER.value, records
    )


@pytest.mark.parametrize("rejection", ["tools", "budget"])
def test_rejected_auxiliary_request_has_no_reservation_binding_or_usage(
    model: Mock, spans: list[Span[GenerationSpanData]], rejection: str
) -> None:
    workflow = gateway(model)
    if rejection == "budget":
        workflow.budget.stop("Budget exhausted")
    with trace("rejected-helper"):
        with pytest.raises(RunStopped):
            workflow.research_proxy().invoke(
                UserMessage(content="Select section"),
                tools=[{"type": "function"}] if rejection == "tools" else None,
            )
    assert workflow.budget.snapshot()["model_calls"] == 0
    model.invoke.assert_not_called()
    assert len(spans) == 1
    assert spans[0].span_data.usage is None
    assert "legal_composite_call_id" not in (spans[0].span_data.model_config or {})


@pytest.mark.parametrize("invocation", ["typed", "fallback", "explicit"])
@pytest.mark.parametrize("exception_type", [RuntimeError, LLMTimeoutError])
def test_provider_failure_marks_generation_and_redacts_private_error(
    model: Mock,
    spans: list[Span[GenerationSpanData]],
    invocation: str,
    exception_type: type[Exception],
) -> None:
    failure = exception_type(
        "provider failed api_key='private-value' Bearer private-bearer " + "x" * 600
    )
    model.invoke.side_effect = failure
    ledger = Mock(spec=EvidenceLedger)
    workflow = gateway(model, ledger)
    prompt = UserMessage(content="Select a legal section")
    with trace("failed-provider"):
        with pytest.raises(
            RunStopped,
            match="timed out"
            if exception_type is LLMTimeoutError
            else "invocation failed",
        ) as stopped:
            if invocation == "typed":
                workflow.complete(
                    "Use original authority",
                    {},
                    DraftAnswer,
                    LLMFlow.LEGAL_COMPOSITE_ANSWER,
                    True,
                )
            elif invocation == "fallback":
                workflow.research_proxy().invoke(prompt)
            else:
                helper = ScopedSearchLLM(
                    workflow.research_proxy(), RunContext(timeout_seconds=120), None
                )
                with llm_generation_span(
                    helper,
                    LLMFlow.CLASSIFY_SECTION_RELEVANCE,
                    input_messages=[prompt],
                ):
                    helper.invoke(prompt)
    assert stopped.value.__cause__ is failure
    assert model.invoke.call_count == 1
    assert len(spans) == 1
    span = spans[0]
    assert span.error is not None
    message = span.error["message"]
    assert "private-value" not in message and "private-bearer" not in message
    assert len(message) <= 550
    if invocation != "fallback":
        assert message.startswith(f"{exception_type.__name__}: provider failed")
        assert "[REDACTED]" in message
    assert span.span_data.output is None and span.span_data.usage is None
    captured, _, _, _, operation = _span_contents(span)
    assert operation == (
        LLMFlow.LEGAL_COMPOSITE_ANSWER.value
        if invocation == "typed"
        else LLMFlow.LEGAL_COMPOSITE_RESEARCH.value
    )
    config = captured["model_config"]
    assert float(config["legal_composite_allocated_call_seconds"]) == 30
    assert config["legal_composite_transport_timeout_seconds"] == "30"
    assert config["legal_composite_compat_attempt_bound"] == "3"
    assert config["legal_composite_call_id"]
    assert "legal_composite_response_id" not in config
    ledger.record_delivery.assert_not_called()
    assert workflow.budget.snapshot()["unsettled_calls"] == 1
