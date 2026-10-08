"""Native Decisions wire contracts and conservative source filtering fences."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from threading import Event
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.decisions import DecisionsClassifier
from onyx.legal_composite.models import ResearchNeed, ResearchPlan, WorkflowPolicy
from onyx.legal_composite.selection import SourceCandidate, SourceSelectionRequest
from onyx.llm.interfaces import LLMConfig
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import trace
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.provider import DefaultTraceProvider
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span

ROLES = ("relevant", "irrelevant", "uncertain")


def candidate(
    citation: int, text: str = "A prerequisite applies to this procedure."
) -> SourceCandidate:
    return SourceCandidate(
        citation=citation,
        source_id=f"source-{citation}",
        chunk_id=f"chunk-{citation}",
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
        metadata={"source_kind": "legislation"},
    )


def selection_request(
    *candidates: SourceCandidate, needs: int = 1
) -> SourceSelectionRequest:
    return SourceSelectionRequest(
        question="What procedure and prerequisites apply?",
        plan=ResearchPlan(
            language="en",
            requires_sources=True,
            needs=[
                ResearchNeed(
                    need_id=f"need-{index}",
                    question=f"Question {index}",
                    governing_source="Statutory rule",
                    conditions_to_check=["prerequisite"],
                )
                for index in range(needs)
            ],
            initial_actions=[],
            missing_user_facts=[],
        ),
        candidates=list(candidates) or [candidate(1)],
    )


def choice(
    role: str = "relevant", probability: float = 0.99, confidence: float = 0.99
) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": role,
        "confidence": confidence,
        "probabilities": {
            key: probability if key == role else (1 - probability) / 2 for key in ROLES
        },
    }


def classifier(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    provider: str = "openai",
    policy: WorkflowPolicy | None = None,
    candidates: list[SourceCandidate] | None = None,
    **kwargs: Any,
) -> DecisionsClassifier:
    config = LLMConfig(
        model_provider=provider,
        model_name="gpt-6-luna" if provider == "openai" else "typesafe/jev-1.13",
        temperature=1,
        api_key="test-secret-not-real",
        max_input_tokens=32_000,
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id=item.source_id, chunk_id=item.chunk_id, text=item.text
            )
            for item in (candidates or [candidate(1)])
        ],
        RunContext(),
    )
    return DecisionsClassifier(
        config=config,
        budget=WorkflowBudget(policy or WorkflowPolicy()),
        ledger=ledger,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def response_body(
    payload: dict[str, Any], role: str = "relevant", *, provider: str = "openai"
) -> dict[str, Any]:
    if provider == "openai":
        answers = [
            {
                "type": "predicate",
                "name": item["name"],
                "probability": 0.01 if role == "irrelevant" else 0.99,
            }
            for item in payload["questions"]
        ]
    else:
        answers = {name: choice(role) for name in payload["questions"]}
    return {
        "model": payload["model"],
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 0},
        "id": "decision-response-id",
    }


@pytest.mark.parametrize(
    "provider,endpoint",
    [
        ("openai", "https://api.openai.com/v1/decisions"),
        ("openrouter", "https://openrouter.ai/api/alpha/decisions"),
    ],
)
def test_native_batched_contract_preserves_every_original_and_need(
    provider: str, endpoint: str
) -> None:
    observed: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == endpoint
        assert request.headers["Authorization"] == "Bearer test-secret-not-real"
        payload = json.loads(request.content)
        observed.append(payload)
        state = (
            json.loads(payload["input"]) if provider == "openai" else payload["state"]
        )
        assert state["candidates"]["1"]["text"] == first.text
        assert state["candidates"]["2"]["text_hash"] == second.text_hash
        assert len(payload["questions"]) == 4
        assert not {
            "temperature",
            "reasoning",
            "max_tokens",
            "response_format",
        }.intersection(payload)
        return httpx.Response(200, json=response_body(payload, provider=provider))

    first, second = (
        candidate(1),
        candidate(2, "The complete exception to this condition."),
    )
    workflow = classifier(handler, provider=provider, candidates=[first, second])
    result = workflow.classify(selection_request(first, second, needs=2))
    assert result.failure is None
    assert result.delivered_citations == [1, 2]
    assert result.decision is not None
    assert [row.relevant for row in result.decision.needs] == [[1, 2], [1, 2]]
    assert len(observed) == 1
    assert workflow.budget.snapshot()["model_calls"] == 1
    assert workflow.budget.snapshot()["output_tokens"] == 0
    assert workflow.budget.snapshot()["estimated_cost_usd"] == (
        0.00001 if provider == "openai" else 0.0000042
    )


@pytest.mark.parametrize(
    "role,probability,confidence,expected",
    [
        ("irrelevant", 0.99, 0.99, "irrelevant"),
        ("irrelevant", 0.97, 0.99, "uncertain"),
        ("irrelevant", 0.99, 0.97, "uncertain"),
        ("relevant", 0.60, 0.99, "uncertain"),
        ("relevant", 0.99, 0.99, "relevant"),
    ],
)
def test_only_explicit_confident_irrelevance_can_authorize_rejection(
    role: str, probability: float, confidence: float, expected: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "answers": {"n0_c1": choice(role, probability, confidence)},
                "usage": {"input_tokens": 20},
            },
        )

    result = classifier(handler, provider="openrouter").classify(selection_request())
    assert result.decision is not None
    row = result.decision.needs[0]
    values = getattr(row, expected)
    assert (values[0].citation if expected == "irrelevant" else values[0]) == 1
    if expected != "irrelevant":
        assert row.irrelevant == []


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {"type": "refusal"},
        {**choice("irrelevant"), "confidence": None},
        {**choice("irrelevant"), "probabilities": {"irrelevant": 0.99}},
        {**choice("irrelevant"), "probabilities": {key: float("nan") for key in ROLES}},
        {**choice("irrelevant"), "probabilities": {key: True for key in ROLES}},
    ],
)
def test_refusal_missing_or_malformed_distributions_retain_uncertain(
    answer: dict[str, Any] | None,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "model": "typesafe/jev-1.13",
                    "answers": {} if answer is None else {"n0_c1": answer},
                    "usage": {"input_tokens": 20},
                }
            ),
        )

    result = classifier(handler, provider="openrouter").classify(selection_request())
    assert result.decision is not None
    assert result.decision.needs[0].uncertain == [1]
    assert result.decision.needs[0].irrelevant == []


def test_duplicate_names_cannot_replace_a_refusal_with_an_irrelevance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        body = response_body(payload, "irrelevant")
        body["answers"].insert(0, {"name": "n0_c1", "type": "refusal"})
        return httpx.Response(200, json=body)

    result = classifier(handler).classify(selection_request())
    assert result.decision is None
    assert result.failure is not None


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "irrelevant"),
        (0.02, "irrelevant"),
        (0.021, "uncertain"),
        (0.799, "uncertain"),
        (0.80, "relevant"),
        (1.0, "relevant"),
        (True, "uncertain"),
        (float("nan"), "uncertain"),
        (-0.01, "uncertain"),
        (None, "uncertain"),
    ],
)
def test_native_predicate_thresholds_are_conservative(
    probability: object, expected: str
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "predicate",
                            "name": "n0_c1",
                            "probability": probability,
                        }
                    ],
                    "usage": {"input_tokens": 20},
                }
            ),
        )

    result = classifier(handler).classify(selection_request())
    assert result.decision is not None
    row = result.decision.needs[0]
    values = getattr(row, expected)
    assert (values[0].citation if expected == "irrelevant" else values[0]) == 1
    assert row.direct == []


@pytest.mark.parametrize(
    "usage,model",
    [
        ({"input_tokens": True}, "gpt-6-luna"),
        ({"input_tokens": -1}, "gpt-6-luna"),
        ({}, "gpt-6-luna"),
        ({"input_tokens": 20}, "other-model"),
    ],
)
def test_invalid_usage_or_wrong_model_keeps_reservation_and_uncertainty(
    usage: dict[str, Any], model: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = response_body(json.loads(request.content), "irrelevant")
        body.update(usage=usage, model=model)
        return httpx.Response(200, json=body)

    workflow = classifier(handler)
    result = workflow.classify(selection_request())
    assert result.decision is None
    assert workflow.budget.snapshot()["unsettled_calls"] == 1


def test_high_usage_overrun_settles_then_blocks_future_spend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = response_body(json.loads(request.content))
        body["usage"]["input_tokens"] = 100_000
        return httpx.Response(200, json=body)

    workflow = classifier(handler)
    workflow.classify(selection_request())
    assert workflow.budget.snapshot()["usage_overrun"] is True
    assert workflow.classify(selection_request()).call_id is None
    assert workflow.budget.snapshot()["model_calls"] == 1


def test_typed_jev_output_is_preserved_as_nonbillable_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = response_body(json.loads(request.content), provider="openrouter")
        body["usage"]["output_tokens"] = 38
        return httpx.Response(200, json=body)

    workflow = classifier(handler, provider="openrouter")
    result = workflow.classify(selection_request())
    assert result.failure is None
    assert workflow.budget.snapshot()["output_tokens"] == 0
    assert workflow.budget.snapshot()["estimated_cost_usd"] == 0.0000042


@pytest.mark.parametrize("status", [307, 400, 401, 429, 500])
def test_provider_failures_never_retry_follow_redirect_or_expose_secret(
    status: int,
) -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(
            status,
            headers={"Location": "https://other.example/decisions"},
            json={"error": "test-secret-not-real"},
        )

    workflow = classifier(handler)
    result = workflow.classify(selection_request())
    assert len(attempts) == 1
    assert result.decision is None and result.failure is not None
    assert "test-secret" not in result.model_dump_json()
    assert workflow.budget.snapshot()["unsettled_calls"] == 1


@pytest.mark.parametrize(
    "base",
    [
        "http://api.openai.com/v1",
        "https://other.example/v1",
        "https://api.openai.com/v1?key=secret",
        "https://secret@api.openai.com/v1",
        "https://api.openai.com/other",
        "https://api.openai.com:444/v1",
    ],
)
def test_keys_cannot_be_forwarded_to_another_endpoint(base: str) -> None:
    config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6-luna",
        temperature=0,
        api_key="test-secret",
        api_base=base,
        max_input_tokens=32_000,
    )
    with pytest.raises(ValueError, match="official provider base"):
        DecisionsClassifier(
            config=config,
            budget=WorkflowBudget(WorkflowPolicy()),
            ledger=EvidenceLedger(),
        )


def test_packing_skips_whole_large_or_unread_sources_without_text_truncation() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(json.loads(payload["input"]))
        return httpx.Response(200, json=response_body(payload))

    candidates = [
        candidate(1, "TOO_LARGE complete original"),
        candidate(2),
        candidate(3).model_copy(update={"truncated": True}),
    ]
    workflow = classifier(
        handler,
        candidates=candidates,
        token_counter=lambda body: 50_000 if "TOO_LARGE" in body else 1000,
    )
    result = workflow.classify(selection_request(*candidates))
    assert result.delivered_citations == [2]
    assert set(captured[0]["candidates"]) == {"2"}
    assert captured[0]["candidates"]["2"]["text"] == candidate(2).text


@pytest.mark.parametrize(
    "update",
    [
        {"source_id": "forged-source"},
        {"chunk_id": "forged-chunk"},
        {
            "text": "forged original",
            "text_hash": hashlib.sha256(b"forged original").hexdigest(),
        },
    ],
)
def test_original_identity_must_match_the_run_ledger_before_any_delivery(
    update: dict[str, object],
) -> None:
    handler = Mock(side_effect=AssertionError("Forged original must remain unread"))
    workflow = classifier(handler)
    result = workflow.classify(
        selection_request(candidate(1).model_copy(update=update))
    )
    assert result.decision is None
    assert result.delivered_citations == []
    assert handler.call_count == 0


def test_budget_declines_transport_before_any_paid_request() -> None:
    handler = Mock(side_effect=AssertionError("No transport admitted"))
    workflow = classifier(handler, policy=WorkflowPolicy(max_cost_usd=0.000001))
    result = workflow.classify(selection_request())
    assert result.decision is None and result.call_id is None
    assert handler.call_count == 0
    assert workflow.budget.snapshot()["model_calls"] == 0


def test_host_deadline_stops_future_spend_and_retains_timed_out_allocation() -> None:
    release = Event()

    def handler(request: httpx.Request) -> httpx.Response:
        release.wait(2)
        return httpx.Response(200, json=response_body(json.loads(request.content)))

    workflow = classifier(handler, policy=WorkflowPolicy(max_call_seconds=0.02))
    try:
        result = workflow.classify(selection_request())
        assert result.decision is None
        assert workflow.budget.snapshot()["stop_reason"] is not None
        assert workflow.budget.snapshot()["unsettled_calls"] == 1
        assert workflow.classify(selection_request()).call_id is None
        assert workflow.budget.snapshot()["model_calls"] == 1
    finally:
        release.set()


def test_one_tagged_generation_has_scope_hash_call_originals_and_actual_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed: list[Span[GenerationSpanData]] = []
    processor = Mock(spec=TracingProcessor)

    def collect(span: Span[Any]) -> None:
        if isinstance(span.span_data, GenerationSpanData):
            completed.append(span)

    processor.on_span_end.side_effect = collect
    provider = DefaultTraceProvider()
    provider.register_processor(processor)
    monkeypatch.setattr("onyx.tracing.framework.setup.GLOBAL_TRACE_PROVIDER", provider)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_body(json.loads(request.content)),
            headers={"x-request-id": "native-request-id"},
        )

    workflow = classifier(
        handler, run_id="run-exact", scope={"access_control_list": ["u:1"]}
    )
    with trace("decision-selection"):
        result = workflow.classify(selection_request())
    assert len(completed) == 1
    captured = completed[0].span_data
    config = captured.model_config or {}
    assert config["legal_composite_call_id"] == result.call_id
    assert config["legal_composite_run_id"] == "run-exact"
    assert config["legal_composite_response_id"] == "decision-response-id"
    assert (
        config["legal_composite_scope_sha256"]
        == hashlib.sha256(config["legal_composite_scope"].encode()).hexdigest()
    )
    assert (
        json.loads(config["legal_composite_delivered_originals"])[0]["text_hash"]
        == candidate(1).text_hash
    )
    assert captured.usage == {
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
    }
    assert "test-secret" not in json.dumps(captured.export())
    assert config["flow"] == LLMFlow.LEGAL_COMPOSITE_SELECTION.value


def test_completed_provider_attempt_records_native_payload_and_usage_without_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[str, dict[str, Any], Mock]] = []

    @contextmanager
    def capture(operation: str, input_value: dict[str, Any]) -> Any:
        graph = Mock()
        attempts.append((operation, input_value, graph))
        yield graph

    monkeypatch.setattr("onyx.legal_composite.decisions.graph_step", capture)

    def handler(request: httpx.Request) -> httpx.Response:
        body = response_body(json.loads(request.content))
        body.pop("id")
        return httpx.Response(
            200, json=body, headers={"x-request-id": "native-header-response-id"}
        )

    result = classifier(handler).classify(selection_request())
    assert result.failure is None
    assert len(attempts) == 1
    operation, captured, graph = attempts[0]
    assert operation == "llm.provider_attempt"
    assert captured["request_body"]["questions"][0]["type"] == "predicate"
    assert graph.output_value["id"] == "native-header-response-id"
    assert graph.output_value["decisions_response_id_source"] == "x-request-id"
    assert graph.output_value["usage"] == {"input_tokens": 100, "output_tokens": 0}
    assert "test-secret" not in json.dumps(captured)
    assert "headers" not in captured
