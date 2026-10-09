"""Legal Review admissions account schemas and normalize provider timeout errors."""

import json
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.legal_review.gateway import GeminiGateway
from onyx.legal_review.models import WorkflowPolicy
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.tracing.flows import LLMFlow


class Response(BaseModel):
    decision: str


def gateway(*, policy: WorkflowPolicy | None = None) -> GeminiGateway:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=200_000,
    )
    return GeminiGateway(
        llm=cast(LLM, llm),
        context=RunContext(
            timeout_seconds=240,
            research_reserve_seconds=110,
            budget=SharedBudget(max_decisions=32),
        ),
        policy=policy or WorkflowPolicy(),
        token_counter=len,
    )


def test_schema_and_json_reminder_can_exhaust_capacity_before_provider_call() -> None:
    transport = gateway(policy=WorkflowPolicy(max_context_tokens=100))
    with patch("onyx.legal_review.gateway.generate_structured") as call:
        with pytest.raises(RunStopped, match="context exceeds"):
            transport.complete(
                "Read.", {"request": "x"}, Response, LLMFlow.LEGAL_REVIEW_READING
            )
    call.assert_not_called()


def test_provider_timeout_is_safe_and_no_automatic_retry_is_added() -> None:
    transport = gateway()
    with patch(
        "onyx.legal_review.gateway.generate_structured",
        side_effect=LLMTimeoutError("sensitive-provider-detail"),
    ) as call:
        with pytest.raises(TimeoutError) as error:
            transport.complete(
                "Read.", {"request": "x"}, Response, LLMFlow.LEGAL_REVIEW_READING
            )
    assert str(error.value) == "Legal Review Gemini provider timeout"
    assert isinstance(error.value.__cause__, LLMTimeoutError)
    assert call.call_count == 1
    assert call.call_args.kwargs["max_attempts"] == 1
    assert call.call_args.kwargs["provider_max_attempts"] == 1
    assert call.call_args.kwargs["timeout_override"] == 45
    assert transport.context.budget.snapshot()["decisions"] == 1


def test_remaining_phase_deadline_bounds_provider_timeout() -> None:
    transport = gateway()
    with (
        patch(
            "onyx.legal_review.gateway.time.monotonic",
            return_value=transport.context.research_deadline - 7.9,
        ),
        patch(
            "onyx.legal_review.gateway.generate_structured",
            return_value=Response(decision="ok"),
        ) as call,
    ):
        transport.complete(
            "Read.", {"request": "x"}, Response, LLMFlow.LEGAL_REVIEW_READING
        )
    assert call.call_args.kwargs["timeout_override"] == 7
    assert call.call_args.kwargs["deadline"] == transport.context.research_deadline


def test_writer_provider_receives_normalized_view_without_mutating_host_state() -> None:
    transport = gateway()
    state: dict[str, JsonValue] = {
        "request": "x",
        "original_evidence": [],
        "tools": [{"name": "search"}],
    }
    with patch(
        "onyx.legal_review.gateway.generate_structured",
        return_value=Response(decision="ok"),
    ) as call:
        result = transport.complete(
            "Write.", state, Response, LLMFlow.LEGAL_REVIEW_DRAFT, finalizing=True
        )
    assert result == Response(decision="ok")
    actual = json.loads(call.call_args.kwargs["user_prompt"])
    assert "tools" not in actual and actual["source_registry"] == {}
    assert "tools" in state
