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


@pytest.mark.parametrize("size,admitted", [(195_000, True), (205_000, False)])
def test_default_context_admission_uses_configured_provider_capacity(
    size: int, admitted: bool
) -> None:
    transport = gateway()
    state: dict[str, JsonValue] = {"request": "x" * size}
    with patch(
        "onyx.legal_review.gateway.generate_structured",
        return_value=Response(decision="ok"),
    ) as call:
        if admitted:
            transport.complete("Read.", state, Response, LLMFlow.LEGAL_REVIEW_READING)
            assert json.loads(call.call_args.kwargs["user_prompt"]) == state
        else:
            with pytest.raises(RunStopped, match="context exceeds"):
                transport.complete(
                    "Read.", state, Response, LLMFlow.LEGAL_REVIEW_READING
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


def test_correction_reports_root_errors_instead_of_a_nested_passage_object() -> None:
    from pydantic import Field, ValidationError

    from onyx.legal_review.gateway import root_validation_error
    from onyx.regulatory.structured_llm import _validate_json_object

    class Row(BaseModel):
        value: int = Field(gt=0)

    class Batch(BaseModel):
        rows: list[Row]

    content = json.dumps({"rows": [{"value": -1} for _ in range(5)]})
    with pytest.raises(ValidationError) as generic:
        _validate_json_object(content, Batch)
    # The shared prose-tolerant parser selects the smaller nested-object error.
    assert generic.value.errors()[0]["loc"] == ("rows",)
    actual = root_validation_error(content, Batch, generic.value)
    assert [row["loc"] for row in actual.errors()] == [
        ("rows", index, "value") for index in range(5)
    ]
    assert (
        root_validation_error("prose " + content, Batch, generic.value) is generic.value
    )


@pytest.mark.parametrize(
    "flow",
    [
        LLMFlow.LEGAL_REVIEW_PLANNER,
        LLMFlow.LEGAL_REVIEW_READING,
        LLMFlow.LEGAL_REVIEW_SOURCE_ACCOUNTING,
        LLMFlow.LEGAL_REVIEW_DRAFT,
        LLMFlow.LEGAL_REVIEW_REPAIR,
    ],
)
def test_every_structured_phase_streams_with_its_actual_workflow_deadline(
    flow: LLMFlow,
) -> None:
    from onyx.legal_review.gateway import MeteredLLM, UsageMeter

    transport = gateway()
    selected = cast(MagicMock, transport.llm)
    selected.with_stream_cancellation_check.return_value = selected
    transport.llm = MeteredLLM(
        selected, transport.context, UsageMeter(transport.policy)
    )
    finalizing = flow in {LLMFlow.LEGAL_REVIEW_DRAFT, LLMFlow.LEGAL_REVIEW_REPAIR}
    with patch(
        "onyx.legal_review.gateway.generate_structured",
        return_value=Response(decision="ok"),
    ) as call:
        transport.complete("Read or write.", {}, Response, flow, finalizing=finalizing)
    invocation = call.call_args.args[0]
    assert isinstance(invocation, MeteredLLM)
    assert invocation.reader_deadline == (
        transport.context.deadline - transport.policy.publication_reserve_seconds
        if finalizing
        else transport.context.research_deadline
    )
