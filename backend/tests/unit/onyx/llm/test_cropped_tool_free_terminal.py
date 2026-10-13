"""Tool-free Cropped reviews retain provider terminal and usage evidence."""

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler

from onyx.llm.model_response import ModelResponse
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.native_responses_stream import NativeResponseTerminalError
from onyx.tracing.framework.create import generation_span, trace
from tests.unit.onyx.llm.test_asv3_response_terminal_capture import (
    REPORTED_USAGE,
    events,
    model,
)
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream, tool_chunk


def invoke_tool_free(
    llm: LitellmLLM,
    stream: EventStream,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[ModelResponse, dict[str, Any]]:
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        stream=stream,
        headers={"content-type": "text/event-stream"},
    )
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
    ):
        try:
            result = llm.invoke(
                [UserMessage(content="Check all conditions; return JSON.")],
                tools=tools,
                max_tokens=12,
            )
        finally:
            ordinary.assert_not_called()
            client.close.assert_called_once()
    return result, client.post.call_args.kwargs["json"]


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_review_keeps_exact_text_terminal_usage_and_tier(
    tools: list[dict[str, Any]] | None, status: str
) -> None:
    llm = model(capture=False, service_tier="default")
    cropped = llm.with_cropped_response_terminal_capture()
    raw = events(status, copy.deepcopy(REPORTED_USAGE))
    content = '{"conditions": ["Özgün koşul [1]"' + (
        "]}" if status == "completed" else ""
    )
    raw[1]["delta"] = content
    raw[-1]["response"]["service_tier"] = "priority"
    stream = EventStream(raw)
    with (
        trace("Cropped tool-free review"),
        generation_span(model="gpt-6.1-sol") as span,
        patch.object(cropped, "_track_llm_cost") as track_cost,
    ):
        response, wire = invoke_tool_free(cropped, stream, tools)
    assert response.choice.message.content == content
    assert response.choice.finish_reason == (
        "stop" if status == "completed" else "length"
    )
    assert response.usage is not None
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 12
    assert response.usage.total_tokens == 112
    assert response.usage.cache_read_input_tokens == 4
    assert response.usage.cache_creation_input_tokens == 0
    assert response.usage.reasoning_tokens == 2
    assert response.usage.service_tier == "priority"
    track_cost.assert_called_once_with(response.usage)
    assert span.span_data.usage == response.usage
    assert span.span_data.request_params is not None
    diagnostic = span.span_data.request_params["asv3_exp_cropped_response_terminal"]
    assert diagnostic["status"] == status
    assert diagnostic["usage_status"] == "provider_reported"
    assert diagnostic["provider_reported_usage"] == REPORTED_USAGE
    assert diagnostic["service_tier"] == "priority"
    assert wire["service_tier"] == "default"
    assert wire["max_output_tokens"] == 12
    assert wire["store"] is False
    assert not wire.get("tools")
    assert stream.closed
    assert not llm._cropped_response_terminal_capture


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 100}])
def test_review_missing_usage_does_not_fall_back_to_builder_estimates(
    usage: Any,
) -> None:
    llm = model(capture=False).with_cropped_response_terminal_capture()
    with (
        trace("Cropped unknown review usage"),
        generation_span(model="gpt-6.1-sol") as span,
        patch.object(llm, "_track_llm_cost") as track_cost,
    ):
        response, _ = invoke_tool_free(llm, EventStream(events("completed", usage)))
    assert response.usage is None
    assert span.span_data.usage is None
    track_cost.assert_not_called()


@pytest.mark.parametrize("empty", [False, True])
def test_cut_review_fails_closed_without_dispatch_or_estimated_usage(
    empty: bool,
) -> None:
    raw = events("completed", REPORTED_USAGE)[: 1 if empty else -1]
    stream = EventStream(raw)
    llm = model(capture=False).with_cropped_response_terminal_capture()
    with (
        trace("Cropped cut review"),
        generation_span(model="gpt-6.1-sol") as span,
        patch.object(llm, "_track_llm_cost") as track_cost,
        pytest.raises(NativeResponseTerminalError) as raised,
    ):
        invoke_tool_free(llm, stream)
    assert raised.value.terminal["status"] == "missing_terminal_event"
    assert raised.value.usage is None
    assert span.span_data.usage is None
    assert span.error is not None
    track_cost.assert_not_called()
    assert stream.closed


@pytest.mark.parametrize("status", ["failed", "content_filter", "mismatched"])
def test_invalid_review_terminal_is_not_a_repairable_length_finish(status: str) -> None:
    raw = events(
        "failed" if status == "failed" else "incomplete",
        REPORTED_USAGE,
        reason="content_filter" if status == "content_filter" else "max_output_tokens",
    )
    if status == "mismatched":
        raw[-1]["response"]["status"] = "completed"
    stream = EventStream(raw)
    llm = model(capture=False).with_cropped_response_terminal_capture()
    with (
        patch.object(llm, "_track_llm_cost") as track_cost,
        pytest.raises(NativeResponseTerminalError) as raised,
    ):
        invoke_tool_free(llm, stream)
    assert raised.value.usage is not None
    track_cost.assert_called_once_with(raised.value.usage)
    assert stream.closed


def test_tool_free_review_cancellation_propagates_and_closes_stream() -> None:
    class Cancelled(RuntimeError):
        pass

    check_active = MagicMock(side_effect=Cancelled("Stop"))
    llm = (
        model(capture=False)
        .with_cropped_response_terminal_capture()
        .with_stream_cancellation_check(check_active)
    )
    stream = EventStream(events("completed", REPORTED_USAGE))
    with (
        patch.object(llm, "_track_llm_cost") as track_cost,
        pytest.raises(Cancelled, match="Stop"),
    ):
        invoke_tool_free(llm, stream)
    check_active.assert_called_once()
    track_cost.assert_not_called()
    assert stream.closed


@pytest.mark.parametrize(
    "variant",
    [
        "ordinary",
        "experimental",
        "cheap",
        "cropped_cheap",
        "proxy",
        "unguarded",
        "other_model",
        "other_provider",
    ],
)
def test_other_tool_free_routes_do_not_acquire_native_capture(variant: str) -> None:
    llm = model(capture=variant == "experimental")
    if variant in {"cheap", "cropped_cheap"}:
        llm = llm.with_asv3_exp_cheap_cache()
    if variant in {
        "cropped_cheap",
        "proxy",
        "unguarded",
        "other_model",
        "other_provider",
    }:
        if variant == "proxy":
            llm = model(capture=False, api_base="https://proxy.example/v1")
        if variant == "unguarded":
            llm = LitellmLLM(
                api_key="test-key",
                model_provider="openai",
                model_name="gpt-6.1-sol",
                max_input_tokens=900000,
            )
        if variant in {"other_model", "other_provider"}:
            llm = LitellmLLM(
                api_key="test-key",
                model_provider="openrouter"
                if variant == "other_provider"
                else "openai",
                model_name="custom-model"
                if variant == "other_model"
                else "gpt-6.1-sol",
                max_input_tokens=900000,
            ).with_tool_argument_stream_guard()
        llm = llm.with_cropped_response_terminal_capture()
    with (
        patch("litellm.HTTPHandler"),
        patch(
            "litellm.completion", return_value=iter([tool_chunk(0, "{}", "test")])
        ) as ordinary,
        patch("onyx.llm.native_responses_stream.native_responses_stream") as native,
    ):
        llm.invoke([UserMessage(content="Existing path")], tools=None)
    ordinary.assert_called_once()
    native.assert_not_called()


def test_cropped_tool_free_nonstream_retains_existing_route() -> None:
    from litellm import ModelResponse as LiteLLMModelResponse

    llm = model(capture=False).with_cropped_response_terminal_capture()
    with (
        patch("litellm.HTTPHandler"),
        patch("litellm.completion", return_value=LiteLLMModelResponse()) as ordinary,
        patch("onyx.llm.native_responses_stream.native_responses_stream") as native,
    ):
        llm.invoke(
            [UserMessage(content="Existing path")], tools=None, use_streaming=False
        )
    ordinary.assert_called_once()
    native.assert_not_called()


def test_explicit_zero_write_still_requires_the_existing_tool_contract() -> None:
    llm = model(capture=False).with_cropped_response_terminal_capture(
        explicit_cache=True
    )
    with (
        patch.object(llm, "_completion") as completion,
        pytest.raises(ValueError, match="Zero-write Cropped cache"),
    ):
        llm.invoke([UserMessage(content="Existing contract")], tools=None)
    completion.assert_not_called()
