import copy
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler

from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.native_responses_stream import CroppedResponseTerminalError
from onyx.llm.tool_argument_stream import (
    TOOL_ARGUMENT_STREAM_ABORT,
    TOOL_ARGUMENT_WHITESPACE_RUN,
)
from onyx.tracing.framework.create import generation_span, trace
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream, tool_chunk

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_provision",
            "parameters": {
                "type": "object",
                "properties": {"article": {"type": "string"}},
            },
        },
    }
]
REPORTED_USAGE = {
    "input_tokens": 100,
    "output_tokens": 12,
    "total_tokens": 112,
    "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 2},
}


def model(
    *,
    capture: bool = True,
    api_base: str | None = None,
    explicit_cache: bool = False,
) -> LitellmLLM:
    llm = LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=900000,
        api_base=api_base,
        extra_body={"store": False},
    ).with_tool_argument_stream_guard()
    return (
        llm.with_cropped_response_terminal_capture(explicit_cache=explicit_cache)
        if capture
        else llm
    )


def events(
    status: str,
    usage: Any,
    *,
    tool: bool = False,
    reason: str = "max_output_tokens",
) -> list[dict[str, Any]]:
    response: dict[str, Any] = {
        "id": "resp-test",
        "created_at": 1,
        "object": "response",
        "model": "gpt-6.1-sol",
        "status": "in_progress",
        "output": [],
    }
    result: list[dict[str, Any]] = [
        {"type": "response.created", "response": copy.deepcopy(response)}
    ]
    if tool:
        item = {
            "type": "function_call",
            "id": "fc-test",
            "call_id": "call-test",
            "name": "read_provision",
            "arguments": "",
            "status": "in_progress",
        }
        result.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": copy.deepcopy(item),
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "item_id": "fc-test",
                    "delta": '{"article":',
                },
            ]
        )
        item["arguments"] = '{"article":'
        response["output"] = [item]
    else:
        result.append(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg-test",
                "delta": "Original source qualification remains unresolved.",
            }
        )
    response.update(status=status, usage=usage)
    if status == "incomplete":
        response["incomplete_details"] = {"reason": reason}
    if status == "failed":
        response["error"] = {"code": "server_error", "message": "Generation failed"}
    result.append({"type": f"response.{status}", "response": response})
    return result


def invoke(
    llm: LitellmLLM,
    stream: EventStream,
    *,
    prompt: list[ChatCompletionMessage] | None = None,
) -> tuple[Any, dict[str, Any]]:
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        stream=stream,
        headers={"content-type": "text/event-stream"},
    )
    selected_prompt: list[ChatCompletionMessage] = [
        UserMessage(content="Retain all original conditions.")
    ]
    if prompt:
        selected_prompt = prompt
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
    ):
        try:
            result = llm.invoke(
                selected_prompt,
                tools=copy.deepcopy(TOOLS),
                max_tokens=12,
            )
        finally:
            ordinary.assert_not_called()
            client.close.assert_called_once()
    return result, client.post.call_args.kwargs["json"]


@pytest.mark.parametrize("tool", [False, True])
def test_incomplete_preserves_actual_usage_and_normalizes_length(tool: bool) -> None:
    stream = EventStream(events("incomplete", REPORTED_USAGE, tool=tool))
    with trace("Cropped incomplete"), generation_span(model="gpt-6.1-sol") as span:
        response, wire = invoke(model(), stream)
    assert response.choice.finish_reason == "length"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 12
    assert response.usage.cache_read_input_tokens == 4
    assert response.usage.reasoning_tokens == 2
    if tool:
        assert response.choice.message.tool_calls
        assert response.choice.message.tool_calls[0].function.arguments == '{"article":'
    else:
        assert response.choice.message.content == (
            "Original source qualification remains unresolved."
        )
    assert span.span_data.request_params is not None
    diagnostic = span.span_data.request_params["asv3_exp_cropped_response_terminal"]
    assert diagnostic["status"] == "incomplete"
    assert diagnostic["incomplete_reason"] == "max_output_tokens"
    assert diagnostic["usage_status"] == "provider_reported"
    assert wire["max_output_tokens"] == 12 and wire["store"] is False
    assert stream.closed


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 100}])
def test_missing_or_invalid_terminal_usage_never_uses_builder_estimates(
    usage: Any,
) -> None:
    stream = EventStream(events("incomplete", usage))
    response, _ = invoke(model(), stream)
    assert response.choice.finish_reason == "length"
    assert response.usage is None
    assert stream.closed


@pytest.mark.parametrize(
    ("incoming", "cached", "writes"),
    [(100, 0, 101), (100, 4, 97), (100, 100, 1)],
)
def test_invalid_read_write_sum_retains_raw_counts_without_inferred_usage(
    incoming: int, cached: int, writes: int
) -> None:
    usage = {
        **REPORTED_USAGE,
        "input_tokens": incoming,
        "total_tokens": incoming + 12,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes},
    }
    with trace("Cropped invalid usage"), generation_span(model="gpt-6.1-sol") as span:
        response, _ = invoke(model(), EventStream(events("incomplete", usage)))
    assert response.usage is None
    assert span.span_data.usage is None
    assert span.span_data.request_params is not None
    diagnostic = span.span_data.request_params["asv3_exp_cropped_response_terminal"]
    assert diagnostic["usage_status"] == "invalid"
    assert diagnostic["provider_reported_usage"] == usage


@pytest.mark.parametrize("cached", [0, 3])
def test_live_write_counts_remain_part_of_reported_total_input(cached: int) -> None:
    usage = {
        "input_tokens": 31_493,
        "output_tokens": 272,
        "total_tokens": 31_765,
        "input_tokens_details": {
            "cached_tokens": cached,
            "cache_write_tokens": 31_490,
        },
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    with trace("Cropped cache writes"), generation_span(model="gpt-6.1-sol") as span:
        response, _ = invoke(model(), EventStream(events("completed", usage)))
    assert response.usage is not None
    assert response.usage.prompt_tokens == 31_493
    assert response.usage.cache_read_input_tokens == cached
    assert response.usage.cache_creation_input_tokens == 31_490
    assert response.usage.completion_tokens == 272
    assert response.usage.total_tokens == 31_765
    assert response.choice.finish_reason == "stop"
    assert span.span_data.request_params is not None
    diagnostic = span.span_data.request_params["asv3_exp_cropped_response_terminal"]
    assert diagnostic["usage_status"] == "provider_reported"
    assert diagnostic["provider_reported_usage"] == usage


@pytest.mark.parametrize("usage", [None, REPORTED_USAGE])
def test_failed_preserves_actual_error_and_usage_without_returning_actions(
    usage: Any,
) -> None:
    stream = EventStream(events("failed", usage, tool=True))
    with (
        trace("Cropped failed"),
        generation_span(model="gpt-6.1-sol") as span,
        pytest.raises(CroppedResponseTerminalError) as raised,
    ):
        invoke(model(), stream)
    assert raised.value.provider_error == {
        "code": "server_error",
        "message": "Generation failed",
    }
    assert (raised.value.usage is None) == (usage is None)
    assert span.error is not None
    assert isinstance(span.error["data"], dict)
    assert span.error["data"]["status"] == "failed"
    assert span.span_data.usage == raised.value.usage
    assert stream.closed


def test_other_incomplete_reasons_fail_closed() -> None:
    stream = EventStream(events("incomplete", REPORTED_USAGE, reason="content_filter"))
    with pytest.raises(CroppedResponseTerminalError) as raised:
        invoke(model(), stream)
    assert raised.value.terminal["incomplete_reason"] == "content_filter"
    assert raised.value.usage is not None
    assert stream.closed


def test_completed_response_and_wire_match_default_behavior() -> None:
    result, captured_wire = invoke(
        model(), EventStream(events("completed", REPORTED_USAGE))
    )
    baseline, baseline_wire = invoke(
        model(capture=False), EventStream(events("completed", REPORTED_USAGE))
    )
    assert result.choice == baseline.choice
    assert result.choice.finish_reason == "stop"
    assert (
        result.usage
        and result.usage.completion_tokens == captured_wire["max_output_tokens"]
    )
    assert result.usage == baseline.usage
    assert captured_wire == baseline_wire


def test_aborted_stream_preserves_guard_and_terminal_diagnostics_without_usage() -> (
    None
):
    raw = events("incomplete", REPORTED_USAGE, tool=True)
    raw[2]["delta"] = '{"article":' + "\t" * TOOL_ARGUMENT_WHITESPACE_RUN
    stream = EventStream(raw)
    with trace("Cropped aborted"), generation_span(model="gpt-6.1-sol") as span:
        response, _ = invoke(model(), stream)
    assert response.choice.finish_reason == TOOL_ARGUMENT_STREAM_ABORT
    assert response.usage is None
    assert span.span_data.usage is None
    params = span.span_data.request_params
    assert params is not None
    assert params["max_tokens"] == 12
    assert params["reasoning_effort"] == "auto"
    assert "sampling" in params and "sent_kwargs" in params
    abort = params[TOOL_ARGUMENT_STREAM_ABORT]
    assert abort["usage_status"] == "unknown"
    assert abort["whitespace_run"] == TOOL_ARGUMENT_WHITESPACE_RUN
    terminal = params["asv3_exp_cropped_response_terminal"]
    assert terminal["event_type"] is None
    assert terminal["usage_status"] == "missing"
    assert span.error is not None
    assert span.error["data"] == abort
    assert stream.received == 3
    assert stream.closed


def test_bound_flag_isolated_and_unrelated_proxy_does_not_capture() -> None:
    ordinary = model(capture=False)
    bound = ordinary.with_cropped_response_terminal_capture()
    assert bound is not ordinary
    assert not ordinary._cropped_response_terminal_capture
    assert bound._cropped_response_terminal_capture
    proxy = model(api_base="https://proxy.example/v1")
    with patch.object(
        proxy, "_completion", return_value=iter([tool_chunk(0, "{}", "read_provision")])
    ) as completion:
        response = proxy.invoke([UserMessage(content="Unchanged")], tools=TOOLS)
    assert response.choice.finish_reason != "length"
    assert "terminal_capture" not in completion.call_args.kwargs


def test_missing_terminal_event_fails_closed_without_inferred_usage() -> None:
    stream = EventStream(events("completed", REPORTED_USAGE)[:-1])
    with pytest.raises(CroppedResponseTerminalError) as raised:
        invoke(model(), stream)
    assert raised.value.usage is None
    assert raised.value.terminal["status"] == "missing_terminal_event"
    assert stream.closed


def test_mismatched_terminal_status_fails_closed_with_reported_usage() -> None:
    raw = events("incomplete", REPORTED_USAGE, tool=True)
    raw[-1]["response"]["status"] = "completed"
    stream = EventStream(raw)
    with pytest.raises(CroppedResponseTerminalError) as raised:
        invoke(model(), stream)
    assert raised.value.usage is not None
    assert raised.value.terminal["status"] == "invalid_terminal_status"
    assert stream.closed
