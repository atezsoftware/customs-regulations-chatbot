import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler
from litellm.types.utils import ModelResponseStream as LiteLLMModelResponseStream

from onyx.llm.model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    FunctionCall,
    ModelResponseStream,
    StreamingChoice,
)
from onyx.llm.models import (
    ReasoningEffort,
    SystemMessage,
    ToolChoiceOptions,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.tool_argument_stream import (
    TOOL_ARGUMENT_STREAM_ABORT,
    TOOL_ARGUMENT_WHITESPACE_RUN,
    ToolArgumentStreamGuard,
)
from onyx.tracing.framework.create import generation_span, trace


def delta(arguments: str, index: int = 0) -> ModelResponseStream:
    return ModelResponseStream(
        id="test",
        created="1",
        choice=StreamingChoice(
            delta=Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        index=index, function=FunctionCall(arguments=arguments)
                    )
                ]
            )
        ),
    )


@pytest.mark.parametrize("split", range(1, 35))
def test_strings_and_escapes_crossing_deltas_are_preserved(split: int) -> None:
    arguments = json.dumps({"answer": '\\"Quoted law ' + " " * 10000 + "\\"})
    guard = ToolArgumentStreamGuard()
    assert guard.accept(delta(arguments[:split])) is None
    assert guard.accept(delta(arguments[split:])) is None


def test_large_complete_object_and_benign_formatting_are_retained() -> None:
    guard = ToolArgumentStreamGuard()
    assert guard.accept(delta('{"task":' + " " * 10000 + '"full research"}')) is None
    assert guard.accept(delta(" " * 10000)) is None
    guard = ToolArgumentStreamGuard()
    for _ in range(100):
        assert guard.accept(delta("\n" * 20 + '"field":1,')) is None


def test_only_consecutive_formatting_whitespace_triggers_each_independent_stream() -> (
    None
):
    for _ in range(2):
        guard = ToolArgumentStreamGuard()
        assert guard.accept(delta('{"task":"preserved",')) is None
        assert guard.accept(delta("\t" * (TOOL_ARGUMENT_WHITESPACE_RUN - 1))) is None
        abort = guard.accept(delta(" "))
        assert (
            abort is not None and abort.whitespace_run == TOOL_ARGUMENT_WHITESPACE_RUN
        )
        assert abort.call_index == 0


def llm() -> LitellmLLM:
    return LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=900000,
    )


def tool_chunk(index: int, arguments: str, name: str | None = None) -> Any:
    return LiteLLMModelResponseStream(
        id="resp-test",
        created=1,
        model="gpt-6.1-sol",
        choices=cast(
            Any,
            [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": f"call-{index}" if name else None,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ]
                    },
                }
            ],
        ),
    )


def test_abort_keeps_valid_sibling_exactly_and_records_unknown_usage() -> None:
    closed = False
    received = 0

    def stream() -> Iterator[Any]:
        nonlocal closed, received
        try:
            received += 1
            yield tool_chunk(0, '{"citation":1}', "read_evidence")
            received += 1
            yield tool_chunk(1, '{"task":"full research",', "spawn_researcher")
            received += 1
            yield tool_chunk(1, "\t" * TOOL_ARGUMENT_WHITESPACE_RUN)
            received += 1
            yield tool_chunk(1, "\t" * 100000)
        finally:
            closed = True

    model = llm()
    guarded = model.with_tool_argument_stream_guard()
    assert guarded is not model and not model._tool_argument_stream_guard
    tools = [{"type": "function", "function": {"name": "read_evidence"}}]
    with (
        patch.object(guarded, "_completion", return_value=stream()),
        patch.object(guarded, "_track_llm_cost") as cost,
        trace("aborted arguments"),
        generation_span(model="gpt-6.1-sol") as span,
    ):
        response = guarded.invoke([UserMessage(content="Research fully")], tools=tools)
    assert closed and received == 3
    assert response.choice.finish_reason == TOOL_ARGUMENT_STREAM_ABORT
    assert response.usage is None
    cost.assert_not_called()
    calls = response.choice.message.tool_calls
    assert calls and calls[0].function.arguments == '{"citation":1}'
    assert calls[1].function.arguments == (
        '{"task":"full research",' + "\t" * TOOL_ARGUMENT_WHITESPACE_RUN
    )
    assert span.error is not None
    assert span.span_data.request_params[TOOL_ARGUMENT_STREAM_ABORT][
        "usage_status"
    ] == ("unknown")


class EventStream(httpx.SyncByteStream):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.received = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for event in self.events:
            self.received += 1
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("api_base", [None, "", "https://api.openai.com/v1"])
def test_missing_model_map_still_sends_native_sse_and_closes_early(
    api_base: str | None,
) -> None:
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp-test",
                "created_at": 1,
                "object": "response",
                "status": "in_progress",
                "model": "gpt-6.1-sol",
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc-test",
                "call_id": "call-0",
                "name": "spawn_researcher",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "fc-test",
            "delta": '{"task":"preserved",' + "\t" * TOOL_ARGUMENT_WHITESPACE_RUN,
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": "fc-test",
            "delta": "\t" * 10000,
        },
    ]
    event_stream = EventStream(events)
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        stream=event_stream,
        headers={"content-type": "text/event-stream"},
    )
    guarded = LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=900000,
        api_base=api_base,
        extra_headers={"x-test-request": "preserved"},
        extra_body={"store": False},
        model_kwargs={"user": "original-user"},
    ).with_tool_argument_stream_guard()
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
    ):
        response = guarded.invoke(
            [
                SystemMessage(content="Preserve original source conditions"),
                UserMessage(content="Retain full evidence"),
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "spawn_researcher",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            max_tokens=128000,
            tool_choice=ToolChoiceOptions.AUTO,
            reasoning_effort=ReasoningEffort.HIGH,
        )
    ordinary.assert_not_called()
    assert response.choice.finish_reason == TOOL_ARGUMENT_STREAM_ABORT
    assert event_stream.closed and event_stream.received == 3
    request = client.post.call_args.kwargs
    assert request["stream"] is True
    assert request["json"]["stream"] is True
    assert request["json"]["model"] == "gpt-6.1-sol"
    assert request["json"]["max_output_tokens"] == 128000
    assert request["json"]["parallel_tool_calls"] is True
    assert request["json"]["reasoning"]["effort"] == "high"
    assert request["json"]["temperature"] == 1
    assert request["json"]["tool_choice"] == "auto"
    assert request["json"]["instructions"] == "Preserve original source conditions"
    assert request["json"]["user"] == "original-user"
    assert request["json"]["store"] is False
    assert request["headers"]["x-test-request"] == "preserved"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["input"][0]["content"][0]["text"] == "Retain full evidence"
    client.close.assert_called_once()


@pytest.mark.parametrize(
    ("provider", "model", "api_base"),
    [
        ("openai", "gpt-6.1-sol", "https://proxy.example/v1"),
        ("openai", "custom-model", None),
        ("openrouter", "gpt-6.1-sol", None),
    ],
)
def test_absent_map_fallback_does_not_guess_proxy_capabilities(
    provider: str, model: str, api_base: str | None
) -> None:
    guarded = LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=model,
        max_input_tokens=900000,
        api_base=api_base,
    ).with_tool_argument_stream_guard()
    assert not guarded._uses_guarded_openai_responses()


def test_native_raw_bad_request_preserves_provider_compatibility_retry() -> None:
    from litellm.llms.openai.common_utils import OpenAIError

    guarded = llm().with_tool_argument_stream_guard()
    with (
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.response_api_handler",
            side_effect=[
                OpenAIError(400, "Unsupported reasoning parameter"),
                iter([tool_chunk(0, '{"citation":1}', "read_evidence")]),
            ],
        ) as transport,
    ):
        # Keep the ordinary conversion path mocked: this case isolates error mapping.
        with patch(
            "litellm.completion_extras.litellm_responses_transformation.transformation.LiteLLMResponsesTransformationHandler.get_model_response_iterator",
            side_effect=lambda streaming_response, **_kwargs: streaming_response,
        ):
            response = guarded.invoke(
                [UserMessage(content="Read complete original")],
                tools=[{"type": "function", "function": {"name": "read_evidence"}}],
            )
    ordinary.assert_not_called()
    assert transport.call_count == 2
    assert (
        "reasoning"
        in transport.call_args_list[0].kwargs["response_api_optional_request_params"]
    )
    assert (
        "reasoning"
        not in transport.call_args_list[1].kwargs[
            "response_api_optional_request_params"
        ]
    )
    assert response.choice.message.tool_calls
    assert response.choice.message.tool_calls[0].function.arguments == '{"citation":1}'


def test_native_timeout_remains_a_provider_timeout() -> None:
    from litellm.llms.openai.common_utils import OpenAIError

    from onyx.llm.multi_llm import LLMTimeoutError

    guarded = llm().with_tool_argument_stream_guard()
    with (
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.response_api_handler",
            side_effect=OpenAIError(408, "provider timed out"),
        ),
        pytest.raises(LLMTimeoutError),
    ):
        guarded.invoke(
            [UserMessage(content="Read complete original")],
            tools=[{"type": "function", "function": {"name": "read_evidence"}}],
        )
    ordinary.assert_not_called()
