import copy
from typing import Any

import pytest

from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    FunctionCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.tracing.framework.create import generation_span, trace
from tests.unit.onyx.llm.test_asv3_response_terminal_capture import (
    REPORTED_USAGE,
    events,
    invoke,
    model,
)
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream


def prompt(turns: int) -> list[ChatCompletionMessage]:
    result: list[ChatCompletionMessage] = [
        SystemMessage(content="Preserve every original, qualification and citation."),
        UserMessage(content="Explain the same legal question."),
    ]
    for index in range(turns):
        result.extend(
            [
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id=f"call-{index}",
                            function=FunctionCall(
                                name="read_provision",
                                arguments='{"article":"8/1"}',
                            ),
                        )
                    ]
                ),
                ToolMessage(
                    tool_call_id=f"call-{index}",
                    content=f'{{"citation":{index},"text":"Original condition {index}"}}',
                ),
            ]
        )
    result.append(UserMessage(content=f'{{"current_decision":{turns}}}'))
    return result


def without_breakpoints(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_breakpoints(item)
            for key, item in value.items()
            if key != "prompt_cache_breakpoint"
        }
    if isinstance(value, list):
        return [without_breakpoints(item) for item in value]
    return value


@pytest.mark.parametrize("turns", [0, 2, 60])
def test_actual_wire_diff_is_only_tool_endpoint_metadata(turns: int) -> None:
    messages = prompt(turns)
    original = copy.deepcopy(messages)
    _, baseline = invoke(
        model(capture=False),
        EventStream(events("completed", REPORTED_USAGE)),
        prompt=messages,
    )
    with (
        trace("Native stable endpoint metadata"),
        generation_span(model="gpt-6.1-sol") as span,
    ):
        _, marked = invoke(
            model(), EventStream(events("completed", REPORTED_USAGE)), prompt=messages
        )
    assert without_breakpoints(marked) == baseline
    assert without_breakpoints(baseline) == baseline
    assert messages == original
    assert "prompt_cache_options" not in marked
    endpoints = [
        item for item in marked["input"] if item["type"] == "function_call_output"
    ]
    assert len(endpoints) == turns
    assert span.span_data.request_params is not None
    diagnostic = span.span_data.request_params["asv3_native_response_terminal"]
    assert diagnostic["requested_tool_cache_breakpoints"] == turns
    for endpoint in endpoints:
        assert endpoint["output"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    for item in marked["input"]:
        if item["type"] != "function_call_output":
            assert without_breakpoints(item) == item


def test_growing_history_preserves_previous_marked_endpoints() -> None:
    _, previous = invoke(
        model(), EventStream(events("completed", REPORTED_USAGE)), prompt=prompt(2)
    )
    _, following = invoke(
        model(), EventStream(events("completed", REPORTED_USAGE)), prompt=prompt(3)
    )
    assert following["input"][: len(previous["input"]) - 1] == previous["input"][:-1]
    assert following["input"][-1] != previous["input"][-1]
    assert "prompt_cache_breakpoint" not in following["input"][-1]["content"][0]


@pytest.mark.parametrize("reported_tier", [None, "priority"])
def test_terminal_effective_tier_prefers_provider_then_requested(
    reported_tier: str | None,
) -> None:
    raw = events("completed", REPORTED_USAGE)
    if reported_tier is not None:
        raw[-1]["response"]["service_tier"] = reported_tier
    with (
        trace("Native effective service tier"),
        generation_span(model="gpt-6.1-sol") as span,
    ):
        response, wire = invoke(model(service_tier="flex"), EventStream(raw))
    assert wire["service_tier"] == "flex"
    effective_tier = reported_tier or "flex"
    assert response.usage is not None
    assert response.usage.service_tier == effective_tier
    assert span.span_data.request_params is not None
    assert span.span_data.request_params["service_tier"] == effective_tier


def test_requested_tier_fallback_matches_extra_body_precedence() -> None:
    llm = (
        LitellmLLM(
            api_key="test-key",
            model_provider="openai",
            model_name="gpt-6.1-sol",
            max_input_tokens=900000,
            model_kwargs={"service_tier": "flex"},
            extra_body={"store": False, "service_tier": "priority"},
        )
        .with_tool_argument_stream_guard()
        .with_asv3_response_terminal_capture()
    )
    response, wire = invoke(llm, EventStream(events("completed", REPORTED_USAGE)))
    assert wire["service_tier"] == "priority"
    assert response.usage is not None and response.usage.service_tier == "priority"
