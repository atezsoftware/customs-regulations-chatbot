import copy
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler

from onyx.llm.asv3_exp_cheap_cache import ASv3ExpCheapCachePolicy, NativeCacheRequest
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import (
    AssistantMessage,
    FunctionCall,
    ReasoningEffort,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.tool_argument_stream import TOOL_ARGUMENT_WHITESPACE_RUN
from onyx.tracing.framework.create import generation_span, trace


def strip_cache_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_cache_metadata(item)
            for key, item in value.items()
            if key not in {"prompt_cache_options", "prompt_cache_breakpoint"}
        }
    if isinstance(value, list):
        return [strip_cache_metadata(item) for item in value]
    return value


def request_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def request(tool_count: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Original question"}],
        }
    ]
    for index in range(tool_count):
        items.extend(
            [
                {
                    "type": "function_call",
                    "call_id": f"call-{index}",
                    "name": "read",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call-{index}",
                    "output": [
                        {
                            "type": "input_text",
                            "text": f"Exact original {index}\nAND/OR; exception; dates.",
                        }
                    ],
                },
            ]
        )
    items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": '{"state":"rebuilt"}'}],
        }
    )
    return {
        "model": "gpt-6.1-sol",
        "instructions": "All legal conditions",
        "input": items,
        "tools": [],
        "reasoning": {"effort": "medium"},
        "extra_body": {"store": False},
    }


def complete(policy: ASv3ExpCheapCachePolicy, capture: NativeCacheRequest) -> None:
    events = iter([{"type": "response.completed", "response": {"id": "resp-previous"}}])
    assert list(policy.capture_response(events, capture))


@pytest.mark.parametrize("tool_count", [0, 1, 8])
def test_default_explicit_mode_leaves_changing_latest_state_outside_breakpoints(
    tool_count: int,
) -> None:
    policy = ASv3ExpCheapCachePolicy()
    current = request(tool_count)
    current["input"][-1]["content"][0]["text"] = (
        '{"state":"changed","draft":"Şartlar AND/OR; istisna; 01.01.2026"}'
    )
    before = request_bytes(current)
    policy.prepare_request(current)
    assert current["extra_body"]["prompt_cache_options"]["mode"] == "explicit"
    assert "prompt_cache_breakpoint" not in current["input"][-1]["content"][0]
    assert request_bytes(strip_cache_metadata(current)) == before


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_explicit_caller_cache_options_are_preserved(mode: str) -> None:
    policy = ASv3ExpCheapCachePolicy()
    complete(policy, policy.prepare_request(request(1)))
    current = request(2)
    caller_options = {
        "mode": mode,
        "ttl": "30m",
        "comparison_response_id": "resp-caller",
    }
    current["extra_body"]["prompt_cache_options"] = caller_options
    before = request_bytes(strip_cache_metadata(current))
    policy.prepare_request(current)
    assert current["extra_body"]["prompt_cache_options"] == caller_options
    assert current["extra_body"]["prompt_cache_options"] is not caller_options
    assert caller_options["comparison_response_id"] == "resp-caller"
    assert request_bytes(strip_cache_metadata(current)) == before


def test_history_boundary_requires_completion_and_excludes_initial_question() -> None:
    policy = ASv3ExpCheapCachePolicy()
    assert not policy.has_recorded_history_boundary
    initial = policy.prepare_request(request(0))
    assert initial.boundary_hashes and not initial.history_boundary_hashes
    complete(policy, initial)
    assert not policy.has_recorded_history_boundary
    history = policy.prepare_request(request(1))
    assert history.history_boundary_hashes
    assert not policy.has_recorded_history_boundary
    complete(policy, history)
    assert policy.has_recorded_history_boundary
    complete(policy, policy.prepare_request(request(0)))
    assert policy.has_recorded_history_boundary
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert not worker.submit(lambda: policy.has_recorded_history_boundary).result()


def test_llm_history_boundary_exposure_does_not_mutate_original_or_new_clone() -> None:
    original = model()
    cheap = original.with_asv3_exp_cheap_cache()
    assert not original.asv3_exp_cheap_has_history_cache_boundary
    assert not cheap.asv3_exp_cheap_has_history_cache_boundary
    policy = cheap._asv3_exp_cheap_cache
    assert policy is not None
    complete(policy, policy.prepare_request(request(1)))
    assert cheap.asv3_exp_cheap_has_history_cache_boundary
    assert not original.asv3_exp_cheap_has_history_cache_boundary
    assert (
        not cheap.with_asv3_exp_cheap_cache().asv3_exp_cheap_has_history_cache_boundary
    )


def test_prior_written_endpoint_survives_a_batch_of_new_tools_without_input_changes() -> (
    None
):
    policy = ASv3ExpCheapCachePolicy()
    first = request(1)
    before = copy.deepcopy(first)
    complete(policy, policy.prepare_request(first))
    assert strip_cache_metadata(first) == before
    later = request(8)
    before_later = copy.deepcopy(later)
    policy.prepare_request(later)
    assert strip_cache_metadata(later) == before_later
    assert request_bytes(strip_cache_metadata(later)) == request_bytes(before_later)
    assert later["extra_body"]["prompt_cache_options"]["mode"] == "explicit"
    assert "prompt_cache_breakpoint" not in later["input"][-1]["content"][0]
    marked = [
        index
        for index, item in enumerate(later["input"])
        if item.get("type") == "function_call_output"
        and "prompt_cache_breakpoint" in item["output"][-1]
    ]
    assert marked == [2, 16]
    assert (
        later["extra_body"]["prompt_cache_options"]["comparison_response_id"]
        == "resp-previous"
    )


def test_changed_tool_prefix_does_not_retain_stale_boundary_and_threads_are_separate() -> (
    None
):
    policy = ASv3ExpCheapCachePolicy()
    first = request(1)
    complete(policy, policy.prepare_request(first))
    changed = request(2)
    changed["tools"] = [{"type": "function", "name": "changed"}]
    policy.prepare_request(changed)
    assert "prompt_cache_breakpoint" not in changed["input"][2]["output"][0]
    with ThreadPoolExecutor(max_workers=1) as worker:
        isolated = request(1)
        worker.submit(policy.prepare_request, isolated).result()
    assert (
        "comparison_response_id" not in isolated["extra_body"]["prompt_cache_options"]
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        "malformed",
        {"input_tokens_details": "malformed"},
        {"input_tokens_details": {"cache_write_tokens": "malformed"}},
    ],
)
def test_optional_response_metadata_cannot_break_or_change_provider_events(
    usage: Any,
) -> None:
    policy = ASv3ExpCheapCachePolicy()
    capture = policy.prepare_request(request(1))
    event = {
        "type": "response.completed",
        "response": {
            "id": None,
            "usage": usage,
            "prompt_cache_diagnostics": ["malformed"],
        },
    }
    assert list(policy.capture_response(iter([event]), capture)) == [event]
    assert capture.cache_write_tokens is None and capture.diagnostics is None


class EventStream(httpx.SyncByteStream):
    def __init__(self, *, stalled: bool = False) -> None:
        self.closed = False
        self.arguments = '{"answer":"MADDE 10/A; AND/OR; istisna; 01.01.2026"}'
        self.events: list[dict[str, Any]] = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-true",
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
                    "call_id": "call-new",
                    "name": "read",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "fc-test",
                "delta": '{"task":' + "\t" * TOOL_ARGUMENT_WHITESPACE_RUN
                if stalled
                else self.arguments,
            },
            {
                "type": "response.completed",
                "sequence_number": 3,
                "response": {
                    "id": "resp-true",
                    "created_at": 1,
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-6.1-sol",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-new",
                            "name": "read",
                            "arguments": self.arguments,
                        }
                    ],
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 20,
                        "total_tokens": 1020,
                        "input_tokens_details": {
                            "cached_tokens": 100,
                            "cache_write_tokens": 500,
                        },
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                    "prompt_cache_diagnostics": {"type": "cache_hit"},
                },
            },
        ]

    def __iter__(self) -> Iterator[bytes]:
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    def close(self) -> None:
        self.closed = True


def invoke(
    model: LitellmLLM, events: EventStream
) -> tuple[dict[str, Any], ModelResponse, dict[str, Any]]:
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        stream=events,
        headers={"content-type": "text/event-stream"},
    )
    messages = [
        SystemMessage(content="Retain every condition, exception, date, and original"),
        UserMessage(content="Exact original question"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-old",
                    function=FunctionCall(
                        name="read", arguments='{"source":"verbatim"}'
                    ),
                )
            ],
        ),
        ToolMessage(
            tool_call_id="call-old",
            content="MADDE 10/A\nŞartlar AND/OR; istisna.\n" + " " * 10000,
        ),
        UserMessage(content='{"metadata":"retain all","draft":"full answer"}'),
    ]
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
        trace("cheap cache transport"),
        generation_span(model="gpt-6.1-sol") as span,
    ):
        response = model.invoke(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                }
            ],
            max_tokens=128000,
            reasoning_effort=ReasoningEffort.HIGH,
        )
    ordinary.assert_not_called()
    assert events.closed
    return (
        client.post.call_args.kwargs["json"],
        response,
        dict(span.span_data.request_params or {}),
    )


def model(*, api_base: str | None = None) -> LitellmLLM:
    return LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=900000,
        api_base=api_base,
        extra_body={"store": False},
    ).with_tool_argument_stream_guard()


def test_opt_in_actual_http_body_matches_baseline_and_preserves_complete_output_and_usage() -> (
    None
):
    original = model()
    cheap = original.with_asv3_exp_cheap_cache()
    assert cheap is not original and original._asv3_exp_cheap_cache is None
    assert cheap._model_kwargs is not original._model_kwargs
    baseline_body, baseline, _ = invoke(original, EventStream())
    cheap_body, optimized, metadata = invoke(cheap, EventStream())
    assert strip_cache_metadata(cheap_body) == baseline_body
    assert request_bytes(strip_cache_metadata(cheap_body)) == request_bytes(
        baseline_body
    )
    assert cheap_body["prompt_cache_options"]["mode"] == "explicit"
    assert "prompt_cache_breakpoint" not in cheap_body["input"][-1]["content"][0]
    assert baseline.choice.message.tool_calls == optimized.choice.message.tool_calls
    assert optimized.usage and baseline.usage
    assert baseline.usage.cache_creation_input_tokens == 0
    assert optimized.usage.cache_creation_input_tokens == 500
    assert optimized.usage.cache_read_input_tokens == 100
    assert metadata["reasoning_effort"] == "high"
    assert metadata["asv3_exp_cheap_cache"] == {
        "provider_response_id": "resp-true",
        "cache_write_tokens": 500,
        "prompt_cache_diagnostics": {"type": "cache_hit"},
        "metadata_source": "native_response_completed",
    }
    assert "prompt_cache_options" not in original._model_kwargs["extra_body"]


def test_opt_in_retains_early_stream_cancellation_ownership() -> None:
    _, response, metadata = invoke(
        model().with_asv3_exp_cheap_cache(), EventStream(stalled=True)
    )
    assert response.usage is None
    assert metadata["asv3_exp_cheap_cache"] == {
        "provider_response_id": None,
        "cache_write_tokens": None,
        "prompt_cache_diagnostics": None,
        "metadata_source": "native_request",
    }


@pytest.mark.parametrize("api_base", ["https://proxy.example/v1"])
def test_opt_in_does_not_enable_guarded_native_route_for_a_proxy(api_base: str) -> None:
    assert (
        not model(api_base=api_base)
        .with_asv3_exp_cheap_cache()
        ._uses_guarded_openai_responses()
    )
