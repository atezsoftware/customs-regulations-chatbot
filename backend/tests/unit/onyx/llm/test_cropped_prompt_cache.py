import copy
import json
from dataclasses import FrozenInstanceError, asdict
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import BadRequestError, HTTPHandler

from onyx.llm.asv3_exp_cheap_cache import ASv3ExpCheapCachePolicy
from onyx.llm.cropped_prompt_cache import CroppedPromptCachePolicy
from onyx.llm.models import ChatCompletionMessage, SystemMessage, UserMessage
from onyx.llm.native_responses_stream import (
    CroppedResponseTerminalCapture,
    native_responses_stream,
)
from onyx.tracing.framework.create import generation_span, trace
from tests.unit.onyx.llm.test_cropped_response_terminal_capture import (
    REPORTED_USAGE,
    TOOLS,
    events,
    invoke,
    model,
)
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream


def without_cache_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_cache_metadata(item)
            for key, item in value.items()
            if key not in {"prompt_cache_options", "prompt_cache_breakpoint"}
        }
    if isinstance(value, list):
        return [without_cache_metadata(item) for item in value]
    return value


def test_zero_boundaries_preserve_all_instruction_and_source_blocks() -> None:
    request: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "reasoning": {"effort": "medium", "summary": "auto"},
        "tools": copy.deepcopy(TOOLS),
        "input": [
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "Stable instruction A"},
                    {"type": "input_text", "text": "Stable instruction B"},
                ],
            },
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "Later instructions"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Full original source\n" + " " * 32}
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "read1",
                "output": [{"type": "input_text", "text": "Exact original result"}],
            },
        ],
        "extra_body": {"store": False},
    }
    baseline = copy.deepcopy(request)
    CroppedPromptCachePolicy().prepare_request(request)
    assert request["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    assert '"prompt_cache_breakpoint"' not in json.dumps(request)
    assert json.dumps(without_cache_metadata(request)) == json.dumps(baseline)
    assert "prompt_cache_breakpoint" not in baseline["input"][0]["content"][1]


@pytest.mark.parametrize(
    "inputs",
    [
        "Exact input",
        [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Only dynamic input"}],
            }
        ],
        [{"role": "developer", "content": "Unconverted exact instruction"}],
    ],
)
def test_no_eligible_instruction_block_adds_no_synthetic_boundary(inputs: Any) -> None:
    request: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "input": copy.deepcopy(inputs),
        "extra_body": {"store": False},
    }
    original = copy.deepcopy(request)
    CroppedPromptCachePolicy().prepare_request(request)
    assert "prompt_cache_breakpoint" not in json.dumps(request)
    extra_body = cast(dict[str, Any], request["extra_body"])
    assert extra_body["prompt_cache_options"]["mode"] == "explicit"
    assert without_cache_metadata(request) == original


def test_real_sdk_native_body_is_identical_after_cache_metadata_is_removed() -> None:
    prompt: list[ChatCompletionMessage] = [
        SystemMessage(content="Stable task instructions."),
        UserMessage(
            content="Full legal original; source/chunk/hash retained.\n" + " " * 20
        ),
    ]
    with trace("Cropped stable prefix"), generation_span(model="gpt-6.1-sol") as span:
        response, wire = invoke(
            model(explicit_cache=True),
            EventStream(events("completed", REPORTED_USAGE)),
            prompt=prompt,
        )
    baseline, ordinary = invoke(
        model(capture=False),
        EventStream(events("completed", REPORTED_USAGE)),
        prompt=prompt,
    )
    assert response.choice == baseline.choice
    assert response.usage == baseline.usage
    assert json.dumps(without_cache_metadata(wire)) == json.dumps(ordinary)
    assert wire["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    # LiteLLM extracts the actual string system message into instructions.
    assert wire["instructions"] == prompt[0].content
    assert "prompt_cache_breakpoint" not in json.dumps(wire)
    assert span.span_data.request_params is not None
    params = span.span_data.request_params
    assert params["asv3_exp_cropped_prompt_cache"]["marker_message_index"] is None
    assert params["asv3_exp_cropped_prompt_cache"]["marker_count"] == 0
    proof = model(explicit_cache=True).cropped_native_cache_write_contract
    assert proof is not None
    assert params["asv3_exp_cropped_prompt_cache"]["native_contract"] == asdict(proof)
    assert "previous_response_id" not in wire
    assert (
        params["asv3_exp_cropped_response_terminal"]["provider_reported_usage"]
        == REPORTED_USAGE
    )
    assert params["max_tokens"] == 12


def test_real_sdk_clears_existing_developer_markers_and_preserves_all_content() -> None:
    messages = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Exact stable instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"role": "user", "content": "Original body with exact source/chunk/hash"},
    ]
    wires: list[dict[str, Any]] = []
    for policy in [None, CroppedPromptCachePolicy()]:
        client = MagicMock(spec=HTTPHandler)
        stream = EventStream(events("completed", REPORTED_USAGE))
        client.post.return_value = httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            stream=stream,
            headers={"content-type": "text/event-stream"},
        )
        capture = CroppedResponseTerminalCapture()
        response = native_responses_stream(
            model="gpt-6.1-sol",
            messages=copy.deepcopy(messages),
            tools=copy.deepcopy(TOOLS),
            optional_params={"temperature": 1},
            passthrough_kwargs={"api_key": "test-key", "extra_body": {"store": False}},
            api_base="https://api.openai.com/v1",
            timeout=30,
            max_tokens=12,
            client=client,
            cache_policy=policy,
            terminal_capture=capture,
        )
        assert list(response)
        assert capture.status == "completed" and capture.usage is not None
        assert capture.usage.cache_creation_input_tokens == 0
        assert stream.closed
        wires.append(client.post.call_args.kwargs["json"])
    baseline, explicit = wires
    assert json.dumps(without_cache_metadata(explicit)) == json.dumps(
        without_cache_metadata(baseline)
    )
    assert "prompt_cache_breakpoint" not in json.dumps(explicit)
    assert explicit["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert "previous_response_id" not in explicit


def test_cropped_clone_does_not_bind_cheap_history_policy_or_mutate_original() -> None:
    original = model(capture=False)
    cheap = original.with_asv3_exp_cheap_cache()
    assert original._cropped_prompt_cache is None
    assert not original.cropped_native_cache_writes_disabled
    assert cheap._cropped_prompt_cache is None
    assert isinstance(cheap._asv3_exp_cheap_cache, ASv3ExpCheapCachePolicy)
    terminal_only = original.with_cropped_response_terminal_capture()
    assert terminal_only._cropped_prompt_cache is None
    assert terminal_only.cropped_native_cache_write_contract is None
    bound = original.with_cropped_response_terminal_capture(explicit_cache=True)
    assert isinstance(bound._cropped_prompt_cache, CroppedPromptCachePolicy)
    assert bound.cropped_native_cache_writes_disabled
    proof = bound.cropped_native_cache_write_contract
    assert proof is not None
    assert asdict(proof) == {
        "schema_version": 1,
        "policy_id": "cropped_native_explicit_zero_boundaries_v1",
        "mode": "explicit",
        "ttl": "30m",
        "marked_boundary_count": 0,
        "provider": "openai",
        "model": "gpt-6.1-sol",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(proof, "marked_boundary_count", 1)
    with pytest.raises(AttributeError):
        setattr(bound, "cropped_native_cache_writes_disabled", False)
    assert bound._asv3_exp_cheap_cache is None
    assert not bound.asv3_exp_cheap_has_history_cache_boundary
    assert not original._cropped_response_terminal_capture
    assert original._model_kwargs == bound._model_kwargs == cheap._model_kwargs
    assert bound.with_cropped_response_terminal_capture()._cropped_prompt_cache is None


def test_terminal_only_capture_preserves_existing_cheap_cache_wire() -> None:
    original = model(capture=False).with_asv3_exp_cheap_cache()
    bound = original.with_cropped_response_terminal_capture()
    assert bound._asv3_exp_cheap_cache is original._asv3_exp_cheap_cache
    assert bound._cropped_prompt_cache is None
    prompt: list[ChatCompletionMessage] = [
        SystemMessage(content="Stable instructions"),
        UserMessage(content="Exact source body"),
    ]
    # Use independent policies so the comparison does not share completed-call state.
    _, baseline_wire = invoke(
        original, EventStream(events("completed", REPORTED_USAGE)), prompt=prompt
    )
    _, wire = invoke(
        model(capture=False)
        .with_asv3_exp_cheap_cache()
        .with_cropped_response_terminal_capture(),
        EventStream(events("completed", REPORTED_USAGE)),
        prompt=prompt,
    )
    assert json.dumps(wire) == json.dumps(baseline_wire)


def test_existing_boundaries_removed_without_changing_literal_source_json() -> None:
    text = '{"prompt_cache_breakpoint":{"mode":"explicit"},"source":"literal"}\n  '
    inputs: list[dict[str, Any]] = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Exact instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "read1",
            "output": [
                {
                    "type": "input_text",
                    "text": text,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
    ]
    request: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "input": copy.deepcopy(inputs),
        "extra_body": {
            "input": copy.deepcopy(inputs),
            "store": False,
            "prompt_cache_options": {
                "mode": "implicit",
                "comparison_response_id": "resp-old",
            },
        },
    }
    original = copy.deepcopy(request)
    CroppedPromptCachePolicy().prepare_request(request)
    assert request["input"][1]["output"][0]["text"] == text
    assert request["extra_body"]["input"][1]["output"][0]["text"] == text
    assert json.dumps(without_cache_metadata(request)) == json.dumps(
        without_cache_metadata(original)
    )
    assert request["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    assert request["input"][0]["content"][0].get("prompt_cache_breakpoint") is None
    assert original["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


@pytest.mark.parametrize(
    "extra_body",
    [{"previous_response_id": "resp-old"}, {"model": "other-model"}, {"stream": False}],
)
def test_implicit_history_or_routing_override_rejected_before_transport(
    extra_body: dict[str, Any],
) -> None:
    request = {
        "model": "gpt-6.1-sol",
        "input": "All explicit input",
        "extra_body": extra_body,
    }
    original = copy.deepcopy(request)
    with pytest.raises(ValueError, match="Zero-write Cropped"):
        CroppedPromptCachePolicy().prepare_request(request)
    assert request == original


@pytest.mark.parametrize(
    "use_streaming,tools", [(False, TOOLS), (True, None), (True, [])]
)
def test_zero_write_contract_rejects_non_native_invoke_before_client_creation(
    use_streaming: bool, tools: list[dict[str, Any]] | None
) -> None:
    bound = model(explicit_cache=True)
    with (
        patch("litellm.HTTPHandler") as client,
        patch("litellm.completion") as ordinary,
        pytest.raises(ValueError, match="guarded native tool streaming"),
    ):
        bound.invoke(
            [UserMessage(content="Full literal")],
            tools=tools,
            use_streaming=use_streaming,
        )
    client.assert_not_called()
    ordinary.assert_not_called()


def test_zero_write_contract_rejects_unguarded_proxy_and_changed_model() -> None:
    with pytest.raises(ValueError, match="first-party guarded Responses"):
        model(api_base="https://proxy.example/v1", explicit_cache=True)
    bound = model(explicit_cache=True).with_model("other-model", max_input_tokens=10000)
    assert not bound.cropped_native_cache_writes_disabled
    assert bound.cropped_native_cache_write_contract is None
    with (
        patch("litellm.HTTPHandler") as client,
        pytest.raises(ValueError, match="guarded native tool streaming"),
    ):
        bound.invoke([UserMessage(content="Full literal")], tools=TOOLS)
    client.assert_not_called()


def test_zero_write_contract_rejects_capture_missing_direct_completion() -> None:
    bound = model(explicit_cache=True)
    with (
        patch("litellm.completion") as ordinary,
        pytest.raises(ValueError, match="guarded native tool streaming"),
    ):
        bound._completion(
            [UserMessage(content="Full literal")],
            tools=TOOLS,
            tool_choice=None,
            stream=True,
            parallel_tool_calls=True,
        )
    ordinary.assert_not_called()


def test_zero_write_policy_does_not_replace_actual_provider_usage() -> None:
    actual = copy.deepcopy(REPORTED_USAGE)
    actual["input_tokens_details"] = {"cached_tokens": 4, "cache_write_tokens": 20}
    with trace("Actual write usage"), generation_span(model="gpt-6.1-sol") as span:
        response, wire = invoke(
            model(explicit_cache=True), EventStream(events("completed", actual))
        )
    assert "prompt_cache_breakpoint" not in json.dumps(wire)
    assert response.usage.cache_creation_input_tokens == 20
    assert span.span_data.request_params is not None
    assert (
        span.span_data.request_params["asv3_exp_cropped_response_terminal"][
            "provider_reported_usage"
        ]
        == actual
    )


def test_unsupported_cache_option_fails_closed_without_compatibility_retry() -> None:
    requests: list[httpx.Request] = []

    def unsupported(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": "Unsupported parameter: prompt_cache_options",
                    "type": "invalid_request_error",
                    "param": "prompt_cache_options",
                    "code": "unsupported_parameter",
                }
            },
        )

    client = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(unsupported))
    )
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
        pytest.raises(BadRequestError, match="prompt_cache_options"),
    ):
        model(explicit_cache=True).invoke(
            [
                SystemMessage(content="Exact instructions"),
                UserMessage(content="Full unchanged sources"),
            ],
            tools=copy.deepcopy(TOOLS),
            max_tokens=12,
        )
    assert len(requests) == 1
    assert client.client.is_closed
    ordinary.assert_not_called()
