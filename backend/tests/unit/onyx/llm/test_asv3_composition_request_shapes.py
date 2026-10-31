"""Real provider request shaping for registered composition models, without network."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler

from onyx.asv3.model_composition.profiles import MODEL_SPECS
from onyx.llm.model_capabilities import is_true_openai_model
from onyx.llm.models import (
    ReasoningEffort,
    SystemMessage,
    ToolChoiceOptions,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_provision",
            "parameters": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}},
                "required": ["source_id"],
                "additionalProperties": False,
            },
        },
    }
]


def selected(provider: str, name: str) -> LitellmLLM:
    return LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=name,
        max_input_tokens=900_000,
    )


@pytest.mark.parametrize("alias", list(MODEL_SPECS))
def test_all_seven_registered_routes_preserve_native_tools_and_low_request_shape(
    alias: str,
) -> None:
    spec = MODEL_SPECS[alias]
    current = selected(spec.provider_type, spec.model_name)
    if spec.provider_type == "openai":
        assert is_true_openai_model(spec.provider_type, spec.model_name)
    with patch("litellm.completion", return_value=[]) as completion:
        assert (
            list(
                current.stream(
                    [UserMessage(content="Read the complete operative provision.")],
                    tools=TOOLS,
                    tool_choice=ToolChoiceOptions.AUTO,
                    reasoning_effort=ReasoningEffort.LOW,
                    max_tokens=1024,
                )
            )
            == []
        )
    sent = completion.call_args.kwargs
    expected = (
        f"openai/responses/{spec.model_name}"
        if spec.provider_type == "openai"
        else f"{spec.provider_type}/{spec.model_name}"
    )
    assert sent["model"] == expected
    assert sent["tools"] == TOOLS
    assert sent["tool_choice"] == "auto"
    assert sent["messages"][0]["content"] == "Read the complete operative provision."
    if spec.provider_type == "openai":
        assert sent["reasoning"]["effort"] == "low"
    if spec.provider_type == "anthropic":
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"]["effort"] == "low"


@pytest.mark.parametrize("name", ["gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"])
def test_real_openai_bridge_posts_responses_native_tools_and_returns_function_call(
    name: str,
) -> None:
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        json={
            "id": "resp-test",
            "created_at": 1,
            "object": "response",
            "status": "completed",
            "model": name,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-test",
                    "call_id": "call-0",
                    "name": "read_provision",
                    "arguments": '{"source_id":"retained"}',
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        },
    )
    current = selected("openai", name)
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("httpx.Client.send", side_effect=AssertionError("network forbidden")),
    ):
        result = current.invoke(
            [
                SystemMessage(content="Preserve every original condition."),
                UserMessage(content="Full original evidence."),
            ],
            tools=TOOLS,
            tool_choice=ToolChoiceOptions.AUTO,
            reasoning_effort=ReasoningEffort.LOW,
            max_tokens=1024,
            use_streaming=False,
        )
    calls = result.choice.message.tool_calls
    assert calls and calls[0].function.name == "read_provision"
    assert calls[0].function.arguments == '{"source_id":"retained"}'
    request = client.post.call_args
    url = request.kwargs.get("url") or request.args[0]
    assert url == "https://api.openai.com/v1/responses"
    body: dict[str, Any] = request.kwargs["json"]
    assert body["model"] == name
    assert body["max_output_tokens"] == 1024
    assert body["reasoning"]["effort"] == "low"
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["name"] == "read_provision"
    assert body["instructions"] == "Preserve every original condition."
    assert body["input"][0]["content"][0]["text"] == "Full original evidence."
    client.close.assert_called_once()
