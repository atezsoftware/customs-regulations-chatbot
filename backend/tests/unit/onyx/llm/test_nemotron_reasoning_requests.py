"""Ensure Ultra's default-on reasoning cannot consume the summary's output budget."""

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


@pytest.mark.parametrize("provider", ["openrouter", "nvidia_nim"])
@pytest.mark.parametrize("registered_reasoning_model", [True, False])
def test_ultra_off_sends_explicit_disable_with_full_summary_output_budget(
    provider: str, registered_reasoning_model: bool
) -> None:
    llm = LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=ULTRA_MODEL,
        max_input_tokens=200_000,
        timeout=90,
        temperature=0,
    )
    with (
        patch(
            "onyx.llm.multi_llm.model_is_reasoning_model",
            return_value=registered_reasoning_model,
        ),
        patch(
            "onyx.llm.litellm_singleton.litellm.completion", return_value=object()
        ) as completion,
    ):
        llm._completion(
            prompt=[UserMessage(content="Kaynakta Madde 1 bulundu [1].")],
            tools=None,
            tool_choice=None,
            stream=False,
            parallel_tool_calls=False,
            reasoning_effort=ReasoningEffort.OFF,
            max_tokens=1800,
        )
    sent = completion.call_args.kwargs
    assert sent["model"] == f"{provider}/{ULTRA_MODEL}"
    if provider == "openrouter":
        assert sent["extra_body"]["reasoning"] == {"enabled": False}
    else:
        assert sent["reasoning_effort"] == "none"
    assert sent["max_tokens"] == 1800
    assert sent["stream"] is False


@pytest.mark.parametrize(
    ("provider", "model", "effort", "expected"),
    [
        ("openrouter", ULTRA_MODEL, ReasoningEffort.MEDIUM, "medium"),
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b", ReasoningEffort.OFF, None),
        (
            "openrouter",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            ReasoningEffort.OFF,
            None,
        ),
        ("anthropic", ULTRA_MODEL, ReasoningEffort.OFF, None),
    ],
)
def test_ultra_off_mapping_leaves_other_models_providers_and_efforts_unchanged(
    provider: str, model: str, effort: ReasoningEffort, expected: str | None
) -> None:
    llm = LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=model,
        max_input_tokens=200_000,
    )
    with (
        patch("onyx.llm.multi_llm.model_is_reasoning_model", return_value=True),
        patch(
            "onyx.llm.litellm_singleton.litellm.completion", return_value=object()
        ) as completion,
    ):
        llm._completion(
            prompt=[UserMessage(content="Kaynakları tanıt.")],
            tools=None,
            tool_choice=None,
            stream=False,
            parallel_tool_calls=False,
            reasoning_effort=effort,
        )
    sent: dict[str, Any] = dict(completion.call_args.kwargs)
    assert sent.get("reasoning_effort") == expected


def test_ultra_disable_survives_litellm_provider_transformation_on_http_wire() -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "mock-response",
                "object": "chat.completion",
                "created": 0,
                "model": ULTRA_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Bölüm bulundu [1].",
                        },
                    }
                ],
            },
        )

    existing_body: dict[str, Any] = {
        "provider": {"sort": "latency"},
        "reasoning": {"enabled": True},
    }
    llm = LitellmLLM(
        api_key="test-key",
        model_provider="openrouter",
        model_name=ULTRA_MODEL,
        max_input_tokens=200_000,
        timeout=90,
        extra_body=existing_body,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with patch("onyx.llm.multi_llm.model_is_reasoning_model", return_value=False):
            llm._completion(
                prompt=[UserMessage(content="Kaynakları tanıt.")],
                tools=None,
                tool_choice=None,
                stream=False,
                parallel_tool_calls=False,
                reasoning_effort=ReasoningEffort.OFF,
                max_tokens=1800,
                client=HTTPHandler(client=http_client),
            )
    assert len(requests) == 1
    assert requests[0]["model"] == ULTRA_MODEL
    assert requests[0]["reasoning"] == {"enabled": False}
    assert requests[0]["provider"] == {"sort": "latency"}
    assert requests[0]["max_tokens"] == 1800
    assert existing_body["reasoning"] == {"enabled": True}
