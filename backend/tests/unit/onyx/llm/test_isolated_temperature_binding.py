from typing import Any
from unittest.mock import patch

import pytest

from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM


def _llm(model: str, provider: str = "vertex_ai") -> LitellmLLM:
    return LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=model,
        max_input_tokens=1048576,
        temperature=0.0,
    )


def _sent_kwargs(llm: LitellmLLM) -> dict[str, Any]:
    with patch("litellm.completion", return_value=[]) as call:
        list(
            llm.stream(
                [UserMessage(content="Hello")], reasoning_effort=ReasoningEffort.HIGH
            )
        )
        return dict(call.call_args.kwargs)


def test_explicit_gemini_sampling_reaches_provider_without_mutating_shared_model() -> (
    None
):
    llm = _llm("gemini-3.8-flash")
    low = llm.with_temperature(0.1)
    other = llm.with_temperature(0.8)
    assert _sent_kwargs(low)["temperature"] == 0.1
    assert _sent_kwargs(other)["temperature"] == 0.8
    assert _sent_kwargs(llm)["temperature"] == 1
    assert llm.config.temperature == 0.0
    assert low.config.temperature == 0.1
    assert low.config.seed is None
    assert low.config.max_input_tokens == llm.config.max_input_tokens


def test_unsupported_sonnet_sampling_is_still_omitted() -> None:
    llm = _llm("claude-sonnet-5-5", provider="anthropic").with_temperature(0.1)
    assert "temperature" not in _sent_kwargs(llm)


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("nan"), float("inf"), True])
def test_invalid_sampling_is_rejected_before_provider_call(temperature: float) -> None:
    with pytest.raises(ValueError, match="Temperature"):
        _llm("gemini-3.8-flash").with_temperature(temperature)
