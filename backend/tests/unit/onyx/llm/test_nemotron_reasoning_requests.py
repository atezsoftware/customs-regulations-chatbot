"""Ensure Ultra's default-on reasoning cannot consume the summary's output budget."""

from typing import Any
from unittest.mock import patch

import pytest

from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


@pytest.mark.parametrize("provider", ["openrouter", "nvidia_nim"])
@pytest.mark.parametrize("registered_reasoning_model", [True, False])
def test_ultra_off_sends_explicit_none_with_full_summary_output_budget(
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
