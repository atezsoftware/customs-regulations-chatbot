"""Only composed Sol search uses its supported minimum reasoning setting."""

from unittest.mock import MagicMock

import pytest

from onyx.asv3 import search_adapter
from onyx.asv3.model_composition import (
    MODEL_COMPOSITION_SERVICE,
    ModelBinding,
    ModelComposition,
)
from onyx.asv3.model_composition.models import COMPOSITION_ROLES
from onyx.asv3.models import RunContext
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.tracing.framework.span_data import GenerationSpanData
from tests.unit.onyx.asv3.test_runtime import response


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "provider,model_name,composed,requested,expected",
    [
        ("openai", "gpt-6.1-sol", True, ReasoningEffort.OFF, ReasoningEffort.LOW),
        ("openai", "gpt-6.1-terra", True, ReasoningEffort.OFF, ReasoningEffort.OFF),
        ("openai", "gpt-6-luna", True, ReasoningEffort.OFF, ReasoningEffort.OFF),
        (
            "vertex_ai",
            "gemini-3.5-flash-lite",
            True,
            ReasoningEffort.OFF,
            ReasoningEffort.OFF,
        ),
        ("openai", "gpt-6.1-sol", False, ReasoningEffort.OFF, ReasoningEffort.OFF),
        ("openai", "gpt-6.1-sol", True, ReasoningEffort.LOW, ReasoningEffort.LOW),
        ("openai", "gpt-6.1-sol", True, ReasoningEffort.HIGH, ReasoningEffort.HIGH),
    ],
)
def test_search_invoke_and_stream_preserve_requested_and_actual_effort(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    provider: str,
    model_name: str,
    composed: bool,
    requested: ReasoningEffort,
    expected: ReasoningEffort,
) -> None:
    selected = MagicMock(spec=LLM)
    selected.config = LLMConfig(
        model_provider=provider,
        model_name=model_name,
        temperature=0,
        max_input_tokens=200000,
    )
    selected.invoke.return_value = response("query")
    selected.stream.return_value = iter([])
    binding = ModelBinding(selected, len, 101)
    bundle = ModelComposition(
        "sol_reference_v1", 1, dict.fromkeys(COMPOSITION_ROLES, binding)
    )
    context = RunContext(
        services={MODEL_COMPOSITION_SERVICE: bundle} if composed else {}
    )
    span = MagicMock()
    span.span_data = GenerationSpanData(
        model=model_name, request_params={"provider_mapped": "preserved"}
    )
    monkeypatch.setattr(search_adapter, "get_current_span", lambda: span)
    scoped = ScopedSearchLLM(selected, context, None)
    if stream:
        assert (
            list(
                scoped.stream(
                    [UserMessage(content="query")], reasoning_effort=requested
                )
            )
            == []
        )
        actual = selected.stream.call_args.kwargs
    else:
        scoped.invoke([UserMessage(content="query")], reasoning_effort=requested)
        actual = selected.invoke.call_args.kwargs
    assert actual["reasoning_effort"] == expected
    parameters = span.span_data.request_params
    assert parameters is not None
    assert parameters["provider_mapped"] == "preserved"
    if composed:
        assert parameters["asv3_requested_search_reasoning_effort"] == requested.value
        assert parameters["asv3_effective_search_reasoning_effort"] == expected.value
        identity = bundle.identity()
        assert identity["requested_search_reasoning_effort"] == "off"
        assert identity["effective_search_reasoning_effort"] == (
            "low" if provider == "openai" and model_name == "gpt-6.1-sol" else "off"
        )
    else:
        assert parameters == {"provider_mapped": "preserved"}
