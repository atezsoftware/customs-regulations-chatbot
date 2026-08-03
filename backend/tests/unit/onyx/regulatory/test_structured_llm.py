from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ReasoningEffort
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow


class _TinyResult(BaseModel):
    value: str


class _ConstrainedResult(BaseModel):
    values: list[str] = Field(min_length=2, max_length=3)
    index: int = Field(ge=0, lt=10)


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        id="test-response",
        created="2026-08-01T00:00:00Z",
        choice=Choice(message=Message(content=content)),
    )


def _generate(
    llm: MagicMock,
    *,
    timeout_override: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_attempts: int = 2,
) -> _TinyResult:
    with (
        patch(
            "onyx.regulatory.structured_llm.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.regulatory.structured_llm.record_llm_response"),
    ):
        return generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_ANSWER_AUDIT,
            system_prompt="Return the requested data.",
            user_prompt="payload",
            response_model=_TinyResult,
            timeout_override=timeout_override,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            max_attempts=max_attempts,
        )


def test_generate_structured_forwards_optional_invoke_limits() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response('{"value":"ok"}')

    result = _generate(
        llm,
        timeout_override=17,
        max_tokens=321,
        reasoning_effort=ReasoningEffort.OFF,
        max_attempts=3,
    )

    assert result == _TinyResult(value="ok")
    invoke_kwargs = llm.invoke.call_args.kwargs
    assert invoke_kwargs["timeout_override"] == 17
    assert invoke_kwargs["max_tokens"] == 321
    assert invoke_kwargs["reasoning_effort"] is ReasoningEffort.OFF
    assert "structured_response_format" in invoke_kwargs


def test_generate_structured_omits_unsupplied_optional_invoke_limits() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response('{"value":"ok"}')

    _generate(llm)

    invoke_kwargs = llm.invoke.call_args.kwargs
    assert "timeout_override" not in invoke_kwargs
    assert "max_tokens" not in invoke_kwargs
    assert "reasoning_effort" not in invoke_kwargs


def test_generate_structured_retries_validation_failure() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = [
        _response('{"wrong":"shape"}'),
        _response('{"value":"repaired"}'),
    ]

    result = _generate(llm, max_attempts=3)

    assert result.value == "repaired"
    assert llm.invoke.call_count == 2
    retry_messages = llm.invoke.call_args.args[0]
    assert len(retry_messages) == 3
    assert "previous response was not valid JSON" in retry_messages[-1].content


def test_generate_structured_rejects_zero_attempts_without_invoking() -> None:
    llm = MagicMock()

    with pytest.raises(ValueError, match="at least 1"):
        _generate(llm, max_attempts=0)

    llm.invoke.assert_not_called()


def test_generate_structured_sends_portable_provider_schema() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response('{"values":["a","b"],"index":1}')

    with (
        patch(
            "onyx.regulatory.structured_llm.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.regulatory.structured_llm.record_llm_response"),
    ):
        result = generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_ANSWER_AUDIT,
            system_prompt="Return the requested data.",
            user_prompt="payload",
            response_model=_ConstrainedResult,
        )

    assert result.index == 1
    provider_schema = llm.invoke.call_args.kwargs["structured_response_format"][
        "json_schema"
    ]["schema"]
    serialized_schema = str(provider_schema)
    for unsupported_key in (
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maxLength",
        "maximum",
        "minLength",
        "minimum",
        "multipleOf",
    ):
        assert unsupported_key not in serialized_schema
    assert provider_schema["properties"]["values"]["minItems"] == 1
