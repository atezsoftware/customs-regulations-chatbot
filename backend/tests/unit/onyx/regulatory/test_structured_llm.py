from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ReasoningEffort
from onyx.llm.multi_llm import LLMRateLimitError
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow


class _TinyResult(BaseModel):
    value: str


class _ConstrainedResult(BaseModel):
    values: list[str] = Field(min_length=2, max_length=3)
    index: int = Field(ge=0, lt=10)


def _response(content: str, *, finish_reason: str | None = None) -> ModelResponse:
    return ModelResponse(
        id="test-response",
        created="2026-08-01T00:00:00Z",
        choice=Choice(
            finish_reason=finish_reason,
            message=Message(content=content),
        ),
    )


def _generate(
    llm: MagicMock,
    *,
    timeout_override: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_attempts: int = 2,
    provider_max_attempts: int = 3,
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
            provider_max_attempts=provider_max_attempts,
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
    assert len(retry_messages) == 4
    assert retry_messages[-2].content == '{"wrong":"shape"}'
    assert "previous response was not valid JSON" in retry_messages[-1].content


def test_generate_structured_extracts_schema_matching_json_from_prose() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response(
        'Evaluation notes {"wrong":"shape"}.\n'
        'Final result:\n```json\n{"value":"recovered"}\n```\nDone.'
    )

    result = _generate(llm, max_attempts=1)

    assert result.value == "recovered"
    llm.invoke.assert_called_once()


def test_generate_structured_retries_transient_provider_error_separately() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = [
        LLMRateLimitError("capacity window exhausted"),
        _response('{"value":"recovered"}'),
    ]

    with (
        patch("onyx.regulatory.structured_llm.random.uniform", return_value=2.25),
        patch("onyx.regulatory.structured_llm.time.sleep") as sleep,
        patch("onyx.regulatory.structured_llm.LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S", 2.0),
        patch("onyx.regulatory.structured_llm.LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S", 10.0),
        patch(
            "onyx.regulatory.structured_llm.LLM_FIRST_CHUNK_RETRY_JITTER_RATIO", 0.25
        ),
    ):
        result = _generate(llm, max_attempts=1, provider_max_attempts=3)

    assert result.value == "recovered"
    assert llm.invoke.call_count == 2
    assert llm.invoke.call_args_list[0].args[0] == llm.invoke.call_args_list[1].args[0]
    sleep.assert_called_once_with(2.25)


def test_generate_structured_restarts_without_echoing_truncated_json() -> None:
    llm = MagicMock()
    truncated = '{"value":"' + ("x" * 40_000)
    llm.invoke.side_effect = [
        _response(truncated),
        _response('{"value":"repaired"}'),
    ]

    result = _generate(llm, max_attempts=2)

    assert result.value == "repaired"
    retry_messages = llm.invoke.call_args.args[0]
    assert len(retry_messages) == 3
    assert all(truncated not in message.content for message in retry_messages)
    assert "truncated" in retry_messages[-1].content
    assert "from scratch" in retry_messages[-1].content


def test_generate_structured_treats_output_limit_finish_as_truncation() -> None:
    llm = MagicMock()
    invalid = '{"wrong":"shape"}'
    llm.invoke.side_effect = [
        _response(invalid, finish_reason="length"),
        _response('{"value":"repaired"}', finish_reason="stop"),
    ]

    result = _generate(llm, max_attempts=2)

    assert result.value == "repaired"
    retry_messages = llm.invoke.call_args.args[0]
    assert len(retry_messages) == 3
    assert all(invalid not in message.content for message in retry_messages)
    assert "truncated" in retry_messages[-1].content


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
