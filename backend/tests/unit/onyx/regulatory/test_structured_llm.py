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


class _TwoFieldResult(BaseModel):
    first: str
    second: str


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
    deadline: float | None = None,
    use_streaming: bool | None = None,
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
            deadline=deadline,
            use_streaming=use_streaming,
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
    assert "use_streaming" not in invoke_kwargs


def test_generate_structured_can_disable_streaming_for_bounded_source_calls() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response('{"value":"ok"}')

    assert _generate(llm, use_streaming=False).value == "ok"
    assert llm.invoke.call_args.kwargs["use_streaming"] is False


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


def test_generate_structured_terminal_error_identifies_invalid_fields() -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response('{"wrong":"shape"}')

    with pytest.raises(ValueError) as raised:
        _generate(llm, max_attempts=1)

    message = str(raised.value)
    assert "value [missing]: Field required" in message
    assert '"wrong":"shape"' not in message


def test_generate_structured_keeps_best_validation_error_from_embedded_objects() -> (
    None
):
    llm = MagicMock()
    llm.invoke.return_value = _response(
        '{"first":"present","nested":{"unrelated":"value"}}'
    )

    with (
        patch(
            "onyx.regulatory.structured_llm.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.regulatory.structured_llm.record_llm_response"),
        pytest.raises(ValueError) as raised,
    ):
        generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_ANSWER_AUDIT,
            system_prompt="Return the requested data.",
            user_prompt="payload",
            response_model=_TwoFieldResult,
            max_attempts=1,
        )

    message = str(raised.value)
    assert "second [missing]: Field required" in message
    assert "first [missing]: Field required" not in message


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


def test_structured_image_parts_survive_validation_retry() -> None:
    from onyx.llm.models import ImageContentPart, ImageUrlDetail

    llm = MagicMock()
    llm.invoke.side_effect = [_response('{"wrong":1}'), _response('{"value":"ok"}')]
    part = ImageContentPart(
        image_url=ImageUrlDetail(url="data:image/png;base64,aGVsbG8=")
    )
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
            system_prompt="Extract",
            user_prompt="Evidence",
            image_parts=[part],
            response_model=_TinyResult,
        )
    assert result.value == "ok"
    for call in llm.invoke.call_args_list:
        assert call.args[0][1].content[1] == part
        assert call.args[0][1].content[0].text == "Evidence"


@pytest.mark.parametrize("header", ["7", "0", "invalid"])
def test_wrapped_retry_after_and_monotonic_deadline(
    monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    import httpx
    from litellm.exceptions import RateLimitError

    from onyx.regulatory import structured_llm as module

    clock = [100.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module.random, "uniform", lambda _low, _high: 2.0)
    wrapped = LLMRateLimitError("fixture")
    wrapped.__context__ = RateLimitError(
        "fixture",
        llm_provider="openrouter",
        model="fixture",
        response=httpx.Response(429, headers={"Retry-After": header}),
    )
    llm = MagicMock()

    def invoke(*_args: object, **_kwargs: object) -> ModelResponse:
        clock[0] += 1
        if llm.invoke.call_count == 1:
            raise wrapped
        return _response('{"value":"ok"}')

    llm.invoke.side_effect = invoke
    result = _generate(llm, max_attempts=1, deadline=120.0)
    assert result.value == "ok"
    delay = 7.0 if header == "7" else 0.0 if header == "0" else 2.0
    assert sleeps == [delay]
    assert [call.kwargs["timeout_override"] for call in llm.invoke.call_args_list] == [
        20,
        int(19 - delay),
    ]


def test_retry_after_that_cannot_fit_is_never_shortened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from litellm.exceptions import RateLimitError

    from onyx.regulatory import structured_llm as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    sleep = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    error = RateLimitError(
        "fixture",
        llm_provider="openrouter",
        model="fixture",
        headers={"rEtRy-AfTeR": "30"},
        response=httpx.Response(429),
    )
    llm = MagicMock()
    llm.invoke.side_effect = error
    with pytest.raises(RateLimitError):
        _generate(llm, max_attempts=1, deadline=110.0)
    llm.invoke.assert_called_once()
    sleep.assert_not_called()


def test_deadline_rejects_expired_and_late_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory import structured_llm as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    llm = MagicMock()
    with pytest.raises(TimeoutError):
        _generate(llm, deadline=100.0)
    llm.invoke.assert_not_called()

    def late(*_args: object, **_kwargs: object) -> ModelResponse:
        clock[0] = 106.0
        return _response('{"value":"late"}')

    llm.invoke.side_effect = late
    with pytest.raises(TimeoutError):
        _generate(llm, deadline=105.0)


def test_retry_after_date_and_bounded_cycle_use_existing_parser() -> None:
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    import httpx
    from litellm.exceptions import RateLimitError

    from onyx.regulatory import structured_llm as module

    header = format_datetime(
        datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True
    )
    error = RateLimitError(
        "fixture",
        llm_provider="vertex_ai",
        model="fixture",
        response=httpx.Response(429, headers={"Retry-After": header}),
    )
    wrapper = LLMRateLimitError(error)
    error.__context__ = wrapper
    delay = module._retry_after_seconds(wrapper)
    assert delay is not None and 28 <= delay <= 30


@pytest.mark.parametrize("retry_header, attempts", [("100000", 1), ("0", 3)])
def test_provider_delay_ceiling_and_attempt_cap_without_deadline(
    monkeypatch: pytest.MonkeyPatch, retry_header: str, attempts: int
) -> None:
    import httpx
    from litellm.exceptions import RateLimitError

    from onyx.regulatory import structured_llm as module

    sleep = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    error = RateLimitError(
        "fixture",
        llm_provider="vertex_ai",
        model="fixture",
        response=httpx.Response(429, headers={"Retry-After": retry_header}),
    )
    llm = MagicMock()
    llm.invoke.side_effect = error
    with pytest.raises(RateLimitError):
        _generate(llm, max_attempts=1)
    assert llm.invoke.call_count == attempts
    assert sleep.call_count == attempts - 1


def test_nonretryable_provider_failure_never_repeats() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = ValueError("fixture invalid request")
    with (
        patch("onyx.regulatory.structured_llm.time.sleep") as sleep,
        pytest.raises(ValueError),
    ):
        _generate(llm)
    llm.invoke.assert_called_once()
    sleep.assert_not_called()


def test_validation_must_also_finish_within_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory import structured_llm as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def validate(_content: str, _model: type[BaseModel]) -> _TinyResult:
        clock[0] = 106.0
        return _TinyResult(value="late")

    monkeypatch.setattr(module, "_validate_json_object", validate)
    llm = MagicMock()
    llm.invoke.return_value = _response('{"value":"ok"}')
    with pytest.raises(TimeoutError):
        _generate(llm, deadline=105.0)
