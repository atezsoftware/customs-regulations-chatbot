import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from fs_explorer_api.llm import gemini as gemini_module
from fs_explorer_api.llm.base import ChatTurn
from fs_explorer_api.llm.gemini import (
    GeminiLLMClient,
    GeminiStructuredOutputError,
)


class Reply(BaseModel):
    answer: str


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _AsyncStream:
    def __init__(self) -> None:
        self._chunks = iter([_Chunk("hello"), _Chunk(" world")])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _AsyncModels:
    async def generate_content_stream(self, **_kwargs):
        return _AsyncStream()


class _Aio:
    models = _AsyncModels()


class _Client:
    aio = _Aio()


class _StructuredModels:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


class _StructuredClient:
    def __init__(self, responses: list[object]) -> None:
        self.models = _StructuredModels(responses)
        self.aio = SimpleNamespace(models=self.models)


def _usage(
    *,
    prompt: int,
    candidates: int,
    thoughts: int = 0,
    cached: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
    )


def _response(
    text: str | None,
    *,
    finish_reason: str | None = "STOP",
    finish_message: str | None = None,
    block_reason: str | None = None,
    safety_blocked: bool = False,
    usage: object | None = None,
    response_id: str | None = None,
) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=finish_reason,
        finish_message=finish_message,
        safety_ratings=[SimpleNamespace(blocked=True)] if safety_blocked else [],
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        usage_metadata=usage,
        response_id=response_id,
    )


def _content_texts(contents: list[object]) -> list[str]:
    return [
        "".join(
            str(getattr(part, "text", "") or "")
            for part in (getattr(content, "parts", None) or [])
        )
        for content in contents
    ]


def test_retry_limits_default_to_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FS_EXPLORER_LLM_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("GEMINI_STRUCTURED_RETRIES", raising=False)

    assert gemini_module._optional_retry_limit("FS_EXPLORER_LLM_RETRY_ATTEMPTS") is None
    assert GeminiLLMClient(client=_Client())._structured_retries is None


@pytest.mark.asyncio
async def test_stream_text_accepts_awaitable_sdk_stream() -> None:
    client = GeminiLLMClient(client=_Client())

    chunks = [
        chunk async for chunk in client.stream_text([], "system", thinking_level="high")
    ]

    assert chunks == ["hello", " world"]


@pytest.mark.asyncio
async def test_truncated_structured_response_is_regenerated_from_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "1")
    broken_output = '{"answer":"TOP_SECRET_BROKEN_FRAGMENT"}'
    raw_client = _StructuredClient(
        [
            _response(
                broken_output,
                finish_reason="MAX_TOKENS",
                usage=_usage(prompt=10, candidates=6, thoughts=2, cached=1),
                response_id="gemini-truncated",
            ),
            _response(
                '{"answer":"repaired"}',
                usage=_usage(prompt=12, candidates=3, cached=2),
                response_id="gemini-complete",
            ),
        ]
    )
    client = GeminiLLMClient(client=raw_client)
    history = [ChatTurn(role="user", text="original request")]

    result, usage = await client.generate_structured(
        history,
        "system",
        Reply,
        thinking_level="high",
    )

    assert result.answer == "repaired"
    assert len(raw_client.models.calls) == 2
    first, second = raw_client.models.calls
    assert _content_texts(first["contents"]) == ["original request"]
    second_texts = _content_texts(second["contents"])
    assert second_texts[0] == "original request"
    assert "complete, concise JSON" in second_texts[-1]
    assert "TOP_SECRET_BROKEN_FRAGMENT" not in " ".join(second_texts)
    assert second["config"]["system_instruction"] == "system"
    assert second["config"]["response_schema"] is Reply
    assert second["config"]["response_mime_type"] == "application/json"
    assert second["config"]["thinking_config"] == {"thinking_level": "high"}
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.thinking_tokens,
        usage.cached_input_tokens,
    ) == (22, 9, 2, 3)
    assert usage.generation_id == "gemini-complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_text", [None, "not json", "{}"])
async def test_missing_or_invalid_structured_response_gets_one_correction(
    invalid_text: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "1")
    raw_client = _StructuredClient(
        [
            _response(invalid_text),
            _response('{"answer":"valid"}'),
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    result, _usage = await client.generate_structured(
        [ChatTurn(role="user", text="original")],
        "system",
        Reply,
    )

    assert result.answer == "valid"
    assert len(raw_client.models.calls) == 2
    correction = _content_texts(raw_client.models.calls[1]["contents"])[-1]
    assert "Regenerate it from the original request" in correction
    assert "errors.pydantic.dev" not in correction


@pytest.mark.asyncio
async def test_unlimited_structured_corrections_continue_until_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_STRUCTURED_RETRIES", raising=False)
    raw_client = _StructuredClient(
        [
            _response(None),
            _response("not json"),
            _response("{}"),
            _response('{"answer":"eventually valid"}'),
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    result, _usage = await client.generate_structured(
        [ChatTurn(role="user", text="original")],
        "system",
        Reply,
    )

    assert result.answer == "eventually valid"
    assert len(raw_client.models.calls) == 4


@pytest.mark.asyncio
async def test_repeated_invalid_json_raises_sanitized_error_with_failed_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "1")
    broken_output = '{"answer":"TOP_SECRET_BROKEN_FRAGMENT'
    raw_client = _StructuredClient(
        [
            _response(
                broken_output,
                finish_reason="MAX_TOKENS",
                usage=_usage(prompt=7, candidates=4, thoughts=1),
            ),
            _response(
                broken_output,
                finish_reason="MAX_TOKENS",
                usage=_usage(prompt=9, candidates=5, thoughts=2),
            ),
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    with pytest.raises(GeminiStructuredOutputError) as caught:
        await client.generate_structured(
            [ChatTurn(role="user", text="original")],
            "system",
            Reply,
        )

    error = caught.value
    assert error.attempts == 2
    assert error.truncated is True
    assert (
        error.usage.input_tokens,
        error.usage.output_tokens,
        error.usage.thinking_tokens,
    ) == (16, 9, 3)
    assert "output-token boundary" in str(error)
    assert "TOP_SECRET_BROKEN_FRAGMENT" not in str(error)
    assert "input_value" not in str(error)
    assert "errors.pydantic.dev" not in str(error)
    retry_text = " ".join(_content_texts(raw_client.models.calls[1]["contents"]))
    assert "TOP_SECRET_BROKEN_FRAGMENT" not in retry_text


@pytest.mark.asyncio
async def test_schema_mismatch_never_leaks_raw_pydantic_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "0")
    raw_client = _StructuredClient(
        [
            _response(
                "{}",
                usage=_usage(prompt=5, candidates=1),
            )
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    with pytest.raises(GeminiStructuredOutputError) as caught:
        await client.generate_structured(
            [ChatTurn(role="user", text="original")],
            "system",
            Reply,
        )

    message = str(caught.value)
    assert "missing at answer" in message
    assert "input_value" not in message
    assert "errors.pydantic.dev" not in message
    assert caught.value.usage.input_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "finish_reason",
        "finish_message",
        "block_reason",
        "safety_blocked",
    ),
    [
        ("SAFETY", None, None, False),
        (None, None, "SAFETY", False),
        ("OTHER", "The model refused this request under policy.", None, False),
        ("STOP", None, None, True),
    ],
)
async def test_structured_refusal_or_safety_block_is_not_retried(
    finish_reason: str | None,
    finish_message: str | None,
    block_reason: str | None,
    safety_blocked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "2")
    raw_client = _StructuredClient(
        [
            _response(
                None,
                finish_reason=finish_reason,
                finish_message=finish_message,
                block_reason=block_reason,
                safety_blocked=safety_blocked,
                usage=_usage(prompt=5, candidates=1),
            )
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    with pytest.raises(
        GeminiStructuredOutputError, match="blocked or refused"
    ) as caught:
        await client.generate_structured(
            [ChatTurn(role="user", text="original")],
            "system",
            Reply,
        )

    assert len(raw_client.models.calls) == 1
    assert caught.value.attempts == 1
    assert caught.value.usage.input_tokens == 5


@pytest.mark.asyncio
async def test_transient_provider_retry_preserves_original_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_RETRIES", "1")
    monkeypatch.setattr(gemini_module, "_LLM_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(gemini_module, "_LLM_RETRY_BACKOFF_SECONDS", 0)
    raw_client = _StructuredClient(
        [
            asyncio.TimeoutError(),
            _response(
                '{"answer":"after transient retry"}',
                usage=_usage(prompt=4, candidates=2),
            ),
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    result, usage = await client.generate_structured(
        [ChatTurn(role="user", text="original")],
        "system",
        Reply,
    )

    assert result.answer == "after transient retry"
    assert len(raw_client.models.calls) == 2
    assert all(
        _content_texts(call["contents"]) == ["original"]
        for call in raw_client.models.calls
    )
    assert usage.input_tokens == 4


@pytest.mark.asyncio
async def test_unlimited_transient_provider_retries_continue_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini_module, "_LLM_RETRY_ATTEMPTS", None)
    monkeypatch.setattr(gemini_module, "_LLM_RETRY_BACKOFF_SECONDS", 0)
    raw_client = _StructuredClient(
        [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            _response('{"answer":"eventually available"}'),
        ]
    )
    client = GeminiLLMClient(client=raw_client)

    result, _usage = await client.generate_structured(
        [ChatTurn(role="user", text="original")],
        "system",
        Reply,
    )

    assert result.answer == "eventually available"
    assert len(raw_client.models.calls) == 4
