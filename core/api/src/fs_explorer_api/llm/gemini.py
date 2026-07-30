"""Gemini implementation of the provider-agnostic `LLMClient` interface."""

import asyncio
import inspect
import os
import time
from typing import Any, AsyncIterator

from google.genai import Client as GenAIClient
from google.genai import errors as genai_errors
from google.genai.types import Content, Part
from fs_explorer_shared.google_genai import build_genai_client
from pydantic import ValidationError

from .base import ChatTurn, LLMUsage, SchemaT, ThinkingLevel

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


def _optional_retry_limit(name: str) -> int | None:
    """Return an explicit retry override; production defaults to unlimited."""

    raw = os.getenv(name)
    if raw is None or raw.strip().casefold() in {"", "none", "unlimited"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a non-negative integer or unlimited."
        ) from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or unlimited.")
    return value


class _UnlimitedAsyncLimiter:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None


# Production has no application concurrency ceiling. An operator can opt into
# one for a constrained deployment; every client then shares that semaphore.
_MAX_CONCURRENT_GEMINI_CALLS = _optional_retry_limit("FS_EXPLORER_LLM_MAX_CONCURRENCY")
_llm_semaphore = (
    asyncio.Semaphore(max(_MAX_CONCURRENT_GEMINI_CALLS, 1))
    if _MAX_CONCURRENT_GEMINI_CALLS is not None
    else _UnlimitedAsyncLimiter()
)

# Transient-failure retry: `429 RESOURCE_EXHAUSTED` (rate limit) and
# `503 UNAVAILABLE` (transient overload) are worth a short wait-and-retry
# instead of failing the whole run outright — a chatbot request that's
# already spent several tool-call steps shouldn't die because one call
# got rate-limited for a moment. Provider/network timeout exceptions remain
# retryable, but this client does not impose its own wall-clock deadline.
_LLM_RETRY_ATTEMPTS = _optional_retry_limit("FS_EXPLORER_LLM_RETRY_ATTEMPTS")
_LLM_RETRY_BACKOFF_SECONDS = float(
    os.getenv("FS_EXPLORER_LLM_RETRY_BACKOFF_SECONDS", "2")
)
_RETRYABLE_STATUS_CODES = {429, 503}
_NON_RETRYABLE_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}
_REFUSAL_MARKERS = (
    "block",
    "policy",
    "prohibit",
    "recitation",
    "refus",
    "safety",
)


class GeminiStructuredOutputError(RuntimeError):
    """A bounded structured-output failure with any observable token usage."""

    def __init__(
        self,
        message: str,
        *,
        usage: LLMUsage,
        attempts: int,
        truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.attempts = attempts
        self.truncated = truncated


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, genai_errors.APIError):
        return exc.code in _RETRYABLE_STATUS_CODES
    return False


def _to_contents(history: list[ChatTurn]) -> list[Content]:
    return [
        Content(role=turn.role, parts=[Part.from_text(text=turn.text)])
        for turn in history
    ]


def _member(value: object, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _enum_name(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw).rsplit(".", 1)[-1].strip().upper()
    return normalized or None


def _first_candidate(response: object) -> object | None:
    candidates = _member(response, "candidates")
    if isinstance(candidates, (list, tuple)) and candidates:
        return candidates[0]
    return None


def _finish_reason(response: object) -> str | None:
    candidate = _first_candidate(response)
    return _enum_name(_member(candidate, "finish_reason"))


def _blocked_reason(response: object) -> str | None:
    """Classify refusals and safety blocks before attempting correction."""

    prompt_feedback = _member(response, "prompt_feedback")
    prompt_block = _enum_name(_member(prompt_feedback, "block_reason"))
    if prompt_block not in {None, "BLOCKED_REASON_UNSPECIFIED"}:
        return prompt_block

    candidate = _first_candidate(response)
    finish_reason = _enum_name(_member(candidate, "finish_reason"))
    if finish_reason in _NON_RETRYABLE_FINISH_REASONS:
        return finish_reason

    safety_ratings = _member(candidate, "safety_ratings")
    if isinstance(safety_ratings, (list, tuple)) and any(
        _member(rating, "blocked") is True for rating in safety_ratings
    ):
        return "SAFETY"

    finish_message = _member(candidate, "finish_message")
    if isinstance(finish_message, str):
        normalized = finish_message.casefold()
        if any(marker in normalized for marker in _REFUSAL_MARKERS):
            return "REFUSAL"
    return None


def _safe_response_text(response: object) -> str | None:
    """Read SDK convenience text without leaking its raw-response exception."""

    try:
        text = _member(response, "text")
    except (AttributeError, TypeError, ValueError):
        return None
    return text if isinstance(text, str) and text.strip() else None


def _usage_from_response(response: object) -> LLMUsage:
    metadata = _member(response, "usage_metadata")
    if metadata is None:
        return LLMUsage()
    response_id = _member(response, "response_id")
    return LLMUsage(
        input_tokens=int(_member(metadata, "prompt_token_count") or 0),
        output_tokens=int(_member(metadata, "candidates_token_count") or 0),
        thinking_tokens=int(_member(metadata, "thoughts_token_count") or 0),
        cached_input_tokens=int(_member(metadata, "cached_content_token_count") or 0),
        generation_id=str(response_id) if response_id else None,
    )


def _merge_usage(total: LLMUsage, current: LLMUsage) -> LLMUsage:
    """Account for every provider response consumed by structured recovery."""

    return LLMUsage(
        input_tokens=total.input_tokens + current.input_tokens,
        output_tokens=total.output_tokens + current.output_tokens,
        thinking_tokens=total.thinking_tokens + current.thinking_tokens,
        cached_input_tokens=total.cached_input_tokens + current.cached_input_tokens,
        cache_write_tokens=(total.cache_write_tokens + current.cache_write_tokens),
        duration_ms=total.duration_ms + current.duration_ms,
        generation_id=current.generation_id or total.generation_id,
        billed_cost_usd=current.billed_cost_usd or total.billed_cost_usd,
        upstream_cost_usd=current.upstream_cost_usd or total.upstream_cost_usd,
        cost_source=current.cost_source or total.cost_source,
    )


def _bounded_validation_summary(error: ValidationError) -> str:
    """Return one schema hint without echoing the model's invalid output."""

    errors = error.errors(include_input=False, include_url=False)
    if not errors:
        return "schema validation failed"
    first = errors[0]
    location = ".".join(str(item) for item in first.get("loc", ())) or "$"
    error_type = " ".join(str(first.get("type") or "validation_error").split())[:100]
    message = " ".join(str(first.get("msg") or "schema validation failed").split())[
        :200
    ]
    return f"{error_type} at {location}: {message}"[:400]


def _structured_correction(
    *,
    schema_name: str,
    failure: str,
) -> ChatTurn:
    """Ask for compact regeneration from the original request only."""

    return ChatTurn(
        role="user",
        text=(
            f"The previous {schema_name} response was incomplete or schema-invalid "
            f"({failure}). Regenerate it from the original request. Return exactly "
            "one complete, concise JSON object matching the supplied schema. "
            "Preserve distinct requirements; do not add commentary or repeat items."
        ),
    )


class GeminiLLMClient:
    """Wraps `google.genai.Client`. Same calls/config the agent always made."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        client: GenAIClient | None = None,
    ) -> None:
        self.model = model or DEFAULT_GEMINI_MODEL
        self.temperature = temperature
        if client is not None:
            self.raw_client = client
        else:
            self.raw_client = build_genai_client(
                api_key=api_key,
            )
        self._last_stream_usage: LLMUsage | None = None
        self._structured_retries = _optional_retry_limit("GEMINI_STRUCTURED_RETRIES")

    def _generation_config(self, thinking_level: ThinkingLevel | None = None) -> dict:
        config: dict = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if thinking_level is not None:
            config["thinking_config"] = {"thinking_level": thinking_level}
        return config

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema: type[SchemaT],
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> tuple[SchemaT, LLMUsage]:
        started_at = time.monotonic()
        base_history = list(history)
        request_history = base_history
        total_usage = LLMUsage()
        last_failure = "provider returned no structured content"
        saw_truncation = False
        structured_attempt = 0

        while True:
            provider_attempt = 0
            while True:
                try:
                    async with _llm_semaphore:
                        response = await self.raw_client.aio.models.generate_content(
                            model=self.model,
                            contents=_to_contents(request_history),  # ty: ignore[invalid-argument-type]
                            config={
                                "system_instruction": system_prompt,
                                "response_mime_type": "application/json",
                                "response_schema": schema,
                                **self._generation_config(thinking_level),
                            },
                        )
                    break
                except Exception as exc:
                    provider_attempt += 1
                    if not _is_retryable(exc) or (
                        _LLM_RETRY_ATTEMPTS is not None
                        and provider_attempt > _LLM_RETRY_ATTEMPTS
                    ):
                        raise
                    await asyncio.sleep(_LLM_RETRY_BACKOFF_SECONDS)

            total_usage = _merge_usage(total_usage, _usage_from_response(response))
            blocked_reason = _blocked_reason(response)
            if blocked_reason is not None:
                total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                raise GeminiStructuredOutputError(
                    (
                        f"Gemini blocked or refused the {schema.__name__} "
                        f"structured response ({blocked_reason})."
                    ),
                    usage=total_usage,
                    attempts=structured_attempt + 1,
                )

            finish_reason = _finish_reason(response)
            if finish_reason == "MAX_TOKENS":
                saw_truncation = True
                last_failure = "output ended at the model token boundary"
            elif finish_reason not in {None, "FINISH_REASON_UNSPECIFIED", "STOP"}:
                last_failure = f"generation stopped with {finish_reason}"
            else:
                text = _safe_response_text(response)
                if text is None:
                    last_failure = "provider returned no structured content"
                else:
                    try:
                        result = schema.model_validate_json(text)
                    except ValidationError as exc:
                        last_failure = _bounded_validation_summary(exc)
                    else:
                        total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                        return result, total_usage

            if (
                self._structured_retries is not None
                and structured_attempt >= self._structured_retries
            ):
                break
            request_history = [
                *base_history,
                _structured_correction(
                    schema_name=schema.__name__,
                    failure=last_failure,
                ),
            ]
            structured_attempt += 1

        total_usage.duration_ms = (time.monotonic() - started_at) * 1000
        truncation_hint = (
            " At least one response stopped at the provider output-token boundary."
            if saw_truncation
            else ""
        )
        raise GeminiStructuredOutputError(
            (
                f"Gemini could not produce a complete valid {schema.__name__} "
                f"response after {structured_attempt + 1} attempts: "
                f"{last_failure}.{truncation_hint}"
            ),
            usage=total_usage,
            attempts=structured_attempt + 1,
            truncated=saw_truncation,
        )

    async def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        self._last_stream_usage = None
        stream_fn = getattr(self.raw_client.aio.models, "generate_content_stream", None)
        if stream_fn is None:
            return

        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0
        saw_usage = False
        started_at = time.monotonic()
        yielded_any = False
        attempt = 0

        # Retries the whole stream setup — only safe (no duplicated output)
        # as long as nothing has been yielded to the caller yet. A failure
        # after real text already streamed out is *not* retried here; it
        # propagates to the caller (`FsExplorerAgent.stream_final_answer`),
        # which falls back to whatever partial/fallback answer it has
        # rather than risk sending duplicate or out-of-order text.
        while True:
            try:
                async with _llm_semaphore:
                    stream_result = stream_fn(
                        model=self.model,
                        contents=_to_contents(history),
                        config={
                            "system_instruction": system_prompt,
                            **self._generation_config(thinking_level),
                        },
                    )
                    if inspect.isawaitable(stream_result):
                        stream_result = await stream_result
                    stream = stream_result.__aiter__()
                    while True:
                        try:
                            chunk = await stream.__anext__()
                        except StopAsyncIteration:
                            break

                        if getattr(chunk, "usage_metadata", None):
                            saw_usage = True
                            usage = chunk.usage_metadata
                            input_tokens = usage.prompt_token_count or input_tokens
                            output_tokens = (
                                usage.candidates_token_count or output_tokens
                            )
                            thinking_tokens = (
                                getattr(usage, "thoughts_token_count", None)
                                or thinking_tokens
                            )

                        text = getattr(chunk, "text", None)
                        if text:
                            yielded_any = True
                            yield text
                break
            except Exception as exc:
                attempt += 1
                if (
                    yielded_any
                    or not _is_retryable(exc)
                    or (
                        _LLM_RETRY_ATTEMPTS is not None
                        and attempt > _LLM_RETRY_ATTEMPTS
                    )
                ):
                    raise
                await asyncio.sleep(_LLM_RETRY_BACKOFF_SECONDS)

        if saw_usage:
            self._last_stream_usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                duration_ms=(time.monotonic() - started_at) * 1000,
            )

    def last_stream_usage(self) -> LLMUsage | None:
        return self._last_stream_usage
