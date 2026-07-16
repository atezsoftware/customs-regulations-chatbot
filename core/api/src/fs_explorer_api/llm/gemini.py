"""Gemini implementation of the provider-agnostic `LLMClient` interface."""

import asyncio
import inspect
import os
import time
from typing import AsyncIterator

from google.genai import Client as GenAIClient
from google.genai import errors as genai_errors
from google.genai.types import Content, Part
from fs_explorer_shared.google_genai import build_genai_client

from .base import ChatTurn, LLMUsage, SchemaT, ThinkingLevel

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Every `GeminiLLMClient` instance shares this one semaphore (module-level,
# not per-instance) so the cap holds process-wide regardless of how many
# clients get constructed — the singleton chat agent's client and a
# freshly-built one per amendments-analysis request both draw from the same
# pool. Bounds how many Gemini calls this process has in flight at once,
# which is what actually avoids 429 RESOURCE_EXHAUSTED under concurrent
# chatbot traffic (a request that can't get a slot simply waits its turn
# instead of firing and getting rate-limited).
_MAX_CONCURRENT_GEMINI_CALLS = int(os.getenv("FS_EXPLORER_LLM_MAX_CONCURRENCY", "8"))
_llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GEMINI_CALLS)

# Per-call ceiling, applied to every individual Gemini call (each tool-
# planning step, each context-summary compaction, and — via `stream_text`'s
# per-chunk wait below — the final-answer stream too). Without this, a
# single stalled call had no bound at all: it could hang indefinitely,
# holding a `_llm_semaphore` slot forever and eventually starving the whole
# pool, and on the WebSocket side it surfaced only as the connection going
# quiet until something upstream (backend, proxy) gave up and closed it —
# "Core stream closed before completion" with nothing to explain why. A
# bound here means a stuck call now fails fast with a clear timeout instead.
_LLM_CALL_TIMEOUT_SECONDS = float(
    os.getenv("FS_EXPLORER_LLM_CALL_TIMEOUT_SECONDS", "90")
)

# Transient-failure retry: `429 RESOURCE_EXHAUSTED` (rate limit) and
# `503 UNAVAILABLE` (transient overload) are worth a short wait-and-retry
# instead of failing the whole run outright — a chatbot request that's
# already spent several tool-call steps shouldn't die because one call
# got rate-limited for a moment. A call that timed out (see
# `_LLM_CALL_TIMEOUT_SECONDS`) is retried too, on the same assumption that
# it was a transient stall rather than a truly stuck request.
_LLM_RETRY_ATTEMPTS = int(os.getenv("FS_EXPLORER_LLM_RETRY_ATTEMPTS", "3"))
_LLM_RETRY_BACKOFF_SECONDS = float(
    os.getenv("FS_EXPLORER_LLM_RETRY_BACKOFF_SECONDS", "2")
)
_RETRYABLE_STATUS_CODES = {429, 503}


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
        attempt = 0
        while True:
            try:
                async with _llm_semaphore:
                    response = await asyncio.wait_for(
                        self.raw_client.aio.models.generate_content(
                            model=self.model,
                            contents=_to_contents(history),  # ty: ignore[invalid-argument-type]
                            config={
                                "system_instruction": system_prompt,
                                "response_mime_type": "application/json",
                                "response_schema": schema,
                                **self._generation_config(thinking_level),
                            },
                        ),
                        timeout=_LLM_CALL_TIMEOUT_SECONDS,
                    )
                break
            except Exception as exc:
                attempt += 1
                if attempt > _LLM_RETRY_ATTEMPTS or not _is_retryable(exc):
                    raise
                await asyncio.sleep(_LLM_RETRY_BACKOFF_SECONDS)
        duration_ms = (time.monotonic() - started_at) * 1000

        usage = LLMUsage(duration_ms=duration_ms)
        if response.usage_metadata:
            usage = LLMUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                thinking_tokens=getattr(
                    response.usage_metadata, "thoughts_token_count", None
                )
                or 0,
                duration_ms=duration_ms,
            )

        if response.text is None:
            raise ValueError("Gemini returned no text for a structured request.")

        return schema.model_validate_json(response.text), usage

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
                    # Bounded per-chunk, not per-stream: a healthy stream can
                    # legitimately run longer than one timeout window as long
                    # as chunks keep arriving. What this catches is the
                    # stream going silent for `_LLM_CALL_TIMEOUT_SECONDS` —
                    # previously unbounded, so a stalled final-answer stream
                    # could hang the whole request indefinitely.
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.__anext__(), timeout=_LLM_CALL_TIMEOUT_SECONDS
                            )
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
                    or attempt > _LLM_RETRY_ATTEMPTS
                    or not _is_retryable(exc)
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
