"""OpenRouter Chat Completions implementation of the LLM client protocol."""

import asyncio
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator

import httpx

from .base import ChatTurn, LLMUsage, SchemaT, ThinkingLevel

DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
_RETRYABLE_STATUS_CODES = {408, 429, 502, 503}


class OpenRouterError(RuntimeError):
    """Sanitized OpenRouter failure safe for persistence and user handling."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def usage_from_openrouter(raw: dict[str, Any], *, duration_ms: float = 0) -> LLMUsage:
    """Normalize provider usage while keeping reasoning out of visible output."""
    details = raw.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    completion_tokens = _integer(raw.get("completion_tokens"))
    thinking_tokens = min(completion_tokens, _integer(details.get("reasoning_tokens")))
    prompt_details = raw.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    billed_cost = _decimal(raw.get("cost"))
    return LLMUsage(
        input_tokens=_integer(raw.get("prompt_tokens")),
        output_tokens=max(completion_tokens - thinking_tokens, 0),
        thinking_tokens=thinking_tokens,
        cached_input_tokens=_integer(prompt_details.get("cached_tokens")),
        cache_write_tokens=_integer(prompt_details.get("cache_write_tokens")),
        duration_ms=duration_ms,
        generation_id=str(raw["id"]) if raw.get("id") else None,
        billed_cost_usd=billed_cost,
        upstream_cost_usd=_decimal(
            raw.get("cost_details", {}).get("upstream_inference_cost")
            if isinstance(raw.get("cost_details"), dict)
            else None
        ),
        cost_source="provider" if billed_cost is not None else None,
    )


class OpenRouterLLMClient:
    """Adapter for compatible OpenRouter models used by the research agent."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is not configured.")
        self.model = model or DEFAULT_OPENROUTER_MODEL
        self.temperature = temperature
        self._client = client or httpx.AsyncClient(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=float(os.getenv("OPENROUTER_REQUEST_TIMEOUT_SECONDS", "90")),
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Customs Regulations Chatbot"),
            },
        )
        self._last_stream_usage: LLMUsage | None = None
        self._max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))

    @staticmethod
    def _messages(history: list[ChatTurn], system_prompt: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": "assistant" if turn.role == "model" else "user", "content": turn.text}
            for turn in history
        )
        return messages

    def _payload(
        self, history: list[ChatTurn], system_prompt: str, thinking_level: ThinkingLevel | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": self._messages(history, system_prompt)}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if thinking_level and thinking_level != "minimal":
            payload["reasoning"] = {"effort": thinking_level}
        return payload

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response
                if attempt == self._max_retries:
                    raise OpenRouterError("OpenRouter request could not be completed.", status_code=response.status_code)
                delay = float(response.headers.get("Retry-After", "1"))
            except httpx.TimeoutException:
                if attempt == self._max_retries:
                    raise OpenRouterError("OpenRouter request timed out.") from None
                delay = 1
            except httpx.HTTPStatusError as exc:
                raise OpenRouterError("OpenRouter rejected the request.", status_code=exc.response.status_code) from None
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def generate_structured(
        self, history: list[ChatTurn], system_prompt: str, schema: type[SchemaT], *, thinking_level: ThinkingLevel | None = None
    ) -> tuple[SchemaT, LLMUsage]:
        started_at = time.monotonic()
        payload = self._payload(history, system_prompt, thinking_level)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": schema.model_json_schema()},
        }
        response = await self._post(payload)
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OpenRouterError("OpenRouter returned no structured content.")
        usage = usage_from_openrouter(body.get("usage", {}), duration_ms=(time.monotonic() - started_at) * 1000)
        usage.generation_id = str(body.get("id")) if body.get("id") else usage.generation_id
        return schema.model_validate_json(content), usage

    async def stream_text(
        self, history: list[ChatTurn], system_prompt: str, *, thinking_level: ThinkingLevel | None = None
    ) -> AsyncIterator[str]:
        self._last_stream_usage = None
        started_at = time.monotonic()
        payload = self._payload(history, system_prompt, thinking_level)
        payload["stream"] = True
        yielded = False
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    raise OpenRouterError("OpenRouter stream could not be started.", status_code=response.status_code)
                async for line in response.aiter_lines():
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        raise OpenRouterError("OpenRouter stream ended with an error.")
                    if not isinstance(event, dict):
                        continue
                    if isinstance(event.get("usage"), dict):
                        self._last_stream_usage = usage_from_openrouter(
                            event["usage"], duration_ms=(time.monotonic() - started_at) * 1000
                        )
                    if event.get("id") and self._last_stream_usage:
                        self._last_stream_usage.generation_id = str(event["id"])
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        yielded = True
                        yield content
        except httpx.TimeoutException:
            raise OpenRouterError("OpenRouter stream timed out.") from None

    def last_stream_usage(self) -> LLMUsage | None:
        return self._last_stream_usage
