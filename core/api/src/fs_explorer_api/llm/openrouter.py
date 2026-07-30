"""OpenRouter Chat Completions implementation of the LLM client protocol."""

import asyncio
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator

import httpx

from .base import ChatTurn, LLMUsage, SchemaT, ThinkingLevel

DEFAULT_OPENROUTER_MODEL = "google/gemini-3.6-flash"
_RETRYABLE_STATUS_CODES = {408, 429, 502, 503}
_MAX_COMPLETION_TOKEN_MODEL_PREFIXES = (
    "openai/gpt-5",
    "openai/o1",
    "openai/o3",
    "openai/o4",
)
_REF_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}


class OpenRouterError(RuntimeError):
    """Sanitized OpenRouter failure safe for persistence and user handling."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.provider_code = provider_code


def _bounded_message(value: object) -> str:
    return " ".join(value.split())[:300] if isinstance(value, str) else ""


def _nested_provider_message(value: object) -> str:
    """Extract only a provider error message, never an arbitrary raw body."""
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            message = _bounded_message(error.get("message"))
            if message:
                return message
        message = _bounded_message(value.get("message"))
        if message:
            return message
    if not isinstance(value, str):
        return ""
    try:
        decoded = json.loads(value)
    except ValueError:
        return ""
    return _nested_provider_message(decoded)


def _error_fields(
    response: httpx.Response,
) -> tuple[str, str | None, str | None]:
    """Return safe message/type/code fields without retaining request data."""
    try:
        body = response.json()
    except ValueError:
        return "", None, None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = _bounded_message(error.get("message"))
            metadata = error.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            error_type = _bounded_message(metadata.get("error_type")) or None
            provider_code = _bounded_message(metadata.get("provider_code")) or None
            provider_detail = _nested_provider_message(metadata.get("raw"))
            if provider_detail and provider_detail.casefold() != detail.casefold():
                detail = f"{detail} — {provider_detail}" if detail else provider_detail
            return detail[:600], error_type, provider_code
        if isinstance(error, str):
            return _bounded_message(error), None, None
        elif isinstance(body.get("message"), str):
            return _bounded_message(body["message"]), None, None
    return "", None, None


def _rejection_error(
    response: httpx.Response,
    *,
    model: str,
    stream: bool = False,
) -> OpenRouterError:
    """Turn a rejected response into an actionable, bounded safe error."""
    message = (
        "OpenRouter stream could not be started"
        if stream
        else "OpenRouter rejected the request"
    )
    detail, error_type, provider_code = _error_fields(response)
    if detail:
        message = f"{message} (HTTP {response.status_code}): {detail}"
    else:
        message = f"{message} (HTTP {response.status_code})."
    normalized_detail = detail.casefold()
    if response.status_code in {404, 503} and (
        "no endpoints available" in normalized_detail
        or "guardrail restrictions" in normalized_detail
        or "data policy" in normalized_detail
    ):
        message += (
            f" Configured model: {model}. No endpoint for this model satisfies "
            "the API key/workspace Privacy and Guardrail policy; allow this "
            "model/provider in OpenRouter or configure an eligible role model."
        )
    return OpenRouterError(
        message,
        status_code=response.status_code,
        error_type=error_type,
        provider_code=provider_code,
    )


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _is_endpoint_eligibility_error(error: OpenRouterError) -> bool:
    """Whether another parameter spelling may expose an eligible endpoint."""
    if error.status_code not in {404, 503}:
        return False
    message = str(error).casefold()
    return (
        "no endpoints available" in message
        or "guardrail restrictions" in message
        or "data policy" in message
        or "routing requirements" in message
    )


def _is_parameter_compatibility_error(error: OpenRouterError) -> bool:
    """Whether the same model should be retried with a leaner parameter set."""
    if _is_endpoint_eligibility_error(error):
        return True
    if error.status_code != 400:
        return False
    if error.error_type in {
        "content_policy_violation",
        "refusal",
        "context_length_exceeded",
        "max_tokens_exceeded",
        "token_limit_exceeded",
        "string_too_long",
        "invalid_prompt",
        "invalid_image",
        "image_too_large",
        "image_too_small",
        "unsupported_image_format",
        "image_not_found",
        "image_download_failed",
    }:
        return False
    if error.error_type == "invalid_request":
        return True
    detail = str(error).casefold()
    provider_code = (error.provider_code or "").casefold()
    return (
        "provider returned error" in detail
        or "unsupported parameter" in detail
        or "unknown parameter" in detail
        or "invalid parameter" in detail
        or "not supported" in detail
        or provider_code
        in {
            "bad_request",
            "invalid_argument",
            "invalid_request",
            "unsupported_parameter",
        }
    )


def _request_payload_variants(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build safe same-model variants for heterogeneous endpoint parameters.

    OpenRouter endpoints do not share one output-token parameter: most expose
    ``max_tokens``, while modern OpenAI/Azure endpoints expose
    ``max_completion_tokens``. ``provider.require_parameters`` also makes
    optional reasoning/temperature fields part of endpoint eligibility. Try
    only parameter-compatible variants; never change the requested model,
    provider policy, messages, schema, or privacy rules.
    """

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        signature = json.dumps(candidate, sort_keys=True, default=str)
        if signature not in seen:
            seen.add(signature)
            variants.append(candidate)

    def swap_token_parameter(candidate: dict[str, Any]) -> dict[str, Any] | None:
        if "max_tokens" in candidate:
            swapped = dict(candidate)
            swapped["max_completion_tokens"] = swapped.pop("max_tokens")
            return swapped
        if "max_completion_tokens" in candidate:
            swapped = dict(candidate)
            swapped["max_tokens"] = swapped.pop("max_completion_tokens")
            return swapped
        return None

    base = dict(payload)
    add(base)
    swapped = swap_token_parameter(base)
    if swapped is not None:
        add(swapped)

    relaxed = dict(base)
    relaxed.pop("reasoning", None)
    relaxed.pop("temperature", None)
    add(relaxed)
    relaxed_swapped = swap_token_parameter(relaxed)
    if relaxed_swapped is not None:
        add(relaxed_swapped)

    without_token_limit = dict(relaxed)
    without_token_limit.pop("max_tokens", None)
    without_token_limit.pop("max_completion_tokens", None)
    add(without_token_limit)
    return variants


def _provider_compatible_json_schema(value: Any, *, path: str = "$") -> Any:
    """Normalize Pydantic schemas for strict OpenAI-compatible endpoints.

    JSON Schema permits annotation keywords beside ``$ref``. Some OpenRouter
    providers implement the narrower OpenAI structured-output dialect and
    reject those valid schemas. Annotations do not affect validation, so they
    can be removed without changing the Pydantic contract. Validation keywords
    are never discarded silently.
    """
    if isinstance(value, list):
        return [
            _provider_compatible_json_schema(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _provider_compatible_json_schema(item, path=f"{path}.{key}")
        for key, item in value.items()
    }
    if "$ref" not in normalized:
        return normalized

    siblings = set(normalized) - {"$ref"}
    unsupported = siblings - _REF_ANNOTATION_KEYS
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Cannot normalize validation keywords beside $ref at {path}: {joined}"
        )
    return {"$ref": normalized["$ref"]}


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
        *,
        enable_reasoning_effort: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is not configured.")
        self.model = model or DEFAULT_OPENROUTER_MODEL
        self.temperature = temperature
        self._enable_reasoning_effort = enable_reasoning_effort
        self._client = client or httpx.AsyncClient(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            # Planning and research stages may legitimately take longer than
            # a fixed request deadline. Connection-level heartbeats keep the
            # user-facing stream alive while this call remains pending.
            timeout=None,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                "X-Title": os.getenv(
                    "OPENROUTER_APP_TITLE", "Customs Regulations Chatbot"
                ),
                # Adds typed provider/guardrail routing diagnostics to errors;
                # it does not opt into prompt or completion logging.
                "X-OpenRouter-Metadata": "enabled",
            },
        )
        self._last_stream_usage: LLMUsage | None = None
        self._max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
        self._max_output_tokens = int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "8000"))

    @staticmethod
    def _messages(history: list[ChatTurn], system_prompt: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {
                "role": "assistant" if turn.role == "model" else "user",
                "content": turn.text,
            }
            for turn in history
        )
        return messages

    def _payload(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        thinking_level: ThinkingLevel | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(history, system_prompt),
            # Reasoning models (e.g. z-ai/glm-5.2) spend completion tokens on
            # hidden chain-of-thought before ever emitting visible content.
            # Without an explicit ceiling, requests fall back to whatever
            # small default the upstream provider applies, which a verbose
            # reasoner can exhaust entirely on thinking — OpenRouter then
            # returns `finish_reason: "length"` with an empty `content`,
            # surfacing here as "no structured content" even though the
            # request itself succeeded. `max_tokens` is a baseline
            # OpenAI-compatible field every provider honors, unlike the
            # provider-specific `reasoning` controls below.
        }
        token_parameter = (
            "max_completion_tokens"
            if self.model.startswith(_MAX_COMPLETION_TOKEN_MODEL_PREFIXES)
            else "max_tokens"
        )
        payload[token_parameter] = self._max_output_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self._enable_reasoning_effort and thinking_level is not None:
            # OpenRouter normalizes this shape across providers, including
            # OpenAI reasoning effort and Gemini thinking levels. This is
            # enabled only for curated role profiles; arbitrary legacy model
            # selections omit the model-specific parameter for compatibility.
            payload["reasoning"] = {"effort": thinking_level}
        return payload

    async def _post_variant(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    if response.is_error:
                        raise _rejection_error(response, model=self.model)
                    return response
                routing_error = _rejection_error(response, model=self.model)
                if _is_parameter_compatibility_error(routing_error):
                    raise routing_error
                if attempt == self._max_retries:
                    raise OpenRouterError(
                        "OpenRouter request could not be completed.",
                        status_code=response.status_code,
                    )
                delay = float(response.headers.get("Retry-After", "1"))
            except httpx.TimeoutException:
                if attempt == self._max_retries:
                    raise OpenRouterError("OpenRouter request timed out.") from None
                delay = 1
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: OpenRouterError | None = None
        for candidate in _request_payload_variants(payload):
            try:
                return await self._post_variant(candidate)
            except OpenRouterError as exc:
                if not _is_parameter_compatibility_error(exc):
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema: type[SchemaT],
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> tuple[SchemaT, LLMUsage]:
        started_at = time.monotonic()
        payload = self._payload(history, system_prompt, thinking_level)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": _provider_compatible_json_schema(schema.model_json_schema()),
            },
        }
        payload["provider"] = {"require_parameters": True}
        response = await self._post(payload)
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        message = (
            choices[0].get("message", {})
            if isinstance(choices, list) and choices
            else {}
        )
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            finish_reason = (
                choices[0].get("finish_reason")
                if isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                else None
            )
            hint = (
                " (truncated before emitting content — the model likely ran out of "
                "output tokens; see OPENROUTER_MAX_OUTPUT_TOKENS)"
                if finish_reason == "length"
                else ""
            )
            raise OpenRouterError(f"OpenRouter returned no structured content.{hint}")
        usage = usage_from_openrouter(
            body.get("usage", {}), duration_ms=(time.monotonic() - started_at) * 1000
        )
        usage.generation_id = (
            str(body.get("id")) if body.get("id") else usage.generation_id
        )
        return schema.model_validate_json(content), usage

    async def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        self._last_stream_usage = None
        started_at = time.monotonic()
        payload = self._payload(history, system_prompt, thinking_level)
        payload["stream"] = True
        last_error: OpenRouterError | None = None
        for candidate in _request_payload_variants(payload):
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=candidate
                ) as response:
                    if response.status_code >= 400:
                        error = _rejection_error(
                            response,
                            model=self.model,
                            stream=True,
                        )
                        if _is_parameter_compatibility_error(error):
                            last_error = error
                            continue
                        raise error
                    async for line in response.aiter_lines():
                        if (
                            not line
                            or line.startswith(":")
                            or not line.startswith("data:")
                        ):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            raise OpenRouterError(
                                "OpenRouter stream ended with an error."
                            )
                        if not isinstance(event, dict):
                            continue
                        if isinstance(event.get("usage"), dict):
                            self._last_stream_usage = usage_from_openrouter(
                                event["usage"],
                                duration_ms=(time.monotonic() - started_at) * 1000,
                            )
                        if event.get("id") and self._last_stream_usage:
                            self._last_stream_usage.generation_id = str(event["id"])
                        choices = event.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        delta = (
                            choices[0].get("delta")
                            if isinstance(choices[0], dict)
                            else None
                        )
                        content = (
                            delta.get("content") if isinstance(delta, dict) else None
                        )
                        if isinstance(content, str) and content:
                            yield content
                    return
            except httpx.TimeoutException:
                raise OpenRouterError("OpenRouter stream timed out.") from None
        assert last_error is not None
        raise last_error

    def last_stream_usage(self) -> LLMUsage | None:
        return self._last_stream_usage
