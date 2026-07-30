"""OpenRouter Chat Completions implementation of the LLM client protocol."""

import asyncio
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator

import httpx
from pydantic import ValidationError

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
        usage: LLMUsage | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.provider_code = provider_code
        self.usage = usage
        self.attempts = attempts


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


def _optional_retry_limit(name: str) -> int | None:
    """Read an opt-in retry ceiling; unset means keep recovering."""

    raw = os.getenv(name)
    if raw is None or raw.strip().casefold() in {
        "",
        "none",
        "unlimited",
        "infinite",
    }:
        return None
    return max(int(raw), 0)


def _optional_output_token_limit() -> int | None:
    """Read an opt-in provider token ceiling; unset/zero means no app cap."""

    raw = os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS")
    if raw is None or raw.strip().casefold() in {
        "",
        "0",
        "none",
        "unlimited",
        "infinite",
    }:
        return None
    return max(int(raw), 1)


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


def _merge_openrouter_usage(total: LLMUsage, current: LLMUsage) -> LLMUsage:
    """Account for every paid response consumed by structured recovery."""

    def add_decimal(left: Decimal | None, right: Decimal | None) -> Decimal | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right

    return LLMUsage(
        input_tokens=total.input_tokens + current.input_tokens,
        output_tokens=total.output_tokens + current.output_tokens,
        thinking_tokens=total.thinking_tokens + current.thinking_tokens,
        cached_input_tokens=total.cached_input_tokens + current.cached_input_tokens,
        cache_write_tokens=(total.cache_write_tokens + current.cache_write_tokens),
        duration_ms=total.duration_ms + current.duration_ms,
        generation_id=current.generation_id or total.generation_id,
        billed_cost_usd=add_decimal(
            total.billed_cost_usd,
            current.billed_cost_usd,
        ),
        upstream_cost_usd=add_decimal(
            total.upstream_cost_usd,
            current.upstream_cost_usd,
        ),
        cost_source=(
            "provider"
            if "provider" in {total.cost_source, current.cost_source}
            else current.cost_source or total.cost_source
        ),
    )


def _structured_validation_summary(error: ValidationError) -> str:
    """Return one safe validation hint without echoing the model output."""

    errors = error.errors(include_input=False, include_url=False)
    if not errors:
        return "schema validation failed"
    first = errors[0]
    location = ".".join(str(item) for item in first.get("loc", ())) or "$"
    error_type = _bounded_message(first.get("type")) or "validation_error"
    message = _bounded_message(first.get("msg")) or "schema validation failed"
    return f"{error_type} at {location}: {message}"[:400]


def _validation_looks_truncated(error: ValidationError) -> bool:
    """Recognize incomplete JSON even when a provider omits its finish reason."""

    for item in error.errors(include_input=False, include_url=False):
        if item.get("type") != "json_invalid":
            continue
        message = _bounded_message(item.get("msg")).casefold()
        if any(
            marker in message
            for marker in (
                "eof",
                "end of input",
                "unexpected end",
                "unterminated",
            )
        ):
            return True
    return False


def _normalized_finish_reasons(choice: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for key in ("finish_reason", "native_finish_reason"):
        value = choice.get(key)
        if isinstance(value, str) and value.strip():
            reasons.add(value.strip().casefold().replace("-", "_").replace(" ", "_"))
    return reasons


def _is_token_boundary(reasons: set[str]) -> bool:
    return any(
        reason
        in {
            "length",
            "max_completion_tokens",
            "max_output_tokens",
            "max_tokens",
            "model_length",
            "token_limit",
        }
        or "max_token" in reason
        for reason in reasons
    )


def _is_safety_stop(reasons: set[str]) -> bool:
    return any(
        reason
        in {
            "blocked",
            "content_filter",
            "prohibited_content",
            "recitation",
            "safety",
        }
        or "safety" in reason
        or "content_filter" in reason
        for reason in reasons
    )


def _choice_error_summary(choice: dict[str, Any]) -> tuple[str, bool]:
    """Return a bounded choice-level error and whether it is a policy stop."""

    error = choice.get("error")
    if not isinstance(error, dict):
        return "", False
    message = _bounded_message(error.get("message"))
    code = _bounded_message(error.get("code"))
    error_type = _bounded_message(error.get("type"))
    summary = message or code or error_type or "provider choice failed"
    normalized = " ".join((message, code, error_type)).casefold()
    policy_stop = any(
        marker in normalized
        for marker in ("content_filter", "content policy", "refusal", "safety")
    )
    return summary[:400], policy_stop


def _structured_message_content(message: dict[str, Any]) -> str | None:
    """Normalize structured content shapes used by OpenRouter providers."""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).casefold()
            text = part.get("text")
            if part_type in {"text", "output_text"} and isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "".join(text_parts)

    parsed = message.get("parsed")
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        tool_call = tool_calls[0]
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(arguments, str):
            return arguments
        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return None


def _with_output_token_budget(
    payload: dict[str, Any],
    output_tokens: int,
) -> dict[str, Any]:
    adjusted = dict(payload)
    if "max_completion_tokens" in adjusted:
        adjusted["max_completion_tokens"] = output_tokens
    elif "max_tokens" in adjusted:
        adjusted["max_tokens"] = output_tokens
    else:
        adjusted["max_tokens"] = output_tokens
    return adjusted


def _structured_retry_payload(
    payload: dict[str, Any],
    *,
    schema_name: str,
    failure: str,
) -> dict[str, Any]:
    """Repeat the original request without feeding back a huge broken output."""

    retry = dict(payload)
    messages = payload.get("messages")
    messages = list(messages) if isinstance(messages, list) else []
    retry["messages"] = [
        *messages,
        {
            "role": "user",
            "content": (
                f"The previous {schema_name} structured response was incomplete "
                f"or schema-invalid ({failure}). Generate it again from the "
                "original request. Return exactly one COMPLETE, concise JSON "
                "object satisfying the supplied response schema. Preserve every "
                "distinct requirement, but use compact wording. Do not repeat "
                "items, add commentary, or pad with whitespace. Close every "
                "array, object, and string."
            ),
        },
    ]
    return retry


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
        self._max_retries = _optional_retry_limit("OPENROUTER_MAX_RETRIES")
        self._structured_retries = _optional_retry_limit(
            "OPENROUTER_STRUCTURED_RETRIES"
        )
        self._max_output_tokens = _optional_output_token_limit()

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
            # No application output-token ceiling is sent by default. The
            # selected provider/model may use its full native capacity.
        }
        if self._max_output_tokens is not None:
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
        attempt = 0
        while True:
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    if response.is_error:
                        raise _rejection_error(response, model=self.model)
                    return response
                routing_error = _rejection_error(response, model=self.model)
                if _is_parameter_compatibility_error(routing_error):
                    raise routing_error
                if self._max_retries is not None and attempt >= self._max_retries:
                    raise OpenRouterError(
                        "OpenRouter request could not be completed.",
                        status_code=response.status_code,
                    )
                delay = float(response.headers.get("Retry-After", "1"))
            except httpx.TimeoutException:
                if self._max_retries is not None and attempt >= self._max_retries:
                    raise OpenRouterError("OpenRouter request timed out.") from None
                delay = 1
            attempt += 1
            await asyncio.sleep(delay)

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
        base_payload = self._payload(history, system_prompt, thinking_level)
        base_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": _provider_compatible_json_schema(schema.model_json_schema()),
            },
        }
        base_payload["provider"] = {"require_parameters": True}
        payload = base_payload
        total_usage = LLMUsage()
        last_failure = "provider returned no structured response"
        last_was_truncated = False
        output_token_budget = self._max_output_tokens
        attempt = 0

        while True:
            attempt += 1
            response = await self._post(payload)
            try:
                body = response.json()
            except ValueError:
                body = {}
                last_failure = "invalid JSON response envelope"

            raw_usage = body.get("usage", {}) if isinstance(body, dict) else {}
            current_usage = usage_from_openrouter(
                raw_usage if isinstance(raw_usage, dict) else {}
            )
            if isinstance(body, dict) and body.get("id"):
                current_usage.generation_id = str(body["id"])
            total_usage = _merge_openrouter_usage(total_usage, current_usage)

            choices = body.get("choices") if isinstance(body, dict) else None
            top_level_error = (
                _nested_provider_message(body)
                if isinstance(body, dict) and body.get("error")
                else ""
            )
            choice = (
                choices[0]
                if isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                else {}
            )
            message = choice.get("message", {})
            message = message if isinstance(message, dict) else {}
            reasons = _normalized_finish_reasons(choice)
            last_was_truncated = _is_token_boundary(reasons)
            refusal = message.get("refusal")
            if refusal:
                total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                raise OpenRouterError(
                    f"OpenRouter refused the {schema.__name__} structured request.",
                    usage=total_usage,
                    attempts=attempt,
                )
            if _is_safety_stop(reasons):
                total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                raise OpenRouterError(
                    f"OpenRouter blocked the {schema.__name__} structured response.",
                    usage=total_usage,
                    attempts=attempt,
                )

            choice_error, policy_stop = _choice_error_summary(choice)
            top_level_policy_stop = any(
                marker in top_level_error.casefold()
                for marker in (
                    "content_filter",
                    "content policy",
                    "refusal",
                    "safety",
                )
            )
            if policy_stop or top_level_policy_stop:
                total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                raise OpenRouterError(
                    f"OpenRouter blocked the {schema.__name__} structured response.",
                    usage=total_usage,
                    attempts=attempt,
                )
            if top_level_error:
                last_was_truncated = False
                last_failure = f"provider response failed: {top_level_error}"
            elif choice_error or "error" in reasons:
                last_was_truncated = False
                last_failure = (
                    f"provider choice failed: {choice_error}"
                    if choice_error
                    else "provider choice failed"
                )
            elif last_was_truncated:
                # A syntactically valid prefix can still satisfy a permissive
                # schema. A provider token-boundary signal therefore wins over
                # parsing so a silently incomplete plan is never accepted.
                last_failure = "output ended at the model token boundary"
            else:
                content = _structured_message_content(message)
                if isinstance(content, str) and content.strip():
                    try:
                        result = schema.model_validate_json(content)
                    except ValidationError as exc:
                        last_was_truncated = _validation_looks_truncated(exc)
                        last_failure = (
                            "output ended before the JSON object was complete"
                            if last_was_truncated
                            else _structured_validation_summary(exc)
                        )
                    else:
                        total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                        return result, total_usage
                else:
                    last_failure = "provider returned no structured content"

            if (
                self._structured_retries is not None
                and attempt > self._structured_retries
            ):
                total_usage.duration_ms = (time.monotonic() - started_at) * 1000
                truncation_hint = (
                    " The provider repeatedly stopped at its output-token boundary."
                    if last_was_truncated
                    else ""
                )
                raise OpenRouterError(
                    (
                        f"OpenRouter could not produce a complete valid "
                        f"{schema.__name__} response after {attempt} attempts: "
                        f"{last_failure}.{truncation_hint}"
                    ),
                    usage=total_usage,
                    attempts=attempt,
                )

            if last_was_truncated and output_token_budget is not None:
                output_token_budget *= 2
            retry_base = (
                _with_output_token_budget(base_payload, output_token_budget)
                if output_token_budget is not None
                else base_payload
            )
            payload = _structured_retry_payload(
                retry_base,
                schema_name=schema.__name__,
                failure=last_failure,
            )

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
