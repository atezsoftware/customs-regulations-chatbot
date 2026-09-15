"""Schema-constrained JSON generation for regulatory LLM workflows.

The amendment pipeline (segmenter/matcher/drafter) needs typed, validated LLM
output rather than free text. `structured_response_format` on `LLM.invoke`
maps straight to the provider's JSON-schema mode when supported; on the
(common, multi-provider) case where that support is inconsistent, one retry
with the validation error fed back closes the gap without depending on any
single provider's structured-output guarantees.
"""

import json
import math
import random
import time
from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

from onyx.configs.chat_configs import (
    LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S,
    LLM_FIRST_CHUNK_RETRY_JITTER_RATIO,
    LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S,
)
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    ImageContentPart,
    ReasoningEffort,
    SystemMessage,
    TextContentPart,
    UserMessage,
)
from onyx.llm.multi_llm import LLMRateLimitError, LLMTimeoutError
from onyx.llm.utils import llm_response_to_string
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger
from onyx.utils.retry_after import parse_retry_after_seconds

logger = setup_logger()

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredOutputValidationError(ValueError):
    """The provider answered, but exhausted schema-correction attempts."""


class _StructuredInvokeOptions(TypedDict):
    structured_response_format: dict
    timeout_override: NotRequired[int]
    max_tokens: NotRequired[int]
    reasoning_effort: NotRequired[ReasoningEffort]
    use_streaming: NotRequired[bool]


_JSON_ONLY_REMINDER = (
    "\n\nCRITICAL: Respond with ONLY a single valid JSON object matching the "
    "schema below. No prose, no markdown code fences, no explanation outside "
    "the JSON.\n\nSchema:\n{schema}"
)
_MAX_INVALID_RESPONSE_FEEDBACK_CHARS = 48_000
_PORTABLE_SCHEMA_UNSUPPORTED_KEYS = frozenset(
    {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maxLength",
        "maximum",
        "minLength",
        "minimum",
        "multipleOf",
    }
)


def _portable_structured_output_schema(value: Any) -> Any:
    """Remove validation keywords rejected by stricter provider routes.

    Pydantic still validates the original model after generation, so this only
    makes the provider-side grammar portable across OpenRouter upstreams.
    """

    if isinstance(value, dict):
        portable: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key in _PORTABLE_SCHEMA_UNSUPPORTED_KEYS:
                continue
            if key == "minItems" and isinstance(nested_value, int):
                portable[key] = min(nested_value, 1)
                continue
            portable[key] = _portable_structured_output_schema(nested_value)
        return portable
    if isinstance(value, list):
        return [_portable_structured_output_schema(item) for item in value]
    return value


def _extract_json_object(content: str) -> str:
    """Strip a markdown code fence around a JSON object, if the model added one
    despite instructions not to."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _validate_json_object(
    content: str, response_model: type[ResponseModel]
) -> ResponseModel:
    """Validate a structured response, tolerating provider-added prose.

    Some provider/model routes ignore JSON-only response formatting and wrap
    the object in analysis or a closing note. Try the normal strict path first,
    then examine each complete JSON object embedded in the response and accept
    only one that validates against the requested schema. Schema validation is
    what prevents an incidental object in the prose from being mistaken for the
    actual answer.
    """

    normalized = _extract_json_object(content)
    try:
        return response_model.model_validate_json(normalized)
    except ValidationError as primary_error:
        best_candidate_error = primary_error
        best_candidate_error_count = len(primary_error.errors())
        decoder = json.JSONDecoder()
        for start, character in enumerate(content):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content, start)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            try:
                return response_model.model_validate(candidate)
            except ValidationError as candidate_error:
                candidate_error_count = len(candidate_error.errors())
                if candidate_error_count < best_candidate_error_count:
                    best_candidate_error = candidate_error
                    best_candidate_error_count = candidate_error_count

        raise best_candidate_error


def _is_truncated_json_error(error: ValidationError) -> bool:
    return any(
        issue.get("type") == "json_invalid"
        and "eof" in str(issue.get("ctx", {})).casefold()
        for issue in error.errors(include_input=False)
    )


def _response_hit_output_limit(finish_reason: str | None) -> bool:
    if finish_reason is None:
        return False
    return finish_reason.casefold() in {"length", "max_tokens", "max_output_tokens"}


def _validation_error_summary(error: ValidationError, limit: int = 5) -> str:
    """Return actionable schema errors without echoing model-supplied content."""

    summaries: list[str] = []
    for issue in error.errors(include_url=False, include_input=False)[:limit]:
        location = ".".join(str(part) for part in issue.get("loc", ())) or "<root>"
        summaries.append(
            f"{location} [{issue.get('type', 'validation_error')}]: "
            f"{issue.get('msg', 'invalid value')}"
        )
    return "; ".join(summaries)


def is_retryable_provider_error(error: BaseException) -> bool:
    from litellm.exceptions import (
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
    )
    from litellm.exceptions import Timeout as LiteLLMTimeout

    return isinstance(
        error,
        (
            LLMRateLimitError,
            LLMTimeoutError,
            RateLimitError,
            LiteLLMTimeout,
            APIConnectionError,
            ServiceUnavailableError,
            InternalServerError,
        ),
    )


def _retry_after_seconds(error: BaseException) -> float | None:
    pending = [error]
    seen: set[int] = set()
    delays: list[float] = []
    while pending and len(seen) < 8:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        for owner in (current, getattr(current, "response", None)):
            headers = getattr(owner, "headers", None)
            if not isinstance(headers, Mapping):
                continue
            for key, value in headers.items():
                if (
                    isinstance(key, str)
                    and key.casefold() == "retry-after"
                    and isinstance(value, str)
                ):
                    delay = parse_retry_after_seconds(value)
                    if delay is not None:
                        delays.append(delay)
        pending.extend(
            item
            for item in (current.__cause__, current.__context__, *current.args[:1])
            if isinstance(item, BaseException) and id(item) not in seen
        )
    return max(delays) if delays else None


def generate_structured(
    llm: LLM,
    *,
    flow: LLMFlow,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModel],
    image_parts: list[ImageContentPart] | None = None,
    timeout_override: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_attempts: int = 2,
    provider_max_attempts: int = 3,
    deadline: float | None = None,
    use_streaming: bool | None = None,
) -> ResponseModel:
    """Call the LLM and parse+validate its response as `response_model`.

    By default, retries once and feeds back the validation error if the first
    response isn't valid JSON matching the schema. Optional invocation limits
    are forwarded only when supplied. An optional absolute monotonic deadline
    bounds calls and retry waits across both provider and validation attempts.
    """
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if provider_max_attempts < 1:
        raise ValueError("provider_max_attempts must be at least 1")

    validation_schema_json = response_model.model_json_schema()
    provider_schema_json = _portable_structured_output_schema(validation_schema_json)
    full_system_prompt = system_prompt + _JSON_ONLY_REMINDER.format(
        schema=validation_schema_json
    )

    messages: list[ChatCompletionMessage] = [
        SystemMessage(content=full_system_prompt),
        UserMessage(
            content=[TextContentPart(text=user_prompt), *image_parts]
            if image_parts
            else user_prompt
        ),
    ]

    structured_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": provider_schema_json,
            "strict": False,
        },
    }

    invoke_options: _StructuredInvokeOptions = {
        "structured_response_format": structured_response_format
    }
    if timeout_override is not None:
        invoke_options["timeout_override"] = timeout_override
    if max_tokens is not None:
        invoke_options["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        invoke_options["reasoning_effort"] = reasoning_effort
    if use_streaming is not None:
        invoke_options["use_streaming"] = use_streaming

    last_error: ValidationError | None = None
    for attempt in range(max_attempts):
        response: ModelResponse | None = None
        for provider_attempt in range(provider_max_attempts):
            if deadline is not None:
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("structured LLM deadline exhausted")
                invoke_options["timeout_override"] = (
                    min(timeout_override, remaining)
                    if timeout_override is not None
                    else remaining
                )
            try:
                with llm_generation_span(
                    llm=llm, flow=flow, input_messages=messages
                ) as span:
                    response = llm.invoke(messages, **invoke_options)
                    record_llm_response(span, response)
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("structured LLM deadline exhausted")
                break
            except Exception as error:
                if not is_retryable_provider_error(error):
                    raise
                if provider_attempt >= provider_max_attempts - 1:
                    raise
                scheduled_delay_s = min(
                    LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S,
                    LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S * (2**provider_attempt),
                )
                jitter_s = scheduled_delay_s * LLM_FIRST_CHUNK_RETRY_JITTER_RATIO
                retry_delay_s = random.uniform(
                    max(0.0, scheduled_delay_s - jitter_s),
                    scheduled_delay_s + jitter_s,
                )
                provider_delay = _retry_after_seconds(error)
                if provider_delay is not None:
                    retry_delay_s = provider_delay
                    if (
                        deadline is None
                        and retry_delay_s > LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S
                    ):
                        raise
                if (
                    deadline is not None
                    and retry_delay_s >= deadline - time.monotonic()
                ):
                    raise
                logger.warning(
                    "generate_structured: retrying transient provider error for %s "
                    "after %s on attempt %d/%d in %.1f seconds",
                    response_model.__name__,
                    type(error).__name__,
                    provider_attempt + 1,
                    provider_max_attempts,
                    retry_delay_s,
                )
                if (
                    deadline is not None
                    and retry_delay_s >= deadline - time.monotonic()
                ):
                    raise
                time.sleep(retry_delay_s)

        if response is None:
            raise RuntimeError("structured LLM invocation produced no response")
        content = llm_response_to_string(response)
        try:
            result = _validate_json_object(content, response_model)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("structured LLM deadline exhausted")
            return result
        except ValidationError as e:
            last_error = e
            finish_reason = response.choice.finish_reason
            logger.warning(
                "generate_structured: invalid response for %s (attempt %d) "
                "finish_reason=%s: %s",
                response_model.__name__,
                attempt + 1,
                finish_reason,
                e,
            )
            if _is_truncated_json_error(e) or _response_hit_output_limit(finish_reason):
                messages = [
                    messages[0],
                    messages[1],
                    UserMessage(
                        content=(
                            "Your previous JSON response was truncated. Restart from "
                            "scratch and keep every string and array concise enough to "
                            "finish within the output limit. Obey all schema size "
                            "limits. Respond with ONLY the complete JSON object."
                        )
                    ),
                ]
            else:
                messages = [
                    *messages,
                    AssistantMessage(
                        content=content[:_MAX_INVALID_RESPONSE_FEEDBACK_CHARS].rstrip()
                    ),
                    UserMessage(
                        content=(
                            f"Your previous response was not valid JSON for the "
                            f"required schema. Error:\n{e}\n\nRespond again with "
                            "ONLY the corrected JSON object."
                        )
                    ),
                ]

    assert last_error is not None
    raise StructuredOutputValidationError(
        f"LLM failed to produce valid {response_model.__name__} after "
        f"{max_attempts} attempts: {_validation_error_summary(last_error)}"
    ) from last_error
