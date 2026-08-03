"""Schema-constrained JSON generation for regulatory LLM workflows.

The amendment pipeline (segmenter/matcher/drafter) needs typed, validated LLM
output rather than free text. `structured_response_format` on `LLM.invoke`
maps straight to the provider's JSON-schema mode when supported; on the
(common, multi-provider) case where that support is inconsistent, one retry
with the validation error fed back closes the gap without depending on any
single provider's structured-output guarantees.
"""

from typing import Any, NotRequired, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

from onyx.llm.interfaces import LLM
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.utils import llm_response_to_string
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger

logger = setup_logger()

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class _StructuredInvokeOptions(TypedDict):
    structured_response_format: dict
    timeout_override: NotRequired[int]
    max_tokens: NotRequired[int]
    reasoning_effort: NotRequired[ReasoningEffort]

_JSON_ONLY_REMINDER = (
    "\n\nCRITICAL: Respond with ONLY a single valid JSON object matching the "
    "schema below. No prose, no markdown code fences, no explanation outside "
    "the JSON.\n\nSchema:\n{schema}"
)

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


def generate_structured(
    llm: LLM,
    *,
    flow: LLMFlow,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModel],
    timeout_override: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_attempts: int = 2,
) -> ResponseModel:
    """Call the LLM and parse+validate its response as `response_model`.

    By default, retries once and feeds back the validation error if the first
    response isn't valid JSON matching the schema. Optional invocation limits
    are forwarded only when supplied, preserving existing callers' behavior.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    validation_schema_json = response_model.model_json_schema()
    provider_schema_json = _portable_structured_output_schema(
        validation_schema_json
    )
    full_system_prompt = system_prompt + _JSON_ONLY_REMINDER.format(
        schema=validation_schema_json
    )

    messages: list[ChatCompletionMessage] = [
        SystemMessage(content=full_system_prompt),
        UserMessage(content=user_prompt),
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

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        with llm_generation_span(llm=llm, flow=flow, input_messages=messages) as span:
            response = llm.invoke(messages, **invoke_options)
            record_llm_response(span, response)

        content = llm_response_to_string(response)
        try:
            return response_model.model_validate_json(_extract_json_object(content))
        except ValidationError as e:
            last_error = e
            logger.warning(
                "generate_structured: invalid response for %s (attempt %d): %s",
                response_model.__name__,
                attempt + 1,
                e,
            )
            messages = [
                *messages,
                UserMessage(
                    content=(
                        f"Your previous response was not valid JSON for the "
                        f"required schema. Error:\n{e}\n\nRespond again with "
                        "ONLY the corrected JSON object."
                    )
                ),
            ]

    assert last_error is not None
    raise ValueError(
        f"LLM failed to produce valid {response_model.__name__} after "
        f"{max_attempts} attempts"
    ) from last_error
