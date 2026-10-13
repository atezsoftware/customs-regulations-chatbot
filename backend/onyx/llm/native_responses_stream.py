"""Instance-scoped native Responses streaming for models absent from LiteLLM's map."""

import copy
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from onyx.llm.model_response import Usage
from onyx.llm.tool_argument_stream import close_sync_stream

NATIVE_RESPONSE_TERMINAL = "asv3_native_response_terminal"

if TYPE_CHECKING:
    from litellm import HTTPHandler
    from litellm.types.utils import ModelResponseStream


class NativeResponseTerminalError(RuntimeError):
    """A failed or unsupported terminal response must not dispatch actions."""

    def __init__(self, capture: "NativeResponseTerminalCapture") -> None:
        self.usage = capture.usage
        self.terminal = capture.diagnostic()
        self.provider_error = capture.provider_error
        super().__init__(f"Native Responses terminal failure: {capture.status}")


@dataclass
class NativeResponseTerminalCapture:
    """Per-invocation provider metadata, never inferred from visible output."""

    event_type: str | None = None
    status: str | None = None
    response_id: str | None = None
    incomplete_reason: str | None = None
    provider_error: Any = None
    usage: Usage | None = None
    usage_status: str = "missing"
    reported_usage: dict[str, Any] | None = None

    def diagnostic(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "status": self.status,
            "provider_response_id": self.response_id,
            "incomplete_reason": self.incomplete_reason,
            "provider_error": self.provider_error,
            "usage_status": self.usage_status,
            "provider_reported_usage": self.reported_usage,
        }

    @property
    def failed(self) -> bool:
        return self.status not in {None, "completed", "incomplete"} or (
            self.status == "incomplete"
            and self.incomplete_reason != "max_output_tokens"
        )

    def events(self, response: Iterator[Any]) -> Iterator[Any]:
        try:
            for chunk in response:
                event = _response_event(chunk)
                kind = event.get("type") if event is not None else None
                if kind not in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    yield chunk
                    continue
                assert event is not None and isinstance(kind, str)
                self.event_type = kind
                data = event.get("response")
                if not isinstance(data, dict):
                    self.status = "invalid_terminal_response"
                    yield json.dumps(
                        {
                            **event,
                            "type": "response.completed",
                            "response": {"output": [], "usage": None},
                        }
                    )
                    continue
                expected_status = kind.removeprefix("response.")
                self.status = data.get("status")
                self.response_id = (
                    data["id"] if isinstance(data.get("id"), str) else None
                )
                self.provider_error = data.get("error")
                details = data.get("incomplete_details")
                reason = details.get("reason") if isinstance(details, dict) else None
                self.incomplete_reason = reason if isinstance(reason, str) else None
                raw_usage = data.get("usage")
                self.reported_usage = (
                    {
                        key: copy.deepcopy(raw_usage[key])
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "input_tokens_details",
                            "output_tokens_details",
                        )
                        if key in raw_usage
                    }
                    if isinstance(raw_usage, dict)
                    else None
                )
                self.usage = _reported_response_usage(raw_usage)
                self.usage_status = (
                    "provider_reported"
                    if self.usage is not None
                    else "missing"
                    if data.get("usage") is None
                    else "invalid"
                )
                if self.status != expected_status:
                    self.status = "invalid_terminal_status"
                if kind == "response.completed" and self.usage_status != "invalid":
                    yield chunk
                else:
                    # The installed bridge only normalizes completed events.
                    # Retain the actual status independently; it is never approval.
                    yield json.dumps(
                        {
                            **event,
                            "type": "response.completed",
                            "response": {
                                **data,
                                "usage": data.get("usage")
                                if self.usage is not None
                                else None,
                            },
                        }
                    )
        finally:
            close_sync_stream(response)


def _response_event(chunk: Any) -> dict[str, Any] | None:
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        value = chunk.model_dump()
        return value if isinstance(value, dict) else None
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8")
    if not isinstance(chunk, str):
        return None
    payload = chunk.lstrip()
    if payload.startswith("data:"):
        payload = payload[5:]
    try:
        value = json.loads(payload)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _reported_response_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    counts = [raw.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
    if any(type(value) is not int or value < 0 for value in counts):
        return None
    incoming, outgoing, total = cast(tuple[int, int, int], tuple(counts))
    if total != incoming + outgoing:
        return None
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        return None
    cached = input_details.get("cached_tokens", 0)
    writes = input_details.get("cache_write_tokens", 0)
    reasoning = output_details.get("reasoning_tokens")
    if (
        type(cached) is not int
        or not 0 <= cached <= incoming
        or type(writes) is not int
        or writes < 0
        or cached + writes > incoming
        or (
            reasoning is not None
            and (type(reasoning) is not int or not 0 <= reasoning <= outgoing)
        )
    ):
        return None
    return Usage(
        prompt_tokens=incoming,
        completion_tokens=outgoing,
        total_tokens=total,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=writes,
        reasoning_tokens=reasoning,
    )


def _native_sse_events(response: Any) -> Iterator[Any]:
    """Read raw SDK SSE events before LiteLLM can discard a failed terminal."""
    from httpx import Response
    from openai._streaming import SSEDecoder

    http_response = getattr(response, "response", None)
    try:
        if isinstance(http_response, Response):
            for event in SSEDecoder().iter_bytes(http_response.iter_bytes()):
                yield event.data
        else:
            yield from response
    finally:
        close_sync_stream(response)


def native_responses_stream(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict],
    optional_params: dict[str, Any],
    passthrough_kwargs: dict[str, Any],
    api_base: str | None,
    timeout: int | None,
    max_tokens: int | None,
    client: "HTTPHandler | None",
    terminal_capture: NativeResponseTerminalCapture | None = None,
) -> Iterator["ModelResponseStream"]:
    """Use the ordinary LiteLLM transforms/transport with fake_stream disabled.

    The public Responses entry point chooses fake streaming from global model
    metadata and has no per-call override. Keep this bypass confined to a bound
    OpenAI tool-decision call; no provider capability or global registry changes.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.types.router import GenericLiteLLMParams

    options: dict[str, Any] = {
        **optional_params,
        **passthrough_kwargs,
        "tools": tools,
        "stream": True,
        "max_tokens": max_tokens,
    }
    provider_config = OpenAIResponsesAPIConfig()
    logging = Logging(
        model=model,
        messages=messages,
        stream=True,
        call_type="responses",
        start_time=datetime.now(timezone.utc),
        litellm_call_id=str(uuid4()),
        function_id=str(uuid4()),
        kwargs=passthrough_kwargs,
    )
    transformation = LiteLLMResponsesTransformationHandler()
    request = transformation.transform_request(
        model=model,
        messages=cast(Any, messages),
        optional_params=options,
        litellm_params={**passthrough_kwargs, "api_base": api_base},
        headers=passthrough_kwargs.get("extra_headers") or {},
        litellm_logging_obj=logging,
        client=client,
    )
    requested = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
        request
    )
    request_params = ResponsesAPIRequestUtils.get_optional_params_responses_api(
        model=model,
        responses_api_provider_config=provider_config,
        response_api_optional_params=requested,
        allowed_openai_params=options.get("allowed_openai_params"),
    )
    credential_params: dict[str, Any] = {
        **passthrough_kwargs,
        "api_base": api_base,
        "custom_llm_provider": "openai",
    }
    params = GenericLiteLLMParams(**credential_params)
    logging.update_from_kwargs(
        kwargs=passthrough_kwargs,
        model=model,
        optional_params=request_params,
        litellm_params={**dict(params), "aresponses": False},
        custom_llm_provider="openai",
    )
    try:
        response = BaseLLMHTTPHandler().response_api_handler(
            model=model,
            input=request["input"],
            responses_api_provider_config=provider_config,
            response_api_optional_request_params=request_params,
            custom_llm_provider="openai",
            litellm_params=params,
            logging_obj=logging,
            extra_headers=request.get("extra_headers"),
            extra_body=request.get("extra_body"),
            timeout=timeout,
            client=client,
            fake_stream=False,
            litellm_metadata=request.get("litellm_metadata") or {},
        )
    except Exception as error:
        from onyx.llm.litellm_singleton import litellm

        raise litellm.exception_type(
            model=model,
            custom_llm_provider="openai",
            original_exception=error,
            completion_kwargs=request,
            extra_kwargs=passthrough_kwargs,
        )
    if terminal_capture is not None:
        response = terminal_capture.events(_native_sse_events(response))
    completion_stream = transformation.get_model_response_iterator(
        streaming_response=cast(Any, response), sync_stream=True
    )
    return cast(
        Iterator["ModelResponseStream"],
        CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="openai",
            logging_obj=logging,
        ),
    )
