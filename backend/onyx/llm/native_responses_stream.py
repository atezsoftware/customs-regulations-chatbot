"""Instance-scoped native Responses streaming for models absent from LiteLLM's map."""

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from litellm import HTTPHandler
    from litellm.types.utils import ModelResponseStream


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
