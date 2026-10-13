"""Zero cache-write transport for a bound Cropped native Responses invocation."""

import copy
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from onyx.llm.asv3_exp_cheap_cache import NativeCacheRequest
from onyx.llm.tool_argument_stream import close_sync_stream
from onyx.tracing.framework.create import get_current_span
from onyx.tracing.framework.span_data import GenerationSpanData

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


@dataclass(frozen=True)
class CroppedNoCacheWriteContract:
    schema_version: Literal[1] = 1
    policy_id: Literal["cropped_native_explicit_zero_boundaries_v1"] = (
        "cropped_native_explicit_zero_boundaries_v1"
    )
    mode: Literal["explicit"] = "explicit"
    ttl: Literal["30m"] = "30m"
    marked_boundary_count: Literal[0] = 0
    provider: Literal["openai"] = "openai"
    model: Literal["gpt-6.1-sol"] = "gpt-6.1-sol"


def _without_breakpoints(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_breakpoints(item)
            for key, item in value.items()
            if key != "prompt_cache_breakpoint"
        }
    if isinstance(value, list):
        return [_without_breakpoints(item) for item in value]
    return copy.deepcopy(value)


class CroppedPromptCachePolicy:
    """Explicit mode without eligible write boundaries leaves content unchanged."""

    @property
    def cache_writes_disabled(self) -> Literal[True]:
        return True

    @property
    def contract(self) -> CroppedNoCacheWriteContract:
        return CroppedNoCacheWriteContract()

    def prepare_request(self, request: dict[str, Any]) -> NativeCacheRequest:
        if request.get("model") != self.contract.model:
            raise ValueError("Zero-write Cropped cache requires the bound Sol model")
        extra_body = copy.deepcopy(request.get("extra_body") or {})
        if ("model" in extra_body and extra_body["model"] != self.contract.model) or (
            "stream" in extra_body and extra_body["stream"] is not True
        ):
            raise ValueError("Zero-write Cropped cache cannot override native routing")
        if request.get("previous_response_id") or extra_body.get(
            "previous_response_id"
        ):
            raise ValueError("Zero-write Cropped requests require explicit full input")
        if "input" in extra_body:
            extra_body["input"] = _without_breakpoints(extra_body["input"])
        extra_body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        request["extra_body"] = extra_body
        if "input" in request:
            request["input"] = _without_breakpoints(request["input"])
        span = get_current_span()
        if span is not None and isinstance(span.span_data, GenerationSpanData):
            span.span_data.request_params = {
                **(span.span_data.request_params or {}),
                "asv3_exp_cropped_prompt_cache": {
                    "mode": "explicit",
                    "ttl": "30m",
                    "marker_message_index": None,
                    "marker_content_index": None,
                    "marker_count": 0,
                    "cache_writes_disabled": True,
                    "native_contract": asdict(self.contract),
                    "wire_observed": True,
                },
            }
        return NativeCacheRequest(boundary_hashes=())

    def capture_response(
        self, response: Iterator[Any], _capture: NativeCacheRequest
    ) -> Iterator[Any]:
        # Actual usage is independently retained by the terminal-event capture.
        try:
            yield from response
        finally:
            close_sync_stream(response)

    def preserve_cache_write_usage(
        self,
        response: Iterator["ModelResponseStream"],
        _capture: NativeCacheRequest,
    ) -> Iterator["ModelResponseStream"]:
        try:
            yield from response
        finally:
            close_sync_stream(response)
