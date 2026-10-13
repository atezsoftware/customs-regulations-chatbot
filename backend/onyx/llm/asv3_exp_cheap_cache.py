"""Cache transport state for an explicitly bound ASv3Exp Cheap model."""

import copy
import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from onyx.llm.tool_argument_stream import close_sync_stream
from onyx.tracing.framework.create import get_current_span
from onyx.tracing.framework.span_data import GenerationSpanData

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream
    from litellm.types.utils import Usage as LiteLLMUsage


@dataclass
class _ThreadCacheState:
    boundaries: tuple[str, ...] = ()
    history_boundaries: tuple[str, ...] = ()
    response_id: str | None = None


@dataclass
class NativeCacheRequest:
    boundary_hashes: tuple[str, ...]
    history_boundary_hashes: tuple[str, ...] = ()
    response_id: str | None = None
    cache_write_tokens: int | None = None
    diagnostics: dict[str, Any] | None = None
    completed: bool = False


def _without_breakpoints(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_breakpoints(item)
            for key, item in value.items()
            if key != "prompt_cache_breakpoint"
        }
    if isinstance(value, list):
        return [_without_breakpoints(item) for item in value]
    return value


def _text_block(item: dict[str, Any]) -> dict[str, Any] | None:
    blocks = (
        item.get("output")
        if item.get("type") == "function_call_output"
        else item.get("content")
    )
    if not isinstance(blocks, list) or not blocks:
        return None
    block = blocks[-1]
    return (
        block if isinstance(block, dict) and block.get("type") == "input_text" else None
    )


def _record_request_metadata(capture: NativeCacheRequest, metadata_source: str) -> None:
    span = get_current_span()
    if span is not None and isinstance(span.span_data, GenerationSpanData):
        span.span_data.request_params = {
            **(span.span_data.request_params or {}),
            "asv3_exp_cheap_cache": {
                "provider_response_id": capture.response_id,
                "cache_write_tokens": capture.cache_write_tokens,
                "prompt_cache_diagnostics": capture.diagnostics,
                "metadata_source": metadata_source,
            },
        }


class ASv3ExpCheapCachePolicy:
    """Keep at most three explicit write boundaries, separately per thread."""

    def __init__(self) -> None:
        self._threads = threading.local()

    def _state(self) -> _ThreadCacheState:
        state = getattr(self._threads, "state", None)
        if state is None:
            state = _ThreadCacheState()
            self._threads.state = state
        return cast(_ThreadCacheState, state)

    @property
    def has_recorded_history_boundary(self) -> bool:
        """A completed request marked native history; a cache hit is not implied."""
        return bool(self._state().history_boundaries)

    def prepare_request(self, request: dict[str, Any]) -> NativeCacheRequest:
        """Annotate converted blocks, preserving every rendered input value."""
        state = self._state()
        request["input"] = copy.deepcopy(request["input"])
        extra_body = copy.deepcopy(request.get("extra_body") or {})
        options = copy.deepcopy(extra_body.get("prompt_cache_options") or {})
        options.setdefault("mode", "explicit")
        options.setdefault("ttl", "30m")
        if state.response_id is not None:
            options.setdefault("comparison_response_id", state.response_id)
        extra_body["prompt_cache_options"] = options
        request["extra_body"] = extra_body

        context = {
            key: request.get(key)
            for key in (
                "model",
                "instructions",
                "tools",
                "reasoning",
                "text",
                "parallel_tool_calls",
                "tool_choice",
            )
        }
        prefix = hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True).encode()
        )
        hashes: dict[int, str] = {}
        tools: list[int] = []
        users: list[int] = []
        for index, item in enumerate(request["input"]):
            prefix.update(
                json.dumps(
                    _without_breakpoints(item), ensure_ascii=False, sort_keys=True
                ).encode()
            )
            prefix.update(b"\0")
            if not isinstance(item, dict) or _text_block(item) is None:
                continue
            if item.get("type") == "function_call_output":
                tools.append(index)
                hashes[index] = prefix.hexdigest()
            elif item.get("role") == "user":
                users.append(index)
                hashes[index] = prefix.hexdigest()

        latest = tools[-1] if tools else users[0] if len(users) > 1 else None
        selected: list[int] = []
        if latest is not None:
            previous = [
                index
                for index, value in hashes.items()
                if value in state.boundaries and index != latest
            ]
            selected = sorted([*previous[-2:], latest])
        for index in selected:
            block = _text_block(request["input"][index])
            assert block is not None
            block["prompt_cache_breakpoint"] = {"mode": "explicit"}
        capture = NativeCacheRequest(
            boundary_hashes=tuple(hashes[index] for index in selected),
            history_boundary_hashes=tuple(
                hashes[index] for index in selected if index in tools
            ),
        )
        _record_request_metadata(capture, "native_request")
        return capture

    def capture_response(
        self, response: Iterator[Any], capture: NativeCacheRequest
    ) -> Iterator[Any]:
        """Observe completed SSE metadata while yielding provider events intact."""
        try:
            yield from self._capture_events(response, capture)
        finally:
            close_sync_stream(response)

    def _capture_events(
        self, response: Iterator[Any], capture: NativeCacheRequest
    ) -> Iterator[Any]:
        for chunk in response:
            event: dict[str, Any] | None = None
            inspected = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            if isinstance(inspected, str) and "response.completed" in inspected:
                payload = (
                    inspected[inspected.find("data:") + 5 :]
                    if "data:" in inspected
                    else inspected
                )
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    event = parsed
            elif isinstance(chunk, dict):
                event = chunk
            elif hasattr(chunk, "model_dump"):
                event = chunk.model_dump()
            if event is not None and event.get("type") == "response.completed":
                completed = event.get("response")
                if not isinstance(completed, dict):
                    yield chunk
                    continue
                response_id = completed.get("id")
                if isinstance(response_id, str):
                    from litellm.responses.utils import ResponsesAPIRequestUtils

                    capture.response_id = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
                        response_id
                    )
                usage = completed.get("usage")
                details = (
                    usage.get("input_tokens_details")
                    if isinstance(usage, dict)
                    else None
                )
                details = details if isinstance(details, dict) else {}
                writes = details.get("cache_write_tokens")
                if (
                    isinstance(writes, int)
                    and not isinstance(writes, bool)
                    and writes >= 0
                ):
                    capture.cache_write_tokens = writes
                diagnostics = completed.get("prompt_cache_diagnostics")
                capture.diagnostics = (
                    diagnostics if isinstance(diagnostics, dict) else None
                )
                capture.completed = True
                state = self._state()
                state.boundaries = capture.boundary_hashes
                state.history_boundaries = (
                    capture.history_boundary_hashes or state.history_boundaries
                )
                state.response_id = capture.response_id
                _record_request_metadata(capture, "native_response_completed")
            yield chunk

    def preserve_cache_write_usage(
        self, response: Iterator["ModelResponseStream"], capture: NativeCacheRequest
    ) -> Iterator["ModelResponseStream"]:
        try:
            for chunk in response:
                usage = cast("LiteLLMUsage | None", getattr(chunk, "usage", None))
                if (
                    capture.completed
                    and capture.cache_write_tokens is not None
                    and usage is not None
                ):
                    # Keep writes distinct; absent provider-specific pricing must remain incomplete.
                    setattr(
                        usage, "cache_creation_input_tokens", capture.cache_write_tokens
                    )
                yield chunk
        finally:
            close_sync_stream(response)
