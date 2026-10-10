"""One Gemini stream bounded by the workflow's absolute phase deadline."""

from __future__ import annotations

import ssl
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, cast

import httpcore
import httpx
from httpcore._backends.base import SOCKET_OPTION

from onyx.llm.interfaces import LLMUserIdentity
from onyx.llm.model_response import ModelResponse, Usage
from onyx.llm.models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions
from onyx.llm.multi_llm import LitellmLLM


class Deadline:
    def __init__(
        self, end: float, idle_seconds: float, check_active: Callable[[], None]
    ) -> None:
        self.end = end
        self.idle_seconds = idle_seconds
        self.check_active = check_active

    def remaining(self, timeout: float | None = None) -> float:
        self.check_active()
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Legal Review reading phase deadline exceeded")
        return min(remaining, self.idle_seconds, timeout or self.idle_seconds)


class DeadlineStream(httpcore.NetworkStream):
    def __init__(self, stream: httpcore.NetworkStream, deadline: Deadline) -> None:
        self.stream = stream
        self.deadline = deadline

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self.stream.read(max_bytes, timeout=self.deadline.remaining(timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.stream.write(buffer, timeout=self.deadline.remaining(timeout))

    def close(self) -> None:
        self.stream.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        return DeadlineStream(
            self.stream.start_tls(
                ssl_context,
                server_hostname,
                timeout=self.deadline.remaining(timeout),
            ),
            self.deadline,
        )

    def get_extra_info(self, info: str) -> Any:
        return self.stream.get_extra_info(info)


class DeadlineBackend(httpcore.NetworkBackend):
    def __init__(self, backend: httpcore.NetworkBackend, deadline: Deadline) -> None:
        self.backend = backend
        self.deadline = deadline

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return DeadlineStream(
            self.backend.connect_tcp(
                host,
                port,
                timeout=self.deadline.remaining(timeout),
                local_address=local_address,
                socket_options=socket_options,
            ),
            self.deadline,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return DeadlineStream(
            self.backend.connect_unix_socket(
                path,
                timeout=self.deadline.remaining(timeout),
                socket_options=socket_options,
            ),
            self.deadline,
        )


def read_stream(
    selected: LitellmLLM,
    *,
    deadline: Deadline,
    prompt: LanguageModelInput,
    tools: list[dict] | None,
    tool_choice: ToolChoiceOptions | None,
    structured_response_format: dict | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort,
    user_identity: LLMUserIdentity | None,
    record_usage: Callable[[int, int], None],
) -> ModelResponse:
    from litellm import HTTPHandler, stream_chunk_builder
    from litellm.exceptions import Timeout as ProviderTimeout
    from litellm.types.utils import ModelResponse as RawResponse
    from litellm.types.utils import ModelResponseStream as RawChunk

    from onyx.llm.model_response import (
        from_litellm_model_response,
        from_litellm_model_response_stream,
    )

    transport = httpx.HTTPTransport(retries=0)
    # httpcore caches the body read timeout. Clamp each socket operation instead.
    pool = transport._pool
    assert isinstance(pool, httpcore.ConnectionPool)
    pool._network_backend = DeadlineBackend(pool._network_backend, deadline)
    client = HTTPHandler(
        client=httpx.Client(
            transport=transport,
            timeout=deadline.remaining(),
            follow_redirects=False,
        )
    )
    chunks: list[RawChunk] = []
    usage: Usage | None = None
    stream: Iterator[RawChunk] | None = None
    try:
        stream = cast(
            Iterator[RawChunk],
            selected._completion(
                prompt=prompt,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                parallel_tool_calls=True,
                reasoning_effort=reasoning_effort,
                structured_response_format=structured_response_format,
                timeout_override=max(1, int(deadline.remaining())),
                max_tokens=max_tokens,
                user_identity=user_identity,
                client=client,
                provider_compatibility_attempts=1,
            ),
        )
        for chunk in stream:
            converted = from_litellm_model_response_stream(chunk)
            if converted.usage is not None:
                usage = converted.usage
            deadline.remaining()
            chunks.append(chunk)
        deadline.remaining()
        raw_response = stream_chunk_builder(chunks)
        if not isinstance(raw_response, RawResponse):
            raise ValueError("Legal Review reader returned an invalid stream")
        response = from_litellm_model_response(raw_response)
        if response.choice.finish_reason != "stop":
            raise ValueError("Legal Review reader did not return a complete response")
        return response
    except (
        ProviderTimeout,
        httpx.TimeoutException,
        httpcore.TimeoutException,
    ) as error:
        raise TimeoutError("Legal Review Gemini reader deadline exceeded") from error
    finally:
        try:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        finally:
            try:
                client.close()
            finally:
                if usage is not None:
                    record_usage(usage.prompt_tokens, usage.completion_tokens)
                    selected._track_llm_cost(usage)
