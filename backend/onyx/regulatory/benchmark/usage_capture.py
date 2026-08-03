"""Captures exact per-item LLM usage for a benchmark run.

A benchmark item runs the real chat agent loop, which can call the LLM many
times across tool-calling cycles (search, think, etc.). The per-user
`UserUsage` rollup (onyx.db.user_usage) exists for billing, but it accumulates
asynchronously into a shared per-window bucket — not safely readable back
for one specific request.

Instead, this hooks the tracing framework directly: `BenchmarkUsageProcessor`
is registered once as a global trace processor (a no-op for all non-benchmark
traffic) and only collects when `benchmark_usage_capture()` has an active
contextvar scope, so it captures exactly and only the spans emitted by one
`handle_stream_message_objects()` call — every LLM call in every cycle of
that one turn, nothing from concurrent unrelated requests.
"""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any

from onyx.tracing.framework import add_trace_processor
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace


@dataclass(frozen=True)
class LLMCallUsage:
    model: str
    provider: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    flow: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "flow": self.flow,
        }


def _usage_field(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if value is not None:
            return int(value)
    return 0


_capture_var: ContextVar[list[LLMCallUsage] | None] = ContextVar(
    "benchmark_usage_capture", default=None
)


class BenchmarkUsageProcessor(TracingProcessor):
    """No-op unless a benchmark capture scope is active in the current
    context (contextvars are task/thread-local, so concurrent non-benchmark
    requests are never observed here)."""

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        bucket = _capture_var.get()
        if bucket is None:
            return
        span_data = span.span_data
        if not isinstance(span_data, GenerationSpanData):
            return
        if not span_data.usage or not span_data.model:
            return
        model_config = span_data.model_config or {}
        provider_value = model_config.get("model_provider")
        provider = str(provider_value) if provider_value is not None else None
        flow_value = model_config.get("flow")
        flow = str(flow_value) if flow_value is not None else None
        bucket.append(
            LLMCallUsage(
                model=span_data.model,
                provider=provider,
                input_tokens=_usage_field(
                    span_data.usage, "input_tokens", "prompt_tokens"
                ),
                output_tokens=_usage_field(
                    span_data.usage, "output_tokens", "completion_tokens"
                ),
                cache_read_tokens=_usage_field(
                    span_data.usage, "cache_read_input_tokens"
                ),
                flow=flow,
            )
        )

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


_processor = BenchmarkUsageProcessor()
_registered = False
_registration_lock = Lock()


def ensure_registered() -> None:
    global _registered
    if _registered:
        return
    with _registration_lock:
        if not _registered:
            add_trace_processor(_processor)
            _registered = True


@contextlib.contextmanager
def benchmark_usage_capture() -> Iterator[list[LLMCallUsage]]:
    """Yields a list that fills with one `LLMCallUsage` per LLM call made by
    code running inside this `with` block."""
    ensure_registered()
    bucket: list[LLMCallUsage] = []
    token = _capture_var.set(bucket)
    try:
        yield bucket
    finally:
        _capture_var.reset(token)
