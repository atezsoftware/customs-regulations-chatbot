from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

from onyx.regulatory.benchmark.usage_capture import (
    BenchmarkUsageProcessor,
    LLMCallUsage,
    benchmark_usage_capture,
)
from onyx.tracing.framework.create import ensure_trace, generation_span
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span


def _generation_span(
    *,
    model: str | None = "test-model",
    provider: str | None = "test-provider",
    flow: str | None = None,
    usage: dict[str, Any] | None = None,
) -> Span[Any]:
    model_config: dict[str, str] = {}
    if provider is not None:
        model_config["model_provider"] = provider
    if flow is not None:
        model_config["flow"] = flow
    span_data = GenerationSpanData(
        model=model,
        model_config=model_config or None,
        usage=usage,
    )
    return cast(Span[Any], SimpleNamespace(span_data=span_data))


def test_capture_collects_generation_usage_and_aliases() -> None:
    processor = BenchmarkUsageProcessor()

    with benchmark_usage_capture() as bucket:
        processor.on_span_end(
            _generation_span(
                flow="chat_response",
                usage={
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cost": 0.000123,
                },
            )
        )

    assert len(bucket) == 1
    usage = bucket[0]
    assert usage.model == "test-model"
    assert usage.provider == "test-provider"
    assert usage.input_tokens == 17
    assert usage.output_tokens == 5
    assert usage.cache_read_tokens == 3
    assert usage.cost_cents == 0.0123
    assert usage.flow == "chat_response"


def test_usage_is_backward_compatible_and_serializes_nullable_flow() -> None:
    usage = LLMCallUsage("legacy-model", None, 3, 2, 1)

    assert usage.flow is None
    assert usage.to_dict() == {
        "model": "legacy-model",
        "provider": None,
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_read_tokens": 1,
        "flow": None,
    }


def test_capture_ignores_spans_outside_scope_or_without_usage() -> None:
    processor = BenchmarkUsageProcessor()

    processor.on_span_end(_generation_span(usage={"input_tokens": 10}))
    with benchmark_usage_capture() as bucket:
        processor.on_span_end(_generation_span(usage=None))
        processor.on_span_end(_generation_span(model=None, usage={"input_tokens": 10}))

    assert bucket == []


def test_nested_capture_scopes_are_isolated() -> None:
    processor = BenchmarkUsageProcessor()

    with benchmark_usage_capture() as outer_bucket:
        processor.on_span_end(_generation_span(usage={"input_tokens": 1}))
        with benchmark_usage_capture() as inner_bucket:
            processor.on_span_end(_generation_span(usage={"output_tokens": 2}))
        processor.on_span_end(_generation_span(usage={"input_tokens": 3}))

    assert [usage.input_tokens for usage in outer_bucket] == [1, 3]
    assert [usage.output_tokens for usage in inner_bucket] == [2]


def test_concurrent_capture_scopes_do_not_mix_item_usage() -> None:
    processor = BenchmarkUsageProcessor()
    barrier = Barrier(2)

    def capture(model: str, input_tokens: int) -> list[LLMCallUsage]:
        with benchmark_usage_capture() as bucket:
            barrier.wait()
            processor.on_span_end(
                _generation_span(
                    model=model,
                    usage={"input_tokens": input_tokens},
                )
            )
        return bucket

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(capture, "first-model", 11)
        second = executor.submit(capture, "second-model", 22)

    assert [(usage.model, usage.input_tokens) for usage in first.result()] == [
        ("first-model", 11)
    ]
    assert [(usage.model, usage.input_tokens) for usage in second.result()] == [
        ("second-model", 22)
    ]


def test_capture_requires_and_observes_a_real_trace() -> None:
    with benchmark_usage_capture() as bucket:
        with ensure_trace("benchmark_usage_test"):
            with generation_span(
                model="judge-model",
                model_config={
                    "model_provider": "OpenRouter",
                    "flow": "benchmark_judge",
                },
                usage={"input_tokens": 21, "output_tokens": 8},
            ):
                pass

    assert [(usage.input_tokens, usage.output_tokens) for usage in bucket] == [(21, 8)]
    assert [usage.flow for usage in bucket] == ["benchmark_judge"]
