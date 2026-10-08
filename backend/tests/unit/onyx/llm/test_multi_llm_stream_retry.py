from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import Timeout as LiteLLMTimeout

from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.llm.interfaces import LanguageModelInput
from onyx.llm.model_response import Delta, ModelResponseStream, StreamingChoice
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM


def _make_fake_llm() -> MagicMock:
    llm = MagicMock()
    llm.config.model_name = "gpt-test"
    llm.config.model_provider = "openai"
    llm._timeout = 30
    llm._track_llm_cost = MagicMock()
    llm._stream_cancellation_check = None
    return llm


def _make_prompt() -> LanguageModelInput:
    return [UserMessage(content="hello")]


def _make_stream_response(content: str) -> ModelResponseStream:
    return ModelResponseStream(
        id="chunk-1",
        created="1",
        choice=StreamingChoice(delta=Delta(content=content)),
    )


def test_stream_retries_timeout_before_first_chunk() -> None:
    fake_llm = _make_fake_llm()
    translated_chunk = _make_stream_response("hello")
    attempt_count = 0

    def completion_side_effect(**_kwargs: object) -> list[object]:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise LiteLLMTimeout("timed out", "gpt-test", "openai")
        return [object()]

    fake_llm._completion = MagicMock(side_effect=completion_side_effect)

    with (
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_MAX_RETRIES", 1),
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S", 2.0),
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S", 10.0),
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_RETRY_JITTER_RATIO", 0.25),
        patch("onyx.llm.multi_llm.random.uniform", return_value=2.25) as jitter,
        patch("onyx.llm.multi_llm.time.sleep") as sleep,
        patch("onyx.llm.multi_llm.is_true_openai_model", return_value=False),
        patch(
            "onyx.llm.model_response.from_litellm_model_response_stream",
            return_value=translated_chunk,
        ),
        patch("onyx.llm.multi_llm.logger") as mock_logger,
    ):
        # Bind the unbound method to a fake self to isolate retry behavior.
        results = list(LitellmLLM.stream(fake_llm, prompt=_make_prompt()))

    assert len(results) == 1
    assert results[0].choice.delta.content == "hello"
    assert fake_llm._completion.call_count == 2
    jitter.assert_called_once_with(1.5, 2.5)
    sleep.assert_called_once_with(2.25)
    mock_logger.warning.assert_called_once()


def test_stream_does_not_retry_after_first_chunk() -> None:
    fake_llm = _make_fake_llm()
    translated_chunk = _make_stream_response("partial")

    def stream_then_timeout() -> Iterator[object]:
        yield object()
        raise LiteLLMTimeout("timed out", "gpt-test", "openai")

    fake_llm._completion = MagicMock(return_value=stream_then_timeout())

    with (
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_MAX_RETRIES", 2),
        patch("onyx.llm.multi_llm.time.sleep") as sleep,
        patch("onyx.llm.multi_llm.is_true_openai_model", return_value=False),
        patch(
            "onyx.llm.model_response.from_litellm_model_response_stream",
            return_value=translated_chunk,
        ),
        patch("onyx.llm.multi_llm.logger") as mock_logger,
    ):
        # Bind the unbound method to a fake self to isolate retry behavior.
        with pytest.raises(LiteLLMTimeout):
            list(LitellmLLM.stream(fake_llm, prompt=_make_prompt()))

    assert fake_llm._completion.call_count == 1
    sleep.assert_not_called()
    mock_logger.warning.assert_not_called()


@pytest.mark.parametrize("seed", [None, 0, -(2**31)])
def test_real_llm_retries_timeout_deferred_until_first_chunk(seed: int | None) -> None:
    llm = LitellmLLM(
        api_key="test-key",
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        max_input_tokens=100000,
        seed=seed,
    )
    attempt_count = 0

    def completion(**_kwargs: object) -> Iterator[object]:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise LiteLLMTimeout("timed out", "gemini-3.8-flash", "vertex_ai")
        yield object()

    with (
        patch(
            "onyx.llm.litellm_singleton.litellm.completion", side_effect=completion
        ) as provider,
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_MAX_RETRIES", 1),
        patch("onyx.llm.multi_llm.time.sleep"),
        patch(
            "onyx.llm.model_response.from_litellm_model_response_stream",
            return_value=_make_stream_response("hello"),
        ),
    ):
        results = list(llm.stream(prompt=_make_prompt()))

    assert [result.choice.delta.content for result in results] == ["hello"]
    assert attempt_count == 2
    for call in provider.call_args_list:
        if seed is None:
            assert "seed" not in call.kwargs
        else:
            assert call.kwargs["seed"] == seed
        assert "top_p" not in call.kwargs


@pytest.mark.parametrize("stopped_before_call", [False, True])
def test_run_bound_stream_never_retries_a_provider_after_owner_stop(
    stopped_before_call: bool,
) -> None:
    original_llm = LitellmLLM(
        api_key="test-key",
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        max_input_tokens=100000,
    )
    stopped = stopped_before_call
    context = RunContext(
        cancelled=lambda: stopped, budget=SharedBudget(unlimited_execution=True)
    )
    bound = original_llm.with_stream_cancellation_check(context.check_active)

    def timed_out_without_chunks(**_kwargs: object) -> Iterator[object]:
        nonlocal stopped
        stopped = True
        raise LiteLLMTimeout("timed out", "gemini-3.8-flash", "vertex_ai")
        yield  # pragma: no cover

    with (
        patch(
            "onyx.llm.litellm_singleton.litellm.completion",
            side_effect=timed_out_without_chunks,
        ) as provider,
        patch("onyx.llm.multi_llm.LLM_FIRST_CHUNK_MAX_RETRIES", 2),
        patch("onyx.llm.multi_llm.time.sleep") as sleep,
    ):
        with pytest.raises(RunStopped, match="cancel"):
            list(bound.stream(prompt=_make_prompt()))

    assert provider.call_count == (0 if stopped_before_call else 1)
    sleep.assert_not_called()
    assert bound is not original_llm
    assert bound.config == original_llm.config
    assert original_llm._stream_cancellation_check is None
