"""Per-turn cancellation survives cache recovery, concurrent readers and new turns."""

import socket
import threading
import time
from collections.abc import Generator
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, call
from uuid import UUID, uuid4

import pytest
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from onyx.asv3 import runtime
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.performance import FastCancellationProbe
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
)
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.stop_signal_checker import (
    FENCE_PREFIX,
    TurnCancellationProbe,
    reset_cancel_status,
    set_fence,
)
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.models import ReasoningEffort
from onyx.server.query_and_chat.streaming_models import OverallStop, Packet
from tests.unit.onyx.chat.test_cropped_stream_writer_lifecycle import _CroppedRun
from tests.unit.onyx.chat.test_stop_signal_checker import _MemoryCacheBackend


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("external connection")
        raise AssertionError("Cancellation tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
    assert not attempts


def test_healthy_probe_reads_cache_once_every_call_without_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = MagicMock(spec=CacheBackend)
    cache.exists.return_value = False
    session_id = uuid4()
    connected = TurnCancellationProbe(session_id, cache)

    def no_sleep(_seconds: float) -> None:
        raise AssertionError("Healthy cancellation reads must not sleep")

    monkeypatch.setattr(time, "sleep", no_sleep)
    for expected_calls in range(1, 51):
        assert connected()
        assert cache.exists.call_count == expected_calls
    assert cache.mock_calls == [call.exists(f"{FENCE_PREFIX}_{session_id}")] * 50


def test_observed_stop_stays_with_old_turn_after_reset() -> None:
    cache = _MemoryCacheBackend()
    session_id = uuid4()
    old_turn = TurnCancellationProbe(session_id, cache)
    assert old_turn()
    set_fence(session_id, cache, True)
    assert not old_turn()
    reset_cancel_status(session_id, cache)
    new_turn = TurnCancellationProbe(session_id, cache)
    assert new_turn()
    assert not old_turn()
    native_context = RunContext(cancelled=lambda: not old_turn())
    with pytest.raises(RunStopped, match="cancelled"):
        native_context.check_active()


@pytest.mark.parametrize("error_type", [RedisError, SQLAlchemyError])
def test_transient_failure_retries_on_next_call_and_does_not_latch(
    error_type: type[Exception],
) -> None:
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = [error_type("cache unavailable"), False, True, False]
    connected = TurnCancellationProbe(uuid4(), cache)
    assert connected()
    assert cache.exists.call_count == 1
    assert connected()
    assert cache.exists.call_count == 2
    assert not connected()
    assert cache.exists.call_count == 3
    assert not connected()
    assert cache.exists.call_count == 3


def test_unknown_failure_propagates_without_becoming_cancellation() -> None:
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = [ValueError("invalid cache implementation"), False]
    connected = TurnCancellationProbe(uuid4(), cache)
    with pytest.raises(ValueError, match="invalid cache implementation"):
        connected()
    assert connected()
    assert cache.exists.call_count == 2


@pytest.mark.parametrize("late_error", [None, RedisError, SQLAlchemyError])
def test_concurrent_late_response_cannot_revive_observed_stop(
    late_error: type[Exception] | None,
) -> None:
    cache = MagicMock(spec=CacheBackend)
    first_entered, first_release, second_finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    order_lock = threading.Lock()
    calls = 0
    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def exists(_key: str) -> bool:
        nonlocal calls
        with order_lock:
            calls += 1
            position = calls
        if position == 1:
            first_entered.set()
            assert first_release.wait(5)
            if late_error is not None:
                raise late_error("late transient failure")
            return False
        return True

    cache.exists.side_effect = exists
    connected = TurnCancellationProbe(uuid4(), cache)

    def run(name: str) -> None:
        try:
            results[name] = connected()
        except BaseException as error:
            errors.append(error)
        finally:
            if name == "stop":
                second_finished.set()

    stale = threading.Thread(target=run, args=("stale",))
    stop = threading.Thread(target=run, args=("stop",))
    stale.start()
    try:
        assert first_entered.wait(5)
        stop.start()
        assert second_finished.wait(5)
        assert results["stop"] is False
    finally:
        first_release.set()
        stale.join(5)
        if stop.ident is not None:
            stop.join(5)
    assert not stale.is_alive() and not stop.is_alive()
    assert not errors
    assert results == {"stop": False, "stale": False}
    assert not connected()
    assert cache.exists.call_count == 2


def test_transient_logging_is_bounded_without_throttling_cache_reads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = MagicMock(return_value=100.0)
    monkeypatch.setattr("onyx.chat.stop_signal_checker.time.monotonic", clock)
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = RedisError("cache unavailable")
    connected = TurnCancellationProbe(uuid4(), cache)
    for _ in range(10):
        assert connected()
    assert cache.exists.call_count == 10
    records = [r for r in caplog.records if r.name == "onyx.chat.stop_signal_checker"]
    assert len(records) == 1
    clock.return_value = 161.0
    assert connected()
    assert cache.exists.call_count == 11
    records = [r for r in caplog.records if r.name == "onyx.chat.stop_signal_checker"]
    assert len(records) == 2


def test_fast_timing_wrapper_keeps_the_same_active_lookup_counts() -> None:
    cache = MagicMock(spec=CacheBackend)
    cache.exists.return_value = False
    connected = TurnCancellationProbe(uuid4(), cache)
    cancelled = FastCancellationProbe(lambda: not connected())
    for _ in range(10):
        assert not cancelled()
    assert cache.exists.call_count == 10
    assert cancelled.snapshot()["count"] == 10


def _capture_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
    cache: CacheBackend,
    session_id: UUID,
    variant: str,
    connected: TurnCancellationProbe | None,
) -> RunContext:
    class ContextCaptured(Exception):
        pass

    contexts: list[RunContext] = []

    def capture(**options: Any) -> RunContext:
        contexts.append(RunContext(**options))
        raise ContextCaptured

    monkeypatch.setattr(runtime, "RunContext", capture)
    monkeypatch.setattr(
        runtime, "ensure_trace", lambda *_args, **_kwargs: nullcontext()
    )
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="offline",
        max_input_tokens=200000,
        temperature=0,
    )
    options: dict[str, Any] = dict(
        emitter=MagicMock(),
        state_container=ChatStateContainer(),
        simple_chat_history=[],
        tools=[],
        llm=llm,
        token_counter=len,
        user=MagicMock(id=uuid4()),
        filters=None,
        document_set_names_override=None,
        user_identity=None,
        chat_session_id=session_id,
        user_message_id=1,
        assistant_message_id=2,
        reasoning_effort=ReasoningEffort.AUTO,
        include_citations=True,
        cache=cache,
        research_profile="normal",
        workflow_variant=variant,
    )
    if connected is not None:
        options["check_is_connected"] = connected
    with pytest.raises(ContextCaptured):
        runtime.run_asv3_loop(**options)
    llm.invoke.assert_not_called()
    return contexts[0]


@pytest.mark.parametrize(
    "variant", [ASV3_EXP_CROPPED_VARIANT, ASV3_FAST_CROPPED_VARIANT]
)
@pytest.mark.parametrize("provided", [False, True])
def test_actual_runtime_uses_shared_or_fresh_latched_probe(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    provided: bool,
) -> None:
    cache = _MemoryCacheBackend()
    session_id = uuid4()
    connected = TurnCancellationProbe(session_id, cache) if provided else None
    old_context = _capture_runtime_context(
        monkeypatch, cache, session_id, variant, connected
    )
    old_context.check_active()
    set_fence(session_id, cache, True)
    if connected is not None:
        assert not connected()
    else:
        assert old_context.is_cancelled()
    reset_cancel_status(session_id, cache)
    new_context = _capture_runtime_context(
        monkeypatch, cache, session_id, variant, None
    )
    new_context.check_active()
    with pytest.raises(RunStopped, match="cancelled"):
        old_context.check_active()


def test_actual_writer_passes_its_latch_to_native_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _CroppedRun(monkeypatch)
    connected = TurnCancellationProbe(run.setup.chat_session.id, run.cache)
    run.setup.check_is_connected = connected
    contexts: list[RunContext] = []
    original_native = run._native

    def native(**kwargs: Any) -> None:
        assert kwargs["check_is_connected"] is connected
        contexts.append(
            RunContext(cancelled=lambda: not kwargs["check_is_connected"]())
        )
        original_native(**kwargs)

    monkeypatch.setattr(runtime, "run_asv3_loop", native)
    try:
        run.start()
        set_fence(run.setup.chat_session.id, run.cache, True)
        assert run.persistence_entered.wait(5)
        assert run.persisted_connections == [False]
        reset_cancel_status(run.setup.chat_session.id, run.cache)
        assert TurnCancellationProbe(run.setup.chat_session.id, run.cache)()
        with pytest.raises(RunStopped, match="cancelled"):
            contexts[0].check_active()
        run.persistence_release.set()
        assert run.reader is not None
        stop = next(run.reader)
        assert isinstance(stop, Packet) and isinstance(stop.obj, OverallStop)
        assert stop.obj.stop_reason == "user_cancelled"
        assert list(run.reader) == []
    finally:
        run.cleanup()
