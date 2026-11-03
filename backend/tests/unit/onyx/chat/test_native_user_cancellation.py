"""Native RunStopped is user cancellation only with the shared explicit-stop latch."""

import queue
import threading
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from onyx.asv3 import runtime
from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.stop_signal_checker import (
    TurnCancellationProbe,
    reset_cancel_status,
    set_fence,
)
from onyx.chat.stream_lifecycle import wait_for_chat_stream_writers
from onyx.regulatory.publication_reads import PublicationReadChanged
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import Packet, ReasoningStart
from tests.unit.onyx.chat.test_stream_persistence_truthfulness import _SavingRun


class _NativeRun:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ordered: bool = True,
        n_models: int = 1,
        cause: str = "explicit",
    ) -> None:
        self.saving = _SavingRun(monkeypatch, n_models=n_models)
        self.connected = TurnCancellationProbe(
            self.saving.run.setup.chat_session.id, self.saving.run.cache
        )
        self.saving.run.setup.check_is_connected = self.connected
        for release in self.saving.release:
            release.set()
        self.writer_release = threading.Event()
        self.workers_finished = threading.Event()
        self.natives_started = threading.Event()
        self.lock = threading.Lock()
        self.started_count = 0
        self.finished_count = 0
        self.queue_count = 0
        self.starter: threading.Thread | None = None
        self.start_errors: list[BaseException] = []
        owner = self
        actual_submit = ThreadPoolExecutor.submit

        def unexpected_database(*_args: Any, **_kwargs: Any) -> None:
            self.saving.run.external_attempts.append("unexpected DB session")
            raise AssertionError("Native cancellation tests must not open a DB session")

        monkeypatch.setattr(
            "onyx.db.engine.sql_engine.get_session_with_current_tenant",
            unexpected_database,
        )

        def finished(_future: Future[Any]) -> None:
            with self.lock:
                self.finished_count += 1
                if self.finished_count == n_models:
                    self.workers_finished.set()

        def submit(
            executor: ThreadPoolExecutor,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> Future[Any]:
            future = actual_submit(executor, function, *args, **kwargs)
            future.add_done_callback(finished)
            return future

        class WorkerGatedQueue(queue.Queue[Any]):
            def __init__(self) -> None:
                super().__init__()
                self.is_merged = owner.queue_count == 0
                owner.queue_count += 1

            def get(self, block: bool = True, timeout: float | None = None) -> Any:
                if self.is_merged and ordered:
                    assert owner.workers_finished.wait(15)
                    assert owner.writer_release.wait(15)
                return super().get(block=block, timeout=timeout)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
        monkeypatch.setattr(queue, "Queue", WorkerGatedQueue)

        def native(**kwargs: Any) -> None:
            assert kwargs["check_is_connected"] is self.connected
            index = kwargs["assistant_message_id"] - 2
            state = cast(ChatStateContainer, kwargs["state_container"])
            state.set_answer_tokens(f"Synthetic native partial answer {index}")
            self.saving.states[index] = state
            self.saving.run.native_thread = threading.current_thread()
            kwargs["emitter"].emit(
                Packet(placement=Placement(turn_index=0), obj=ReasoningStart())
            )
            self.saving.run.native_started.set()
            with self.lock:
                self.started_count += 1
                if self.started_count == n_models:
                    self.natives_started.set()
            assert self.saving.run.native_release.wait(15)
            context = RunContext(
                cancelled=lambda: not self.connected(),
                timeout_seconds=0 if cause == "deadline" else float("inf"),
                budget=SharedBudget(max_tools=0 if cause == "budget" else 64),
            )
            try:
                context.check_active()
                if cause == "budget":
                    context.budget.consume("tools")
                if cause == "source":
                    raise RunStopped("Source access interrupted")
                raise AssertionError("Native RunContext must stop")
            finally:
                self.saving.run.native_exited.set()

        monkeypatch.setattr(runtime, "run_asv3_loop", native)

    def native_first(self, *, explicit: bool = True, reset: bool = False) -> None:
        def start() -> None:
            try:
                self.saving.run.start()
            except BaseException as error:
                self.start_errors.append(error)

        self.starter = threading.Thread(target=start)
        self.starter.start()
        assert self.natives_started.wait(5)
        if explicit:
            set_fence(
                self.saving.run.setup.chat_session.id, self.saving.run.cache, True
            )
        self.saving.run.native_release.set()
        assert self.workers_finished.wait(5)
        if reset:
            reset_cancel_status(
                self.saving.run.setup.chat_session.id, self.saving.run.cache
            )
            assert TurnCancellationProbe(
                self.saving.run.setup.chat_session.id, self.saving.run.cache
            )()
            assert not self.connected()
        self.writer_release.set()
        self.starter.join(5)
        assert not self.starter.is_alive() and not self.start_errors
        assert wait_for_chat_stream_writers(timeout=5)
        assert self.saving.run.buffer_done.wait(5)

    def cleanup(self) -> None:
        self.writer_release.set()
        self.saving.run.native_release.set()
        for release in self.saving.release:
            release.set()
        if self.starter is not None:
            self.starter.join(5)
        self.saving.cleanup()


@pytest.fixture
def make_run(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Callable[..., _NativeRun], None, None]:
    runs: list[_NativeRun] = []

    def make(**kwargs: Any) -> _NativeRun:
        run = _NativeRun(monkeypatch, **kwargs)
        runs.append(run)
        return run

    try:
        yield make
    finally:
        for run in reversed(runs):
            run.cleanup()


@pytest.mark.parametrize("reset", [False, True])
def test_native_first_explicit_cancel_saves_partial_without_native_error(
    make_run: Callable[..., _NativeRun],
    reset: bool,
) -> None:
    run = make_run()
    run.native_first(reset=reset)
    assert [row["obj"]["stop_reason"] for row in run.saving.stops()] == [
        "user_cancelled"
    ]
    assert not run.saving.errors_in_buffer()
    assert run.saving.save.call_count == 1
    assert "stopped by the user" in run.saving.save.call_args.kwargs["message_text"]
    assert run.saving.saved == [True]
    run.saving.run.final_recording.assert_called_once_with(2)


def test_writer_first_cancel_remains_prompt_and_old_latch_survives_reset(
    make_run: Callable[..., _NativeRun],
) -> None:
    run = make_run(ordered=False)
    run.saving.run.start()
    set_fence(run.saving.run.setup.chat_session.id, run.saving.run.cache, True)
    assert run.saving.run.buffer_done.wait(5)
    assert not run.saving.run.native_exited.is_set()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    reset_cancel_status(run.saving.run.setup.chat_session.id, run.saving.run.cache)
    assert TurnCancellationProbe(
        run.saving.run.setup.chat_session.id, run.saving.run.cache
    )()
    assert not run.connected()
    run.saving.run.native_release.set()
    assert wait_for_chat_stream_writers(timeout=5)
    assert not run.saving.errors_in_buffer()
    assert [row["obj"]["stop_reason"] for row in run.saving.stops()] == [
        "user_cancelled"
    ]
    assert run.saving.save.call_count == 1
    assert "stopped by the user" in run.saving.save.call_args.kwargs["message_text"]


@pytest.mark.parametrize("cause", ["deadline", "budget", "source"])
def test_connected_run_stopped_is_a_real_error(
    make_run: Callable[..., _NativeRun],
    cause: str,
) -> None:
    run = make_run(cause=cause)
    run.native_first(explicit=False)
    assert run.connected()
    assert not run.saving.stops()
    errors = run.saving.errors_in_buffer()
    assert len(errors) == 1
    assert errors[0]["error_code"] not in {"PERSISTENCE_ERROR", "STREAM_WRITER_ERROR"}
    run.saving.save.assert_not_called()
    assert run.saving.run.db_session.commit.call_count == 1
    run.saving.run.final_recording.assert_called_once_with(2)


@pytest.mark.parametrize("error_type", [RedisError, SQLAlchemyError])
def test_transient_probe_then_connected_does_not_reclassify_source_stop(
    make_run: Callable[..., _NativeRun],
    error_type: type[Exception],
) -> None:
    run = make_run(cause="source")
    run.saving.run.stop_probe_error = error_type("controlled transient lookup")
    run.native_first(explicit=False)
    assert run.saving.run.stop_probe_failed.is_set()
    assert run.saving.run.stop_probe_recovered.is_set()
    assert run.connected()
    assert not run.saving.stops()
    assert len(run.saving.errors_in_buffer()) == 1
    run.saving.save.assert_not_called()


def test_multiple_native_cancel_markers_emit_one_turn_cancel(
    make_run: Callable[..., _NativeRun],
) -> None:
    run = make_run(n_models=2)
    run.native_first()
    assert run.finished_count == 2
    assert not run.saving.errors_in_buffer()
    assert [row["obj"]["stop_reason"] for row in run.saving.stops()] == [
        "user_cancelled"
    ]
    assert run.saving.save.call_count == 2
    assert all(
        "stopped by the user" in call.kwargs["message_text"]
        for call in run.saving.save.call_args_list
    )
    assert run.saving.saved == [True, True]


def test_native_cancel_partial_save_failure_remains_explicit_error(
    make_run: Callable[..., _NativeRun],
) -> None:
    run = make_run()
    run.saving.errors[0] = RuntimeError("SENSITIVE_PARTIAL_SAVE_DETAIL")
    run.native_first()
    assert [row["error_code"] for row in run.saving.errors_in_buffer()] == [
        "PERSISTENCE_ERROR"
    ]
    assert [row["obj"]["stop_reason"] for row in run.saving.stops()] == [
        "user_cancelled"
    ]
    assert run.saving.save.call_count == 1
    assert run.saving.saved == [False]
    assert "SENSITIVE_PARTIAL_SAVE_DETAIL" not in str(run.saving.run.records())
    run.saving.run.final_recording.assert_not_called()


def test_native_cancel_does_not_bypass_publication_invalidation(
    make_run: Callable[..., _NativeRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = make_run(ordered=False)
    run.saving.run.start()
    run.saving.run.detach_both_readers()

    def changed() -> None:
        raise PublicationReadChanged()

    invalidation = MagicMock()
    monkeypatch.setattr(
        "onyx.db.chat.invalidate_publication_chat_message", invalidation
    )
    monkeypatch.setattr(run.saving.states[0].publication_reads, "validate", changed)
    set_fence(run.saving.run.setup.chat_session.id, run.saving.run.cache, True)
    run.saving.run.native_release.set()
    assert wait_for_chat_stream_writers(timeout=5)
    run.saving.save.assert_not_called()
    assert [row["error_code"] for row in run.saving.errors_in_buffer()] == [
        "PUBLICATION_SOURCE_CHANGED"
    ]
    assert not any(
        row["obj"]["stop_reason"] == "complete" for row in run.saving.stops()
    )
    assert invalidation.call_count == 1
