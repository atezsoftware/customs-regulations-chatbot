"""Raw chat terminals reflect real completion handling and critical save outcomes."""

import json
import threading
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import runtime
from onyx.chat import process_message
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.stop_signal_checker import set_fence
from onyx.chat.stream_lifecycle import wait_for_chat_stream_writers
from onyx.regulatory.publication_reads import PublicationReadChanged
from tests.unit.onyx.chat.test_cropped_stream_writer_lifecycle import _CroppedRun

_REAL_COMPLETION = process_message.llm_loop_completion_handle


class _SavingRun:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, n_models: int = 1) -> None:
        self.run = _CroppedRun(monkeypatch)
        self.entered = [threading.Event() for _ in range(n_models)]
        self.release = [threading.Event() for _ in range(n_models)]
        self.saved = [False] * n_models
        self.errors: dict[int, Exception] = {}
        self.states: dict[int, ChatStateContainer] = {}
        self.flush_while_saving = threading.Event()
        self.messages = {
            i + 2: SimpleNamespace(
                id=i + 2, chat_session_id=self.run.setup.chat_session.id
            )
            for i in range(n_models)
        }
        for i in range(1, n_models):
            model = MagicMock()
            model.config = self.run.setup.llms[0].config
            self.run.setup.llms.append(model)
            self.run.setup.model_display_names.append(f"model-{i}")
            self.run.setup.reserved_messages.append(MagicMock(id=i + 2))
        for message in self.run.setup.reserved_messages:
            message.chat_session_id = self.run.setup.chat_session.id
        self.run.db_session.get.side_effect = lambda _kind, identifier: (
            self.messages.get(identifier)
        )
        self.run.db_session.commit.side_effect = None
        self.save = MagicMock(side_effect=self._save)
        monkeypatch.setattr(
            process_message, "llm_loop_completion_handle", _REAL_COMPLETION
        )
        monkeypatch.setattr(process_message, "save_chat_turn", self.save)
        monkeypatch.setattr(
            process_message, "create_chat_history_chain", lambda **_: []
        )
        monkeypatch.setattr(
            process_message, "calculate_total_history_tokens", lambda _: 0
        )
        monkeypatch.setattr(
            process_message,
            "get_compression_params",
            lambda **_: SimpleNamespace(should_compress=False),
        )
        native = self.run._native

        def answer(**kwargs: Any) -> None:
            index = kwargs["assistant_message_id"] - 2
            state = cast(ChatStateContainer, kwargs["state_container"])
            state.set_answer_tokens(f"Synthetic answer for model {index}")
            self.states[index] = state
            native(**kwargs)

        monkeypatch.setattr(runtime, "run_asv3_loop", answer)
        actual_flush = self.run.buffer.flush

        def flush() -> None:
            actual_flush()
            if self.run.native_exited.is_set() and any(
                entered.is_set() and not released.is_set()
                for entered, released in zip(self.entered, self.release, strict=True)
            ):
                self.flush_while_saving.set()

        monkeypatch.setattr(self.run.buffer, "flush", flush)

    def _save(self, **kwargs: Any) -> None:
        index = kwargs["assistant_message"].id - 2
        self.entered[index].set()
        assert self.release[index].wait(15)
        if index in self.errors:
            raise self.errors[index]
        kwargs["assistant_message"].message = kwargs["message_text"]
        kwargs["db_session"].commit()
        self.saved[index] = True

    def start_saving(self) -> None:
        self.run.start()
        self.run.detach_both_readers()
        self.run.native_release.set()
        assert all(event.wait(5) for event in self.entered)
        assert self.flush_while_saving.wait(5)

    def finish(self) -> None:
        self.run.native_release.set()
        for event in self.release:
            event.set()
        assert wait_for_chat_stream_writers(timeout=5)
        assert self.run.buffer_done.wait(5)

    def stops(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.run.records()
            if row.get("obj", {}).get("type") == "stop"
        ]

    def errors_in_buffer(self) -> list[dict[str, Any]]:
        return [row for row in self.run.records() if row.get("error_code")]

    def cleanup(self) -> None:
        for event in self.release:
            event.set()
        self.run.cleanup()


@pytest.fixture
def saving_run(monkeypatch: pytest.MonkeyPatch) -> Generator[_SavingRun, None, None]:
    driver = _SavingRun(monkeypatch)
    try:
        yield driver
    finally:
        driver.cleanup()


def test_complete_stop_waits_for_actual_critical_save(saving_run: _SavingRun) -> None:
    saving_run.start_saving()
    assert not saving_run.stops()
    assert not saving_run.run.snapshot().done
    assert not saving_run.saved[0]
    saving_run.run.final_recording.assert_not_called()
    saving_run.finish()
    assert saving_run.saved == [True]
    assert len(saving_run.stops()) == 1
    assert saving_run.stops()[0]["obj"]["stop_reason"] == "complete"
    assert not saving_run.errors_in_buffer()
    assert saving_run.save.call_count == 1
    assert saving_run.run.final_recording.call_count == 1


def test_critical_save_failure_is_one_safe_error_without_complete_or_retry(
    saving_run: _SavingRun,
) -> None:
    saving_run.errors[0] = RuntimeError("SENSITIVE_DATABASE_DETAIL")
    saving_run.start_saving()
    saving_run.finish()
    assert saving_run.saved == [False]
    assert not saving_run.stops()
    errors = saving_run.errors_in_buffer()
    assert [error["error_code"] for error in errors] == ["PERSISTENCE_ERROR"]
    assert errors[0]["details"]["model_index"] == 0
    assert "SENSITIVE_DATABASE_DETAIL" not in str(saving_run.run.records())
    assert saving_run.save.call_count == 1
    saving_run.run.final_recording.assert_not_called()


def test_graph_recording_failure_does_not_reclassify_a_saved_answer(
    saving_run: _SavingRun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    saving_run.run.final_recording.side_effect = RuntimeError("SENSITIVE_GRAPH_DETAIL")
    saving_run.start_saving()
    saving_run.finish()
    assert saving_run.saved == [True]
    assert len(saving_run.stops()) == 1
    assert saving_run.stops()[0]["obj"]["stop_reason"] == "complete"
    assert not saving_run.errors_in_buffer()
    assert saving_run.save.call_count == 1
    assert saving_run.run.final_recording.call_count == 1
    assert "SENSITIVE_GRAPH_DETAIL" not in str(saving_run.run.records())
    assert any(record.levelno >= 30 for record in caplog.records)


def test_cancelled_partial_save_failure_is_truthful_before_native_returns(
    saving_run: _SavingRun,
) -> None:
    saving_run.errors[0] = RuntimeError("SENSITIVE_PARTIAL_SAVE_DETAIL")
    saving_run.run.start()
    saving_run.run.detach_both_readers()
    set_fence(saving_run.run.setup.chat_session.id, saving_run.run.cache, True)
    assert saving_run.entered[0].wait(5)
    saving_run.release[0].set()
    assert saving_run.run.buffer_done.wait(5)
    assert not saving_run.run.native_exited.is_set()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    assert [error["error_code"] for error in saving_run.errors_in_buffer()] == [
        "PERSISTENCE_ERROR"
    ]
    assert [row["obj"]["stop_reason"] for row in saving_run.stops()] == [
        "user_cancelled"
    ]
    assert "stopped by the user" in saving_run.save.call_args.kwargs["message_text"]
    assert "SENSITIVE_PARTIAL_SAVE_DETAIL" not in str(saving_run.run.records())
    saving_run.finish()
    assert saving_run.save.call_count == 1
    assert saving_run.saved == [False]
    saving_run.run.final_recording.assert_not_called()


def test_cancel_during_existing_save_still_reports_its_later_failure(
    saving_run: _SavingRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    saving_run.errors[0] = RuntimeError("SENSITIVE_INFLIGHT_SAVE_DETAIL")
    cancelled = threading.Event()
    actual_append = saving_run.run.buffer.append_line

    def append(line: str, **kwargs: Any) -> None:
        actual_append(line, **kwargs)
        if json.loads(line).get("obj", {}).get("stop_reason") == "user_cancelled":
            cancelled.set()

    monkeypatch.setattr(saving_run.run.buffer, "append_line", append)
    saving_run.start_saving()
    set_fence(saving_run.run.setup.chat_session.id, saving_run.run.cache, True)
    assert cancelled.wait(5)
    assert not saving_run.release[0].is_set()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    saving_run.finish()
    assert saving_run.saved == [False]
    assert saving_run.save.call_count == 1
    assert [error["error_code"] for error in saving_run.errors_in_buffer()] == [
        "PERSISTENCE_ERROR"
    ]
    assert [row["obj"]["stop_reason"] for row in saving_run.stops()] == [
        "user_cancelled"
    ]
    saving_run.run.final_recording.assert_not_called()


def test_cancel_save_failure_before_finally_snapshot_is_not_lost(
    saving_run: _SavingRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    saving_run.errors[0] = RuntimeError("SENSITIVE_CANCEL_RACE_SAVE_DETAIL")
    futures: list[Future[Any]] = []
    actual_submit = ThreadPoolExecutor.submit
    actual_append = saving_run.run.buffer.append_line
    failed_before_append_return = threading.Event()

    def submit(
        executor: ThreadPoolExecutor,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future = actual_submit(executor, function, *args, **kwargs)
        futures.append(future)
        return future

    def append(line: str, **kwargs: Any) -> None:
        actual_append(line, **kwargs)
        if json.loads(line).get("obj", {}).get("stop_reason") == "user_cancelled":
            assert len(futures) == 1
            saving_run.release[0].set()
            futures[0].result(timeout=5)
            failed_before_append_return.set()

    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr(saving_run.run.buffer, "append_line", append)
    saving_run.start_saving()
    set_fence(saving_run.run.setup.chat_session.id, saving_run.run.cache, True)
    assert failed_before_append_return.wait(5)
    saving_run.finish()
    assert saving_run.saved == [False]
    assert saving_run.save.call_count == 1
    assert [row["error_code"] for row in saving_run.errors_in_buffer()] == [
        "PERSISTENCE_ERROR"
    ]
    assert [row["obj"]["stop_reason"] for row in saving_run.stops()] == [
        "user_cancelled"
    ]
    saving_run.run.final_recording.assert_not_called()


def test_models_keep_independent_saved_and_failed_terminals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _SavingRun(monkeypatch, n_models=2)
    driver.errors[0] = RuntimeError("SENSITIVE_MODEL_ZERO_SAVE_DETAIL")
    try:
        driver.start_saving()
        assert not driver.stops()
        driver.finish()
        assert driver.saved == [False, True]
        assert [
            (row["placement"]["model_index"], row["obj"]["stop_reason"])
            for row in driver.stops()
        ] == [(1, "complete")]
        errors = driver.errors_in_buffer()
        assert [
            (row["error_code"], row["details"]["model_index"]) for row in errors
        ] == [("PERSISTENCE_ERROR", 0)]
        assert driver.save.call_count == 2
        assert sorted(
            call.kwargs["assistant_message"].id for call in driver.save.call_args_list
        ) == [2, 3]
        driver.run.final_recording.assert_called_once_with(3)
    finally:
        driver.cleanup()


def test_publication_change_still_prevents_save_and_success(
    saving_run: _SavingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saving_run.run.start()
    saving_run.run.detach_both_readers()

    def changed() -> None:
        raise PublicationReadChanged()

    monkeypatch.setattr(saving_run.states[0].publication_reads, "validate", changed)
    invalidation = MagicMock()
    monkeypatch.setattr(
        "onyx.db.chat.invalidate_publication_chat_message", invalidation
    )
    saving_run.finish()
    saving_run.save.assert_not_called()
    assert not saving_run.stops()
    assert [error["error_code"] for error in saving_run.errors_in_buffer()] == [
        "PUBLICATION_SOURCE_CHANGED"
    ]
    assert invalidation.call_count == 1


def test_unhandled_worker_future_failure_is_not_silent_success(
    saving_run: _SavingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WorkerAborted(BaseException):
        pass

    native = saving_run.run._native

    def aborted(**kwargs: Any) -> None:
        native(**kwargs)
        raise WorkerAborted("SENSITIVE_WORKER_DETAIL")

    monkeypatch.setattr(runtime, "run_asv3_loop", aborted)
    saving_run.run.start()
    saving_run.run.detach_both_readers()
    saving_run.finish()
    assert not saving_run.stops()
    assert len(saving_run.errors_in_buffer()) == 1
    saving_run.save.assert_not_called()
    saving_run.run.final_recording.assert_not_called()
    assert "SENSITIVE_WORKER_DETAIL" not in str(saving_run.run.records())


def test_native_error_after_terminal_never_becomes_saved_success(
    saving_run: _SavingRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = saving_run.run._native

    def failed(**kwargs: Any) -> None:
        native(**kwargs)
        raise RuntimeError("controlled native failure after terminal")

    monkeypatch.setattr(runtime, "run_asv3_loop", failed)
    saving_run.run.start()
    saving_run.run.detach_both_readers()
    saving_run.finish()
    assert not saving_run.stops()
    errors = saving_run.errors_in_buffer()
    assert len(errors) == 1
    assert errors[0]["error"] == "controlled native failure after terminal"
    assert errors[0]["error_code"] not in {"PERSISTENCE_ERROR", "STREAM_WRITER_ERROR"}
    saving_run.save.assert_not_called()
    assert saving_run.run.db_session.commit.call_count == 1
    saving_run.run.final_recording.assert_called_once_with(2)


@pytest.mark.parametrize(
    "boundary",
    [
        "create_chat_history_chain",
        "calculate_total_history_tokens",
        "get_compression_params",
        "compress_chat_history",
    ],
)
def test_post_save_maintenance_failure_preserves_saved_success(
    saving_run: _SavingRun, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    if boundary == "compress_chat_history":
        monkeypatch.setattr(
            process_message,
            "get_compression_params",
            lambda **_: SimpleNamespace(should_compress=True),
        )
    maintenance = MagicMock(side_effect=RuntimeError("controlled maintenance failure"))
    monkeypatch.setattr(process_message, boundary, maintenance)
    saving_run.start_saving()
    saving_run.finish()
    assert saving_run.saved == [True]
    assert [row["obj"]["stop_reason"] for row in saving_run.stops()] == ["complete"]
    assert not saving_run.errors_in_buffer()
    assert saving_run.save.call_count == 1
    assert maintenance.call_count == 1
    saving_run.run.final_recording.assert_called_once_with(2)


def test_post_save_source_invalidation_remains_critical(
    saving_run: _SavingRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        process_message,
        "create_chat_history_chain",
        MagicMock(side_effect=PublicationReadChanged()),
    )
    invalidation = MagicMock()
    monkeypatch.setattr(
        "onyx.db.chat.invalidate_publication_chat_message", invalidation
    )
    saving_run.start_saving()
    saving_run.finish()
    assert saving_run.saved == [True]
    assert not saving_run.stops()
    assert [row["error_code"] for row in saving_run.errors_in_buffer()] == [
        "PUBLICATION_SOURCE_CHANGED"
    ]
    assert saving_run.save.call_count == 1
    assert invalidation.call_count == 1
