"""Exercise detached Cropped production with controlled native and persistence gates."""

import ast
import json
import socket
import threading
import time
from collections.abc import Callable, Generator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from onyx.cache.interface import CacheBackend
from onyx.chat import process_message, stream_lifecycle
from onyx.chat.chat_processing_checker import (
    is_chat_session_processing,
    set_processing_status,
)
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.chat.stop_signal_checker import FENCE_PREFIX, is_connected, set_fence
from onyx.chat.stream_buffer import (
    StreamBufferWriter,
    StreamChunkRead,
    read_stream_chunks,
)
from onyx.chat.stream_lifecycle import wait_for_chat_stream_writers
from onyx.llm.interfaces import LLMConfig
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    OverallStop,
    Packet,
    ReasoningStart,
)
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


class _CroppedRun:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.native_release = threading.Event()
        self.native_started = threading.Event()
        self.native_exited = threading.Event()
        self.persistence_release = threading.Event()
        self.persistence_entered = threading.Event()
        self.buffer_available = threading.Event()
        self.buffer_done = threading.Event()
        self.stop_probe_failed = threading.Event()
        self.stop_probe_recovered = threading.Event()
        self.writer_failure_requested = threading.Event()
        self.native_error: Exception | None = None
        self.persistence_error: Exception | None = None
        self.stop_probe_error: Exception | None = None
        self.persisted_connections: list[bool] = []
        self.native_thread: threading.Thread | None = None
        self.reader: Generator[Packet | StreamingError, None, None] | None = None
        self.values: dict[str, bytes] = {}
        self.external_attempts: list[str] = []
        self.cache = MagicMock(spec=CacheBackend)
        self.cache.set.side_effect = self._store
        self.cache.get.side_effect = self.values.get
        self.cache.exists.side_effect = self._exists
        self.cache.delete.side_effect = lambda key: self.values.pop(key, None)
        self.setup = _make_setup()
        self.setup.persona.id = process_message.DEFAULT_PERSONA_ID
        self.setup.cache = self.cache
        self.setup.processing_run_id = 1
        self.setup.new_msg_req = SendMessageRequest(
            message="offline lifecycle fixture",
            atez_search_v3=True,
            asv3_exp_cropped=True,
            asv3_research_profile="normal",
            asv3_parallel_research=False,
        )
        self.setup.user_message.id = 1
        self.setup.reserved_messages[0].id = 2
        self.setup.llms[0].config = LLMConfig(
            model_provider="openai",
            model_name="offline-fixture",
            max_input_tokens=200000,
            temperature=0,
        )
        self.setup.check_is_connected = lambda: is_connected(
            self.setup.chat_session.id, self.cache
        )
        self.buffer = StreamBufferWriter(self.cache, self.setup.chat_session.id, 1)
        actual_flush = self.buffer.flush

        def flush() -> None:
            actual_flush()
            if self.writer_failure_requested.is_set():
                self.writer_failure_requested.clear()
                raise RuntimeError("controlled writer failure")

        monkeypatch.setattr(self.buffer, "flush", flush)
        monkeypatch.setattr("onyx.asv3.runtime.run_asv3_loop", self._native)
        monkeypatch.setattr(process_message, "construct_tools", lambda **_: {})
        monkeypatch.setattr(process_message, "get_llm_token_counter", lambda _: len)
        monkeypatch.setattr(
            process_message,
            "load_settings",
            lambda: SimpleNamespace(auto_detect_search_filters=False),
        )
        self.completion = MagicMock(side_effect=self._persist)
        self.final_recording = MagicMock()
        monkeypatch.setattr(
            process_message, "llm_loop_completion_handle", self.completion
        )
        monkeypatch.setattr(
            process_message, "record_final_answer_message", self.final_recording
        )
        self.db_session = MagicMock()
        self.db_session.commit.side_effect = self._persist
        monkeypatch.setattr(
            process_message,
            "get_session_with_current_tenant",
            lambda: nullcontext(self.db_session),
        )

        def external_call(*_args: Any, **_kwargs: Any) -> None:
            self.external_attempts.append("forbidden external call")
            raise AssertionError(
                "Lifecycle tests must not use network, DB or providers"
            )

        monkeypatch.setattr(socket.socket, "connect", external_call)
        monkeypatch.setattr(socket.socket, "connect_ex", external_call)
        monkeypatch.setattr(socket, "create_connection", external_call)
        monkeypatch.setattr(process_message, "get_llm_for_persona", external_call)
        self.setup.llms[0].invoke.side_effect = external_call

    def _store(self, key: str, value: str | bytes | int | float, **_: Any) -> None:
        self.values[key] = value if isinstance(value, bytes) else str(value).encode()
        if key.endswith(":meta"):
            self.buffer_available.set()
            if json.loads(self.values[key])["done"]:
                self.buffer_done.set()

    def _exists(self, key: str) -> bool:
        if key.startswith(FENCE_PREFIX):
            if self.stop_probe_error is not None:
                error, self.stop_probe_error = self.stop_probe_error, None
                self.stop_probe_failed.set()
                raise error
            if self.stop_probe_failed.is_set():
                self.stop_probe_recovered.set()
        return key in self.values

    def _native(self, **kwargs: Any) -> None:
        assert kwargs["workflow_variant"] == "asv3_exp_cropped"
        self.native_thread = threading.current_thread()
        emitter = kwargs["emitter"]
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=ReasoningStart()))
        self.native_started.set()
        try:
            assert self.native_release.wait(15)
            if self.native_error is not None:
                raise self.native_error
            emitter.emit(
                Packet(
                    placement=Placement(turn_index=0),
                    obj=OverallStop(stop_reason="complete"),
                )
            )
        finally:
            self.native_exited.set()

    def _persist(self, *_args: Any, **kwargs: Any) -> None:
        if "is_connected" in kwargs:
            self.persisted_connections.append(
                cast(Callable[[], bool], kwargs["is_connected"])()
            )
        self.persistence_entered.set()
        assert self.persistence_release.wait(15)
        if self.persistence_error is not None:
            raise self.persistence_error

    def start(self) -> None:
        set_processing_status(self.setup.chat_session.id, self.cache, True, run_id=1)
        self.reader = cast(
            Generator[Packet | StreamingError, None, None],
            process_message._run_models(
                cast(ChatTurnSetup, self.setup), MagicMock(), stream_buffer=self.buffer
            ),
        )
        assert self.native_started.wait(5)
        assert isinstance(next(self.reader), Packet)

    def detach_both_readers(self) -> None:
        assert self.reader is not None
        self.reader.close()
        assert self.buffer_available.wait(5)
        # Extract only the route's reader so authorization/session DB calls are not run.
        path = (
            Path(process_message.__file__).parents[1]
            / "server/query_and_chat/chat_backend.py"
        )
        route = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "resume_chat_stream"
        )
        reader = next(
            node
            for node in route.body
            if isinstance(node, ast.FunctionDef) and node.name == "stream_buffered_run"
        )
        namespace: dict[str, Any] = {
            "Generator": Generator,
            "cursor": 0,
            "time": time,
            "cache": self.cache,
            "session_id": self.setup.chat_session.id,
            "run_id": 1,
            "read_stream_chunks": read_stream_chunks,
            "_RESUME_MAX_CHUNKS_PER_READ": 32,
            "is_chat_session_processing": is_chat_session_processing,
        }
        exec(
            compile(ast.Module(body=[reader], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        resumed = cast(Generator[str, None, None], namespace["stream_buffered_run"]())
        assert "reasoning_start" in next(resumed)
        resumed.close()

    def snapshot(self) -> StreamChunkRead:
        read = read_stream_chunks(self.cache, self.setup.chat_session.id, 1, 0)
        assert read is not None and not read.gap
        return read

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for block in self.snapshot().blocks
            for line in block.splitlines()
        ]

    def finish(self) -> None:
        self.native_release.set()
        assert self.persistence_entered.wait(5)
        assert not wait_for_chat_stream_writers(timeout=0.05)
        self.persistence_release.set()
        assert wait_for_chat_stream_writers(timeout=5)
        assert self.buffer_done.wait(5)
        assert not is_chat_session_processing(self.setup.chat_session.id, self.cache)

    def cleanup(self) -> None:
        self.native_release.set()
        self.persistence_release.set()
        if self.reader is not None:
            self.reader.close()
        assert wait_for_chat_stream_writers(timeout=5)
        if self.native_thread is not None:
            self.native_thread.join(5)
            assert not self.native_thread.is_alive()
        assert not self.external_attempts


@pytest.fixture
def cropped_run(monkeypatch: pytest.MonkeyPatch) -> Generator[_CroppedRun, None, None]:
    run = _CroppedRun(monkeypatch)
    try:
        yield run
    finally:
        run.cleanup()


def test_post_and_resume_disconnect_preserve_native_and_persistence(
    cropped_run: _CroppedRun,
) -> None:
    cropped_run.start()
    cropped_run.detach_both_readers()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    assert not cropped_run.snapshot().done
    cropped_run.finish()
    assert cropped_run.persisted_connections == [True]
    assert cropped_run.final_recording.call_count == 1
    assert [
        r["obj"]["stop_reason"]
        for r in cropped_run.records()
        if r.get("obj", {}).get("type") == "stop"
    ] == ["complete"]
    assert not any(r.get("error_code") for r in cropped_run.records())


def test_native_work_waits_for_writer_registration(
    cropped_run: _CroppedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_entered = threading.Event()
    registration_release = threading.Event()
    actual_start = stream_lifecycle.start_chat_stream_writer
    start_errors: list[BaseException] = []

    def held_registration(target: Callable[[], None]) -> None:
        registration_entered.set()
        assert registration_release.wait(5)
        actual_start(target)

    def start() -> None:
        try:
            cropped_run.start()
        except BaseException as error:
            start_errors.append(error)

    monkeypatch.setattr(stream_lifecycle, "start_chat_stream_writer", held_registration)
    starter = threading.Thread(target=start)
    starter.start()
    try:
        assert registration_entered.wait(5)
        assert not cropped_run.native_started.is_set()
    finally:
        registration_release.set()
        starter.join(5)
    assert not starter.is_alive() and not start_errors
    cropped_run.finish()


def test_worker_submission_failure_releases_registration(
    cropped_run: _CroppedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_submit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("controlled worker submission failure")

    monkeypatch.setattr(process_message.ThreadPoolExecutor, "submit", fail_submit)
    set_processing_status(
        cropped_run.setup.chat_session.id, cropped_run.cache, True, run_id=1
    )
    result = list(
        process_message._run_models(
            cast(ChatTurnSetup, cropped_run.setup),
            MagicMock(),
            stream_buffer=cropped_run.buffer,
        )
    )
    assert any(
        isinstance(item, StreamingError) and item.error_code == "STREAM_WRITER_ERROR"
        for item in result
    )
    assert wait_for_chat_stream_writers(timeout=5)
    assert cropped_run.buffer_done.wait(5)
    assert not cropped_run.native_started.is_set()
    assert not is_chat_session_processing(
        cropped_run.setup.chat_session.id, cropped_run.cache
    )
    cropped_run.final_recording.assert_not_called()


@pytest.mark.parametrize("error_type", [RedisError, SQLAlchemyError])
def test_transient_stop_lookup_error_keeps_run_alive(
    cropped_run: _CroppedRun, error_type: type[Exception]
) -> None:
    cropped_run.start()
    cropped_run.detach_both_readers()
    cropped_run.stop_probe_error = error_type("controlled stop lookup failure")
    assert cropped_run.stop_probe_failed.wait(5)
    assert cropped_run.stop_probe_recovered.wait(5)
    assert not wait_for_chat_stream_writers(timeout=0.05)
    assert not cropped_run.snapshot().done
    assert is_chat_session_processing(
        cropped_run.setup.chat_session.id, cropped_run.cache
    )
    cropped_run.finish()
    assert cropped_run.persisted_connections == [True]
    assert not any(r.get("error_code") for r in cropped_run.records())
    assert not any(
        r.get("obj", {}).get("stop_reason") == "user_cancelled"
        for r in cropped_run.records()
    )


def test_writer_error_does_not_release_native_producer(
    cropped_run: _CroppedRun,
) -> None:
    cropped_run.start()
    cropped_run.writer_failure_requested.set()
    assert cropped_run.reader is not None
    error = next(cropped_run.reader)
    assert (
        isinstance(error, StreamingError) and error.error_code == "STREAM_WRITER_ERROR"
    )
    cropped_run.reader.close()
    assert not cropped_run.native_exited.is_set()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    cropped_run.finish()
    assert cropped_run.persisted_connections == [True]
    assert [r["error_code"] for r in cropped_run.records() if r.get("error_code")] == [
        "STREAM_WRITER_ERROR"
    ]
    assert not any(
        r.get("obj", {}).get("stop_reason") == "user_cancelled"
        for r in cropped_run.records()
    )


def test_explicit_cancel_keeps_native_registered_and_persists_partial_once(
    cropped_run: _CroppedRun,
) -> None:
    cropped_run.start()
    set_fence(cropped_run.setup.chat_session.id, cropped_run.cache, True)
    assert cropped_run.persistence_entered.wait(5)
    assert cropped_run.persisted_connections == [False]
    assert not wait_for_chat_stream_writers(timeout=0.05)
    cropped_run.persistence_release.set()
    assert cropped_run.reader is not None
    stop = next(cropped_run.reader)
    assert isinstance(stop, Packet) and isinstance(stop.obj, OverallStop)
    assert stop.obj.stop_reason == "user_cancelled"
    assert list(cropped_run.reader) == []
    assert not is_chat_session_processing(
        cropped_run.setup.chat_session.id, cropped_run.cache
    )
    assert not cropped_run.native_exited.is_set()
    assert not wait_for_chat_stream_writers(timeout=0.05)
    cropped_run.native_release.set()
    assert wait_for_chat_stream_writers(timeout=5)
    assert cropped_run.buffer_done.wait(5)
    assert cropped_run.completion.call_count == 1
    assert cropped_run.final_recording.call_count == 1
    assert [
        r["obj"]["stop_reason"]
        for r in cropped_run.records()
        if r.get("obj", {}).get("type") == "stop"
    ] == ["user_cancelled"]


@pytest.mark.parametrize("native_error", [False, True])
def test_native_or_persistence_failure_settles_registration(
    cropped_run: _CroppedRun, native_error: bool
) -> None:
    if native_error:
        cropped_run.native_error = RuntimeError("controlled native failure")
    else:
        cropped_run.persistence_error = RuntimeError("controlled persistence failure")
    cropped_run.start()
    cropped_run.detach_both_readers()
    cropped_run.finish()
    assert cropped_run.snapshot().done
    if native_error:
        assert cropped_run.completion.call_count == 0
        assert cropped_run.db_session.commit.call_count == 1
        assert cropped_run.final_recording.call_count == 1
        errors = [r for r in cropped_run.records() if r.get("error_code")]
        assert len(errors) == 1
        assert errors[0]["error"] == "controlled native failure"
        assert errors[0]["error_code"] != "STREAM_WRITER_ERROR"
        assert not any(
            r.get("obj", {}).get("stop_reason") == "complete"
            for r in cropped_run.records()
        )
    else:
        assert cropped_run.completion.call_count == 1
        cropped_run.final_recording.assert_not_called()
    assert not any(
        r.get("obj", {}).get("stop_reason") == "user_cancelled"
        for r in cropped_run.records()
    )
