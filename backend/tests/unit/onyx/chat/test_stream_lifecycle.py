import contextvars
import threading

import pytest

from onyx.chat.stream_lifecycle import (
    start_chat_stream_writer,
    wait_for_chat_stream_writers,
)


def test_shutdown_waits_for_every_detached_writer_and_preserves_context() -> None:
    tenant = contextvars.ContextVar[str]("stream_lifecycle_test_tenant")
    token = tenant.set("tenant-one")
    first_finished = threading.Event()
    second_finished = threading.Event()
    drain_finished = threading.Event()
    observed: list[str] = []

    def writer(finish: threading.Event) -> None:
        assert finish.wait(5)
        observed.append(tenant.get())

    try:
        start_chat_stream_writer(lambda: writer(first_finished))
        start_chat_stream_writer(lambda: writer(second_finished))
        assert not wait_for_chat_stream_writers(timeout=0)

        def drain() -> None:
            assert wait_for_chat_stream_writers(timeout=5)
            drain_finished.set()

        shutdown = threading.Thread(target=drain)
        shutdown.start()
        first_finished.set()
        assert not drain_finished.wait(0.05)
        second_finished.set()
        assert drain_finished.wait(5)
        shutdown.join(timeout=5)
        assert observed == ["tenant-one", "tenant-one"]
    finally:
        first_finished.set()
        second_finished.set()
        assert wait_for_chat_stream_writers(timeout=5)
        tenant.reset(token)


def test_writer_start_failure_does_not_leave_shutdown_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(_self: threading.Thread) -> None:
        raise RuntimeError("thread could not start")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread could not start"):
        start_chat_stream_writer(lambda: None)
    assert wait_for_chat_stream_writers(timeout=0)
