"""Keep detached chat writers alive until their persistence finishes on shutdown."""

import contextvars
import threading
import time
from collections.abc import Callable

_writers: set[threading.Thread] = set()
_writers_changed = threading.Condition()
_SHUTDOWN_DRAIN_SECONDS = 14 * 60


def start_chat_stream_writer(target: Callable[[], None]) -> None:
    context = contextvars.copy_context()

    def run() -> None:
        try:
            context.run(target)
        finally:
            with _writers_changed:
                _writers.discard(threading.current_thread())
                _writers_changed.notify_all()

    writer = threading.Thread(target=run, name="chat-stream-writer")
    # Register before starting so shutdown cannot miss a scheduled writer.
    with _writers_changed:
        _writers.add(writer)
        try:
            writer.start()
        except BaseException:
            _writers.discard(writer)
            _writers_changed.notify_all()
            raise


def wait_for_chat_stream_writers(timeout: float = _SHUTDOWN_DRAIN_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    with _writers_changed:
        while _writers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _writers_changed.wait(remaining)
    return True
