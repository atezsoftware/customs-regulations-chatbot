"""Keep an owned metadata lease live while provider or storage calls are pending."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import copy_context
from math import isfinite
from threading import TIMEOUT_MAX, Event, Lock, Thread


@contextmanager
def maintain_metadata_lease(
    heartbeat: Callable[[], None], *, interval_seconds: float = 30.0
) -> Iterator[Callable[[], None]]:
    if not isfinite(interval_seconds) or not 0 < interval_seconds <= TIMEOUT_MAX:
        raise ValueError("Metadata heartbeat interval must be finite and positive")
    stopped = Event()
    failure_lock = Lock()
    heartbeat_lock = Lock()
    failure: Exception | None = None

    def check_owner() -> None:
        with failure_lock:
            error = failure
        if error is not None:
            raise error

    def guarded_heartbeat() -> None:
        nonlocal failure
        with heartbeat_lock:
            check_owner()
            if stopped.is_set():
                raise RuntimeError("Metadata lease keepalive is closed")
            try:
                heartbeat()
            except Exception as error:
                with failure_lock:
                    failure = error
                stopped.set()
                raise
            check_owner()

    def renew() -> None:
        while not stopped.wait(interval_seconds):
            try:
                guarded_heartbeat()
            except Exception:
                return

    context = copy_context()
    thread = Thread(
        target=lambda: context.run(renew),
        name="intake-metadata-lease",
        daemon=True,
    )
    thread.start()
    completed = False
    try:
        yield guarded_heartbeat
        completed = True
    finally:
        stopped.set()
        thread.join()
        # Preserve the primary extraction error if both operations failed.
        if completed:
            check_owner()
