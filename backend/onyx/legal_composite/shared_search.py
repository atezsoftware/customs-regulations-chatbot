"""Request-local sharing for physically identical prepared source searches."""

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Lock

from onyx.context.search.models import InferenceChunk


class SharedPreparedRetrieval:
    def __init__(self, max_entries: int = 128) -> None:
        self._lock = Lock()
        self._max_entries = max_entries
        self._queries: dict[str, Future[list[InferenceChunk]]] = {}

    def run(
        self,
        key: str,
        execute: Callable[[], list[InferenceChunk]],
        check_active: Callable[[], None],
    ) -> tuple[list[InferenceChunk], bool]:
        check_active()
        with self._lock:
            future = self._queries.get(key)
            shared = future is not None
            if future is None and len(self._queries) < self._max_entries:
                future = Future()
                self._queries[key] = future
        if future is None:
            return execute(), False
        if not shared:
            try:
                result = execute()
                check_active()
                future.set_result([chunk.model_copy(deep=True) for chunk in result])
            except BaseException as error:
                future.set_exception(error)
                with self._lock:
                    self._queries.pop(key, None)
                raise
        while True:
            check_active()
            try:
                return (
                    [
                        chunk.model_copy(deep=True)
                        for chunk in future.result(timeout=0.05)
                    ],
                    shared,
                )
            except FutureTimeout:
                if future.done():
                    raise
