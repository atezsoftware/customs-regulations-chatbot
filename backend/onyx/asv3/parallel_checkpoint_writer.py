"""Run-local coalescing persistence for fully captured parallel checkpoints."""

from __future__ import annotations

import contextvars
import copy
import threading
from collections.abc import Callable

from pydantic import JsonValue


class ParallelCheckpointWriter:
    def __init__(self, persist: Callable[[dict[str, JsonValue]], None]) -> None:
        self._persist = persist
        self._changed = threading.Condition()
        self._issued = 0
        self._durable = 0
        self._pending: tuple[int, dict[str, JsonValue]] | None = None
        self._closing = False
        self._error: BaseException | None = None
        captured = contextvars.copy_context()
        self._thread = threading.Thread(
            target=captured.run,
            args=(self._run,),
            name="asv3-checkpoint-writer",
            daemon=False,
        )
        self._thread.start()

    def raise_if_failed(self) -> None:
        with self._changed:
            if self._error is not None:
                raise self._error

    def submit(self, snapshot: dict[str, JsonValue]) -> int:
        self.raise_if_failed()
        captured = copy.deepcopy(snapshot)
        with self._changed:
            if self._error is not None:
                raise self._error
            if self._closing:
                raise RuntimeError("Parallel checkpoint writer is closed")
            self._issued += 1
            self._pending = (self._issued, captured)
            self._changed.notify_all()
            return self._issued

    def flush(self, revision: int | None = None) -> None:
        with self._changed:
            target = self._issued if revision is None else revision
            if type(target) is not int or not 0 <= target <= self._issued:
                raise ValueError("Unknown parallel checkpoint revision")
            while self._durable < target and self._error is None:
                self._changed.wait()
            if self._error is not None:
                raise self._error

    def close(self) -> None:
        with self._changed:
            self._closing = True
            self._changed.notify_all()
        self._thread.join()
        self.raise_if_failed()

    def _run(self) -> None:
        while True:
            with self._changed:
                while self._pending is None and not self._closing:
                    self._changed.wait()
                if self._pending is None:
                    return
                revision, snapshot = self._pending
                self._pending = None
            try:
                self._persist(snapshot)
            except BaseException as error:
                with self._changed:
                    self._error = error
                    self._pending = None
                    self._changed.notify_all()
                return
            with self._changed:
                self._durable = revision
                self._changed.notify_all()
