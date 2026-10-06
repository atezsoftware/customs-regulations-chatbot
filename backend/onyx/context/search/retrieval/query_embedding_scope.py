import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


class ParallelQueryEmbeddingScope:
    """Bound concurrent cold probes without retaining queries or vectors."""

    def __init__(self) -> None:
        self._cold_probe_slots = threading.BoundedSemaphore(4)

    @contextmanager
    def cold_probe(self, check_active: Callable[[], None]) -> Iterator[None]:
        check_active()
        while not self._cold_probe_slots.acquire(timeout=0.05):
            check_active()
        try:
            check_active()
            yield
        finally:
            self._cold_probe_slots.release()


@dataclass(frozen=True)
class QueryEmbeddingScopeBinding:
    scope: ParallelQueryEmbeddingScope
    check_active: Callable[[], None]


_query_embedding_scope: contextvars.ContextVar[QueryEmbeddingScopeBinding | None] = (
    contextvars.ContextVar("experimental_parallel_query_embeddings", default=None)
)


def current_parallel_query_embedding_scope() -> QueryEmbeddingScopeBinding | None:
    return _query_embedding_scope.get()


@contextmanager
def experimental_parallel_query_embeddings(
    scope: ParallelQueryEmbeddingScope,
    *,
    check_active: Callable[[], None],
) -> Iterator[None]:
    check_active()
    token = _query_embedding_scope.set(QueryEmbeddingScopeBinding(scope, check_active))
    try:
        yield
    finally:
        _query_embedding_scope.reset(token)
