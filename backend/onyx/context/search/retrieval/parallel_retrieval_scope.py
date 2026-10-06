"""Opt-in acquisition concurrency for one authorized parallel research call."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT_CHECK: ContextVar[Callable[[], None] | None] = ContextVar(
    "experimental_parallel_retrieval", default=None
)


@contextmanager
def experimental_parallel_retrieval(
    *, check_active: Callable[[], None]
) -> Iterator[None]:
    token = _CURRENT_CHECK.set(check_active)
    try:
        check_active()
        yield
    finally:
        _CURRENT_CHECK.reset(token)


def parallel_retrieval_enabled() -> bool:
    return _CURRENT_CHECK.get() is not None


def check_parallel_retrieval_active() -> None:
    check_active = _CURRENT_CHECK.get()
    if check_active is not None:
        check_active()
