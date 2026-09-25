"""Optional shared admission for nested retrieval lanes in an analysis scope."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import BoundedSemaphore


@dataclass(frozen=True)
class _SearchBudget:
    slots: BoundedSemaphore
    check_resources: Callable[[], None] | None


_budget: ContextVar[_SearchBudget | None] = ContextVar("search_budget", default=None)


@contextmanager
def limit_search_concurrency(
    limit: int, *, check_resources: Callable[[], None] | None = None
) -> Iterator[None]:
    if type(limit) is not int or limit < 1:
        raise ValueError("Search concurrency must be a positive integer")
    token = _budget.set(_SearchBudget(BoundedSemaphore(limit), check_resources))
    try:
        yield
    finally:
        _budget.reset(token)


@contextmanager
def search_slot() -> Iterator[None]:
    budget = _budget.get()
    if budget is None:
        yield
        return
    while True:
        if budget.check_resources is not None:
            budget.check_resources()
        if budget.slots.acquire(timeout=0.1):
            break
    try:
        if budget.check_resources is not None:
            budget.check_resources()
        yield
    finally:
        budget.slots.release()
