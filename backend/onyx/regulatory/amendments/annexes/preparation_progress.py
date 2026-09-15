"""Task-local checkpoints shared by evidence and context preparation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft


@dataclass(frozen=True)
class PreparationObserver:
    checkpoint: Callable[[AnnexChangeDraft], None]
    progress: Callable[[str, int, int], None]


_observer: ContextVar[PreparationObserver | None] = ContextVar(
    "annex_preparation_observer", default=None
)


@contextmanager
def observe_preparation(observer: PreparationObserver) -> Iterator[None]:
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def save_preparation_checkpoint(draft: AnnexChangeDraft) -> None:
    if observer := _observer.get():
        observer.checkpoint(draft)


def report_preparation_progress(stage: str, completed: int = 0, total: int = 0) -> None:
    _context_stage.set(stage)
    if observer := _observer.get():
        observer.progress(stage, completed, total)


_context_stage: ContextVar[str] = ContextVar("annex_context_stage", default="context")


def report_context_progress(completed: int, total: int) -> None:
    report_preparation_progress(_context_stage.get(), completed, total)
