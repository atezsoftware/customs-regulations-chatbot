from types import SimpleNamespace
from typing import cast

from sqlalchemy.orm import Session

from onyx.db.enums import BenchmarkRunItemStatus, BenchmarkRunStatus
from onyx.db.models import BenchmarkRun
from onyx.db.regulatory_benchmark import (
    cancel_benchmark_run,
    mark_benchmark_run_failed,
)


class _ScalarResult:
    def __init__(self, run: BenchmarkRun) -> None:
        self._run = run

    def one_or_none(self) -> BenchmarkRun:
        return self._run


class _LockAwareSession:
    def __init__(self, run: BenchmarkRun) -> None:
        self._run = run
        self.locked_reads = 0
        self.commits = 0

    def scalars(self, statement: object) -> _ScalarResult:
        if getattr(statement, "_for_update_arg", None) is None:
            raise AssertionError("terminal transition must lock the benchmark run")
        self.locked_reads += 1
        return _ScalarResult(self._run)

    def commit(self) -> None:
        self.commits += 1


def test_competing_terminal_transitions_lock_and_preserve_first_winner() -> None:
    pending_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.PENDING.value,
        execution_phase="starting",
        error_message=None,
        completed_at=None,
    )
    run = cast(
        BenchmarkRun,
        SimpleNamespace(
            id=7,
            status=BenchmarkRunStatus.RUNNING.value,
            report_error=None,
            failure_code=None,
            failure_message=None,
            heartbeat_at=None,
            completed_at=None,
            completed_items=0,
            failed_items=0,
            items=[pending_item],
        ),
    )
    session = _LockAwareSession(run)

    cancelled = cancel_benchmark_run(cast(Session, session), run.id)
    failed = mark_benchmark_run_failed(
        cast(Session, session), run.id, "persona selection exploded"
    )

    assert cancelled is run
    assert failed is run
    assert session.locked_reads == 2
    assert session.commits == 1
    assert run.status == BenchmarkRunStatus.CANCELLED.value
    assert run.report_error is None
    assert pending_item.status == BenchmarkRunItemStatus.CANCELLED.value
    assert pending_item.execution_phase is None


def test_fully_terminalized_error_transition_is_idempotent() -> None:
    completed_at = object()
    error_item = SimpleNamespace(status=BenchmarkRunItemStatus.ERROR.value)
    run = cast(
        BenchmarkRun,
        SimpleNamespace(
            id=9,
            status=BenchmarkRunStatus.ERROR.value,
            report_error=None,
            failure_code="execution_failed",
            failure_message="original failure",
            heartbeat_at=completed_at,
            completed_at=completed_at,
            completed_items=0,
            failed_items=1,
            items=[error_item],
        ),
    )
    session = _LockAwareSession(run)

    result = mark_benchmark_run_failed(
        cast(Session, session), run.id, "duplicate failure"
    )

    assert result is run
    assert session.locked_reads == 1
    assert session.commits == 0
    assert run.report_error is None
    assert run.failure_message == "original failure"
    assert run.completed_at is completed_at


def test_repeated_cancellation_repairs_unfinished_items() -> None:
    completed_at = object()
    pending_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.PENDING.value,
        execution_phase="starting",
        completed_at=None,
    )
    running_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.RUNNING.value,
        execution_phase="answering",
        completed_at=None,
    )
    completed_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        execution_phase=None,
        completed_at=completed_at,
    )
    run = cast(
        BenchmarkRun,
        SimpleNamespace(
            id=11,
            status=BenchmarkRunStatus.CANCELLED.value,
            completed_at=completed_at,
            items=[pending_item, running_item, completed_item],
        ),
    )
    session = _LockAwareSession(run)

    result = cancel_benchmark_run(cast(Session, session), run.id)

    assert result is run
    assert session.locked_reads == 1
    assert session.commits == 1
    assert pending_item.status == BenchmarkRunItemStatus.CANCELLED.value
    assert pending_item.execution_phase is None
    assert pending_item.completed_at is completed_at
    assert running_item.status == BenchmarkRunItemStatus.CANCELLED.value
    assert running_item.execution_phase is None
    assert running_item.completed_at is completed_at
    assert completed_item.status == BenchmarkRunItemStatus.COMPLETED.value
    assert completed_item.completed_at is completed_at
