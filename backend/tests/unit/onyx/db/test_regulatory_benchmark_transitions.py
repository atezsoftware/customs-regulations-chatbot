from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from onyx.db.enums import BenchmarkRunItemStatus, BenchmarkRunStatus
from onyx.db.models import BenchmarkRun, BenchmarkRunItem
from onyx.db.regulatory_benchmark import (
    cancel_benchmark_run,
    list_benchmark_runs,
    mark_benchmark_run_failed,
    reset_benchmark_run_for_retry,
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


def test_run_history_query_does_not_select_large_report_columns() -> None:
    db_session = MagicMock(spec=Session)
    db_session.scalars.return_value.all.return_value = []

    list_benchmark_runs(db_session)

    statement = db_session.scalars.call_args.args[0]
    selected_columns = str(statement).partition(" FROM ")[0]
    assert "benchmark_run.report" not in selected_columns
    assert "benchmark_run.report_error" not in selected_columns
    assert "benchmark_run.report_input_tokens" not in selected_columns
    assert "benchmark_run.report_output_tokens" not in selected_columns


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


def test_retry_preserves_completed_answer_when_only_judging_failed() -> None:
    judge_item = BenchmarkRunItem(
        status=BenchmarkRunItemStatus.ERROR.value,
        final_result="Preserved candidate answer",
        judge_error="structured validation failed",
        error_message="Judge failed",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        duration_ms=1500,
        cost_cents=0.5,
        cited_chunk_ids=["chunk-1"],
        cited_sources=[{"regulatory_chunk_id": "chunk-1"}],
        execution_steps=[{"kind": "answer"}],
        llm_calls=[{"phase": "answer"}],
        citation_recall=1.0,
        citation_precision=1.0,
    )
    answer_item = BenchmarkRunItem(
        status=BenchmarkRunItemStatus.ERROR.value,
        final_result=None,
        judge_error=None,
        error_message="Answer failed",
        input_tokens=10,
        llm_calls=[{"phase": "answer"}],
    )
    run = BenchmarkRun(
        status=BenchmarkRunStatus.ERROR.value,
        completed_items=0,
        failed_items=2,
        items=[judge_item, answer_item],
    )

    reset_benchmark_run_for_retry(run)

    assert judge_item.status == BenchmarkRunItemStatus.PENDING.value
    assert judge_item.final_result == "Preserved candidate answer"
    assert judge_item.input_tokens == 100
    assert judge_item.llm_calls == [{"phase": "answer"}]
    assert judge_item.cited_chunk_ids == ["chunk-1"]
    assert judge_item.judge_error is None
    assert judge_item.error_message is None

    assert answer_item.status == BenchmarkRunItemStatus.PENDING.value
    assert answer_item.final_result is None
    assert answer_item.input_tokens is None
    assert answer_item.llm_calls == []


def test_cancelled_retry_can_rejudge_completed_and_preserve_available_answers() -> None:
    completed_item = BenchmarkRunItem(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        final_result="Completed answer",
        input_tokens=100,
        judgment=MagicMock(),
    )
    cancelled_with_answer = BenchmarkRunItem(
        status=BenchmarkRunItemStatus.CANCELLED.value,
        final_result="Interrupted after answering",
        input_tokens=80,
        judgment=None,
    )
    cancelled_without_answer = BenchmarkRunItem(
        status=BenchmarkRunItemStatus.CANCELLED.value,
        final_result=None,
        input_tokens=20,
        judgment=None,
    )
    run = BenchmarkRun(
        status=BenchmarkRunStatus.CANCELLED.value,
        completed_items=1,
        failed_items=0,
        items=[completed_item, cancelled_with_answer, cancelled_without_answer],
    )

    reset_benchmark_run_for_retry(run, rerun_completed=True)

    assert completed_item.status == BenchmarkRunItemStatus.PENDING.value
    assert completed_item.final_result == "Completed answer"
    assert completed_item.input_tokens == 100
    assert completed_item.judgment is None
    assert cancelled_with_answer.status == BenchmarkRunItemStatus.PENDING.value
    assert cancelled_with_answer.final_result == "Interrupted after answering"
    assert cancelled_with_answer.input_tokens == 80
    assert cancelled_without_answer.status == BenchmarkRunItemStatus.PENDING.value
    assert cancelled_without_answer.final_result is None
    assert cancelled_without_answer.input_tokens is None
    assert run.completed_items == 0
