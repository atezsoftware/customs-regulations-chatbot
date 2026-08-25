import datetime
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from onyx.db.enums import (
    BenchmarkCostSource,
    BenchmarkRunFailureCode,
    BenchmarkRunItemPhase,
    BenchmarkRunItemStatus,
    BenchmarkRunStatus,
)
from onyx.db.models import (
    BenchmarkQuestion,
    BenchmarkRun,
    BenchmarkRunItem,
    BenchmarkRunJudgment,
)


def _question_snapshot(question: BenchmarkQuestion) -> dict[str, object]:
    return {
        "id": question.id,
        "title": question.title,
        "prompt": question.prompt,
        "reference_answer": question.reference_answer,
        "expected_facts": list(question.expected_facts),
        "expected_citations": list(question.expected_citations),
        "rubric_notes": question.rubric_notes,
        "tags": list(question.tags),
        "document_set_id": question.document_set_id,
        "document_set_name": question.document_set.name,
        "as_of_date": (
            question.as_of_date.isoformat() if question.as_of_date else None
        ),
    }


def get_benchmark_question(
    db_session: Session, question_id: int
) -> BenchmarkQuestion | None:
    return db_session.get(BenchmarkQuestion, question_id)


def get_benchmark_run_status(db_session: Session, run_id: int) -> str | None:
    return db_session.scalar(
        select(BenchmarkRun.status).where(BenchmarkRun.id == run_id)
    )


def get_benchmark_run_item(
    db_session: Session, *, run_id: int, item_id: int
) -> BenchmarkRunItem | None:
    stmt = (
        select(BenchmarkRunItem)
        .where(
            BenchmarkRunItem.id == item_id,
            BenchmarkRunItem.run_id == run_id,
        )
        .options(
            selectinload(BenchmarkRunItem.question),
            selectinload(BenchmarkRunItem.judgment),
        )
    )
    return db_session.scalars(stmt).one_or_none()


def touch_benchmark_run_heartbeat(
    db_session: Session, run_id: int, *, heartbeat_at: datetime.datetime
) -> None:
    db_session.execute(
        update(BenchmarkRun)
        .where(
            BenchmarkRun.id == run_id,
            BenchmarkRun.status == BenchmarkRunStatus.RUNNING.value,
        )
        .values(heartbeat_at=heartbeat_at)
    )
    db_session.commit()


def touch_benchmark_run_items(
    db_session: Session,
    run_id: int,
    item_ids: Sequence[int],
    *,
    heartbeat_at: datetime.datetime,
) -> None:
    if not item_ids:
        return
    db_session.execute(
        update(BenchmarkRunItem)
        .where(
            BenchmarkRunItem.run_id == run_id,
            BenchmarkRunItem.id.in_(item_ids),
            BenchmarkRunItem.status == BenchmarkRunItemStatus.RUNNING.value,
        )
        .values(heartbeat_at=heartbeat_at)
    )
    db_session.commit()


def claim_benchmark_run_item(
    db_session: Session,
    *,
    run_id: int,
    item_id: int,
    started_at: datetime.datetime,
) -> bool:
    """Atomically fence duplicate deliveries before an expensive LLM call."""
    claimed_id = db_session.scalar(
        update(BenchmarkRunItem)
        .where(
            BenchmarkRunItem.id == item_id,
            BenchmarkRunItem.run_id == run_id,
            BenchmarkRunItem.status == BenchmarkRunItemStatus.PENDING.value,
            BenchmarkRunItem.run_id.in_(
                select(BenchmarkRun.id).where(
                    BenchmarkRun.id == run_id,
                    BenchmarkRun.status == BenchmarkRunStatus.RUNNING.value,
                )
            ),
        )
        .values(
            status=BenchmarkRunItemStatus.RUNNING.value,
            execution_phase=BenchmarkRunItemPhase.STARTING.value,
            heartbeat_at=started_at,
            started_at=started_at,
            completed_at=None,
        )
        .returning(BenchmarkRunItem.id)
    )
    db_session.commit()
    return claimed_id is not None


def mark_benchmark_run_item_failed(
    db_session: Session,
    *,
    run_id: int,
    item_id: int,
    error_message: str,
    completed_at: datetime.datetime,
) -> bool:
    failed_id = db_session.scalar(
        update(BenchmarkRunItem)
        .where(
            BenchmarkRunItem.id == item_id,
            BenchmarkRunItem.run_id == run_id,
            BenchmarkRunItem.status.in_(
                {
                    BenchmarkRunItemStatus.PENDING.value,
                    BenchmarkRunItemStatus.RUNNING.value,
                }
            ),
        )
        .values(
            status=BenchmarkRunItemStatus.ERROR.value,
            execution_phase=None,
            error_message=error_message,
            heartbeat_at=completed_at,
            completed_at=completed_at,
        )
        .returning(BenchmarkRunItem.id)
    )
    db_session.commit()
    return failed_id is not None


def mark_benchmark_run_report_failed(
    db_session: Session, run_id: int, error_message: str
) -> None:
    db_session.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .values(report_error=error_message[:4000])
    )
    db_session.commit()


def list_benchmark_questions(
    db_session: Session, *, active_only: bool = False
) -> Sequence[BenchmarkQuestion]:
    stmt = select(BenchmarkQuestion).order_by(BenchmarkQuestion.created_at.desc())
    if active_only:
        stmt = stmt.where(BenchmarkQuestion.is_active.is_(True))
    return db_session.scalars(stmt).all()


def question_has_run_items(db_session: Session, question_id: int) -> bool:
    count = db_session.scalar(
        select(func.count(BenchmarkRunItem.id)).where(
            BenchmarkRunItem.question_id == question_id
        )
    )
    return bool(count)


def document_set_has_benchmark_questions(
    db_session: Session, document_set_id: int
) -> bool:
    count = db_session.scalar(
        select(func.count(BenchmarkQuestion.id)).where(
            BenchmarkQuestion.document_set_id == document_set_id
        )
    )
    return bool(count)


def create_benchmark_run(
    db_session: Session,
    *,
    label: str | None,
    judge_provider: str,
    judge_provider_id: int,
    judge_model: str,
    deep_research: bool,
    created_by: UUID,
    questions: Sequence[BenchmarkQuestion],
    candidates: Sequence[tuple[str, int, str]],
) -> BenchmarkRun:
    run = BenchmarkRun(
        label=label,
        judge_provider=judge_provider,
        judge_provider_id=judge_provider_id,
        judge_model=judge_model,
        deep_research=deep_research,
        created_by=created_by,
        total_items=len(questions) * len(candidates),
    )
    db_session.add(run)
    db_session.flush()
    for provider, provider_id, model_id in candidates:
        for question in questions:
            db_session.add(
                BenchmarkRunItem(
                    run_id=run.id,
                    provider=provider,
                    provider_id=provider_id,
                    model_id=model_id,
                    question_id=question.id,
                    question_snapshot=_question_snapshot(question),
                )
            )
    db_session.commit()
    return get_benchmark_run(db_session, run.id) or run


def get_benchmark_run(db_session: Session, run_id: int) -> BenchmarkRun | None:
    stmt = (
        select(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .options(
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.question),
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.judgment),
        )
    )
    return db_session.scalars(stmt).one_or_none()


def get_benchmark_run_for_update(
    db_session: Session, run_id: int
) -> BenchmarkRun | None:
    """Lock one run while its dispatch state is changed.

    The lock closes the double-click / multi-replica race where two API requests
    could both observe ``pending`` and publish the same expensive run.
    """
    stmt = (
        select(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .with_for_update()
        .options(
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.question),
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.judgment),
        )
    )
    return db_session.scalars(stmt).one_or_none()


def claim_stale_benchmark_runs_for_recovery(
    db_session: Session,
    *,
    stale_before: datetime.datetime,
    claimed_at: datetime.datetime,
    limit: int = 20,
) -> list[int]:
    """Claim stale queued or in-flight runs for idempotent re-delivery.

    Queue age and execution heartbeat are separate so the public start time remains
    immutable. ``SKIP LOCKED`` keeps concurrent API replicas from publishing the
    same recovery probe.
    """
    stmt = (
        select(BenchmarkRun)
        .where(
            or_(
                (
                    (BenchmarkRun.status == BenchmarkRunStatus.QUEUED.value)
                    & BenchmarkRun.queued_at.is_not(None)
                    & (BenchmarkRun.queued_at <= stale_before)
                ),
                (
                    (BenchmarkRun.status == BenchmarkRunStatus.RUNNING.value)
                    & BenchmarkRun.heartbeat_at.is_not(None)
                    & (BenchmarkRun.heartbeat_at <= stale_before)
                ),
            )
        )
        .order_by(
            func.coalesce(BenchmarkRun.heartbeat_at, BenchmarkRun.queued_at),
            BenchmarkRun.id,
        )
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    runs = list(db_session.scalars(stmt).all())
    for run in runs:
        if run.status == BenchmarkRunStatus.QUEUED.value:
            run.queued_at = claimed_at
        else:
            run.heartbeat_at = claimed_at
    if runs:
        db_session.commit()
    return [run.id for run in runs]


def list_benchmark_runs(db_session: Session) -> Sequence[BenchmarkRun]:
    stmt = (
        select(BenchmarkRun)
        .options(
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.question),
            selectinload(BenchmarkRun.items).selectinload(BenchmarkRunItem.judgment),
        )
        .order_by(BenchmarkRun.created_at.desc())
    )
    return db_session.scalars(stmt).all()


def refresh_benchmark_run_counts(db_session: Session, run: BenchmarkRun) -> None:
    rows = db_session.execute(
        select(BenchmarkRunItem.status, func.count(BenchmarkRunItem.id))
        .where(BenchmarkRunItem.run_id == run.id)
        .group_by(BenchmarkRunItem.status)
    ).all()
    counts = {status: count for status, count in rows}
    run.completed_items = counts.get(BenchmarkRunItemStatus.COMPLETED.value, 0)
    run.failed_items = counts.get(BenchmarkRunItemStatus.ERROR.value, 0)


def reset_benchmark_run_for_retry(
    run: BenchmarkRun, *, rerun_completed: bool = False
) -> None:
    for item in run.items:
        original_status = item.status
        retryable_statuses = {
            BenchmarkRunItemStatus.ERROR.value,
            BenchmarkRunItemStatus.CANCELLED.value,
        }
        if rerun_completed:
            retryable_statuses.add(BenchmarkRunItemStatus.COMPLETED.value)
        if original_status not in retryable_statuses:
            continue
        retry_judging_only = item.final_result is not None and (
            item.judge_error is not None
            or original_status
            in {
                BenchmarkRunItemStatus.CANCELLED.value,
                BenchmarkRunItemStatus.COMPLETED.value,
            }
        )
        item.status = BenchmarkRunItemStatus.PENDING.value
        item.error_message = None
        if not retry_judging_only:
            item.final_result = None
            item.input_tokens = None
            item.output_tokens = None
            item.total_tokens = None
            item.duration_ms = None
            item.cost_cents = None
            item.cost_source = BenchmarkCostSource.UNAVAILABLE.value
            item.cited_chunk_ids = []
            item.cited_sources = []
            item.execution_steps = []
            item.llm_calls = []
            item.answer_reasoning = None
            item.chat_session_id = None
            item.assistant_message_id = None
            item.citation_recall = None
            item.citation_precision = None
        item.judge_error = None
        item.execution_phase = None
        item.heartbeat_at = None
        item.started_at = None
        item.completed_at = None
        item.judgment = None
    run.completed_items = sum(
        item.status == BenchmarkRunItemStatus.COMPLETED.value for item in run.items
    )
    run.failed_items = 0
    run.queued_at = None
    run.started_at = None
    run.heartbeat_at = None
    run.completed_at = None
    run.failure_code = None
    run.failure_message = None
    run.report = None
    run.report_error = None
    run.report_input_tokens = None
    run.report_output_tokens = None
    run.report_cost_cents = None


def mark_benchmark_run_failed(
    db_session: Session,
    run_id: int,
    error_message: str,
    failure_code: BenchmarkRunFailureCode = BenchmarkRunFailureCode.EXECUTION_FAILED,
) -> BenchmarkRun | None:
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None or run.status in {
        BenchmarkRunStatus.COMPLETED.value,
        BenchmarkRunStatus.CANCELLED.value,
    }:
        return run
    unfinished_items = [
        item
        for item in run.items
        if item.status
        in {
            BenchmarkRunItemStatus.PENDING.value,
            BenchmarkRunItemStatus.RUNNING.value,
        }
    ]
    if (
        run.status == BenchmarkRunStatus.ERROR.value
        and run.failure_message is not None
        and not unfinished_items
    ):
        return run
    completed_at = run.completed_at or datetime.datetime.now(datetime.timezone.utc)
    diagnostic = run.failure_message or error_message
    run.status = BenchmarkRunStatus.ERROR.value
    run.failure_code = run.failure_code or failure_code.value
    run.failure_message = diagnostic
    run.heartbeat_at = completed_at
    run.completed_at = completed_at
    for item in unfinished_items:
        item.status = BenchmarkRunItemStatus.ERROR.value
        item.execution_phase = None
        item.error_message = item.error_message or diagnostic
        item.heartbeat_at = completed_at
        item.completed_at = completed_at
    run.completed_items = sum(
        item.status == BenchmarkRunItemStatus.COMPLETED.value for item in run.items
    )
    run.failed_items = sum(
        item.status == BenchmarkRunItemStatus.ERROR.value for item in run.items
    )
    db_session.commit()
    return run


def cancel_benchmark_run(db_session: Session, run_id: int) -> BenchmarkRun | None:
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None or run.status in {
        BenchmarkRunStatus.COMPLETED.value,
        BenchmarkRunStatus.ERROR.value,
    }:
        return run
    completed_at = run.completed_at or datetime.datetime.now(datetime.timezone.utc)
    run.status = BenchmarkRunStatus.CANCELLED.value
    run.heartbeat_at = completed_at
    run.completed_at = completed_at
    for item in run.items:
        if item.status in {
            BenchmarkRunItemStatus.PENDING.value,
            BenchmarkRunItemStatus.RUNNING.value,
        }:
            item.status = BenchmarkRunItemStatus.CANCELLED.value
            item.execution_phase = None
            item.heartbeat_at = completed_at
            item.completed_at = completed_at
    db_session.commit()
    return run


def add_benchmark_judgment(
    db_session: Session,
    *,
    run_item: BenchmarkRunItem,
    judge_provider: str,
    judge_model: str,
    correctness_score: int,
    groundedness_score: int,
    completeness_score: int,
    clarity_score: int,
    overall_score: int,
    rationale: str,
    report: dict[str, object],
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    cost_cents: float | None,
    cost_source: str,
) -> BenchmarkRunJudgment:
    judgment = BenchmarkRunJudgment(
        run_item_id=run_item.id,
        judge_provider=judge_provider,
        judge_model=judge_model,
        correctness_score=correctness_score,
        groundedness_score=groundedness_score,
        completeness_score=completeness_score,
        clarity_score=clarity_score,
        overall_score=overall_score,
        rationale=rationale,
        report=report,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_cents=cost_cents,
        cost_source=cost_source,
    )
    db_session.add(judgment)
    return judgment
