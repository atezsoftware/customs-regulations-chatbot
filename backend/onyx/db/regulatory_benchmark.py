import datetime
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from onyx.db.enums import (
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
        "project_id": question.project_id,
        "project_name": question.project.name,
        "as_of_date": (
            question.as_of_date.isoformat() if question.as_of_date else None
        ),
    }


def get_benchmark_question(
    db_session: Session, question_id: int
) -> BenchmarkQuestion | None:
    return db_session.get(BenchmarkQuestion, question_id)


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


def create_benchmark_run(
    db_session: Session,
    *,
    label: str | None,
    judge_provider: str,
    judge_model: str,
    deep_research: bool,
    created_by: UUID,
    questions: Sequence[BenchmarkQuestion],
    candidates: Sequence[tuple[str, str]],
) -> BenchmarkRun:
    run = BenchmarkRun(
        label=label,
        judge_provider=judge_provider,
        judge_model=judge_model,
        deep_research=deep_research,
        created_by=created_by,
        total_items=len(questions) * len(candidates),
    )
    db_session.add(run)
    db_session.flush()
    for provider, model_id in candidates:
        for question in questions:
            db_session.add(
                BenchmarkRunItem(
                    run_id=run.id,
                    provider=provider,
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
    """Claim stale in-flight runs for a bounded, idempotent re-delivery probe.

    ``BenchmarkRun.started_at`` doubles as a progress lease timestamp. The runner
    renews it at item boundaries; API polling only republishes a run after that
    lease goes stale. ``SKIP LOCKED`` keeps concurrent API replicas from publishing
    the same recovery probe.
    """
    stmt = (
        select(BenchmarkRun)
        .where(
            BenchmarkRun.status == BenchmarkRunStatus.RUNNING.value,
            BenchmarkRun.started_at.is_not(None),
            BenchmarkRun.started_at <= stale_before,
        )
        .order_by(BenchmarkRun.started_at, BenchmarkRun.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    runs = list(db_session.scalars(stmt).all())
    for run in runs:
        run.started_at = claimed_at
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


def cancel_benchmark_run(db_session: Session, run: BenchmarkRun) -> None:
    if run.status in {
        BenchmarkRunStatus.COMPLETED.value,
        BenchmarkRunStatus.ERROR.value,
        BenchmarkRunStatus.CANCELLED.value,
    }:
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    run.status = BenchmarkRunStatus.CANCELLED.value
    run.completed_at = now
    for item in run.items:
        if item.status in {
            BenchmarkRunItemStatus.PENDING.value,
            BenchmarkRunItemStatus.RUNNING.value,
        }:
            item.status = BenchmarkRunItemStatus.CANCELLED.value
            item.completed_at = now
    db_session.commit()


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
