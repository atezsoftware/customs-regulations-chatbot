"""Exact request-cohort reporting; resetting analytics never clears budget usage."""

from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import KVStore, UsageMeasurement, User
from onyx.db.user_usage import (
    DELETED_USER_EXPORT_EMAIL,
    UsageExportRow,
    UserActivityCounts,
    UserUsageTotalsByEmail,
)
from onyx.tracing.usage_measurement import UsageMeasurementContext

EPOCH_KEY = "usage_measurement_epoch"


class MeasurementPeriod(BaseModel):
    id: str
    started_at: datetime


def get_measurement_period(db_session: Session) -> MeasurementPeriod | None:
    value = db_session.execute(
        select(KVStore.value).where(KVStore.key == EPOCH_KEY)
    ).scalar_one_or_none()
    return MeasurementPeriod.model_validate(value) if value else None


def start_measurement_period(db_session: Session) -> MeasurementPeriod:
    period = MeasurementPeriod(id=str(uuid4()), started_at=datetime.now(timezone.utc))
    stmt = insert(KVStore).values(key=EPOCH_KEY, value=period.model_dump(mode="json"))
    db_session.execute(
        stmt.on_conflict_do_update(
            index_elements=[KVStore.key], set_={"value": stmt.excluded.value}
        )
    )
    db_session.flush()
    return period


def record_measurement(
    db_session: Session,
    context: UsageMeasurementContext,
    *,
    model: str = "",
    flow: str = "",
    provider: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cost_cents: float = 0,
) -> None:
    values = context.model_dump(exclude={"user_id"})
    if model:
        root_workflow = (
            select(UsageMeasurement.workflow)
            .where(
                UsageMeasurement.request_id == context.request_id,
                UsageMeasurement.model == "",
                UsageMeasurement.flow == "",
                UsageMeasurement.provider == "",
            )
            .scalar_subquery()
        )
        # Resume can pin a different profile before buffered setup costs arrive.
        values["workflow"] = func.coalesce(root_workflow, context.workflow)
    stmt = insert(UsageMeasurement).values(
        **values,
        user_id=context.user_id,
        model=model,
        flow=flow,
        provider=provider or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_cents=cost_cents,
    )
    db_session.execute(
        stmt.on_conflict_do_update(
            index_elements=["request_id", "model", "flow", "provider"],
            set_={
                name: getattr(UsageMeasurement, name) + getattr(stmt.excluded, name)
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cost_cents",
                )
            },
        )
    )


def _conditions(
    period: MeasurementPeriod, start: datetime, end: datetime, workflow: str | None
) -> list[ColumnElement[bool]]:
    conditions = [
        UsageMeasurement.epoch == period.id,
        UsageMeasurement.started_at >= start,
        UsageMeasurement.started_at < end,
        UsageMeasurement.benchmark.is_(False),
    ]
    if workflow is not None:
        conditions.append(UsageMeasurement.workflow == workflow)
    return conditions


def measurement_export(
    db_session: Session,
    period: MeasurementPeriod,
    start: datetime,
    end: datetime,
    model: str | None = None,
    workflow: str | None = None,
) -> list[UsageExportRow]:
    email = func.coalesce(User.email, DELETED_USER_EXPORT_EMAIL)
    day = func.date(func.timezone("UTC", UsageMeasurement.started_at))
    query = (
        select(
            email,
            UsageMeasurement.model,
            day,
            func.sum(UsageMeasurement.input_tokens),
            func.sum(UsageMeasurement.output_tokens),
            func.sum(UsageMeasurement.cache_read_tokens),
            func.sum(UsageMeasurement.cost_cents),
        )
        .outerjoin(User, UsageMeasurement.user_id == User.id)
        .where(*_conditions(period, start, end, workflow), UsageMeasurement.model != "")
    )
    if model is not None:
        query = query.where(UsageMeasurement.model == model)
    rows = db_session.execute(
        query.group_by(email, UsageMeasurement.model, day).order_by(
            email, day, UsageMeasurement.model
        )
    ).all()
    return [
        UsageExportRow(
            email=e,
            model=m,
            day=str(d),
            input_tokens=int(i or 0),
            output_tokens=int(o or 0),
            cache_read_tokens=int(c or 0),
            cost_cents=float(cost or 0),
        )
        for e, m, d, i, o, c, cost in rows
    ]


def measurement_activity(
    db_session: Session,
    period: MeasurementPeriod,
    start: datetime,
    end: datetime,
    workflow: str | None = None,
) -> list[UserActivityCounts]:
    email = func.coalesce(User.email, DELETED_USER_EXPORT_EMAIL)
    rows = db_session.execute(
        select(
            email,
            func.count(func.distinct(UsageMeasurement.question_id)),
            func.count(func.distinct(UsageMeasurement.session_id)),
        )
        .outerjoin(User, UsageMeasurement.user_id == User.id)
        .where(*_conditions(period, start, end, workflow))
        .group_by(email)
    ).all()
    counts = {
        str(e): UserActivityCounts(
            email=str(e), query_count=int(q), session_count=int(s)
        )
        for e, q, s in rows
    }
    for e in db_session.execute(select(cast(ColumnElement[str], User.email))).scalars():
        counts.setdefault(
            e, UserActivityCounts(email=e, query_count=0, session_count=0)
        )
    return list(counts.values())


def measurement_totals(
    db_session: Session,
    period: MeasurementPeriod,
    start: datetime,
    end: datetime,
    workflow: str | None = None,
) -> list[UserUsageTotalsByEmail]:
    rows = measurement_export(db_session, period, start, end, workflow=workflow)
    totals: dict[str, UserUsageTotalsByEmail] = {}
    for row in rows:
        total = totals.setdefault(
            row.email,
            UserUsageTotalsByEmail(
                email=row.email,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cost_cents=0,
            ),
        )
        total.input_tokens += row.input_tokens
        total.output_tokens += row.output_tokens
        total.cache_read_tokens += row.cache_read_tokens
        total.cost_cents += row.cost_cents
    return list(totals.values())


def set_measurement_workflow(
    db_session: Session, request_id: str, workflow: str
) -> None:
    db_session.execute(
        update(UsageMeasurement)
        .where(UsageMeasurement.request_id == request_id)
        .values(workflow=workflow)
    )


def latest_session_measurement(
    db_session: Session, user_id: str, session_id: str
) -> UsageMeasurementContext | None:
    row = db_session.execute(
        select(UsageMeasurement)
        .where(
            UsageMeasurement.user_id == user_id,
            UsageMeasurement.session_id == session_id,
        )
        .order_by(UsageMeasurement.started_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return UsageMeasurementContext(
        epoch=row.epoch,
        request_id=row.request_id,
        started_at=row.started_at,
        user_id=user_id,
        session_id=row.session_id,
        question_id=row.question_id,
        workflow=row.workflow,
        benchmark=row.benchmark,
    )
