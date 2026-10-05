"""PostgreSQL cohort tests in an isolated schema, without model calls."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import Table, create_engine, text
from sqlalchemy.orm import Session

from onyx.db.models import KVStore, UsageMeasurement, UserUsage
from onyx.db.usage_measurement import (
    get_measurement_period,
    measurement_activity,
    measurement_export,
    record_measurement,
    set_measurement_workflow,
    start_measurement_period,
)
from onyx.db.user_usage import (
    get_user_cost_cents_since,
    record_user_usage,
    summarize_usage_by_email,
)
from onyx.tracing.usage_measurement import UsageMeasurementContext


@pytest.fixture
def measurement_db() -> Generator[Session, None, None]:
    engine = create_engine(
        "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres"
    )
    schema = "test_usage_measurement_" + uuid4().hex
    with engine.connect() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(text(f'SET search_path TO "{schema}"'))
        connection.execute(
            text('CREATE TABLE "user" (id UUID PRIMARY KEY, email VARCHAR NOT NULL)')
        )
        for table in [
            KVStore.__table__,
            UsageMeasurement.__table__,
            UserUsage.__table__,
        ]:
            cast(Table, table).create(connection)
        connection.commit()
        with Session(bind=connection) as session:
            yield session
        connection.rollback()
        connection.execute(text("SET search_path TO public"))
        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        connection.commit()
    engine.dispose()


def test_reset_late_writes_profiles_and_single_question_denominator(
    measurement_db: Session,
) -> None:
    db = measurement_db
    user, inactive, session_id = uuid4(), uuid4(), uuid4()
    db.execute(
        text(
            'INSERT INTO "user" (id,email) VALUES (:id,:email),(:inactive,:inactive_email)'
        ),
        {
            "id": user,
            "email": "active@example.com",
            "inactive": inactive,
            "inactive_email": "inactive@example.com",
        },
    )
    old_period = start_measurement_period(db)
    db.commit()
    now = datetime.now(timezone.utc)
    old = UsageMeasurementContext(
        epoch=old_period.id,
        request_id=uuid4(),
        started_at=now,
        user_id=str(user),
        session_id=session_id,
        question_id=1,
        workflow="deep",
    )
    record_measurement(db, old)
    db.commit()
    new_period = start_measurement_period(db)
    db.commit()
    assert get_measurement_period(db) == new_period
    assert all(
        row.query_count == 0
        for row in measurement_activity(
            db, new_period, now - timedelta(days=1), now + timedelta(days=1)
        )
    )
    # A buffered old run finishes after reset, on the very same UTC day.
    record_measurement(db, old, model="m", flow="child", input_tokens=999, cost_cents=9)
    deep = old.model_copy(
        update={"epoch": new_period.id, "request_id": uuid4(), "question_id": 2}
    )
    normal = deep.model_copy(
        update={"request_id": uuid4(), "question_id": 3, "workflow": "normal"}
    )
    for context in [deep, normal]:
        record_measurement(db, context)
    for flow in ["coordinator", "child-one", "child-two"]:
        record_measurement(
            db,
            deep,
            model="m",
            flow=flow,
            input_tokens=10,
            output_tokens=5,
            cost_cents=1,
        )
    record_measurement(
        db, normal, model="m", input_tokens=5, output_tokens=5, cost_cents=0.5
    )
    # Regeneration costs count, but the same user question is counted once.
    regenerated = deep.model_copy(update={"request_id": uuid4()})
    record_measurement(db, regenerated, model="m", input_tokens=5, cost_cents=0.5)
    db.commit()
    start, end = now - timedelta(days=1), now + timedelta(days=1)
    rows = measurement_export(db, new_period, start, end)
    activity = measurement_activity(db, new_period, start, end)
    summaries = {
        row.email: row
        for row in summarize_usage_by_email(rows, activity, include_activity_only=True)
    }
    assert summaries["inactive@example.com"].total_tokens == 0
    active = summaries["active@example.com"]
    assert active.total_user_queries == 2
    assert active.total_user_sessions == 1
    assert active.total_tokens == 60
    assert active.cost_cents == 4
    assert active.average_tokens_per_query == 30
    assert (
        sum(
            row.input_tokens + row.output_tokens
            for row in measurement_export(db, new_period, start, end, workflow="deep")
        )
        == 50
    )
    assert len(measurement_export(db, old_period, start, end)) == 1
    # A resumed request pins its original profile before buffered setup costs arrive.
    resumed = normal.model_copy(update={"request_id": uuid4(), "question_id": 4})
    record_measurement(db, resumed)
    set_measurement_workflow(db, str(resumed.request_id), "deep")
    record_measurement(db, resumed, model="setup", input_tokens=7)
    db.commit()
    setup_rows = measurement_export(
        db, new_period, start, end, model="setup", workflow="deep"
    )
    assert len(setup_rows) == 1 and setup_rows[0].input_tokens == 7
    # Analytics reset cannot forgive existing enforcement spend.
    record_user_usage(db, str(user), "m", "chat", None, 10, 0, 0, 12, now)
    db.commit()
    start_measurement_period(db)
    db.commit()
    assert get_user_cost_cents_since(db, str(user), start) == 12


def test_migration_upgrade_and_downgrade(measurement_db: Session) -> None:
    import importlib.util
    from pathlib import Path

    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory

    path = (
        Path(__file__).parents[3] / "alembic/versions/a61e0f9b245c_usage_measurement.py"
    )
    config = Config()
    config.set_main_option("script_location", str(path.parents[1]))
    assert len(ScriptDirectory.from_config(config).get_heads()) == 1
    spec = importlib.util.spec_from_file_location("measurement_migration", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with Operations.context(MigrationContext.configure(measurement_db.connection())):
        migration.downgrade()
        migration.upgrade()
    assert (
        measurement_db.execute(
            text("SELECT count(*) FROM usage_measurement")
        ).scalar_one()
        == 0
    )
