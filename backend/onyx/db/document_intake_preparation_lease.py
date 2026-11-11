"""Fence source preparation separately from checkpointed metadata execution."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import (
    get_session_with_current_tenant,
    get_sqlalchemy_engine,
)
from onyx.db.models import RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from onyx.regulatory.document_intake.metadata_lease import maintain_metadata_lease
from shared_configs.contextvars import get_current_tenant_id


def _execution_key(intake_id: UUID) -> int:
    return int.from_bytes(
        sha256(
            f"intake-preparation:{get_current_tenant_id()}:{intake_id}".encode()
        ).digest()[:8],
        byteorder="big",
        signed=True,
    )


def try_reserve_preparation_execution(session: Session, intake_id: UUID) -> bool:
    return bool(
        session.scalar(
            select(func.pg_try_advisory_xact_lock(_execution_key(intake_id)))
        )
    )


@contextmanager
def preparation_execution_lock(intake_id: UUID) -> Iterator[bool]:
    key = _execution_key(intake_id)
    with (
        get_sqlalchemy_engine()
        .connect()
        .execution_options(isolation_level="AUTOCOMMIT") as connection
    ):
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(key))))
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    connection.execute(select(func.pg_advisory_unlock(key)))
                except Exception:
                    connection.invalidate()


def heartbeat_preparation(
    session: Session, intake_id: UUID, token: UUID, *, skip_locked: bool = False
) -> None:
    """Never renew an expired claim or a successor's token."""
    try:
        session.execute(text("SET LOCAL lock_timeout = '5s'"))
        session.execute(text("SET LOCAL statement_timeout = '10s'"))
        batch = session.scalar(
            select(RegulatoryDocumentIntake)
            .where(RegulatoryDocumentIntake.id == intake_id)
            .with_for_update(skip_locked=skip_locked)
            .execution_options(populate_existing=True)
        )
        # Reprepare renews inside its own batch-locked canonical transaction.
        # Its background renewal must not compete with that transaction.
        if batch is None and skip_locked:
            return
        now = datetime.now(timezone.utc)
        if (
            batch is None
            or batch.status != "preparing"
            or batch.stage not in {"chunking", "reprepare"}
            or batch.lease_token != token
            or batch.lease_expires_at is None
            or batch.lease_expires_at <= now
        ):
            raise MetadataLeaseLost("The source preparation lease expired or changed")
        batch.lease_expires_at = now + timedelta(minutes=10)
        session.flush()
    except (OperationalError, SQLAlchemyTimeoutError) as error:
        raise MetadataLeaseLost(
            "Source preparation ownership could not be renewed"
        ) from error


@contextmanager
def maintain_preparation_lease(
    intake_id: UUID, token: UUID
) -> Iterator[Callable[[], None]]:
    def heartbeat(*, skip_locked: bool = False) -> None:
        with get_session_with_current_tenant() as session:
            heartbeat_preparation(session, intake_id, token, skip_locked=skip_locked)
            session.commit()

    heartbeat()
    with maintain_metadata_lease(lambda: heartbeat(skip_locked=True)) as keepalive:

        def check_owner() -> None:
            keepalive()
            heartbeat()

        yield check_owner
        check_owner()
