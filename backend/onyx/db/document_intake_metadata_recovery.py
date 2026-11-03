"""Reserve bounded metadata-only resumptions without rewriting canonical sources."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata_checkpoint import (
    try_reserve_metadata_execution,
    validate_metadata_recovery_source,
)
from onyx.db.document_intake_queue import INTAKE_REDELIVERY_GRACE_SECONDS
from onyx.db.models import AmendmentSourcePackage, RegulatoryDocumentIntake
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.document_intake.models import INTAKE_DELIVERY_SECONDS

MAX_METADATA_RECOVERIES = 3
METADATA_LEASE_SECONDS = 600


def _fail(batch: RegulatoryDocumentIntake, message: str) -> None:
    batch.status, batch.error = "failed", message
    batch.lease_token, batch.lease_expires_at = None, None


def reserve_intake_metadata_deliveries(
    session: Session,
    *,
    environment: str,
    limit: int = 10,
    now: datetime | None = None,
) -> list[UUID]:
    """A recovery consumes one attempt; a lost broker hint consumes none."""
    if not environment.strip() or not 1 <= limit <= 100:
        raise ValueError("Metadata recovery requires a scope and bounded limit")
    current = now or datetime.now(timezone.utc)
    expired = and_(
        RegulatoryDocumentIntake.lease_token.is_not(None),
        RegulatoryDocumentIntake.lease_expires_at.is_not(None),
        RegulatoryDocumentIntake.lease_expires_at < current,
    )
    pending = and_(
        RegulatoryDocumentIntake.lease_token.is_(None),
        RegulatoryDocumentIntake.lease_expires_at.is_(None),
        RegulatoryDocumentIntake.metadata_recovery_count > 0,
        RegulatoryDocumentIntake.updated_at
        < current
        - timedelta(seconds=INTAKE_DELIVERY_SECONDS + INTAKE_REDELIVERY_GRACE_SECONDS),
    )
    batches = list(
        session.scalars(
            select(RegulatoryDocumentIntake)
            .join(
                AmendmentSourcePackage,
                AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
            )
            .where(
                AmendmentSourcePackage.environment == environment,
                RegulatoryDocumentIntake.status == "preparing",
                RegulatoryDocumentIntake.stage == "metadata",
                RegulatoryDocumentIntake.metadata_checkpoint.is_not(None),
                or_(expired, pending),
            )
            .order_by(RegulatoryDocumentIntake.updated_at, RegulatoryDocumentIntake.id)
            .limit(limit)
            .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
            .execution_options(populate_existing=True)
        )
    )
    deliveries = []
    for batch in batches:
        # An alive execution holds the matching session mutex across provider calls.
        if not try_reserve_metadata_execution(session, batch.id):
            continue
        takeover = batch.lease_token is not None
        if takeover and batch.metadata_recovery_count >= MAX_METADATA_RECOVERIES:
            _fail(
                batch, "Metadata preparation exceeded its bounded worker recovery limit"
            )
            continue
        try:
            with session.begin_nested():
                validate_metadata_recovery_source(session, batch)
        except intake.IntakeConflictError:
            _fail(batch, "The frozen metadata source or request contract changed")
            continue
        if takeover:
            batch.metadata_recovery_count += 1
        batch.lease_token, batch.lease_expires_at = None, None
        batch.updated_at = current
        deliveries.append(batch.id)
    session.flush()
    return deliveries


def claim_intake_metadata_recovery(
    session: Session, intake_id: UUID, *, now: datetime | None = None
) -> UUID | None:
    """Claim only a reserved metadata intent; caller already owns execution mutex."""
    batch = session.scalar(
        select(RegulatoryDocumentIntake)
        .join(
            AmendmentSourcePackage,
            AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
        )
        .where(
            RegulatoryDocumentIntake.id == intake_id,
            AmendmentSourcePackage.environment == config.REGULATORY_ANNEX_ENVIRONMENT,
            RegulatoryDocumentIntake.status == "preparing",
            RegulatoryDocumentIntake.stage == "metadata",
            RegulatoryDocumentIntake.metadata_checkpoint.is_not(None),
            RegulatoryDocumentIntake.metadata_recovery_count.between(
                1, MAX_METADATA_RECOVERIES
            ),
            RegulatoryDocumentIntake.lease_token.is_(None),
            RegulatoryDocumentIntake.lease_expires_at.is_(None),
        )
        .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
        .execution_options(populate_existing=True)
    )
    if batch is None:
        return None
    try:
        with session.begin_nested():
            validate_metadata_recovery_source(session, batch)
    except intake.IntakeConflictError:
        _fail(batch, "The frozen metadata source or request contract changed")
        session.flush()
        return None
    batch.lease_token = uuid4()
    batch.lease_expires_at = (now or datetime.now(timezone.utc)) + timedelta(
        seconds=METADATA_LEASE_SECONDS
    )
    session.flush()
    return batch.lease_token
