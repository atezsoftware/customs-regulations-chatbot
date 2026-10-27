"""Recover only unclaimed preparation intents and metadata-complete label waits."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling
from onyx.db.document_intake_reprepare import latest_reprepare
from onyx.db.models import AmendmentSourcePackage, RegulatoryDocumentIntake
from onyx.regulatory.document_intake.models import INTAKE_DELIVERY_SECONDS
from onyx.utils.logger import setup_logger

logger = setup_logger()
INTAKE_REDELIVERY_GRACE_SECONDS = 30
INTAKE_LABEL_WAIT_SECONDS = 60


@dataclass(frozen=True)
class IntakePreparationDelivery:
    batch_id: UUID
    reprepare_request_id: UUID | None = None


def _require_limit(limit: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("Intake recovery requires a limit from 1 to 100")


def reserve_intake_preparation_deliveries(
    session: Session,
    *,
    environment: str,
    limit: int = 10,
    now: datetime | None = None,
) -> list[IntakePreparationDelivery]:
    """Commit reservations before sending hints; an existing claim is never replayed."""
    _require_limit(limit)
    current = now or datetime.now(timezone.utc)
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
                RegulatoryDocumentIntake.stage.in_(("chunking", "reprepare")),
                RegulatoryDocumentIntake.lease_token.is_(None),
                RegulatoryDocumentIntake.lease_expires_at.is_(None),
                RegulatoryDocumentIntake.updated_at
                < current
                - timedelta(
                    seconds=INTAKE_DELIVERY_SECONDS + INTAKE_REDELIVERY_GRACE_SECONDS
                ),
            )
            .order_by(RegulatoryDocumentIntake.updated_at, RegulatoryDocumentIntake.id)
            .limit(limit)
            .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
            .execution_options(populate_existing=True)
        )
    )
    deliveries = []
    for batch in batches:
        request_id = None
        if batch.stage == "reprepare":
            request = latest_reprepare(session, batch.id)
            if request is None or request.status == "prepared":
                intake.fail_intake(
                    session,
                    batch.id,
                    None,
                    "The queued reprepare request is unavailable",
                )
                continue
            request_id = request.id
        batch.updated_at = current
        deliveries.append(IntakePreparationDelivery(batch.id, request_id))
    session.flush()
    return deliveries


def resume_waiting_intake_labeling(
    session: Session,
    *,
    environment: str,
    limit: int = 10,
    now: datetime | None = None,
) -> list[UUID]:
    """Start frozen, already-prepared scope when the document-set slot is free."""
    _require_limit(limit)
    current = now or datetime.now(timezone.utc)
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
                RegulatoryDocumentIntake.stage == "waiting_labeling",
                RegulatoryDocumentIntake.lease_token.is_(None),
                RegulatoryDocumentIntake.lease_expires_at.is_(None),
                RegulatoryDocumentIntake.updated_at
                < current - timedelta(seconds=INTAKE_LABEL_WAIT_SECONDS),
            )
            .order_by(RegulatoryDocumentIntake.updated_at, RegulatoryDocumentIntake.id)
            .limit(limit)
            .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
            .execution_options(populate_existing=True)
        )
    )
    runs = []
    for batch in batches:
        # Busy document sets rotate behind the remaining bounded candidates.
        batch.updated_at = current
        try:
            with session.begin_nested():
                run_id = intake.resume_intake_labeling(session, batch.id)
        except (
            ValueError,
            intake.IntakeConflictError,
            regulatory_labeling.LabelingStateConflictError,
        ) as error:
            intake.fail_intake(session, batch.id, None, str(error))
            logger.warning("Waiting intake labeling could not start: %s", batch.id)
            continue
        if run_id is not None:
            runs.append(run_id)
    session.flush()
    return runs
