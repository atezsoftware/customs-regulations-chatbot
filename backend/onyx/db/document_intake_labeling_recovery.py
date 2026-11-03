"""Reconcile committed labeling outcomes without depending on a review-page GET."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata import validate_intake_metadata_receipts
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()
_TERMINAL = ("completed", "completed_with_errors", "failed", "cancelled")


def reconcile_terminal_intake_labeling(
    session: Session, *, environment: str, limit: int = 10
) -> list[UUID]:
    """Commit through the caller; a lost worker delivery is repaired next tick."""
    if not environment.strip() or not 1 <= limit <= 100:
        raise ValueError("Intake labeling recovery requires a scope and bounded limit")
    batches = list(
        session.scalars(
            select(RegulatoryDocumentIntake)
            .join(
                AmendmentSourcePackage,
                AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
            )
            .join(
                RegulatoryLabelingRun,
                RegulatoryLabelingRun.id == RegulatoryDocumentIntake.labeling_run_id,
            )
            .where(
                AmendmentSourcePackage.environment == environment,
                RegulatoryDocumentIntake.status == "preparing",
                RegulatoryDocumentIntake.stage == "labeling",
                RegulatoryDocumentIntake.lease_token.is_(None),
                RegulatoryDocumentIntake.lease_expires_at.is_(None),
                RegulatoryLabelingRun.status.in_(_TERMINAL),
                RegulatoryLabelingRun.stage == "finished",
                RegulatoryLabelingRun.lease_token.is_(None),
                RegulatoryLabelingRun.lease_expires_at.is_(None),
                RegulatoryLabelingRun.document_set_id
                == RegulatoryDocumentIntake.document_set_id,
                RegulatoryLabelingRun.model_configuration_id
                == RegulatoryDocumentIntake.labeling_model_configuration_id,
                RegulatoryLabelingRun.requested_by_id
                == RegulatoryDocumentIntake.created_by,
            )
            .order_by(RegulatoryDocumentIntake.updated_at, RegulatoryDocumentIntake.id)
            .limit(limit)
            .with_for_update(of=RegulatoryDocumentIntake, skip_locked=True)
            .execution_options(populate_existing=True)
        )
    )
    reconciled = []
    for batch in batches:
        try:
            with session.begin_nested():
                # No run lock is taken while the batch is held. Refresh a possibly
                # cached run after acquiring the batch lock, then use normal guards.
                run = session.get_one(
                    RegulatoryLabelingRun,
                    batch.labeling_run_id,
                    populate_existing=True,
                )
                if run.status == "completed":
                    validate_intake_metadata_receipts(session, batch)
                intake.refresh_intake(session, batch)
        except (ValueError, intake.IntakeConflictError):
            intake.fail_intake(
                session,
                batch.id,
                None,
                "Completed labeling source evidence changed; review this import before approval",
            )
            logger.warning(
                "Finished intake labeling evidence failed validation: %s", batch.id
            )
        reconciled.append(batch.id)
    session.flush()
    return reconciled
