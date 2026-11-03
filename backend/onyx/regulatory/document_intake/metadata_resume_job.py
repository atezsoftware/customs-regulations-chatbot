"""Resume frozen metadata work without replaying original canonical preparation."""

from uuid import UUID

from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata_checkpoint import metadata_execution_lock
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from onyx.regulatory.document_intake.metadata_job import run_intake_metadata_locked


def run_document_intake_metadata_recovery(
    *, intake_id: UUID, tenant_id: str
) -> UUID | None:
    from onyx.db.document_intake_metadata_recovery import claim_intake_metadata_recovery

    with metadata_execution_lock(intake_id) as acquired:
        if not acquired:
            return None
        with get_session_with_current_tenant() as session:
            token = claim_intake_metadata_recovery(session, intake_id)
            session.commit()
        if token is None:
            return None
        try:
            run_intake_metadata_locked(
                intake_id=intake_id, lease_token=token, tenant_id=tenant_id
            )
            with get_session_with_current_tenant() as session:
                run_id = intake.start_intake_labeling(session, intake_id, token)
                session.commit()
            return run_id
        except MetadataLeaseLost:
            return None
        except Exception as error:
            with get_session_with_current_tenant() as session:
                intake.fail_intake(session, intake_id, token, str(error))
                session.commit()
            raise
