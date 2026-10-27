"""Reserve durable intake deliveries before emitting expendable broker hints."""

from celery import shared_task

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.regulatory.amendments.annexes import config
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()


@shared_task(name="recover_document_intakes", ignore_result=True)
def recover_document_intakes(
    *, tenant_id: str, environment: str, database_identity: str
) -> None:
    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or tenant_id != get_current_tenant_id()
        or (not MULTI_TENANT and tenant_id != POSTGRES_DEFAULT_SCHEMA)
    ):
        raise ValueError("New document recovery scope mismatch")
    if not config.REGULATORY_ANNEX_WORKER_ENABLED:
        return

    from onyx.db.document_intake_queue import (
        reserve_intake_preparation_deliveries,
        resume_waiting_intake_labeling,
    )

    with get_session_with_current_tenant() as session:
        deliveries = reserve_intake_preparation_deliveries(
            session, environment=environment, limit=10
        )
        run_ids = resume_waiting_intake_labeling(
            session, environment=environment, limit=10
        )
        session.commit()

    # The database intent survives a failed broker send; the next tick can retry.
    if deliveries:
        from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
            enqueue_intake_batch,
        )

        for delivery in deliveries:
            try:
                enqueue_intake_batch(
                    batch_id=delivery.batch_id,
                    tenant_id=tenant_id,
                    reprepare_request_id=delivery.reprepare_request_id,
                )
            except Exception:
                logger.warning(
                    "New document preparation redelivery failed; recovery will retry",
                    extra={"batch_id": str(delivery.batch_id)},
                )
    if run_ids:
        from onyx.background.celery.tasks.regulatory_labeling.tasks import (
            enqueue_labeling_run,
        )

        for run_id in run_ids:
            try:
                enqueue_labeling_run(run_id=run_id, tenant_id=tenant_id)
            except Exception:
                logger.warning(
                    "New document labeling redelivery failed; recovery will retry",
                    extra={"run_id": str(run_id)},
                )
