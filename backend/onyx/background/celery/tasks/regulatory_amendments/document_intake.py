"""New-document delivery uses the existing environment-scoped source worker."""

from uuid import UUID

from celery import shared_task

from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.document_intake.models import INTAKE_DELIVERY_SECONDS

INTAKE_TASK_NAME = "regulatory_document_intake_prepare"


def enqueue_intake_batch(
    *, batch_id: UUID, tenant_id: str, reprepare_request_id: UUID | None = None
) -> None:
    from onyx.background.celery.versioned_apps.client import app

    app.send_task(
        INTAKE_TASK_NAME,
        kwargs={
            "batch_id": str(batch_id),
            "tenant_id": tenant_id,
            "environment": config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": config.ANNEX_DATABASE_IDENTITY,
            **(
                {"reprepare_request_id": str(reprepare_request_id)}
                if reprepare_request_id
                else {}
            ),
        },
        queue=config.source_queue_name(),
        expires=INTAKE_DELIVERY_SECONDS,
    )


def enqueue_intake_indexes(deliveries: list[tuple[UUID, UUID]], tenant_id: str) -> None:
    from onyx.background.celery.versioned_apps.client import app

    for file_id, token in deliveries:
        app.send_task(
            OnyxCeleryTask.INDEX_SINGLE_USER_FILE,
            kwargs={
                "user_file_id": str(file_id),
                "tenant_id": tenant_id,
                "index_request_attempt_id": str(token),
            },
            queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
            priority=OnyxCeleryPriority.HIGH,
            expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
        )


@shared_task(name=INTAKE_TASK_NAME, ignore_result=True)
def prepare_document_intake(
    *,
    batch_id: str,
    tenant_id: str,
    environment: str,
    database_identity: str,
    reprepare_request_id: str | None = None,
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling.tasks import (
        enqueue_labeling_run,
    )
    from onyx.regulatory.document_intake.job import run_document_intake
    from shared_configs.contextvars import get_current_tenant_id

    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or get_current_tenant_id() != tenant_id
    ):
        raise ValueError("New document task belongs to another environment or database")
    if reprepare_request_id:
        from onyx.regulatory.document_intake.reprepare_job import (
            run_document_intake_reprepare,
        )

        run_id = run_document_intake_reprepare(
            intake_id=UUID(batch_id),
            request_id=UUID(reprepare_request_id),
            tenant_id=tenant_id,
        )
    else:
        run_id = run_document_intake(intake_id=UUID(batch_id), tenant_id=tenant_id)
    if run_id is not None:
        try:
            enqueue_labeling_run(run_id=run_id, tenant_id=tenant_id)
        except Exception:
            from onyx.db.document_intake import mark_intake_label_dispatch_failed
            from onyx.db.engine.sql_engine import get_session_with_current_tenant

            with get_session_with_current_tenant() as session:
                mark_intake_label_dispatch_failed(session, UUID(batch_id))
                session.commit()
            raise
