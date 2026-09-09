from uuid import UUID

from celery import shared_task

from onyx.configs.constants import OnyxCeleryPriority
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.job import run_source_package
from shared_configs.contextvars import get_current_tenant_id

_TASK_NAME = "acquire_amendment_sources"


def enqueue_source_package(*, package_id: UUID, tenant_id: str) -> None:
    from onyx.background.celery.versioned_apps.client import app

    app.send_task(
        _TASK_NAME,
        kwargs={
            "package_id": str(package_id),
            "tenant_id": tenant_id,
            "environment": config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": config.ANNEX_DATABASE_IDENTITY,
        },
        queue=config.source_queue_name(),
        priority=OnyxCeleryPriority.MEDIUM,
        expires=3600,
        retry=False,
    )


@shared_task(name=_TASK_NAME, ignore_result=True)
def acquire_amendment_sources(
    *, package_id: str, tenant_id: str, environment: str, database_identity: str
) -> None:
    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or tenant_id != get_current_tenant_id()
    ):
        raise ValueError("Source acquisition worker scope mismatch")
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise ValueError("Annex updates are disabled")
    run_source_package(package_id=UUID(package_id), environment=environment)
