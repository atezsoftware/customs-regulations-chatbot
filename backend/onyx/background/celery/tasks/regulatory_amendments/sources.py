from uuid import UUID

from celery import shared_task

from onyx.configs.constants import OnyxCeleryPriority
from onyx.db.amendment_sources import (
    mark_source_package_failed,
    source_packages_for_redelivery,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.job import run_source_package
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import get_current_tenant_id

_TASK_NAME = "acquire_amendment_sources"
logger = setup_logger()


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


def _validate_scope(
    *, tenant_id: str, environment: str, database_identity: str
) -> None:
    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or tenant_id != get_current_tenant_id()
        or (not MULTI_TENANT and tenant_id != POSTGRES_DEFAULT_SCHEMA)
    ):
        raise ValueError("Source acquisition worker scope mismatch")


@shared_task(name="recover_amendment_sources", ignore_result=True)
def recover_amendment_sources(
    *, tenant_id: str, environment: str, database_identity: str
) -> None:
    _validate_scope(
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
    )
    with get_session_with_current_tenant() as session:
        package_ids = source_packages_for_redelivery(session, environment=environment)
        session.commit()
    for package_id in package_ids:
        try:
            enqueue_source_package(package_id=package_id, tenant_id=tenant_id)
        except Exception:
            logger.warning(
                "Source package redelivery failed; recovery will retry",
                extra={"package_id": str(package_id)},
            )


@shared_task(name=_TASK_NAME, ignore_result=True)
def acquire_amendment_sources(
    *, package_id: str, tenant_id: str, environment: str, database_identity: str
) -> None:
    identifier = UUID(package_id)
    try:
        _validate_scope(
            tenant_id=tenant_id,
            environment=environment,
            database_identity=database_identity,
        )
        if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
            raise ValueError("Annex updates are disabled")
        run_source_package(package_id=identifier, environment=environment)
    except Exception as error:
        # Validation happens before the worker can claim a lease. Without a
        # terminal transition the UI polls `processing` forever and recovery
        # redelivers the same permanently invalid task.
        from shared_configs.contextvars import get_current_tenant_id

        if (
            tenant_id
            and tenant_id == get_current_tenant_id()
            and (MULTI_TENANT or tenant_id == POSTGRES_DEFAULT_SCHEMA)
        ):
            with get_session_with_current_tenant() as session:
                mark_source_package_failed(
                    session,
                    package_id=identifier,
                    environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                    failure=error,
                )
        logger.exception("Source package %s failed before completion", package_id)
        raise
