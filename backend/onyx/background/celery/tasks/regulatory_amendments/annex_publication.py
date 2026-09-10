"""Dispatch immutable publication intent; the Task5 publisher consumes this queue."""

from celery import shared_task

from onyx.configs.constants import OnyxCeleryPriority
from onyx.db.models import AnnexPublicationIntent
from onyx.regulatory.amendments.annexes import config

ANNEX_PUBLICATION_TASK = "publish_annex_change"


def enqueue_annex_publication(intent: AnnexPublicationIntent) -> None:
    from onyx.background.celery.versioned_apps.client import app

    if (
        intent.environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or intent.database_identity != config.ANNEX_DATABASE_IDENTITY
    ):
        raise ValueError("publication intent environment/database mismatch")
    app.send_task(
        ANNEX_PUBLICATION_TASK,
        kwargs={
            "intent_id": str(intent.id),
            "change_set_id": str(intent.change_set_id),
            "logical_group_id": str(intent.logical_group_id),
            "review_revision": intent.review_revision,
            "review_sha256": intent.review_sha256,
            "publication_generation": intent.publication_generation,
            "tenant_id": intent.tenant_id,
            "environment": intent.environment,
            "database_identity": intent.database_identity,
        },
        queue=config.publication_queue_name(),
        priority=OnyxCeleryPriority.MEDIUM,
        expires=3600,
        retry=False,
    )


@shared_task(name=ANNEX_PUBLICATION_TASK, ignore_result=True)
def publish_annex_change(
    *,
    intent_id: str,
    change_set_id: str,
    logical_group_id: str,
    review_revision: int,
    review_sha256: str,
    publication_generation: int,
    tenant_id: str,
    environment: str,
    database_identity: str,
) -> str:
    from onyx.regulatory.amendments.annexes.publication_execution import (
        execute_publication,
    )
    from onyx.regulatory.amendments.annexes.publication_execution_models import (
        AnnexPublicationDelivery,
    )
    from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
    from shared_configs.contextvars import get_current_tenant_id

    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or tenant_id != get_current_tenant_id()
        or not MULTI_TENANT
        and tenant_id != POSTGRES_DEFAULT_SCHEMA
    ):
        raise ValueError("publication worker scope mismatch")
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise ValueError("Annex updates are disabled")
    return execute_publication(
        AnnexPublicationDelivery.model_validate(
            dict(
                intent_id=intent_id,
                change_set_id=change_set_id,
                logical_group_id=logical_group_id,
                review_revision=review_revision,
                review_sha256=review_sha256,
                publication_generation=publication_generation,
                tenant_id=tenant_id,
                environment=environment,
                database_identity=database_identity,
            )
        )
    )


@shared_task(name="recover_annex_publications", ignore_result=True)
def recover_annex_publications(
    *, tenant_id: str, environment: str, database_identity: str
) -> int:
    from onyx.background.celery.versioned_apps.client import app
    from onyx.db.regulatory_annex_execution import pending_deliveries
    from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
    from shared_configs.contextvars import get_current_tenant_id

    if (
        not config.REGULATORY_ANNEX_UPDATES_ENABLED
        or environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or tenant_id != get_current_tenant_id()
        or not MULTI_TENANT
        and tenant_id != POSTGRES_DEFAULT_SCHEMA
    ):
        raise ValueError("publication recovery scope mismatch")
    deliveries = pending_deliveries(
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
    )
    for delivery in deliveries:
        app.send_task(
            ANNEX_PUBLICATION_TASK,
            kwargs=delivery.model_dump(mode="json"),
            queue=config.publication_queue_name(),
            priority=OnyxCeleryPriority.MEDIUM,
            expires=3600,
            retry=False,
        )
    return len(deliveries)
