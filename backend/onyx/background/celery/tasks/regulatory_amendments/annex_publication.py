"""Dispatch immutable publication intent; the Task5 publisher consumes this queue."""

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
