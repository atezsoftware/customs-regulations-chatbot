from datetime import timedelta
from typing import Any

from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE
from onyx.configs.constants import (
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.regulatory.amendments.annexes import config as annex_config

# This schedule is intentionally separate from the full-runtime Beat schedule.
# Production-lite must recover durable indexing jobs and emit queue metrics, but
# must not dispatch connector ingestion or generic document-indexing work.
PRODUCTION_LITE_TASK_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "recover-amendment-sources",
        "task": "recover_amendment_sources",
        "schedule": timedelta(minutes=1),
        "kwargs": {
            "environment": annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": annex_config.ANNEX_DATABASE_IDENTITY,
        },
        "options": {
            "priority": OnyxCeleryPriority.LOW,
            "expires": 60,
            "queue": annex_config.publication_queue_name(),
        },
    },
    {
        "name": "recover-annex-publications",
        "task": "recover_annex_publications",
        "schedule": timedelta(minutes=1),
        "kwargs": {
            "environment": annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": annex_config.ANNEX_DATABASE_IDENTITY,
        },
        "options": {
            "priority": OnyxCeleryPriority.LOW,
            "expires": 5 * 60,
            "queue": annex_config.publication_queue_name(),
        },
    },
    {
        "name": "recover-stale-regulatory-amendments",
        "task": OnyxCeleryTask.REGULATORY_AMENDMENT_RECOVER_STALE,
        "schedule": timedelta(minutes=1),
        "options": {
            "priority": OnyxCeleryPriority.LOW,
            "expires": 5 * 60,
            "queue": REGULATORY_AMENDMENT_QUEUE,
        },
    },
    {
        "name": "recover-stale-regulatory-indexing",
        "task": OnyxCeleryTask.REGULATORY_INDEXING_RECOVER_STALE,
        "schedule": timedelta(minutes=1),
        "options": {
            "priority": OnyxCeleryPriority.LOW,
            "expires": 5 * 60,
            "queue": OnyxCeleryQueues.REGULATORY_INDEXING,
        },
    },
    {
        "name": "recover-stale-regulatory-labeling",
        "task": OnyxCeleryTask.REGULATORY_LABELING_RECOVER_STALE,
        "schedule": timedelta(minutes=1),
        "options": {
            "priority": OnyxCeleryPriority.LOW,
            "expires": 5 * 60,
            "queue": OnyxCeleryQueues.REGULATORY_INDEXING,
        },
    },
    {
        "name": "monitor-celery-queues",
        "task": OnyxCeleryTask.MONITOR_CELERY_QUEUES,
        "schedule": timedelta(seconds=10),
        "options": {
            "priority": OnyxCeleryPriority.MEDIUM,
            "expires": 15 * 60,
            "queue": OnyxCeleryQueues.MONITORING,
        },
    },
)
