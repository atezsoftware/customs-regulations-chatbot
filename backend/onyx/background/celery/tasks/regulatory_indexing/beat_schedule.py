from datetime import timedelta
from typing import Any

from onyx.configs.constants import (
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)

# This schedule is intentionally separate from the full-runtime Beat schedule.
# Production-lite must recover durable indexing jobs and emit queue metrics, but
# must not dispatch connector ingestion or generic document-indexing work.
PRODUCTION_LITE_TASK_TEMPLATES: tuple[dict[str, Any], ...] = (
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
