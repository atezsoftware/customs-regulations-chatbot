"""Bounded worker observations; no file, lease, queue or provider mutations."""

import json
from collections import Counter
from datetime import datetime, timezone

from onyx.background.celery.apps.client import celery_app
from onyx.background.celery.celery_redis import (
    celery_get_broker_client,
    celery_get_queue_length,
)
from onyx.configs.constants import OnyxCeleryQueues, OnyxRedisLocks
from onyx.redis.redis_pool import get_redis_client

FILES = {
    "17e7eec4-2bf8-4eee-b64a-370c7f9cc96a",
    "6a9e0ada-2939-4397-ae41-5b7c8d7da69a",
    "9e421263-6504-4b6b-8e4a-ad36f7d5df43",
    "eb5aba9a-5f27-4b50-b977-30f5539f9676",
}
result = {
    "observed_at": datetime.now(timezone.utc).isoformat(),
    "business_mutations": 0,
    "provider_calls": 0,
    "queue_depth": celery_get_queue_length(
        OnyxCeleryQueues.USER_FILE_PROCESSING, celery_get_broker_client(celery_app)
    ),
    "processing_beat_lock_ttl": get_redis_client(tenant_id="public").ttl(
        OnyxRedisLocks.USER_FILE_PROCESSING_BEAT_LOCK
    ),
    "workers": {},
}
inspector = celery_app.control.inspect(timeout=5)
for category in ("active", "reserved", "scheduled"):
    replies = getattr(inspector, category)() or {}
    for worker, tasks in replies.items():
        selected = []
        names = Counter()
        for envelope in tasks:
            task = envelope.get("request", envelope)
            names[task.get("name", "unknown")] += 1
            kwargs = task.get("kwargs", {})
            if isinstance(kwargs, dict) and kwargs.get("user_file_id") in FILES:
                selected.append(
                    {
                        key: task.get(key)
                        for key in (
                            "id",
                            "name",
                            "time_start",
                            "acknowledged",
                            "worker_pid",
                        )
                    }
                    | {
                        "kwargs": {
                            key: kwargs.get(key)
                            for key in (
                                "user_file_id",
                                "tenant_id",
                                "index_request_attempt_id",
                            )
                        }
                    }
                )
        result["workers"].setdefault(worker, {})[category] = {
            "task_counts": dict(names),
            "target_tasks": selected,
        }
print(json.dumps(result, sort_keys=True))
