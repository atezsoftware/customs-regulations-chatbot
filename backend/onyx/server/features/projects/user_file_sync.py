from uuid import UUID

from fastapi import BackgroundTasks

from onyx.configs.app_configs import DISABLE_VECTOR_DB
from onyx.configs.constants import (
    USER_FILE_PROJECT_SYNC_MAX_QUEUE_DEPTH,
    OnyxCeleryPriority,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


def trigger_user_file_metadata_sync(
    user_file_id: UUID,
    tenant_id: str,
    background_tasks: BackgroundTasks | None = None,
) -> None:
    """Schedule synchronization of mutable UserFile metadata and ACLs."""
    if DISABLE_VECTOR_DB and background_tasks is not None:
        from onyx.background.task_utils import drain_project_sync_loop

        background_tasks.add_task(drain_project_sync_loop, tenant_id)
        logger.info("Queued in-process metadata sync for user_file_id=%s", user_file_id)
        return

    from onyx.background.celery.tasks.user_file_processing.tasks import (
        enqueue_user_file_project_sync_task,
        get_user_file_project_sync_queue_depth,
    )
    from onyx.background.celery.versioned_apps.client import app as client_app
    from onyx.redis.redis_pool import get_redis_client

    queue_depth = get_user_file_project_sync_queue_depth(client_app)
    if queue_depth > USER_FILE_PROJECT_SYNC_MAX_QUEUE_DEPTH:
        logger.warning(
            "Skipping immediate metadata sync for user_file_id=%s due to queue "
            "depth %s>%s. It will be picked up by beat later.",
            user_file_id,
            queue_depth,
            USER_FILE_PROJECT_SYNC_MAX_QUEUE_DEPTH,
        )
        return

    redis_client = get_redis_client(tenant_id=tenant_id)
    enqueued = enqueue_user_file_project_sync_task(
        celery_app=client_app,
        redis_client=redis_client,
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        priority=OnyxCeleryPriority.HIGHEST,
    )
    if not enqueued:
        logger.info(
            "Skipped duplicate metadata sync enqueue for user_file_id=%s",
            user_file_id,
        )
        return

    logger.info("Triggered metadata sync for user_file_id=%s", user_file_id)
