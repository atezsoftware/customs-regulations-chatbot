from celery import Task, shared_task

from onyx.background.celery.apps.app_base import task_logger
from onyx.configs.app_configs import AUTO_LLM_CONFIG_URL
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.llm.well_known_providers.auto_update_service import (
    sync_llm_models_from_github,
    sync_vertex_models_from_google,
)
from onyx.server.manage.llm.provider_cache import invalidate_provider_listing_cache


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_AUTO_LLM_UPDATE,
    ignore_result=True,
    trail=False,
    bind=True,
)
def check_for_auto_llm_updates(
    self: Task,  # noqa: ARG001
    *,
    tenant_id: str,  # noqa: ARG001
) -> bool:
    """Refresh Auto providers from Google and the configured recommendation feed."""
    vertex_results = sync_vertex_models_from_google()
    if vertex_results:
        task_logger.info("Vertex model sync results: %s", vertex_results)
    if not AUTO_LLM_CONFIG_URL:
        return True

    try:
        # Sync to database
        with get_session_with_current_tenant() as db_session:
            results = sync_llm_models_from_github(db_session)

            if results:
                invalidate_provider_listing_cache()
                task_logger.info(f"Auto mode sync results: {results}")
            else:
                task_logger.debug("No model updates applied")

    except Exception:
        task_logger.exception("Error in auto LLM update task")
        raise

    return True
