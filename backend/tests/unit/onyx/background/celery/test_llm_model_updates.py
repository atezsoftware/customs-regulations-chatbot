import configparser
import importlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from onyx.background.celery.tasks.llm_model_update import tasks


def test_lite_scheduler_routes_discovery_to_a_registered_worker() -> None:
    from onyx.background.celery.apps.light import celery_app
    from onyx.configs import app_configs

    backend = Path(__file__).resolve().parents[5]
    dockerfile = (backend / "Dockerfile.runtime-lite").read_text()
    match = re.search(r'ENV BEAT_TASK_ALLOWLIST="([^"]+)"', dockerfile)
    assert match is not None
    schedule = importlib.import_module("onyx.background.celery.tasks.beat_schedule")
    try:
        with (
            patch.object(app_configs, "BEAT_TASK_ALLOWLIST", match.group(1).split(",")),
            patch.object(app_configs, "AUTO_LLM_CONFIG_URL", ""),
        ):
            importlib.reload(schedule)
            entry = next(
                template
                for template in schedule.get_tasks_to_schedule()
                if template["task"] == "check_for_auto_llm_update"
            )
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(backend / "supervisord-lite.conf")
        command = parser["program:celery_worker_light"]["command"]
        queues = command.split("-Q ", 1)[1].split()[0].split(",")
        assert entry["options"]["queue"] in queues
        celery_app.loader.import_default_modules()
        assert entry["task"] in celery_app.tasks
    finally:
        importlib.reload(schedule)


def test_google_discovery_runs_without_github_recommendation_url() -> None:
    with (
        patch.object(tasks, "AUTO_LLM_CONFIG_URL", ""),
        patch.object(
            tasks, "sync_vertex_models_from_google", return_value={"7": 1}
        ) as sync,
    ):
        result = tasks.check_for_auto_llm_updates.run(tenant_id="public")
    assert result is True
    sync.assert_called_once_with()


def test_github_sync_invalidates_chat_provider_cache() -> None:
    with (
        patch.object(tasks, "AUTO_LLM_CONFIG_URL", "https://example.test/models.json"),
        patch.object(tasks, "sync_vertex_models_from_google", return_value={}),
        patch.object(
            tasks, "get_session_with_current_tenant", return_value=MagicMock()
        ),
        patch.object(tasks, "sync_llm_models_from_github", return_value={"3": 1}),
        patch.object(tasks, "invalidate_provider_listing_cache") as invalidate,
    ):
        tasks.check_for_auto_llm_updates.run(tenant_id="public")
    invalidate.assert_called_once_with()
