from collections.abc import ItemsView
from datetime import timedelta
from typing import Any

from celery import Celery, signals
from celery.beat import PersistentScheduler
from celery.signals import beat_init
from celery.utils.log import get_task_logger

import onyx.background.celery.apps.app_base as app_base
from onyx.background.celery.celery_utils import make_probe_path
from onyx.background.celery.tasks.regulatory_indexing.beat_schedule import (
    PRODUCTION_LITE_TASK_TEMPLATES,
)
from onyx.configs.constants import (
    POSTGRES_CELERY_BEAT_REGULATORY_INDEXING_APP_NAME,
)
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.engine.tenant_utils import get_all_tenant_ids
from shared_configs.configs import IGNORED_SYNCING_TENANT_LIST

task_logger = get_task_logger(__name__)

_BEAT_HOSTNAME = "regulatory_indexing_beat@hostname"

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.regulatory_indexing_beat")


class RegulatoryIndexingScheduler(PersistentScheduler):
    """Tenant-aware production-lite schedule with no full-runtime tasks."""

    RELOAD_INTERVAL_SECONDS = 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reload_interval = timedelta(seconds=self.RELOAD_INTERVAL_SECONDS)
        self._last_reload = self.app.now() - self._reload_interval
        self._liveness_probe_path = make_probe_path("liveness", _BEAT_HOSTNAME)

    def tick(self) -> float:  # ty: ignore[invalid-method-override]
        next_interval = super().tick()
        now = self.app.now()
        if now - self._last_reload >= self._reload_interval:
            try:
                self.update_schedule()
                self._liveness_probe_path.touch()
            except Exception:
                task_logger.exception(
                    "Failed to refresh the regulatory indexing Beat schedule"
                )
            self._last_reload = now
        return next_interval

    @staticmethod
    def generate_schedule(tenant_ids: list[str]) -> dict[str, dict[str, Any]]:
        schedule: dict[str, dict[str, Any]] = {}
        for tenant_id in tenant_ids:
            if IGNORED_SYNCING_TENANT_LIST and tenant_id in IGNORED_SYNCING_TENANT_LIST:
                task_logger.info("Skipping ignored tenant %s", tenant_id)
                continue
            for template in PRODUCTION_LITE_TASK_TEMPLATES:
                name = f"{template['name']}-{tenant_id}"
                schedule[name] = {
                    "task": template["task"],
                    "schedule": template["schedule"],
                    "kwargs": {"tenant_id": tenant_id},
                    "options": template["options"],
                }
        return schedule

    @staticmethod
    def _schedules_match(
        current: ItemsView[str, Any], desired: dict[str, dict[str, Any]]
    ) -> bool:
        current_by_name = dict(current)
        if set(current_by_name) != set(desired):
            return False
        for name, desired_entry in desired.items():
            current_entry = current_by_name[name]
            current_schedule = getattr(
                current_entry.schedule, "run_every", current_entry.schedule
            )
            if (
                current_entry.task != desired_entry["task"]
                or current_schedule != desired_entry["schedule"]
                or current_entry.kwargs != desired_entry["kwargs"]
                or current_entry.options != desired_entry["options"]
            ):
                return False
        return True

    def update_schedule(self) -> None:
        tenant_ids = get_all_tenant_ids()
        desired = self.generate_schedule(tenant_ids)
        if self._schedules_match(self.schedule.items(), desired):
            task_logger.info(
                "Regulatory indexing Beat schedule unchanged: tasks=%s", len(desired)
            )
            return

        entries = {
            name: self.Entry(
                name=name,
                app=self.app,
                task=entry["task"],
                schedule=entry["schedule"],
                kwargs=entry["kwargs"],
                options=entry["options"],
            )
            for name, entry in desired.items()
        }
        self.schedule.clear()
        self.schedule.update(entries)
        self.sync()
        task_logger.info(
            "Regulatory indexing Beat schedule updated: tenants=%s tasks=%s",
            len(tenant_ids),
            len(desired),
        )


@beat_init.connect
def on_beat_init(sender: Any, **kwargs: Any) -> None:
    task_logger.info("regulatory indexing beat_init signal received")
    SqlEngine.set_app_name(POSTGRES_CELERY_BEAT_REGULATORY_INDEXING_APP_NAME)
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    app_base.wait_for_redis(sender, **kwargs)
    app_base.wait_for_db(sender, **kwargs)

    scheduler: RegulatoryIndexingScheduler = sender.scheduler
    scheduler.update_schedule()
    make_probe_path("liveness", _BEAT_HOSTNAME).touch()
    readiness_path = make_probe_path("readiness", _BEAT_HOSTNAME)
    readiness_path.touch()
    task_logger.info("Regulatory indexing Beat is ready at %s", readiness_path)


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)


celery_app.conf.beat_scheduler = RegulatoryIndexingScheduler
celery_app.conf.task_default_base = app_base.TenantAwareTask
