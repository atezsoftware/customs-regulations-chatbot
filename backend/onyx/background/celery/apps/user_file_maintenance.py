from typing import Any, cast

from celery import Celery, Task, signals
from celery.apps.worker import Worker
from celery.signals import (
    celeryd_init,
    worker_init,
    worker_process_init,
    worker_ready,
    worker_shutdown,
)

import onyx.background.celery.apps.app_base as app_base
from onyx.configs.constants import (
    POSTGRES_CELERY_WORKER_USER_FILE_MAINTENANCE_APP_NAME,
)
from onyx.db.engine.sql_engine import SqlEngine
from onyx.server.metrics.celery_task_metrics import (
    on_celery_task_postrun,
    on_celery_task_prerun,
    on_celery_task_rejected,
    on_celery_task_retry,
    on_celery_task_revoked,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT

logger = setup_logger()

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.user_file_maintenance")
celery_app.Task = app_base.TenantAwareTask  # ty: ignore[invalid-assignment]


@signals.task_prerun.connect
def on_task_prerun(
    sender: Any | None = None,
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    **kwds: Any,
) -> None:
    app_base.on_task_prerun(sender, task_id, task, args, kwargs, **kwds)
    on_celery_task_prerun(task_id, task)


@signals.task_postrun.connect
def on_task_postrun(
    sender: Any | None = None,
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    retval: Any | None = None,
    state: str | None = None,
    **kwds: Any,
) -> None:
    app_base.on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **kwds)
    on_celery_task_postrun(task_id, task, state)


@signals.task_retry.connect
def on_task_retry(sender: Any | None = None, **kwargs: Any) -> None:  # noqa: ARG001
    task_id = getattr(getattr(sender, "request", None), "id", None)
    on_celery_task_retry(task_id, sender)


@signals.task_revoked.connect
def on_task_revoked(sender: Any | None = None, **kwargs: Any) -> None:
    task_name = getattr(sender, "name", None) or str(sender)
    on_celery_task_revoked(kwargs.get("task_id"), task_name)


@signals.task_rejected.connect
def on_task_rejected(sender: Any | None = None, **kwargs: Any) -> None:  # noqa: ARG001
    message = kwargs.get("message")
    task_name: str | None = None
    if message is not None:
        headers = getattr(message, "headers", None) or {}
        task_name = headers.get("task")
    if task_name is None:
        task_name = "unknown"
    on_celery_task_rejected(None, task_name)


@celeryd_init.connect
def on_celeryd_init(sender: str, conf: Any = None, **kwargs: Any) -> None:
    app_base.on_celeryd_init(sender, conf, **kwargs)


@worker_init.connect
def on_worker_init(sender: Worker, **kwargs: Any) -> None:
    logger.info("worker_init signal received.")

    SqlEngine.set_app_name(POSTGRES_CELERY_WORKER_USER_FILE_MAINTENANCE_APP_NAME)
    pool_size = cast(int, sender.concurrency)  # ty: ignore[unresolved-attribute]
    SqlEngine.init_engine(pool_size=pool_size, max_overflow=2)

    app_base.wait_for_redis(sender, **kwargs)
    app_base.wait_for_db(sender, **kwargs)
    app_base.wait_for_document_index_or_shutdown()

    if MULTI_TENANT:
        return

    app_base.on_secondary_worker_init(sender, **kwargs)


@worker_ready.connect
def on_worker_ready(sender: Any, **kwargs: Any) -> None:
    app_base.on_worker_ready(sender, **kwargs)


@worker_shutdown.connect
def on_worker_shutdown(sender: Any, **kwargs: Any) -> None:
    app_base.on_worker_shutdown(sender, **kwargs)


@worker_process_init.connect
def init_worker(**kwargs: Any) -> None:  # noqa: ARG001
    SqlEngine.reset_engine()


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)


for bootstep in app_base.get_bootsteps():
    celery_app.steps["worker"].add(bootstep)

celery_app.autodiscover_tasks(["onyx.background.celery.tasks.user_file_maintenance"])
