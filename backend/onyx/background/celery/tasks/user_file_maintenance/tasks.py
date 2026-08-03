"""Parser-free registration boundary for user-file maintenance tasks.

The full worker keeps the canonical implementations in the user-file processing
module. The production-lite worker imports only these adapters at startup and
loads that implementation module after it has received maintenance work. This
keeps ingestion and parser modules out of the lite worker's startup graph while
preserving the public Celery task names and behavior.
"""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from celery import Task, shared_task

from onyx.configs.constants import OnyxCeleryTask

_IMPLEMENTATION_MODULE = "onyx.background.celery.tasks.user_file_processing.tasks"


class _MaintenanceImplementation(Protocol):
    def __call__(
        self, *, user_file_id: str, tenant_id: str, redis_locking: bool
    ) -> None: ...


def _load_implementation(name: str) -> _MaintenanceImplementation:
    module = import_module(_IMPLEMENTATION_MODULE)
    implementation = getattr(module, name)
    if not isinstance(implementation, Callable):
        raise TypeError(
            f"User-file maintenance implementation {name!r} is not callable"
        )
    return cast(_MaintenanceImplementation, implementation)


def _run_delete(*, user_file_id: str, tenant_id: str) -> None:
    delete_user_file_impl = _load_implementation("delete_user_file_impl")
    delete_user_file_impl(
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        redis_locking=True,
    )


@shared_task(
    name=OnyxCeleryTask.DELETE_SINGLE_USER_FILE,
    bind=True,
    ignore_result=True,
)
def process_single_user_file_delete(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    _run_delete(user_file_id=user_file_id, tenant_id=tenant_id)


def _run_project_sync(*, user_file_id: str, tenant_id: str) -> None:
    project_sync_user_file_impl = _load_implementation("project_sync_user_file_impl")
    project_sync_user_file_impl(
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        redis_locking=True,
    )


@shared_task(
    name=OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC,
    bind=True,
    ignore_result=True,
)
def process_single_user_file_project_sync(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    _run_project_sync(user_file_id=user_file_id, tenant_id=tenant_id)
