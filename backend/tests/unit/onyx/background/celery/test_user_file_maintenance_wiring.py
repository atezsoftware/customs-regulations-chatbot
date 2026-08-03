import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from onyx.background.celery.tasks.user_file_maintenance import tasks


def _backend_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "onyx").is_dir() and (parent / "tests").is_dir()
    )


@patch.object(tasks, "_load_implementation")
def test_delete_adapter_delegates_to_existing_implementation(
    mock_load_implementation: MagicMock,
) -> None:
    implementation = MagicMock()
    mock_load_implementation.return_value = implementation

    tasks._run_delete(
        user_file_id="file-id",
        tenant_id="tenant-id",
    )

    mock_load_implementation.assert_called_once_with("delete_user_file_impl")
    implementation.assert_called_once_with(
        user_file_id="file-id",
        tenant_id="tenant-id",
        redis_locking=True,
    )


@patch.object(tasks, "_load_implementation")
def test_project_sync_adapter_delegates_to_existing_implementation(
    mock_load_implementation: MagicMock,
) -> None:
    implementation = MagicMock()
    mock_load_implementation.return_value = implementation

    tasks._run_project_sync(
        user_file_id="file-id",
        tenant_id="tenant-id",
    )

    mock_load_implementation.assert_called_once_with("project_sync_user_file_impl")
    implementation.assert_called_once_with(
        user_file_id="file-id",
        tenant_id="tenant-id",
        redis_locking=True,
    )


def test_maintenance_worker_has_parser_free_registration_boundary() -> None:
    backend_root = _backend_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(backend_root), env.get("PYTHONPATH")) if value
    )
    verification = """
import sys


blocked_import_roots = {
    "docling",
    "docling_core",
    "markitdown",
    "playwright",
    "pypdfium2",
    "unstructured",
    "unstructured_client",
}


class BlockedImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked_import_roots:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockedImportFinder())

from onyx.background.celery.versioned_apps.user_file_maintenance import app
from onyx.configs.constants import OnyxCeleryTask

app.loader.import_default_modules()
app.finalize()

known_onyx_tasks = {
    value
    for name, value in vars(OnyxCeleryTask).items()
    if name.isupper() and isinstance(value, str)
}
registered_onyx_tasks = known_onyx_tasks.intersection(app.tasks)
expected_tasks = {
    OnyxCeleryTask.DELETE_SINGLE_USER_FILE,
    OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC,
}
assert registered_onyx_tasks == expected_tasks, registered_onyx_tasks

unexpected_modules = {
    "onyx.background.celery.tasks.user_file_processing.tasks",
    "onyx.connectors.file.connector",
    "onyx.indexing.indexing_pipeline",
    "onyx.regulatory.indexing",
}
assert unexpected_modules.isdisjoint(sys.modules), unexpected_modules.intersection(
    sys.modules
)
assert blocked_import_roots.isdisjoint(sys.modules), blocked_import_roots.intersection(
    sys.modules
)

# Cross the lazy execution boundary as well. These fakes stop at the external
# Redis/DB edges, after both canonical implementations have been imported and
# entered through the same adapters used by Celery.
from contextlib import contextmanager

from onyx.background.celery.tasks.user_file_maintenance import tasks


class FakeLock:
    def acquire(self, blocking=False):
        return True

    def owned(self):
        return True

    def release(self):
        return None


class FakeRedis:
    def delete(self, key):
        return None

    def lock(self, key, **kwargs):
        return FakeLock()


class FakeSession:
    def get(self, model, identifier):
        return None


class FakeHeartbeat:
    def __init__(self, lock, **kwargs):
        self.lock = lock

    def start(self):
        return None

    def stop(self):
        return True


@contextmanager
def fake_session():
    yield FakeSession()


tasks._load_implementation("delete_user_file_impl")
implementation_module = sys.modules[tasks._IMPLEMENTATION_MODULE]
implementation_module.get_redis_client = lambda **kwargs: FakeRedis()
implementation_module.get_session_with_current_tenant = fake_session
implementation_module.fetch_user_files_with_access_relationships = (
    lambda *args, **kwargs: []
)
implementation_module._RedisLockHeartbeat = FakeHeartbeat

file_id = "00000000-0000-0000-0000-000000000001"
app.tasks[OnyxCeleryTask.DELETE_SINGLE_USER_FILE].run(
    user_file_id=file_id,
    tenant_id="tenant-id",
)
app.tasks[OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC].run(
    user_file_id=file_id,
    tenant_id="tenant-id",
)

assert blocked_import_roots.isdisjoint(sys.modules), blocked_import_roots.intersection(
    sys.modules
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", verification],
        cwd=backend_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_lite_supervisor_consumes_user_file_maintenance_queues() -> None:
    supervisor_config = (_backend_root() / "supervisord-lite.conf").read_text(
        encoding="utf-8"
    )

    assert "[program:celery_worker_user_file_maintenance]" in supervisor_config
    assert (
        "-A onyx.background.celery.versioned_apps.user_file_maintenance worker"
        in supervisor_config
    )
    assert "-Q user_file_project_sync,user_file_delete" in supervisor_config
