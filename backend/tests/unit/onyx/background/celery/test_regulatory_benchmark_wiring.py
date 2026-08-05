import inspect
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.routing import APIRoute

from onyx.background.celery.tasks.regulatory_benchmark import tasks as benchmark_tasks
from onyx.configs.app_configs import REGULATORY_BENCHMARK_MAX_QUESTIONS
from onyx.configs.constants import (
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.enums import BenchmarkRunStatus, Permission
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory import benchmark_api
from onyx.server.features.regulatory.benchmark_models import (
    BenchmarkModelSelection,
    BenchmarkRunCreate,
)


def _backend_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "onyx").is_dir() and (parent / "tests").is_dir()
    )


@patch.object(benchmark_api, "benchmark_run_snapshot")
@patch.object(benchmark_api, "get_current_tenant_id", return_value="test_tenant")
@patch.object(benchmark_api, "get_benchmark_run_for_update")
@patch.object(benchmark_api.celery_app, "send_task")
def test_start_run_routes_to_regulatory_benchmark_queue(
    mock_send_task: MagicMock,
    mock_get_benchmark_run_for_update: MagicMock,
    mock_get_tenant_id: MagicMock,
    mock_benchmark_run_snapshot: MagicMock,
) -> None:
    run = MagicMock()
    run.id = 42
    run.status = BenchmarkRunStatus.PENDING.value
    mock_get_benchmark_run_for_update.return_value = run
    expected_snapshot = MagicMock()
    mock_benchmark_run_snapshot.return_value = expected_snapshot
    db_session = MagicMock()

    result = benchmark_api.start_run(
        run_id=run.id,
        user=MagicMock(),
        db_session=db_session,
    )

    assert result is expected_snapshot
    assert run.status == BenchmarkRunStatus.RUNNING.value
    db_session.commit.assert_called_once_with()
    mock_get_tenant_id.assert_called_once_with()
    mock_send_task.assert_called_once_with(
        OnyxCeleryTask.REGULATORY_BENCHMARK_RUN,
        kwargs={"run_id": run.id, "tenant_id": "test_tenant"},
        queue=OnyxCeleryQueues.REGULATORY_BENCHMARK,
        priority=OnyxCeleryPriority.MEDIUM,
        expires=24 * 60 * 60,
    )


@patch.object(benchmark_api, "benchmark_run_snapshot")
@patch.object(benchmark_api, "get_current_tenant_id", return_value="test_tenant")
@patch.object(benchmark_api, "get_benchmark_run_for_update")
@patch.object(benchmark_api.celery_app, "send_task")
def test_concurrent_start_observes_locked_running_state_and_does_not_republish(
    mock_send_task: MagicMock,
    mock_get_benchmark_run_for_update: MagicMock,
    _mock_get_tenant_id: MagicMock,
    mock_benchmark_run_snapshot: MagicMock,
) -> None:
    run = MagicMock(id=42, status=BenchmarkRunStatus.PENDING.value)
    run.id = 42
    run.status = BenchmarkRunStatus.PENDING.value
    mock_get_benchmark_run_for_update.return_value = run
    mock_benchmark_run_snapshot.return_value = MagicMock()

    benchmark_api.start_run(42, user=MagicMock(), db_session=MagicMock())
    with pytest.raises(OnyxError, match="not pending"):
        benchmark_api.start_run(42, user=MagicMock(), db_session=MagicMock())

    mock_send_task.assert_called_once()


def test_polling_requeues_only_runs_claimed_by_the_recovery_lease() -> None:
    with (
        patch.object(
            benchmark_api,
            "claim_stale_benchmark_runs_for_recovery",
            return_value=[7, 9],
        ) as claim,
        patch.object(benchmark_api, "_enqueue_benchmark_run") as enqueue,
    ):
        benchmark_api._recover_stale_runs(MagicMock())

    assert claim.call_count == 1
    assert [call.args for call in enqueue.call_args_list] == [(7,), (9,)]


def test_every_benchmark_route_requires_full_admin_panel_access() -> None:
    def requires_full_admin(call: object) -> bool:
        closure = getattr(call, "__closure__", None) or ()
        return any(
            cell.cell_contents == Permission.FULL_ADMIN_PANEL_ACCESS for cell in closure
        )

    for route in benchmark_api.router.routes:
        assert isinstance(route, APIRoute)
        assert any(
            requires_full_admin(dependency.call)
            for dependency in route.dependant.dependencies
        ), route.path

    assert "current_curator_or_admin_user" not in inspect.getsource(benchmark_api)


def test_run_request_has_bounded_candidate_and_question_lists() -> None:
    selection = BenchmarkModelSelection(provider="openrouter", model_id="model")
    with pytest.raises(ValueError):
        BenchmarkRunCreate(
            question_ids=list(range(1, 10_000)),
            candidates=[selection],
            judge=selection,
        )
    with pytest.raises(ValueError):
        BenchmarkRunCreate(
            question_ids=[1],
            candidates=[selection] * 10_000,
            judge=selection,
        )


def test_implicit_all_questions_is_still_bounded() -> None:
    selection = BenchmarkModelSelection(provider="openrouter", model_id="model")
    request = BenchmarkRunCreate(candidates=[selection], judge=selection)
    questions = [MagicMock() for _ in range(REGULATORY_BENCHMARK_MAX_QUESTIONS + 1)]

    with (
        patch.object(benchmark_api, "_validate_model"),
        patch.object(benchmark_api, "list_benchmark_questions", return_value=questions),
        pytest.raises(OnyxError, match="Too many benchmark questions"),
    ):
        benchmark_api.create_run(request, user=MagicMock(), db_session=MagicMock())


def test_benchmark_task_uses_late_ack_worker_loss_redelivery_and_run_lease() -> None:
    task = benchmark_tasks.run_regulatory_benchmark_task
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True

    lock = MagicMock()
    lock.acquire.return_value = True
    lock.owned.return_value = True
    cache = MagicMock()
    cache.lock.return_value = lock
    db_session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session
    with (
        patch.object(benchmark_tasks, "get_cache_backend", return_value=cache),
        patch.object(benchmark_tasks, "_RunLeaseHeartbeat") as heartbeat_type,
        patch.object(
            benchmark_tasks,
            "get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch("onyx.regulatory.benchmark.runner.run_benchmark") as run_benchmark,
    ):
        benchmark_tasks.run_regulatory_benchmark_task.run(run_id=12, tenant_id="tenant")

    cache.lock.assert_called_once()
    heartbeat_type.return_value.start.assert_called_once_with()
    heartbeat_type.return_value.ensure_owned.assert_called_once_with()
    run_benchmark.assert_called_once_with(db_session, 12)
    lock.release.assert_called_once_with()


def test_regulatory_benchmark_worker_only_registers_benchmark_task() -> None:
    backend_root = _backend_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(backend_root), env.get("PYTHONPATH")) if value
    )
    verification = """
import sys


blocked_import_roots = {
    "PIL",
    "docling",
    "docling_core",
    "docx",
    "markitdown",
    "openpyxl",
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

from onyx.background.celery.versioned_apps.regulatory_benchmark import app
from onyx.configs.constants import OnyxCeleryTask

app.loader.import_default_modules()
app.finalize()

known_onyx_tasks = {
    value
    for name, value in vars(OnyxCeleryTask).items()
    if name.isupper() and isinstance(value, str)
}
registered_onyx_tasks = known_onyx_tasks.intersection(app.tasks)
assert registered_onyx_tasks == {OnyxCeleryTask.REGULATORY_BENCHMARK_RUN}, (
    registered_onyx_tasks
)
assert "onyx.background.celery.tasks.scheduled_tasks.tasks" not in sys.modules
assert "onyx.regulatory.benchmark.runner" not in sys.modules
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


def test_full_and_lite_supervisors_consume_regulatory_benchmark_queue() -> None:
    backend_root = _backend_root()
    expected_program = "[program:celery_worker_regulatory_benchmark]"
    expected_app = (
        "-A onyx.background.celery.versioned_apps.regulatory_benchmark worker"
    )

    for config_name in ("supervisord.conf", "supervisord-lite.conf"):
        config = (backend_root / config_name).read_text(encoding="utf-8")
        assert expected_program in config
        assert expected_app in config
        assert "-Q regulatory_benchmark" in config

    lite_config = (backend_root / "supervisord-lite.conf").read_text(encoding="utf-8")
    assert "celery_worker_scheduled_tasks" not in lite_config
    assert "celery_worker_user_file_processing" not in lite_config
    assert "celery_beat" not in lite_config
    assert "elasticsearch_migration" not in lite_config
