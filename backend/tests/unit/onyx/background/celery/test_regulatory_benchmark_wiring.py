import datetime
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
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
from onyx.db.enums import BenchmarkRunItemStatus, BenchmarkRunStatus, Permission
from onyx.db.models import BenchmarkRun
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory import benchmark_api
from onyx.server.features.regulatory.benchmark_models import (
    BenchmarkModelSelection,
    BenchmarkRunCreate,
    BenchmarkRunItemSnapshot,
    benchmark_run_snapshot,
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
    assert run.status == BenchmarkRunStatus.QUEUED.value
    assert run.queued_at is not None
    assert run.started_at is None
    assert run.heartbeat_at is None
    assert run.failure_code is None
    assert run.failure_message is None
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
def test_concurrent_start_observes_locked_queued_state_and_does_not_republish(
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


def test_error_run_retry_resets_failed_items_and_preserves_completed_items() -> None:
    failed_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.ERROR.value,
        final_result="failed answer",
        error_message="candidate failed",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        duration_ms=40,
        cost_cents=0.5,
        cost_source="measured",
        cited_chunk_ids=["chunk-1"],
        cited_sources=[{"document_id": "doc-1"}],
        execution_steps=[{"kind": "answer"}],
        llm_calls=[{"model": "candidate"}],
        answer_reasoning="reasoning",
        chat_session_id="chat-id",
        assistant_message_id=99,
        citation_recall=0.5,
        citation_precision=0.25,
        judge_error="judge failed",
        started_at=MagicMock(),
        completed_at=MagicMock(),
        judgment=MagicMock(),
    )
    completed_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        final_result="completed answer",
        completed_at=MagicMock(),
    )
    run = SimpleNamespace(
        id=42,
        status=BenchmarkRunStatus.ERROR.value,
        items=[failed_item, completed_item],
        started_at=MagicMock(),
        completed_at=MagicMock(),
        completed_items=1,
        failed_items=1,
        report={"summary": "stale"},
        report_error="startup failed",
        report_input_tokens=11,
        report_output_tokens=12,
        report_cost_cents=0.7,
    )

    with (
        patch.object(benchmark_api, "get_benchmark_run_for_update", return_value=run),
        patch.object(benchmark_api, "_enqueue_benchmark_run"),
        patch.object(benchmark_api, "benchmark_run_snapshot", return_value=MagicMock()),
    ):
        benchmark_api.start_run(42, user=MagicMock(), db_session=MagicMock())

    assert run.status == BenchmarkRunStatus.QUEUED.value
    assert run.queued_at is not None
    assert run.started_at is None
    assert run.heartbeat_at is None
    assert run.failure_code is None
    assert run.failure_message is None
    assert run.completed_at is None
    assert run.completed_items == 1
    assert run.failed_items == 0
    assert run.report is None
    assert run.report_error is None
    assert run.report_input_tokens is None
    assert run.report_output_tokens is None
    assert run.report_cost_cents is None
    assert failed_item.status == BenchmarkRunItemStatus.PENDING.value
    assert failed_item.final_result is None
    assert failed_item.error_message is None
    assert failed_item.input_tokens is None
    assert failed_item.output_tokens is None
    assert failed_item.total_tokens is None
    assert failed_item.duration_ms is None
    assert failed_item.cost_cents is None
    assert failed_item.cost_source == "unavailable"
    assert failed_item.cited_chunk_ids == []
    assert failed_item.cited_sources == []
    assert failed_item.execution_steps == []
    assert failed_item.llm_calls == []
    assert failed_item.answer_reasoning is None
    assert failed_item.chat_session_id is None
    assert failed_item.assistant_message_id is None
    assert failed_item.citation_recall is None
    assert failed_item.citation_precision is None
    assert failed_item.judge_error is None
    assert failed_item.started_at is None
    assert failed_item.completed_at is None
    assert failed_item.judgment is None
    assert completed_item.status == BenchmarkRunItemStatus.COMPLETED.value
    assert completed_item.final_result == "completed answer"


def test_start_publish_failure_restores_pending_state() -> None:
    run = SimpleNamespace(
        id=42,
        status=BenchmarkRunStatus.PENDING.value,
        queued_at=None,
        started_at=None,
        heartbeat_at=None,
        completed_at=None,
        failure_code=None,
        failure_message=None,
        items=[],
    )
    db_session = MagicMock()

    with (
        patch.object(benchmark_api, "get_benchmark_run_for_update", return_value=run),
        patch.object(
            benchmark_api,
            "_enqueue_benchmark_run",
            side_effect=RuntimeError("broker unavailable"),
        ),
        pytest.raises(OnyxError, match="Failed to queue benchmark run"),
    ):
        benchmark_api.start_run(42, user=MagicMock(), db_session=db_session)

    assert run.status == BenchmarkRunStatus.PENDING.value
    assert run.queued_at is None
    assert run.started_at is None
    assert run.heartbeat_at is None
    assert run.completed_at is None
    assert run.failure_code == "dispatch_failed"
    assert run.failure_message == "Failed to queue benchmark run"


def test_benchmark_models_exclude_hidden_openrouter_configurations() -> None:
    visible = SimpleNamespace(
        name="visible/model",
        custom_display_name=None,
        display_name="Visible",
        max_input_tokens=100,
        is_visible=True,
    )
    hidden = SimpleNamespace(
        name="hidden/model",
        custom_display_name=None,
        display_name="Hidden",
        max_input_tokens=100,
        is_visible=False,
    )
    provider = SimpleNamespace(
        provider="openrouter",
        name="OpenRouter Prod",
        id=7,
        model_configurations=[hidden, visible],
    )

    with patch.object(
        benchmark_api, "fetch_existing_llm_providers", return_value=[provider]
    ):
        models = benchmark_api.list_models(user=MagicMock(), db_session=MagicMock())

    assert [(model.provider, model.model_id) for model in models] == [
        ("OpenRouter Prod", "visible/model")
    ]


def test_benchmark_models_include_nameless_openrouter_provider() -> None:
    visible = SimpleNamespace(
        name="openai/gpt-5-mini",
        custom_display_name=None,
        display_name="GPT-5 mini",
        max_input_tokens=100,
        is_visible=True,
    )
    provider = SimpleNamespace(
        provider="openrouter",
        name=None,
        id=8,
        model_configurations=[visible],
    )

    with patch.object(
        benchmark_api, "fetch_existing_llm_providers", return_value=[provider]
    ):
        models = benchmark_api.list_models(user=MagicMock(), db_session=MagicMock())

    assert [
        (model.provider, model.provider_id, model.model_id) for model in models
    ] == [("openrouter", 8, "openai/gpt-5-mini")]


def test_benchmark_models_give_each_nameless_provider_a_stable_selector() -> None:
    configuration = SimpleNamespace(
        name="visible/model",
        custom_display_name=None,
        display_name="Visible",
        max_input_tokens=100,
        is_visible=True,
    )
    named_collision = SimpleNamespace(
        provider="openrouter",
        name="openrouter",
        id=7,
        model_configurations=[configuration],
    )
    first_nameless = SimpleNamespace(
        provider="openrouter",
        name=None,
        id=8,
        model_configurations=[configuration],
    )
    second_nameless = SimpleNamespace(
        provider="openrouter",
        name=None,
        id=9,
        model_configurations=[configuration],
    )

    with patch.object(
        benchmark_api,
        "fetch_existing_llm_providers",
        return_value=[named_collision, first_nameless, second_nameless],
    ):
        models = benchmark_api.list_models(user=MagicMock(), db_session=MagicMock())

    assert [(model.provider, model.provider_id) for model in models] == [
        ("openrouter", 7),
        ("openrouter", 8),
        ("openrouter", 9),
    ]


def test_nameless_openrouter_selector_cannot_be_captured_by_named_collision() -> None:
    configuration = SimpleNamespace(
        name="visible/model",
        custom_display_name=None,
        display_name="Visible",
        max_input_tokens=100,
        is_visible=True,
    )
    named_collision = SimpleNamespace(
        provider="openrouter",
        name="openrouter",
        id=7,
        model_configurations=[configuration],
    )
    nameless = SimpleNamespace(
        provider="openrouter",
        name=None,
        id=8,
        model_configurations=[configuration],
    )

    with patch.object(
        benchmark_api,
        "fetch_existing_llm_providers",
        return_value=[named_collision, nameless],
    ):
        models = benchmark_api.list_models(user=MagicMock(), db_session=MagicMock())

    assert [(model.provider, model.provider_id) for model in models] == [
        ("openrouter", 7),
        ("openrouter", 8),
    ]


def test_benchmark_model_validation_rejects_hidden_configuration() -> None:
    hidden = SimpleNamespace(name="hidden/model", is_visible=False)
    provider = SimpleNamespace(
        provider="openrouter",
        name="OpenRouter Prod",
        id=7,
        model_configurations=[hidden],
    )
    selection = BenchmarkModelSelection(
        provider="OpenRouter Prod", model_id="hidden/model"
    )

    with (
        patch.object(
            benchmark_api,
            "fetch_existing_llm_provider_by_name_and_type",
            return_value=provider,
        ),
        pytest.raises(OnyxError, match="not available through OpenRouter"),
    ):
        benchmark_api._validate_model(MagicMock(), selection)


def test_benchmark_model_validation_resolves_nameless_selector_by_provider_id() -> None:
    visible = SimpleNamespace(name="visible/model", is_visible=True)
    provider = SimpleNamespace(
        provider="openrouter",
        name=None,
        id=8,
        model_configurations=[visible],
    )
    selection = BenchmarkModelSelection(
        provider="openrouter", provider_id=8, model_id="visible/model"
    )

    db_session = MagicMock()
    with (
        patch.object(
            benchmark_api,
            "fetch_existing_llm_provider_by_id",
            return_value=provider,
            create=True,
        ) as resolve_provider,
    ):
        benchmark_api._validate_model(db_session, selection)

    resolve_provider.assert_called_once_with(8, db_session)


def test_provider_id_allows_a_named_provider_that_looks_like_old_reserved_syntax() -> (
    None
):
    named_provider = SimpleNamespace(
        provider="openrouter",
        name="openrouter::8",
        id=8,
        model_configurations=[SimpleNamespace(name="visible/model", is_visible=True)],
    )
    selection = BenchmarkModelSelection(
        provider="openrouter::8", provider_id=8, model_id="visible/model"
    )

    with patch.object(
        benchmark_api,
        "fetch_existing_llm_provider_by_id",
        return_value=named_provider,
    ):
        resolved = benchmark_api._validate_model(MagicMock(), selection)

    assert resolved is named_provider


def test_run_creation_persists_candidate_and_judge_provider_ids() -> None:
    candidate_provider = SimpleNamespace(id=7, name=None, provider="openrouter")
    judge_provider = SimpleNamespace(id=8, name="Judge", provider="openrouter")
    candidate = BenchmarkModelSelection(
        provider="openrouter", provider_id=7, model_id="candidate/model"
    )
    judge = BenchmarkModelSelection(
        provider="Judge", provider_id=8, model_id="judge/model"
    )
    request = BenchmarkRunCreate(candidates=[candidate], judge=judge)
    question = SimpleNamespace(id=11, document_set_id=3)
    created_run = MagicMock()

    with (
        patch.object(
            benchmark_api,
            "_validate_model",
            side_effect=[candidate_provider, judge_provider],
        ),
        patch.object(
            benchmark_api, "list_benchmark_questions", return_value=[question]
        ),
        patch.object(benchmark_api, "_get_editable_document_set"),
        patch.object(
            benchmark_api, "create_benchmark_run", return_value=created_run
        ) as create_run,
        patch.object(benchmark_api, "benchmark_run_snapshot", return_value=MagicMock()),
    ):
        benchmark_api.create_run(request, user=MagicMock(), db_session=MagicMock())

    assert create_run.call_args.kwargs["judge_provider_id"] == 8
    assert create_run.call_args.kwargs["candidates"] == [
        ("openrouter", 7, "candidate/model")
    ]


def test_run_snapshot_keeps_same_named_models_separate_by_provider_id() -> None:
    def item(provider_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            provider="openrouter",
            provider_id=provider_id,
            model_id="shared/model",
            judgment=None,
            total_tokens=None,
            duration_ms=None,
            cost_cents=None,
            citation_recall=None,
            citation_precision=None,
            status="completed",
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    run = SimpleNamespace(
        id=1,
        label=None,
        status="completed",
        judge_provider="openrouter",
        judge_provider_id=10,
        judge_model="judge/model",
        deep_research=False,
        total_items=2,
        completed_items=2,
        failed_items=0,
        queued_at=now,
        started_at=now,
        heartbeat_at=now,
        completed_at=now,
        created_at=now,
        failure_code=None,
        failure_message=None,
        report=None,
        report_error=None,
        report_input_tokens=None,
        report_output_tokens=None,
        report_cost_cents=None,
        items=[item(8), item(9)],
    )
    empty_item_snapshot = BenchmarkRunItemSnapshot.model_construct()

    with patch.object(
        BenchmarkRunItemSnapshot,
        "from_model",
        return_value=empty_item_snapshot,
    ):
        snapshot = benchmark_run_snapshot(cast(BenchmarkRun, run))

    assert [
        (aggregate.provider_id, aggregate.item_count)
        for aggregate in snapshot.aggregates
    ] == [
        (8, 1),
        (9, 1),
    ]


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
    job = MagicMock()
    client = MagicMock()
    client.submit.return_value = job
    with (
        patch.object(benchmark_tasks, "get_cache_backend", return_value=cache),
        patch.object(benchmark_tasks, "_RunLeaseHeartbeat") as heartbeat_type,
        patch.object(benchmark_tasks, "_prepare_benchmark_items", return_value=[34]),
        patch.object(benchmark_tasks, "_run_benchmark_items", return_value=False),
        patch.object(
            benchmark_tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.RUNNING.value,
        ),
        patch.object(benchmark_tasks, "SimpleJobClient", return_value=client),
        patch.object(benchmark_tasks, "_monitor_finalization_job") as monitor,
    ):
        benchmark_tasks.run_regulatory_benchmark_task.run(run_id=12, tenant_id="tenant")

    cache.lock.assert_called_once()
    heartbeat_type.return_value.start.assert_called_once_with()
    heartbeat_type.return_value.ensure_owned.assert_called_once_with()
    client.submit.assert_called_once_with(
        benchmark_tasks._execute_benchmark_finalization,
        12,
        "tenant",
        False,
    )
    monitor.assert_called_once_with(
        job, run_id=12, heartbeat=heartbeat_type.return_value
    )
    lock.release.assert_called_once_with()


def test_benchmark_task_persists_startup_crash_and_terminalizes_items() -> None:
    pending_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.PENDING.value,
        error_message=None,
        completed_at=None,
    )
    running_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.RUNNING.value,
        error_message=None,
        completed_at=None,
    )
    completed_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        error_message=None,
        completed_at=MagicMock(),
    )
    run = SimpleNamespace(
        id=12,
        status=BenchmarkRunStatus.RUNNING.value,
        report_error=None,
        failure_code=None,
        failure_message=None,
        heartbeat_at=None,
        completed_at=None,
        completed_items=1,
        failed_items=0,
        items=[pending_item, running_item, completed_item],
    )
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
        patch.object(benchmark_tasks, "_RunLeaseHeartbeat"),
        patch.object(
            benchmark_tasks,
            "_prepare_benchmark_items",
            side_effect=RuntimeError("persona selection exploded"),
        ),
        patch.object(
            benchmark_tasks,
            "get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.db.regulatory_benchmark.get_benchmark_run_for_update",
            return_value=run,
        ),
        pytest.raises(RuntimeError, match="persona selection exploded"),
    ):
        benchmark_tasks.run_regulatory_benchmark_task.run(run_id=12, tenant_id="tenant")

    assert run.status == BenchmarkRunStatus.ERROR.value
    assert run.report_error is None
    assert run.failure_code == "execution_failed"
    assert run.failure_message == "persona selection exploded"
    assert run.completed_items == 1
    assert run.failed_items == 2
    assert pending_item.status == BenchmarkRunItemStatus.ERROR.value
    assert running_item.status == BenchmarkRunItemStatus.ERROR.value
    assert pending_item.error_message == "persona selection exploded"
    assert running_item.error_message == "persona selection exploded"
    assert pending_item.completed_at == run.completed_at
    assert running_item.completed_at == run.completed_at
    assert completed_item.status == BenchmarkRunItemStatus.COMPLETED.value


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
