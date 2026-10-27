"""Short recovery reserves first, keeps scope, and survives lost broker hints."""

import ast
import importlib
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from onyx.background.celery.tasks.regulatory_intake_recovery import tasks
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues


@dataclass(frozen=True)
class _Delivery:
    batch_id: UUID
    reprepare_request_id: UUID | None = None


@dataclass
class _Harness:
    session: MagicMock
    session_factory: MagicMock
    reserve: MagicMock
    resume: MagicMock
    prepare_send: MagicMock
    label_send: MagicMock
    events: list[str]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(tasks.config, "REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setattr(tasks.config, "ANNEX_DATABASE_IDENTITY", "owned-db")
    monkeypatch.setattr(tasks.config, "REGULATORY_ANNEX_WORKER_ENABLED", True)
    monkeypatch.setattr(tasks, "MULTI_TENANT", False)
    monkeypatch.setattr(tasks, "POSTGRES_DEFAULT_SCHEMA", "public")
    monkeypatch.setattr(tasks, "get_current_tenant_id", lambda: "public")
    session = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(tasks, "get_session_with_current_tenant", factory)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    reserve, resume = MagicMock(return_value=[]), MagicMock(return_value=[])
    db_module = ModuleType("onyx.db.document_intake_queue")
    monkeypatch.setattr(
        db_module, "reserve_intake_preparation_deliveries", reserve, raising=False
    )
    monkeypatch.setattr(
        db_module, "resume_waiting_intake_labeling", resume, raising=False
    )
    monkeypatch.setitem(sys.modules, db_module.__name__, db_module)
    prepare_send, label_send = MagicMock(), MagicMock()
    for module_name, symbol, send in (
        (
            "onyx.background.celery.tasks.regulatory_amendments.document_intake",
            "enqueue_intake_batch",
            prepare_send,
        ),
        (
            "onyx.background.celery.tasks.regulatory_labeling.tasks",
            "enqueue_labeling_run",
            label_send,
        ),
    ):
        module = ModuleType(module_name)
        monkeypatch.setattr(module, symbol, send, raising=False)
        monkeypatch.setitem(sys.modules, module_name, module)
    return _Harness(session, factory, reserve, resume, prepare_send, label_send, events)


def _recover(**updates: str) -> None:
    tasks.recover_document_intakes.run(
        **{
            "tenant_id": "public",
            "environment": "dev",
            "database_identity": "owned-db",
            **updates,
        }
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"tenant_id": ""},
        {"tenant_id": "other"},
        {"environment": "prod"},
        {"database_identity": "foreign-db"},
    ],
)
def test_wrong_scope_never_opens_database(
    harness: _Harness, updates: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="scope mismatch"):
        _recover(**updates)
    harness.session_factory.assert_not_called()


def test_self_hosted_nondefault_schema_is_rejected_before_database(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "get_current_tenant_id", lambda: "other")
    with pytest.raises(ValueError, match="scope mismatch"):
        _recover(tenant_id="other")
    harness.session_factory.assert_not_called()


def test_disabled_source_worker_never_reserves(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks.config, "REGULATORY_ANNEX_WORKER_ENABLED", False)
    _recover()
    harness.session_factory.assert_not_called()


def test_reservations_commit_before_each_scoped_send(harness: _Harness) -> None:
    delivery = _Delivery(uuid4(), uuid4())
    run_id = uuid4()
    harness.reserve.return_value = [delivery]
    harness.resume.return_value = [run_id]

    def sent(**_kwargs: object) -> None:
        assert harness.events == ["commit"]

    harness.prepare_send.side_effect = sent
    harness.label_send.side_effect = sent
    _recover()
    harness.reserve.assert_called_once_with(
        harness.session, environment="dev", limit=10
    )
    harness.resume.assert_called_once_with(harness.session, environment="dev", limit=10)
    harness.session.commit.assert_called_once_with()
    harness.prepare_send.assert_called_once_with(
        batch_id=delivery.batch_id,
        reprepare_request_id=delivery.reprepare_request_id,
        tenant_id="public",
    )
    harness.label_send.assert_called_once_with(run_id=run_id, tenant_id="public")


def test_one_broker_failure_does_not_fail_or_drop_other_reservations(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    deliveries = [_Delivery(uuid4()), _Delivery(uuid4(), uuid4())]
    run_ids = [uuid4(), uuid4()]
    harness.reserve.return_value = deliveries
    harness.resume.return_value = run_ids
    harness.prepare_send.side_effect = [RuntimeError("broker unavailable"), None]
    harness.label_send.side_effect = [RuntimeError("broker unavailable"), None]
    log = MagicMock()
    monkeypatch.setattr(tasks, "logger", log)
    _recover()
    assert harness.prepare_send.call_count == 2
    assert harness.label_send.call_count == 2
    assert log.warning.call_count == 2
    assert harness.events == ["commit"]
    harness.session.execute.assert_not_called()


def test_database_failure_does_not_commit_or_emit_any_hint(harness: _Harness) -> None:
    harness.reserve.return_value = [_Delivery(uuid4())]
    harness.resume.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        _recover()
    harness.session.commit.assert_not_called()
    harness.prepare_send.assert_not_called()
    harness.label_send.assert_not_called()


def test_no_reserved_work_emits_no_delivery(harness: _Harness) -> None:
    _recover()
    harness.session.commit.assert_called_once_with()
    harness.prepare_send.assert_not_called()
    harness.label_send.assert_not_called()


def test_full_and_lite_schedules_use_short_light_queue_and_frozen_scope() -> None:
    from onyx.background.celery.tasks import beat_schedule
    from onyx.background.celery.tasks.regulatory_indexing.beat_schedule import (
        PRODUCTION_LITE_TASK_TEMPLATES,
    )

    for entries in (beat_schedule.beat_task_templates, PRODUCTION_LITE_TASK_TEMPLATES):
        matching = [e for e in entries if e["task"] == "recover_document_intakes"]
        assert len(matching) == 1
        entry = matching[0]
        assert entry["schedule"] == timedelta(seconds=60)
        assert entry["options"] == {
            "queue": OnyxCeleryQueues.CHECKPOINT_CLEANUP,
            "priority": OnyxCeleryPriority.LOW,
            "expires": 60,
        }
        assert entry["kwargs"] == {
            "environment": tasks.config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": tasks.config.ANNEX_DATABASE_IDENTITY,
        }


def test_full_scheduler_preserves_scope_kwargs_and_overrides_template_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.apps import beat

    template = {
        "name": "owned-recovery",
        "task": "recover_document_intakes",
        "schedule": timedelta(seconds=60),
        "kwargs": {
            "environment": "dev",
            "database_identity": "owned-db",
            "tenant_id": "untrusted-template",
        },
    }
    monkeypatch.setattr(beat, "MULTI_TENANT", False)
    monkeypatch.setattr(beat, "IGNORED_SYNCING_TENANT_LIST", [])
    monkeypatch.setattr(
        beat, "fetch_versioned_implementation", lambda *_args: lambda: [template]
    )
    scheduler = object.__new__(beat.DynamicTenantScheduler)
    schedule = scheduler._generate_schedule(["public"], 1)
    assert schedule["owned-recovery-public"]["kwargs"] == {
        "environment": "dev",
        "database_identity": "owned-db",
        "tenant_id": "public",
    }
    assert template["kwargs"]["tenant_id"] == "untrusted-template"


def test_cloud_generator_contract_does_not_forward_new_scope_fields() -> None:
    from onyx.background.celery.tasks.beat_schedule import make_cloud_generator_task

    generated = make_cloud_generator_task(
        {
            "name": "existing-task",
            "task": "existing_task",
            "schedule": timedelta(seconds=60),
            "kwargs": {"environment": "dev", "database_identity": "owned-db"},
            "options": {"queue": "existing-queue", "expires": 60},
        }
    )
    assert generated["kwargs"] == {
        "task_name": "existing_task",
        "queue": "existing-queue",
        "expires": 60,
    }


def test_full_cloud_schedule_does_not_add_an_unscoped_intake_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.tasks import beat_schedule
    from shared_configs import configs

    try:
        with monkeypatch.context() as context:
            context.setattr(configs, "MULTI_TENANT", True)
            cloud_schedule = importlib.reload(beat_schedule)
            assert not any(
                entry["task"] == "recover_document_intakes"
                for entry in cloud_schedule.beat_task_templates
            )
    finally:
        importlib.reload(beat_schedule)


def test_light_discovers_only_the_new_short_recovery_package() -> None:
    backend = Path(__file__).resolve().parents[5]
    source = (backend / "onyx/background/celery/apps/light.py").read_text()
    tree = ast.parse(source)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "onyx.background.celery.tasks.regulatory_intake_recovery" in strings
    assert "onyx.background.celery.tasks.regulatory_amendments" not in strings
    assert "onyx.background.celery.tasks.regulatory_labeling" not in strings
