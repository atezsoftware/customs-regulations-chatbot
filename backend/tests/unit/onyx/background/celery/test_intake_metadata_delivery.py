"""Metadata recovery is a scoped hint to the metadata-only worker entry point."""

from contextlib import nullcontext
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.regulatory_amendments import document_intake as tasks
from onyx.background.celery.tasks.regulatory_labeling import tasks as labeling_tasks
from onyx.background.celery.versioned_apps.client import app
from onyx.db import document_intake as repository
from onyx.db.engine import sql_engine
from onyx.regulatory.document_intake import metadata_resume_job
from shared_configs import configs, contextvars


@pytest.fixture(autouse=True)
def scoped_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks.config, "REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setattr(tasks.config, "ANNEX_DATABASE_IDENTITY", "owned-db")
    monkeypatch.setattr(contextvars, "get_current_tenant_id", lambda: "public")


def test_metadata_hint_uses_normal_scoped_queue_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = MagicMock()
    monkeypatch.setattr(app, "send_task", send)
    monkeypatch.setattr(tasks.config, "source_queue_name", lambda: "owned-source-queue")
    identifier = uuid4()
    tasks.enqueue_intake_metadata_recovery(batch_id=identifier, tenant_id="public")
    send.assert_called_once_with(
        tasks.METADATA_RECOVERY_TASK_NAME,
        kwargs={
            "batch_id": str(identifier),
            "tenant_id": "public",
            "environment": "dev",
            "database_identity": "owned-db",
        },
        queue="owned-source-queue",
        expires=900,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"environment": "prod"},
        {"database_identity": "foreign"},
        {"tenant_id": "other"},
        {"tenant_id": ""},
    ],
)
def test_foreign_metadata_hint_never_executes(
    monkeypatch: pytest.MonkeyPatch, updates: dict[str, str]
) -> None:
    resume = MagicMock()
    monkeypatch.setattr(
        metadata_resume_job, "run_document_intake_metadata_recovery", resume
    )
    with pytest.raises(ValueError, match="another environment or database"):
        tasks.recover_document_intake_metadata.run(
            **{
                "batch_id": str(uuid4()),
                "tenant_id": "public",
                "environment": "dev",
                "database_identity": "owned-db",
                **updates,
            }
        )
    resume.assert_not_called()


@pytest.mark.parametrize("created", [False, True])
def test_metadata_only_worker_dispatches_only_its_result(
    monkeypatch: pytest.MonkeyPatch, created: bool
) -> None:
    identifier, run_id = uuid4(), uuid4()
    resume, send = MagicMock(return_value=run_id if created else None), MagicMock()
    monkeypatch.setattr(
        metadata_resume_job, "run_document_intake_metadata_recovery", resume
    )
    monkeypatch.setattr(labeling_tasks, "enqueue_labeling_run", send)
    tasks.recover_document_intake_metadata.run(
        batch_id=str(identifier),
        tenant_id="public",
        environment="dev",
        database_identity="owned-db",
    )
    resume.assert_called_once_with(intake_id=identifier, tenant_id="public")
    if created:
        send.assert_called_once_with(run_id=run_id, tenant_id="public")
    else:
        send.assert_not_called()


def test_lost_labeling_send_preserves_normal_durable_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier, run_id = uuid4(), uuid4()
    session, mark = MagicMock(), MagicMock()
    monkeypatch.setattr(
        metadata_resume_job,
        "run_document_intake_metadata_recovery",
        lambda **_k: run_id,
    )
    monkeypatch.setattr(
        labeling_tasks,
        "enqueue_labeling_run",
        MagicMock(side_effect=RuntimeError("broker unavailable")),
    )
    monkeypatch.setattr(repository, "mark_intake_label_dispatch_failed", mark)
    monkeypatch.setattr(
        sql_engine, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    with pytest.raises(RuntimeError, match="broker unavailable"):
        tasks.recover_document_intake_metadata.run(
            batch_id=str(identifier),
            tenant_id="public",
            environment="dev",
            database_identity="owned-db",
        )
    mark.assert_called_once_with(session, identifier)
    session.commit.assert_called_once_with()


def test_self_hosted_foreign_schema_never_executes_even_when_context_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = MagicMock()
    monkeypatch.setattr(
        metadata_resume_job, "run_document_intake_metadata_recovery", resume
    )
    monkeypatch.setattr(configs, "MULTI_TENANT", False)
    monkeypatch.setattr(configs, "POSTGRES_DEFAULT_SCHEMA", "public")
    monkeypatch.setattr(contextvars, "get_current_tenant_id", lambda: "foreign")
    with pytest.raises(ValueError, match="another environment or database"):
        tasks.recover_document_intake_metadata.run(
            batch_id=str(uuid4()),
            tenant_id="foreign",
            environment="dev",
            database_identity="owned-db",
        )
    resume.assert_not_called()
