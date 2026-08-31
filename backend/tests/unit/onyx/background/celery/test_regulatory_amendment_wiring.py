from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from onyx.background.celery.tasks.regulatory_amendments import tasks
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues, OnyxCeleryTask


def test_enqueue_routes_redundant_expiring_deliveries() -> None:
    app = MagicMock()

    tasks.enqueue_amendment_batch(app, batch_id=42, tenant_id="public")

    expected = call(
        OnyxCeleryTask.REGULATORY_AMENDMENT_RUN,
        kwargs={"batch_id": 42, "tenant_id": "public"},
        queue=OnyxCeleryQueues.REGULATORY_AMENDMENT,
        priority=OnyxCeleryPriority.HIGH,
        expires=24 * 60 * 60,
        retry=False,
    )
    delayed = call(
        OnyxCeleryTask.REGULATORY_AMENDMENT_RUN,
        kwargs={"batch_id": 42, "tenant_id": "public"},
        queue=OnyxCeleryQueues.REGULATORY_AMENDMENT,
        priority=OnyxCeleryPriority.HIGH,
        expires=24 * 60 * 60,
        retry=False,
        countdown=5,
    )
    assert app.send_task.call_args_list == [expected, delayed]


def test_enqueue_approval_routes_expiring_background_delivery() -> None:
    app = MagicMock()

    tasks.enqueue_amendment_proposal_approval(
        app,
        proposal_id=9,
        tenant_id="public",
    )

    app.send_task.assert_called_once_with(
        OnyxCeleryTask.REGULATORY_AMENDMENT_APPROVE,
        kwargs={"proposal_id": 9, "tenant_id": "public"},
        queue=OnyxCeleryQueues.REGULATORY_AMENDMENT,
        priority=OnyxCeleryPriority.HIGH,
        expires=24 * 60 * 60,
        retry=False,
    )


def test_run_task_claims_and_executes_batch() -> None:
    lease = MagicMock(generation=7)
    with (
        patch.object(tasks, "get_session_with_current_tenant") as session_factory,
        patch.object(tasks, "claim_batch_for_analysis", return_value=lease),
        patch.object(tasks, "run_amendment_batch") as run_batch,
    ):
        session_factory.return_value.__enter__.return_value = MagicMock()
        tasks.regulatory_amendment_run.run(batch_id=42, tenant_id="public")

    run_batch.assert_called_once_with(batch_id=42, lease_generation=7)


def test_approval_task_projects_affected_file_before_committing() -> None:
    proposal = SimpleNamespace(id=9, status="approving")
    result = SimpleNamespace(new_chunk=SimpleNamespace(user_file_id="file-id"))
    user_file = SimpleNamespace(id="file-id")
    db_session = MagicMock()
    db_session.get.return_value = user_file

    with (
        patch.object(tasks, "get_session_with_current_tenant") as session_factory,
        patch.object(tasks, "get_proposal", return_value=proposal),
        patch.object(
            tasks, "approve_amendment_proposal", return_value=result
        ) as approve,
        patch.object(tasks, "project_user_file_to_index") as project,
        patch.object(tasks, "get_current_tenant_id", return_value="tenant-a"),
    ):
        session_factory.return_value.__enter__.return_value = db_session
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    approve.assert_called_once_with(db_session, proposal)
    project.assert_called_once_with(db_session, user_file, "tenant-a")
    db_session.commit.assert_called_once_with()


def test_lease_watchdog_renews_heartbeat_with_tenant_context() -> None:
    stop = MagicMock()
    stop.wait.side_effect = [False, True]
    thread = MagicMock()

    def build_thread(*, target, daemon):  # noqa: ANN001, ANN202, ARG001
        thread.start.side_effect = target
        return thread

    with (
        patch.object(tasks, "Event", return_value=stop),
        patch.object(tasks, "Thread", side_effect=build_thread),
        patch.object(tasks, "get_session_with_current_tenant") as session_factory,
        patch.object(tasks, "touch_batch_heartbeat", return_value=True) as touch,
    ):
        heartbeat_session = MagicMock()
        session_factory.return_value.__enter__.return_value = heartbeat_session
        with tasks._renew_batch_lease(batch_id=42, lease_generation=7):
            pass

    touch.assert_called_once_with(
        heartbeat_session,
        batch_id=42,
        lease_generation=7,
    )
    stop.set.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=2)


def test_runtime_lite_worker_consumes_amendment_queue() -> None:
    backend_root = Path(__file__).resolve().parents[5]
    supervisor = (backend_root / "supervisord-lite.conf").read_text(encoding="utf-8")

    assert "-Q regulatory_benchmark,regulatory_amendment" in supervisor
