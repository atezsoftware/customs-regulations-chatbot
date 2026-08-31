from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from uuid import UUID

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


def test_approval_task_commits_version_before_durable_indexing() -> None:
    proposal = SimpleNamespace(
        id=9,
        status="approving",
        applied_new_chunk_id=None,
    )
    user_file_id = UUID("00000000-0000-0000-0000-000000000111")
    result = SimpleNamespace(new_chunk=SimpleNamespace(user_file_id=user_file_id))
    job_id = UUID("00000000-0000-0000-0000-000000000909")
    apply_session = MagicMock()
    prepare_session = MagicMock()
    link_session = MagicMock()
    events: list[str] = []
    apply_session.commit.side_effect = lambda: events.append("version-commit")
    link_session.commit.side_effect = lambda: events.append("link-commit")

    session_contexts = []
    for session in (apply_session, prepare_session, link_session):
        context = MagicMock()
        context.__enter__.return_value = session
        session_contexts.append(context)

    with (
        patch.object(tasks, "get_session_with_current_tenant") as session_factory,
        patch.object(tasks, "get_proposal", return_value=proposal),
        patch.object(
            tasks, "approve_amendment_proposal", return_value=result
        ) as approve,
        patch.object(
            tasks,
            "prepare_regulatory_indexing_job_from_chunks",
            return_value=job_id,
        ) as prepare,
        patch.object(
            tasks, "link_amendment_proposal_indexing_job", return_value=True
        ) as link,
        patch.object(tasks, "enqueue_prepared_regulatory_indexing_job") as enqueue,
    ):
        session_factory.side_effect = session_contexts
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    approve.assert_called_once_with(apply_session, proposal)
    prepare.assert_called_once_with(user_file_id, "tenant-a", prepare_session)
    link.assert_called_once_with(
        link_session,
        proposal_id=9,
        job_id=job_id,
    )
    enqueue.assert_called_once_with(job_id=job_id, tenant_id="tenant-a")
    assert events == ["version-commit", "link-commit"]


def test_approval_task_does_not_use_locked_full_file_projection() -> None:
    assert not hasattr(tasks, "project_user_file_to_index")


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


def test_recovery_redelivers_applied_unlinked_approvals() -> None:
    batch_context = MagicMock()
    batch_context.__enter__.return_value = MagicMock()
    proposal_context = MagicMock()
    proposal_context.__enter__.return_value = MagicMock()

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            side_effect=[batch_context, proposal_context],
        ),
        patch.object(tasks, "claim_stale_batches_for_recovery", return_value=[42]),
        patch.object(
            tasks,
            "recover_stale_amendment_proposal_approvals",
            return_value=[9],
        ),
        patch.object(tasks, "enqueue_amendment_batch") as enqueue_batch,
        patch.object(tasks, "enqueue_amendment_proposal_approval") as enqueue_approval,
    ):
        tasks.regulatory_amendment_recover_stale.run(tenant_id="tenant-a")

    enqueue_batch.assert_called_once()
    enqueue_approval.assert_called_once()
    assert enqueue_batch.call_args.kwargs == {
        "batch_id": 42,
        "tenant_id": "tenant-a",
    }
    assert enqueue_approval.call_args.kwargs == {
        "proposal_id": 9,
        "tenant_id": "tenant-a",
    }


def test_runtime_lite_worker_consumes_amendment_queue() -> None:
    backend_root = Path(__file__).resolve().parents[5]
    supervisor = (backend_root / "supervisord-lite.conf").read_text(encoding="utf-8")

    assert "-Q regulatory_benchmark,regulatory_amendment" in supervisor
