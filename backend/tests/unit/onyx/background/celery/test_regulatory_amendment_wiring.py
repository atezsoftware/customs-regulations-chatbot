from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from uuid import UUID

import pytest

from onyx.background.celery.tasks.regulatory_amendments import tasks
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues, OnyxCeleryTask
from shared_configs.enums import EmbeddingProvider


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


def test_approval_task_commits_version_before_active_index_projection() -> None:
    proposal = SimpleNamespace(
        id=9,
        status="approving",
        applied_new_chunk_id=None,
    )
    user_file_id = UUID("00000000-0000-0000-0000-000000000111")
    result = SimpleNamespace(
        new_chunk=SimpleNamespace(id="new-chunk", user_file_id=user_file_id),
        old_chunk=SimpleNamespace(id="old-chunk"),
    )
    user_file = SimpleNamespace(id=user_file_id)
    apply_session = MagicMock()
    projection_session = MagicMock()
    projection_session.get.return_value = user_file
    events: list[str] = []
    apply_session.commit.side_effect = lambda: events.append("version-commit")
    projection_session.commit.side_effect = lambda: events.append("projection-commit")

    session_contexts = []
    for session in (apply_session, projection_session):
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
            tasks, "project_amendment_to_index", create=True
        ) as project_to_active_index,
        patch.object(
            tasks,
            "validate_amendment_projection_search_settings",
            create=True,
        ) as validate_search_settings,
        patch.object(
            tasks,
            "finalize_amendment_proposal_projection",
            return_value=True,
            create=True,
        ) as finalize,
    ):
        validate_search_settings.return_value = 11
        project_to_active_index.return_value = 461
        session_factory.side_effect = session_contexts
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    approve.assert_called_once_with(apply_session, proposal)
    assert validate_search_settings.call_args_list == [
        call(projection_session),
        call(projection_session, expected_id=11, for_update=True),
    ]
    project_to_active_index.assert_called_once_with(
        projection_session,
        user_file,
        "tenant-a",
        old_chunk_id="old-chunk",
        new_chunk_id="new-chunk",
        current_search_settings_id=11,
    )
    finalize.assert_called_once_with(
        projection_session,
        proposal_id=9,
        succeeded=True,
    )
    assert events == ["version-commit", "projection-commit"]


@pytest.mark.parametrize(
    ("provider_type", "model_name", "dimension", "expected_error"),
    [
        (
            EmbeddingProvider.OPENROUTER,
            "openai/text-embedding-3-large",
            1024,
            "Google Gemini",
        ),
        (EmbeddingProvider.GOOGLE, "text-embedding-004", 1024, "gemini-embedding-2"),
        (EmbeddingProvider.GOOGLE, "gemini-embedding-2", 2048, "1024"),
    ],
)
def test_amendment_projection_rejects_non_gemini_1024_search_settings(
    provider_type: EmbeddingProvider,
    model_name: str,
    dimension: int,
    expected_error: str,
) -> None:
    search_settings = SimpleNamespace(
        provider_type=provider_type,
        model_name=model_name,
        final_embedding_dim=dimension,
    )
    with (
        patch.object(
            tasks,
            "get_current_search_settings",
            return_value=search_settings,
            create=True,
        ),
        pytest.raises(RuntimeError, match=expected_error),
    ):
        tasks.validate_amendment_projection_search_settings(MagicMock())


def test_amendment_projection_accepts_gemini_1024_search_settings() -> None:
    search_settings = SimpleNamespace(
        id=11,
        provider_type=EmbeddingProvider.GOOGLE,
        model_name="gemini-embedding-2",
        final_embedding_dim=1024,
    )
    with patch.object(
        tasks,
        "get_current_search_settings",
        return_value=search_settings,
    ):
        assert tasks.validate_amendment_projection_search_settings(MagicMock()) == 11


def test_amendment_projection_lock_rejects_a_different_current_setting() -> None:
    search_settings = SimpleNamespace(
        id=12,
        provider_type=EmbeddingProvider.GOOGLE,
        model_name="gemini-embedding-2",
        final_embedding_dim=1024,
    )
    db_session = MagicMock()
    with (
        patch.object(
            tasks,
            "get_current_search_settings",
            return_value=search_settings,
        ) as get_settings,
        pytest.raises(RuntimeError, match="changed during amendment projection"),
    ):
        tasks.validate_amendment_projection_search_settings(
            db_session,
            expected_id=11,
            for_update=True,
        )

    get_settings.assert_called_once_with(db_session, for_update=True)


def test_approval_task_marks_projection_failure_terminal_after_version_commit() -> None:
    proposal = SimpleNamespace(id=9, status="approving", applied_new_chunk_id=None)
    user_file_id = UUID("00000000-0000-0000-0000-000000000111")
    result = SimpleNamespace(
        new_chunk=SimpleNamespace(id="new-chunk", user_file_id=user_file_id),
        old_chunk=None,
    )
    apply_session = MagicMock()
    projection_session = MagicMock()
    projection_session.get.return_value = SimpleNamespace(id=user_file_id)
    failure_session = MagicMock()
    contexts = []
    for session in (apply_session, projection_session, failure_session):
        context = MagicMock()
        context.__enter__.return_value = session
        contexts.append(context)

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            side_effect=contexts,
        ),
        patch.object(tasks, "get_proposal", return_value=proposal),
        patch.object(tasks, "approve_amendment_proposal", return_value=result),
        patch.object(tasks, "validate_amendment_projection_search_settings"),
        patch.object(
            tasks,
            "project_amendment_to_index",
            side_effect=RuntimeError("provider secret detail"),
        ),
        patch.object(
            tasks,
            "finalize_amendment_proposal_projection",
            return_value=True,
        ) as finalize,
        pytest.raises(RuntimeError, match="provider secret detail"),
    ):
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    finalize.assert_called_once_with(
        failure_session,
        proposal_id=9,
        succeeded=False,
        error_message="Indexing failed. The approval was not published.",
    )
    failure_session.commit.assert_called_once_with()


def test_approval_task_rejects_an_empty_projection() -> None:
    proposal = SimpleNamespace(id=9, status="approving", applied_new_chunk_id=None)
    user_file_id = UUID("00000000-0000-0000-0000-000000000111")
    result = SimpleNamespace(
        new_chunk=SimpleNamespace(id="new-chunk", user_file_id=user_file_id),
        old_chunk=None,
    )
    apply_session = MagicMock()
    projection_session = MagicMock()
    projection_session.get.return_value = SimpleNamespace(id=user_file_id)
    failure_session = MagicMock()
    contexts = []
    for session in (apply_session, projection_session, failure_session):
        context = MagicMock()
        context.__enter__.return_value = session
        contexts.append(context)

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            side_effect=contexts,
        ),
        patch.object(tasks, "get_proposal", return_value=proposal),
        patch.object(tasks, "approve_amendment_proposal", return_value=result),
        patch.object(
            tasks,
            "validate_amendment_projection_search_settings",
            return_value=11,
        ),
        patch.object(tasks, "project_amendment_to_index", return_value=0),
        patch.object(
            tasks,
            "finalize_amendment_proposal_projection",
            return_value=True,
        ) as finalize,
        pytest.raises(RuntimeError, match="did not write any chunks"),
    ):
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    finalize.assert_called_once_with(
        failure_session,
        proposal_id=9,
        succeeded=False,
        error_message="Indexing failed. The approval was not published.",
    )


def test_approval_task_rejects_promotion_during_projection() -> None:
    proposal = SimpleNamespace(id=9, status="approving", applied_new_chunk_id=None)
    user_file_id = UUID("00000000-0000-0000-0000-000000000111")
    result = SimpleNamespace(
        new_chunk=SimpleNamespace(id="new-chunk", user_file_id=user_file_id),
        old_chunk=None,
    )
    apply_session = MagicMock()
    projection_session = MagicMock()
    projection_session.get.return_value = SimpleNamespace(id=user_file_id)
    failure_session = MagicMock()
    contexts = []
    for session in (apply_session, projection_session, failure_session):
        context = MagicMock()
        context.__enter__.return_value = session
        contexts.append(context)

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            side_effect=contexts,
        ),
        patch.object(tasks, "get_proposal", return_value=proposal),
        patch.object(tasks, "approve_amendment_proposal", return_value=result),
        patch.object(
            tasks,
            "validate_amendment_projection_search_settings",
            side_effect=[11, RuntimeError("changed after projection")],
        ) as validate_settings,
        patch.object(tasks, "project_amendment_to_index", return_value=4),
        patch.object(
            tasks,
            "finalize_amendment_proposal_projection",
            return_value=True,
        ) as finalize,
        pytest.raises(RuntimeError, match="changed after projection"),
    ):
        tasks.regulatory_amendment_approve.run(
            proposal_id=9,
            tenant_id="tenant-a",
        )

    assert validate_settings.call_args_list == [
        call(projection_session),
        call(projection_session, expected_id=11, for_update=True),
    ]
    finalize.assert_called_once_with(
        failure_session,
        proposal_id=9,
        succeeded=False,
        error_message="Indexing failed. The approval was not published.",
    )


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


def test_approval_watchdog_keeps_long_gemini_projection_recoverable() -> None:
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
        patch.object(
            tasks,
            "touch_amendment_proposal_approval",
            return_value=True,
            create=True,
        ) as touch,
    ):
        heartbeat_session = MagicMock()
        session_factory.return_value.__enter__.return_value = heartbeat_session
        with tasks._renew_amendment_approval(proposal_id=9):
            pass

    touch.assert_called_once_with(heartbeat_session, proposal_id=9)
    stop.set.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=2)


def test_recovery_redelivers_applied_unlinked_approvals() -> None:
    batch_context = MagicMock()
    batch_context.__enter__.return_value = MagicMock()
    proposal_context = MagicMock()
    proposal_context.__enter__.return_value = MagicMock()
    task_app = MagicMock()

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
        patch.object(tasks.regulatory_amendment_recover_stale, "app", task_app),
    ):
        tasks.regulatory_amendment_recover_stale.run(tenant_id="tenant-a")

    enqueue_batch.assert_called_once_with(
        task_app,
        batch_id=42,
        tenant_id="tenant-a",
    )
    enqueue_approval.assert_called_once_with(
        task_app,
        proposal_id=9,
        tenant_id="tenant-a",
    )


def test_runtime_lite_worker_consumes_amendment_queue() -> None:
    backend_root = Path(__file__).resolve().parents[5]
    supervisor = (backend_root / "supervisord-lite.conf").read_text(encoding="utf-8")

    assert "-Q regulatory_benchmark,regulatory_amendment" in supervisor
