import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from onyx.background.celery import regulatory_worker
from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE
from onyx.background.celery.tasks.regulatory_amendments import tasks
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryTask
from shared_configs.enums import EmbeddingProvider


def test_amendment_queue_is_isolated_by_database_identity() -> None:
    script = (
        "from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE; "
        "print(REGULATORY_AMENDMENT_QUEUE)"
    )

    def read_prefix(database_name: str) -> str:
        environment = os.environ.copy()
        environment["POSTGRES_DB"] = database_name
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[5])
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result.stdout.rstrip("\n")

    dev_queue = read_prefix("customs-regulations-dev")
    test_queue = read_prefix("customs-regulations-test")

    assert dev_queue.startswith("regulatory_amendment_")
    assert test_queue.startswith("regulatory_amendment_")
    assert dev_queue != test_queue
    assert dev_queue == read_prefix("customs-regulations-dev")


def test_amendment_queue_does_not_namespace_unrelated_broker_keys() -> None:
    from onyx.background.celery.configs.base import broker_transport_options

    assert "global_keyprefix" not in broker_transport_options


def test_enqueue_routes_redundant_expiring_deliveries() -> None:
    app = MagicMock()

    tasks.enqueue_amendment_batch(app, batch_id=42, tenant_id="public")

    expected = call(
        OnyxCeleryTask.REGULATORY_AMENDMENT_RUN,
        kwargs={
            "batch_id": 42,
            "tenant_id": "public",
            "environment": tasks.annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": tasks.annex_config.ANNEX_DATABASE_IDENTITY,
        },
        queue=tasks.annex_config.analysis_delivery_queue_name(),
        priority=OnyxCeleryPriority.HIGH,
        expires=24 * 60 * 60,
        retry=False,
    )
    delayed = call(
        OnyxCeleryTask.REGULATORY_AMENDMENT_RUN,
        kwargs={
            "batch_id": 42,
            "tenant_id": "public",
            "environment": tasks.annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": tasks.annex_config.ANNEX_DATABASE_IDENTITY,
        },
        queue=tasks.annex_config.analysis_delivery_queue_name(),
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
        queue=REGULATORY_AMENDMENT_QUEUE,
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

    assert "python -m onyx.background.celery.regulatory_worker" in supervisor


def test_regulatory_worker_launcher_consumes_scoped_amendment_queue() -> None:
    with (
        patch.object(sys, "argv", ["regulatory_worker", "--hostname=worker@%n"]),
        patch.object(regulatory_worker.os, "execvp") as execvp,
    ):
        regulatory_worker.main()

    command = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.regulatory_benchmark",
        "worker",
        "--hostname=worker@%n",
        "-Q",
        f"regulatory_benchmark,{REGULATORY_AMENDMENT_QUEUE}",
    ]
    execvp.assert_called_once_with(command[0], command)


def test_approval_task_validates_target_before_owned_atomic_publication() -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from onyx.db import regulatory_writer_publication
    from onyx.regulatory import writer_publication

    events: list[str] = []
    session = MagicMock()

    @contextmanager
    def scoped_session() -> Iterator[MagicMock]:
        events.append("read_target")
        yield session
        events.append("release_read_session")

    @contextmanager
    def heartbeat(*, proposal_id: int) -> Iterator[None]:
        assert proposal_id == 9
        events.append("heartbeat_start")
        yield
        events.append("heartbeat_stop")

    def publish(proposal_id: int, tenant_id: str, current_id: int) -> int:
        assert (proposal_id, tenant_id, current_id) == (9, "tenant-a", 11)
        assert events == ["read_target", "release_read_session", "heartbeat_start"]
        events.append("owned_publish")
        return 461

    with (
        patch.object(
            tasks, "get_session_with_current_tenant", side_effect=scoped_session
        ),
        patch.object(
            tasks, "validate_amendment_projection_search_settings", return_value=11
        ) as validate,
        patch.object(tasks, "_renew_amendment_approval", side_effect=heartbeat),
        patch.object(
            writer_publication, "approve_owned_amendment", side_effect=publish
        ),
        patch.object(
            regulatory_writer_publication, "record_owned_amendment_failure"
        ) as failure,
    ):
        tasks.regulatory_amendment_approve.run(proposal_id=9, tenant_id="tenant-a")
    assert events == [
        "read_target",
        "release_read_session",
        "heartbeat_start",
        "owned_publish",
        "heartbeat_stop",
    ]
    validate.assert_called_once_with(session)
    session.commit.assert_not_called()
    failure.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("provider detail"),
        ValueError("current amendment search settings changed"),
    ],
)
def test_approval_task_preserves_owned_failure_for_scoped_recovery(
    error: Exception,
) -> None:
    from contextlib import nullcontext

    from onyx.db import regulatory_writer_publication
    from onyx.regulatory import writer_publication

    with (
        patch.object(tasks, "get_session_with_current_tenant") as sessions,
        patch.object(
            tasks, "validate_amendment_projection_search_settings", return_value=11
        ),
        patch.object(tasks, "_renew_amendment_approval", return_value=nullcontext()),
        patch.object(
            writer_publication, "approve_owned_amendment", side_effect=error
        ) as publish,
        patch.object(
            regulatory_writer_publication, "record_owned_amendment_failure"
        ) as failure,
        pytest.raises(type(error)) as raised,
    ):
        tasks.regulatory_amendment_approve.run(proposal_id=9, tenant_id="tenant-a")
    assert raised.value is error
    publish.assert_called_once_with(9, "tenant-a", 11)
    failure.assert_called_once_with(9, "tenant-a", error)
    sessions.return_value.__enter__().commit.assert_not_called()


def test_approval_task_accepts_owned_idempotent_noop_without_marking_failure() -> None:
    from contextlib import nullcontext

    from onyx.db import regulatory_writer_publication
    from onyx.regulatory import writer_publication

    with (
        patch.object(tasks, "get_session_with_current_tenant") as sessions,
        patch.object(
            tasks, "validate_amendment_projection_search_settings", return_value=11
        ),
        patch.object(tasks, "_renew_amendment_approval", return_value=nullcontext()),
        patch.object(
            writer_publication, "approve_owned_amendment", return_value=0
        ) as publish,
        patch.object(
            regulatory_writer_publication, "record_owned_amendment_failure"
        ) as failure,
    ):
        assert (
            tasks.regulatory_amendment_approve.run(proposal_id=9, tenant_id="tenant-a")
            is None
        )
    publish.assert_called_once_with(9, "tenant-a", 11)
    failure.assert_not_called()
    sessions.return_value.__enter__().commit.assert_not_called()


def test_approval_task_rejects_invalid_initial_target_before_ownership() -> None:
    from onyx.db import regulatory_writer_publication
    from onyx.regulatory import writer_publication

    error = RuntimeError("current embedding setting is unsupported")
    with (
        patch.object(tasks, "get_session_with_current_tenant"),
        patch.object(
            tasks, "validate_amendment_projection_search_settings", side_effect=error
        ),
        patch.object(tasks, "_renew_amendment_approval") as heartbeat,
        patch.object(writer_publication, "approve_owned_amendment") as publish,
        patch.object(
            regulatory_writer_publication, "record_owned_amendment_failure"
        ) as failure,
        pytest.raises(RuntimeError) as raised,
    ):
        tasks.regulatory_amendment_approve.run(proposal_id=9, tenant_id="tenant-a")
    assert raised.value is error
    heartbeat.assert_not_called()
    publish.assert_not_called()
    failure.assert_called_once_with(9, "tenant-a", error)


def test_worker_retains_exception_with_current_failure_lease() -> None:
    from contextlib import nullcontext

    error = RuntimeError("private-provider-detail")
    with (
        patch.object(tasks, "get_session_with_current_tenant") as factory,
        patch.object(
            tasks,
            "claim_batch_for_analysis",
            return_value=SimpleNamespace(generation=7),
        ),
        patch.object(tasks, "_renew_batch_lease", return_value=nullcontext()),
        patch.object(tasks, "run_amendment_batch", side_effect=error),
        patch.object(tasks, "mark_batch_failed") as mark_failed,
        patch.object(tasks.logger, "exception"),
    ):
        with pytest.raises(RuntimeError) as raised:
            tasks.regulatory_amendment_run.run(batch_id=44, tenant_id="public")
    assert raised.value is error
    mark_failed.assert_called_once_with(
        factory.return_value.__enter__.return_value,
        batch_id=44,
        lease_generation=7,
        error_message=tasks._SAFE_FAILURE_MESSAGE,
        failure=error,
    )
