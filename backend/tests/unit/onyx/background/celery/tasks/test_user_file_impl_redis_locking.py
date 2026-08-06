"""Tests for the _impl functions' redis_locking parameter.

Verifies that:
- redis_locking=True acquires/releases Redis locks and clears queued keys
- redis_locking=False skips all Redis operations entirely
- Both paths execute the same business logic (DB lookup, status check)
"""

from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.user_file_processing.tasks import (
    delete_user_file_impl,
    process_user_file_impl,
    project_sync_user_file_impl,
    user_file_project_sync_lock_key,
)
from onyx.configs.constants import CELERY_USER_FILE_PROJECT_SYNC_LOCK_TIMEOUT
from onyx.db.enums import UserFileStatus

TASKS_MODULE = "onyx.background.celery.tasks.user_file_processing.tasks"


def _mock_session_returning_none() -> MagicMock:
    """Return a mock session whose .get() returns None (file not found)."""
    session = MagicMock()
    session.get.return_value = None
    return session


# ------------------------------------------------------------------
# process_user_file_impl
# ------------------------------------------------------------------


class TestProcessUserFileImpl:
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_acquires_and_releases_lock(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = True
        lock.owned.return_value = True
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        user_file_id = str(uuid4())
        process_user_file_impl(
            user_file_id=user_file_id,
            tenant_id="test-tenant",
            redis_locking=True,
        )

        mock_get_redis.assert_called_once_with(tenant_id="test-tenant")
        redis_client.delete.assert_called_once()
        lock.acquire.assert_called_once_with(blocking=False)
        lock.release.assert_called_once()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_skips_when_lock_held(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = False
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        process_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=True,
        )

        lock.acquire.assert_called_once()
        mock_get_session.assert_not_called()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_false_skips_redis_entirely(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        process_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=False,
        )

        mock_get_redis.assert_not_called()
        mock_get_session.assert_called_once()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_both_paths_call_db_get(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        """Both redis_locking=True and False should call db_session.get(UserFile, ...)."""
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = True
        lock.owned.return_value = True
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        uid = str(uuid4())

        process_user_file_impl(user_file_id=uid, tenant_id="t", redis_locking=True)
        call_count_true = session.get.call_count

        session.reset_mock()
        mock_get_session.reset_mock()
        mock_get_session.return_value.__enter__.return_value = session

        process_user_file_impl(user_file_id=uid, tenant_id="t", redis_locking=False)
        call_count_false = session.get.call_count

        assert call_count_true == call_count_false == 1


# ------------------------------------------------------------------
# delete_user_file_impl
# ------------------------------------------------------------------


class TestDeleteUserFileImpl:
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_acquires_and_releases_lock(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = True
        lock.owned.return_value = True
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        delete_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=True,
        )

        mock_get_redis.assert_called_once()
        lock.acquire.assert_called_once_with(blocking=False)
        lock.release.assert_called_once()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_skips_when_lock_held(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = False
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        delete_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=True,
        )

        lock.acquire.assert_called_once()
        mock_get_session.assert_not_called()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_false_skips_redis_entirely(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        delete_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=False,
        )

        mock_get_redis.assert_not_called()
        mock_get_session.assert_called_once()


# ------------------------------------------------------------------
# project_sync_user_file_impl
# ------------------------------------------------------------------


@patch(
    f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
    return_value=[],
)
class TestProjectSyncUserFileImpl:
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_acquires_and_releases_lock(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
        _mock_fetch: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = True
        lock.owned.return_value = True
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        user_file_id = str(uuid4())
        project_sync_user_file_impl(
            user_file_id=user_file_id,
            tenant_id="test-tenant",
            redis_locking=True,
        )

        mock_get_redis.assert_called_once()
        redis_client.delete.assert_called_once()
        redis_client.lock.assert_called_once_with(
            user_file_project_sync_lock_key(user_file_id),
            timeout=CELERY_USER_FILE_PROJECT_SYNC_LOCK_TIMEOUT,
            thread_local=False,
        )
        lock.acquire.assert_called_once_with(blocking=False)
        lock.release.assert_called_once()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_true_skips_when_lock_held(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
        _mock_fetch: MagicMock,
    ) -> None:
        redis_client = MagicMock()
        lock = MagicMock()
        lock.acquire.return_value = False
        redis_client.lock.return_value = lock
        mock_get_redis.return_value = redis_client

        project_sync_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=True,
        )

        lock.acquire.assert_called_once()
        redis_client.delete.assert_not_called()
        mock_get_session.assert_not_called()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.get_redis_client")
    def test_redis_locking_false_skips_redis_entirely(
        self,
        mock_get_redis: MagicMock,
        mock_get_session: MagicMock,
        _mock_fetch: MagicMock,
    ) -> None:
        session = _mock_session_returning_none()
        mock_get_session.return_value.__enter__.return_value = session

        project_sync_user_file_impl(
            user_file_id=str(uuid4()),
            tenant_id="test-tenant",
            redis_locking=False,
        )

        mock_get_redis.assert_not_called()
        mock_get_session.assert_called_once()


def _run_lost_lock_after_canonical_projection(*, active_target_id: int) -> MagicMock:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.status = UserFileStatus.COMPLETED
    user_file.needs_project_sync = True
    user_file.needs_persona_sync = True
    user_file.secondary_reconcile_pending = True
    user_file.projects = []
    user_file.assistants = []
    user_file.chunk_count = 3

    projected_target = MagicMock()
    projected_target.id = 41
    primary = MagicMock()
    primary.port_backfill_source_id = None
    active_settings = MagicMock()
    active_settings.primary = primary
    active_settings.secondary = projected_target
    current_target = MagicMock()
    current_target.id = active_target_id

    session = MagicMock()
    session.get.return_value = user_file
    session_context = MagicMock()
    session_context.__enter__.return_value = session

    lock = MagicMock()
    lock.acquire.return_value = True
    lock.owned.return_value = False
    redis_client = MagicMock()
    redis_client.lock.return_value = lock

    heartbeat = MagicMock()
    heartbeat.ensure_owned.side_effect = [None, RuntimeError("lease lost")]
    heartbeat.stop.return_value = True

    with (
        patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", False),
        patch(f"{TASKS_MODULE}.get_redis_client", return_value=redis_client),
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[user_file],
        ),
        patch(
            f"{TASKS_MODULE}.get_active_search_settings", return_value=active_settings
        ),
        patch(f"{TASKS_MODULE}.get_all_document_indices", return_value=[MagicMock()]),
        patch(f"{TASKS_MODULE}.build_access_for_user_files", return_value={}),
        patch(f"{TASKS_MODULE}.httpx_init_vespa_pool"),
        patch(
            f"{TASKS_MODULE}._sync_metadata_and_reconcile_secondary",
            return_value=(True, True),
        ),
        patch(
            f"{TASKS_MODULE}.active_secondary_port_target",
            return_value=current_target,
        ),
        patch(f"{TASKS_MODULE}._RedisLockHeartbeat", return_value=heartbeat),
    ):
        project_sync_user_file_impl(
            user_file_id=str(user_file.id),
            tenant_id="test-tenant",
            redis_locking=True,
        )

    heartbeat.ensure_owned.assert_has_calls([call(), call()])
    return user_file


def test_lost_redis_lock_accepts_only_completed_canonical_content() -> None:
    user_file = _run_lost_lock_after_canonical_projection(active_target_id=41)

    assert user_file.secondary_reconcile_pending is False
    # A different owner may now own these operations; the expired owner must not
    # clear them even though its exact canonical content write can be accepted.
    assert user_file.needs_project_sync is True
    assert user_file.needs_persona_sync is True


def test_lost_redis_lock_keeps_pending_when_future_target_changed() -> None:
    user_file = _run_lost_lock_after_canonical_projection(active_target_id=42)

    assert user_file.secondary_reconcile_pending is True
    assert user_file.needs_project_sync is True
    assert user_file.needs_persona_sync is True


def test_failed_file_sync_requires_document_set_dirty_flag() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.status = UserFileStatus.FAILED
    user_file.needs_project_sync = True
    user_file.needs_persona_sync = True
    user_file.needs_document_set_sync = False
    user_file.secondary_reconcile_pending = True

    session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session

    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[user_file],
        ),
        patch(f"{TASKS_MODULE}.get_active_search_settings") as get_search_settings,
    ):
        project_sync_user_file_impl(
            user_file_id=str(user_file.id),
            tenant_id="test-tenant",
            redis_locking=False,
        )

    get_search_settings.assert_not_called()
    session.add.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    ("stored_chunk_count", "regulatory_chunk_count", "expected_delete_count"),
    [
        (None, 3, 3),
        (5, 3, 5),
        (None, 0, None),
    ],
)
def test_failed_document_set_sync_removes_indexed_chunks_without_reconcile(
    stored_chunk_count: int | None,
    regulatory_chunk_count: int,
    expected_delete_count: int | None,
) -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.status = UserFileStatus.FAILED
    user_file.needs_project_sync = True
    user_file.needs_persona_sync = True
    user_file.needs_document_set_sync = True
    user_file.secondary_reconcile_pending = True
    user_file.projects = []
    user_file.assistants = []
    user_file.chunk_count = stored_chunk_count

    primary = MagicMock()
    primary.port_backfill_source_id = None
    active_settings = MagicMock()
    active_settings.primary = primary
    active_settings.secondary = MagicMock()
    session = MagicMock()
    session.get.return_value = user_file
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    retry_index = MagicMock()

    with (
        patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", False),
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[user_file],
        ),
        patch(
            f"{TASKS_MODULE}.get_active_search_settings",
            return_value=active_settings,
        ),
        patch(f"{TASKS_MODULE}.get_all_document_indices", return_value=[MagicMock()]),
        patch(f"{TASKS_MODULE}.RetryDocumentIndex", return_value=retry_index),
        patch(
            f"{TASKS_MODULE}.get_chunk_counts_for_files",
            return_value={user_file.id: regulatory_chunk_count},
        ),
        patch(f"{TASKS_MODULE}.httpx_init_vespa_pool"),
        patch(
            f"{TASKS_MODULE}._sync_metadata_and_reconcile_secondary",
            return_value=(False, False),
        ) as sync_metadata,
    ):
        project_sync_user_file_impl(
            user_file_id=str(user_file.id),
            tenant_id="test-tenant",
            redis_locking=False,
        )

    retry_index.delete.assert_called_once_with(
        str(user_file.id), chunk_count=expected_delete_count
    )
    sync_metadata.assert_not_called()
    assert user_file.needs_project_sync is True
    assert user_file.needs_persona_sync is True
    assert user_file.needs_document_set_sync is False
    assert user_file.secondary_reconcile_pending is True
    session.add.assert_called_once_with(user_file)
    session.commit.assert_called_once()
