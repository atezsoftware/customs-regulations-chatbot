"""
Unit test verifying that the upload API path sends tasks with expires=.

The upload_files_to_user_files_with_indexing function must include expires=
on every send_task call to prevent phantom task accumulation if the worker
is down or slow.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from onyx.auth.schemas import UserRole
from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.models import UserFile
from onyx.db.projects import (
    check_project_access,
    check_project_write_access,
    upload_files_to_user_files_with_indexing,
)


def _make_mock_user_file() -> MagicMock:
    uf = MagicMock(spec=UserFile)
    uf.id = str(uuid4())
    return uf


@patch("onyx.db.projects.get_current_tenant_id", return_value="test_tenant")
@patch("onyx.db.projects.create_user_files")
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_send_task_includes_expires(
    mock_client_app: MagicMock,
    mock_create: MagicMock,
    mock_tenant: MagicMock,  # noqa: ARG001
) -> None:
    """Every send_task call from the upload path must include expires=."""
    user_files = [_make_mock_user_file(), _make_mock_user_file()]
    mock_create.return_value = MagicMock(
        user_files=user_files,
        rejected_files=[],
        id_to_temp_id={},
        skip_indexing_filenames=set(),
        indexable_files=user_files,
    )

    mock_user = MagicMock()
    mock_db_session = MagicMock()

    upload_files_to_user_files_with_indexing(
        files=[],
        project_id=None,
        user=mock_user,
        temp_id_map=None,
        db_session=mock_db_session,
    )

    assert mock_client_app.send_task.call_count == len(user_files)

    for call in mock_client_app.send_task.call_args_list:
        assert call.args[0] == OnyxCeleryTask.PROCESS_SINGLE_USER_FILE
        assert call.kwargs.get("queue") == OnyxCeleryQueues.USER_FILE_PROCESSING
        assert call.kwargs.get("expires") == CELERY_USER_FILE_PROCESSING_TASK_EXPIRES, (
            "send_task must include expires= to prevent phantom task accumulation"
        )


def test_admin_can_write_to_an_existing_project() -> None:
    user = MagicMock()
    user.role = UserRole.ADMIN
    db_session = MagicMock()
    db_session.get.return_value = MagicMock()

    assert check_project_write_access(42, user, db_session)
    db_session.get.assert_called_once()


def test_admin_can_access_an_existing_project() -> None:
    user = MagicMock()
    user.role = UserRole.ADMIN
    db_session = MagicMock()
    db_session.get.return_value = MagicMock()

    assert check_project_access(42, user, db_session)
    db_session.get.assert_called_once()


@patch("onyx.db.projects.check_project_ownership", return_value=False)
def test_non_admin_cannot_write_to_another_users_project(
    mock_check_ownership: MagicMock,
) -> None:
    user = MagicMock()
    user.role = UserRole.BASIC
    user.id = uuid4()
    db_session = MagicMock()

    assert not check_project_write_access(42, user, db_session)
    mock_check_ownership.assert_called_once_with(42, user.id, db_session)
