from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.server.features.document_set.api import (
    link_document_set_file,
    list_document_set_files,
    unlink_document_set_file,
)


@pytest.mark.parametrize("needs_sync", [False, True])
def test_link_schedules_metadata_sync_only_for_dirty_files(needs_sync: bool) -> None:
    file_id = uuid4()
    user_file = MagicMock(id=file_id, needs_document_set_sync=needs_sync)

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api.get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.link_user_file_to_document_set",
            return_value=True,
        ),
        patch(
            "onyx.server.features.document_set.api.UserFileSnapshot.from_model",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.server.features.document_set.api.trigger_user_file_metadata_sync"
        ) as trigger_sync,
    ):
        link_document_set_file(
            document_set_id=7,
            file_id=file_id,
            bg_tasks=MagicMock(),
            user=MagicMock(),
            db_session=MagicMock(),
            tenant_id="tenant",
        )

    if needs_sync:
        trigger_sync.assert_called_once()
    else:
        trigger_sync.assert_not_called()


@pytest.mark.parametrize("needs_sync", [False, True])
def test_unlink_schedules_metadata_sync_only_for_dirty_files(needs_sync: bool) -> None:
    file_id = uuid4()
    user_file = MagicMock(id=file_id, needs_document_set_sync=needs_sync)

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api.get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.unlink_user_file_from_document_set",
            return_value=True,
        ),
        patch(
            "onyx.server.features.document_set.api.trigger_user_file_metadata_sync"
        ) as trigger_sync,
    ):
        unlink_document_set_file(
            document_set_id=7,
            file_id=file_id,
            bg_tasks=MagicMock(),
            user=MagicMock(),
            db_session=MagicMock(),
            tenant_id="tenant",
        )

    if needs_sync:
        trigger_sync.assert_called_once()
    else:
        trigger_sync.assert_not_called()


def test_list_files_includes_latest_durable_indexing_progress() -> None:
    file_id = uuid4()
    user_file = MagicMock(id=file_id)
    progress = MagicMock()
    snapshot = MagicMock()

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api.fetch_user_files_for_document_set",
            return_value=[user_file],
        ),
        patch(
            "onyx.server.features.document_set.api.fetch_latest_regulatory_indexing_progress_for_user_files",
            return_value={file_id: progress},
        ) as fetch_progress,
        patch(
            "onyx.server.features.document_set.api.DocumentSetUserFileSnapshot.from_user_file",
            return_value=snapshot,
        ) as from_model,
    ):
        result = list_document_set_files(
            document_set_id=7,
            user=MagicMock(),
            db_session=MagicMock(),
        )

    assert result == [snapshot]
    fetch_progress.assert_called_once()
    assert fetch_progress.call_args.kwargs["user_file_ids"] == [file_id]
    from_model.assert_called_once_with(user_file, progress=progress)
