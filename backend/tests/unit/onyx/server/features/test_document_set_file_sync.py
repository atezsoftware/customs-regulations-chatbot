from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.document_set.api import (
    get_completed_document_set_file_reprojection,
    link_document_set_file,
    list_document_set_files,
    reproject_completed_document_set_file,
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


def test_completed_regulatory_file_reprojection_is_queued() -> None:
    file_id = uuid4()
    attempt_id = uuid4()
    user_file = MagicMock(id=file_id, status=UserFileStatus.COMPLETED)
    updated_at = datetime.now(timezone.utc)
    repair = SimpleNamespace(
        user_file_id=file_id,
        attempt_id=attempt_id,
        status=UserFileProjectionRepairStatus.PENDING,
        updated_at=updated_at,
    )
    db_session = MagicMock()

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api."
            "get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.get_chunk_counts_for_files",
            return_value={file_id: 1},
        ),
        patch(
            "onyx.server.features.document_set.api.claim_user_file_projection_repair",
            return_value=attempt_id,
        ) as claim,
        patch(
            "onyx.server.features.document_set.api._enqueue_user_file_indexing"
        ) as enqueue,
        patch(
            "onyx.server.features.document_set.api.get_user_file_projection_repair",
            return_value=repair,
        ),
    ):
        result = reproject_completed_document_set_file(
            document_set_id=7,
            file_id=file_id,
            user=MagicMock(),
            db_session=db_session,
            tenant_id="tenant",
        )

    assert result.attempt_id == attempt_id
    assert result.status is UserFileProjectionRepairStatus.PENDING
    assert result.updated_at == updated_at
    claim.assert_called_once()
    enqueue.assert_called_once_with(
        file_id,
        "tenant",
        reproject_completed=True,
        projection_repair_attempt_id=attempt_id,
    )


def test_projection_repair_status_is_returned() -> None:
    file_id = uuid4()
    attempt_id = uuid4()
    updated_at = datetime.now(timezone.utc)
    user_file = MagicMock(id=file_id)
    repair = SimpleNamespace(
        user_file_id=file_id,
        attempt_id=attempt_id,
        status=UserFileProjectionRepairStatus.FAILED,
        updated_at=updated_at,
    )

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api."
            "get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.get_user_file_projection_repair",
            return_value=repair,
        ),
    ):
        result = get_completed_document_set_file_reprojection(
            document_set_id=7,
            file_id=file_id,
            user=MagicMock(),
            db_session=MagicMock(),
        )

    assert result.attempt_id == attempt_id
    assert result.status is UserFileProjectionRepairStatus.FAILED


def test_pending_regulatory_file_reprojection_is_not_queued_twice() -> None:
    file_id = uuid4()
    user_file = MagicMock(id=file_id, status=UserFileStatus.COMPLETED)

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api."
            "get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.get_chunk_counts_for_files",
            return_value={file_id: 1},
        ),
        patch(
            "onyx.server.features.document_set.api.claim_user_file_projection_repair",
            return_value=None,
        ),
        patch(
            "onyx.server.features.document_set.api._enqueue_user_file_indexing"
        ) as enqueue,
        pytest.raises(OnyxError, match="already in progress"),
    ):
        reproject_completed_document_set_file(
            document_set_id=7,
            file_id=file_id,
            user=MagicMock(),
            db_session=MagicMock(),
            tenant_id="tenant",
        )

    enqueue.assert_not_called()


def test_reprojection_publish_failure_is_recorded() -> None:
    file_id = uuid4()
    attempt_id = uuid4()
    user_file = MagicMock(id=file_id, status=UserFileStatus.COMPLETED)
    db_session = MagicMock()

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api."
            "get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.document_set.api.get_chunk_counts_for_files",
            return_value={file_id: 1},
        ),
        patch(
            "onyx.server.features.document_set.api.claim_user_file_projection_repair",
            return_value=attempt_id,
        ),
        patch(
            "onyx.server.features.document_set.api._enqueue_user_file_indexing",
            side_effect=RuntimeError("broker unavailable"),
        ),
        patch(
            "onyx.server.features.document_set.api.finish_user_file_projection_repair",
            return_value=True,
        ) as finish,
        pytest.raises(RuntimeError, match="broker unavailable"),
    ):
        reproject_completed_document_set_file(
            document_set_id=7,
            file_id=file_id,
            user=MagicMock(),
            db_session=db_session,
            tenant_id="tenant",
        )

    finish.assert_called_once_with(db_session, file_id, attempt_id, succeeded=False)
    assert db_session.commit.call_count == 2


def test_chunked_file_cannot_use_reprojection_repair() -> None:
    file_id = uuid4()
    user_file = MagicMock(id=file_id, status=UserFileStatus.CHUNKED)

    with (
        patch(
            "onyx.server.features.document_set.api._get_editable_document_set_or_raise"
        ),
        patch(
            "onyx.server.features.document_set.api."
            "get_user_file_for_document_set_management",
            return_value=user_file,
        ),
        pytest.raises(OnyxError, match="Only completed files"),
    ):
        reproject_completed_document_set_file(
            document_set_id=7,
            file_id=file_id,
            user=MagicMock(),
            db_session=MagicMock(),
            tenant_id="tenant",
        )
