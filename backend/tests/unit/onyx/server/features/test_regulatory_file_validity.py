import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.regulatory_chunks import (
    RegulatoryFileValidityUpdateResult,
    RegulatoryFileValidityWindow,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.validity_projection import AppliedRegulatoryValidityPatch
from onyx.server.features.regulatory.api import patch_file_validity
from onyx.server.features.regulatory.models import (
    RegulatoryFileValidityUpdateRequest,
)


@pytest.mark.parametrize(
    ("previous_window", "updated_window"),
    [
        (
            None,
            RegulatoryFileValidityWindow(
                start=datetime.date(2025, 1, 1), end=None
            ),
        ),
        (RegulatoryFileValidityWindow(start=None, end=None), None),
    ],
)
def test_missing_uniform_window_requires_canonical_reindex_without_projection(
    previous_window: RegulatoryFileValidityWindow | None,
    updated_window: RegulatoryFileValidityWindow | None,
) -> None:
    db_session = MagicMock()

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=406,
                skipped_versioned_chunk_count=0,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices"
        ) as patch_metadata,
        patch(
            "onyx.server.features.regulatory.api.project_user_file_to_index"
        ) as full_projection,
        pytest.raises(OnyxError) as exc_info,
    ):
        patch_file_validity(
            user_file_id=uuid4(),
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    assert exc_info.value.error_code is OnyxErrorCode.CONFLICT
    assert "canonical reindex is required" in exc_info.value.detail
    db_session.flush.assert_called_once_with()
    db_session.rollback.assert_called_once_with()
    db_session.commit.assert_not_called()
    patch_metadata.assert_not_called()
    full_projection.assert_not_called()


def test_versioned_file_requires_canonical_reindex_without_projection() -> None:
    db_session = MagicMock()
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=10,
                skipped_versioned_chunk_count=1,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices"
        ) as patch_metadata,
        patch(
            "onyx.server.features.regulatory.api.project_user_file_to_index"
        ) as full_projection,
        pytest.raises(OnyxError) as exc_info,
    ):
        patch_file_validity(
            user_file_id=uuid4(),
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    assert exc_info.value.error_code is OnyxErrorCode.CONFLICT
    assert "canonical reindex is required" in exc_info.value.detail
    db_session.rollback.assert_called_once_with()
    db_session.commit.assert_not_called()
    patch_metadata.assert_not_called()
    full_projection.assert_not_called()


def test_metadata_preflight_none_requires_reindex_without_full_projection() -> None:
    user_file = MagicMock()
    db_session = MagicMock()
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=10,
                skipped_versioned_chunk_count=0,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices",
            return_value=None,
        ) as patch_metadata,
        patch(
            "onyx.server.features.regulatory.api.project_user_file_to_index"
        ) as full_projection,
        pytest.raises(OnyxError) as exc_info,
    ):
        patch_file_validity(
            user_file_id=uuid4(),
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    assert exc_info.value.error_code is OnyxErrorCode.CONFLICT
    assert "canonical reindex is required" in exc_info.value.detail
    patch_metadata.assert_called_once_with(
        db_session,
        user_file,
        previous_window=previous_window,
        updated_window=updated_window,
    )
    db_session.rollback.assert_called_once_with()
    db_session.commit.assert_not_called()
    full_projection.assert_not_called()


def test_uniform_file_validity_uses_metadata_only_projection() -> None:
    user_file_id = uuid4()
    user_file = MagicMock()
    db_session = MagicMock()
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )
    applied_patch = AppliedRegulatoryValidityPatch(
        document_id=str(user_file_id),
        expected_regulatory_chunk_ids=("rc-a",),
        previous_window=previous_window,
        updated_window=updated_window,
        updated_indices=(MagicMock(),),
    )

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=406,
                skipped_versioned_chunk_count=0,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices",
            return_value=applied_patch,
        ) as patch_metadata,
        patch(
            "onyx.server.features.regulatory.api.project_user_file_to_index"
        ) as full_projection,
    ):
        response = patch_file_validity(
            user_file_id=user_file_id,
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    patch_metadata.assert_called_once_with(
        db_session,
        user_file,
        previous_window=previous_window,
        updated_window=updated_window,
    )
    full_projection.assert_not_called()
    db_session.commit.assert_called_once_with()
    assert response.updated_chunk_count == 406


def test_metadata_only_present_failure_never_commits_canonical_dates() -> None:
    user_file = MagicMock()
    db_session = MagicMock()
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=10,
                skipped_versioned_chunk_count=0,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices",
            side_effect=RuntimeError("present patch failed"),
        ),
        patch(
            "onyx.server.features.regulatory.api.project_user_file_to_index"
        ) as full_projection,
        pytest.raises(RuntimeError, match="present patch failed"),
    ):
        patch_file_validity(
            user_file_id=uuid4(),
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    full_projection.assert_not_called()
    db_session.rollback.assert_called_once_with()
    db_session.commit.assert_not_called()


def test_pg_commit_failure_compensates_successful_present_patch() -> None:
    user_file_id = uuid4()
    user_file = MagicMock()
    db_session = MagicMock()
    db_session.commit.side_effect = RuntimeError("commit failed")
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )
    # FUTURE may already have failed and compensated itself; only PRESENT is
    # listed here as successfully updated.
    applied_patch = AppliedRegulatoryValidityPatch(
        document_id=str(user_file_id),
        expected_regulatory_chunk_ids=("rc-a",),
        previous_window=previous_window,
        updated_window=updated_window,
        updated_indices=(MagicMock(),),
    )

    with (
        patch(
            "onyx.server.features.regulatory.api._get_owned_user_file",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch(
            "onyx.server.features.regulatory.api.update_file_validity_window",
            return_value=RegulatoryFileValidityUpdateResult(
                updated_chunk_count=10,
                skipped_versioned_chunk_count=0,
                previous_window=previous_window,
                updated_window=updated_window,
            ),
        ),
        patch(
            "onyx.server.features.regulatory.api.patch_user_file_validity_in_active_indices",
            return_value=applied_patch,
        ),
        patch(
            "onyx.server.features.regulatory.api.compensate_regulatory_validity_patch"
        ) as compensate,
        pytest.raises(RuntimeError, match="commit failed"),
    ):
        patch_file_validity(
            user_file_id=user_file_id,
            update_request=RegulatoryFileValidityUpdateRequest(
                validity_start_date=datetime.date(2025, 1, 1)
            ),
            user=MagicMock(),
            db_session=db_session,
        )

    compensate.assert_called_once_with(applied_patch)
    db_session.rollback.assert_called_once_with()
