import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import RegulatoryChunkSource, RegulatoryChunkStatus
from onyx.db.regulatory_chunks import RegulatoryFileValidityWindow
from onyx.regulatory.validity_projection import (
    AppliedRegulatoryValidityPatch,
    compensate_regulatory_validity_patch,
    patch_user_file_validity_in_active_indices,
)


def _row(row_id: str, position: int, start: datetime.date) -> MagicMock:
    row = MagicMock()
    row.id = row_id
    row.position = position
    row.source = RegulatoryChunkSource.INDEXED.value
    row.status = RegulatoryChunkStatus.ACTIVE.value
    row.validity_start_date = start
    row.validity_end_date = None
    row.supersedes_chunk_id = None
    row.superseded_by_chunk_id = None
    return row


def _settings(*, current: bool, future: bool = False) -> MagicMock:
    settings = MagicMock()
    settings.id = 1 if current else 2
    settings.status.is_current.return_value = current
    settings.status.is_future.return_value = future
    return settings


def _user_file(
    chunk_count: int, *, secondary_reconcile_pending: bool = True
) -> MagicMock:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.chunk_count = chunk_count
    user_file.secondary_reconcile_pending = secondary_reconcile_pending
    return user_file


@pytest.mark.parametrize("initial_reconcile_pending", [True, False])
def test_validity_patch_preserves_existing_reconcile_state_after_future_success(
    initial_reconcile_pending: bool,
) -> None:
    updated_window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1), end=None
    )
    previous_window = RegulatoryFileValidityWindow(start=None, end=None)
    rows = [
        _row("rc-a", 0, updated_window.start),
        _row("rc-b", 1, updated_window.start),
    ]
    user_file = _user_file(
        len(rows), secondary_reconcile_pending=initial_reconcile_pending
    )
    current_index = MagicMock()
    future_index = MagicMock()

    with (
        patch(
            "onyx.regulatory.validity_projection.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX",
            True,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_chunks_for_file",
            return_value=rows,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_active_search_settings_list",
            return_value=[
                _settings(current=True),
                _settings(current=False, future=True),
            ],
        ),
        patch(
            "onyx.regulatory.validity_projection.build_opensearch_document_index",
            side_effect=[current_index, future_index],
        ) as build_index,
    ):
        applied_patch = patch_user_file_validity_in_active_indices(
            MagicMock(),
            user_file,
            previous_window=previous_window,
            updated_window=updated_window,
        )

    assert applied_patch is not None
    assert applied_patch.updated_indices == (current_index, future_index)
    assert build_index.call_count == 2
    expected_kwargs = {
        "document_id": str(user_file.id),
        "expected_regulatory_chunk_ids": ["rc-a", "rc-b"],
        "previous_start_date": None,
        "previous_end_date": None,
        "updated_start_date": datetime.date(2025, 1, 1),
        "updated_end_date": None,
    }
    current_index.update_regulatory_validity.assert_called_once_with(**expected_kwargs)
    future_index.update_regulatory_validity.assert_called_once_with(**expected_kwargs)
    assert user_file.secondary_reconcile_pending is initial_reconcile_pending


def test_present_failure_stops_future_and_is_raised() -> None:
    window = RegulatoryFileValidityWindow(start=datetime.date(2025, 1, 1), end=None)
    rows = [_row("rc-a", 0, window.start)]
    current_index = MagicMock()
    current_index.update_regulatory_validity.side_effect = RuntimeError(
        "present failed"
    )
    future_index = MagicMock()

    with (
        patch(
            "onyx.regulatory.validity_projection.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX",
            True,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_chunks_for_file",
            return_value=rows,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_active_search_settings_list",
            return_value=[
                _settings(current=True),
                _settings(current=False, future=True),
            ],
        ),
        patch(
            "onyx.regulatory.validity_projection.build_opensearch_document_index",
            side_effect=[current_index, future_index],
        ),
        pytest.raises(RuntimeError, match="present failed"),
    ):
        patch_user_file_validity_in_active_indices(
            MagicMock(),
            _user_file(1),
            previous_window=RegulatoryFileValidityWindow(start=None, end=None),
            updated_window=window,
        )

    future_index.update_regulatory_validity.assert_not_called()


def test_future_failure_is_deferred_after_present_success() -> None:
    window = RegulatoryFileValidityWindow(start=datetime.date(2025, 1, 1), end=None)
    rows = [_row("rc-a", 0, window.start)]
    user_file = _user_file(1, secondary_reconcile_pending=False)
    current_index = MagicMock()
    future_index = MagicMock()
    future_index.update_regulatory_validity.side_effect = RuntimeError("future failed")

    with (
        patch(
            "onyx.regulatory.validity_projection.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX",
            True,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_chunks_for_file",
            return_value=rows,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_active_search_settings_list",
            return_value=[
                _settings(current=True),
                _settings(current=False, future=True),
            ],
        ),
        patch(
            "onyx.regulatory.validity_projection.build_opensearch_document_index",
            side_effect=[current_index, future_index],
        ),
    ):
        applied_patch = patch_user_file_validity_in_active_indices(
            MagicMock(),
            user_file,
            previous_window=RegulatoryFileValidityWindow(start=None, end=None),
            updated_window=window,
        )

    assert applied_patch is not None
    assert applied_patch.updated_indices == (current_index,)
    current_index.update_regulatory_validity.assert_called_once()
    future_index.update_regulatory_validity.assert_called_once()
    assert user_file.secondary_reconcile_pending is True


def test_future_index_construction_failure_is_deferred_after_present_success() -> None:
    window = RegulatoryFileValidityWindow(start=datetime.date(2025, 1, 1), end=None)
    rows = [_row("rc-a", 0, window.start)]
    user_file = _user_file(1, secondary_reconcile_pending=False)
    current_index = MagicMock()

    with (
        patch(
            "onyx.regulatory.validity_projection.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX",
            True,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_chunks_for_file",
            return_value=rows,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_active_search_settings_list",
            return_value=[
                _settings(current=True),
                _settings(current=False, future=True),
            ],
        ),
        patch(
            "onyx.regulatory.validity_projection.build_opensearch_document_index",
            side_effect=[current_index, RuntimeError("future construction failed")],
        ) as build_index,
    ):
        applied_patch = patch_user_file_validity_in_active_indices(
            MagicMock(),
            user_file,
            previous_window=RegulatoryFileValidityWindow(start=None, end=None),
            updated_window=window,
        )

    assert applied_patch is not None
    assert applied_patch.updated_indices == (current_index,)
    assert build_index.call_count == 2
    current_index.update_regulatory_validity.assert_called_once()
    assert user_file.secondary_reconcile_pending is True


def test_fast_path_refuses_unknown_projection_alignment_before_writing() -> None:
    user_file = _user_file(2)
    rows = [_row("rc-a", 0, datetime.date(2025, 1, 1))]

    with (
        patch(
            "onyx.regulatory.validity_projection.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX",
            True,
        ),
        patch(
            "onyx.regulatory.validity_projection.get_chunks_for_file",
            return_value=rows,
        ),
        patch(
            "onyx.regulatory.validity_projection.build_opensearch_document_index"
        ) as build_index,
    ):
        applied_patch = patch_user_file_validity_in_active_indices(
            MagicMock(),
            user_file,
            previous_window=RegulatoryFileValidityWindow(start=None, end=None),
            updated_window=RegulatoryFileValidityWindow(
                start=datetime.date(2025, 1, 1), end=None
            ),
        )

    assert applied_patch is None
    build_index.assert_not_called()


def test_compensation_reverses_only_indices_that_were_successfully_updated() -> None:
    current_index = MagicMock()
    applied_patch = AppliedRegulatoryValidityPatch(
        document_id="doc-1",
        expected_regulatory_chunk_ids=("rc-a",),
        previous_window=RegulatoryFileValidityWindow(start=None, end=None),
        updated_window=RegulatoryFileValidityWindow(
            start=datetime.date(2025, 1, 1), end=None
        ),
        updated_indices=(current_index,),
    )

    compensate_regulatory_validity_patch(applied_patch)

    current_index.update_regulatory_validity.assert_called_once_with(
        document_id="doc-1",
        expected_regulatory_chunk_ids=["rc-a"],
        previous_start_date=datetime.date(2025, 1, 1),
        previous_end_date=None,
        updated_start_date=None,
        updated_end_date=None,
    )
