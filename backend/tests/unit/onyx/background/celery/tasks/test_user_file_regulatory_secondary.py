"""Secondary indexing delegates canonical/history publication to the owned writer."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.user_file_processing.tasks import (
    _index_user_file_to_secondary,
)
from onyx.db.models import SearchSettings


@pytest.mark.parametrize("pass_uuid", [False, True])
def test_secondary_projection_preserves_file_and_exact_target_scope(
    pass_uuid: bool,
) -> None:
    file_id = uuid4()
    secondary = SearchSettings(id=41)
    with patch(
        "onyx.regulatory.writer_publication.republish_user_file", return_value=2
    ) as publish:
        assert (
            _index_user_file_to_secondary(
                file_id if pass_uuid else str(file_id), secondary, "tenant-a"
            )
            is True
        )
    publish.assert_called_once_with(
        file_id,
        "tenant-a",
        target_search_settings_id=41,
        include_chunked=True,
        adopt_original=True,
    )


def test_secondary_projection_reports_no_eligible_projection() -> None:
    with patch(
        "onyx.regulatory.writer_publication.republish_user_file", return_value=0
    ):
        assert (
            _index_user_file_to_secondary(uuid4(), SearchSettings(id=41), "tenant-a")
            is False
        )


def test_secondary_projection_does_not_mask_owned_publication_failure() -> None:
    with patch(
        "onyx.regulatory.writer_publication.republish_user_file",
        side_effect=ValueError("target is no longer FUTURE"),
    ):
        with pytest.raises(ValueError, match="no longer FUTURE"):
            _index_user_file_to_secondary(uuid4(), SearchSettings(id=41), "tenant-a")
