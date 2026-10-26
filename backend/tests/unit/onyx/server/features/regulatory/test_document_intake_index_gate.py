"""Ordinary document-set indexing cannot bypass new-document admission review."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import UserFileStatus
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.document_set import api
from onyx.server.features.regulatory import api as regulatory_api
from onyx.server.features.regulatory.models import (
    RegulatoryChunkUpdateRequest,
    RegulatoryFileValidityUpdateRequest,
    UserFileRenameRequest,
)


def test_individual_index_rejects_an_unapproved_intake() -> None:
    file = SimpleNamespace(id=uuid4(), status=UserFileStatus.CHUNKED)
    with (
        patch.object(api, "_get_editable_document_set_or_raise"),
        patch.object(
            api, "get_user_file_for_document_set_management", return_value=file
        ),
        patch.object(api, "pending_intake_file_ids", return_value={file.id}),
        patch.object(api, "_request_user_file_indexing") as request,
    ):
        with pytest.raises(OnyxError) as error:
            api.index_document_set_file(1, file.id, MagicMock(), MagicMock(), "public")
        assert error.value.error_code == OnyxErrorCode.CONFLICT
        request.assert_not_called()


def test_bulk_index_preserves_ordinary_uploads_and_skips_unapproved_intakes() -> None:
    ordinary = SimpleNamespace(id=uuid4(), status=UserFileStatus.CHUNKED)
    imported = SimpleNamespace(id=uuid4(), status=UserFileStatus.CHUNKED)
    session = MagicMock()
    with (
        patch.object(api, "_get_editable_document_set_or_raise"),
        patch.object(
            api, "fetch_user_files_for_document_set", return_value=[ordinary, imported]
        ),
        patch.object(api, "pending_intake_file_ids", return_value={imported.id}),
        patch.object(api, "_request_user_file_indexing") as request,
    ):
        result = api.index_document_set_chunked_files(1, MagicMock(), session, "public")
        assert result.queued == 1
        request.assert_called_once_with([ordinary], session, "public")


@pytest.mark.parametrize("action", ["chunk", "validity", "rename"])
def test_generic_mutations_cannot_bypass_intake_relabel_and_approval(
    action: str,
) -> None:
    file_id = uuid4()
    session, user = MagicMock(), MagicMock()
    with (
        patch.object(regulatory_api, "_get_owned_user_file"),
        patch.object(
            regulatory_api,
            "get_chunk_snapshot_by_id",
            return_value=SimpleNamespace(user_file_id=file_id),
        ),
        patch.object(
            regulatory_api, "pending_intake_file_ids", return_value={file_id}
        ) as pending,
    ):
        with pytest.raises(OnyxError) as error:
            if action == "chunk":
                regulatory_api.patch_chunk(
                    "pending-chunk",
                    RegulatoryChunkUpdateRequest(text="Edit"),
                    user,
                    session,
                )
            elif action == "validity":
                regulatory_api.patch_file_validity(
                    file_id,
                    RegulatoryFileValidityUpdateRequest(clear_validity_end_date=True),
                    user,
                    session,
                )
            else:
                regulatory_api.rename_user_file(
                    file_id, UserFileRenameRequest(name="Renamed"), user, session
                )
        assert error.value.error_code == OnyxErrorCode.CONFLICT
        assert "new-document review" in error.value.detail
        pending.assert_called_once_with(session, [file_id], include_approving=True)
        session.rollback.assert_not_called()


def test_completed_and_ordinary_files_keep_the_existing_edit_channel() -> None:
    session, file_id = MagicMock(), uuid4()
    with patch.object(regulatory_api, "pending_intake_file_ids", return_value=set()):
        regulatory_api._require_ordinary_edit_channel(session, file_id)
