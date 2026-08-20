"""Serving regulatory documents and chunks as PDF.

The same bytes back both uses in the UI: the browser displays them inline and
saves them unchanged, so the endpoints return the document rather than a
download-only attachment.
"""

import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.models import RegulatoryChunk, UserFile
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory import api

API_MODULE = "onyx.server.features.regulatory.api"


def _user_file() -> MagicMock:
    user_file = MagicMock(spec=UserFile)
    user_file.id = uuid4()
    user_file.name = "mevzuat/1975_tir_sozlesmesi.md"
    user_file.file_id = "stored-file-id"
    return user_file


def _chunk(user_file_id: object) -> MagicMock:
    chunk = MagicMock(spec=RegulatoryChunk)
    chunk.id = "chunk-1"
    chunk.user_file_id = user_file_id
    chunk.text = "Madde 5 - Gümrük müşavirliği şirketleri."
    chunk.heading_path = ["BİRİNCİ BÖLÜM", "Madde 5"]
    chunk.position = 4
    chunk.validity_start_date = datetime.date(2024, 1, 1)
    chunk.validity_end_date = None
    return chunk


def test_document_pdf_is_rendered_from_the_uploaded_markdown() -> None:
    user_file = _user_file()
    file_store = MagicMock()
    file_store.read_file.return_value = BytesIO("# Başlık\n\nMetin.".encode())

    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
    ):
        response = api.get_file_pdf(
            user_file_id=user_file.id,
            user=MagicMock(),
            db_session=MagicMock(),
        )

    file_store.read_file.assert_called_once_with(user_file.file_id, mode="b")
    assert response.media_type == "application/pdf"
    assert response.body.startswith(b"%PDF-")
    # Shown in the browser rather than forced to disk; the same bytes save fine.
    assert "inline" in response.headers["content-disposition"]
    assert "1975_tir_sozlesmesi" in response.headers["content-disposition"]


def test_chunk_pdf_is_rendered_for_a_single_chunk() -> None:
    user_file = _user_file()
    chunk = _chunk(user_file.id)

    with (
        patch(f"{API_MODULE}.get_chunk_by_id", return_value=chunk),
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
    ):
        response = api.get_chunk_pdf(
            chunk_id=chunk.id,
            user=MagicMock(),
            db_session=MagicMock(),
        )

    assert response.media_type == "application/pdf"
    assert response.body.startswith(b"%PDF-")
    assert "inline" in response.headers["content-disposition"]


def test_chunk_pdf_rejects_a_chunk_that_does_not_exist() -> None:
    with patch(f"{API_MODULE}.get_chunk_by_id", return_value=None):
        with pytest.raises(OnyxError):
            api.get_chunk_pdf(
                chunk_id="missing",
                user=MagicMock(),
                db_session=MagicMock(),
            )


def test_chunk_pdf_checks_ownership_of_the_owning_file() -> None:
    """Permission lives on the file, so the chunk route has to go through it."""

    chunk = _chunk(uuid4())

    with (
        patch(f"{API_MODULE}.get_chunk_by_id", return_value=chunk),
        patch(
            f"{API_MODULE}._get_owned_user_file",
            side_effect=OnyxError(api.OnyxErrorCode.UNAUTHORIZED, "Not your file"),
        ),
    ):
        with pytest.raises(OnyxError):
            api.get_chunk_pdf(
                chunk_id=chunk.id,
                user=MagicMock(),
                db_session=MagicMock(),
            )
