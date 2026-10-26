"""Serving regulatory documents and chunks as PDF.

The same bytes back both uses in the UI: the browser displays them inline and
saves them unchanged, so the endpoints return the document rather than a
download-only attachment.
"""

import datetime
import hashlib
from collections.abc import Iterator
from io import BytesIO
from unittest.mock import MagicMock, call, patch
from urllib.parse import quote
from uuid import uuid4

import pytest

from onyx.db.document_intake_preview import IntakePreparedPreview
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.publication_reads import PublicationReadChanged
from onyx.server.features.regulatory import api

API_MODULE = "onyx.server.features.regulatory.api"
READS_MODULE = "onyx.regulatory.publication_reads"


@pytest.fixture(autouse=True)
def publication_reads() -> Iterator[tuple[MagicMock, MagicMock]]:
    observation = MagicMock()
    with (
        patch(f"{READS_MODULE}.observe_publication_read", return_value=observation),
        patch(f"{READS_MODULE}.require_publication_files") as require_files,
        patch(f"{API_MODULE}.intake_prepared_preview", return_value=None),
    ):
        yield observation, require_files


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
    assert bytes(response.body).startswith(b"%PDF-")
    # Shown in the browser rather than forced to disk; the same bytes save fine.
    assert "inline" in response.headers["content-disposition"]
    assert "1975_tir_sozlesmesi" in response.headers["content-disposition"]


@pytest.mark.parametrize("name", ["uploaded.pdf", "without-pdf-extension.md"])
def test_original_pdf_bytes_skip_rendering_and_keep_both_publication_checks(
    publication_reads: tuple[MagicMock, MagicMock], name: str
) -> None:
    observation, require_files = publication_reads
    user_file = _user_file()
    user_file.name = name
    original = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    file_store = MagicMock()
    file_store.read_file.return_value = BytesIO(original)
    user, session = MagicMock(), MagicMock()
    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file) as owned,
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
        patch(f"{API_MODULE}.render_document_pdf") as renderer,
    ):
        response = api.get_file_pdf(user_file.id, user=user, db_session=session)

    assert response.body == original
    assert response.media_type == "application/pdf"
    renderer.assert_not_called()
    owned.assert_called_once_with(session, user_file.id, user)
    file_store.read_file.assert_called_once_with(user_file.file_id, mode="b")
    assert require_files.call_args_list == [
        call(observation, (user_file.id,)),
        call(observation, (user_file.id,)),
    ]


@pytest.mark.parametrize("reject_after_read", [False, True])
def test_pdf_publication_gate_can_reject_before_or_after_the_file_read(
    publication_reads: tuple[MagicMock, MagicMock], reject_after_read: bool
) -> None:
    _observation, require_files = publication_reads
    require_files.side_effect = (
        [None, PublicationReadChanged()]
        if reject_after_read
        else [PublicationReadChanged()]
    )
    user_file = _user_file()
    file_store = MagicMock()
    file_store.read_file.return_value = BytesIO(b"%PDF-1.7\n%%EOF")
    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
        patch(f"{API_MODULE}.render_document_pdf") as renderer,
    ):
        with pytest.raises(PublicationReadChanged):
            api.get_file_pdf(user_file.id, user=MagicMock(), db_session=MagicMock())
    assert file_store.read_file.call_count == int(reject_after_read)
    renderer.assert_not_called()


def test_original_pdf_still_requires_file_ownership(
    publication_reads: tuple[MagicMock, MagicMock],
) -> None:
    _observation, require_files = publication_reads
    user_file = _user_file()
    user_file.user_id = uuid4()
    user = MagicMock(id=uuid4(), role="BASIC")
    session = MagicMock()
    session.get.return_value = user_file
    with patch(f"{API_MODULE}.get_default_file_store") as store:
        with pytest.raises(OnyxError, match="Not your file"):
            api.get_file_pdf(user_file.id, user=user, db_session=session)
    store.assert_not_called()
    require_files.assert_not_called()


def test_non_pdf_content_keeps_the_original_markdown_render_path(
    publication_reads: tuple[MagicMock, MagicMock],
) -> None:
    _observation, require_files = publication_reads
    user_file = _user_file()
    user_file.name = "markdown-with-pdf-extension.pdf"
    markdown = "# Başlık\n\nEkli belge."
    file_store = MagicMock()
    file_store.read_file.return_value = BytesIO(markdown.encode())
    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
        patch(
            f"{API_MODULE}.render_document_pdf", return_value=b"rendered"
        ) as renderer,
    ):
        response = api.get_file_pdf(
            user_file.id, user=MagicMock(), db_session=MagicMock()
        )
    renderer.assert_called_once_with(name=user_file.name, markdown=markdown)
    assert response.body == b"rendered"
    assert require_files.call_count == 2


@pytest.mark.parametrize("extension", ["docx", "xlsx", "html", "png"])
def test_intake_original_renders_verified_prepared_text(
    publication_reads: tuple[MagicMock, MagicMock], extension: str
) -> None:
    _observation, require_files = publication_reads
    user_file = _user_file()
    user_file.name = f"original.{extension}"
    original, text = b"binary-original\x00\xff", "# İnceleme\n\nDoğru metin.".encode()
    source = IntakePreparedPreview(
        original_file_id=user_file.file_id,
        raw_sha256=hashlib.sha256(original).hexdigest(),
        text_file_id="prepared-text",
        text_sha256=hashlib.sha256(text).hexdigest(),
    )
    file_store = MagicMock()
    file_store.read_file.side_effect = [BytesIO(original), BytesIO(text)]
    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
        patch(f"{API_MODULE}.intake_prepared_preview", return_value=source),
        patch(
            f"{API_MODULE}.render_document_pdf", return_value=b"rendered"
        ) as renderer,
    ):
        response = api.get_file_pdf(
            user_file.id, user=MagicMock(), db_session=MagicMock()
        )
    renderer.assert_called_once_with(
        name=f"original.{extension} (extracted text)", markdown=text.decode()
    )
    assert response.body == b"rendered"
    assert require_files.call_count == 2
    assert file_store.read_file.call_args_list == [
        call(user_file.file_id, mode="b"),
        call("prepared-text", mode="b"),
    ]


@pytest.mark.parametrize("failure", ["raw", "text", "utf8", "binding"])
def test_changed_intake_original_or_prepared_text_never_renders(
    publication_reads: tuple[MagicMock, MagicMock], failure: str
) -> None:
    _observation, require_files = publication_reads
    user_file = _user_file()
    original = b"binary-original\xff"
    text = b"prepared" if failure != "utf8" else b"\xff"
    source = IntakePreparedPreview(
        original_file_id="foreign" if failure == "binding" else user_file.file_id,
        raw_sha256="0" * 64
        if failure == "raw"
        else hashlib.sha256(original).hexdigest(),
        text_file_id="prepared-text",
        text_sha256="0" * 64 if failure == "text" else hashlib.sha256(text).hexdigest(),
    )
    file_store = MagicMock()
    file_store.read_file.side_effect = [BytesIO(original), BytesIO(text)]
    with (
        patch(f"{API_MODULE}._get_owned_user_file", return_value=user_file),
        patch(f"{API_MODULE}.get_default_file_store", return_value=file_store),
        patch(f"{API_MODULE}.intake_prepared_preview", return_value=source),
        patch(f"{API_MODULE}.render_document_pdf") as renderer,
    ):
        with pytest.raises(OnyxError, match="Verified reading text is unavailable"):
            api.get_file_pdf(user_file.id, user=MagicMock(), db_session=MagicMock())
    renderer.assert_not_called()
    assert require_files.call_count == 1


@pytest.mark.parametrize("filename", ["document.pdf", "İthalat ve Gümrük Şartları.pdf"])
def test_pdf_filename_supports_utf8_and_preserves_ascii(filename: str) -> None:
    response = api._pdf_response(b"%PDF-1.7\n%%EOF", filename)
    disposition = response.headers["content-disposition"]
    if filename.isascii():
        assert disposition == f'inline; filename="{filename}"'
    else:
        assert f"filename*=UTF-8''{quote(filename, safe='')}" in disposition
        assert disposition.isascii()


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
    assert bytes(response.body).startswith(b"%PDF-")
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
