"""Zip upload on the admin document-set surface.

Verifies the whole path an operator uses to bulk-load regulatory markdown:
one archive in, one indexed UserFile per contained document out.
"""

import zipfile
from io import BytesIO

from onyx.db.enums import UserFileStatus
from tests.integration.common_utils.managers.document_set import DocumentSetManager
from tests.integration.common_utils.test_models import DATestUser

# Mirrors the bundle layout the conversion pipeline emits: a directory per
# source document, holding a generically named markdown rendering plus sidecars.
_BUNDLE_ARCHIVE = {
    "mevzuat/1975_tir_sozlesmesi.docx/document.md": (
        "# TIR Sozlesmesi\n\nMadde 1 - Bu sozlesme transit tasimaciligi duzenler.\n"
    ).encode(),
    "mevzuat/1975_tir_sozlesmesi.docx/routing.json": b"[]",
    "mevzuat/1975_tir_sozlesmesi.docx/.bundle.sha256": b"0" * 64,
    "mevzuat/2006-11_sinir_kapilari.docx/document.md": (
        "# Sinir Kapilari\n\nMadde 1 - Kara sinir kapilarindan yapilan cikislar.\n"
    ).encode(),
    "mevzuat/2006-11_sinir_kapilari.docx/routing.json": b"[]",
    "mevzuat/2006-11_sinir_kapilari.docx/.bundle.sha256": b"1" * 64,
}

# Each bundle directory collapses into the document it holds, and the source
# document's extension is dropped so the indexed name reads as the document.
_EXPECTED_NAMES = {
    "mevzuat/1975_tir_sozlesmesi.md",
    "mevzuat/2006-11_sinir_kapilari.md",
}


def _build_archive(entries: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_zip_upload_indexes_one_document_per_bundle(admin_user: DATestUser) -> None:
    document_set = DocumentSetManager.create(user_performing_action=admin_user)

    result = DocumentSetManager.upload_files(
        document_set_id=document_set.id,
        files=[("mevzuat.zip", _build_archive(_BUNDLE_ARCHIVE), "application/zip")],
        user_performing_action=admin_user,
    )

    # The sidecars are dropped rather than rejected: they are packaging, not
    # documents an operator asked to import.
    assert result.rejected_files == []
    assert {file.name for file in result.user_files} == _EXPECTED_NAMES

    files = DocumentSetManager.wait_for_files_indexed(
        document_set_id=document_set.id,
        expected_count=len(_EXPECTED_NAMES),
        user_performing_action=admin_user,
    )

    assert {file.name for file in files} == _EXPECTED_NAMES
    for file in files:
        assert file.status == UserFileStatus.COMPLETED, f"{file.name}: {file.status}"
        assert file.chunk_count is not None and file.chunk_count > 0


def test_zip_upload_rejects_an_archive_that_escapes_its_root(
    admin_user: DATestUser,
) -> None:
    document_set = DocumentSetManager.create(user_performing_action=admin_user)

    response = DocumentSetManager.upload_files_raw(
        document_set_id=document_set.id,
        files=[
            (
                "kotu.zip",
                _build_archive({"../escape.md": b"# Escape", "madde.md": b"# Madde"}),
                "application/zip",
            )
        ],
        user_performing_action=admin_user,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert DocumentSetManager.get_files(document_set.id, admin_user) == []
