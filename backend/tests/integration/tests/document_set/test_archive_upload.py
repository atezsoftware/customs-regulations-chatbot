"""Zip upload on the admin document-set surface.

Verifies the whole path an operator uses to bulk-load regulatory markdown:
one archive in, one indexed UserFile per contained document out.
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from onyx.db.enums import UserFileStatus
from onyx.server.features.projects.models import UserFileSnapshot
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

_EXPORT_CORPUS = Path(
    "/home/kubilay-payci/mevzuat-datalar-chatbot/data/"
    "PC-output-6b5214873e2c4178b1590649ee26a95b/documents/"
    "PC Külliyat Dosyaları/İhracat Mevzuatı/"
    "İhracat Mevzuatıyla İlgili Tasarruflu Yazılar"
)
_REPRESENTATIVE_DOCUMENTS = (
    "06.09.2018_37024696_dahilde_isleme_izni_belgesi_haricindeki_bugday_unu_"
    "ihracati_yasagi.docx",
    "19.02.2008_04574_kdv_iadesinde_kullanilacak_ihracat_beyannamelerinin_"
    "ilgili_gumruk_idaresince_onaylanmasi_gerekmemektedir.docx",
)


def _build_archive(entries: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _chunk_then_index(
    document_set_id: int,
    expected_count: int,
    admin_user: DATestUser,
) -> list[UserFileSnapshot]:
    chunked = DocumentSetManager.wait_for_files_chunked(
        document_set_id=document_set_id,
        expected_count=expected_count,
        user_performing_action=admin_user,
    )
    assert all(file.status is UserFileStatus.CHUNKED for file in chunked)
    assert DocumentSetManager.index_chunked_files(document_set_id, admin_user) == (
        expected_count
    )
    return DocumentSetManager.wait_for_files_indexed(
        document_set_id=document_set_id,
        expected_count=expected_count,
        user_performing_action=admin_user,
    )


@pytest.fixture
def representative_export_zip(tmp_path: Path) -> bytes:
    """Build a temporary archive from two real converted export documents."""

    selected = [_EXPORT_CORPUS / name for name in _REPRESENTATIVE_DOCUMENTS]
    if any(not (bundle / "document.md").is_file() for bundle in selected):
        pytest.skip("representative PC export corpus is unavailable")
    archive_path = tmp_path / "representative-ihracat.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for bundle in selected:
            prefix = f"ihracat/{bundle.name}"
            archive.write(bundle / "document.md", f"{prefix}/document.md")
            for sidecar_name in ("routing.json", ".bundle.sha256"):
                sidecar = bundle / sidecar_name
                if sidecar.is_file():
                    archive.write(sidecar, f"{prefix}/{sidecar_name}")
    return archive_path.read_bytes()


def test_zip_upload_chunks_then_indexes_one_document_per_bundle(
    admin_user: DATestUser,
) -> None:
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

    files = _chunk_then_index(
        document_set.id,
        len(_EXPECTED_NAMES),
        admin_user,
    )

    assert {file.name for file in files} == _EXPECTED_NAMES
    for file in files:
        assert file.status == UserFileStatus.COMPLETED, f"{file.name}: {file.status}"
        assert file.chunk_count is not None and file.chunk_count > 0


def test_representative_pc_export_zip_preserves_each_markdown_child(
    admin_user: DATestUser,
    representative_export_zip: bytes,
) -> None:
    document_set = DocumentSetManager.create(user_performing_action=admin_user)
    result = DocumentSetManager.upload_files(
        document_set_id=document_set.id,
        files=[
            (
                "representative-ihracat.zip",
                representative_export_zip,
                "application/zip",
            )
        ],
        user_performing_action=admin_user,
    )

    assert result.rejected_files == []
    assert len(result.user_files) == len(_REPRESENTATIVE_DOCUMENTS)
    files = _chunk_then_index(
        document_set.id,
        len(_REPRESENTATIVE_DOCUMENTS),
        admin_user,
    )
    assert len({file.id for file in files}) == len(_REPRESENTATIVE_DOCUMENTS)
    assert all(file.status is UserFileStatus.COMPLETED for file in files)


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
