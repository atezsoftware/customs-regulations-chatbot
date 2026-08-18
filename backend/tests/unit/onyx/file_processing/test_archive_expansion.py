import zipfile
from io import BytesIO

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing import archive_expansion
from onyx.file_processing.archive_expansion import expand_archive_uploads


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _as_upload(filename: str, payload: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(payload),
        size=len(payload),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def _zip_upload(entries: dict[str, bytes], filename: str = "sources.zip") -> UploadFile:
    return _as_upload(filename, _build_zip(entries), "application/zip")


def _read(upload: UploadFile) -> bytes:
    upload.file.seek(0)
    return upload.file.read()


def test_expands_archive_into_one_upload_per_markdown_entry() -> None:
    upload = _zip_upload(
        {
            "madde-1.md": b"# Madde 1",
            "madde-2.md": b"# Madde 2",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["madde-1.md", "madde-2.md"]
    assert [_read(item) for item in expanded] == [b"# Madde 1", b"# Madde 2"]


def test_passes_through_non_archive_uploads_in_order() -> None:
    plain = _as_upload("tebligat.md", b"# Tebligat", "text/markdown")
    archive = _zip_upload({"madde-1.md": b"# Madde 1", "madde-2.md": b"# Madde 2"})

    expanded = expand_archive_uploads([plain, archive])

    assert [item.filename for item in expanded] == [
        "tebligat.md",
        "madde-1.md",
        "madde-2.md",
    ]
    assert expanded[0] is plain


def test_guesses_content_type_from_entry_name() -> None:
    upload = _zip_upload({"madde-1.md": b"# Madde 1", "madde-2.md": b"# Madde 2"})

    expanded = expand_archive_uploads([upload])

    assert expanded[0].content_type == "text/markdown"


# --- what gets dropped -------------------------------------------------------


def test_drops_non_markdown_entries() -> None:
    upload = _zip_upload(
        {
            "madde.md": b"# Madde",
            "notlar.mdx": b"# Notlar",
            "routing.json": b"[]",
            "okuma.txt": b"duz metin",
            "tablo.xlsx": b"binary",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["madde.md", "notlar.mdx"]


def test_skips_directory_entries() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(zipfile.ZipInfo("bolum-1/"), b"")
        archive.writestr("bolum-1/madde-a.md", b"# A")
        archive.writestr("bolum-1/madde-b.md", b"# B")
    upload = _as_upload("sources.zip", buffer.getvalue(), "application/zip")

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == [
        "bolum-1/madde-a.md",
        "bolum-1/madde-b.md",
    ]


def test_skips_macos_metadata_and_dotfiles() -> None:
    upload = _zip_upload(
        {
            "__MACOSX/._madde.md": b"resource fork",
            ".DS_Store": b"junk",
            ".onyx_metadata.json": b"{}",
            "madde-1.md": b"# Madde 1",
            "madde-2.md": b"# Madde 2",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["madde-1.md", "madde-2.md"]


# --- naming ------------------------------------------------------------------


def test_names_a_lone_markdown_file_after_its_directory() -> None:
    # The bundle layout the importer pipeline emits: one directory per source
    # document, holding a generically named markdown rendering of it.
    upload = _zip_upload(
        {
            "mevzuat/1975_tir_sozlesmesi.docx/document.md": b"# TIR",
            "mevzuat/1975_tir_sozlesmesi.docx/.bundle.sha256": b"abc",
            "mevzuat/2006-11_sinir_kapilari.docx/document.md": b"# Sinir",
            "mevzuat/2006-11_sinir_kapilari.docx/.bundle.sha256": b"def",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == [
        "mevzuat/1975_tir_sozlesmesi.md",
        "mevzuat/2006-11_sinir_kapilari.md",
    ]


def test_directory_collapse_survives_sidecars_that_were_filtered_out() -> None:
    upload = _zip_upload(
        {
            "paket.docx/document.md": b"# Madde",
            "paket.docx/routing.json": b"[]",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["paket.md"]


def test_keeps_filenames_when_a_directory_holds_several_markdown_files() -> None:
    upload = _zip_upload(
        {
            "bolum/madde-1.md": b"# Madde 1",
            "bolum/madde-2.md": b"# Madde 2",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == [
        "bolum/madde-1.md",
        "bolum/madde-2.md",
    ]


def test_collapsed_names_stay_distinct_for_identically_named_documents() -> None:
    upload = _zip_upload(
        {
            "bolum-1/document.md": b"birinci",
            "bolum-2/document.md": b"ikinci",
        }
    )

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["bolum-1.md", "bolum-2.md"]
    assert [_read(item) for item in expanded] == [b"birinci", b"ikinci"]


def test_does_not_collapse_a_lone_markdown_file_at_the_archive_root() -> None:
    upload = _zip_upload({"madde.md": b"# Madde"})

    expanded = expand_archive_uploads([upload])

    assert [item.filename for item in expanded] == ["madde.md"]


# --- hostile archives --------------------------------------------------------


@pytest.mark.parametrize(
    "entry_name",
    ["../escape.md", "bolum/../../escape.md", "/etc/passwd"],
)
def test_rejects_entry_escaping_the_archive_root(entry_name: str) -> None:
    upload = _zip_upload({entry_name: b"# Madde", "madde.md": b"# Madde"})

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.INVALID_INPUT


def test_rejects_symlink_entry() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        symlink = zipfile.ZipInfo("madde.md")
        symlink.create_system = 3  # unix
        symlink.external_attr = (0o120777 << 16) | 0o600
        archive.writestr(symlink, b"/etc/passwd")
    upload = _as_upload("sources.zip", buffer.getvalue(), "application/zip")

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.INVALID_INPUT


def test_rejects_nested_archive_even_though_it_is_not_markdown() -> None:
    upload = _zip_upload(
        {
            "madde.md": b"# Madde",
            "inner.zip": _build_zip({"gizli.md": b"# Gizli"}),
        }
    )

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.INVALID_INPUT


def test_rejects_corrupt_archive() -> None:
    upload = _as_upload("sources.zip", b"not really a zip", "application/zip")

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.INVALID_INPUT


def test_rejects_archive_with_too_many_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_expansion, "MAX_ARCHIVE_ENTRIES", 2)
    upload = _zip_upload(
        {"madde-1.md": b"a", "madde-2.md": b"b", "madde-3.md": b"c"},
    )

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.PAYLOAD_TOO_LARGE


def test_rejects_archive_exceeding_expanded_size_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_expansion, "MAX_ARCHIVE_EXPANDED_BYTES", 16)
    upload = _zip_upload({"madde-1.md": b"a" * 10, "madde-2.md": b"b" * 10})

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.PAYLOAD_TOO_LARGE


def test_rejects_entry_with_implausible_compression_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_expansion, "COMPRESSION_RATIO_CHECK_MIN_BYTES", 1024)
    upload = _zip_upload({"bomb.md": b"\0" * 4_000_000})

    with pytest.raises(OnyxError) as raised:
        expand_archive_uploads([upload])

    assert raised.value.error_code is OnyxErrorCode.PAYLOAD_TOO_LARGE


def test_allows_ordinary_text_compression_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_expansion, "COMPRESSION_RATIO_CHECK_MIN_BYTES", 1024)
    # Repetitive but realistic regulatory prose, well under the bomb threshold.
    body = b"\n\n".join(
        f"## Madde {index}\n\nBu madde uyarinca gumruk beyannamesi duzenlenir.".encode()
        for index in range(400)
    )
    upload = _zip_upload({"mevzuat.md": body, "ikinci.md": b"# Ikinci"})

    expanded = expand_archive_uploads([upload])

    assert _read(expanded[0]) == body
