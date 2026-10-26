import io
import json
import stat
import zipfile
from hashlib import sha256
from typing import TypedDict

import pytest

from onyx.regulatory.document_intake import markdown_package as package


class _ManifestFixture(TypedDict):
    format: str
    main: str
    documents: list[dict[str, str]]


def _manifest(*, parent: str = "ana.md") -> _ManifestFixture:
    return {
        "format": "legislation-markdown-v1",
        "main": "ana.md",
        "documents": [
            {"path": "ekler/ek-1.md", "title": "EK-1", "parent": parent},
            {
                "path": "ana.md",
                "title": "Ana belge",
                "source_url": "https://example.com/mevzuat",
            },
        ],
    }


def _zip(
    files: dict[str, bytes],
    *,
    manifest: object | None = None,
    prefix: str = "",
    extra: list[zipfile.ZipInfo] | None = None,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        if manifest is not None:
            archive.writestr(
                prefix + "manifest.json",
                json.dumps(manifest, ensure_ascii=False).encode(),
            )
        for path, data in files.items():
            archive.writestr(prefix + path, data)
        for entry in extra or []:
            archive.writestr(entry, b"special")
    return stream.getvalue()


def _pair(root: str = "# Ana\n\n[EK-1](ekler/ek-1.md)\n") -> dict[str, bytes]:
    return {
        "ana.md": root.encode(),
        "ekler/ek-1.md": "# EK-1\n\nTürkçe kural.\n".encode(),
    }


@pytest.mark.parametrize("prefix", ["", "paket/"])
def test_zip_preserves_bytes_text_and_declared_parent_before_cross_reference(
    prefix: str,
) -> None:
    root = b"\xef\xbb\xbf# Ana\r\n\r\n[EK-1][ek]\r\n\r\n[ek]: ekler/ek-1.md#tablo\r\n"
    annex = "# EK-1\n\n| Başlık | Değer |\n| --- | --- |\n| A | ıİşŞğĞ |\n\n[Ana](../ana.md)\n".encode()
    result = package.parse_markdown_package(
        _zip(
            {"ana.md": root, "ekler/ek-1.md": annex},
            manifest=_manifest(),
            prefix=prefix,
        ),
        "Paket.ZIP",
    )
    assert result.status == "ready" and not result.issues
    main, child = result.assets
    assert main.display_name == "Ana belge" and child.display_name == "EK-1"
    assert main.content == root and main.text == root.decode("utf-8-sig")
    assert child.content == annex and child.text == annex.decode()
    assert main.sha256 == sha256(root).hexdigest()
    assert main.mime_type == child.mime_type == "text/markdown"
    assert main.original_url == main.final_url == "https://example.com/mevzuat"
    parent, crossref = result.links
    assert parent.parent_asset_hash == main.sha256
    assert parent.target_asset_hash == child.sha256
    assert parent.requested_url == "ekler/ek-1.md"
    assert parent.original_url == "ekler/ek-1.md#tablo"
    assert parent.source_field == "markdown:ana.md:link:1"
    assert parent.kind == "embedded"
    assert (
        crossref.parent_asset_hash == child.sha256
        and crossref.target_asset_hash == main.sha256
    )


def test_real_markdown_reference_and_raw_html_links_preserve_literal_urls() -> None:
    root = '# Ana\n\n`[sahte](eksik.md)`\n\n```md\n![tarama](scan.png)\n```\n\n<a data-href="ignored.md" data-note=" href=\'shadow.md\'" href="ekler/ek-1.md#tablo">EK &amp; 1</a>\n\n[Kaynak](https://example.com/source.pdf?a=1&amp;b=2)\n'
    result = package.parse_markdown_package(
        _zip(_pair(root), manifest=_manifest()), "source.zip"
    )
    assert len(result.links) == 1
    assert result.links[0].original_url == "ekler/ek-1.md#tablo"
    assert result.links[0].label == "EK & 1"
    assert "https://example.com/source.pdf?a=1&amp;b=2" in result.assets[0].text


def test_nested_parent_tree_is_explicit_and_parent_edges_precede_crossrefs() -> None:
    manifest = _manifest()
    documents = manifest["documents"]
    assert isinstance(documents, list)
    documents.insert(
        0, {"path": "alt.md", "title": "Alt ek", "parent": "ekler/ek-1.md"}
    )
    files = _pair("[Çapraz](alt.md)\n[Ek](ekler/ek-1.md)\n")
    files["ekler/ek-1.md"] = b"# Annex\n[Alt](../alt.md)\n"
    files["alt.md"] = b"# Child\nUnique nested text.\n"
    result = package.parse_markdown_package(
        _zip(files, manifest=manifest), "nested.zip"
    )
    assert [asset.display_name for asset in result.assets] == [
        "Ana belge",
        "EK-1",
        "Alt ek",
    ]
    assert [link.requested_url for link in result.links] == [
        "ekler/ek-1.md",
        "alt.md",
        "alt.md",
    ]
    assert [link.source_field for link in result.links] == [
        "markdown:ana.md:link:2",
        "markdown:ekler/ek-1.md:link:1",
        "markdown:ana.md:link:1",
    ]


def test_percent_encoded_unicode_link_resolves_only_exact_package_path() -> None:
    manifest = _manifest()
    documents = manifest["documents"]
    assert isinstance(documents, list)
    documents[0]["path"] = "ekler/çizelge.md"
    files = {
        "ana.md": b"[Ek](ekler/%C3%A7izelge.md)\n",
        "ekler/çizelge.md": "# Çizelge\n".encode(),
    }
    result = package.parse_markdown_package(
        _zip(files, manifest=manifest), "unicode.zip"
    )
    assert result.links[0].original_url == "ekler/%C3%A7izelge.md"
    assert result.links[0].requested_url == "ekler/çizelge.md"


@pytest.mark.parametrize(
    "source,title",
    [("# **Başlık**\n\nTam metin.\n", "Başlık"), ("MADDE 1 - Tam metin.\r\n", "kural")],
)
def test_standalone_uses_first_h1_or_filename_without_rewriting(
    source: str, title: str
) -> None:
    raw = b"\xef\xbb\xbf" + source.encode()
    result = package.parse_markdown_package(raw, "kural.md")
    assert len(result.assets) == 1 and not result.links
    assert result.assets[0].text == source and result.assets[0].content == raw
    assert result.assets[0].display_name == title


@pytest.mark.parametrize(
    "source,code",
    [
        ("[Ek](missing.md)", "missing_local_markdown_target"),
        ("[Ek](../outside.md)", "missing_local_markdown_target"),
        ("[Ek](/ana.md)", "unsupported_or_unsafe_markdown_link"),
        ("[Ek](file:///tmp/source.md)", "unsupported_or_unsafe_markdown_link"),
        ("![Tarama](https://example.com/scan.png)", "images_require_transcription"),
        ('<img src="scan.png">', "images_or_embedded_resources_require_transcription"),
        (
            '<object data="source.pdf"></object>',
            "images_or_embedded_resources_require_transcription",
        ),
        ("[Tarama](https://example.com/scan.jpg)", "images_require_transcription"),
    ],
)
def test_standalone_rejects_missing_local_sources_and_images(
    source: str, code: str
) -> None:
    with pytest.raises(package.MarkdownPackageError, match=code):
        package.parse_markdown_package(source.encode(), "root.md")


@pytest.mark.parametrize(
    "path",
    ["../escape.md", "/absolute.md", "C:/drive.md", "a\\b.md", "a//b.md", "a/./b.md"],
)
def test_zip_never_accepts_unsafe_entry_paths(path: str) -> None:
    with pytest.raises(package.MarkdownPackageError, match="unsafe_package_path"):
        package.parse_markdown_package(
            _zip({path: b"text"}, manifest=_manifest()), "bad.zip"
        )


@pytest.mark.parametrize("collision", ["ANA.md", "a\u0301.md"])
def test_zip_rejects_case_and_unicode_normalization_collisions(collision: str) -> None:
    original = "ana.md" if collision == "ANA.md" else "á.md"
    with pytest.raises(
        package.MarkdownPackageError, match="duplicate_or_ambiguous_package_path"
    ):
        package.parse_markdown_package(
            _zip({original: b"first", collision: b"second"}, manifest=_manifest()),
            "bad.zip",
        )


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO])
def test_zip_rejects_symlinks_and_special_files(mode: int) -> None:
    entry = zipfile.ZipInfo("unsafe.md")
    entry.create_system = 3
    entry.external_attr = (mode | 0o777) << 16
    with pytest.raises(
        package.MarkdownPackageError, match="encrypted_or_special_zip_entry"
    ):
        package.parse_markdown_package(
            _zip(_pair(), manifest=_manifest(), extra=[entry]), "bad.zip"
        )


def test_zip_rejects_encrypted_entry_before_reading() -> None:
    content = _zip(_pair(), manifest=_manifest())
    # Set the encryption bit in each central directory record; no decryption occurs.
    tampered = content.replace(
        b"PK\x01\x02\x14\x03\x14\x00\x00\x00", b"PK\x01\x02\x14\x03\x14\x00\x01\x00"
    )
    assert tampered != content
    with pytest.raises(
        package.MarkdownPackageError, match="encrypted_or_special_zip_entry"
    ):
        package.parse_markdown_package(tampered, "encrypted.zip")


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("no_manifest", "exactly_one_root_manifest_required"),
        ("unlisted", "manifest_must_exactly_list_all_markdown_files"),
        ("missing", "manifest_must_exactly_list_all_markdown_files"),
        ("wrong_case", "manifest_must_exactly_list_all_markdown_files"),
        ("nested_archive", "manifest_must_exactly_list_all_markdown_files"),
        ("parent_missing", "annex_requires_one_valid_parent"),
        ("parent_self", "annex_requires_one_valid_parent"),
        ("main_parent", "main_document_cannot_have_parent"),
        ("no_parent_link", "declared_parent_must_link_to_annex"),
        ("duplicate_content", "duplicate_markdown_content"),
    ],
)
def test_zip_fails_incomplete_or_ambiguous_packages(mutation: str, code: str) -> None:
    manifest = _manifest()
    files = _pair()
    documents = manifest["documents"]
    assert isinstance(documents, list)
    if mutation == "unlisted":
        files["extra.md"] = b"extra"
    elif mutation == "missing":
        files.pop("ekler/ek-1.md")
    elif mutation == "wrong_case":
        documents[0]["path"] = "ekler/EK-1.md"
    elif mutation == "nested_archive":
        files["nested.zip"] = b"PK"
    elif mutation == "parent_missing":
        documents[0].pop("parent")
    elif mutation == "parent_self":
        documents[0]["parent"] = "ekler/ek-1.md"
    elif mutation == "main_parent":
        documents[1]["parent"] = "ekler/ek-1.md"
    elif mutation == "no_parent_link":
        files["ana.md"] = b"# Ana\n`[Ek](ekler/ek-1.md)`\n"
    elif mutation == "duplicate_content":
        files["ekler/ek-1.md"] = files["ana.md"]
    with pytest.raises(package.MarkdownPackageError, match=code):
        package.parse_markdown_package(
            _zip(files, manifest=None if mutation == "no_manifest" else manifest),
            "bad.zip",
        )


def test_zip_rejects_parent_cycle_even_if_markdown_links_exist() -> None:
    manifest = _manifest(parent="alt.md")
    documents = manifest["documents"]
    assert isinstance(documents, list)
    documents.append({"path": "alt.md", "title": "Alt", "parent": "ekler/ek-1.md"})
    files = _pair()
    files["alt.md"] = b"# Alt\n[Ek](ekler/ek-1.md)\n"
    with pytest.raises(
        package.MarkdownPackageError, match="manifest_parent_tree_required"
    ):
        package.parse_markdown_package(_zip(files, manifest=manifest), "cycle.zip")


def test_limits_are_checked_without_truncating_or_partial_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package, "MAX_MARKDOWN_UPLOAD_BYTES", 10)
    with pytest.raises(
        package.MarkdownPackageError, match="markdown_upload_byte_limit"
    ):
        package.parse_markdown_package(b"x" * 11, "large.md")
    monkeypatch.setattr(package, "MAX_MARKDOWN_UPLOAD_BYTES", 10000)
    monkeypatch.setattr(package, "MAX_MARKDOWN_DOCUMENT_BYTES", 10)
    with pytest.raises(
        package.MarkdownPackageError, match="markdown_document_byte_limit"
    ):
        package.parse_markdown_package(b"x" * 11, "large.md")


def test_zip_compression_bomb_is_rejected_from_metadata() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.md", b"x" * 100000)
    with pytest.raises(
        package.MarkdownPackageError, match="markdown_archive_compression_limit"
    ):
        package.parse_markdown_package(stream.getvalue(), "bomb.zip")


def test_template_is_deterministic_complete_and_accepted() -> None:
    first = package.build_markdown_package_template()
    assert first == package.build_markdown_package_template()
    result = package.parse_markdown_package(first, "ornek.zip")
    assert result.status == "ready" and len(result.assets) == 3
    assert [link.requested_url for link in result.links[:2]] == [
        "ekler/ek-1.md",
        "ekler/ek-2.md",
    ]


@pytest.mark.parametrize(
    "raw", [b"\xff\xfe", b"text\x00binary", b"", b"   ", b"%PDF-1.7\nASCII PDF objects"]
)
def test_markdown_is_nonempty_utf8_text(raw: bytes) -> None:
    with pytest.raises(package.MarkdownPackageError):
        package.parse_markdown_package(raw, "invalid.md")


def test_error_never_contains_document_text_or_unbounded_path() -> None:
    secret_text = "private text must not be an error detail"
    with pytest.raises(package.MarkdownPackageError) as error:
        package.parse_markdown_package(
            _zip({"a" * 4000: secret_text.encode()}, manifest=_manifest()), "bad.zip"
        )
    assert secret_text not in str(error.value) and len(str(error.value)) < 1100
    assert secret_text not in error.value.public_message
    assert len(error.value.public_message) < 500


def test_finder_metadata_does_not_change_outer_folder_or_asset_graph() -> None:
    content = _zip(_pair(), manifest=_manifest(), prefix="Paket/")
    stream = io.BytesIO(content)
    with zipfile.ZipFile(stream, "a") as archive:
        archive.writestr(".DS_Store", b"Finder metadata")
        archive.writestr("Paket/.DS_Store", b"Finder metadata")
        archive.writestr("__MACOSX/Paket/._ana.md", b"resource fork")
    result = package.parse_markdown_package(stream.getvalue(), "Finder.zip")
    assert result.status == "ready" and len(result.assets) == 2


def test_ignored_finder_metadata_still_obeys_entry_safety() -> None:
    entry = zipfile.ZipInfo("__MACOSX/._ana.md")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(
        package.MarkdownPackageError, match="encrypted_or_special_zip_entry"
    ):
        package.parse_markdown_package(
            _zip(_pair(), manifest=_manifest(), extra=[entry]), "unsafe-finder.zip"
        )


def test_manifest_duplicate_fields_and_markdown_reference_definitions_are_visible_errors() -> (
    None
):
    with pytest.raises(package.MarkdownPackageError, match="duplicate_manifest_key"):
        package.parse_markdown_package(
            _zip(
                {
                    "manifest.json": b'{"format":"legislation-markdown-v1","format":"other"}'
                }
            ),
            "duplicate.zip",
        )
    with pytest.raises(
        package.MarkdownPackageError, match="duplicate_markdown_reference"
    ):
        package.parse_markdown_package(
            b"[Source][ref]\n\n[ref]: https://example.com/a\n[ref]: https://example.com/b\n",
            "root.md",
        )


@pytest.mark.parametrize("source", ["<a href>Bad</a>", '<a href="root.md">Unclosed'])
def test_invalid_html_links_fail_with_controlled_diagnostics(source: str) -> None:
    with pytest.raises(package.MarkdownPackageError):
        package.parse_markdown_package(source.encode(), "root.md")


def test_archive_count_and_total_size_limits_include_ignored_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package, "MAX_MARKDOWN_ZIP_ENTRIES", 1)
    with pytest.raises(package.MarkdownPackageError, match="markdown_archive_limit"):
        package.parse_markdown_package(_zip(_pair(), manifest=_manifest()), "many.zip")
    monkeypatch.setattr(package, "MAX_MARKDOWN_ZIP_ENTRIES", 128)
    monkeypatch.setattr(package, "MAX_MARKDOWN_PACKAGE_BYTES", 10)
    with pytest.raises(package.MarkdownPackageError, match="markdown_archive_limit"):
        package.parse_markdown_package(
            _zip({".DS_Store": b"metadata bigger than ten bytes"}), "big.zip"
        )


def test_link_limit_counts_external_citations_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package, "MAX_MARKDOWN_LINKS", 1)
    with pytest.raises(package.MarkdownPackageError, match="markdown_link_limit"):
        package.parse_markdown_package(
            b"[A](https://example.com/a) [B](https://example.com/b)", "cites.md"
        )


def test_same_document_fragment_is_an_internal_local_reference() -> None:
    result = package.parse_markdown_package(
        b"# Rule\n[Section](#section)\n## Section\nFull text.\n", "root.md"
    )
    link = result.links[0]
    assert link.kind == "internal"
    assert link.parent_asset_hash == link.target_asset_hash == result.assets[0].sha256
    assert link.requested_url == "root.md" and link.original_url == "#section"


@pytest.mark.parametrize("document_count", [21, 22])
def test_document_count_matches_source_package_admission_bound(
    document_count: int,
) -> None:
    documents = [{"path": "ana.md", "title": "Ana belge"}]
    files = {}
    links = []
    for number in range(1, document_count):
        path = f"ekler/ek-{number}.md"
        documents.append({"path": path, "title": f"EK-{number}", "parent": "ana.md"})
        files[path] = f"# EK-{number}\n\nÖzgün ek hükmü {number}.\n".encode()
        links.append(f"[EK-{number}]({path})")
    files["ana.md"] = ("# Ana belge\n\n" + "\n".join(links)).encode()
    content = _zip(
        files,
        manifest={
            "format": "legislation-markdown-v1",
            "main": "ana.md",
            "documents": documents,
        },
    )
    if document_count == 21:
        result = package.parse_markdown_package(content, "sinir.zip")
        assert len(result.assets) == package.MAX_MARKDOWN_DOCUMENTS == 21
    else:
        with pytest.raises(package.MarkdownPackageError) as failure:
            package.parse_markdown_package(content, "fazla.zip")
        assert failure.value.code == "markdown_document_count_limit"
        assert "20 ek" in failure.value.public_message
        assert "21 belge" in failure.value.public_message
        assert "bağlı ekleri ayrı paketlere bölmeyin" in failure.value.public_message
