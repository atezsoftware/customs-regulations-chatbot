"""Mixed ZIP acquisition preserves file identity and a source-backed forest."""

import hashlib
import io
import json
import stat
import zipfile
from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from onyx.regulatory.amendments.annexes.sources import (
    DownloadedSource,
    SourceAcquisitionError,
    inspect_source,
)
from onyx.regulatory.document_intake import zip_package as zip_source


def archive(
    files: dict[str, bytes], *, manifest: Mapping[str, object] | None = None
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as result:
        for path, raw in files.items():
            result.writestr(path, raw)
        if manifest is not None:
            result.writestr("manifest.json", json.dumps(manifest))
    return output.getvalue()


def pdf() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.write(output)
    return output.getvalue()


def docx() -> bytes:
    return archive(
        {
            "[Content_Types].xml": b"<Types/>",
            "word/document.xml": b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>MADDE 4 - Office metni</w:t></w:r></w:p></w:body></w:document>',
        }
    )


@pytest.fixture(autouse=True)
def _in_process_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    def isolated(
        fn: Callable[..., Any], *args: Any, timeout: float, **kwargs: Any
    ) -> Any:
        assert timeout > 0
        return fn(*args, **kwargs)

    monkeypatch.setattr(zip_source, "run_in_isolated_process", isolated)
    monkeypatch.setattr(
        zip_source,
        "_bounded_inspect",
        lambda raw, declared, _deadline: inspect_source(raw, declared),
    )


def test_mixed_archive_uses_actual_types_and_each_independent_document_is_root() -> (
    None
):
    files = {
        "folder/b.pdf": pdf(),
        "a.docx": docx(),
        "main.html": b"<html><body><p>MADDE 1 - HTML metni</p></body></html>",
        "source.md": "# Mevzuat\n\nMADDE 2 - Sağlanan metin.\n".encode(),
    }
    result = zip_source.acquire_zip_package(archive(files), "mixed.zip")
    assert result.status == "ready" and result.issues == []
    assert {asset.sha256 for asset in result.assets} == set(
        result.intake_graph.root_hashes
    )
    assert result.intake_graph.primary_parent_by_hash == {}
    assets = {asset.display_name: asset for asset in result.assets}
    assert {asset.mime_type for asset in result.assets} == {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/html",
        "text/markdown",
    }
    assert assets["source.md"].text.encode() == files["source.md"]
    assert "Office metni" in assets["a.docx"].text
    assert assets["folder/b.pdf"].page_count == 1
    for path, raw in files.items():
        assert assets[path].content == raw
        assert assets[path].sha256 == hashlib.sha256(raw).hexdigest()
    assert "mixed.zip" not in assets


def test_local_annex_and_backlink_keep_separate_files_and_correct_parent() -> None:
    files = {
        "root.md": b"# Main\n\n[Ekler icin tiklayiniz](annex/ek-1.html)\n",
        "annex/ek-1.html": b"<html><body><p>EK-1 table</p></body></html>",
        "other.md": b"# Other law\n\n[Main source](root.md)\n",
    }
    result = zip_source.acquire_zip_package(archive(files), "mixed.zip")
    hashes = {path: hashlib.sha256(raw).hexdigest() for path, raw in files.items()}
    assert result.intake_graph.primary_parent_by_hash == {
        hashes["annex/ek-1.html"]: hashes["root.md"]
    }
    assert set(result.intake_graph.root_hashes) == {
        hashes["root.md"],
        hashes["other.md"],
    }
    assert len(result.links) == 2
    assert result.links[0].original_url == "annex/ek-1.html"


def test_markdown_link_preserves_literal_url_and_resolves_its_decoded_path() -> None:
    result = zip_source.acquire_zip_package(
        archive(
            {
                "main.md": b"# Main\n[EK-1](ek&amp;%C3%A7.html)",
                "ek&ç.html": b"<html><p>Exact annex content</p></html>",
            }
        ),
        "source.zip",
    )
    assert result.status == "ready"
    assert result.links[0].original_url == "ek&amp;%C3%A7.html"
    assert result.links[0].requested_url == "ek&ç.html"


def test_explicit_manifest_requires_actual_source_link_and_can_resolve_shared_annex() -> (
    None
):
    files = {
        "one.md": b"# One\n[EK-1](shared.md)",
        "two.md": b"# Two\n[EK-1](shared.md)",
        "shared.md": b"# Shared annex\nData",
    }
    with pytest.raises(zip_source.ZipPackageError, match="invalid_zip_graph"):
        zip_source.acquire_zip_package(archive(files), "source.zip")
    manifest = {
        "documents": [
            {"path": "one.md"},
            {"path": "two.md"},
            {"path": "shared.md", "parent": "one.md"},
        ]
    }
    result = zip_source.acquire_zip_package(
        archive(files, manifest=manifest), "source.zip"
    )
    assert result.intake_graph.primary_parent_by_hash == {
        hashlib.sha256(files["shared.md"]).hexdigest(): hashlib.sha256(
            files["one.md"]
        ).hexdigest()
    }
    files["one.md"] = b"# One without a source link"
    with pytest.raises(zip_source.ZipPackageError, match="invalid_zip_graph"):
        zip_source.acquire_zip_package(archive(files, manifest=manifest), "source.zip")


def test_exact_remote_alias_uses_included_file_without_downloading() -> None:
    files = {
        "main.md": b"# Main\n[EK-1](https://example.com/attachment)",
        "annex.html": b"<html><p>Annex content</p></html>",
    }
    manifest = {
        "documents": [
            {"path": "main.md"},
            {
                "path": "annex.html",
                "parent": "main.md",
                "source_url": "https://example.com/attachment",
            },
        ]
    }
    fetch = MagicMock(side_effect=AssertionError("Included exact URL must not fetch"))
    result = zip_source.acquire_zip_package(
        archive(files, manifest=manifest), "source.zip", fetch=fetch
    )
    assert result.status == "ready" and len(result.assets) == 2
    fetch.assert_not_called()


def test_remote_annex_uses_existing_download_contract_and_shared_depth_limit() -> None:
    files = {"main.md": b"# Main\n[EK-1](https://example.com/one.html)"}
    fetch = MagicMock(
        side_effect=lambda address: DownloadedSource(
            f'<html><p>EK next <a href="https://example.com/{"two" if "one" in address else "three"}.html">EK</a></p></html>'.encode(),
            "text/html",
            address,
        )
    )
    result = zip_source.acquire_zip_package(archive(files), "source.zip", fetch=fetch)
    assert result.status == "blocked" and len(result.assets) == 3
    assert [issue.code for issue in result.issues] == ["depth_limit"]
    assert fetch.call_count == 2


@pytest.mark.parametrize("missing_second_annex", [False, True])
def test_identical_remote_content_expands_each_distinct_base_url(
    missing_second_annex: bool,
) -> None:
    listing = b'<html><p><a href="ek.html">EK-1</a></p></html>'
    fixtures = {
        "https://example.com/a/list.html": listing,
        "https://example.com/b/list.html": listing,
        "https://example.com/a/ek.html": b"<html><p>A annex norm</p></html>",
        "https://example.com/b/ek.html": b"<html><p>B annex norm</p></html>",
    }
    if missing_second_annex:
        fixtures.pop("https://example.com/b/ek.html")
    visited: list[str] = []

    def fetch(address: str) -> DownloadedSource:
        visited.append(address)
        if address not in fixtures:
            raise SourceAcquisitionError("download_failed")
        return DownloadedSource(fixtures[address], "text/html", address)

    result = zip_source.acquire_zip_package(
        archive(
            {
                "main.md": b"# Main\n[EK-A](https://example.com/a/list.html)\n[EK-B](https://example.com/b/list.html)",
            }
        ),
        "source.zip",
        fetch=fetch,
    )
    assert set(visited) == set(fixtures) | {"https://example.com/b/ek.html"}
    assert len(result.assets) == (3 if missing_second_annex else 4)
    assert len(result.links) == 4
    aliases = [
        link
        for link in result.links
        if link.parent_asset_hash == hashlib.sha256(listing).hexdigest()
    ]
    assert {link.parent_url for link in aliases} == {
        "https://example.com/a/list.html",
        "https://example.com/b/list.html",
    }
    assert {link.requested_url for link in aliases} == {
        "https://example.com/a/ek.html",
        "https://example.com/b/ek.html",
    }
    if missing_second_annex:
        assert result.status == "blocked"
        assert result.issues[0].code == "download_failed"
        assert result.issues[0].locator == "https://example.com/b/ek.html"
    else:
        assert result.status == "ready" and not result.issues


def test_downloaded_duplicate_bytes_share_package_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = b"# Main\n[EK-A](https://example.com/a.html)\n[EK-B](https://example.com/b.html)"
    same = b"<html><p>Same norm</p></html>"
    monkeypatch.setattr(zip_source, "MAX_PACKAGE_BYTES", len(main) + len(same) * 2 - 1)
    fetch = MagicMock(
        side_effect=lambda address: DownloadedSource(same, "text/html", address)
    )
    result = zip_source.acquire_zip_package(
        archive({"main.md": main}), "source.zip", fetch=fetch
    )
    assert result.status == "blocked"
    assert result.issues[0].code == "package_byte_limit"
    assert len(result.assets) == 2 and fetch.call_count == 2


def test_remote_failure_is_explicit_and_ordinary_external_markdown_link_is_not_an_annex() -> (
    None
):
    files = {
        "main.md": b"# Main\n[EK-1](https://example.com/annex.pdf)\n[Reference](https://example.com/statute)"
    }
    fetch = MagicMock(side_effect=SourceAcquisitionError("unsafe_url"))
    result = zip_source.acquire_zip_package(archive(files), "source.zip", fetch=fetch)
    assert result.status == "blocked"
    assert result.issues[0].code == "unsafe_url"
    assert len(result.links) == 1 and result.links[0].target_asset_hash is None
    fetch.assert_called_once_with("https://example.com/annex.pdf")


@pytest.mark.parametrize("target", ["missing.pdf", "../outside.pdf", "wrongcase.HTML"])
def test_local_missing_or_unsafe_target_never_guesses_or_fetches(target: str) -> None:
    fetch = MagicMock(side_effect=AssertionError("Missing local file cannot fetch"))
    result = zip_source.acquire_zip_package(
        archive(
            {
                "main.md": f"# Main\n[EK-1]({target})".encode(),
                "wrongcase.html": b"<html><p>Annex</p></html>",
            }
        ),
        "source.zip",
        fetch=fetch,
    )
    assert result.status == "blocked"
    assert result.issues[0].code in {"missing_local_zip_target", "unsafe_local_link"}
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "files",
    [
        {"a.md": b"# Same", "b.md": b"# Same"},
        {"a.md": b"# One", "A.md": b"# Two"},
        {"../outside.md": b"# Unsafe"},
        {"a.md": b"# A\n[EK-1](b.md)", "b.md": b"# B\n[EK-2](a.md)"},
    ],
)
def test_duplicate_path_content_traversal_and_cycle_fail_closed(
    files: dict[str, bytes],
) -> None:
    with pytest.raises(zip_source.ZipPackageError):
        zip_source.acquire_zip_package(archive(files), "source.zip")


def test_unsupported_member_blocks_whole_package_and_reports_path() -> None:
    result = zip_source.acquire_zip_package(
        archive({"valid.md": b"# Valid law", "binary.exe": b"MZ\x00binary"}),
        "source.zip",
    )
    assert result.status == "blocked" and len(result.assets) == 1
    assert result.issues[0].locator == "binary.exe"
    assert result.issues[0].code == "unsupported_format"


def test_symlink_and_count_limit_rejected_before_inspection() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as result:
        info = zipfile.ZipInfo("link.md")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        result.writestr(info, "target")
    with pytest.raises(zip_source.ZipPackageError, match="unsafe_zip"):
        zip_source.acquire_zip_package(output.getvalue(), "source.zip")
    with pytest.raises(zip_source.ZipPackageError, match="asset_count_limit"):
        zip_source.acquire_zip_package(
            archive({f"{index}.md": f"# Law {index}".encode() for index in range(22)}),
            "source.zip",
        )


def test_all_invalid_members_report_bounded_member_error() -> None:
    with pytest.raises(zip_source.ZipPackageError) as raised:
        zip_source.acquire_zip_package(archive({"bad.pdf": b"not a pdf"}), "source.zip")
    assert raised.value.path == "bad.pdf"
    assert "bad.pdf" in raised.value.public_message


@pytest.mark.parametrize(
    "content", [b"# Text\n![Figure](figure.png)", b"%PDF-disguised as Markdown"]
)
def test_markdown_binary_and_untranscribed_image_are_explicit_failures(
    content: bytes,
) -> None:
    with pytest.raises(zip_source.ZipPackageError):
        zip_source.acquire_zip_package(archive({"source.md": content}), "source.zip")


def test_worker_retry_rereads_original_archive_and_markdown_never_uses_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from uuid import uuid4

    from onyx.regulatory.amendments import analysis_llm
    from onyx.regulatory.amendments.annexes import job

    files = {"one.md": b"# One\nMADDE 1 - One.", "two.md": b"# Two\nMADDE 2 - Two."}
    uploaded = archive(files)
    first_hash = hashlib.sha256(files["one.md"]).hexdigest()
    previous_manifest = json.dumps(
        {"assets": [{"sha256": first_hash, "mime_type": "text/markdown"}], "links": []}
    ).encode()
    blobs = {
        "zip-input": uploaded,
        "old-manifest": previous_manifest,
        "first-md": files["one.md"],
    }
    store = MagicMock()
    store.read_file.side_effect = lambda identifier: io.BytesIO(blobs[identifier])

    def save(stream: io.BytesIO, **_kwargs: object) -> str:
        identifier = f"saved-{len(blobs)}"
        blobs[identifier] = stream.read()
        return identifier

    store.save_file.side_effect = save
    package = SimpleNamespace(
        input_spec={
            "kind": zip_source.ZIP_INPUT_KIND,
            "input_sha256": hashlib.sha256(uploaded).hexdigest(),
            "display_name": "source.zip",
        },
        input_file_id="zip-input",
        manifest_file_id="old-manifest",
        manifest_sha256=hashlib.sha256(previous_manifest).hexdigest(),
    )
    existing = SimpleNamespace(
        sha256=first_hash,
        file_id="first-md",
        mime_type="text/markdown",
        display_name="one.md",
        byte_count=len(files["one.md"]),
        original_url=None,
        final_url=None,
        text_file_id=None,
        text_sha256=None,
    )
    monkeypatch.setattr(
        job, "claim_source_package", MagicMock(return_value=(package, uuid4()))
    )
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job, "list_source_assets", MagicMock(return_value=[existing]))
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(
        job, "extend_source_package_lease", MagicMock(return_value=True)
    )
    finish = MagicMock(return_value=True)
    monkeypatch.setattr(job, "finish_source_package", finish)
    no_conversion = MagicMock(
        side_effect=AssertionError("Supplied Markdown must not convert")
    )
    monkeypatch.setattr(analysis_llm, "get_amendment_analysis_llm", no_conversion)
    monkeypatch.setattr(job, "acquire_source_package", no_conversion)
    monkeypatch.setattr(job, "parse_markdown_package", no_conversion)
    job.run_source_package(package_id=uuid4(), environment="local-test")
    result = finish.call_args.kwargs["result"]
    assert {asset.content for asset in result.assets} == set(files.values())
    frozen = json.loads(blobs[finish.call_args.kwargs["manifest_file_id"]])
    assert frozen["intake_graph"]["root_hashes"] == sorted(
        hashlib.sha256(raw).hexdigest() for raw in files.values()
    )
    assert frozen["input_sha256"] == hashlib.sha256(uploaded).hexdigest()
    assert blobs["zip-input"] == uploaded
    no_conversion.assert_not_called()


def test_worker_changed_original_archive_fails_before_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from uuid import uuid4

    from onyx.regulatory.amendments.annexes import job

    uploaded = archive({"one.md": b"# One"})
    package = SimpleNamespace(
        input_spec={
            "kind": zip_source.ZIP_INPUT_KIND,
            "input_sha256": "0" * 64,
            "display_name": "source.zip",
        },
        input_file_id="zip-input",
        manifest_file_id=None,
    )
    store = MagicMock()
    store.read_file.side_effect = lambda _identifier: io.BytesIO(uploaded)
    monkeypatch.setattr(
        job, "claim_source_package", MagicMock(return_value=(package, uuid4()))
    )
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job, "list_source_assets", MagicMock(return_value=[]))
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    acquire = MagicMock()
    failed = MagicMock()
    monkeypatch.setattr(job, "acquire_zip_package", acquire)
    monkeypatch.setattr(job, "fail_zip_source_package", failed)
    with pytest.raises(zip_source.ZipPackageError, match="zip_input_integrity_changed"):
        job.run_source_package(package_id=uuid4(), environment="local-test")
    acquire.assert_not_called()
    assert "ZIP" in failed.call_args.kwargs["detail"]
