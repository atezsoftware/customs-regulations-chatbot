import hashlib
from io import BytesIO
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from onyx.connectors.models import Document
from onyx.file_processing.original_ingestion import LoadedUserFileDocuments
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import (
    build_intake_document,
    prepare_intake_file,
)


def source_asset(
    package_id: UUID, *, name: str = "regulation.html", content: bytes = b"original"
) -> IntakeSourceAsset:
    return IntakeSourceAsset(
        id=uuid4(),
        package_id=package_id,
        sha256=hashlib.sha256(content).hexdigest(),
        file_id=f"original-{name}",
        text_file_id=f"text-{name}",
        text_sha256=hashlib.sha256(b"MADDE 1 - Original rule.").hexdigest(),
        mime_type="application/pdf",
        display_name=name,
        byte_count=len(content),
        original_url=f"https://example.gov/{name}",
        final_url=f"https://example.gov/current/{name}",
    )


def test_child_keeps_own_text_title_url_and_parent_context() -> None:
    package_id = uuid4()
    parent = source_asset(package_id)
    child = source_asset(package_id, name="download.pdf", content=b"child-original")
    link = SourceLink(
        parent_asset_hash=parent.sha256,
        target_asset_hash=child.sha256,
        source_field="html:0",
        label="Ekleri için tıklayınız",
        kind="url",
        original_url="download.pdf",
        requested_url=child.original_url,
        final_url=child.final_url,
    )
    child_text = "| Tariff | Rate |\n| 01 | 5% |"
    document = build_intake_document(
        package_id=package_id,
        asset=child,
        user_file_id=uuid4(),
        text=child_text,
        links=[link],
        package_assets=[parent, child],
    )
    assert document.title == "download.pdf"
    assert document.get_text_content() == child_text
    assert document.sections[0].link == child.final_url
    assert document.doc_metadata is not None
    context = document.doc_metadata["regulatory_document_intake"]
    assert context["parent_asset_ids"] == [str(parent.id)]
    assert context["parent_titles"] == ["regulation.html"]
    assert context["source_links"][0]["label"] == "Ekleri için tıklayınız"


def test_preparation_verifies_original_and_supplies_initial_ingestion_receipt() -> None:
    package_id, file_id = uuid4(), uuid4()
    asset = source_asset(package_id)
    stored = {
        asset.file_id: b"original",
        asset.text_file_id: b"MADDE 1 - Original rule.",
    }
    store = cast(FileStore, SimpleNamespace(read_file=lambda key: BytesIO(stored[key])))
    prepared: list[Document] = []
    plaintext: list[str] = []

    def chunk(identifier: UUID, tenant: str, documents: list[Document]) -> bool:
        assert (identifier, tenant) == (file_id, "tenant")
        assert isinstance(documents, LoadedUserFileDocuments)
        assert documents.original_extraction.file_id == asset.file_id
        assert documents.original_extraction.raw_sha256 == asset.sha256
        prepared.extend(documents)
        return True

    result = prepare_intake_file(
        package_id=package_id,
        asset=asset,
        user_file_id=file_id,
        tenant_id="tenant",
        links=[],
        store=store,
        chunk_file=chunk,
        store_plaintext=lambda _identifier, text: plaintext.append(text) is None,
    )
    assert result.initial_chunks_created
    assert prepared[0].id == str(file_id)
    assert prepared[0].file_id == asset.file_id
    assert plaintext == ["MADDE 1 -\nOriginal rule."]
    assert result.plaintext_sha256 == hashlib.sha256(plaintext[0].encode()).hexdigest()


@pytest.mark.parametrize("corrupt", ["original", "text", "byte_count"])
def test_corrupt_prepared_asset_never_reaches_chunking(corrupt: str) -> None:
    package_id = uuid4()
    asset = source_asset(package_id)
    if corrupt == "byte_count":
        asset = asset.model_copy(update={"byte_count": asset.byte_count + 1})
    stored = {
        asset.file_id: b"changed" if corrupt == "original" else b"original",
        asset.text_file_id: (
            b"changed" if corrupt == "text" else b"MADDE 1 - Original rule."
        ),
    }
    store = cast(FileStore, SimpleNamespace(read_file=lambda key: BytesIO(stored[key])))
    with pytest.raises(ValueError, match="integrity|byte count"):
        prepare_intake_file(
            package_id=package_id,
            asset=asset,
            user_file_id=uuid4(),
            tenant_id="tenant",
            links=[],
            store=store,
            chunk_file=lambda *_args: pytest.fail("Corrupt source was chunked"),
        )


def test_foreign_package_and_empty_text_are_rejected() -> None:
    package_id = uuid4()
    asset = source_asset(package_id)
    with pytest.raises(ValueError, match="another intake package"):
        build_intake_document(
            package_id=uuid4(), asset=asset, user_file_id=uuid4(), text="rule", links=[]
        )
    with pytest.raises(ValueError, match="no prepared text"):
        build_intake_document(
            package_id=package_id,
            asset=asset,
            user_file_id=uuid4(),
            text="  ",
            links=[],
        )


def test_html_canonical_input_retains_table_headers_rows_and_columns() -> None:
    package_id = uuid4()
    raw = (
        b"<html><body><nav>Navigation</nav><main><h1>Tariff table</h1>"
        b"<table><tr><th>GTIP</th><th>Rate</th></tr>"
        b"<tr><td>0101</td><td>5%</td></tr>"
        b"<tr><td>0202</td><td>8%</td></tr></table>"
        b"</main><footer>Footer</footer></body></html>"
    )
    prepared_text = b"Tariff table\nGTIP\nRate\n0101\n5%\n0202\n8%"
    asset = source_asset(package_id, content=raw).model_copy(
        update={
            "mime_type": "text/html",
            "text_sha256": hashlib.sha256(prepared_text).hexdigest(),
        }
    )
    stored = {asset.file_id: raw, asset.text_file_id: prepared_text}
    store = cast(FileStore, SimpleNamespace(read_file=lambda key: BytesIO(stored[key])))
    documents: list[Document] = []
    result = prepare_intake_file(
        package_id=package_id,
        asset=asset,
        user_file_id=uuid4(),
        tenant_id="tenant",
        links=[],
        store=store,
        chunk_file=lambda _identifier, _tenant, items: documents.extend(items) is None,
        store_plaintext=lambda *_args: True,
    )
    text = documents[0].get_text_content()
    lines = [line.strip() for line in text.splitlines()]
    assert "GTIP\tRate" in lines
    assert "0101\t5%" in lines
    assert "0202\t8%" in lines
    assert "Navigation" not in text and "Footer" not in text
    assert result.plaintext_sha256 != asset.text_sha256


def test_utf8_byte_limit_does_not_reject_valid_multibyte_character_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import preparation

    monkeypatch.setattr(preparation, "MAX_TEXT_CHARS", 32)
    text = ("ğ" * 24).encode()
    package_id = uuid4()
    asset = source_asset(package_id).model_copy(
        update={"text_sha256": hashlib.sha256(text).hexdigest()}
    )
    stored = {asset.file_id: b"original", asset.text_file_id: text}
    store = cast(FileStore, SimpleNamespace(read_file=lambda key: BytesIO(stored[key])))
    result = prepare_intake_file(
        package_id=package_id,
        asset=asset,
        user_file_id=uuid4(),
        tenant_id="tenant",
        links=[],
        store=store,
        chunk_file=lambda *_args: True,
        store_plaintext=lambda *_args: True,
    )
    assert result.initial_chunks_created


@pytest.mark.parametrize("mime_type", ["application/pdf", "text/html", "text/markdown"])
def test_native_markdown_heading_preservation_does_not_change_converted_format_normalization(
    mime_type: str,
) -> None:
    package_id = uuid4()
    asset = source_asset(package_id).model_copy(update={"mime_type": mime_type})
    text = "# Ana belge\n\n## MADDE 1 — Başvuru elektronik ortamda verilir."
    document = build_intake_document(
        package_id=package_id, asset=asset, user_file_id=uuid4(), text=text, links=[]
    )
    canonical = document.get_text_content()
    if mime_type == "text/markdown":
        assert canonical == text
    else:
        assert "## MADDE 1 —\nBaşvuru elektronik ortamda verilir." in canonical
        assert "".join(canonical.split()) == "".join(text.split())


@pytest.mark.parametrize("mime_type", ["application/pdf", "text/html", "text/markdown"])
def test_navigation_cleanup_runs_before_chunking_for_every_new_intake_format(
    mime_type: str,
) -> None:
    package_id = uuid4()
    prepared_text = (
        "MADDE 1 — Beyanname zorunludur.\n\n[Ekler için tıklayınız](ek-1.pdf)".encode()
    )
    raw = (
        b'<main><h1>Ana belge</h1><p>MADDE 1 - Beyanname zorunludur.</p><p><a href="ek-1.pdf">Ekler icin tiklayiniz</a></p></main>'
        if mime_type == "text/html"
        else b"original-file-bytes"
    )
    parent = source_asset(package_id, content=raw).model_copy(
        update={
            "mime_type": mime_type,
            "text_sha256": hashlib.sha256(prepared_text).hexdigest(),
        }
    )
    child = source_asset(package_id, name="EK-1.pdf", content=b"annex-original")
    link = SourceLink(
        parent_asset_hash=parent.sha256,
        target_asset_hash=child.sha256,
        original_url="ek-1.pdf",
        requested_url=child.original_url,
        label="Ekler için tıklayınız",
        kind="url",
        source_field="href",
    )
    original_link = link.model_dump()
    stored = {parent.file_id: raw, parent.text_file_id: prepared_text}
    store = cast(FileStore, SimpleNamespace(read_file=lambda key: BytesIO(stored[key])))
    captured: list[Document] = []
    prepare_intake_file(
        package_id=package_id,
        asset=parent,
        user_file_id=uuid4(),
        links=[link],
        package_assets=[parent, child],
        tenant_id="tenant",
        store=store,
        chunk_file=lambda _id, _tenant, documents: captured.extend(documents) is None,
        store_plaintext=lambda *_args: True,
    )
    canonical_input = captured[0].get_text_content()
    assert (
        "tıklayınız" not in canonical_input.lower()
        and "tiklayiniz" not in canonical_input.lower()
    )
    assert "EK-1" in canonical_input and "Beyanname zorunludur." in canonical_input
    assert stored[parent.file_id] == raw
    assert link.model_dump() == original_link
