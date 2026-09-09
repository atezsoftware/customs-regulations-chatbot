from io import BytesIO
from typing import NoReturn

import pytest
from pypdf import PdfWriter
from pypdf.annotations import Link


def pdf_bytes(*, linked: bool = False, embedded: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    if linked:
        writer.add_annotation(
            0, Link(rect=(0, 0, 100, 100), url="https://example.gov/annex.pdf")
        )
        writer.add_annotation(0, Link(rect=(0, 100, 100, 200), target_page_index=0))
    if embedded:
        writer.add_attachment("annex.pdf", pdf_bytes())
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_relative_annex_graph_deduplicates_urls_and_ignores_navigation() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    fixtures = {
        "https://example.gov/update": DownloadedSource(
            b'<main><a href="step">Ek listesi</a><a href="step">Ek listesi tekrar</a><a href="privacy">Privacy</a></main>',
            "text/html",
            "https://example.gov/update",
        ),
        "https://example.gov/step": DownloadedSource(
            b'<main><a href="annex.pdf">Ek 1</a></main>',
            "text/html",
            "https://example.gov/step",
        ),
        "https://example.gov/annex.pdf": DownloadedSource(
            pdf_bytes(), "application/pdf", "https://example.gov/annex.pdf"
        ),
    }
    result = acquire_source_package(
        url="https://example.gov/update", fetch=fixtures.pop
    )
    assert result.status == "ready"
    assert len(result.assets) == 3
    assert len(result.links) == 3
    assert not fixtures
    assert result.links[0].original_url == "step"
    assert result.links[0].final_url == "https://example.gov/step"


def test_pdf_annotations_internal_targets_and_embedded_annex_are_evidence() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    result = acquire_source_package(
        content=pdf_bytes(linked=True, embedded=True),
        mime_type="application/pdf",
        fetch=lambda url: DownloadedSource(pdf_bytes(), "application/pdf", url),
    )
    assert result.status == "ready"
    assert (
        len(result.assets) == 2
    )  # identical embedded and downloaded bytes deduplicate
    assert {link.kind for link in result.links} == {"url", "internal", "embedded"}
    assert all(
        link.source_page == 1 for link in result.links if link.kind != "embedded"
    )
    assert all(link.target_asset_hash for link in result.links)


@pytest.mark.parametrize("kind", ["404", "truncated", "redirect_loop"])
def test_missing_annex_never_marks_package_ready(kind: str) -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        SourceAcquisitionError,
        acquire_source_package,
    )

    def fetch(_url: str) -> NoReturn:
        raise SourceAcquisitionError(kind)

    result = acquire_source_package(
        content=b'<main><a href="https://example.gov/annex.pdf">Annex</a></main>',
        mime_type="text/html",
        fetch=fetch,
    )
    assert result.status == "partial"
    assert result.issues[0].code == kind
    assert result.links[0].target_asset_hash is None


def test_relative_annex_without_base_url_is_explicitly_partial() -> None:
    from onyx.regulatory.amendments.annexes.sources import acquire_source_package

    result = acquire_source_package(
        content=b'<main><a href="annex.pdf">Ek 1</a></main>', mime_type="text/html"
    )
    assert result.status == "partial"
    assert result.issues[0].code == "missing_base_url"


def test_depth_and_byte_limits_block_complete_replacement() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    result = acquire_source_package(
        url="https://example.gov/0",
        fetch=lambda url: DownloadedSource(
            f'<a href="https://example.gov/{int(url.rsplit("/", 1)[1]) + 1}">Annex</a>'.encode(),
            "text/html",
            url,
        ),
    )
    assert result.status == "blocked"
    assert result.issues[-1].code == "depth_limit"
    too_large = acquire_source_package(
        content=b"x" * (25 * 1024 * 1024 + 1), mime_type="image/png"
    )
    assert too_large.status == "blocked"
    assert too_large.issues[0].code == "asset_byte_limit"


def test_mime_signature_mismatch_is_rejected() -> None:
    from onyx.regulatory.amendments.annexes.sources import acquire_source_package

    result = acquire_source_package(
        content=b"<html>Not a PDF</html>", mime_type="application/pdf"
    )
    assert result.status == "failed"
    assert result.issues[0].code == "mime_mismatch"


def test_worker_queues_separate_environments_and_databases() -> None:
    from onyx.regulatory.amendments.annexes.config import source_queue_name

    assert source_queue_name(
        environment="dev", database_identity="pg/dev"
    ) != source_queue_name(environment="test", database_identity="pg/dev")
    assert source_queue_name(
        environment="dev", database_identity="pg/dev"
    ) != source_queue_name(environment="dev", database_identity="pg/test")


def test_worker_rejects_foreign_scope_before_acquiring() -> None:
    from onyx.background.celery.tasks.regulatory_amendments.sources import (
        acquire_amendment_sources,
    )

    with pytest.raises(ValueError, match="scope"):
        acquire_amendment_sources.run(
            package_id="00000000-0000-0000-0000-000000000000",
            tenant_id="public",
            environment="other",
            database_identity="other",
        )


def test_source_request_requires_exactly_one_bounded_input() -> None:
    from pydantic import ValidationError

    from onyx.server.features.regulatory.models import (
        CreateAmendmentSourcePackageRequest,
    )

    for values in ({}, {"url": "https://example.gov/a", "text": "duplicate"}):
        with pytest.raises(ValidationError):
            CreateAmendmentSourcePackageRequest(
                document_set_id=1, idempotency_key="request", **values
            )
    assert (
        CreateAmendmentSourcePackageRequest(
            document_set_id=1, idempotency_key="request", text="amendment"
        ).text
        == "amendment"
    )


def test_annex_heading_supplies_context_for_generic_download_label() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    result = acquire_source_package(
        content=b'<main><h2>Ekler</h2><div><a href="https://example.gov/download?id=1">Tiklayiniz</a></div><h2>Help</h2><a href="https://example.gov/privacy">Privacy</a></main>',
        mime_type="text/html",
        fetch=lambda url: DownloadedSource(pdf_bytes(), "application/pdf", url),
    )
    assert result.status == "ready"
    assert len(result.assets) == 2
    assert len(result.links) == 1


def test_invalid_pdf_destination_blocks_readiness() -> None:
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject

    from onyx.regulatory.amendments.annexes.sources import acquire_source_package

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.pages[0][NameObject("/Annots")] = ArrayObject(
        [
            DictionaryObject(
                {
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): ArrayObject([NumberObject(0)] * 4),
                    NameObject("/Dest"): ArrayObject(
                        [NumberObject(999), NameObject("/Fit")]
                    ),
                }
            )
        ]
    )
    stream = BytesIO()
    writer.write(stream)
    result = acquire_source_package(
        content=stream.getvalue(), mime_type="application/pdf"
    )
    assert result.status != "ready"
    assert result.issues[0].code == "missing_internal_target"


def test_pdf_annex_page_does_not_authorize_unrelated_footer_url() -> None:
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from onyx.regulatory.amendments.annexes.sources import acquire_source_package

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 10 250 Td (Annex update) Tj ET")
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    page[NameObject("/Contents")] = content
    writer.add_annotation(
        0, Link(rect=(0, 0, 100, 20), url="https://example.gov/privacy")
    )
    output = BytesIO()
    writer.write(output)
    result = acquire_source_package(
        content=output.getvalue(),
        mime_type="application/pdf",
        fetch=lambda url: pytest.fail(f"Unrelated URL fetched: {url}"),
    )
    assert result.status == "ready"
    assert result.links == []


@pytest.mark.parametrize(
    "status, declared, expected",
    [(404, "4", "404"), (200, "100", "truncated"), (302, "4", "redirect_loop")],
)
def test_bounded_transport_reports_actual_response_failures(
    monkeypatch: pytest.MonkeyPatch, status: int, declared: str, expected: str
) -> None:
    import requests

    from onyx.regulatory.amendments.annexes import sources

    def response_for(url: str, **_kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.url = url
        response.headers.update({"content-length": declared, "location": url})
        response.raw = BytesIO(b"body")
        return response

    monkeypatch.setattr(sources, "ssrf_safe_get", response_for)
    with pytest.raises(sources.SourceAcquisitionError) as captured:
        sources.download_source("https://example.gov/annex")
    assert captured.value.code == expected


def test_transport_rejects_private_metadata_addresses() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        SourceAcquisitionError,
        download_source,
    )

    with pytest.raises(SourceAcquisitionError):
        download_source("http://169.254.169.254/latest/meta-data/")


def test_docx_relationship_preserves_annex_label_and_relative_target() -> None:
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    document = Document()
    paragraph = document.add_paragraph()
    relation = document.part.relate_to(
        "annex.pdf", RELATIONSHIP_TYPE.HYPERLINK, is_external=True
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relation)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Ek 1"
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    output = BytesIO()
    document.save(output)
    result = acquire_source_package(
        content=output.getvalue(),
        url="https://example.gov/update.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        fetch=lambda url: DownloadedSource(pdf_bytes(), "application/pdf", url),
    )
    assert result.status == "ready"
    assert result.links[0].label == "Ek 1"
    assert result.links[0].original_url == "annex.pdf"
    assert result.links[0].final_url == "https://example.gov/annex.pdf"


def test_pdf_file_attachment_annotation_is_acquired() -> None:
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        TextStringObject,
    )

    from onyx.regulatory.amendments.annexes.sources import acquire_source_package

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    stream.set_data(pdf_bytes())
    file_spec = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Filespec"),
            NameObject("/F"): TextStringObject("annex.pdf"),
            NameObject("/EF"): DictionaryObject({NameObject("/F"): stream}),
        }
    )
    page[NameObject("/Annots")] = ArrayObject(
        [
            DictionaryObject(
                {
                    NameObject("/Subtype"): NameObject("/FileAttachment"),
                    NameObject("/FS"): file_spec,
                }
            )
        ]
    )
    output = BytesIO()
    writer.write(output)
    result = acquire_source_package(
        content=output.getvalue(), mime_type="application/pdf"
    )
    assert result.status == "ready"
    assert len(result.assets) == 2
    assert result.links[0].source_page == 1
    assert result.links[0].kind == "embedded"


def test_failed_pdf_download_cannot_be_accepted_as_html_annex() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    result = acquire_source_package(
        content=b'<main><a href="https://example.gov/annex.pdf">Ek 1</a></main>',
        mime_type="text/html",
        fetch=lambda url: DownloadedSource(
            b"<html>Document unavailable</html>", "text/html", url
        ),
    )
    assert result.status == "partial"
    assert result.issues[0].code == "mime_mismatch"


def test_duplicate_downloads_still_consume_package_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import sources

    root = b'<a href="https://example.gov/a.pdf">Annex A</a><a href="https://example.gov/b.pdf">Annex B</a>'
    pdf = pdf_bytes()
    monkeypatch.setattr(sources, "MAX_PACKAGE_BYTES", len(root) + len(pdf) + 1)
    result = sources.acquire_source_package(
        content=root,
        mime_type="text/html",
        fetch=lambda url: sources.DownloadedSource(pdf, "application/pdf", url),
    )
    assert result.status == "blocked"
    assert result.issues[0].code == "package_byte_limit"


def test_worker_download_has_a_hard_subprocess_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import time

    from onyx.regulatory.amendments.annexes import sources

    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("download", 1)

    monkeypatch.setattr(sources.subprocess, "run", timed_out)
    with pytest.raises(sources.SourceAcquisitionError) as captured:
        sources.download_source_bounded(
            "https://example.gov/annex.pdf", deadline=time.monotonic() + 1
        )
    assert captured.value.code == "download_time_limit"


def test_identical_html_at_distinct_bases_preserves_each_relative_annex() -> None:
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        acquire_source_package,
    )

    urls = [
        "https://example.gov/a/step",
        "https://example.gov/b/step",
        "https://example.gov/a/annex.pdf",
        "https://example.gov/b/annex.pdf",
    ]
    fixtures = {
        url: DownloadedSource(
            b'<a href="annex.pdf">Ek 1</a>' if url.endswith("step") else pdf_bytes(),
            "text/html" if url.endswith("step") else "application/pdf",
            url,
        )
        for url in urls
    }
    result = acquire_source_package(
        content=b'<a href="https://example.gov/a/step">Annex A</a><a href="https://example.gov/b/step">Annex B</a>',
        mime_type="text/html",
        fetch=fixtures.pop,
    )
    assert result.status == "ready"
    assert not fixtures
    assert len(result.assets) == 3
