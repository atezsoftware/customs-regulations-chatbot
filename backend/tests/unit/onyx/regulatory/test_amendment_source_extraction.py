import pytest

from onyx.regulatory.amendments import source_extraction
from onyx.regulatory.amendments.source_extraction import (
    AmendmentSourceExtractionError,
    extract_amendment_html,
    extract_amendment_pdf,
    fetch_and_extract_amendment_url,
)


class _Response:
    def __init__(self, content: bytes, content_type: str | None = None) -> None:
        self.headers = {"content-type": content_type or ""}
        self.url = "https://example.gov/update"
        self._content = content

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self._content[index : index + chunk_size]
            for index in range(0, len(self._content), chunk_size)
        ]

    def raise_for_status(self) -> None:
        return None


def test_extract_amendment_html_removes_page_chrome() -> None:
    result = extract_amendment_html(
        b"<html><nav>Menu</nav><main><h1>TEBLIG</h1><p>MADDE 1- Yeni metin.</p></main><script>bad()</script></html>",
        "text/html; charset=utf-8",
    )

    assert result == "TEBLIG\n\nMADDE 1- Yeni metin."


def test_extract_amendment_pdf_rejects_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_extraction, "extract_file_text", lambda *_args, **_kwargs: ""
    )

    with pytest.raises(AmendmentSourceExtractionError, match="text-searchable"):
        extract_amendment_pdf(b"%PDF-1.7", "update.pdf")


def test_fetch_url_detects_pdf_by_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_extraction,
        "ssrf_safe_get",
        lambda *_args, **_kwargs: _Response(b"%PDF-1.7"),
    )
    monkeypatch.setattr(
        source_extraction,
        "extract_amendment_pdf",
        lambda *_args, **_kwargs: "MADDE 1- Yeni metin.",
    )

    result = fetch_and_extract_amendment_url("https://example.gov/update")

    assert result.text == "MADDE 1- Yeni metin."
    assert result.source_type == "pdf"
    assert result.display_name == "update"


def test_fetch_url_rejects_oversized_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_extraction,
        "ssrf_safe_get",
        lambda *_args, **_kwargs: _Response(
            b"x" * (source_extraction.MAX_AMENDMENT_SOURCE_BYTES + 1),
            "text/html",
        ),
    )

    with pytest.raises(AmendmentSourceExtractionError, match="too large"):
        fetch_and_extract_amendment_url("https://example.gov/update.htm")
