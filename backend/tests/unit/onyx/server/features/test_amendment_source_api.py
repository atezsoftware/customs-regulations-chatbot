from io import BytesIO
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import UploadFile

from onyx.db.models import User
from onyx.regulatory.amendments.source_extraction import AmendmentSourceExtraction
from onyx.server.features.regulatory import api as regulatory_api
from onyx.server.features.regulatory.models import AmendmentSourceUrlRequest


def test_url_source_endpoint_returns_normalized_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regulatory_api,
        "fetch_and_extract_amendment_url",
        lambda _url: AmendmentSourceExtraction(
            text="MADDE 1- Yeni metin.",
            source_type="html",
            display_name="20260826-2.htm",
        ),
    )

    result = regulatory_api.extract_amendment_url(
        AmendmentSourceUrlRequest(url="https://example.gov/20260826-2.htm"),
        user=cast(User, SimpleNamespace()),
    )

    assert result.model_dump() == {
        "text": "MADDE 1- Yeni metin.",
        "source_type": "html",
        "display_name": "20260826-2.htm",
    }


def test_pdf_source_endpoint_rejects_non_pdf_bytes() -> None:
    upload = UploadFile(filename="update.pdf", file=BytesIO(b"not a pdf"))

    with pytest.raises(regulatory_api.OnyxError, match="not a valid PDF"):
        regulatory_api.extract_amendment_pdf(upload, user=cast(User, SimpleNamespace()))


def test_docx_source_endpoint_returns_extracted_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regulatory_api,
        "extract_amendment_docx_text",
        lambda _content, _file_name: "MADDE 2- Word metni.",
    )
    upload = UploadFile(filename="değişiklik.docx", file=BytesIO(b"docx bytes"))

    result = regulatory_api.extract_amendment_docx(
        upload, user=cast(User, SimpleNamespace())
    )

    assert result.model_dump() == {
        "text": "MADDE 2- Word metni.",
        "source_type": "docx",
        "display_name": "değişiklik.docx",
    }
