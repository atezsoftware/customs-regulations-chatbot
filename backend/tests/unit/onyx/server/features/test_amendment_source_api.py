from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

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
        user=SimpleNamespace(),
    )

    assert result.model_dump() == {
        "text": "MADDE 1- Yeni metin.",
        "source_type": "html",
        "display_name": "20260826-2.htm",
    }


def test_pdf_source_endpoint_rejects_non_pdf_bytes() -> None:
    upload = UploadFile(filename="update.pdf", file=BytesIO(b"not a pdf"))

    with pytest.raises(regulatory_api.OnyxError, match="not a valid PDF"):
        regulatory_api.extract_amendment_pdf(upload, user=SimpleNamespace())
