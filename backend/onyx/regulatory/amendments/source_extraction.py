"""Normalize one amendment HTML or PDF source into reviewable text."""

import atexit
import io
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from onyx.configs.app_configs import (
    MAX_AMENDMENT_SOURCE_BYTES,
    MAX_AMENDMENT_SOURCE_TEXT_CHARS,
    MIN_AMENDMENT_PDF_TEXT_CHARS,
)
from onyx.file_processing.extract_file_text import extract_file_text
from onyx.utils.url import ssrf_safe_get
from onyx.utils.web_content import (
    decode_html_bytes,
    has_pdf_signature,
    is_pdf_resource,
    title_from_url,
)

_DOWNLOAD_CHUNK_SIZE = 64 * 1024
_URL_TIMEOUT_SECONDS = (5, 20)
_NON_CONTENT_TAGS = ("script", "style", "template", "noscript", "nav", "footer")
_AMENDMENT_SOURCE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
}
_RESMI_GAZETE_HOST = "resmigazete.gov.tr"
_RESMI_GAZETE_INTERMEDIATE_CA = (
    Path(__file__).with_name("certs") / "geotrust_tls_rsa_ca_g1.pem"
)


class AmendmentSourceExtractionError(ValueError):
    """A supplied amendment source cannot safely produce usable text."""


@dataclass(frozen=True)
class AmendmentSourceExtraction:
    text: str
    source_type: Literal["html", "pdf"]
    display_name: str


def _is_resmi_gazete_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    return hostname == _RESMI_GAZETE_HOST or hostname.endswith(f".{_RESMI_GAZETE_HOST}")


def _remove_temporary_ca_bundle(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@lru_cache(maxsize=1)
def _resmi_gazete_ca_bundle_path() -> str:
    """Combine Requests' trust store with the intermediate omitted by the site."""
    default_bundle = Path(requests.certs.where()).read_bytes()
    intermediate = _RESMI_GAZETE_INTERMEDIATE_CA.read_bytes()

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="onyx-resmigazete-ca-",
        suffix=".pem",
        delete=False,
    ) as bundle:
        bundle.write(default_bundle)
        if not default_bundle.endswith(b"\n"):
            bundle.write(b"\n")
        bundle.write(intermediate)
        bundle_path = bundle.name

    atexit.register(_remove_temporary_ca_bundle, bundle_path)
    return bundle_path


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise AmendmentSourceExtractionError("The source did not contain usable text.")
    if len(normalized) > MAX_AMENDMENT_SOURCE_TEXT_CHARS:
        raise AmendmentSourceExtractionError(
            "The extracted text is too large to analyze. Use a smaller or more specific source."
        )
    return normalized


def extract_amendment_html(content: bytes, content_type: str | None) -> str:
    """Extract readable main/body text while discarding common page chrome."""
    soup = BeautifulSoup(decode_html_bytes(content, content_type), "html.parser")
    for tag in soup.find_all(_NON_CONTENT_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    return _normalize_text(root.get_text("\n\n", strip=True))


def extract_amendment_pdf(content: bytes, file_name: str) -> str:
    """Extract text from a valid PDF and reject image-only documents."""
    if len(content) > MAX_AMENDMENT_SOURCE_BYTES:
        raise AmendmentSourceExtractionError(
            "The PDF is too large to upload for analysis."
        )
    if not has_pdf_signature(content):
        raise AmendmentSourceExtractionError("The uploaded file is not a valid PDF.")

    extracted_text = extract_file_text(io.BytesIO(content), file_name, extension=".pdf")
    if (
        not extracted_text.strip()
        or len(extracted_text.strip()) < MIN_AMENDMENT_PDF_TEXT_CHARS
    ):
        raise AmendmentSourceExtractionError(
            "This PDF does not contain enough text to analyze. Upload a text-searchable PDF or paste OCR output."
        )
    return _normalize_text(extracted_text)


def _read_response_content(response: requests.Response) -> bytes:
    response.raise_for_status()
    payload = bytearray()
    for chunk in response.iter_content(_DOWNLOAD_CHUNK_SIZE):
        if not chunk:
            continue
        payload.extend(chunk)
        if len(payload) > MAX_AMENDMENT_SOURCE_BYTES:
            raise AmendmentSourceExtractionError(
                "The source is too large to download for analysis."
            )
    if not payload:
        raise AmendmentSourceExtractionError("The source response was empty.")
    return bytes(payload)


def fetch_and_extract_amendment_url(url: str) -> AmendmentSourceExtraction:
    """Fetch one public URL through the shared SSRF-safe HTTP helper."""
    request_kwargs: dict[str, object] = {}
    if _is_resmi_gazete_url(url):
        request_kwargs["verify"] = _resmi_gazete_ca_bundle_path()

    try:
        response = ssrf_safe_get(
            url,
            headers=_AMENDMENT_SOURCE_HEADERS,
            timeout=_URL_TIMEOUT_SECONDS,
            stream=True,
            **request_kwargs,
        )
        content = _read_response_content(response)
    except AmendmentSourceExtractionError:
        raise
    except requests.exceptions.SSLError as exc:
        raise AmendmentSourceExtractionError(
            "The URL's TLS certificate chain could not be verified."
        ) from exc
    except Exception as exc:
        raise AmendmentSourceExtractionError(
            "The URL could not be downloaded. Check that it is publicly accessible."
        ) from exc

    content_type = response.headers.get("content-type")
    final_url = str(response.url)
    display_name = title_from_url(final_url) or title_from_url(url) or "source"
    if is_pdf_resource(final_url, content_type, content[:16]):
        return AmendmentSourceExtraction(
            text=extract_amendment_pdf(content, display_name or "source.pdf"),
            source_type="pdf",
            display_name=display_name,
        )

    if content_type and "html" not in content_type.lower():
        raise AmendmentSourceExtractionError(
            "The URL must point to an HTML page or PDF document."
        )
    return AmendmentSourceExtraction(
        text=extract_amendment_html(content, content_type),
        source_type="html",
        display_name=display_name,
    )
