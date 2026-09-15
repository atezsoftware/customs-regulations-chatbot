"""Normalize one amendment HTML, PDF, or DOCX source into reviewable text."""

import atexit
import base64
import binascii
import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag
from docx import Document
from docx.opc.exceptions import OpcError
from docx.table import Table
from docx.text.paragraph import Paragraph
from requests.utils import DEFAULT_CA_BUNDLE_PATH

if TYPE_CHECKING:
    from onyx.llm.interfaces import LLM

from onyx.configs.app_configs import (
    MAX_AMENDMENT_SOURCE_BYTES,
    MAX_AMENDMENT_SOURCE_TEXT_CHARS,
    MAX_ARCHIVE_COMPRESSION_RATIO,
    MAX_ARCHIVE_ENTRIES,
    MIN_AMENDMENT_PDF_TEXT_CHARS,
)
from onyx.file_processing.archive_expansion import COMPRESSION_RATIO_CHECK_MIN_BYTES
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
_MAX_AMENDMENT_DOCX_EXPANDED_BYTES = 50 * 1024 * 1024
_MAX_AMENDMENT_DOCX_XML_BYTES = 10 * 1024 * 1024
_DOCX_HEADING_STYLE_PATTERN = re.compile(r"Heading ([1-9])")
_NON_CONTENT_TAGS = ("script", "style", "template", "noscript", "nav", "footer")
_DECORATIVE_IMAGE_PATTERN = re.compile(
    r"(logo|icon|favicon|sprite|pixel|spacer|badge|avatar)", re.IGNORECASE
)
_MIN_DESCRIBED_IMAGE_DIMENSION_PX = 40
_MAX_DESCRIBED_IMAGES = 6
_MAX_IMAGE_DOWNLOAD_BYTES = 15 * 1024 * 1024
_MAX_DESCRIBED_PDF_LINKS = 3
_MAX_DESCRIBED_PDF_PAGES = 5
_PDF_LINK_PATTERN = re.compile(r"\.pdf(?:[?#]|$)", re.IGNORECASE)
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
    source_type: Literal["html", "pdf", "docx"]
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
    default_bundle = Path(DEFAULT_CA_BUNDLE_PATH).read_bytes()
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


def _is_decorative_image(img: Tag) -> bool:
    """Skip logos/icons/tracking pixels — not the scanned/charted content
    amendments actually embed."""
    src = str(img.get("src") or "")
    alt = str(img.get("alt") or "")
    if _DECORATIVE_IMAGE_PATTERN.search(src) or _DECORATIVE_IMAGE_PATTERN.search(alt):
        return True
    for attr in ("width", "height"):
        raw_value = img.get(attr)
        if raw_value is None:
            continue
        digits = re.sub(r"[^0-9]", "", str(raw_value))
        if digits and int(digits) < _MIN_DESCRIBED_IMAGE_DIMENSION_PX:
            return True
    return False


def _download_embedded_asset_bytes(url: str, *, max_bytes: int) -> bytes | None:
    """Best-effort bounded download for an <img>/<a> target discovered inline
    in an amendment source page. Never raises — callers treat None as
    "skip this one", not as a reason to fail the surrounding extraction."""
    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        if ";base64" not in url or not encoded:
            return None
        try:
            return base64.b64decode(encoded)
        except (binascii.Error, ValueError):
            return None
    try:
        options = amendment_source_http_options(url)
        response = ssrf_safe_get(
            url,
            headers=options.headers,
            timeout=_URL_TIMEOUT_SECONDS,
            stream=True,
            verify=options.verify,
        )
        response.raise_for_status()
        payload = bytearray()
        for chunk in response.iter_content(_DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > max_bytes:
                return None
        return bytes(payload) if payload else None
    except Exception:
        return None


def _describe_pdf_bytes(content: bytes, *, llm: "LLM", label: str) -> str | None:
    """Text layer when there is one (verbatim — the most faithful reading of
    "add it in exactly, fully"); otherwise render pages and describe each via
    vision. Returns None on any failure — best-effort, same as images."""
    try:
        extracted_text = extract_file_text(io.BytesIO(content), label, extension=".pdf")
    except Exception:
        extracted_text = ""
    if (
        extracted_text.strip()
        and len(extracted_text.strip()) >= MIN_AMENDMENT_PDF_TEXT_CHARS
    ):
        return extracted_text.strip()

    from onyx.file_processing.image_summarization import (
        summarize_image_with_error_handling,
    )
    from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
    from onyx.utils.process_isolation import run_in_isolated_process

    try:
        rendered_pages = run_in_isolated_process(
            render_annex_pages, content, "application/pdf", timeout=30
        )
    except Exception:
        return None

    page_descriptions: list[str] = []
    for page in rendered_pages[:_MAX_DESCRIBED_PDF_PAGES]:
        description = summarize_image_with_error_handling(
            llm,
            page.png,
            f"{label} (sayfa {page.page})",
        )
        if description:
            page_descriptions.append(f"Sayfa {page.page}: {description}")
    if not page_descriptions:
        return None
    return "\n\n".join(page_descriptions)


def _describe_linked_pdfs(root: Tag, base_url: str, *, llm: "LLM") -> str:
    """Best-effort: fetch and describe PDF attachments linked from the page
    (not just images) so a scanned/annex PDF's content reaches analysis
    alongside the page's own prose instead of silently vanishing."""
    seen_urls: set[str] = set()
    candidates: list[str] = []
    for link in root.find_all("a"):
        href = link.get("href")
        if not href or not _PDF_LINK_PATTERN.search(str(href)):
            continue
        resolved = urljoin(base_url, str(href))
        if resolved in seen_urls:
            continue
        seen_urls.add(resolved)
        candidates.append(resolved)
        if len(candidates) >= _MAX_DESCRIBED_PDF_LINKS:
            break
    if not candidates:
        return ""

    descriptions: list[str] = []
    for url in candidates:
        content = _download_embedded_asset_bytes(
            url, max_bytes=MAX_AMENDMENT_SOURCE_BYTES
        )
        if not content or not has_pdf_signature(content):
            continue
        label = title_from_url(url) or url
        description = _describe_pdf_bytes(content, llm=llm, label=label)
        if description:
            descriptions.append(f"[Ekli PDF: {label}]\n{description}")

    if not descriptions:
        return ""
    return "--- Ekli PDF içerikleri ---\n\n" + "\n\n".join(descriptions)


def _describe_embedded_images(root: Tag, base_url: str, *, llm: "LLM") -> str:
    """Best-effort: describe qualifying <img> elements via a vision LLM so
    charts/scanned content embedded in the source page reach the same
    analysis text the segmenter reads, not just the surrounding prose."""
    images = [
        img
        for img in root.find_all("img")
        if img.get("src") and not _is_decorative_image(img)
    ][:_MAX_DESCRIBED_IMAGES]
    if not images:
        return ""

    from onyx.file_processing.image_summarization import (
        summarize_image_with_error_handling,
    )

    descriptions: list[str] = []
    for img in images:
        src = urljoin(base_url, str(img.get("src")))
        image_bytes = _download_embedded_asset_bytes(
            src, max_bytes=_MAX_IMAGE_DOWNLOAD_BYTES
        )
        if not image_bytes:
            continue
        label = str(img.get("alt") or "").strip() or src
        description = summarize_image_with_error_handling(llm, image_bytes, label)
        if description:
            descriptions.append(f"[Gömülü görsel: {label}]\n{description}")

    if not descriptions:
        return ""
    return "--- Gömülü görsel açıklamaları ---\n\n" + "\n\n".join(descriptions)


def _describe_embedded_media(root: Tag, base_url: str) -> str:
    """Describe both inline <img> elements and linked PDF attachments so
    everything an amendment page embeds — not just its surrounding prose —
    reaches the same text the segmenter analyzes.

    Best-effort throughout: any failure (no vision provider configured,
    download error, render error, unsupported format) silently contributes
    nothing rather than failing the underlying text extraction.
    """
    has_images = any(
        img.get("src") and not _is_decorative_image(img) for img in root.find_all("img")
    )
    has_pdf_links = any(
        link.get("href") and _PDF_LINK_PATTERN.search(str(link.get("href")))
        for link in root.find_all("a")
    )
    if not has_images and not has_pdf_links:
        return ""

    from onyx.llm.factory import get_default_llm_with_vision

    llm = get_default_llm_with_vision()
    if llm is None:
        return ""

    blocks = [
        block
        for block in (
            _describe_embedded_images(root, base_url, llm=llm),
            _describe_linked_pdfs(root, base_url, llm=llm),
        )
        if block
    ]
    return "\n\n".join(blocks)


def extract_amendment_html(
    content: bytes, content_type: str | None, *, base_url: str | None = None
) -> str:
    """Extract readable main/body text while discarding common page chrome.

    When ``base_url`` is supplied, non-decorative <img> elements and linked
    PDF attachments are also described (best-effort, via a vision LLM) and
    appended, so an amendment page's embedded charts/scans/attachments reach
    analysis alongside its prose instead of silently vanishing.
    """
    soup = BeautifulSoup(decode_html_bytes(content, content_type), "html.parser")
    for tag in soup.find_all(_NON_CONTENT_TAGS):
        tag.decompose()

    root = cast(Tag, soup.find("main") or soup.find("article") or soup.body or soup)
    text = root.get_text("\n\n", strip=True)
    if base_url is not None:
        media_descriptions = _describe_embedded_media(root, base_url)
        if media_descriptions:
            text = f"{text}\n\n{media_descriptions}"
    return _normalize_text(text)


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


def _docx_paragraph_text(paragraph: Paragraph) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = paragraph.style.name if paragraph.style is not None else ""
    heading_match = _DOCX_HEADING_STYLE_PATTERN.fullmatch(style_name)
    if heading_match is None:
        return text

    heading_level = int(heading_match.group(1))
    return f"{'#' * heading_level} {text}"


def _docx_table_rows(table: Table) -> list[str]:
    rows: list[str] = []
    for row in table.rows:
        cells = [re.sub(r"\s*\n\s*", " ", cell.text).strip() for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _extract_docx_text(content: bytes) -> str:
    try:
        document = Document(io.BytesIO(content))
    except (KeyError, OpcError, SyntaxError, ValueError, zipfile.BadZipFile) as exc:
        raise AmendmentSourceExtractionError(
            "The uploaded Word .docx document could not be read."
        ) from exc

    sections: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            if paragraph_text := _docx_paragraph_text(block):
                sections.append(paragraph_text)
        elif isinstance(block, Table):
            sections.extend(_docx_table_rows(block))
    return "\n\n".join(sections)


def extract_amendment_docx(content: bytes, file_name: str) -> str:
    """Extract normalized text from a Word Open XML document."""
    if len(content) > MAX_AMENDMENT_SOURCE_BYTES:
        raise AmendmentSourceExtractionError(
            "The Word document is too large to upload for analysis."
        )
    if Path(file_name).suffix.lower() != ".docx":
        raise AmendmentSourceExtractionError(
            "The uploaded Word file must use the .docx format; only .docx files are supported."
        )
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            archive_entries = archive.infolist()
            archive_names = {entry.filename for entry in archive_entries}
    except (OSError, zipfile.BadZipFile) as exc:
        raise AmendmentSourceExtractionError(
            "The uploaded file is not a valid Word .docx document."
        ) from exc
    if len(archive_entries) > MAX_ARCHIVE_ENTRIES:
        raise AmendmentSourceExtractionError(
            "The Word document contains too many entries to extract safely."
        )
    for entry in archive_entries:
        if (
            Path(entry.filename).suffix.lower() == ".xml"
            and entry.file_size > _MAX_AMENDMENT_DOCX_XML_BYTES
        ):
            raise AmendmentSourceExtractionError(
                "The Word document contains an XML part that is too large to extract safely."
            )
        if entry.file_size < COMPRESSION_RATIO_CHECK_MIN_BYTES:
            continue
        if (
            entry.compress_size == 0
            or entry.file_size / entry.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO
        ):
            raise AmendmentSourceExtractionError(
                "The Word document contains an unsafe compression ratio."
            )
    expanded_size = sum(entry.file_size for entry in archive_entries)
    if expanded_size > _MAX_AMENDMENT_DOCX_EXPANDED_BYTES:
        raise AmendmentSourceExtractionError(
            "The Word document expands beyond the safe extraction limit."
        )
    if (
        "[Content_Types].xml" not in archive_names
        or "word/document.xml" not in archive_names
    ):
        raise AmendmentSourceExtractionError(
            "The uploaded file is not a valid Word .docx document."
        )

    return _normalize_text(_extract_docx_text(content))


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
    verify: bool | str = (
        _resmi_gazete_ca_bundle_path() if _is_resmi_gazete_url(url) else True
    )

    try:
        response = ssrf_safe_get(
            url,
            headers=_AMENDMENT_SOURCE_HEADERS,
            timeout=_URL_TIMEOUT_SECONDS,
            stream=True,
            verify=verify,
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
        text=extract_amendment_html(content, content_type, base_url=final_url),
        source_type="html",
        display_name=display_name,
    )


@dataclass(frozen=True)
class AmendmentSourceHttpOptions:
    headers: dict[str, str]
    verify: bool | str


def amendment_source_http_options(url: str) -> AmendmentSourceHttpOptions:
    """Shared public transport settings for original and annex source downloads."""
    return AmendmentSourceHttpOptions(
        headers=dict(_AMENDMENT_SOURCE_HEADERS),
        verify=_resmi_gazete_ca_bundle_path() if _is_resmi_gazete_url(url) else True,
    )
