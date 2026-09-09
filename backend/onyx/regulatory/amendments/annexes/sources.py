"""Fetch only source-derived annex relationships; never follow model-produced URLs."""

import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag
from defusedxml import ElementTree
from PIL import Image

from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AcquisitionResult,
    DiscoveredLink,
    SourceInspection,
    SourceIssue,
    SourceLink,
)
from onyx.regulatory.amendments.source_extraction import (
    amendment_source_http_options,
)
from onyx.utils.url import ssrf_safe_get
from onyx.utils.web_content import decode_html_bytes

MAX_ASSET_BYTES = 25 * 1024 * 1024
MAX_PACKAGE_BYTES = 100 * 1024 * 1024
MAX_ANNEX_ASSETS = 20
MAX_LINK_DEPTH = 2
MAX_PACKAGE_SECONDS = 180
MAX_TEXT_CHARS = 2_000_000
MAX_RELATIONSHIPS = 500
_ANNEX = re.compile(
    r"(?:\bek(?:ler|leri)?(?:\b|[-–:]?\d)|annex|appendix|attachment|ekli|ilişik|değişiklik|amendment)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s<>\"']+")
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class SourceAcquisitionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DownloadedSource:
    content: bytes
    mime_type: str | None
    final_url: str


def download_source(url: str, *, deadline: float | None = None) -> DownloadedSource:
    deadline = deadline or time.monotonic() + MAX_PACKAGE_SECONDS
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SourceAcquisitionError("time_limit")
    response: requests.Response | None = None
    try:
        current_url = url
        visited: set[str] = set()
        for redirect_count in range(11):
            if current_url in visited:
                raise SourceAcquisitionError("redirect_loop")
            visited.add(current_url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SourceAcquisitionError("time_limit")
            options = amendment_source_http_options(current_url)
            response = ssrf_safe_get(
                current_url,
                headers={**options.headers, "Accept-Encoding": "identity"},
                stream=True,
                follow_redirects=False,
                timeout=(min(5, remaining), min(20, remaining)),
                verify=options.verify,
            )
            if not response.is_redirect:
                break
            if redirect_count == 10:
                raise SourceAcquisitionError("redirect_loop")
            current_url = urljoin(current_url, response.headers["location"])
            response.close()
        if response is None:
            raise SourceAcquisitionError("download_failed")
        if response.status_code == 404:
            raise SourceAcquisitionError("404")
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared and int(declared) > MAX_ASSET_BYTES:
            raise SourceAcquisitionError("asset_byte_limit")
        payload = bytearray()
        for chunk in response.iter_content(64 * 1024):
            if time.monotonic() >= deadline:
                raise SourceAcquisitionError("time_limit")
            payload.extend(chunk)
            if len(payload) > MAX_ASSET_BYTES:
                raise SourceAcquisitionError("asset_byte_limit")
        if (
            declared
            and not response.headers.get("content-encoding")
            and len(payload) != int(declared)
        ):
            raise SourceAcquisitionError("truncated")
        if not payload:
            raise SourceAcquisitionError("empty_source")
        return DownloadedSource(
            bytes(payload), response.headers.get("content-type"), current_url
        )
    except SourceAcquisitionError:
        raise
    except requests.exceptions.TooManyRedirects as exc:
        raise SourceAcquisitionError("redirect_loop") from exc
    except requests.exceptions.ChunkedEncodingError as exc:
        raise SourceAcquisitionError("truncated") from exc
    except Exception as exc:
        raise SourceAcquisitionError("download_failed") from exc
    finally:
        if response is not None:
            response.close()


def _archive(content: bytes) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(io.BytesIO(content))
    entries = archive.infolist()
    if (
        len(entries) > 1000
        or sum(entry.file_size for entry in entries) > MAX_PACKAGE_BYTES
    ):
        archive.close()
        raise SourceAcquisitionError("archive_limit")
    for entry in entries:
        if entry.file_size > MAX_ASSET_BYTES or (
            entry.file_size > 1024 * 1024
            and entry.file_size / max(1, entry.compress_size) > 100
        ):
            archive.close()
            raise SourceAcquisitionError("archive_limit")
    return archive


def _detect_mime(content: bytes, declared: str | None) -> str:
    mime = (declared or "").split(";", 1)[0].strip().lower()
    detected: str | None = None
    if content.startswith(b"%PDF-"):
        detected = "application/pdf"
    elif content.startswith(b"PK\x03\x04"):
        with _archive(content) as archive:
            names = archive.namelist()
            if "[Content_Types].xml" in names:
                detected = (
                    _DOCX_MIME
                    if "word/document.xml" in names
                    else _XLSX_MIME
                    if "xl/workbook.xml" in names
                    else None
                )
    elif content.startswith(
        (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*")
    ) or (content[:4] == b"RIFF" and content[8:12] == b"WEBP"):
        with Image.open(io.BytesIO(content)) as image:
            detected = Image.MIME.get(image.format or "")
            image.verify()
    elif re.search(
        rb"<(?:!doctype\s+html|html|head|body|main|article|a|p|div|table)\b",
        content[:4096],
        re.I,
    ):
        detected = "text/html"
    elif mime == "text/plain":
        content.decode("utf-8")
        detected = mime
    if detected is None:
        raise SourceAcquisitionError("unsupported_format")
    if mime not in ("", "application/octet-stream", detected) and not (
        detected == "text/html" and mime == "application/xhtml+xml"
    ):
        raise SourceAcquisitionError("mime_mismatch")
    return detected


def _inspect_html(content: bytes, mime: str) -> SourceInspection:
    soup = BeautifulSoup(decode_html_bytes(content, mime), "html.parser")
    for tag in soup.find_all(
        ("nav", "footer", "script", "style", "template", "noscript")
    ):
        tag.decompose()
    root = cast(Tag, soup.find("main") or soup.find("article") or soup.body or soup)
    links: list[DiscoveredLink] = []
    for index, anchor in enumerate(root.find_all(("a", "img"))):
        target = str(anchor.get("href") or anchor.get("src") or "")
        label = anchor.get_text(" ", strip=True) or str(anchor.get("alt") or "")
        context = anchor.find_parent(("p", "li", "td", "figure"))
        heading = anchor.find_previous(("h1", "h2", "h3", "h4", "h5", "h6"))
        heading_text = (
            heading.get_text(" ", strip=True)
            if isinstance(heading, Tag) and heading in root.descendants
            else ""
        )
        if not target or not _ANNEX.search(
            label
            + " "
            + heading_text
            + " "
            + (context.get_text(" ", strip=True) if context else "")
            + " "
            + target
        ):
            continue
        links.append(
            DiscoveredLink(
                kind="internal" if target.startswith("#") else "url",
                target=target,
                label=label,
                source_field=f"html:{index}",
            )
        )
        if (
            target.startswith("#")
            and root.find(id=target[1:]) is None
            and root.find(attrs={"name": target[1:]}) is None
        ):
            raise SourceAcquisitionError("missing_internal_target")
    return SourceInspection(
        mime_type=mime, text=root.get_text("\n", strip=True), links=links
    )


def _inspect_pdf(content: bytes) -> SourceInspection:
    from pypdf import PdfReader
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        IndirectObject,
        NumberObject,
    )

    reader = PdfReader(io.BytesIO(content), strict=True)
    if reader.is_encrypted:
        raise SourceAcquisitionError("encrypted_pdf")
    if len(reader.pages) > 500:
        raise SourceAcquisitionError("page_limit")
    links: list[DiscoveredLink] = []
    texts: list[str] = []
    for number, page in enumerate(reader.pages, 1):
        fragments: list[tuple[str, float, float]] = []

        def visit_text(
            fragment: str,
            cm: list[float],
            tm: list[float],
            _font: DictionaryObject | None,
            _size: float,
        ) -> None:
            if fragment.strip():
                fragments.append((fragment, cm[4] + tm[4], cm[5] + tm[5]))

        text = page.extract_text(visitor_text=visit_text) or ""
        texts.append(text)
        for index, reference in enumerate(page.get("/Annots", [])):
            annotation = reference.get_object()
            action = annotation.get("/A", {})
            if hasattr(action, "get_object"):
                action = action.get_object()
            target = str(action.get("/URI", ""))
            label = str(annotation.get("/Contents", ""))
            rect = annotation.get("/Rect", [])
            if len(rect) == 4:
                label += " ".join(
                    fragment
                    for fragment, x, y in fragments
                    if float(rect[0]) - 10 <= x <= float(rect[2]) + 10
                    and float(rect[1]) - 10 <= y <= float(rect[3]) + 10
                )
            if annotation.get("/Subtype") == "/FileAttachment":
                file_spec = annotation.get("/FS")
                if file_spec is None:
                    raise SourceAcquisitionError("missing_embedded_source")
                file_spec = file_spec.get_object()
                embedded = file_spec.get("/EF", {}).get("/F")
                if embedded is None:
                    raise SourceAcquisitionError("missing_embedded_source")
                payload = embedded.get_object().get_data()
                if len(payload) > MAX_ASSET_BYTES:
                    raise SourceAcquisitionError("asset_byte_limit")
                name = str(file_spec.get("/UF") or file_spec.get("/F") or "attachment")
                links.append(
                    DiscoveredLink(
                        kind="embedded",
                        target=name,
                        label=label or name,
                        source_page=number,
                        source_field=f"pdf:annotation:{index}",
                        embedded_base64=base64.b64encode(payload).decode(),
                    )
                )
            if target and _ANNEX.search(target + " " + label):
                links.append(
                    DiscoveredLink(
                        kind="url",
                        target=target,
                        label=label,
                        source_page=number,
                        source_field=f"pdf:annotation:{index}",
                    )
                )
            destination = annotation.get("/Dest") or action.get("/D")
            if destination is not None:
                resolved = False
                if isinstance(destination, ArrayObject) and destination:
                    target_page = destination[0]
                    if isinstance(target_page, NumberObject):
                        resolved = 0 <= int(target_page) < len(reader.pages)
                    elif isinstance(target_page, IndirectObject):
                        resolved = any(
                            candidate.indirect_reference == target_page
                            for candidate in reader.pages
                        )
                elif isinstance(destination, str):
                    named = reader.named_destinations.get(
                        destination
                    ) or reader.named_destinations.get(destination.lstrip("/"))
                    resolved = (
                        named is not None
                        and reader.get_destination_page_number(named) is not None
                    )
                if not resolved:
                    raise SourceAcquisitionError("missing_internal_target")
                links.append(
                    DiscoveredLink(
                        kind="internal",
                        target=str(destination),
                        label=label,
                        source_page=number,
                        source_field=f"pdf:annotation:{index}",
                    )
                )
        for index, match in enumerate(_URL.finditer(text)):
            if _ANNEX.search(text[max(0, match.start() - 100) : match.end() + 100]):
                links.append(
                    DiscoveredLink(
                        kind="url",
                        target=match.group().rstrip(".,);"),
                        label="",
                        source_page=number,
                        source_field=f"pdf:text:{index}",
                    )
                )
        if sum(len(part) for part in texts) > MAX_TEXT_CHARS:
            raise SourceAcquisitionError("text_limit")
    for name, payloads in reader.attachments.items():
        for index, payload in enumerate(payloads):
            if len(payload) > MAX_ASSET_BYTES:
                raise SourceAcquisitionError("asset_byte_limit")
            links.append(
                DiscoveredLink(
                    kind="embedded",
                    target=name,
                    label=name,
                    source_field=f"pdf:attachment:{name}:{index}",
                    embedded_base64=base64.b64encode(payload).decode(),
                )
            )
    return SourceInspection(
        mime_type="application/pdf", text="\n\n".join(texts), links=links
    )


def _inspect_office(content: bytes, mime: str) -> SourceInspection:
    texts: list[str] = []
    links: list[DiscoveredLink] = []
    with _archive(content) as archive:
        names = set(archive.namelist())
        for part in sorted(names):
            if not part.endswith(".xml") or not part.startswith(("word/", "xl/")):
                continue
            root = ElementTree.fromstring(archive.read(part))
            texts.extend(
                element.text
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] in ("t", "v") and element.text
            )
            rel_path = str(Path(part).parent / "_rels" / (Path(part).name + ".rels"))
            if rel_path not in names:
                continue
            relationships = ElementTree.fromstring(archive.read(rel_path))
            for relation in relationships:
                rel_id, target = relation.get("Id", ""), relation.get("Target", "")
                contexts = []
                for element in root.iter():
                    if rel_id in element.attrib.values():
                        contexts.append(" ".join(element.itertext()))
                label = " ".join(contexts)
                if not _ANNEX.search(label + " " + target):
                    continue
                if relation.get("TargetMode") == "External":
                    links.append(
                        DiscoveredLink(
                            kind="url",
                            target=target,
                            label=label,
                            source_field=f"{part}:{rel_id}",
                        )
                    )
                else:
                    from posixpath import normpath

                    embedded_path = normpath(str(Path(part).parent / target))
                    if embedded_path not in names:
                        raise SourceAcquisitionError("missing_embedded_source")
                    links.append(
                        DiscoveredLink(
                            kind="embedded",
                            target=embedded_path,
                            label=label,
                            source_field=f"{part}:{rel_id}",
                            embedded_base64=base64.b64encode(
                                archive.read(embedded_path)
                            ).decode(),
                        )
                    )
    return SourceInspection(mime_type=mime, text="\n".join(texts), links=links)


def inspect_source(content: bytes, declared: str | None) -> SourceInspection:
    """Parser entry point executed only inside the bounded child process."""
    mime = _detect_mime(content, declared)
    if mime == "application/pdf":
        result = _inspect_pdf(content)
    elif mime == "text/html":
        result = _inspect_html(content, mime)
    elif mime in (_DOCX_MIME, _XLSX_MIME):
        result = _inspect_office(content, mime)
    else:
        result = SourceInspection(
            mime_type=mime, text=content.decode("utf-8") if mime == "text/plain" else ""
        )
    if len(result.links) > MAX_RELATIONSHIPS:
        raise SourceAcquisitionError("relationship_limit")
    if len(result.text) > MAX_TEXT_CHARS:
        raise SourceAcquisitionError("text_limit")
    return result


def _bounded_inspect(
    content: bytes, mime: str | None, deadline: float
) -> SourceInspection:
    with tempfile.TemporaryDirectory(prefix="onyx-annex-") as directory:
        source = Path(directory) / "source"
        output = Path(directory) / "inspection.json"
        source.write_bytes(content)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onyx.regulatory.amendments.annexes.source_parser",
                    str(source),
                    str(output),
                    mime or "",
                ],
                cwd=Path(__file__).resolve().parents[4],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, min(30, deadline - time.monotonic())),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SourceAcquisitionError("parse_time_limit") from exc
        if completed.returncode != 0 or not output.exists():
            raise SourceAcquisitionError("parse_failed")
        result = json.loads(output.read_text())
        if "error" in result:
            raise SourceAcquisitionError(result["error"])
        return SourceInspection.model_validate(result)


def acquire_source_package(
    *,
    url: str | None = None,
    content: bytes | None = None,
    mime_type: str | None = None,
    display_name: str = "source",
    fetch: Callable[[str], DownloadedSource] | None = None,
) -> AcquisitionResult:
    deadline = time.monotonic() + MAX_PACKAGE_SECONDS
    fetch_source = fetch or (
        lambda address: download_source_bounded(address, deadline=deadline)
    )
    result = AcquisitionResult(status="processing")
    by_hash: dict[str, AcquiredAsset] = {}
    by_url: dict[str, str] = {}
    inspections: dict[str, SourceInspection] = {}
    expanded: set[tuple[str, str | None]] = set()
    pending: deque[
        tuple[str | None, bytes | None, str | None, str, int, SourceLink | None]
    ] = deque([(url, content, mime_type, display_name, 0, None)])
    total = 0
    while pending:
        address, payload, declared, name, depth, relation = pending.popleft()
        try:
            if time.monotonic() >= deadline:
                raise SourceAcquisitionError("time_limit")
            if address in by_url:
                if relation is not None:
                    relation.target_asset_hash = by_url[address]
                    relation.final_url = by_hash[by_url[address]].final_url
                continue
            if depth > MAX_LINK_DEPTH:
                raise SourceAcquisitionError("depth_limit")
            final_url = address
            if payload is None:
                if not address:
                    raise SourceAcquisitionError("missing_source")
                downloaded = fetch_source(address)
                payload, declared, final_url = (
                    downloaded.content,
                    downloaded.mime_type,
                    downloaded.final_url,
                )
            if len(payload) > MAX_ASSET_BYTES:
                raise SourceAcquisitionError("asset_byte_limit")
            digest = hashlib.sha256(payload).hexdigest()
            total += len(payload)
            if total > MAX_PACKAGE_BYTES:
                raise SourceAcquisitionError("package_byte_limit")
            if digest not in by_hash:
                if len(by_hash) >= MAX_ANNEX_ASSETS + 1:
                    raise SourceAcquisitionError("asset_count_limit")
                inspection = _bounded_inspect(payload, declared, deadline)
                inspections[digest] = inspection
                expected_mime = {
                    ".pdf": "application/pdf",
                    ".docx": _DOCX_MIME,
                    ".xlsx": _XLSX_MIME,
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".tif": "image/tiff",
                    ".tiff": "image/tiff",
                }.get(Path(name).suffix.lower())
                if expected_mime and expected_mime != inspection.mime_type:
                    raise SourceAcquisitionError("mime_mismatch")
                asset = AcquiredAsset(
                    sha256=digest,
                    content=payload,
                    mime_type=inspection.mime_type,
                    display_name=name,
                    original_url=address,
                    final_url=final_url,
                    text=inspection.text,
                )
                by_hash[digest] = asset
                result.assets.append(asset)
            if (digest, final_url) not in expanded:
                expanded.add((digest, final_url))
                for discovered in inspections[digest].links:
                    if len(result.links) >= MAX_RELATIONSHIPS:
                        raise SourceAcquisitionError("relationship_limit")
                    link = SourceLink(
                        parent_asset_hash=digest,
                        source_page=discovered.source_page,
                        source_field=discovered.source_field,
                        label=discovered.label,
                        kind=discovered.kind,
                        original_url=discovered.target,
                    )
                    result.links.append(link)
                    if discovered.kind == "internal":
                        link.target_asset_hash = digest
                        link.final_url = (
                            (final_url or "") + "#" + discovered.target.lstrip("#")
                        )
                        continue
                    if discovered.kind == "embedded":
                        pending.append(
                            (
                                None,
                                base64.b64decode(
                                    discovered.embedded_base64 or "", validate=True
                                ),
                                None,
                                discovered.target,
                                depth + 1,
                                link,
                            )
                        )
                        continue
                    target = urljoin(final_url or "", discovered.target)
                    if not urlsplit(target).scheme:
                        result.issues.append(
                            SourceIssue(
                                code="missing_base_url", locator=discovered.source_field
                            )
                        )
                        continue
                    if urlsplit(target).scheme not in ("http", "https"):
                        result.issues.append(
                            SourceIssue(
                                code="unsafe_url",
                                locator=discovered.source_field,
                                retryable=False,
                            )
                        )
                        continue
                    target = urldefrag(target)[0]
                    pending.append(
                        (
                            target,
                            None,
                            None,
                            Path(urlsplit(target).path).name or "source",
                            depth + 1,
                            link,
                        )
                    )
            if address:
                by_url[address] = digest
            if final_url:
                by_url[final_url] = digest
            if relation:
                relation.target_asset_hash = digest
                relation.final_url = final_url
        except SourceAcquisitionError as exc:
            result.issues.append(
                SourceIssue(
                    code=exc.code, locator=relation.source_field if relation else None
                )
            )
    blocked = any(issue.code.endswith("limit") for issue in result.issues)
    result.status = (
        "blocked"
        if blocked
        else "partial"
        if result.issues and result.assets
        else "failed"
        if result.issues
        else "ready"
    )
    return result


def download_source_bounded(url: str, *, deadline: float) -> DownloadedSource:
    """Enforce total wall time even when a peer continually trickles socket bytes."""
    with tempfile.TemporaryDirectory(prefix="onyx-annex-download-") as directory:
        source = Path(directory) / "source"
        metadata = Path(directory) / "metadata.json"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onyx.regulatory.amendments.annexes.source_downloader",
                    url,
                    str(source),
                    str(metadata),
                ],
                cwd=Path(__file__).resolve().parents[4],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, min(30, deadline - time.monotonic())),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SourceAcquisitionError("download_time_limit") from exc
        if completed.returncode != 0 or not metadata.exists():
            raise SourceAcquisitionError("download_failed")
        result = json.loads(metadata.read_text())
        if "error" in result:
            raise SourceAcquisitionError(result["error"])
        return DownloadedSource(
            source.read_bytes(), result["mime_type"], result["final_url"]
        )
