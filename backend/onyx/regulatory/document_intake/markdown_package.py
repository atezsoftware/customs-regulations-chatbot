"""Validate supplied Markdown without fetching, converting, or rewriting its text."""

import io
import json
import posixpath
import re
import stat
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Literal, Match, TypedDict, cast
from urllib.parse import unquote, urlsplit

from mistune import BlockParser, BlockState, InlineParser, InlineState, Markdown
from mistune.helpers import parse_link_href, parse_link_label, unescape_char
from mistune.inline_parser import _parse_link_text
from mistune.util import unikey
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AcquisitionResult,
    SourceLink,
)
from onyx.regulatory.amendments.annexes.sources import MAX_ANNEX_ASSETS

MAX_MARKDOWN_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_MARKDOWN_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_MARKDOWN_TEXT_CHARS = 2_000_000
MAX_MARKDOWN_PACKAGE_BYTES = 32 * 1024 * 1024
MAX_MARKDOWN_DOCUMENTS = MAX_ANNEX_ASSETS + 1
MAX_MARKDOWN_ZIP_ENTRIES = 128
MAX_MARKDOWN_COMPRESSION_RATIO = 100
MAX_MARKDOWN_LINKS = 500
_MANIFEST_BYTES = 256 * 1024
_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".tif",
    ".tiff",
    ".bmp",
    ".heic",
    ".avif",
}


class MarkdownPackageError(ValueError):
    """A bounded diagnostic containing a code and package path, never source text."""

    def __init__(self, code: str, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = "" if path is None else f": {ascii(path[:512])}"
        super().__init__(f"{code}{detail}")

    @property
    def public_message(self) -> str:
        messages = {
            "markdown_document_count_limit": f"Bir Markdown paketinde bir ana belge ve en fazla {MAX_ANNEX_ASSETS} ek bulunabilir (toplam {MAX_MARKDOWN_DOCUMENTS} belge). Bağımsız ana belgeleri ayrı paketlerde yükleyin. Tek mevzuatın bağlı ekleri bu sınırı aşıyorsa yöneticinize başvurun; bağlı ekleri ayrı paketlere bölmeyin.",
            "declared_parent_must_link_to_annex": "Üst Markdown belgesi, manifest'te belirtilen ek dosyasına gerçek bir yerel bağlantı içermelidir.",
            "missing_local_markdown_target": "Markdown bağlantısının hedefi pakette bulunamadı. Dosya yolunu ve büyük/küçük harfleri kontrol edin.",
            "exactly_one_root_manifest_required": "ZIP kökünde veya tek bir dış klasörde bir manifest.json dosyası bulunmalıdır.",
            "invalid_markdown_manifest": "manifest.json biçimi geçersiz. Örnek paketteki alan adlarını ve belge listesini kullanın.",
            "manifest_must_exactly_list_all_markdown_files": "Manifest, paketteki bütün ve yalnızca Markdown dosyalarını birebir yollarıyla listelemelidir.",
            "markdown_requires_utf8": "Markdown dosyası UTF-8 metni olmalıdır.",
            "empty_binary_or_oversized_markdown": "Dosya boş, ikili veri içeriyor veya metin sınırını aşıyor. UTF-8 biçiminde eksiksiz Markdown metni gereklidir.",
            "binary_source_requires_markdown_transcription": "PDF veya görsel dosyası Markdown olarak yeniden adlandırılamaz. İçeriğini eksiksiz Markdown metnine veya tablosuna aktarın.",
            "images_require_transcription": "Görsel veya tarama bağlantısı kabul edilmez. İçeriğini eksiksiz Markdown metnine veya tablosuna aktarın.",
            "images_or_embedded_resources_require_transcription": "Görsel veya gömülü dosya kabul edilmez. İçeriğini eksiksiz Markdown metnine veya tablosuna aktarın.",
            "duplicate_markdown_content": "İki Markdown dosyasının içeriği aynı. Ayrı mevzuat kayıtları için bu tekrar düzeltilmelidir.",
            "duplicate_or_ambiguous_package_path": "ZIP'te aynı veya harf/Unicode yazımı nedeniyle çakışan dosya yolları var.",
            "duplicate_or_invalid_manifest_document": "Manifest'te yinelenen veya geçersiz bir Markdown dosya yolu var.",
            "duplicate_manifest_key": "manifest.json içinde aynı alan birden fazla kez tanımlanmış.",
            "duplicate_markdown_reference": "Markdown içinde aynı bağlantı referansı birden fazla kez tanımlanmış.",
            "annex_requires_one_valid_parent": "Her ekin manifest'te bulunan tek bir üst belgesi olmalıdır.",
            "manifest_parent_tree_required": "Üst belge ilişkileri ana belgede köklenen ve döngü içermeyen bir ağaç olmalıdır.",
        }
        message = messages.get(self.code)
        if message is None:
            message = (
                "Markdown paketi boyut, belge sayısı, bağlantı veya sıkıştırma sınırını aşıyor. İçerik kesilmeden reddedildi."
                if "limit" in self.code
                else f"Markdown paketi doğrulanamadı ({self.code}). Örnek paket biçimini kontrol edin."
            )
        if self.path is not None:
            message += " Dosya: " + json.dumps(self.path[:200], ensure_ascii=False)
        return message


class _ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    title: str = Field(min_length=1, max_length=500)
    parent: str | None = None
    source_url: str | None = None


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["legislation-markdown-v1"]
    main: str
    documents: list[_ManifestDocument] = Field(
        min_length=1, max_length=MAX_MARKDOWN_DOCUMENTS
    )


class _Token(TypedDict, total=False):
    type: str
    raw: str
    attrs: dict[str, object]
    children: list["_Token"]
    literal_url: str


@dataclass
class _Reference:
    literal_url: str
    url: str
    label_parts: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return unescape("".join(self.label_parts)).strip()[:500]


@dataclass(frozen=True)
class _ParsedDocument:
    specification: _ManifestDocument
    content: bytes
    text: str
    references: list[_Reference]
    title: str | None


class _LiteralBlockParser(BlockParser):
    def parse_ref_link(self, m: Match[str], state: BlockState) -> int | None:
        key = unikey(m.group("reflink_1"))
        existed = key in state.env.get("ref_links", {})
        result = super().parse_ref_link(m, state)
        if result is not None and key in state.env.get("ref_links", {}):
            if existed:
                raise MarkdownPackageError("duplicate_markdown_reference")
            literal, _position = parse_link_href(state.src, m.end(), block=True)
            state.env["ref_links"][key]["literal_url"] = literal
        return result


class _LiteralInlineParser(InlineParser):
    def parse_link(self, m: Match[str], state: InlineState) -> int | None:
        count = len(state.tokens)
        result = super().parse_link(m, state)
        if result is None or len(state.tokens) == count:
            return result
        token = state.tokens[-1]
        if token["type"] not in {"link", "image"}:
            return result
        if "ref" in token:
            token["literal_url"] = state.env["ref_links"][token["ref"]]["literal_url"]
        else:
            label, end = parse_link_label(state.src, m.end())
            if label is None:
                _text, end = _parse_link_text(state, m.end())
            if end is not None and state.src[end : end + 1] == "(":
                literal, _position = parse_link_href(state.src, end + 1)
                token["literal_url"] = literal
        return result


class _HtmlReferences(HTMLParser):
    def __init__(self, references: list[_Reference]) -> None:
        super().__init__(convert_charrefs=True)
        self.references = references
        self.anchor: _Reference | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {
            "img",
            "svg",
            "picture",
            "object",
            "embed",
            "iframe",
            "video",
            "audio",
            "canvas",
            "base",
        } or any(key in attributes for key in ("src", "srcset", "poster")):
            raise MarkdownPackageError(
                "images_or_embedded_resources_require_transcription"
            )
        if tag == "a" and "href" in attributes:
            if (
                self.anchor is not None
                or sum(key == "href" for key, _value in attrs) != 1
            ):
                raise MarkdownPackageError("ambiguous_html_link")
            # HTMLParser resolves character references; retain the source attribute too.
            raw = self.get_starttag_text() or ""
            attributes_in_source = re.finditer(
                r"\s+([^\s=/>]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+)))?",
                raw,
            )
            href = next(
                (
                    item
                    for item in attributes_in_source
                    if item.group(1).lower() == "href"
                ),
                None,
            )
            literal = (
                None
                if href is None
                else next(
                    (value for value in href.groups()[1:] if value is not None), None
                )
            )
            if literal is None or attributes["href"] is None:
                raise MarkdownPackageError("invalid_html_link")
            self.anchor = _Reference(literal, attributes["href"] or "")
            self.references.append(self.anchor)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.anchor = None

    def handle_data(self, data: str) -> None:
        if self.anchor is not None:
            self.anchor.label_parts.append(data)


def _path(value: str) -> str:
    if (
        not value
        or len(value) > 512
        or "\\" in value
        or ":" in value
        or value.startswith("/")
    ):
        raise MarkdownPackageError("unsafe_package_path", value)
    if any(
        part in {"", ".", ".."} or len(part) > 255 for part in value.split("/")
    ) or any(unicodedata.category(character).startswith("C") for character in value):
        raise MarkdownPackageError("unsafe_package_path", value)
    return value


def _path_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _decode(content: bytes, path: str) -> str:
    if len(content) > MAX_MARKDOWN_DOCUMENT_BYTES:
        raise MarkdownPackageError("markdown_document_byte_limit", path)
    if content.startswith(
        (b"%PDF-", b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")
    ):
        raise MarkdownPackageError(
            "binary_source_requires_markdown_transcription", path
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MarkdownPackageError("markdown_requires_utf8", path) from None
    if not text.strip() or "\x00" in text or len(text) > MAX_MARKDOWN_TEXT_CHARS:
        raise MarkdownPackageError("empty_binary_or_oversized_markdown", path)
    return text


def _token_text(token: _Token) -> str:
    return token.get("raw", "") + "".join(
        _token_text(child) for child in token.get("children", [])
    )


def _markdown_references(text: str, path: str) -> tuple[list[_Reference], str | None]:
    markdown = Markdown(block=_LiteralBlockParser(), inline=_LiteralInlineParser())
    try:
        tokens = cast(list[_Token], markdown(text))
        references: list[_Reference] = []
        html = _HtmlReferences(references)
        title: str | None = None

        def visit(token: _Token) -> None:
            nonlocal title
            kind = token.get("type")
            if kind == "image":
                raise MarkdownPackageError("images_require_transcription", path)
            if (
                kind == "heading"
                and token.get("attrs", {}).get("level") == 1
                and title is None
            ):
                title = unescape(_token_text(token)).strip()[:500]
            if kind == "link":
                url = token.get("attrs", {}).get("url")
                if not isinstance(url, str):
                    raise MarkdownPackageError("invalid_markdown_link", path)
                literal = token.get("literal_url", url)
                references.append(
                    _Reference(
                        literal, unescape(unescape_char(literal)), [_token_text(token)]
                    )
                )
            elif kind in {"inline_html", "block_html"}:
                html.feed(token.get("raw", ""))
                return
            elif html.anchor is not None and kind in {"text", "codespan"}:
                html.feed(escape(token.get("raw", "")))
            if kind not in {"block_code", "codespan"}:
                for child in token.get("children", []):
                    visit(child)

        for token in tokens:
            visit(token)
        html.close()
        if html.anchor is not None:
            raise MarkdownPackageError("unclosed_html_link", path)
        return references, title
    except MarkdownPackageError as error:
        if error.path is None:
            raise MarkdownPackageError(error.code, path) from None
        raise
    except (RecursionError, ValueError):
        raise MarkdownPackageError("invalid_markdown", path) from None


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MarkdownPackageError("duplicate_manifest_key")
        result[key] = value
    return result


def _is_ignored_macos_metadata(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return parts[-1] == ".DS_Store" or (
        parts[0] == "__MACOSX"
        and len(parts) > 1
        and parts[-1].startswith("._")
        and len(parts[-1]) > 2
    )


def _read_archive(content: bytes) -> tuple[_Manifest, dict[str, bytes]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > MAX_MARKDOWN_ZIP_ENTRIES
                or sum(entry.file_size for entry in entries)
                > MAX_MARKDOWN_PACKAGE_BYTES
            ):
                raise MarkdownPackageError("markdown_archive_limit")
            paths: set[str] = set()
            files: dict[str, zipfile.ZipInfo] = {}
            for entry in entries:
                path = _path(
                    entry.filename.removesuffix("/")
                    if entry.is_dir()
                    else entry.filename
                )
                identity = _path_identity(path)
                if identity in paths:
                    raise MarkdownPackageError(
                        "duplicate_or_ambiguous_package_path", path
                    )
                paths.add(identity)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    entry.flag_bits & 1
                    or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or (mode == stat.S_IFDIR) != entry.is_dir()
                    and mode != 0
                ):
                    raise MarkdownPackageError("encrypted_or_special_zip_entry", path)
                if (
                    entry.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or entry.file_size / max(1, entry.compress_size)
                    > MAX_MARKDOWN_COMPRESSION_RATIO
                ):
                    raise MarkdownPackageError(
                        "markdown_archive_compression_limit", path
                    )
                maximum = (
                    _MANIFEST_BYTES
                    if PurePosixPath(path).name == "manifest.json"
                    else MAX_MARKDOWN_DOCUMENT_BYTES
                )
                if entry.file_size > maximum:
                    raise MarkdownPackageError("markdown_document_byte_limit", path)
                if not entry.is_dir():
                    if not _is_ignored_macos_metadata(path):
                        files[path] = entry
            manifests = [
                path for path in files if PurePosixPath(path).name == "manifest.json"
            ]
            if len(manifests) != 1 or len(PurePosixPath(manifests[0]).parts) > 2:
                raise MarkdownPackageError("exactly_one_root_manifest_required")
            manifest_path = manifests[0]
            prefix = manifest_path.removesuffix("manifest.json")
            if any(not path.startswith(prefix) for path in files):
                raise MarkdownPackageError("files_outside_manifest_root")
            raw_files: dict[str, bytes] = {}
            for path, entry in files.items():
                relative = path.removeprefix(prefix)
                maximum = (
                    _MANIFEST_BYTES
                    if path == manifest_path
                    else MAX_MARKDOWN_DOCUMENT_BYTES
                )
                if entry.file_size > maximum:
                    raise MarkdownPackageError("markdown_document_byte_limit", relative)
                with archive.open(entry) as stream:
                    data = stream.read(maximum + 1)
                if len(data) != entry.file_size or len(data) > maximum:
                    raise MarkdownPackageError(
                        "invalid_or_oversized_zip_entry", relative
                    )
                raw_files[relative] = data
            try:
                manifest_data = json.loads(
                    raw_files.pop("manifest.json").decode("utf-8-sig"),
                    object_pairs_hook=_json_object,
                )
                if (
                    isinstance(manifest_data, dict)
                    and isinstance(manifest_data.get("documents"), list)
                    and len(manifest_data["documents"]) > MAX_MARKDOWN_DOCUMENTS
                ):
                    raise MarkdownPackageError("markdown_document_count_limit")
                manifest = _Manifest.model_validate(manifest_data)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValidationError,
                RecursionError,
            ):
                raise MarkdownPackageError("invalid_markdown_manifest") from None
            return manifest, raw_files
    except MarkdownPackageError:
        raise
    except (
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        zlib.error,
    ):
        raise MarkdownPackageError("invalid_markdown_zip") from None


def _ordered_documents(
    manifest: _Manifest, files: dict[str, bytes]
) -> list[_ManifestDocument]:
    main = _path(manifest.main)
    documents: dict[str, _ManifestDocument] = {}
    identities: set[str] = set()
    for document in manifest.documents:
        path = _path(document.path)
        identity = _path_identity(path)
        if identity in identities or PurePosixPath(path).suffix.lower() != ".md":
            raise MarkdownPackageError("duplicate_or_invalid_manifest_document", path)
        identities.add(identity)
        documents[path] = document
        if not document.title.strip():
            raise MarkdownPackageError("empty_document_title", path)
        if document.source_url is not None:
            try:
                url = urlsplit(document.source_url)
                if (
                    url.scheme not in {"http", "https"}
                    or not url.netloc
                    or url.username
                    or url.password
                    or any(character.isspace() for character in document.source_url)
                ):
                    raise ValueError
            except ValueError:
                raise MarkdownPackageError(
                    "invalid_document_source_url", path
                ) from None
    if main not in documents or set(documents) != set(files):
        raise MarkdownPackageError("manifest_must_exactly_list_all_markdown_files")
    if documents[main].parent is not None:
        raise MarkdownPackageError("main_document_cannot_have_parent", main)
    for document in documents.values():
        if document.path == main:
            continue
        parent = document.parent
        if parent is None or _path(parent) not in documents or parent == document.path:
            raise MarkdownPackageError("annex_requires_one_valid_parent", document.path)
        visited = {document.path}
        while parent != main:
            if (
                parent in visited
                or parent not in documents
                or documents[parent].parent is None
            ):
                raise MarkdownPackageError(
                    "manifest_parent_tree_required", document.path
                )
            visited.add(parent)
            parent = documents[parent].parent
    ordered = [documents[main]]
    for document in ordered:
        ordered.extend(
            child for child in documents.values() if child.parent == document.path
        )
    return ordered


def _target(reference: _Reference, parent: str, paths: set[str]) -> str | None:
    url = reference.url
    if any(unicodedata.category(character).startswith("C") for character in url):
        raise MarkdownPackageError("unsafe_markdown_link", parent)
    try:
        parsed = urlsplit(url)
        decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        raise MarkdownPackageError("invalid_markdown_link", parent) from None
    if PurePosixPath(decoded).suffix.lower() in _IMAGE_SUFFIXES:
        raise MarkdownPackageError("images_require_transcription", parent)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return None
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or decoded.startswith("/")
        or "\\" in decoded
    ):
        raise MarkdownPackageError("unsupported_or_unsafe_markdown_link", parent)
    target = (
        posixpath.normpath(posixpath.join(posixpath.dirname(parent), decoded))
        if decoded
        else parent
    )
    if target.startswith("../") or target == ".." or target not in paths:
        raise MarkdownPackageError("missing_local_markdown_target", parent)
    return target


def parse_markdown_package(content: bytes, display_name: str) -> AcquisitionResult:
    """Return a complete local asset graph, preserving supplied bytes and text."""
    if len(content) > MAX_MARKDOWN_UPLOAD_BYTES:
        raise MarkdownPackageError("markdown_upload_byte_limit")
    filename = display_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    extension = PurePosixPath(filename).suffix.lower()
    if extension == ".zip":
        manifest, files = _read_archive(content)
    elif extension == ".md":
        filename = _path(filename)
        files = {filename: content}
        manifest = _Manifest(
            format="legislation-markdown-v1",
            main=filename,
            documents=[
                _ManifestDocument(path=filename, title=PurePosixPath(filename).stem)
            ],
        )
    else:
        raise MarkdownPackageError("markdown_or_zip_required")
    ordered = _ordered_documents(manifest, files)
    parsed: dict[str, _ParsedDocument] = {}
    digests: dict[str, str] = {}
    assets: list[AcquiredAsset] = []
    for specification in ordered:
        path = specification.path
        raw = files[path]
        text = _decode(raw, path)
        references, title = _markdown_references(text, path)
        digest = sha256(raw).hexdigest()
        if digest in digests.values():
            raise MarkdownPackageError("duplicate_markdown_content", path)
        digests[path] = digest
        parsed[path] = _ParsedDocument(specification, raw, text, references, title)
        assets.append(
            AcquiredAsset(
                sha256=digest,
                content=raw,
                mime_type="text/markdown",
                display_name=title
                if extension == ".md" and title
                else specification.title,
                text=text,
                original_url=specification.source_url,
                final_url=specification.source_url,
            )
        )
    if (
        sum(len(document.references) for document in parsed.values())
        > MAX_MARKDOWN_LINKS
    ):
        raise MarkdownPackageError("markdown_link_limit")
    links: list[SourceLink] = []
    paths = set(parsed)
    for path, document in parsed.items():
        for ordinal, reference in enumerate(document.references, 1):
            target = _target(reference, path, paths)
            if target is not None:
                links.append(
                    SourceLink(
                        parent_url=document.specification.source_url,
                        parent_asset_hash=digests[path],
                        target_asset_hash=digests[target],
                        requested_url=target,
                        original_url=reference.literal_url,
                        source_field=f"markdown:{path}:link:{ordinal}",
                        label=reference.label or target,
                        kind="internal" if target == path else "embedded",
                    )
                )
    parent_links: list[SourceLink] = []
    for specification in ordered[1:]:
        match = next(
            (
                link
                for link in links
                if link.parent_asset_hash == digests[cast(str, specification.parent)]
                and link.target_asset_hash == digests[specification.path]
            ),
            None,
        )
        if match is None:
            raise MarkdownPackageError(
                "declared_parent_must_link_to_annex", specification.path
            )
        parent_links.append(match)
    return AcquisitionResult(
        status="ready",
        assets=assets,
        links=parent_links + [link for link in links if link not in parent_links],
    )


def build_markdown_package_template() -> bytes:
    """Build the downloadable format example from the same documented contract."""
    manifest = {
        "format": "legislation-markdown-v1",
        "main": "ana.md",
        "documents": [
            {
                "path": "ana.md",
                "title": "Örnek Ana Mevzuat",
                "source_url": "https://example.com/mevzuat",
            },
            {
                "path": "ekler/ek-1.md",
                "title": "EK-1 — Başvuru Bilgileri",
                "parent": "ana.md",
            },
            {"path": "ekler/ek-2.md", "title": "EK-2 — Süreler", "parent": "ana.md"},
        ],
    }
    files = {
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "ana.md": "# Örnek Ana Mevzuat\n\nMADDE 1 - Başvurunun esasları aşağıdaki eklerde yer alır.\n\n[EK-1: Başvuru Bilgileri](ekler/ek-1.md)\n\n[EK-2: Süreler](ekler/ek-2.md)\n",
        "ekler/ek-1.md": "# EK-1 — Başvuru Bilgileri\n\n| Alan | Açıklama |\n| --- | --- |\n| Başvuran | Adı ve unvanı |\n| Belge | Başvuru numarası |\n\n[Ana mevzuat](../ana.md)\n",
        "ekler/ek-2.md": "# EK-2 — Süreler\n\nMADDE 1 - Başvuru süresi otuz gündür.\n\n[Ana mevzuat](../ana.md)\n",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, text in files.items():
            info = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, text.encode("utf-8"))
    return output.getvalue()
