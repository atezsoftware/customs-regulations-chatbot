"""Acquire mixed document archives without making the archive a legislation."""

import base64
import hashlib
import io
import json
import posixpath
import stat
import time
import unicodedata
import zipfile
import zlib
from collections import deque
from collections.abc import Callable
from pathlib import PurePosixPath
from urllib.parse import unquote, urldefrag, urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AcquisitionResult,
    DiscoveredLink,
    SourceInspection,
    SourceIssue,
    SourceLink,
)
from onyx.regulatory.amendments.annexes.sources import (
    _ANNEX,
    MAX_ANNEX_ASSETS,
    MAX_ASSET_BYTES,
    MAX_LINK_DEPTH,
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_SECONDS,
    MAX_RELATIONSHIPS,
    DownloadedSource,
    SourceAcquisitionError,
    _bounded_inspect,
    _validate_occurrence_mime,
    download_source_bounded,
)
from onyx.regulatory.document_intake.markdown_package import (
    MarkdownPackageError,
    _decode,
    _is_ignored_macos_metadata,
    _json_object,
    _markdown_references,
    _path,
    _path_identity,
)
from onyx.regulatory.document_intake.source_graph import (
    ZIP_INPUT_KIND,
    IntakeGraph,
    parse_intake_graph,
)
from onyx.utils.process_isolation import run_in_isolated_process

MAX_ZIP_UPLOAD_BYTES = MAX_ASSET_BYTES
MAX_ZIP_ENTRIES = 128
_MANIFEST_BYTES = 256 * 1024


class ZipPackageError(ValueError):
    """Bounded diagnostics expose a member path, never extracted source text."""

    def __init__(self, code: str, path: str | None = None) -> None:
        self.code = code
        self.path = path
        super().__init__(code, path)

    @property
    def public_message(self) -> str:
        messages = {
            "unsafe_zip": "ZIP güvenli biçimde okunamadı; şifreleme, özel dosya, çakışan yol veya arşiv sınırlarını kontrol edin.",
            "invalid_zip_manifest": "ZIP manifest.json dosyası bütün belgeleri path, isteğe bağlı parent ve source_url alanlarıyla listelemelidir.",
            "duplicate_zip_content": "ZIP içinde içeriği birebir aynı olan birden çok belge var. Yinelenen dosyayı kaldırın.",
            "invalid_zip_graph": "Eklerin üst belge ilişkisi belirsiz veya döngülü. manifest.json ile gerçek kaynak bağlantısına dayanan tek bir parent belirtin.",
            "zip_input_integrity_changed": "Yüklenen ZIP bütünlüğü değişti; paketi yeniden yükleyin.",
        }
        message = messages.get(self.code, "ZIP belgesi işlenemedi: " + self.code)
        return message + (" Dosya: " + ascii(self.path[:512]) if self.path else "")


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    parent: str | None = None
    source_url: str | None = None


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    documents: list[_Document] = Field(min_length=1, max_length=MAX_ANNEX_ASSETS + 1)


class ZipAcquisitionResult(AcquisitionResult):
    intake_graph: IntakeGraph


class _MarkdownInspection(SourceInspection):
    literal_urls: dict[str, str]


def _read_zip(content: bytes) -> tuple[dict[str, bytes], _Manifest | None]:
    if not content or len(content) > MAX_ZIP_UPLOAD_BYTES:
        raise ZipPackageError("unsafe_zip")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > MAX_ZIP_ENTRIES
                or sum(e.file_size for e in entries) > MAX_PACKAGE_BYTES
            ):
                raise ZipPackageError("unsafe_zip")
            identities: set[str] = set()
            files: dict[str, bytes] = {}
            for entry in entries:
                path = _path(
                    entry.filename.removesuffix("/")
                    if entry.is_dir()
                    else entry.filename
                )
                identity = _path_identity(path)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    identity in identities
                    or entry.flag_bits & 1
                    or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                ):
                    raise ZipPackageError("unsafe_zip", path)
                identities.add(identity)
                if mode and (mode == stat.S_IFDIR) != entry.is_dir():
                    raise ZipPackageError("unsafe_zip", path)
                if (
                    entry.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or entry.file_size / max(1, entry.compress_size) > 100
                ):
                    raise ZipPackageError("unsafe_zip", path)
                maximum = (
                    _MANIFEST_BYTES
                    if PurePosixPath(path).name == "manifest.json"
                    else MAX_ASSET_BYTES
                )
                if entry.file_size > maximum:
                    raise ZipPackageError("unsafe_zip", path)
                if entry.is_dir() or _is_ignored_macos_metadata(path):
                    continue
                with archive.open(entry) as stream:
                    raw = stream.read(maximum + 1)
                if len(raw) != entry.file_size or len(raw) > maximum:
                    raise ZipPackageError("unsafe_zip", path)
                files[path] = raw
    except (
        zipfile.BadZipFile,
        OSError,
        RuntimeError,
        zlib.error,
        MarkdownPackageError,
    ) as error:
        raise ZipPackageError("unsafe_zip") from error
    manifests = [path for path in files if PurePosixPath(path).name == "manifest.json"]
    manifest = None
    if manifests:
        if len(manifests) != 1 or len(PurePosixPath(manifests[0]).parts) > 2:
            raise ZipPackageError("invalid_zip_manifest")
        manifest_path = manifests[0]
        prefix = manifest_path.removesuffix("manifest.json")
        if any(not path.startswith(prefix) for path in files):
            raise ZipPackageError("invalid_zip_manifest")
        try:
            manifest = _Manifest.model_validate(
                json.loads(
                    files.pop(manifest_path).decode("utf-8-sig"),
                    object_pairs_hook=_json_object,
                )
            )
            files = {path.removeprefix(prefix): raw for path, raw in files.items()}
            paths = [_path(document.path) for document in manifest.documents]
            if len(set(paths)) != len(paths) or set(paths) != set(files):
                raise ZipPackageError("invalid_zip_manifest")
            urls: set[str] = set()
            for document in manifest.documents:
                if document.parent is not None and (
                    _path(document.parent) not in files
                    or document.parent == document.path
                ):
                    raise ZipPackageError("invalid_zip_manifest", document.path)
                if document.source_url is not None:
                    parsed = urlsplit(document.source_url)
                    if (
                        parsed.scheme not in {"http", "https"}
                        or not parsed.netloc
                        or parsed.username
                        or parsed.password
                        or parsed.fragment
                        or document.source_url in urls
                    ):
                        raise ZipPackageError("invalid_zip_manifest", document.path)
                    urls.add(document.source_url)
        except (ValueError, RecursionError, MarkdownPackageError) as error:
            if isinstance(error, ZipPackageError):
                raise
            raise ZipPackageError("invalid_zip_manifest") from error
    if not files or len(files) > MAX_ANNEX_ASSETS + 1:
        raise ZipPackageError("asset_count_limit")
    return files, manifest


def _inspect_markdown(content: bytes, path: str) -> _MarkdownInspection:
    try:
        text = _decode(content, path)
        references, _title = _markdown_references(text, path)
    except MarkdownPackageError as error:
        raise ZipPackageError(error.code, error.path) from error
    return _MarkdownInspection(
        mime_type="text/markdown",
        text=text,
        links=[
            DiscoveredLink(
                kind="url",
                target=reference.url,
                label=reference.label,
                source_field=f"markdown:{path}:link:{index}",
            )
            for index, reference in enumerate(references, 1)
        ],
        literal_urls={
            f"markdown:{path}:link:{index}": reference.literal_url
            for index, reference in enumerate(references, 1)
        },
    )


def _local_target(target: str, parent: str) -> str:
    parsed = urlsplit(target)
    decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or decoded.startswith("/")
        or "\\" in decoded
    ):
        raise ZipPackageError("unsafe_local_link", parent)
    path = (
        posixpath.normpath(posixpath.join(posixpath.dirname(parent), decoded))
        if decoded
        else parent
    )
    try:
        return _path(path)
    except MarkdownPackageError as error:
        raise ZipPackageError("unsafe_local_link", parent) from error


def acquire_zip_package(
    content: bytes,
    display_name: str,
    *,
    fetch: Callable[[str], DownloadedSource] | None = None,
    deadline: float | None = None,
) -> ZipAcquisitionResult:
    """Use one bounded queue for every real member and its actual annex links."""
    if not display_name.lower().endswith(".zip"):
        raise ZipPackageError("zip_filename_required")
    deadline = min(deadline or float("inf"), time.monotonic() + MAX_PACKAGE_SECONDS)
    if time.monotonic() >= deadline:
        raise ZipPackageError("time_limit")
    files, manifest = run_in_isolated_process(
        _read_zip, content, timeout=max(0.1, min(30, deadline - time.monotonic()))
    )
    specifications = (
        {document.path: document for document in manifest.documents} if manifest else {}
    )
    by_path = {path: hashlib.sha256(raw).hexdigest() for path, raw in files.items()}
    if len(set(by_path.values())) != len(by_path):
        raise ZipPackageError("duplicate_zip_content")
    by_url = {
        document.source_url: by_path[path]
        for path, document in specifications.items()
        if document.source_url
    }
    final_by_url = {address: address for address in by_url}
    assets: dict[str, AcquiredAsset] = {}
    inspections: dict[str, SourceInspection] = {}
    expanded: set[tuple[str, str | None, str | None]] = set()
    links: list[SourceLink] = []
    issues: list[SourceIssue] = []
    parents: dict[str, set[str]] = {}
    # Included files have independent identities; their archive order has no legal meaning.
    pending: deque[
        tuple[bytes | None, str, str | None, int, SourceLink | None, bool]
    ] = deque(
        (
            raw,
            path,
            specifications[path].source_url if path in specifications else None,
            0,
            None,
            True,
        )
        for path, raw in files.items()
    )
    total = 0
    fetch_source = fetch or (
        lambda address: download_source_bounded(address, deadline=deadline)
    )
    while pending:
        raw, path, address, depth, relation, included = pending.popleft()
        try:
            if time.monotonic() >= deadline:
                raise SourceAcquisitionError("time_limit")
            if depth > MAX_LINK_DEPTH:
                raise SourceAcquisitionError("depth_limit")
            declared: str | None = None
            final_url = address
            count_bytes = True
            if raw is None:
                if address in by_url and by_url[address] in assets:
                    cached = assets[by_url[address]]
                    raw, declared = cached.content, cached.mime_type
                    final_url = final_by_url[address]
                    count_bytes = False
                else:
                    if address is None:
                        raise SourceAcquisitionError("missing_source")
                    downloaded = fetch_source(address)
                    raw, declared, final_url = (
                        downloaded.content,
                        downloaded.mime_type,
                        downloaded.final_url,
                    )
            if not raw or len(raw) > MAX_ASSET_BYTES:
                raise SourceAcquisitionError("asset_byte_limit")
            digest = hashlib.sha256(raw).hexdigest()
            if relation:
                relation.target_asset_hash = digest
                relation.final_url = final_url
            if count_bytes:
                total += len(raw)
            if total > MAX_PACKAGE_BYTES:
                raise SourceAcquisitionError("package_byte_limit")
            if digest not in assets and len(assets) >= MAX_ANNEX_ASSETS + 1:
                raise SourceAcquisitionError("asset_count_limit")
            if digest not in inspections:
                inspections[digest] = (
                    run_in_isolated_process(
                        _inspect_markdown,
                        raw,
                        path,
                        timeout=max(0.1, min(30, deadline - time.monotonic())),
                    )
                    if PurePosixPath(path).suffix.lower() == ".md"
                    else _bounded_inspect(
                        raw,
                        "text/plain" if path.lower().endswith(".txt") else declared,
                        deadline,
                    )
                )
            inspection = inspections[digest]
            if inspection.mime_type != "text/markdown":
                _validate_occurrence_mime(inspection.mime_type, declared, path)
            if digest not in assets:
                assets[digest] = AcquiredAsset(
                    sha256=digest,
                    content=raw,
                    mime_type=inspection.mime_type,
                    display_name=path,
                    original_url=address,
                    final_url=final_url,
                    text=inspection.text,
                    page_count=inspection.page_count,
                )
            if address:
                by_url[address] = digest
                final_by_url[address] = final_url
            if final_url:
                by_url[final_url] = digest
                final_by_url[final_url] = final_url
            occurrence = (digest, final_url, path if included else None)
            if occurrence in expanded:
                continue
            expanded.add(occurrence)
            for discovered in inspection.links:
                remote_reference = urlsplit(discovered.target)
                if (
                    inspection.mime_type == "text/markdown"
                    and remote_reference.scheme in {"http", "https"}
                    and not (
                        _ANNEX.search(discovered.label)
                        or _ANNEX.search(PurePosixPath(remote_reference.path).stem)
                    )
                ):
                    continue
                if len(links) >= MAX_RELATIONSHIPS:
                    raise SourceAcquisitionError("relationship_limit")
                if any(
                    unicodedata.category(character).startswith("C")
                    for character in discovered.target
                ):
                    raise SourceAcquisitionError("unsafe_url")
                link = SourceLink(
                    parent_url=final_url,
                    parent_asset_hash=digest,
                    source_page=discovered.source_page,
                    source_field=discovered.source_field,
                    label=discovered.label,
                    kind=discovered.kind,
                    original_url=(
                        inspection.literal_urls[discovered.source_field]
                        if isinstance(inspection, _MarkdownInspection)
                        else discovered.target
                    ),
                )
                links.append(link)
                if discovered.kind == "internal":
                    link.target_asset_hash = digest
                    continue
                if discovered.kind == "embedded":
                    pending.append(
                        (
                            base64.b64decode(
                                discovered.embedded_base64 or "", validate=True
                            ),
                            discovered.target,
                            None,
                            depth + 1,
                            link,
                            False,
                        )
                    )
                    continue
                target = urlsplit(discovered.target)
                if included and not target.scheme and not target.netloc:
                    local = _local_target(discovered.target, path)
                    link.requested_url = local
                    if local not in by_path:
                        raise ZipPackageError("missing_local_zip_target", local)
                    link.target_asset_hash = by_path[local]
                    link.kind = "internal" if local == path else "embedded"
                else:
                    remote = urldefrag(urljoin(final_url or "", discovered.target))[0]
                    if urlsplit(remote).scheme not in {"http", "https"}:
                        raise SourceAcquisitionError("unsafe_url")
                    link.requested_url = remote
                    if remote in by_url:
                        link.target_asset_hash = by_url[remote]
                        link.final_url = final_by_url[remote]
                    else:
                        pending.append(
                            (
                                None,
                                PurePosixPath(urlsplit(remote).path).name or "source",
                                remote,
                                depth + 1,
                                link,
                                False,
                            )
                        )
        except (
            SourceAcquisitionError,
            MarkdownPackageError,
            ZipPackageError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            issues.append(
                SourceIssue(
                    code=getattr(error, "code", "invalid_zip_member"),
                    locator=getattr(error, "path", None)
                    or (relation.requested_url if relation else None)
                    or path,
                    retryable=not included,
                )
            )
    # Only annex references imply ownership; ordinary cross references remain navigation.
    for link in links:
        child = link.target_asset_hash
        if (
            child
            and child != link.parent_asset_hash
            and (
                (link.kind == "embedded" and link.requested_url is None)
                or _ANNEX.search(link.label)
                or _ANNEX.search(
                    PurePosixPath(urlsplit(link.original_url or "").path).stem
                )
            )
        ):
            parents.setdefault(child, set()).add(link.parent_asset_hash)
    declared_parents = {
        by_path[path]: by_path[document.parent] if document.parent else None
        for path, document in specifications.items()
    }
    ownership: dict[str, str] = {}
    for child, candidates in parents.items():
        if child not in declared_parents:
            if len(candidates) != 1:
                raise ZipPackageError("invalid_zip_graph")
            ownership[child] = next(iter(candidates))
    ownership.update(
        {child: parent for child, parent in declared_parents.items() if parent}
    )
    # A failed included file must not create an apparent complete subset.
    graph_hashes = set(assets)
    if not graph_hashes:
        first = issues[0] if issues else None
        raise ZipPackageError(
            first.code if first else "empty_zip", first.locator if first else None
        )
    ownership = {
        child: parent
        for child, parent in ownership.items()
        if child in graph_hashes and parent in graph_hashes
    }
    try:
        graph = IntakeGraph(
            version=1,
            root_hashes=sorted(graph_hashes.difference(ownership)),
            primary_parent_by_hash=ownership,
        )
    except ValidationError as error:
        raise ZipPackageError("invalid_zip_graph") from error
    result = ZipAcquisitionResult(
        status="blocked" if issues else "ready",
        assets=list(assets.values()),
        links=links,
        issues=issues,
        intake_graph=graph,
    )
    if not issues:
        try:
            parse_intake_graph(
                result.model_dump(),
                input_kind=ZIP_INPUT_KIND,
                asset_hashes=graph_hashes,
                links=links,
            )
        except (ValueError, ValidationError) as error:
            raise ZipPackageError("invalid_zip_graph") from error
    return result
