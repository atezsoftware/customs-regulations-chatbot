"""Expands uploaded zip archives into one upload per contained markdown document.

Callers hand the result straight to the normal user-file upload path, so every
entry is categorized, stored, and indexed exactly as if it had been uploaded on
its own.

Only markdown is taken out of an archive. Bulk regulatory sources arrive as
bundles -- a directory per source document holding its markdown rendering next
to sidecars (checksums, routing manifests) that carry no regulatory text and
would otherwise be indexed as empty documents.

Names keep their path relative to the archive root, because that name becomes
`UserFile.name`, which the indexing pipeline turns into the document's semantic
identifier and, for regulatory documents, the chunk's `source_file`. A directory
holding a single markdown file is collapsed into it: such a directory is the
document's identity (the bundle layout names it after the source file), while
the markdown inside is generically named.

An archive that looks hostile is rejected whole rather than partially expanded,
so an operator never sees a "successful" import that silently dropped documents.
Benign packaging noise (directories, macOS resource forks, dotfiles, sidecars)
is skipped without failing the upload.
"""

import mimetypes
import stat
import zipfile
from collections import Counter
from collections.abc import Sequence
from io import BytesIO
from pathlib import PurePosixPath

from fastapi import UploadFile
from starlette.datastructures import Headers

from onyx.configs.app_configs import (
    MAX_ARCHIVE_COMPRESSION_RATIO,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_EXPANDED_BYTES,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing.file_types import OnyxFileExtensions

# Below this size the compression ratio says more about zip framing overhead
# than about the entry's contents, so the ratio check would fire on honest
# small files (a few KB of whitespace-heavy markdown compresses very well).
COMPRESSION_RATIO_CHECK_MIN_BYTES = 1024 * 1024

ARCHIVE_EXTENSION = ".zip"
ARCHIVE_MIME_TYPES = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-zip",
        "multipart/x-zip",
    }
)
MARKDOWN_EXTENSIONS = frozenset({".md", ".mdx"})
_ARCHIVE_ROOT = PurePosixPath(".")
_MACOS_METADATA_DIR = "__MACOSX"
_DEFAULT_MIME_TYPE = "application/octet-stream"


def is_archive_upload(upload: UploadFile) -> bool:
    if (upload.content_type or "").lower() in ARCHIVE_MIME_TYPES:
        return True
    return (upload.filename or "").lower().endswith(ARCHIVE_EXTENSION)


def _reject(archive_name: str, reason: str) -> OnyxError:
    return OnyxError(
        OnyxErrorCode.INVALID_INPUT,
        f"Archive '{archive_name}' was rejected: {reason}",
    )


def _reject_too_large(archive_name: str, reason: str) -> OnyxError:
    return OnyxError(
        OnyxErrorCode.PAYLOAD_TOO_LARGE,
        f"Archive '{archive_name}' was rejected: {reason}",
    )


def _entry_path(entry: zipfile.ZipInfo) -> PurePosixPath:
    # Zip stores forward slashes; some Windows writers emit backslashes anyway.
    return PurePosixPath(entry.filename.replace("\\", "/"))


def _ensure_entry_is_safe(entry: zipfile.ZipInfo, archive_name: str) -> None:
    """Reject anything that could write outside the archive root or recurse."""

    path = _entry_path(entry)
    if path.is_absolute() or ".." in path.parts:
        raise _reject(
            archive_name, f"entry '{entry.filename}' points outside the archive"
        )
    if stat.S_ISLNK(entry.external_attr >> 16):
        raise _reject(archive_name, f"entry '{entry.filename}' is a symbolic link")
    if entry.filename.lower().endswith(ARCHIVE_EXTENSION):
        raise _reject(
            archive_name,
            f"entry '{entry.filename}' is a nested archive; unpack it first",
        )


def _ensure_entry_is_not_a_bomb(entry: zipfile.ZipInfo, archive_name: str) -> None:
    if (
        entry.file_size >= COMPRESSION_RATIO_CHECK_MIN_BYTES
        and entry.compress_size > 0
        and entry.file_size / entry.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO
    ):
        raise _reject_too_large(
            archive_name,
            f"entry '{entry.filename}' expands more than "
            f"{MAX_ARCHIVE_COMPRESSION_RATIO}x its stored size",
        )


def _is_wanted(entry: zipfile.ZipInfo) -> bool:
    """Markdown documents only; directories, dotfiles, and sidecars are noise."""

    if entry.is_dir():
        return False
    path = _entry_path(entry)
    if _MACOS_METADATA_DIR in path.parts:
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    return path.suffix.lower() in MARKDOWN_EXTENSIONS


def _upload_name(path: PurePosixPath, siblings: int) -> str:
    """Collapse a directory that exists only to hold this one document."""

    directory = path.parent
    if directory == _ARCHIVE_ROOT or siblings > 1:
        return str(path)

    # Bundle directories are named after the source document, extension and
    # all ("...yonergesi.docx/"). Drop that extension so the indexed name reads
    # as the document rather than as the file it was converted from -- but only
    # when it really is one, so a directory whose name merely contains dots
    # ("2006-11_karar") keeps every character.
    stem = (
        directory.stem
        if directory.suffix.lower() in OnyxFileExtensions.TEXT_AND_DOCUMENT_EXTENSIONS
        else directory.name
    )
    return str(directory.parent / f"{stem}{path.suffix}")


def _read_entry(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    archive_name: str,
    remaining_bytes: int,
) -> bytes:
    """Read one entry, refusing to materialize more than the remaining budget.

    Reads one byte past the budget so an entry whose declared size understates
    its real contents is caught rather than trusted.
    """

    with archive.open(entry, "r") as entry_stream:
        payload = entry_stream.read(remaining_bytes + 1)
    if len(payload) > remaining_bytes:
        raise _reject_too_large(
            archive_name,
            f"it expands to more than the {MAX_ARCHIVE_EXPANDED_BYTES} byte limit",
        )
    return payload


def _entry_as_upload(entry_name: str, payload: bytes) -> UploadFile:
    content_type = mimetypes.guess_type(entry_name)[0] or _DEFAULT_MIME_TYPE
    return UploadFile(
        file=BytesIO(payload),
        size=len(payload),
        filename=entry_name,
        headers=Headers({"content-type": content_type}),
    )


def _expand_one_archive(upload: UploadFile) -> list[UploadFile]:
    archive_name = upload.filename or "archive"
    upload.file.seek(0)
    expanded: list[UploadFile] = []
    try:
        with zipfile.ZipFile(upload.file, "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise _reject_too_large(
                    archive_name,
                    f"it holds {len(entries)} entries, more than the "
                    f"{MAX_ARCHIVE_ENTRIES} allowed per archive",
                )
            # Vet every entry, including the ones we won't extract: a nested
            # archive or a symlink is a property of the upload, not of the
            # documents we happen to want out of it.
            for entry in entries:
                _ensure_entry_is_safe(entry, archive_name)
                _ensure_entry_is_not_a_bomb(entry, archive_name)

            wanted = [entry for entry in entries if _is_wanted(entry)]
            # Sidecars are already gone, so a bundle directory now looks like
            # what it is: a directory holding exactly one document.
            per_directory = Counter(_entry_path(entry).parent for entry in wanted)

            remaining_bytes = MAX_ARCHIVE_EXPANDED_BYTES
            for entry in wanted:
                path = _entry_path(entry)
                payload = _read_entry(archive, entry, archive_name, remaining_bytes)
                remaining_bytes -= len(payload)
                expanded.append(
                    _entry_as_upload(
                        _upload_name(path, per_directory[path.parent]), payload
                    )
                )
    except zipfile.BadZipFile as exc:
        raise _reject(
            archive_name, f"it could not be read as a zip file ({exc})"
        ) from exc
    return expanded


def expand_archive_uploads(files: Sequence[UploadFile]) -> list[UploadFile]:
    """Replace each zip upload with its documents, leaving other uploads untouched."""

    expanded: list[UploadFile] = []
    for upload in files:
        if is_archive_upload(upload):
            expanded.extend(_expand_one_archive(upload))
        else:
            expanded.append(upload)
    return expanded
