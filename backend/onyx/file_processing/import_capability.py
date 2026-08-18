"""Which uploads a runtime can actually ingest.

Two capabilities, deliberately separate. Full document import needs the
source-document parser stack (PDF, docx, pptx, …) and is gated by
DOCUMENT_IMPORT_ENABLED. Markdown import needs none of it — markdown is read as
plain text — so a lightweight runtime that ships without those parsers can still
accept markdown and zip archives of it.

Advertising the narrower capability instead of the broader one keeps the API
honest: a deployment that cannot parse a PDF rejects it at upload with a clear
reason rather than failing somewhere deep in the indexing pipeline.
"""

import importlib.util
from pathlib import PurePosixPath

from onyx.configs.app_configs import DOCUMENT_IMPORT_ENABLED, MARKDOWN_IMPORT_ENABLED
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError

DOCUMENT_IMPORT_REQUIRED_MODULES = (
    "markitdown",
    "pypdfium2",
    "unstructured",
    "unstructured_client",
)

# Extensions a markdown-only runtime accepts. Archives are included because the
# upload path expands them to markdown before anything is parsed.
MARKDOWN_UPLOAD_EXTENSIONS = frozenset({".md", ".mdx", ".zip"})


def missing_document_import_modules() -> tuple[str, ...]:
    return tuple(
        module_name
        for module_name in DOCUMENT_IMPORT_REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
    )


def document_import_available() -> bool:
    return DOCUMENT_IMPORT_ENABLED and not missing_document_import_modules()


def markdown_import_available() -> bool:
    """Full document import implies markdown import; the reverse does not hold."""

    return MARKDOWN_IMPORT_ENABLED or document_import_available()


def supported_upload_extensions() -> frozenset[str] | None:
    """Extensions this runtime accepts, or None when everything is supported."""

    if document_import_available():
        return None
    return MARKDOWN_UPLOAD_EXTENSIONS


def unsupported_upload_reason(filename: str) -> str | None:
    """Why this runtime cannot ingest the file, or None when it can.

    Extension-based, so it runs before any parsing is attempted and the caller
    can report a per-file reason instead of failing the whole upload.
    """

    supported = supported_upload_extensions()
    if supported is None:
        return None

    extension = PurePosixPath(filename).suffix.lower()
    if extension in supported:
        return None
    return (
        f"Unsupported file type: {extension or filename}. This deployment "
        "accepts Markdown documents and .zip archives of them; other formats "
        "are converted by the separate importer deployment."
    )


def ensure_document_import_available() -> None:
    if not DOCUMENT_IMPORT_ENABLED:
        raise OnyxError(
            OnyxErrorCode.ENV_VAR_GATED,
            "Document import is disabled in this runtime. Use the separate importer deployment.",
        )

    missing_modules = missing_document_import_modules()
    if missing_modules:
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE,
            "Document import dependencies are unavailable in this runtime: "
            + ", ".join(missing_modules),
        )


def ensure_markdown_import_available() -> None:
    if markdown_import_available():
        return

    raise OnyxError(
        OnyxErrorCode.ENV_VAR_GATED,
        "Document import is disabled in this runtime. Use the separate importer deployment.",
    )
