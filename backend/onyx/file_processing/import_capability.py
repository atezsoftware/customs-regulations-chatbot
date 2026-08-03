import importlib.util

from onyx.configs.app_configs import DOCUMENT_IMPORT_ENABLED
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError

DOCUMENT_IMPORT_REQUIRED_MODULES = (
    "markitdown",
    "pypdfium2",
    "unstructured",
    "unstructured_client",
)


def missing_document_import_modules() -> tuple[str, ...]:
    return tuple(
        module_name
        for module_name in DOCUMENT_IMPORT_REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
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
