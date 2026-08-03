import importlib.util

import pytest

from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing import import_capability


def test_document_import_disabled_fails_before_dependency_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    dependency_probe_called = False

    def unexpected_dependency_probe(_module_name: str) -> object:
        nonlocal dependency_probe_called
        dependency_probe_called = True
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", unexpected_dependency_probe)

    with pytest.raises(OnyxError) as exc_info:
        import_capability.ensure_document_import_available()

    assert exc_info.value.error_code is OnyxErrorCode.ENV_VAR_GATED
    assert dependency_probe_called is False


def test_document_import_enabled_requires_importer_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", True)

    def dependency_probe(module_name: str) -> object | None:
        if module_name in {"markitdown", "unstructured_client"}:
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", dependency_probe)

    with pytest.raises(OnyxError) as exc_info:
        import_capability.ensure_document_import_available()

    assert exc_info.value.error_code is OnyxErrorCode.SERVICE_UNAVAILABLE
    assert "markitdown" in exc_info.value.detail
    assert "unstructured_client" in exc_info.value.detail


def test_document_import_enabled_accepts_complete_importer_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module_name: object())

    import_capability.ensure_document_import_available()
