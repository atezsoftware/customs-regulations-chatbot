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


# --- markdown-only import ----------------------------------------------------
#
# The lightweight runtime ships no source-document parsers, but markdown needs
# none of them: it is read as plain text. These tests pin that a deployment can
# accept markdown (and archives of it) without the heavy importer stack.


def test_markdown_import_available_when_full_document_import_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", True)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module_name: object())

    assert import_capability.markdown_import_available() is True


def test_markdown_import_available_without_importer_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module_name: None)

    assert import_capability.markdown_import_available() is True
    import_capability.ensure_markdown_import_available()


def test_markdown_import_unavailable_when_both_flags_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", False)

    assert import_capability.markdown_import_available() is False

    with pytest.raises(OnyxError) as exc_info:
        import_capability.ensure_markdown_import_available()

    assert exc_info.value.error_code is OnyxErrorCode.ENV_VAR_GATED


def test_non_markdown_uploads_are_unsupported_without_importer_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", True)

    assert import_capability.supported_upload_extensions() == frozenset(
        {".md", ".mdx", ".zip"}
    )


def test_all_extensions_supported_once_the_importer_stack_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", True)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", False)

    assert import_capability.supported_upload_extensions() is None


def test_markdown_only_runtime_explains_why_a_pdf_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", True)

    reason = import_capability.unsupported_upload_reason("2024_teblig.pdf")

    assert reason is not None
    assert ".pdf" in reason


def test_markdown_only_runtime_accepts_markdown_and_archives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", False)
    monkeypatch.setattr(import_capability, "MARKDOWN_IMPORT_ENABLED", True)

    for filename in ("madde.md", "notlar.MDX", "mevzuat.zip"):
        assert import_capability.unsupported_upload_reason(filename) is None


def test_full_runtime_refuses_nothing_on_extension_grounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(import_capability, "DOCUMENT_IMPORT_ENABLED", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module_name: object())

    assert import_capability.unsupported_upload_reason("2024_teblig.pdf") is None
