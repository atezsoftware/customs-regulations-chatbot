"""PDF startup avoids unrelated converters while format-specific hooks delegate."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from onyx.regulatory.amendments.annexes import sources


def test_fresh_pdf_parser_import_does_not_load_unrelated_converters() -> None:
    backend = Path(sources.__file__).resolve().parents[4]
    program = """
import json
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError('Parser imports must remain offline')

socket.socket.connect = forbidden
socket.create_connection = forbidden
from onyx.regulatory.amendments.annexes.source_parser import apply_source_process_limits
apply_source_process_limits()
from onyx.regulatory.amendments.annexes import sources
import pypdf
unrelated = (
    'onyx.regulatory.amendments.source_extraction',
    'onyx.utils.web_content',
    'onyx.file_processing.extract_file_text',
    'onyx.file_processing.unstructured',
    'openpyxl', 'docx', 'numpy', 'unstructured',
)
loaded = [name for name in sys.modules if any(
    name == prefix or name.startswith(prefix + '.') for prefix in unrelated
)]
assert callable(sources.amendment_source_http_options)
assert callable(sources.decode_html_bytes)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=backend,
        env={**os.environ, "PYTHONPATH": str(backend)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_http_options_hook_delegates_without_changing_security_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments import source_extraction

    expected = source_extraction.AmendmentSourceHttpOptions(
        headers={"Accept": "application/pdf"}, verify="existing-ca-bundle.pem"
    )
    calls: list[str] = []

    def options(url: str) -> source_extraction.AmendmentSourceHttpOptions:
        calls.append(url)
        return expected

    monkeypatch.setattr(source_extraction, "amendment_source_http_options", options)
    assert (
        sources.amendment_source_http_options("https://example.test/source") is expected
    )
    assert calls == ["https://example.test/source"]


def test_html_decoder_hook_preserves_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.utils import web_content

    calls: list[tuple[bytes, str | None, str | None]] = []

    def decode(
        content: bytes,
        content_type: str | None = None,
        fallback_encoding: str | None = None,
    ) -> str:
        calls.append((content, content_type, fallback_encoding))
        return "Özgün metin"

    monkeypatch.setattr(web_content, "decode_html_bytes", decode)
    assert (
        sources.decode_html_bytes(b"source", "text/html", "windows-1254")
        == "Özgün metin"
    )
    assert sources.decode_html_bytes(b"source") == "Özgün metin"
    assert calls == [(b"source", "text/html", "windows-1254"), (b"source", None, None)]
