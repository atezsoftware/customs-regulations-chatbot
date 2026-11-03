"""PROPOSED ONLY: one payload-free native6455 probe in a Ready DEV background image.

Root must review and execute this through the existing read-only diagnostic
workflow. Pass the selected immutable background-image SHA as argv[1]. This
never creates or retries a source package and never calls a provider or DB.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

URL = (
    "https://www5.tbmm.gov.tr/tutanaklar/KANUNLAR_KARARLAR/"
    "kanuntbmmc097/kanuntbmmc097/kanuntbmmc09706455.pdf"
)
EXPECTED_BYTES = 396152
EXPECTED_PDF_SHA256 = "3f5ccdac09f51a64037e931deb0ccf1257ac941bd2c0d284ead3885e068d7366"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,64}$")


class DiagnosticFenceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def main() -> int:
    began = time.monotonic()
    report: dict[str, object] = {}
    timing: dict[str, float] = {}
    phase_started: float | None = None
    phase = "fence"
    exit_code = 0
    previous_logging_disable = logging.root.manager.disable
    try:
        # Own diagnostic-process output only. The real parser also discards its
        # child stdout/stderr; no PDF text, URI labels or exception messages leak.
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            logging.disable(logging.CRITICAL)
            from onyx import __version__

            if not _REVISION.fullmatch(__version__):
                raise DiagnosticFenceError("runtime_revision_unavailable")
            report["revision"] = __version__
            expected = sys.argv[1] if len(sys.argv) == 2 else ""
            if not _REVISION.fullmatch(expected):
                raise DiagnosticFenceError("expected_revision_required")
            if __version__ != expected:
                raise DiagnosticFenceError("runtime_revision_mismatch")
            if sys.platform != "linux":
                raise DiagnosticFenceError("linux_background_image_required")

            from onyx.regulatory.amendments.annexes import (
                source_limits,
                source_parser,
                sources,
            )

            report["code_sha256"] = {
                "sources": hashlib.sha256(
                    Path(sources.__file__).read_bytes()
                ).hexdigest(),
                "source_parser": hashlib.sha256(
                    Path(source_parser.__file__).read_bytes()
                ).hexdigest(),
                "source_limits": hashlib.sha256(
                    Path(source_limits.__file__).read_bytes()
                ).hexdigest(),
                "bounded_inspect": hashlib.sha256(
                    inspect.getsource(sources._bounded_inspect).encode()
                ).hexdigest(),
            }
            if (
                sources.MAX_PACKAGE_SECONDS != 180
                or source_limits.SOURCE_ACQUISITION_SECONDS != 180
            ):
                raise DiagnosticFenceError("unexpected_source_acquisition_budget")

            # This is one fresh production-sized acquisition budget, not an
            # extension/reset of a historical package's remaining deadline.
            package_began = time.monotonic()
            deadline = package_began + sources.MAX_PACKAGE_SECONDS
            phase = "download"
            phase_started = time.monotonic()
            downloaded = sources.download_source_bounded(URL, deadline=deadline)
            timing["download_seconds"] = round(time.monotonic() - phase_started, 6)
            phase_started = None
            phase = "integrity"
            payload = downloaded.content
            if len(payload) != EXPECTED_BYTES:
                raise DiagnosticFenceError("official_pdf_byte_count_mismatch")
            if hashlib.sha256(payload).hexdigest() != EXPECTED_PDF_SHA256:
                raise DiagnosticFenceError("official_pdf_sha256_mismatch")
            if not payload.startswith(b"%PDF-"):
                raise DiagnosticFenceError("official_pdf_magic_mismatch")

            remaining = deadline - time.monotonic()
            timing["package_remaining_before_inspect_seconds"] = round(remaining, 6)
            if remaining < 30:
                raise DiagnosticFenceError("insufficient_fresh_parse_budget")

            phase = "inspect"
            phase_started = time.monotonic()
            # Exactly one actual production parser invocation. Its own 30s wall,
            # child CPU25s and Linux AS2GiB limits are neither patched nor raised.
            result = sources._bounded_inspect(payload, "application/pdf", deadline)
            timing["inspect_seconds"] = round(time.monotonic() - phase_started, 6)
            phase_started = None
            timing["package_seconds"] = round(time.monotonic() - package_began, 6)
            report["counts"] = {
                "pages": result.page_count,
                "text_characters": len(result.text),
                "links": len(result.links),
            }
    except Exception as error:
        if phase_started is not None:
            timing[phase + "_seconds"] = round(time.monotonic() - phase_started, 6)
        code = getattr(error, "code", "unexpected_error")
        report["error"] = {
            "class": type(error).__name__,
            "code": code
            if isinstance(code, str) and _ERROR_CODE.fullmatch(code)
            else "unclassified_error",
        }
        exit_code = 2
    finally:
        logging.disable(previous_logging_disable)
        timing["total_seconds"] = round(time.monotonic() - began, 6)
        report["elapsed"] = timing
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
