"""Fixed DEV release verification; no external probe code or fixture inputs."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import NoReturn


def safe_failure_detail(stage: str, error: BaseException) -> str:
    """Retain bounded code locations, never messages, source lines or frame locals."""
    stages = {
        "scope",
        "startup",
        "configuration",
        "native",
        "calibration",
        "pdf_vision",
        "canary",
        "token",
        "capabilities",
        "baseline",
        "source_review",
        "approval",
        "historical_chat",
        "current_chat",
        "markdown",
        "cleanup",
        "chat_cleanup",
        "token_cleanup",
    }
    types = {
        "UnicodeDecodeError",
        "ValueError",
        "ValidationError",
        "RuntimeError",
        "TypeError",
        "TimeoutError",
        "OperationalError",
        "ImportError",
        "ModuleNotFoundError",
        "IsolatedProcessTimeout",
        "IsolatedProcessCrashed",
        "AssertionError",
        "KeyError",
        "BadRequestError",
        "AuthenticationError",
        "RateLimitError",
        "APIError",
        "APIConnectionError",
        "HTTPStatusError",
        "ConnectError",
        "ReadTimeout",
        "PermissionError",
        "IntegrityError",
        "NotFoundError",
    }
    exceptions: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(exceptions) < 3:
        seen.add(id(current))
        frames: list[dict[str, str | int]] = []
        trace = current.__traceback__
        while trace is not None:
            module = trace.tb_frame.f_globals.get("__name__")
            function = trace.tb_frame.f_code.co_name
            if (
                isinstance(module, str)
                and re.fullmatch(
                    r"(?:onyx|ee\.onyx|shared_configs)\.[A-Za-z0-9_.]{1,100}", module
                )
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function)
            ):
                frames.append(
                    {"module": module, "function": function, "line": trace.tb_lineno}
                )
                frames = frames[:2] + frames[-3:] if len(frames) > 5 else frames
            trace = trace.tb_next
        name = type(current).__name__
        exceptions.append(
            {"type": name if name in types else "Exception", "frames": frames}
        )
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return json.dumps(
        {"stage": stage if stage in stages else "unknown", "exceptions": exceptions},
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_scope(*, database: str, environment: str, machine: str) -> None:
    if (database, environment, machine) != ("customs-regulations-dev", "dev", "x86_64"):
        raise ValueError("native_amd64_dev_scope_required")


def load_fixtures() -> dict[str, bytes]:
    root = Path(__file__).with_name("acceptance_fixtures")
    manifest: dict[str, str] = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {"old.pdf", "new.pdf", "page.png"}:
        raise ValueError("fixed_fixture_manifest_required")
    result: dict[str, bytes] = {}
    for name, expected in manifest.items():
        content = (root / name).read_bytes()
        if len(content) > 100_000 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("fixed_fixture_hash_mismatch")
        result[name] = content
    return result


def refuse_fetch(_url: str) -> NoReturn:
    raise ValueError("native_probe_must_not_fetch")


def native_parser_probe() -> dict[str, object]:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
    from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
    from onyx.regulatory.amendments.annexes.sources import acquire_source_package
    from onyx.utils.process_isolation import run_in_isolated_process

    fixtures = load_fixtures()
    pdf = fixtures["old.pdf"]
    package = acquire_source_package(
        content=pdf,
        mime_type="application/pdf",
        display_name="fictional-annex.pdf",
        fetch=refuse_fetch,
    )
    if package.status != "ready" or len(package.assets) != 1:
        raise ValueError("native_source_acquisition_failed")
    extracted = extract_annex_structure(pdf, "application/pdf")
    if extracted.page_count != 4 or extracted.issues != ["vision_model_unavailable"]:
        raise ValueError("native_pdf_extraction_failed")
    text = "\n".join(element.text for element in extracted.elements)
    if not all(value in text for value in ("1001.10", "EK-1", "EK-2")):
        raise ValueError("native_pdf_structure_missing")
    pages = run_in_isolated_process(
        render_annex_pages, pdf, "application/pdf", timeout=30
    )
    if [page.page for page in pages] != [1, 2, 3, 4]:
        raise ValueError("native_render_incomplete")
    image = extract_annex_structure(fixtures["page.png"], "image/png")
    if image.page_count != 1 or image.issues != ["vision_model_unavailable"]:
        raise ValueError("native_image_parser_failed")
    html = extract_annex_structure(
        b"<h1>EK-1</h1><table><tr><td>1001.10</td><td>5%</td></tr></table>",
        "text/html",
    )
    if html.issues or not any(element.text == "5%" for element in html.elements):
        raise ValueError("native_html_parser_failed")
    modules = (
        "onyx.regulatory.amendments.annexes.sources",
        "onyx.regulatory.amendments.annexes.source_parser",
        "onyx.regulatory.amendments.annexes.extraction",
        "onyx.regulatory.amendments.annexes.rendering",
        "onyx.regulatory.amendments.annexes.models",
        "onyx.utils.process_isolation",
    )
    hashes: dict[str, str] = {}
    for name in modules:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError("native_module_file_missing")
        hashes[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return {
        "native_machine": platform.machine(),
        "pdf_pages": 4,
        "module_sha256": hashes,
        "fixture_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in fixtures.items()
        },
        "render_sha256": [hashlib.sha256(page.png).hexdigest() for page in pages],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("pypdfium2", "pillow", "openpyxl", "python-docx")
        },
        "source_fetches": 0,
        "parser_limits_relaxed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "canary"))
    args = parser.parse_args()
    release_sha = os.environ.get("ANNEX_ACCEPTANCE_RELEASE_SHA", "")
    valid_release_sha = bool(re.fullmatch(r"[0-9a-f]{40}", release_sha))
    report: dict[str, object] = {}
    status = "passed"
    stage = "scope"
    try:
        validate_scope(
            database=os.environ.get("POSTGRES_DB", ""),
            environment=os.environ.get("REGULATORY_ANNEX_ENVIRONMENT", ""),
            machine=platform.machine(),
        )
        if not valid_release_sha:
            raise ValueError("exact_release_ownership_metadata_required")
        stage = "startup"
        from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

        set_is_ee_based_on_env_variable()
        stage = "configuration"
        from onyx.db.regulatory_annex_acceptance import verify_dev_configuration

        report["configuration"] = verify_dev_configuration()
        if args.phase == "preflight":
            stage = "native"
            report["native"] = native_parser_probe()
            stage = "calibration"
            from onyx.regulatory.amendments.annexes.acceptance_calibration import (
                run_native_calibration,
            )

            calibration = run_native_calibration()
            report["calibration"] = calibration
            status = "passed" if calibration.get("status") == "passed" else "failed"
            if status == "passed":
                stage = "pdf_vision"
                from onyx.regulatory.amendments.annexes.acceptance_pdf_vision import (
                    run_pdf_vision_probe,
                )

                pdf_probe = run_pdf_vision_probe()
                report["pdf_vision_probe"] = pdf_probe
                if pdf_probe.get("status") != "passed":
                    status = "failed"
                    report["failure_stage"] = stage
                    report["exception_type"] = (
                        pdf_probe.get("exception_type") or "Exception"
                    )
        else:
            stage = "canary"
            from onyx.db.engine.sql_engine import SqlEngine
            from onyx.regulatory.amendments.annexes.acceptance_canary import run_canary

            with SqlEngine.scoped_engine(pool_size=5, max_overflow=2):
                report.update(run_canary(release_sha))
            status = str(report.get("status", "passed"))
    except Exception as exc:
        status = "failed"
        exception_type = type(exc).__name__
        report.update(
            failure=safe_failure_detail(stage, exc),
            failure_stage=stage,
            exception_type=exception_type
            if exception_type
            in {
                "UnicodeDecodeError",
                "ValueError",
                "ValidationError",
                "RuntimeError",
                "TypeError",
                "TimeoutError",
                "OperationalError",
                "ImportError",
                "ModuleNotFoundError",
                "IsolatedProcessTimeout",
                "IsolatedProcessCrashed",
                "AssertionError",
                "KeyError",
            }
            else "Exception",
        )
    print(
        json.dumps(
            {
                **report,
                "phase": args.phase,
                "status": status,
                "release_sha_metadata": release_sha if valid_release_sha else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
