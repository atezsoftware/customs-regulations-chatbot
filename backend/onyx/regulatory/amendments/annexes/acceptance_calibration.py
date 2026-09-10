"""Fixed native correction calibration for the reviewed DEV acceptance hook."""

import hashlib
import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from onyx.llm.interfaces import LLM

FIXTURE_HASHES = {
    "docx": "bcfc88fb5920e1f182ad7080b5ad14fb0d4c8d9812caabadab24f5370b65c8ed",
    "xlsx": "355db77e7b6d79a189a0b40d6ed32893d90955f0a8b0e41ec74c873ea675a719",
}
MODULES = (
    "onyx.regulatory.amendments.annexes.acceptance_calibration",
    "onyx.regulatory.amendments.annexes.corrections",
    "onyx.regulatory.amendments.annexes.extraction",
    "onyx.prompts.regulatory_annex_review",
)
MAX_RATIONALE_CHARACTERS = 4000
# Escaped astral characters need 12 bytes each; reserve 64 KiB for metadata.
MAX_REPORT_LINE_BYTES = 4 * MAX_RATIONALE_CHARACTERS * 12 + 64 * 1024


class CalibrationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["docx", "xlsx"]
    proposed_value: Literal["7%", "9%"]
    expected_supported: bool
    original_value: Literal["7%"] = "7%"
    raw_transcription: Literal["5%"] = "5%"
    supported: bool | None = None
    rationale: str | None = Field(default=None, max_length=MAX_RATIONALE_CHARACTERS)
    rationale_truncated: bool = False
    input_sha256: str | None = None
    status: Literal["running", "passed", "failed"] = "running"
    failure: (
        Literal["case_failed", "completion_retry_refused", "http_retry_refused"] | None
    ) = None
    attempt_count: int = Field(default=0, ge=0, le=1)
    http_request_count: int = Field(default=0, ge=0, le=1)


class CalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["passed", "failed"] = "failed"
    planned_cases: Literal[4] = 4
    cases: list[CalibrationCase] = Field(default_factory=list, max_length=4)
    attempt_count: int = Field(default=0, ge=0, le=4)
    attempt_count_complete: bool = True
    fixture_sha256: dict[str, str] = Field(default_factory=lambda: dict(FIXTURE_HASHES))
    module_sha256: dict[str, str] = Field(default_factory=dict)
    model_snapshot: dict[str, str] = Field(default_factory=dict)
    database_read_only: bool = False
    fixture_verified: bool = False
    failure: str | None = None


def load_native_fixtures() -> dict[str, bytes]:
    root = Path(__file__).with_name("acceptance_fixtures")
    fixtures = {}
    for format, expected in FIXTURE_HASHES.items():
        content = (root / f"native-correction.{format}").read_bytes()
        if len(content) > 50_000 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("fixed_fixture_hash_mismatch")
        fixtures[format] = content
    return fixtures


class FrozenNativeStore:
    def __init__(self, content: bytes, file_id: str) -> None:
        self.content = content
        self.file_id = file_id

    def read_file(self, file_id: str) -> BytesIO:
        if file_id != self.file_id:
            raise ValueError("fixed_original_required")
        return BytesIO(self.content)


def run_cases(
    llm: "LLM", report: CalibrationReport, emit: Callable[[CalibrationReport], None]
) -> None:
    import httpx
    import litellm

    from onyx.regulatory.amendments.annexes import corrections
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
    from onyx.regulatory.amendments.annexes.models import (
        AnnexChangeDraft,
        AnnexElementCorrection,
        AnnexReviewEvidence,
    )

    fixtures = load_native_fixtures()
    report.fixture_verified = True
    report.model_snapshot = {
        "model_provider": llm.config.model_provider,
        "model_name": llm.config.model_name,
    }
    for name in MODULES:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise ValueError("module_file_required")
        report.module_sha256[name] = hashlib.sha256(
            Path(filename).read_bytes()
        ).hexdigest()
    emit(report)
    original_completion = litellm.completion
    original_send = httpx.Client.send
    for format in ("docx", "xlsx"):
        content = fixtures[format]
        mime = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if format == "docx"
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        for proposed, expected in (("7%", True), ("9%", False)):
            case = CalibrationCase(
                format=format, proposed_value=proposed, expected_supported=expected
            )
            report.cases.append(case)
            try:
                original = extract_annex_structure(content, mime)
                matches = [
                    index
                    for index, element in enumerate(original.elements)
                    if element.text == "7%"
                ]
                if original.issues or len(matches) != 1:
                    raise ValueError("fixed_native_value_required")
                position = matches[0]
                raw = original.model_copy(deep=True)
                raw.elements[position].text = "5%"
                file_id = f"frozen-{format}"
                evidence = AnnexReviewEvidence(
                    id=uuid4(),
                    side="new",
                    kind="original",
                    file_id=file_id,
                    sha256=FIXTURE_HASHES[format],
                    mime_type=mime,
                    byte_count=len(content),
                    parent_file_id=f"original-{format}",
                    parent_sha256=FIXTURE_HASHES[format],
                )
                draft = AnnexChangeDraft(
                    instruction_indices=[0],
                    instruction_texts=["EK-1 oranının kaynak doğrulaması."],
                    annex_label="EK-1",
                    raw_new_extraction=raw,
                    new_extraction=raw,
                    evidence=[evidence],
                )
                edit = AnnexElementCorrection(
                    position=position,
                    before_text="5%",
                    corrected_text=proposed,
                    reason="Özgün dosyanın ilgili konumundaki değere göre düzeltme önerisi.",
                )

                def counted_completion(*args: Any, **kwargs: Any) -> Any:
                    if case.attempt_count:
                        case.failure = "completion_retry_refused"
                        raise ValueError("completion_retry_refused")
                    if kwargs.get("mock_response"):
                        raise ValueError("actual_provider_required")
                    kwargs.update(num_retries=0, max_retries=0)
                    case.attempt_count = 1
                    report.attempt_count += 1
                    emit(report)
                    return original_completion(*args, **kwargs)

                def counted_send(
                    client: httpx.Client, request: httpx.Request, **kwargs: Any
                ) -> httpx.Response:
                    if not case.attempt_count:
                        raise ValueError("calibration_auxiliary_http_refused")
                    if case.http_request_count:
                        case.failure = "http_retry_refused"
                        raise ValueError("http_retry_refused")
                    case.http_request_count = 1
                    emit(report)
                    return original_send(client, request, **kwargs)

                with (
                    patch.object(httpx.Client, "send", counted_send),
                    patch.object(
                        corrections,
                        "get_default_file_store",
                        return_value=FrozenNativeStore(content, file_id),
                    ),
                    patch.object(
                        corrections,
                        "generate_structured",
                        partial(
                            corrections.generate_structured,
                            max_attempts=1,
                            provider_max_attempts=1,
                        ),
                    ),
                    patch.object(
                        llm, "invoke", partial(llm.invoke, use_streaming=False)
                    ),
                    patch.object(litellm, "completion", counted_completion),
                ):
                    receipt = corrections.reconcile_corrections(
                        draft=draft, corrections=[edit], llm=llm
                    )
                case.supported = receipt.supported
                case.rationale = receipt.rationale[:MAX_RATIONALE_CHARACTERS]
                case.rationale_truncated = (
                    len(receipt.rationale) > MAX_RATIONALE_CHARACTERS
                )
                case.input_sha256 = receipt.input_sha256
                case.status = (
                    "passed"
                    if receipt.supported is expected
                    and case.attempt_count == 1
                    and case.http_request_count == 1
                    else "failed"
                )
            except Exception:
                case.status = "failed"
                case.failure = case.failure or "case_failed"
            emit(report)
    report.status = (
        "passed"
        if len(report.cases) == 4
        and all(case.status == "passed" for case in report.cases)
        else "failed"
    )
    if report.status == "failed":
        report.failure = "calibration_failed"
    emit(report)


def _child() -> None:
    # A private pipe carries only allowlisted records; SDK logs cannot enter it.
    output = os.fdopen(os.dup(1), "w")
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def emit(report: CalibrationReport) -> None:
        output.write(report.model_dump_json() + "\n")
        output.flush()

    report = CalibrationReport()
    try:
        if (
            os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
            or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
        ):
            raise ValueError("dev_scope_required")
        from onyx.db.engine.sql_engine import SqlEngine
        from onyx.db.regulatory_annex_dev_cutover import configured_indices
        from onyx.llm.factory import get_default_llm_with_vision
        from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
        from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

        set_is_ee_based_on_env_variable()
        CURRENT_TENANT_ID_CONTEXTVAR.set("public")
        # Existing DB helper verifies actual DEV identity in a read-only transaction.
        configured_indices()
        with SqlEngine.scoped_engine(
            pool_size=2,
            max_overflow=0,
            connect_args={
                "options": "-c default_transaction_read_only=on",
                "connect_timeout": 10,
            },
        ):
            report.database_read_only = True
            llm = get_default_llm_with_vision(timeout=60, temperature=0)
            if llm is None:
                raise ValueError("configured_vision_required")
            run_cases(llm, report, emit)
    except Exception:
        report.status = "failed"
        report.failure = "calibration_setup_failed"
        emit(report)
    finally:
        output.close()


def report_from_output(output: bytes, *, failure: str | None) -> dict[str, object]:
    report = CalibrationReport(failure=failure or "child_report_missing")
    invalid_output = False
    for line in output.splitlines(keepends=True):
        if not line.endswith(b"\n") or len(line) > MAX_REPORT_LINE_BYTES:
            invalid_output = True
            continue
        try:
            report = CalibrationReport.model_validate_json(line)
        except ValueError:
            invalid_output = True
    if failure or invalid_output:
        report.status = "failed"
        report.failure = failure or "child_report_invalid"
        report.attempt_count_complete = False
    return report.model_dump(mode="json")


def run_native_calibration() -> dict[str, object]:
    """Return retained evidence even when a model verdict, crash or timeout fails."""
    if (
        os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
        or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
    ):
        return CalibrationReport(failure="dev_scope_required").model_dump(mode="json")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from onyx.regulatory.amendments.annexes.acceptance_calibration import _child; _child()",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                **os.environ,
                "PGOPTIONS": "-c default_transaction_read_only=on",
                "PGCONNECT_TIMEOUT": "10",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PYTHONPATH": os.pathsep.join(sys.path),
            },
        )
        failure = None
        try:
            output, _ = process.communicate(timeout=330)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            failure = "process_timeout"
        if process.returncode and failure is None:
            failure = "process_failed"
        return report_from_output(output, failure=failure)
    except OSError:
        return CalibrationReport(failure="process_start_failed").model_dump(mode="json")
