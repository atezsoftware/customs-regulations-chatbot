"""Fixed, read-only DEV proof of the ordinary mixed-PDF image drafting path."""

import hashlib
import importlib
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import patch
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from onyx.file_store.file_store import FileStore
    from onyx.llm.interfaces import LLM

FIXTURE_SHA = "644179953512c4c5730d23b5a22204a24ded67a873f3d358662ee520c90122c5"
FIXTURE_NAME = "ordinary-mixed-pdf-vision.pdf"
OLD_TEXT = "MADDE 3\nUrun | Oran\nBugday | 5%\nPirinc | 11%\nDiger hukumler degismez."
INSTRUCTION = "Bugday orani asagidaki tabloda gosterilen deger olarak degistirilmistir. Diger oranlar degismemistir."
MODULES = (
    "onyx.regulatory.amendments.annexes.acceptance_pdf_vision",
    "onyx.regulatory.amendments.pdf_vision",
    "onyx.regulatory.amendments.drafter",
    "onyx.regulatory.amendments.annexes.extraction",
)
MAX_REPORT_BYTES = 32_000


class PdfVisionProbeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["running", "passed", "failed"] = "running"
    probe_stage: Literal["setup", "source", "draft", "grounding", "complete"] = "setup"
    attempt_count: int = Field(default=0, ge=0, le=3)
    http_request_count: int = Field(default=0, ge=0, le=3)
    attempt_count_complete: bool = True
    fixture_verified: bool = False
    fixture_sha256: dict[str, str] = Field(
        default_factory=lambda: {FIXTURE_NAME: FIXTURE_SHA}
    )
    module_sha256: dict[str, str] = Field(default_factory=dict)
    model_snapshot: dict[str, str] = Field(default_factory=dict)
    database_read_only: bool = False
    native_value_absent: bool = False
    image_evidence: bool = False
    grounding_verified: bool = False
    page_count: int = Field(default=0, ge=0, le=1)
    transcript_sha256: str | None = None
    draft_sha256: str | None = None
    receipt_sha256: str | None = None
    failure: str | None = None
    exception_type: str | None = None


def load_fixture() -> bytes:
    content = (
        Path(__file__)
        .with_name("acceptance_fixtures")
        .joinpath(FIXTURE_NAME)
        .read_bytes()
    )
    if len(content) > 100_000 or hashlib.sha256(content).hexdigest() != FIXTURE_SHA:
        raise ValueError("fixed_pdf_fixture_hash_mismatch")
    return content


class MemoryEvidenceStore:
    """Only private fixture originals and derivatives; never a persistent store."""

    def __init__(self, content: bytes) -> None:
        self.blobs = {"original": content}

    def read_file(self, identifier: str) -> BytesIO:
        return BytesIO(self.blobs[identifier])

    def save_file(self, stream: BytesIO, **_kwargs: object) -> str:
        content = stream.read(1_000_001)
        if len(content) > 1_000_000 or len(self.blobs) >= 4:
            raise ValueError("fixed_pdf_evidence_limit")
        identifier = f"fixture-{len(self.blobs)}"
        self.blobs[identifier] = content
        return identifier


def normalized_draft(text: str) -> str:
    lines = [
        line.replace("|", " ").replace("**", "").removeprefix("# ")
        for line in text.splitlines()
        if not re.fullmatch(r"[\s|:-]+", line)
    ]
    return " ".join(" ".join(lines).split()).casefold()


def record_failure(report: PdfVisionProbeReport, error: BaseException) -> None:
    from onyx.regulatory.amendments.annexes.dev_acceptance import safe_failure_detail

    report.status = "failed"
    report.failure = safe_failure_detail("pdf_vision", error)
    name = type(error).__name__
    report.exception_type = (
        name
        if name
        in {
            "ValueError",
            "ValidationError",
            "RuntimeError",
            "TypeError",
            "TimeoutError",
            "KeyError",
            "IsolatedProcessTimeout",
            "IsolatedProcessCrashed",
        }
        else "Exception"
    )


def run_probe(
    llm: "LLM",
    report: PdfVisionProbeReport,
    emit: Callable[[PdfVisionProbeReport], None],
) -> None:
    import httpx
    import litellm

    from onyx.regulatory.amendments import drafter, pdf_vision
    from onyx.regulatory.amendments.annexes.dev_acceptance import refuse_fetch
    from onyx.regulatory.amendments.annexes.models import AcquisitionResult
    from onyx.regulatory.amendments.annexes.sources import acquire_source_package
    from onyx.regulatory.amendments.models import AmendmentInstruction

    original_completion, original_send = litellm.completion, httpx.Client.send
    completed: set[str] = set()
    sent: set[str] = set()

    def counted_completion(*args: Any, **kwargs: Any) -> Any:
        stage = report.probe_stage
        if (
            stage not in {"source", "draft", "grounding"}
            or stage in completed
            or kwargs.get("mock_response")
        ):
            raise ValueError("pdf_probe_completion_retry_refused")
        completed.add(stage)
        report.attempt_count += 1
        emit(report)
        kwargs.update(num_retries=0, max_retries=0)
        return original_completion(*args, **kwargs)

    def counted_send(
        client: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        stage = report.probe_stage
        if stage not in completed or stage in sent:
            raise ValueError("pdf_probe_http_retry_refused")
        sent.add(stage)
        report.http_request_count += 1
        emit(report)
        return original_send(client, request, **kwargs)

    try:
        content = load_fixture()
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
        memory = MemoryEvidenceStore(content)
        store = cast("FileStore", memory)
        package = acquire_source_package(
            content=content,
            mime_type="application/pdf",
            display_name=FIXTURE_NAME,
            fetch=refuse_fetch,
        )
        if (
            package.status != "ready"
            or len(package.assets) != 1
            or package.issues
            or package.links
        ):
            raise ValueError("pdf_probe_source_acquisition_failed")
        original = package.assets[0]
        if (
            "17%" in original.text
            or "11%" in original.text
            or "MADDE 3" not in original.text
        ):
            raise ValueError("pdf_probe_native_value_not_absent")
        report.native_value_absent = True
        with (
            patch.object(httpx.Client, "send", counted_send),
            patch.object(litellm, "completion", counted_completion),
            patch.object(llm, "invoke", partial(llm.invoke, use_streaming=False)),
        ):
            report.probe_stage = "source"
            emit(report)
            asset = pdf_vision.prepare_pdf_source(
                original, store=store, llm=llm, deadline=time.monotonic() + 60
            )
            if (
                asset.native_text != original.text
                or asset.pdf_vision is None
                or "17%" not in asset.text
                or "11%" not in asset.text
            ):
                raise ValueError("pdf_probe_visual_value_missing")
            report.transcript_sha256 = asset.pdf_vision.transcript_sha256
            manifest = (
                AcquisitionResult(status="ready", assets=[asset])
                .model_dump_json()
                .encode()
            )
            memory.blobs["manifest"] = manifest
            source = pdf_vision.PdfBatchSource(
                batch_id=1,
                package_id=UUID(int=1),
                source_text_sha256=pdf_vision.digest(asset.text),
                manifest_file_id="manifest",
                manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                originals=[
                    pdf_vision.PdfOriginal(
                        asset_id=UUID(int=2), file_id="original", sha256=FIXTURE_SHA
                    )
                ],
                user_file_ids=[UUID(int=3)],
            )
            instructions = [
                AmendmentInstruction(
                    instruction_text=INSTRUCTION, article_reference="MADDE 3"
                )
            ]
            evidence = pdf_vision.prepare_pdf_draft_evidence(
                source, instructions, store
            )
            if (
                evidence is None
                or len(evidence.image_parts) != 1
                or len(evidence.pages) != 1
                or evidence.pages[0].page != 1
            ):
                raise ValueError("pdf_probe_selected_original_image_required")
            report.image_evidence = True
            report.page_count = 1
            old = {
                "id": "fictional-madde-3",
                "text": OLD_TEXT,
                "chunk_type": "article",
                "heading_path": ["MADDE 3"],
                "chunk_metadata": {"article_no": "3"},
            }
            report.probe_stage = "draft"
            emit(report)
            draft = drafter.draft_combined_chunk(
                llm,
                instructions=instructions,
                old_chunk=old,
                sibling_reference=None,
                reference_date=None,
                pdf_evidence=evidence,
            )
            if (
                normalized_draft(draft.new_chunk.text)
                != normalized_draft(OLD_TEXT.replace("5%", "17%"))
                or draft.new_chunk.metadata_changes
                or draft.new_chunk.heading_path not in (None, ["MADDE 3"])
                or draft.new_chunk.chunk_type not in (None, "article")
                or draft.dates.effective_start_date is not None
                or draft.dates.effective_end_date is not None
            ):
                raise ValueError("pdf_probe_unintended_draft_change")
            report.draft_sha256 = pdf_vision.digest(draft.new_chunk.text)
            report.probe_stage = "grounding"
            emit(report)
            receipt = pdf_vision.verify_pdf_draft(
                llm,
                evidence=evidence,
                instructions=instructions,
                old_chunk=old,
                draft_text=draft.new_chunk.text,
            )
            pdf_vision.validate_pdf_frozen_references(source, receipt, store)
            if (
                receipt.old_chunk_id != old["id"]
                or receipt.old_snapshot_sha256 != pdf_vision.snapshot_digest(old)
                or receipt.draft_text_sha256 != report.draft_sha256
                or receipt.pages != evidence.pages
                or receipt.manifest_sha256 != source.manifest_sha256
                or receipt.source_text_sha256 != source.source_text_sha256
                or receipt.batch_id != source.batch_id
                or receipt.package_id != source.package_id
            ):
                raise ValueError("pdf_probe_grounding_receipt_mismatch")
            report.receipt_sha256 = hashlib.sha256(
                receipt.model_dump_json().encode()
            ).hexdigest()
            report.grounding_verified = True
        if completed != {"source", "draft", "grounding"} or sent != completed:
            raise ValueError("pdf_probe_three_actual_calls_required")
        report.probe_stage = "complete"
        report.status = "passed"
    except Exception as error:
        record_failure(report, error)
    emit(report)


def _child() -> None:
    output = os.fdopen(os.dup(1), "w")
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    report = PdfVisionProbeReport()

    def emit(value: PdfVisionProbeReport) -> None:
        output.write(value.model_dump_json(exclude_none=True) + "\n")
        output.flush()

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
            llm = get_default_llm_with_vision(timeout=45, temperature=0)
            if llm is None:
                raise ValueError("configured_vision_required")
            run_probe(llm, report, emit)
    except Exception as error:
        record_failure(report, error)
        emit(report)
    finally:
        output.close()


def report_from_output(output: bytes, failure: str | None = None) -> dict[str, object]:
    report = PdfVisionProbeReport(status="failed", failure="child_report_missing")
    invalid = False
    for line in output.splitlines(keepends=True):
        if not line.endswith(b"\n") or len(line) > MAX_REPORT_BYTES:
            invalid = True
            continue
        try:
            report = PdfVisionProbeReport.model_validate_json(line)
        except ValueError:
            invalid = True
    if report.status == "passed" and (
        report.probe_stage != "complete"
        or report.attempt_count != 3
        or report.http_request_count != 3
        or not all(
            (
                report.fixture_verified,
                report.native_value_absent,
                report.image_evidence,
                report.grounding_verified,
            )
        )
        or report.page_count != 1
        or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (
                report.transcript_sha256,
                report.draft_sha256,
                report.receipt_sha256,
            )
        )
    ):
        invalid = True
    if failure or invalid or report.status == "running":
        report.status = "failed"
        report.failure = failure or "child_report_invalid"
        report.attempt_count_complete = False
    return report.model_dump(mode="json", exclude_none=True)


def run_pdf_vision_probe() -> dict[str, object]:
    if (
        os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
        or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
    ):
        return PdfVisionProbeReport(
            status="failed", failure="dev_scope_required"
        ).model_dump(mode="json", exclude_none=True)
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from onyx.regulatory.amendments.annexes.acceptance_pdf_vision import _child; _child()",
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
            output, _ = process.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            failure = "process_timeout"
        if process.returncode and failure is None:
            failure = "process_failed"
        return report_from_output(output, failure)
    except OSError:
        return PdfVisionProbeReport(
            status="failed", failure="process_start_failed"
        ).model_dump(mode="json", exclude_none=True)
