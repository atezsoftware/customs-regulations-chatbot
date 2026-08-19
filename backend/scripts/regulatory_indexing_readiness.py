"""Read-only production readiness checks for durable regulatory indexing.

This command intentionally has no write-capable database, object-store, or
Elasticsearch operations. It performs one constant-text embedding request but
never prints or persists the returned vector.
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import datetime
import hashlib
import hmac
import json
import re
import shlex
import socket
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import httpx
from alembic.config import Config
from alembic.script import ScriptDirectory

from onyx.background.celery.regulatory_indexing_beat_health import (
    BEAT_PROCESS_NAME,
    validate_regulatory_indexing_beat,
)
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.llm import fetch_embedding_provider, fetch_model_configuration_by_id
from onyx.db.schema_version import get_database_alembic_heads
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.schema import (
    CONTENT_VECTOR_FIELD_NAME,
    TITLE_VECTOR_FIELD_NAME,
    DocumentSchema,
)
from onyx.llm.well_known_providers.constants import VERTEX_CREDENTIALS_FILE_KWARG
from onyx.natural_language_processing.constants import OPENROUTER_EMBEDDINGS_URL
from onyx.regulatory.indexing_jobs.configuration import (
    resolve_regulatory_indexing_snapshot,
)
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from onyx.regulatory.indexing_jobs.vertex_batch import GoogleVertexBatchGateway
from shared_configs.configs import MULTI_TENANT
from shared_configs.enums import EmbeddingProvider

EXIT_READY = 0
EXIT_NOT_READY = 1
EXIT_USAGE_ERROR = 2

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SUPERVISOR_CONFIG = Path("/etc/supervisor/conf.d/supervisord.conf")
_LOCAL_SUPERVISOR_CONFIG = _BACKEND_ROOT / "supervisord-lite.conf"
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_WORKER_PROCESS_NAME = "celery_worker_regulatory_indexing"
_PROBE_TEXT = "regulatory indexing readiness probe"
_EXPECTED_REGULATORY_QUEUES = frozenset({"user_file_processing", "regulatory_indexing"})
_ATTESTATION_MAX_AGE = datetime.timedelta(hours=24)
_ATTESTATION_OWNER_UID = 1001
_ATTESTATION_OWNER_GID = 1001
_ATTESTATION_MAX_BYTES = 64 * 1024
_CAPABILITY_EVIDENCE_MAX_BYTES = 4 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_CAPABILITY_PERMISSIONS = frozenset(
    {
        "storage.objects.create",
        "storage.objects.get",
        "storage.objects.delete",
        "storage.objects.list",
        "aiplatform.batchPredictionJobs.create",
        "aiplatform.batchPredictionJobs.get",
        "aiplatform.batchPredictionJobs.cancel",
        "aiplatform.batchPredictionJobs.list",
        "aiplatform.models.get",
    }
)


class ReadinessCheckError(RuntimeError):
    """An operator-safe readiness failure message."""


@dataclass(frozen=True)
class ReadinessSnapshot:
    search_settings_id: int
    embedding_provider: str
    embedding_model: str
    effective_dimension: int
    index_name: str
    vertex_model: str
    vertex_project: str
    vertex_location: str
    gcs_uri: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["PASS", "FAIL", "BLOCKED"]
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    results: tuple[CheckResult, ...]

    @property
    def exit_code(self) -> int:
        return (
            EXIT_READY
            if all(result.status == "PASS" for result in self.results)
            else EXIT_NOT_READY
        )


class ReadinessBackend(Protocol):
    def check_migration(self) -> str: ...

    def check_worker_and_beat(self) -> str: ...

    def check_memory_headroom(self) -> str: ...

    def load_snapshot(self) -> ReadinessSnapshot: ...

    def check_capability_attestation(self, snapshot: ReadinessSnapshot) -> str: ...

    def check_gcs_access(self, snapshot: ReadinessSnapshot) -> str: ...

    def check_vertex_access(self, snapshot: ReadinessSnapshot) -> str: ...

    def probe_embedding_dimension(self, snapshot: ReadinessSnapshot) -> int: ...

    def check_elasticsearch_mapping(self, snapshot: ReadinessSnapshot) -> str: ...


def _safe_failure(error: Exception) -> str:
    if isinstance(error, ReadinessCheckError):
        return str(error)
    return type(error).__name__


def _run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, status="PASS", detail=check())
    except Exception as error:
        return CheckResult(name=name, status="FAIL", detail=_safe_failure(error))


def run_readiness_checks(backend: ReadinessBackend) -> ReadinessReport:
    results = [
        _run_check("migration", backend.check_migration),
        _run_check("worker_and_beat", backend.check_worker_and_beat),
        _run_check("memory_headroom", backend.check_memory_headroom),
    ]
    try:
        snapshot = backend.load_snapshot()
    except Exception as error:
        results.append(
            CheckResult(
                name="admin_snapshot",
                status="FAIL",
                detail=_safe_failure(error),
            )
        )
        results.extend(
            CheckResult(
                name=name, status="BLOCKED", detail="admin snapshot unavailable"
            )
            for name in (
                "capability_attestation",
                "gcs_access",
                "vertex_access",
                "embedding_probe",
                "elasticsearch_mapping",
            )
        )
        return ReadinessReport(results=tuple(results))

    results.append(
        CheckResult(
            name="admin_snapshot",
            status="PASS",
            detail=(
                f"search_settings_id={snapshot.search_settings_id} "
                f"provider={snapshot.embedding_provider} "
                f"embedding_model={snapshot.embedding_model} "
                f"effective_dimension={snapshot.effective_dimension} "
                f"index={snapshot.index_name} "
                f"vertex_model={snapshot.vertex_model} "
                f"vertex_location={snapshot.vertex_location}"
            ),
        )
    )
    results.extend(
        (
            _run_check(
                "capability_attestation",
                lambda: backend.check_capability_attestation(snapshot),
            ),
            _run_check("gcs_access", lambda: backend.check_gcs_access(snapshot)),
            _run_check("vertex_access", lambda: backend.check_vertex_access(snapshot)),
            _run_check(
                "embedding_probe",
                lambda: _embedding_probe_detail(backend, snapshot),
            ),
            _run_check(
                "elasticsearch_mapping",
                lambda: backend.check_elasticsearch_mapping(snapshot),
            ),
        )
    )
    return ReadinessReport(results=tuple(results))


def _embedding_probe_detail(
    backend: ReadinessBackend, snapshot: ReadinessSnapshot
) -> str:
    actual_dimension = backend.probe_embedding_dimension(snapshot)
    if actual_dimension != snapshot.effective_dimension:
        raise ReadinessCheckError(
            "embedding response dimension mismatch: "
            f"actual={actual_dimension} expected={snapshot.effective_dimension}"
        )
    return f"response_dimension={actual_dimension}"


def format_report(report: ReadinessReport) -> str:
    lines = ["Durable regulatory indexing readiness"]
    lines.extend(
        f"{result.name}: {result.status} ({result.detail})" for result in report.results
    )
    lines.append(
        f"result: {'READY' if report.exit_code == EXIT_READY else 'NOT READY'}"
    )
    return "\n".join(lines)


class OnyxReadinessBackend:
    """Production adapters restricted to read/list/probe operations."""

    def __init__(
        self,
        *,
        memory_headroom_reviewed: bool,
        capability_attestation_path: Path | None,
        capability_evidence_path: Path | None,
    ) -> None:
        self._memory_headroom_reviewed = memory_headroom_reviewed
        self._capability_attestation_path = capability_attestation_path
        self._capability_evidence_path = capability_evidence_path
        self._attested_identity: str | None = None
        self._config_snapshot: RegulatoryIndexingConfigSnapshot | None = None
        self._embedding_api_key: str | None = None
        self._vertex_gateway: GoogleVertexBatchGateway | None = None

    def check_migration(self) -> str:
        config = Config(str(_BACKEND_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
        application_heads = frozenset(ScriptDirectory.from_config(config).get_heads())
        with get_session_with_current_tenant() as db_session:
            database_heads = get_database_alembic_heads(db_session)
        if not application_heads or database_heads != application_heads:
            raise ReadinessCheckError(
                "database migration heads do not match this application image"
            )
        return f"application/database heads match ({len(application_heads)} head)"

    def check_worker_and_beat(self) -> str:
        config_path = (
            _SUPERVISOR_CONFIG
            if _SUPERVISOR_CONFIG.is_file()
            else _LOCAL_SUPERVISOR_CONFIG
        )
        config_text = config_path.read_text(encoding="utf-8")
        beat_section = f"[program:{BEAT_PROCESS_NAME}]"
        if beat_section not in config_text:
            raise ReadinessCheckError(
                "dedicated regulatory worker or Beat supervisor config is missing"
            )
        configured_queues = _configured_regulatory_worker_queues(config_text)

        result = subprocess.run(
            [
                "supervisorctl",
                "-c",
                str(_SUPERVISOR_CONFIG),
                "status",
                _WORKER_PROCESS_NAME,
                BEAT_PROCESS_NAME,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ReadinessCheckError("supervisor status query failed")
        status_text = result.stdout
        worker_lines = [
            line
            for line in status_text.splitlines()
            if line.startswith(_WORKER_PROCESS_NAME)
        ]
        if len(worker_lines) != 1 or " RUNNING " not in worker_lines[0]:
            raise ReadinessCheckError("dedicated regulatory worker is not RUNNING")
        validate_regulatory_indexing_beat(status_text)
        expected_worker_name = f"regulatory_indexing@{socket.gethostname()}"
        supervisor_pid = _supervisor_worker_pid(worker_lines[0])
        live_queues = _live_regulatory_worker_queues(
            expected_worker_name=expected_worker_name,
            supervisor_pid=supervisor_pid,
        )
        return (
            "dedicated worker RUNNING with exact configured/live queue set "
            f"{','.join(sorted(configured_queues & live_queues))}; "
            "Beat readiness/liveness probes valid"
        )

    def check_memory_headroom(self) -> str:
        if not self._memory_headroom_reviewed:
            raise ReadinessCheckError(
                "operator must review and archive background cgroup/node headroom"
            )
        current = (_CGROUP_ROOT / "memory.current").read_text(encoding="utf-8").strip()
        peak = (_CGROUP_ROOT / "memory.peak").read_text(encoding="utf-8").strip()
        maximum = (_CGROUP_ROOT / "memory.max").read_text(encoding="utf-8").strip()
        event_lines = (
            (_CGROUP_ROOT / "memory.events").read_text(encoding="utf-8").splitlines()
        )
        events = {
            name: int(value)
            for line in event_lines
            for name, value in [line.split(maxsplit=1)]
        }
        oom = events.get("oom", 0)
        oom_kill = events.get("oom_kill", 0)
        if oom or oom_kill:
            raise ReadinessCheckError(
                f"cgroup reports OOM events: oom={oom} oom_kill={oom_kill}"
            )
        return (
            "operator review attested; "
            f"memory.current={current} memory.peak={peak} memory.max={maximum} "
            "oom=0 oom_kill=0; external node/Helm headroom remains operator-owned"
        )

    def load_snapshot(self) -> ReadinessSnapshot:
        with get_session_with_current_tenant() as db_session:
            config_snapshot = resolve_regulatory_indexing_snapshot(db_session)
            embedding_provider = fetch_embedding_provider(
                db_session, EmbeddingProvider.OPENROUTER
            )
            model_configuration = fetch_model_configuration_by_id(
                db_session, config_snapshot.vertex.model_configuration_id
            )
            if embedding_provider is None or embedding_provider.api_key is None:
                raise ReadinessCheckError("OpenRouter credential is unavailable")
            embedding_api_key = embedding_provider.api_key.get_value(apply_mask=False)
            if model_configuration is None:
                raise ReadinessCheckError("Vertex model configuration is unavailable")
            credential_json = cast(
                str | None,
                (model_configuration.llm_provider.custom_config or {}).get(
                    VERTEX_CREDENTIALS_FILE_KWARG
                ),
            )

        self._config_snapshot = config_snapshot
        self._embedding_api_key = embedding_api_key
        self._vertex_gateway = GoogleVertexBatchGateway(
            config=config_snapshot.vertex,
            object_prefix="readiness-only",
            credential_json_provider=lambda: credential_json,
        )
        return ReadinessSnapshot(
            search_settings_id=config_snapshot.search_settings_id,
            embedding_provider=config_snapshot.embedding_provider.value,
            embedding_model=config_snapshot.embedding_model_name,
            effective_dimension=config_snapshot.effective_dimension,
            index_name=config_snapshot.index_name,
            vertex_model=config_snapshot.vertex.model_name,
            vertex_project=config_snapshot.vertex.project,
            vertex_location=config_snapshot.vertex.location,
            gcs_uri=config_snapshot.vertex.gcs_uri,
        )

    def _gateway(self) -> GoogleVertexBatchGateway:
        if self._vertex_gateway is None:
            raise ReadinessCheckError("admin snapshot has not been loaded")
        return self._vertex_gateway

    def check_capability_attestation(self, snapshot: ReadinessSnapshot) -> str:
        if (
            self._capability_attestation_path is None
            or self._capability_evidence_path is None
        ):
            raise ReadinessCheckError(
                "capability attestation and archived IAM evidence are required"
            )
        self._attested_identity = _validate_capability_attestation(
            self._capability_attestation_path,
            self._capability_evidence_path,
            snapshot,
        )
        return (
            "fresh operator evidence matches active identity and exact GCS/Vertex scope"
        )

    def _require_attested_identity(self) -> str:
        if self._attested_identity is None:
            raise ReadinessCheckError(
                "capability attestation must pass before observational probes"
            )
        return self._attested_identity

    def check_gcs_access(self, snapshot: ReadinessSnapshot) -> str:
        bucket_name, prefix = _parse_gcs_uri(snapshot.gcs_uri)
        del prefix
        attested_identity = self._require_attested_identity()
        probe = self._gateway().probe_gcs_read_access()
        if probe.credential_identity != attested_identity:
            raise ReadinessCheckError(
                "GCS probe credential identity does not match capability attestation"
            )
        return f"observed GCS list access for gs://{bucket_name}"

    def check_vertex_access(self, snapshot: ReadinessSnapshot) -> str:
        del snapshot
        attested_identity = self._require_attested_identity()
        probe = self._gateway().probe_vertex_read_access()
        if probe.credential_identity != attested_identity:
            raise ReadinessCheckError(
                "Vertex probe credential identity does not match capability attestation"
            )
        return "observed Vertex model get and batch list access"

    def probe_embedding_dimension(self, snapshot: ReadinessSnapshot) -> int:
        if self._embedding_api_key is None:
            raise ReadinessCheckError("OpenRouter credential is unavailable")
        return asyncio.run(
            _probe_openrouter_embedding(
                api_key=self._embedding_api_key,
                model=snapshot.embedding_model,
                effective_dimension=snapshot.effective_dimension,
            )
        )

    def check_elasticsearch_mapping(self, snapshot: ReadinessSnapshot) -> str:
        client = ElasticsearchIndexClient(snapshot.index_name)
        try:
            expected = DocumentSchema.get_document_schema(
                vector_dimension=snapshot.effective_dimension,
                multitenant=MULTI_TENANT,
            )
            if not client.validate_index(expected):
                raise ReadinessCheckError(
                    "active Elasticsearch index mapping is incompatible"
                )
            _validate_dense_vector_mapping(
                mapping=client.get_index_mapping(),
                expected=expected,
                effective_dimension=snapshot.effective_dimension,
            )
        finally:
            client.close()
        return (
            f"{snapshot.index_name} mapping matches "
            f"effective_dimension={snapshot.effective_dimension}"
        )


def _configured_regulatory_worker_queues(config_text: str) -> set[str]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(config_text)
        command = parser.get(f"program:{_WORKER_PROCESS_NAME}", "command")
        arguments = shlex.split(" ".join(command.split()))
    except (configparser.Error, KeyError, ValueError) as error:
        raise ReadinessCheckError(
            "dedicated regulatory worker supervisor command is invalid"
        ) from error

    queue_values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument in {"-Q", "--queues"}:
            if index + 1 >= len(arguments):
                raise ReadinessCheckError(
                    "dedicated regulatory worker requires an exact queue set"
                )
            queue_values.append(arguments[index + 1])
        elif argument.startswith("--queues="):
            queue_values.append(argument.partition("=")[2])
        elif argument.startswith("-Q") and argument != "-Q":
            queue_values.append(argument[2:])
    if len(queue_values) != 1:
        raise ReadinessCheckError(
            "dedicated regulatory worker requires one exact queue set"
        )
    queues = {queue.strip() for queue in queue_values[0].split(",") if queue.strip()}
    if queues != _EXPECTED_REGULATORY_QUEUES:
        raise ReadinessCheckError(
            "dedicated regulatory worker requires the exact queue set "
            "regulatory_indexing,user_file_processing"
        )
    return queues


def _supervisor_worker_pid(status_line: str) -> int:
    match = re.search(r"\sRUNNING\s+pid ([1-9][0-9]*),", status_line)
    if match is None:
        raise ReadinessCheckError(
            "dedicated regulatory worker has no positive RUNNING PID"
        )
    return int(match.group(1))


def _live_regulatory_worker_queues(
    *, expected_worker_name: str, supervisor_pid: int
) -> set[str]:
    from onyx.background.celery.versioned_apps.regulatory_indexing import app

    inspector = app.control.inspect(timeout=5, destination=[expected_worker_name])
    queues = _validated_live_regulatory_worker_queues(
        inspector.active_queues(),
        expected_worker_name=expected_worker_name,
    )
    _validate_live_worker_pid(
        inspector.stats(),
        expected_worker_name=expected_worker_name,
        supervisor_pid=supervisor_pid,
    )
    return queues


def _validated_live_regulatory_worker_queues(
    responses: object, *, expected_worker_name: str
) -> set[str]:
    if not isinstance(responses, dict):
        raise ReadinessCheckError(
            "Celery active_queues returned no local worker response"
        )
    queue_records = cast(dict[object, object], responses).get(expected_worker_name)
    if not isinstance(queue_records, list):
        raise ReadinessCheckError(
            "Celery active_queues returned no local worker response"
        )
    queues: set[str] = set()
    for record in queue_records:
        if isinstance(record, dict):
            name = cast(dict[str, object], record).get("name")
            if isinstance(name, str):
                queues.add(name)
    if queues != _EXPECTED_REGULATORY_QUEUES:
        raise ReadinessCheckError(
            "live local dedicated worker must consume the exact queue set "
            "regulatory_indexing,user_file_processing"
        )
    return queues


def _validate_live_worker_pid(
    responses: object,
    *,
    expected_worker_name: str,
    supervisor_pid: int,
) -> None:
    if not isinstance(responses, dict):
        raise ReadinessCheckError("Celery stats returned no local worker response")
    stats = cast(dict[object, object], responses).get(expected_worker_name)
    if not isinstance(stats, dict):
        raise ReadinessCheckError("Celery stats returned no local worker response")
    celery_pid = cast(dict[str, object], stats).get("pid")
    if celery_pid != supervisor_pid:
        raise ReadinessCheckError(
            "local Celery worker PID does not match the Supervisor PID"
        )


def _validate_dense_vector_mapping(
    *,
    mapping: dict[str, object],
    expected: dict[str, object],
    effective_dimension: int,
) -> None:
    raw_actual_properties = mapping.get("properties")
    raw_expected_properties = expected.get("properties")
    if not isinstance(raw_actual_properties, dict) or not isinstance(
        raw_expected_properties, dict
    ):
        raise ReadinessCheckError("Elasticsearch mapping has no properties")
    actual_properties = cast(dict[str, object], raw_actual_properties)
    expected_properties = cast(dict[str, object], raw_expected_properties)
    for field_name in (CONTENT_VECTOR_FIELD_NAME, TITLE_VECTOR_FIELD_NAME):
        actual_field = actual_properties.get(field_name)
        expected_field = expected_properties.get(field_name)
        if not isinstance(actual_field, dict) or not isinstance(expected_field, dict):
            raise ReadinessCheckError(
                f"Elasticsearch mapping has no {field_name} contract"
            )
        typed_actual_field = cast(dict[str, object], actual_field)
        typed_expected_field = cast(dict[str, object], expected_field)
        expected_attributes = {
            "type": typed_expected_field.get("type"),
            "dims": effective_dimension,
            "element_type": typed_expected_field.get("element_type", "float"),
            "index": typed_expected_field.get("index"),
            "similarity": typed_expected_field.get("similarity"),
        }
        for attribute, expected_value in expected_attributes.items():
            actual_value = typed_actual_field.get(
                attribute, "float" if attribute == "element_type" else None
            )
            if actual_value != expected_value:
                raise ReadinessCheckError(
                    f"{field_name}.{attribute} is {actual_value!r}; "
                    f"expected {expected_value!r}"
                )


def _validated_secure_file(
    path: Path,
    *,
    expected_mode: int,
    expected_owner_uid: int,
    expected_owner_gid: int,
    maximum_size: int,
    label: str,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReadinessCheckError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReadinessCheckError(f"{label} must be a regular file, not a symlink")
    if (
        stat.S_IMODE(metadata.st_mode) != expected_mode
        or metadata.st_uid != expected_owner_uid
        or metadata.st_gid != expected_owner_gid
    ):
        raise ReadinessCheckError(
            f"{label} must be owned by runtime UID/GID "
            f"{expected_owner_uid}:{expected_owner_gid} with mode "
            f"{expected_mode:04o}"
        )
    if metadata.st_size <= 0 or metadata.st_size > maximum_size:
        raise ReadinessCheckError(f"{label} exceeds its maximum size or is empty")


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_capability_attestation(
    attestation_path: Path,
    evidence_path: Path,
    snapshot: ReadinessSnapshot,
    *,
    now: datetime.datetime | None = None,
    expected_owner_uid: int = _ATTESTATION_OWNER_UID,
    expected_owner_gid: int = _ATTESTATION_OWNER_GID,
) -> str:
    _validated_secure_file(
        attestation_path,
        expected_mode=0o600,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=expected_owner_gid,
        maximum_size=_ATTESTATION_MAX_BYTES,
        label="capability attestation",
    )
    _validated_secure_file(
        evidence_path,
        expected_mode=0o400,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=expected_owner_gid,
        maximum_size=_CAPABILITY_EVIDENCE_MAX_BYTES,
        label="capability evidence",
    )
    try:
        parsed: object = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReadinessCheckError(
            "capability attestation is unavailable or invalid"
        ) from error
    if not isinstance(parsed, dict) or parsed.get("schema_version") != 1:
        raise ReadinessCheckError("capability attestation schema is invalid")
    evidence = cast(dict[str, object], parsed)
    identity = evidence.get("identity")
    reference = evidence.get("evidence_reference")
    evidence_sha256 = evidence.get("evidence_sha256")
    if not isinstance(identity, str) or not identity.strip():
        raise ReadinessCheckError("capability attestation identity is invalid")
    if not isinstance(reference, str) or not reference.strip():
        raise ReadinessCheckError(
            "capability attestation evidence reference is invalid"
        )
    if (
        not isinstance(evidence_sha256, str)
        or _SHA256_PATTERN.fullmatch(evidence_sha256) is None
    ):
        raise ReadinessCheckError("capability attestation evidence digest is invalid")
    if not reference.endswith(f"#sha256={evidence_sha256}"):
        raise ReadinessCheckError(
            "capability attestation evidence reference has invalid digest binding"
        )
    try:
        actual_evidence_sha256 = _streaming_sha256(evidence_path)
    except OSError as error:
        raise ReadinessCheckError("capability evidence cannot be hashed") from error
    if not hmac.compare_digest(evidence_sha256, actual_evidence_sha256):
        raise ReadinessCheckError(
            "capability attestation does not match the actual evidence digest"
        )
    exact_scope = {
        "gcs_uri": snapshot.gcs_uri,
        "vertex_project": snapshot.vertex_project,
        "vertex_location": snapshot.vertex_location,
        "vertex_model": snapshot.vertex_model,
    }
    if any(evidence.get(key) != value for key, value in exact_scope.items()):
        raise ReadinessCheckError(
            "capability attestation does not match the exact active scope"
        )
    permissions = evidence.get("permissions")
    if not isinstance(permissions, list) or not all(
        isinstance(permission, str) for permission in permissions
    ):
        raise ReadinessCheckError("capability attestation permissions are invalid")
    if not REQUIRED_CAPABILITY_PERMISSIONS.issubset(set(permissions)):
        raise ReadinessCheckError(
            "capability attestation is missing required permissions"
        )
    reviewed_at_raw = evidence.get("reviewed_at")
    try:
        if not isinstance(reviewed_at_raw, str):
            raise ValueError
        reviewed_at = datetime.datetime.fromisoformat(reviewed_at_raw)
        if reviewed_at.tzinfo is None:
            raise ValueError
    except ValueError as error:
        raise ReadinessCheckError(
            "capability attestation reviewed_at is invalid"
        ) from error
    checked_at = now or datetime.datetime.now(datetime.timezone.utc)
    age = checked_at - reviewed_at
    if age < -datetime.timedelta(minutes=5):
        raise ReadinessCheckError("capability attestation is dated in the future")
    if age > _ATTESTATION_MAX_AGE:
        raise ReadinessCheckError("capability attestation is older than 24 hours")
    return identity.strip()


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("gs://")
    bucket, separator, prefix = without_scheme.partition("/")
    if not uri.startswith("gs://") or not bucket:
        raise ReadinessCheckError("configured GCS URI is invalid")
    return bucket, prefix.rstrip("/") if separator else ""


async def _probe_openrouter_embedding(
    *, api_key: str, model: str, effective_dimension: int
) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "input": [_PROBE_TEXT],
                "dimensions": effective_dimension,
            },
        )
        response.raise_for_status()
        payload: object = response.json()
    if not isinstance(payload, dict):
        raise ReadinessCheckError("OpenRouter returned an invalid response")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ReadinessCheckError("OpenRouter returned an invalid response")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ReadinessCheckError("OpenRouter returned an invalid response")
    return len(vector)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run read-only durable regulatory indexing readiness checks."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only status/name/detail JSON; secrets and vectors are never included.",
    )
    parser.add_argument(
        "--memory-headroom-reviewed",
        action="store_true",
        help=(
            "Attest that current/peak/max/events, pod limit, and node headroom were "
            "reviewed and archived. Omit to fail closed."
        ),
    )
    parser.add_argument(
        "--capability-attestation",
        type=Path,
        help=(
            "Owner-only JSON evidence (mode 0600) for required GCS and Vertex "
            "permissions, exact active scope, identity, and review time."
        ),
    )
    parser.add_argument(
        "--capability-evidence",
        type=Path,
        help=(
            "Owner-read-only archived IAM review artifact (mode 0400). Its bytes "
            "are hashed but never emitted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    report = run_readiness_checks(
        OnyxReadinessBackend(
            memory_headroom_reviewed=args.memory_headroom_reviewed,
            capability_attestation_path=args.capability_attestation,
            capability_evidence_path=args.capability_evidence,
        )
    )
    if args.json:
        print(
            json.dumps(
                {
                    "status": "READY"
                    if report.exit_code == EXIT_READY
                    else "NOT_READY",
                    "checks": [
                        {
                            "name": result.name,
                            "status": result.status,
                            "detail": result.detail,
                        }
                        for result in report.results
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        print(format_report(report))
    return report.exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(EXIT_USAGE_ERROR) from None
