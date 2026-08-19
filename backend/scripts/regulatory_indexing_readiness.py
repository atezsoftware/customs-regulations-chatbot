"""Read-only production readiness checks for durable regulatory indexing.

This command intentionally has no write-capable database, object-store, or
Elasticsearch operations. It performs one constant-text embedding request but
never prints or persists the returned vector.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from onyx.document_index.elasticsearch.schema import DocumentSchema
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

    def __init__(self, *, memory_headroom_reviewed: bool) -> None:
        self._memory_headroom_reviewed = memory_headroom_reviewed
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
        worker_section = f"[program:{_WORKER_PROCESS_NAME}]"
        beat_section = f"[program:{BEAT_PROCESS_NAME}]"
        if worker_section not in config_text or beat_section not in config_text:
            raise ReadinessCheckError(
                "dedicated regulatory worker or Beat supervisor config is missing"
            )
        worker_block = config_text.split(worker_section, 1)[1].split("[", 1)[0]
        if "regulatory_indexing" not in worker_block:
            raise ReadinessCheckError(
                "dedicated regulatory worker is not bound to its queue"
            )

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
        return "dedicated worker RUNNING and Beat readiness/liveness probes valid"

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

    def check_gcs_access(self, snapshot: ReadinessSnapshot) -> str:
        gateway = self._gateway()
        credentials = gateway._credentials()
        storage_client = gateway._storage_client(credentials)
        bucket_name, prefix = _parse_gcs_uri(snapshot.gcs_uri)
        blobs = storage_client.list_blobs(
            bucket_name,
            prefix=prefix,
            max_results=1,
            timeout=30,
        )
        next(iter(blobs), None)
        return f"read/list access confirmed for gs://{bucket_name}"

    def check_vertex_access(self, snapshot: ReadinessSnapshot) -> str:
        from google.genai import types as genai_types

        gateway = self._gateway()
        credentials = gateway._credentials()
        with gateway._managed_genai_client(credentials) as client:
            client.models.get(model=snapshot.vertex_model)
            pager = client.batches.list(
                config=genai_types.ListBatchJobsConfig(page_size=1)
            )
            pager.page
        return "Vertex model get and batch list access confirmed"

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
        finally:
            client.close()
        return (
            f"{snapshot.index_name} mapping matches "
            f"effective_dimension={snapshot.effective_dimension}"
        )


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    report = run_readiness_checks(
        OnyxReadinessBackend(
            memory_headroom_reviewed=args.memory_headroom_reviewed,
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
