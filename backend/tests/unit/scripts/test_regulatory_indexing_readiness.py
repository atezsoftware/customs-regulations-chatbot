from __future__ import annotations

import datetime
import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from scripts import regulatory_indexing_readiness as readiness
from scripts.regulatory_indexing_readiness import (
    EXIT_NOT_READY,
    EXIT_READY,
    ReadinessSnapshot,
    format_report,
    run_readiness_checks,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _FakeBackend:
    fail_check: str | None = None

    def __post_init__(self) -> None:
        self.dimensions: list[int] = []

    def check_migration(self) -> str:
        return self._result("migration", "database matches application head")

    def check_worker_and_beat(self) -> str:
        return self._result("worker_and_beat", "worker and Beat probes are healthy")

    def check_memory_headroom(self) -> str:
        return self._result(
            "memory_headroom", "operator-reviewed cgroup evidence has no OOM event"
        )

    def check_capability_attestation(self, snapshot: ReadinessSnapshot) -> str:
        return self._result(
            "capability_attestation",
            f"reviewed capability scope for {snapshot.vertex_project}",
        )

    def load_snapshot(self) -> ReadinessSnapshot:
        self._result("admin_snapshot", "")
        return ReadinessSnapshot(
            search_settings_id=17,
            embedding_provider="openrouter",
            embedding_model="openai/text-embedding-3-large",
            effective_dimension=1536,
            index_name="active-index",
            vertex_model="gemini-test",
            vertex_project="project-test",
            vertex_location="europe-west4",
            gcs_uri="gs://test-bucket/regulatory",
        )

    def check_gcs_access(self, snapshot: ReadinessSnapshot) -> str:
        return self._result("gcs_access", f"read access to {snapshot.gcs_uri}")

    def check_vertex_access(self, snapshot: ReadinessSnapshot) -> str:
        return self._result(
            "vertex_access", f"model and batch list access for {snapshot.vertex_model}"
        )

    def probe_embedding_dimension(self, snapshot: ReadinessSnapshot) -> int:
        self._result("embedding_probe", "")
        self.dimensions.append(snapshot.effective_dimension)
        return snapshot.effective_dimension

    def check_elasticsearch_mapping(self, snapshot: ReadinessSnapshot) -> str:
        self._result("elasticsearch_mapping", "")
        self.dimensions.append(snapshot.effective_dimension)
        return f"{snapshot.index_name} mapping is compatible"

    def _result(self, check: str, result: str) -> str:
        if self.fail_check == check:
            raise RuntimeError(
                "request failed Authorization: Bearer top-secret-token "
                '{"private_key":"private-secret"}'
            )
        return result


def test_readiness_uses_active_dynamic_dimension_and_reports_safe_metadata() -> None:
    backend = _FakeBackend()

    report = run_readiness_checks(backend)

    assert report.exit_code == EXIT_READY
    assert backend.dimensions == [1536, 1536]
    rendered = format_report(report)
    assert "effective_dimension=1536" in rendered
    assert "openai/text-embedding-3-large" in rendered
    assert "active-index" in rendered
    assert "vector" not in rendered.lower()
    assert "memory_headroom: PASS" in rendered
    assert "capability_attestation: PASS" in rendered


def test_readiness_failure_is_redacted_and_blocks_dependent_checks() -> None:
    backend = _FakeBackend(fail_check="admin_snapshot")

    report = run_readiness_checks(backend)

    assert report.exit_code == EXIT_NOT_READY
    rendered = format_report(report)
    assert "admin_snapshot: FAIL (RuntimeError)" in rendered
    assert "gcs_access: BLOCKED" in rendered
    assert "capability_attestation: BLOCKED" in rendered
    assert "vertex_access: BLOCKED" in rendered
    assert "embedding_probe: BLOCKED" in rendered
    assert "elasticsearch_mapping: BLOCKED" in rendered
    assert "top-secret-token" not in rendered
    assert "private-secret" not in rendered
    assert backend.dimensions == []


def test_independent_failure_does_not_skip_other_read_only_checks() -> None:
    backend = _FakeBackend(fail_check="gcs_access")

    report = run_readiness_checks(backend)

    assert report.exit_code == EXIT_NOT_READY
    assert backend.dimensions == [1536, 1536]
    statuses = {result.name: result.status for result in report.results}
    assert statuses["gcs_access"] == "FAIL"
    assert statuses["vertex_access"] == "PASS"
    assert statuses["embedding_probe"] == "PASS"
    assert statuses["elasticsearch_mapping"] == "PASS"


def test_runtime_lite_image_contains_the_readiness_command() -> None:
    dockerfile = (_BACKEND_ROOT / "Dockerfile.runtime-lite").read_text(encoding="utf-8")

    assert (
        "COPY --chown=onyx:onyx ./scripts/regulatory_indexing_readiness.py "
        "/app/scripts/regulatory_indexing_readiness.py"
    ) in dockerfile


def test_cli_initializes_a_bounded_read_only_database_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    initialized: list[tuple[int, int]] = []
    reviewed: list[bool] = []

    def fake_backend(
        *,
        memory_headroom_reviewed: bool,
        capability_attestation_path: Path | None,
        capability_evidence_path: Path | None,
    ) -> _FakeBackend:
        reviewed.append(memory_headroom_reviewed)
        assert capability_attestation_path == Path("operator-evidence.json")
        assert capability_evidence_path == Path("archived-evidence.json")
        return backend

    monkeypatch.setattr(
        readiness,
        "OnyxReadinessBackend",
        fake_backend,
    )
    monkeypatch.setattr(
        readiness.SqlEngine,
        "init_engine",
        lambda *, pool_size, max_overflow: initialized.append(
            (pool_size, max_overflow)
        ),
    )

    assert (
        readiness.main(
            [
                "--memory-headroom-reviewed",
                "--capability-attestation",
                "operator-evidence.json",
                "--capability-evidence",
                "archived-evidence.json",
            ]
        )
        == EXIT_READY
    )
    assert initialized == [(1, 0)]
    assert reviewed == [True]


def test_capability_file_only_cli_uses_secure_reader_without_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.write_bytes(b"{}\n")
    attestation_path.chmod(0o600)
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b'{"approved":true,"secret-marker":"never-print"}\n')
    evidence_path.chmod(0o400)
    monkeypatch.setattr(readiness, "_ATTESTATION_OWNER_UID", os.geteuid())
    monkeypatch.setattr(readiness, "_ATTESTATION_OWNER_GID", os.getegid())
    monkeypatch.setattr(
        readiness.SqlEngine,
        "init_engine",
        lambda **_kwargs: pytest.fail("file-only validation initialized the database"),
    )

    assert (
        readiness.main(
            [
                "--validate-capability-files-only",
                "--capability-attestation",
                str(attestation_path),
                "--capability-evidence",
                str(evidence_path),
            ]
        )
        == EXIT_READY
    )
    output = capsys.readouterr()
    assert "secret-marker" not in output.out
    assert "secret-marker" not in output.err


def test_capability_snapshot_cli_revalidates_exact_evidence_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_bytes = b'{"approved":true,"marker":"snapshot-never-print"}\n'
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o600)
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_reference": f"archive://TASK-8#sha256={digest}",
                "evidence_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    attestation_path.chmod(0o600)
    monkeypatch.setattr(
        readiness.SqlEngine,
        "init_engine",
        lambda **_kwargs: pytest.fail("snapshot validation initialized the database"),
    )
    arguments = [
        "--validate-capability-snapshots-only",
        "--capability-attestation",
        str(attestation_path),
        "--capability-evidence",
        str(evidence_path),
    ]

    assert readiness.main(arguments) == EXIT_READY
    evidence_path.write_bytes(evidence_bytes + b"tampered")
    assert readiness.main(arguments) == EXIT_NOT_READY

    output = capsys.readouterr()
    assert "snapshot-never-print" not in output.out
    assert "snapshot-never-print" not in output.err


def test_memory_headroom_requires_attestation_and_rejects_oom_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(readiness.ReadinessCheckError, match="operator must review"):
        readiness.OnyxReadinessBackend(
            memory_headroom_reviewed=False,
            capability_attestation_path=None,
            capability_evidence_path=None,
        ).check_memory_headroom()

    (tmp_path / "memory.current").write_text("100\n", encoding="utf-8")
    (tmp_path / "memory.peak").write_text("200\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("1000\n", encoding="utf-8")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 1\noom_kill 0\n", encoding="utf-8"
    )
    monkeypatch.setattr(readiness, "_CGROUP_ROOT", tmp_path)

    with pytest.raises(readiness.ReadinessCheckError, match="oom=1"):
        readiness.OnyxReadinessBackend(
            memory_headroom_reviewed=True,
            capability_attestation_path=None,
            capability_evidence_path=None,
        ).check_memory_headroom()


def test_worker_queue_parser_rejects_regulatory_substrings_outside_queue_arg() -> None:
    false_positive = """
[program:celery_worker_regulatory_indexing]
command=celery -A onyx.background.celery.versioned_apps.regulatory_indexing worker
    --hostname=regulatory_indexing@%%n
    -Q user_file_processing
stdout_logfile=/var/log/onyx/celery_worker_regulatory_indexing.log

[program:celery_beat_regulatory_indexing]
command=celery beat
"""

    with pytest.raises(readiness.ReadinessCheckError, match="exact queue set"):
        readiness._configured_regulatory_worker_queues(false_positive)

    configured = false_positive.replace(
        "-Q user_file_processing", "-Q user_file_processing,regulatory_indexing"
    )
    assert readiness._configured_regulatory_worker_queues(configured) == {
        "user_file_processing",
        "regulatory_indexing",
    }


def test_live_worker_queue_validation_requires_exact_regulatory_queue_set() -> None:
    with pytest.raises(readiness.ReadinessCheckError, match="exact queue set"):
        readiness._validated_live_regulatory_worker_queues(
            {
                "regulatory_indexing@local-node": [
                    {"name": "user_file_processing"},
                ],
                "regulatory_indexing@remote-node": [
                    {"name": "regulatory_indexing"},
                    {"name": "user_file_processing"},
                ],
            },
            expected_worker_name="regulatory_indexing@local-node",
        )

    assert readiness._validated_live_regulatory_worker_queues(
        {
            "regulatory_indexing@local-node": [
                {"name": "regulatory_indexing"},
                {"name": "user_file_processing"},
            ],
            "regulatory_indexing@remote-node": [{"name": "other"}],
        },
        expected_worker_name="regulatory_indexing@local-node",
    ) == {"regulatory_indexing", "user_file_processing"}

    with pytest.raises(readiness.ReadinessCheckError, match="local worker response"):
        readiness._validated_live_regulatory_worker_queues(
            {
                "regulatory_indexing@remote-node": [
                    {"name": "regulatory_indexing"},
                    {"name": "user_file_processing"},
                ]
            },
            expected_worker_name="regulatory_indexing@local-node",
        )


def test_local_worker_pid_must_match_supervisor_status() -> None:
    status_line = "celery_worker_regulatory_indexing RUNNING pid 4242, uptime 0:01:00"
    assert readiness._supervisor_worker_pid(status_line) == 4242
    readiness._validate_live_worker_pid(
        {"regulatory_indexing@local-node": {"pid": 4242}},
        expected_worker_name="regulatory_indexing@local-node",
        supervisor_pid=4242,
    )
    with pytest.raises(readiness.ReadinessCheckError, match="Supervisor PID"):
        readiness._validate_live_worker_pid(
            {"regulatory_indexing@local-node": {"pid": 5252}},
            expected_worker_name="regulatory_indexing@local-node",
            supervisor_pid=4242,
        )


@pytest.mark.parametrize(
    "status_line",
    [
        "celery_worker_regulatory_indexing RUNNING uptime 0:01:00",
        "celery_worker_regulatory_indexing RUNNING pid 0, uptime 0:01:00",
        "celery_worker_regulatory_indexing RUNNING pid nope, uptime 0:01:00",
        "celery_worker_regulatory_indexing STOPPED pid 4242, uptime 0:01:00",
    ],
)
def test_supervisor_worker_pid_fails_closed_when_not_positive_running_pid(
    status_line: str,
) -> None:
    with pytest.raises(readiness.ReadinessCheckError, match="positive RUNNING PID"):
        readiness._supervisor_worker_pid(status_line)


def test_live_worker_inspection_targets_only_the_local_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_worker_name = "regulatory_indexing@local-node"
    inspector = MagicMock()
    inspector.active_queues.return_value = {
        expected_worker_name: [
            {"name": "regulatory_indexing"},
            {"name": "user_file_processing"},
        ]
    }
    inspector.stats.return_value = {expected_worker_name: {"pid": 4242}}

    from onyx.background.celery.versioned_apps.regulatory_indexing import app

    inspect_mock = MagicMock(return_value=inspector)
    monkeypatch.setattr(app.control, "inspect", inspect_mock)

    assert readiness._live_regulatory_worker_queues(
        expected_worker_name=expected_worker_name,
        supervisor_pid=4242,
    ) == {"regulatory_indexing", "user_file_processing"}
    inspect_mock.assert_called_once_with(
        timeout=5,
        destination=[expected_worker_name],
    )
    inspector.stats.assert_called_once_with()


@pytest.mark.parametrize(
    "probe_name", ["probe_gcs_read_access", "probe_vertex_read_access"]
)
def test_observational_probe_identity_must_match_attestation(
    probe_name: str,
) -> None:
    snapshot = _FakeBackend().load_snapshot()
    backend = readiness.OnyxReadinessBackend(
        memory_headroom_reviewed=True,
        capability_attestation_path=None,
        capability_evidence_path=None,
    )
    backend._attested_identity = "expected@example.iam.gserviceaccount.com"
    gateway = SimpleNamespace(
        probe_gcs_read_access=lambda: SimpleNamespace(
            credential_identity="actual@example.iam.gserviceaccount.com"
        ),
        probe_vertex_read_access=lambda: SimpleNamespace(
            credential_identity="actual@example.iam.gserviceaccount.com"
        ),
    )
    backend._vertex_gateway = cast(readiness.GoogleVertexBatchGateway, gateway)

    check = (
        backend.check_gcs_access
        if probe_name == "probe_gcs_read_access"
        else backend.check_vertex_access
    )
    with pytest.raises(readiness.ReadinessCheckError, match="identity does not match"):
        check(snapshot)


@pytest.mark.parametrize(
    ("field_name", "attribute", "invalid_value"),
    [
        ("content_vector", "dims", 768),
        ("title_vector", "dims", 768),
        ("content_vector", "element_type", "byte"),
        ("title_vector", "index", False),
        ("content_vector", "similarity", "dot_product"),
    ],
)
def test_elasticsearch_mapping_rejects_wrong_dense_vector_contract(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    attribute: str,
    invalid_value: str | int,
) -> None:
    closed: list[bool] = []
    mapping = {
        "properties": {
            field: {
                "type": "dense_vector",
                "dims": 1536,
                "element_type": "float",
                "index": True,
                "similarity": "cosine",
            }
            for field in ("content_vector", "title_vector")
        }
    }
    mapping["properties"][field_name][attribute] = invalid_value
    fake_client = SimpleNamespace(
        validate_index=lambda _expected: True,
        get_index_mapping=lambda: mapping,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        readiness, "ElasticsearchIndexClient", lambda _index_name: fake_client
    )
    backend = readiness.OnyxReadinessBackend(
        memory_headroom_reviewed=True,
        capability_attestation_path=None,
        capability_evidence_path=None,
    )

    with pytest.raises(
        readiness.ReadinessCheckError,
        match=rf"{field_name}\.{attribute}",
    ):
        backend.check_elasticsearch_mapping(_FakeBackend().load_snapshot())

    assert closed == [True]


def test_capability_attestation_requires_exact_scope_permissions_and_freshness(
    tmp_path: Path,
) -> None:
    snapshot = _FakeBackend().load_snapshot()
    evidence_path = tmp_path / "archived-iam-evidence.json"
    evidence_bytes = b'{"approved":true,"change":"TASK-8"}\n'
    evidence_path.write_bytes(evidence_bytes)
    os.chmod(evidence_path, 0o400)
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    attestation_path = tmp_path / "capability-attestation.json"
    attestation = {
        "schema_version": 1,
        "reviewed_at": "2026-08-20T08:00:00+00:00",
        "identity": "service-account@example.iam.gserviceaccount.com",
        "evidence_reference": f"archive://TASK-8#sha256={digest}",
        "evidence_sha256": digest,
        "gcs_uri": snapshot.gcs_uri,
        "vertex_project": snapshot.vertex_project,
        "vertex_location": snapshot.vertex_location,
        "vertex_model": snapshot.vertex_model,
        "permissions": sorted(readiness.REQUIRED_CAPABILITY_PERMISSIONS),
    }
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    os.chmod(attestation_path, 0o600)

    identity = readiness._validate_capability_attestation(
        attestation_path,
        evidence_path,
        snapshot,
        now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
        expected_owner_uid=os.geteuid(),
        expected_owner_gid=os.getegid(),
    )
    assert identity == attestation["identity"]

    attestation["permissions"].remove("storage.objects.delete")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(readiness.ReadinessCheckError, match="required permissions"):
        readiness._validate_capability_attestation(
            attestation_path,
            evidence_path,
            snapshot,
            now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
        )

    attestation["permissions"] = sorted(readiness.REQUIRED_CAPABILITY_PERMISSIONS)
    attestation["reviewed_at"] = "2026-08-18T08:00:00+00:00"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(readiness.ReadinessCheckError, match="older than"):
        readiness._validate_capability_attestation(
            attestation_path,
            evidence_path,
            snapshot,
            now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
        )

    attestation["reviewed_at"] = "2026-08-20T08:00:00+00:00"
    attestation["evidence_sha256"] = "c" * 64
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(readiness.ReadinessCheckError, match="digest binding"):
        readiness._validate_capability_attestation(
            attestation_path,
            evidence_path,
            snapshot,
            now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
        )


def test_capability_attestation_hashes_separate_secure_evidence_file(
    tmp_path: Path,
) -> None:
    snapshot = _FakeBackend().load_snapshot()
    evidence_path = tmp_path / "archived-iam-evidence.json"
    evidence_bytes = b'{"approved":true,"change":"TASK-8"}\n'
    evidence_path.write_bytes(evidence_bytes)
    os.chmod(evidence_path, 0o400)
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewed_at": "2026-08-20T08:00:00+00:00",
                "identity": "service-account@example.iam.gserviceaccount.com",
                "evidence_reference": f"archive://TASK-8#sha256={digest}",
                "evidence_sha256": digest,
                "gcs_uri": snapshot.gcs_uri,
                "vertex_project": snapshot.vertex_project,
                "vertex_location": snapshot.vertex_location,
                "vertex_model": snapshot.vertex_model,
                "permissions": sorted(readiness.REQUIRED_CAPABILITY_PERMISSIONS),
            }
        ),
        encoding="utf-8",
    )
    os.chmod(attestation_path, 0o600)

    identity = readiness._validate_capability_attestation(
        attestation_path,
        evidence_path,
        snapshot,
        now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
        expected_owner_uid=os.geteuid(),
        expected_owner_gid=os.getegid(),
    )
    assert identity == "service-account@example.iam.gserviceaccount.com"

    os.chmod(evidence_path, 0o600)
    evidence_path.write_bytes(evidence_bytes + b"tampered")
    os.chmod(evidence_path, 0o400)
    with pytest.raises(readiness.ReadinessCheckError, match="actual evidence digest"):
        readiness._validate_capability_attestation(
            attestation_path,
            evidence_path,
            snapshot,
            now=datetime.datetime(2026, 8, 20, 9, 0, tzinfo=datetime.timezone.utc),
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
        )


def test_capability_files_reject_symlinks_and_oversize_evidence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"12345")
    os.chmod(target, 0o400)
    symlink = tmp_path / "evidence-link"
    symlink.symlink_to(target)

    with pytest.raises(readiness.ReadinessCheckError, match="regular file"):
        readiness._read_secure_file(
            symlink,
            expected_mode=0o400,
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
            maximum_size=10,
            label="capability evidence",
        )
    with pytest.raises(readiness.ReadinessCheckError, match="maximum size"):
        readiness._read_secure_file(
            target,
            expected_mode=0o400,
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
            maximum_size=4,
            label="capability evidence",
        )


def test_secure_file_reader_owns_descriptor_across_path_swap(tmp_path: Path) -> None:
    trusted_bytes = b"trusted archived evidence\n"
    trusted_path = tmp_path / "capability-evidence.json"
    trusted_path.write_bytes(trusted_bytes)
    trusted_path.chmod(0o400)
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_bytes(b"attacker-controlled replacement\n")
    replacement_path.chmod(0o400)

    opened_descriptor: int | None = None

    def open_then_swap(path: Path, flags: int) -> int:
        nonlocal opened_descriptor
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_CLOEXEC
        descriptor = os.open(path, flags)
        opened_descriptor = descriptor
        os.replace(replacement_path, trusted_path)
        return descriptor

    contents = readiness._read_secure_file(
        trusted_path,
        expected_mode=0o400,
        expected_owner_uid=os.geteuid(),
        expected_owner_gid=os.getegid(),
        maximum_size=1024,
        label="capability evidence",
        open_file=open_then_swap,
    )

    assert contents == trusted_bytes
    assert trusted_path.read_bytes() == b"attacker-controlled replacement\n"
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)


@pytest.mark.parametrize("flag_name", ["O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK"])
def test_secure_file_reader_fails_closed_without_safe_open_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flag_name: str,
) -> None:
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b"evidence")
    evidence_path.chmod(0o400)
    monkeypatch.setattr(readiness.os, flag_name, 0)

    with pytest.raises(readiness.ReadinessCheckError, match="secure-open semantics"):
        readiness._read_secure_file(
            evidence_path,
            expected_mode=0o400,
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
            maximum_size=1024,
            label="capability evidence",
        )


def test_secure_file_reader_rejects_fifo_without_waiting_for_writer(
    tmp_path: Path,
) -> None:
    fifo_path = tmp_path / "capability-evidence.fifo"
    os.mkfifo(fifo_path, 0o400)
    result_reader, result_writer = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(result_reader)
        try:
            readiness._read_secure_file(
                fifo_path,
                expected_mode=0o400,
                expected_owner_uid=os.geteuid(),
                expected_owner_gid=os.getegid(),
                maximum_size=1024,
                label="capability evidence",
            )
        except readiness.ReadinessCheckError:
            os.write(result_writer, b"rejected")
        else:
            os.write(result_writer, b"accepted")
        finally:
            os.close(result_writer)
        os._exit(0)

    os.close(result_writer)
    deadline = time.monotonic() + 1
    child_status: int | None = None
    while time.monotonic() < deadline:
        waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if waited_pid == child_pid:
            child_status = status
            break
        time.sleep(0.01)

    if child_status is None:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        os.close(result_reader)
        pytest.fail("secure reader blocked while opening a FIFO without a writer")

    result = os.read(result_reader, 32)
    os.close(result_reader)
    assert os.waitstatus_to_exitcode(child_status) == 0
    assert result == b"rejected"


def test_secure_file_reader_rejects_growth_beyond_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b"1234")
    evidence_path.chmod(0o400)
    original_read = os.read
    grew = False

    def grow_before_first_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        if not grew:
            grew = True
            evidence_path.chmod(0o600)
            with evidence_path.open("ab") as evidence_file:
                evidence_file.write(b"5")
            evidence_path.chmod(0o400)
        return original_read(descriptor, size)

    monkeypatch.setattr(readiness.os, "read", grow_before_first_read)

    with pytest.raises(readiness.ReadinessCheckError, match="maximum size"):
        readiness._read_secure_file(
            evidence_path,
            expected_mode=0o400,
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
            maximum_size=4,
            label="capability evidence",
        )


def test_secure_file_reader_closes_descriptor_on_validation_failure(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b"evidence")
    evidence_path.chmod(0o400)
    opened_descriptor: int | None = None

    def record_open(path: Path, flags: int) -> int:
        nonlocal opened_descriptor
        opened_descriptor = os.open(path, flags)
        return opened_descriptor

    with pytest.raises(readiness.ReadinessCheckError, match="mode 0600"):
        readiness._read_secure_file(
            evidence_path,
            expected_mode=0o600,
            expected_owner_uid=os.geteuid(),
            expected_owner_gid=os.getegid(),
            maximum_size=1024,
            label="capability evidence",
            open_file=record_open,
        )

    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)
