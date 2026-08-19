from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def test_readiness_failure_is_redacted_and_blocks_dependent_checks() -> None:
    backend = _FakeBackend(fail_check="admin_snapshot")

    report = run_readiness_checks(backend)

    assert report.exit_code == EXIT_NOT_READY
    rendered = format_report(report)
    assert "admin_snapshot: FAIL (RuntimeError)" in rendered
    assert "gcs_access: BLOCKED" in rendered
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
    monkeypatch.setattr(
        readiness,
        "OnyxReadinessBackend",
        lambda *, memory_headroom_reviewed: (
            reviewed.append(memory_headroom_reviewed) or backend
        ),
    )
    monkeypatch.setattr(
        readiness.SqlEngine,
        "init_engine",
        lambda *, pool_size, max_overflow: initialized.append(
            (pool_size, max_overflow)
        ),
    )

    assert readiness.main(["--memory-headroom-reviewed"]) == EXIT_READY
    assert initialized == [(1, 0)]
    assert reviewed == [True]


def test_memory_headroom_requires_attestation_and_rejects_oom_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(readiness.ReadinessCheckError, match="operator must review"):
        readiness.OnyxReadinessBackend(
            memory_headroom_reviewed=False
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
            memory_headroom_reviewed=True
        ).check_memory_headroom()
