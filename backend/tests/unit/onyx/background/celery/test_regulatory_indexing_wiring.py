from __future__ import annotations

import configparser
import importlib
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, call, patch

import pytest
import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from onyx.configs.constants import OnyxCeleryQueues

_BACKEND_ROOT = Path(__file__).resolve().parents[5]
_REPOSITORY_ROOT = _BACKEND_ROOT.parent
_COMPOSE_PATH = (
    _REPOSITORY_ROOT
    / "deployment"
    / "docker_compose"
    / "docker-compose.regulatory-prod-lite.yml"
)
_ENV_TEMPLATE_PATH = (
    _REPOSITORY_ROOT / "deployment" / "docker_compose" / "env.prod.template"
)
_WORKFLOW_PATH = (
    _REPOSITORY_ROOT
    / ".github"
    / "workflows"
    / "customs-regulations-backend-lite-codebuild.yaml"
)
_RUNBOOK_PATH = (
    _REPOSITORY_ROOT
    / "deployment"
    / "docker_compose"
    / "REGULATORY_PRODUCTION_RUNBOOK.md"
)
_HANDOFF_PATH = _REPOSITORY_ROOT / "deployment" / "DEVOPS_PRODUCTION_HANDOFF_TR.md"

_PRODUCTION_LITE_ENVIRONMENT = {
    "MARKDOWN_IMPORT_ENABLED",
    "MAX_ARCHIVE_COMPRESSION_RATIO",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_ARCHIVE_EXPANDED_BYTES",
    "REGULATORY_BATCH_INDEXING_ENABLED",
    "REGULATORY_INDEXING_EMBEDDING_REQUEST_SIZE",
    "REGULATORY_INDEXING_GCS_URI",
    "REGULATORY_INDEXING_LEASE_SECONDS",
    "REGULATORY_INDEXING_MAX_ATTEMPTS",
    "REGULATORY_INDEXING_POLL_SECONDS",
    "REGULATORY_INDEXING_RETRY_BASE_SECONDS",
    "REGULATORY_INDEXING_RETRY_MAX_SECONDS",
}


class _ComposeLoader(yaml.BaseLoader):
    pass


def _construct_compose_tag(loader: _ComposeLoader, node: yaml.Node) -> object:
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported Compose YAML node: {type(node).__name__}")


_ComposeLoader.add_constructor("!reset", _construct_compose_tag)
_ComposeLoader.add_constructor("!override", _construct_compose_tag)


def _load_compose() -> dict[str, object]:
    loaded = yaml.load(_COMPOSE_PATH.read_text(encoding="utf-8"), Loader=_ComposeLoader)
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def _load_workflow() -> dict[str, object]:
    loaded = yaml.load(
        _WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _workflow_step(workflow: dict[str, object], name: str) -> dict[str, str]:
    jobs = _mapping(workflow["jobs"])
    job = _mapping(jobs["build-and-deploy"])
    steps = job["steps"]
    assert isinstance(steps, list)
    for step in steps:
        if isinstance(step, dict):
            typed_step = cast(dict[str, object], step)
            if typed_step.get("name") == name:
                assert isinstance(typed_step.get("run"), str)
                return cast(dict[str, str], typed_step)
    raise AssertionError(f"Workflow step not found: {name}")


def _load_runbook_runtime_contract() -> dict[str, object]:
    runbook = _RUNBOOK_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- production-lite-runtime-contract:start -->\s*"
        r"```yaml\s*(.*?)\s*```\s*"
        r"<!-- production-lite-runtime-contract:end -->",
        runbook,
        re.S,
    )
    assert match is not None, "canonical production-lite runtime contract is missing"
    loaded = yaml.safe_load(match.group(1))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def test_production_lite_scheduler_contains_only_recovery_and_queue_monitoring() -> (
    None
):
    schedule_module = importlib.import_module(
        "onyx.background.celery.tasks.regulatory_indexing.beat_schedule"
    )
    templates = schedule_module.PRODUCTION_LITE_TASK_TEMPLATES

    assert {template["task"] for template in templates} == {
        "regulatory_indexing_recover_stale",
        "monitor_celery_queues",
    }
    assert {template["options"]["queue"] for template in templates} == {
        OnyxCeleryQueues.REGULATORY_INDEXING,
        OnyxCeleryQueues.MONITORING,
    }
    schedules = {template["task"]: template["schedule"] for template in templates}
    assert schedules["regulatory_indexing_recover_stale"] == timedelta(minutes=1)
    assert schedules["monitor_celery_queues"] == timedelta(seconds=10)
    assert all(template["options"]["expires"] > 0 for template in templates)
    assert not any("check-for-indexing" in template["name"] for template in templates)


def test_production_lite_scheduler_expands_every_task_with_each_tenant_id() -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )

    schedule = beat_app.RegulatoryIndexingScheduler.generate_schedule(
        ["public", "tenant-a"]
    )

    assert len(schedule) == 4
    assert {entry["kwargs"]["tenant_id"] for entry in schedule.values()} == {
        "public",
        "tenant-a",
    }
    assert all(set(entry["kwargs"]) == {"tenant_id"} for entry in schedule.values())
    assert (
        beat_app.celery_app.conf.task_default_base is beat_app.app_base.TenantAwareTask
    )


def test_production_lite_scheduler_rebuilds_schedule_across_restart(
    tmp_path: Path,
) -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )
    schedule_path = str(tmp_path / "regulatory-indexing-beat-schedule")
    scheduler = beat_app.RegulatoryIndexingScheduler(
        app=beat_app.celery_app,
        schedule_filename=schedule_path,
        lazy=False,
    )
    with patch.object(beat_app, "get_all_tenant_ids", return_value=["public"]):
        scheduler.update_schedule()
    expected_names = {
        "recover-stale-regulatory-indexing-public",
        "monitor-celery-queues-public",
    }
    assert set(scheduler.schedule) == expected_names
    scheduler.close()

    restarted = beat_app.RegulatoryIndexingScheduler(
        app=beat_app.celery_app,
        schedule_filename=schedule_path,
        lazy=False,
    )
    try:
        with patch.object(beat_app, "get_all_tenant_ids", return_value=["public"]):
            restarted.update_schedule()
        assert set(restarted.schedule) == expected_names
    finally:
        restarted.close()


def test_two_schedulers_publish_once_per_tenant_slot_and_fail_over_after_ttl(
    tmp_path: Path,
) -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )

    class ExpiringRedis:
        def __init__(self) -> None:
            self.now = 0.0
            self.values: dict[tuple[str, str], float] = {}
            self.claims: list[tuple[str, str, int]] = []

        def client(self, tenant_id: str) -> MagicMock:
            client = MagicMock()

            def set_value(
                key: str,
                _value: str,
                *,
                ex: int,
                nx: bool,
            ) -> bool | None:
                assert nx is True
                namespaced_key = (tenant_id, key)
                expiry = self.values.get(namespaced_key)
                if expiry is not None and expiry > self.now:
                    return None
                self.values[namespaced_key] = self.now + ex
                self.claims.append((tenant_id, key, ex))
                return True

            client.set.side_effect = set_value
            return client

    redis = ExpiringRedis()
    fixed_now = datetime(2026, 8, 19, 12, 0, 5, tzinfo=timezone.utc)
    schedulers = [
        beat_app.RegulatoryIndexingScheduler(
            app=beat_app.celery_app,
            schedule_filename=str(tmp_path / f"schedule-{index}"),
            lazy=False,
        )
        for index in range(2)
    ]
    try:
        for scheduler in schedulers:
            with patch.object(
                beat_app, "get_all_tenant_ids", return_value=["public", "tenant-a"]
            ):
                scheduler.update_schedule()
        entry_a = schedulers[0].schedule["monitor-celery-queues-public"]
        entry_b = schedulers[1].schedule["monitor-celery-queues-public"]
        tenant_entry = schedulers[1].schedule["monitor-celery-queues-tenant-a"]

        with (
            patch.object(beat_app.celery_app, "now", return_value=fixed_now),
            patch.object(
                beat_app,
                "get_redis_client",
                side_effect=lambda *, tenant_id: redis.client(tenant_id),
            ),
            patch.object(beat_app.PersistentScheduler, "apply_entry") as publish,
        ):
            schedulers[0].apply_entry(entry_a)
            schedulers[1].apply_entry(entry_b)
            schedulers[1].apply_entry(tenant_entry)

            assert publish.call_count == 2
            assert {claim[0] for claim in redis.claims} == {"public", "tenant-a"}
            public_claim = next(claim for claim in redis.claims if claim[0] == "public")
            assert "monitor-celery-queues-public" in public_claim[1]
            assert public_claim[1].endswith(":178714080")

            redis.now = public_claim[2] + 0.1
            schedulers[1].apply_entry(entry_b)
            assert publish.call_count == 3
    finally:
        for scheduler in schedulers:
            scheduler.close()


def test_scheduler_recovers_a_corrupt_pod_local_schedule(tmp_path: Path) -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )
    schedule_path = tmp_path / "corrupt-schedule"
    schedule_path.write_bytes(b"not-a-shelve-database")

    scheduler = beat_app.RegulatoryIndexingScheduler(
        app=beat_app.celery_app,
        schedule_filename=str(schedule_path),
        lazy=False,
    )
    try:
        with patch.object(beat_app, "get_all_tenant_ids", return_value=["public"]):
            scheduler.update_schedule()
        assert set(scheduler.schedule) == {
            "recover-stale-regulatory-indexing-public",
            "monitor-celery-queues-public",
        }
    finally:
        scheduler.close()


def test_production_lite_scheduler_waits_for_dependencies_before_readiness() -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )
    scheduler = MagicMock()
    sender = MagicMock(scheduler=scheduler)
    startup_events: list[str] = []
    scheduler.clear_probes.side_effect = lambda: startup_events.append("cleanup")
    scheduler.update_schedule.side_effect = lambda: startup_events.append("schedule")
    scheduler.mark_ready.side_effect = lambda: startup_events.append("ready")

    with (
        patch.object(beat_app.SqlEngine, "set_app_name") as set_app_name,
        patch.object(beat_app.SqlEngine, "init_engine") as init_engine,
        patch.object(
            beat_app.app_base,
            "wait_for_redis",
            side_effect=lambda *_args, **_kwargs: startup_events.append("redis"),
        ),
        patch.object(
            beat_app.app_base,
            "wait_for_db",
            side_effect=lambda *_args, **_kwargs: startup_events.append("database"),
        ),
    ):
        beat_app.on_beat_init(sender)

    set_app_name.assert_called_once_with("celery_beat_regulatory_indexing")
    init_engine.assert_called_once_with(pool_size=2, max_overflow=0)
    assert startup_events == ["cleanup", "redis", "database", "schedule", "ready"]
    scheduler.clear_probes.assert_called_once_with()
    scheduler.update_schedule.assert_called_once_with()
    scheduler.mark_ready.assert_called_once_with()


def test_scheduler_clears_probes_on_close_and_does_not_refresh_after_failure(
    tmp_path: Path,
) -> None:
    beat_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing_beat"
    )
    scheduler = beat_app.RegulatoryIndexingScheduler(
        app=beat_app.celery_app,
        schedule_filename=str(tmp_path / "schedule"),
        lazy=False,
    )
    scheduler._readiness_probe_path = tmp_path / "readiness"
    scheduler._liveness_probe_path = tmp_path / "liveness"
    scheduler.mark_ready()
    assert scheduler._readiness_probe_path.exists()
    assert scheduler._liveness_probe_path.exists()

    scheduler._last_reload = scheduler.app.now() - scheduler._reload_interval
    with (
        patch.object(beat_app.PersistentScheduler, "tick", return_value=1.0),
        patch.object(scheduler, "update_schedule", side_effect=RuntimeError("db down")),
        patch.object(scheduler, "mark_alive") as mark_alive,
    ):
        scheduler.tick()
    mark_alive.assert_not_called()

    scheduler._last_reload = scheduler.app.now() - scheduler._reload_interval
    with (
        patch.object(beat_app.PersistentScheduler, "tick", return_value=1.0),
        patch.object(scheduler, "update_schedule"),
        patch.object(scheduler, "mark_alive") as mark_alive,
    ):
        scheduler.tick()
    mark_alive.assert_called_once_with()

    scheduler.close()
    assert not scheduler._readiness_probe_path.exists()
    assert not scheduler._liveness_probe_path.exists()


def test_regulatory_indexing_worker_config_is_single_thread_and_prefetches_one() -> (
    None
):
    worker_config = importlib.import_module(
        "onyx.background.celery.configs.regulatory_indexing"
    )

    assert worker_config.worker_pool == "threads"
    assert worker_config.worker_concurrency == 1
    assert worker_config.worker_prefetch_multiplier == 1
    assert worker_config.task_acks_late is True


def test_regulatory_indexing_worker_discovers_upload_and_orchestration_tasks_without_heavy_dependencies() -> (
    None
):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_BACKEND_ROOT), env.get("PYTHONPATH")) if value
    )
    verification = """
import sys

blocked_import_roots = {
    "docling",
    "docling_core",
    "markitdown",
    "nvidia",
    "pypdfium2",
    "torch",
    "triton",
    "unstructured",
    "unstructured_client",
}


class BlockedImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked_import_roots:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockedImportFinder())

from onyx.background.celery.versioned_apps.regulatory_indexing import app
from onyx.configs.constants import OnyxCeleryTask

app.loader.import_default_modules()
app.finalize()

expected_tasks = {
    OnyxCeleryTask.CHECK_FOR_USER_FILE_DELETE,
    OnyxCeleryTask.CHECK_FOR_USER_FILE_PROCESSING,
    OnyxCeleryTask.CHECK_FOR_USER_FILE_PROJECT_SYNC,
    OnyxCeleryTask.DELETE_SINGLE_USER_FILE,
    OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
    OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC,
    OnyxCeleryTask.REGULATORY_INDEXING_RECOVER_STALE,
    OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
}
assert expected_tasks <= set(app.tasks), expected_tasks - set(app.tasks)
assert "onyx.background.celery.tasks.user_file_processing.tasks" in sys.modules
assert "onyx.background.celery.tasks.regulatory_indexing.tasks" in sys.modules
assert blocked_import_roots.isdisjoint(sys.modules), blocked_import_roots.intersection(
    sys.modules
)
assert not any(
    module_name == "model_server" or module_name.startswith("model_server.")
    for module_name in sys.modules
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", verification],
        cwd=_BACKEND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_regulatory_indexing_worker_initializes_database_redis_and_elasticsearch() -> (
    None
):
    worker_app = importlib.import_module(
        "onyx.background.celery.apps.regulatory_indexing"
    )
    sender = MagicMock(concurrency=1)
    startup_events: list[str] = []

    with (
        patch.object(worker_app.SqlEngine, "set_app_name") as set_app_name,
        patch.object(worker_app.SqlEngine, "init_engine") as init_engine,
        patch.object(
            worker_app.app_base,
            "wait_for_redis",
            side_effect=lambda *_args, **_kwargs: startup_events.append("redis"),
        ),
        patch.object(
            worker_app.app_base,
            "wait_for_db",
            side_effect=lambda *_args, **_kwargs: startup_events.append("database"),
        ),
        patch.object(
            worker_app.app_base,
            "wait_for_document_index_or_shutdown",
            side_effect=lambda: startup_events.append("elasticsearch"),
        ),
        patch.object(worker_app.app_base, "on_secondary_worker_init"),
        patch.object(worker_app, "MULTI_TENANT", False),
    ):
        worker_app.on_worker_init(sender)

    set_app_name.assert_called_once_with("celery_worker_regulatory_indexing")
    init_engine.assert_called_once_with(pool_size=1, max_overflow=2)
    assert startup_events == ["redis", "database", "elasticsearch"]


def test_lite_supervisor_runs_exact_regulatory_indexing_queues_and_forwards_log() -> (
    None
):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(_BACKEND_ROOT / "supervisord-lite.conf", encoding="utf-8")
    section = "program:celery_worker_regulatory_indexing"

    assert parser.has_section(section)
    command = " ".join(parser.get(section, "command").split())
    assert (
        "celery -A onyx.background.celery.versioned_apps.regulatory_indexing worker"
        in command
    )
    assert re.search(
        r"(?:^|\s)-Q user_file_processing,regulatory_indexing(?:\s|$)", command
    )
    assert "--hostname=regulatory_indexing@%%n" in command

    log_path = parser.get(section, "stdout_logfile")
    redirect_command = parser.get("program:log-redirect-handler", "command")
    assert log_path == "/var/log/onyx/celery_worker_regulatory_indexing.log"
    assert log_path in redirect_command


def test_lite_supervisor_runs_durable_regulatory_indexing_beat_and_forwards_log() -> (
    None
):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(_BACKEND_ROOT / "supervisord-lite.conf", encoding="utf-8")
    section = "program:celery_beat_regulatory_indexing"

    assert parser.has_section(section)
    command = " ".join(parser.get(section, "command").split())
    assert (
        "celery -A onyx.background.celery.versioned_apps.regulatory_indexing_beat beat"
        in command
    )
    assert "--schedule=/tmp/regulatory-indexing-beat-schedule" in command
    assert "/var/log/onyx/regulatory-indexing-beat-schedule" not in command
    assert parser.getboolean(section, "autorestart") is True

    log_path = parser.get(section, "stdout_logfile")
    redirect_command = parser.get("program:log-redirect-handler", "command")
    assert log_path == "/var/log/onyx/celery_beat_regulatory_indexing.log"
    assert log_path in redirect_command


def test_beat_probe_verifier_requires_current_pid_instance_and_fresh_liveness(
    tmp_path: Path,
) -> None:
    health = importlib.import_module(
        "onyx.background.celery.regulatory_indexing_beat_health"
    )
    readiness_path = tmp_path / "readiness"
    liveness_path = tmp_path / "liveness"
    marker = "4321:0123456789abcdef0123456789abcdef"
    status = "celery_beat_regulatory_indexing RUNNING pid 4321, uptime 0:00:30"
    readiness_path.write_text(marker, encoding="utf-8")
    liveness_path.write_text(marker, encoding="utf-8")

    health.validate_regulatory_indexing_beat(
        status,
        readiness_path=readiness_path,
        liveness_path=liveness_path,
        now=time.time(),
    )

    stale_mtime = time.time() - health.BEAT_LIVENESS_MAX_AGE_SECONDS - 1
    os.utime(liveness_path, (stale_mtime, stale_mtime))
    with pytest.raises(health.BeatProbeError, match="stale"):
        health.validate_regulatory_indexing_beat(
            status,
            readiness_path=readiness_path,
            liveness_path=liveness_path,
            now=time.time(),
        )

    wrong_pid_marker = "9999:0123456789abcdef0123456789abcdef"
    readiness_path.write_text(wrong_pid_marker, encoding="utf-8")
    liveness_path.write_text(wrong_pid_marker, encoding="utf-8")
    with pytest.raises(health.BeatProbeError, match="current supervisor PID"):
        health.validate_regulatory_indexing_beat(
            status,
            readiness_path=readiness_path,
            liveness_path=liveness_path,
            now=time.time(),
        )


def test_beat_probe_cli_executes_the_same_freshness_check(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness"
    liveness_path = tmp_path / "liveness"
    marker = "4321:0123456789abcdef0123456789abcdef"
    status = "celery_beat_regulatory_indexing RUNNING pid 4321, uptime 0:00:30"
    readiness_path.write_text(marker, encoding="utf-8")
    liveness_path.write_text(marker, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "onyx.background.celery.regulatory_indexing_beat_health",
        "--status-text",
        status,
        "--readiness-path",
        str(readiness_path),
        "--liveness-path",
        str(liveness_path),
    ]

    fresh = subprocess.run(
        command,
        cwd=_BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr

    stale_mtime = time.time() - 151
    os.utime(liveness_path, (stale_mtime, stale_mtime))
    stale = subprocess.run(
        command,
        cwd=_BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode != 0
    assert "stale" in stale.stderr


def test_production_lite_health_requires_every_worker_including_regulatory_indexing(
    tmp_path: Path,
) -> None:
    compose = _load_compose()
    services = _mapping(compose["services"])
    background = _mapping(services["background"])
    healthcheck = _mapping(background["healthcheck"])
    health_command = healthcheck["test"]
    assert isinstance(health_command, list)
    typed_health_command = cast(list[str], health_command)

    supervisorctl = tmp_path / "supervisorctl"
    supervisorctl.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$SUPERVISOR_ROWS\"\n",
        encoding="utf-8",
    )
    supervisorctl.chmod(0o755)
    expected_processes = [
        "celery_worker_regulatory_benchmark",
        "celery_worker_regulatory_indexing",
        "celery_worker_user_file_maintenance",
        "celery_worker_light",
        "celery_worker_monitoring",
        "celery_beat_regulatory_indexing",
        "log-redirect-handler",
    ]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_BACKEND_ROOT), env.get("PYTHONPATH")) if value
    )
    env["SUPERVISOR_ROWS"] = "\n".join(
        f"{process} RUNNING pid 1, uptime 0:00:30" for process in expected_processes
    )

    readiness_path = Path("/tmp/onyx_k8s_regulatoryindexingbeat_readiness.txt")
    liveness_path = Path("/tmp/onyx_k8s_regulatoryindexingbeat_liveness.txt")
    try:
        marker = "1:0123456789abcdef0123456789abcdef"
        readiness_path.write_text(marker, encoding="utf-8")
        liveness_path.write_text(marker, encoding="utf-8")
        successful = subprocess.run(
            [sys.executable, *typed_health_command[2:]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert successful.returncode == 0, successful.stdout + successful.stderr

        stale_mtime = time.time() - 151
        os.utime(liveness_path, (stale_mtime, stale_mtime))
        stale_probe = subprocess.run(
            [sys.executable, *typed_health_command[2:]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert stale_probe.returncode != 0

        liveness_path.write_text(
            "999:fedcba9876543210fedcba9876543210", encoding="utf-8"
        )
        wrong_process = subprocess.run(
            [sys.executable, *typed_health_command[2:]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert wrong_process.returncode != 0

        env["SUPERVISOR_ROWS"] = "\n".join(
            f"{process} RUNNING pid 1, uptime 0:00:30"
            for process in expected_processes
            if process != "celery_beat_regulatory_indexing"
        )
        missing_worker = subprocess.run(
            [sys.executable, *typed_health_command[2:]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert missing_worker.returncode != 0
    finally:
        readiness_path.unlink(missing_ok=True)
        liveness_path.unlink(missing_ok=True)


def test_production_lite_wires_the_complete_regulatory_environment_contract() -> None:
    template_variables = {
        match.group(1)
        for line in _ENV_TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", line))
    }
    assert _PRODUCTION_LITE_ENVIRONMENT <= template_variables

    compose = _load_compose()
    services = _mapping(compose["services"])
    api_server = _mapping(services["api_server"])
    background = _mapping(services["background"])
    api_environment = _mapping(api_server["environment"])
    background_environment = _mapping(background["environment"])

    assert {
        "MARKDOWN_IMPORT_ENABLED",
        "MAX_ARCHIVE_COMPRESSION_RATIO",
        "MAX_ARCHIVE_ENTRIES",
        "MAX_ARCHIVE_EXPANDED_BYTES",
        "REGULATORY_BATCH_INDEXING_ENABLED",
    } <= set(api_environment)
    assert {
        "MARKDOWN_IMPORT_ENABLED",
        "REGULATORY_BATCH_INDEXING_ENABLED",
        "REGULATORY_INDEXING_EMBEDDING_REQUEST_SIZE",
        "REGULATORY_INDEXING_GCS_URI",
        "REGULATORY_INDEXING_LEASE_SECONDS",
        "REGULATORY_INDEXING_MAX_ATTEMPTS",
        "REGULATORY_INDEXING_POLL_SECONDS",
        "REGULATORY_INDEXING_RETRY_BASE_SECONDS",
        "REGULATORY_INDEXING_RETRY_MAX_SECONDS",
    } <= set(background_environment)
    assert api_environment["MARKDOWN_IMPORT_ENABLED"] == (
        "${MARKDOWN_IMPORT_ENABLED:-true}"
    )
    assert background_environment["REGULATORY_BATCH_INDEXING_ENABLED"] == (
        "${REGULATORY_BATCH_INDEXING_ENABLED:-false}"
    )


def test_production_lite_background_has_no_model_server_dependency() -> None:
    compose = _load_compose()
    services = _mapping(compose["services"])
    background = _mapping(services["background"])
    depends_on = _mapping(background["depends_on"])

    assert set(depends_on) == {"indexing_model_server"}
    assert depends_on["indexing_model_server"] == "null"

    runtime_requirements = (_BACKEND_ROOT / "requirements" / "runtime.txt").read_text(
        encoding="utf-8"
    )
    forbidden_packages = (
        "docling",
        "markitdown",
        "nvidia-",
        "pypdfium2",
        "torch",
        "triton",
        "unstructured",
    )
    for package in forbidden_packages:
        assert not re.search(
            rf"^{re.escape(package)}[^=]*==", runtime_requirements, re.M
        )


def test_monitoring_collects_regulatory_indexing_queue_depth() -> None:
    monitoring_tasks = importlib.import_module(
        "onyx.background.celery.tasks.monitoring.tasks"
    )
    redis_client = MagicMock()

    with patch.object(
        monitoring_tasks,
        "celery_get_queue_length",
        return_value=0,
    ) as get_queue_length:
        metrics = monitoring_tasks._collect_queue_metrics(redis_client)

    assert call(OnyxCeleryQueues.REGULATORY_INDEXING, redis_client) in (
        get_queue_length.call_args_list
    )
    assert any(
        metric.name == "regulatory_indexing_queue_length"
        and metric.tags == {"queue": "regulatory_indexing_queue_length"}
        for metric in metrics
    )


def test_codebuild_diagnostics_and_readiness_use_the_exact_worker_name() -> None:
    workflow = _load_workflow()
    diagnostics = _workflow_step(workflow, "Capture pre-deploy worker diagnostics")
    deploy = _workflow_step(workflow, "Deploy api and background with Helm")
    diagnostics_script = diagnostics["run"]
    deploy_script = deploy["run"]

    assert "status celery_worker_regulatory_indexing" in diagnostics_script
    assert "verify_regulatory_indexing_worker" in deploy_script
    assert "status celery_worker_regulatory_indexing" in deploy_script
    assert "/tmp/onyx_k8s_regulatoryindexing_readiness.txt" in deploy_script
    assert "verify_regulatory_indexing_worker\n" in deploy_script
    assert "status celery_beat_regulatory_indexing" in diagnostics_script
    assert "verify_regulatory_indexing_beat" in deploy_script
    assert "status celery_beat_regulatory_indexing" in deploy_script
    assert (
        "python -m onyx.background.celery.regulatory_indexing_beat_health"
        in deploy_script
    )
    assert "capture_background_memory_evidence" in deploy_script

    for script in (diagnostics_script, deploy_script):
        syntax_check = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax_check.returncode == 0, syntax_check.stderr


def test_canonical_runbook_matches_executable_production_lite_topology() -> None:
    contract = _load_runbook_runtime_contract()
    runbook = _RUNBOOK_PATH.read_text(encoding="utf-8")
    workers = _mapping(contract["workers"])
    scheduler = _mapping(contract["scheduler"])
    operations = _mapping(contract["operations"])
    forbidden_queues = contract["forbidden_queues"]
    assert isinstance(forbidden_queues, list)

    assert contract["supervisor_process_count"] == 7
    assert set(workers) == {
        "celery_worker_regulatory_benchmark",
        "celery_worker_regulatory_indexing",
        "celery_worker_user_file_maintenance",
        "celery_worker_light",
        "celery_worker_monitoring",
    }
    assert workers["celery_worker_regulatory_indexing"] == [
        "user_file_processing",
        "regulatory_indexing",
    ]
    assert scheduler == {
        "name": "celery_beat_regulatory_indexing",
        "tasks": ["regulatory_indexing_recover_stale", "monitor_celery_queues"],
        "readiness_file": "/tmp/onyx_k8s_regulatoryindexingbeat_readiness.txt",
        "liveness_file": "/tmp/onyx_k8s_regulatoryindexingbeat_liveness.txt",
        "liveness_max_age_seconds": 150,
        "probe_marker": "pid:instance_uuid",
        "dispatch_dedup": "redis_tenant_entry_utc_slot_set_nx_ex",
    }
    assert "user_file_processing" not in forbidden_queues
    assert operations == {
        "feature_flag": "REGULATORY_BATCH_INDEXING_ENABLED",
        "default_enabled": False,
        "required_workspace": "REGULATORY_INDEXING_GCS_URI",
        "migration_before_enable": "alembic upgrade head",
        "restart_after_config_change": ["api_server", "background"],
    }
    for variable in _PRODUCTION_LITE_ENVIRONMENT:
        assert f"`{variable}`" in runbook

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(_BACKEND_ROOT / "supervisord-lite.conf", encoding="utf-8")
    supervisor_programs = {
        section.removeprefix("program:")
        for section in parser.sections()
        if section.startswith("program:")
    }
    assert supervisor_programs == {
        *workers,
        scheduler["name"],
        "log-redirect-handler",
    }
    for worker_name, documented_queues in workers.items():
        command = " ".join(parser.get(f"program:{worker_name}", "command").split())
        queue_match = re.search(r"(?:^|\s)-Q ([^\s]+)", command)
        assert queue_match is not None
        assert queue_match.group(1).split(",") == documented_queues

    handoff = _HANDOFF_PATH.read_text(encoding="utf-8")
    for process_name in supervisor_programs:
        assert f"`{process_name}`" in handoff
    assert "tam olarak beş worker, bir özel Beat ve bir log yönlendirici" in handoff
    assert "API ve background için birlikte `true`" in handoff
    assert "Yalnız background için `REGULATORY_INDEXING_GCS_URI`" in handoff
