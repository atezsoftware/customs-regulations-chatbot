from __future__ import annotations

import configparser
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.regulatory_labeling.tasks import enqueue_labeling_step
from onyx.configs.constants import OnyxCeleryTask

_BACKEND_ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize("countdown", [0, 30, 2000])
def test_label_deliveries_preserve_tenant_and_expiry_on_independent_queue(
    countdown: int,
) -> None:
    app = MagicMock()
    run_id = uuid4()

    enqueue_labeling_step(
        app,
        run_id=run_id,
        expected_generation=7,
        tenant_id="tenant-a",
        countdown_seconds=countdown,
    )

    args, options = app.send_task.call_args
    assert args == (OnyxCeleryTask.REGULATORY_LABELING_RUN_STEP,)
    assert options["queue"] == "regulatory_labeling"
    assert options["kwargs"] == {
        "run_id": str(run_id),
        "expected_generation": 7,
        "tenant_id": "tenant-a",
    }
    assert countdown < options["expires"] <= 86400
    assert options.get("countdown", 0) == countdown


@pytest.mark.parametrize(
    "module_name,attribute",
    [
        ("onyx.background.celery.tasks.beat_schedule", "beat_task_templates"),
        (
            "onyx.background.celery.tasks.regulatory_indexing.beat_schedule",
            "PRODUCTION_LITE_TASK_TEMPLATES",
        ),
    ],
)
def test_recovery_cannot_occupy_an_indexing_worker_slot(
    module_name: str, attribute: str
) -> None:
    templates = getattr(importlib.import_module(module_name), attribute)
    recovery = next(
        template
        for template in templates
        if template["task"] == OnyxCeleryTask.REGULATORY_LABELING_RECOVER_STALE
    )
    assert recovery["options"]["queue"] == "regulatory_labeling"
    assert recovery["options"]["expires"] > 0


@pytest.mark.parametrize("filename", ["supervisord.conf", "supervisord-lite.conf"])
def test_each_image_runs_one_label_consumer_separate_from_indexing(
    filename: str,
) -> None:
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.read(_BACKEND_ROOT / filename)
    consumers: list[str] = []
    for section in config.sections():
        if not section.startswith("program:"):
            continue
        args = shlex.split(config.get(section, "command"))
        if "-Q" in args:
            queues = set(args[args.index("-Q") + 1].split(","))
            if "regulatory_labeling" in queues:
                consumers.append(section)
                assert queues == {"regulatory_labeling"}
                assert args[args.index("-A") + 1] == (
                    "onyx.background.celery.versioned_apps.regulatory_labeling"
                )
                assert "--hostname=regulatory_labeling@%%n" in args
    assert consumers == ["program:celery_worker_regulatory_labeling"]
    log_path = config.get(consumers[0], "stdout_logfile")
    assert log_path in config.get("program:log-redirect-handler", "command")


def test_label_worker_loads_with_bounded_capacity_and_without_parser_stack() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_BACKEND_ROOT), env.get("PYTHONPATH")) if value
    )
    verification = """
import sys
blocked = {"docling", "docling_core", "torch", "triton", "markitdown", "unstructured", "pypdfium2"}
class BlockedFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockedFinder())
from onyx.background.celery.versioned_apps.regulatory_labeling import app
from onyx.background.celery.apps.app_base import TenantAwareTask
app.loader.import_default_modules()
app.finalize()
assert app.conf.worker_concurrency == 1
assert app.conf.worker_prefetch_multiplier == 1
assert app.conf.worker_pool == "threads"
assert app.conf.task_acks_late is True
assert issubclass(app.Task, TenantAwareTask)
assert {"regulatory_labeling_run_step", "regulatory_labeling_recover_stale"} <= set(app.tasks)
assert blocked.isdisjoint(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", verification],
        cwd=_BACKEND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("multi_tenant", [False, True])
def test_label_worker_initializes_tenant_safe_database_and_dependencies(
    multi_tenant: bool,
) -> None:
    app = importlib.import_module("onyx.background.celery.apps.regulatory_labeling")
    sender = MagicMock(concurrency=1)
    with (
        patch.object(app.SqlEngine, "set_app_name") as set_app_name,
        patch.object(app.SqlEngine, "init_engine") as init_engine,
        patch.object(app.app_base, "wait_for_redis") as redis,
        patch.object(app.app_base, "wait_for_db") as database,
        patch.object(app.app_base, "wait_for_document_index_or_shutdown") as index,
        patch.object(app.app_base, "on_secondary_worker_init") as tenant_init,
        patch.object(app, "MULTI_TENANT", multi_tenant),
    ):
        app.on_worker_init(sender)
    set_app_name.assert_called_once_with("celery_worker_regulatory_labeling")
    init_engine.assert_called_once_with(pool_size=1, max_overflow=2)
    redis.assert_called_once_with(sender)
    database.assert_called_once_with(sender)
    index.assert_called_once_with()
    assert tenant_init.call_count == (0 if multi_tenant else 1)
