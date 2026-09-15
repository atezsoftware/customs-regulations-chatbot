"""Registration and scope failures that otherwise strand reviewed publications."""

import importlib.util
from datetime import timedelta

import pytest


def test_runtime_registers_handlers_and_exclusively_consumes_scoped_work() -> None:
    module = "onyx.background.celery.apps.regulatory_annex"
    assert importlib.util.find_spec(module), "annex runtime consumer is absent"
    from onyx.background.celery.apps.regulatory_annex import celery_app
    from onyx.background.celery.regulatory_annex_worker import worker_command

    celery_app.loader.import_default_modules()
    assert {
        "acquire_amendment_sources",
        "regulatory_amendment_run",
        "publish_annex_change",
        "recover_annex_publications",
        "recover_amendment_sources",
    } <= celery_app.tasks.keys()
    assert celery_app.conf.worker_concurrency == 1
    assert celery_app.conf.worker_prefetch_multiplier == 1
    command = worker_command(["--hostname=regulatory_annex@fixture"])
    queues = command[command.index("-Q") + 1].split(",")
    assert len(set(queues)) == 3
    assert all(queue.startswith("regulatory_annex_") for queue in queues)
    assert not {"celery", "regulatory_amendment", "regulatory_indexing"} & set(queues)
    for arguments in (["-Q", "celery"], ["--queues=celery"], ["-c", "8"]):
        with pytest.raises(ValueError):
            worker_command(arguments)


def test_periodic_recovery_keeps_explicit_scope_when_creation_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.apps.regulatory_indexing_beat import (
        RegulatoryIndexingScheduler,
    )
    from onyx.regulatory.amendments.annexes import config

    monkeypatch.setattr(config, "REGULATORY_ANNEX_WORKER_ENABLED", False)
    assert all(
        entry["task"] != "recover_annex_publications"
        for entry in RegulatoryIndexingScheduler.generate_schedule(["public"]).values()
    )
    monkeypatch.setattr(config, "REGULATORY_ANNEX_WORKER_ENABLED", True)
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", False)
    entries = [
        entry
        for entry in RegulatoryIndexingScheduler.generate_schedule(["public"]).values()
        if entry["task"] == "recover_annex_publications"
    ]
    assert len(entries) == 1, "authorized intents have no scheduled recovery"
    entry = entries[0]
    assert entry["kwargs"] == {
        "tenant_id": "public",
        "environment": config.REGULATORY_ANNEX_ENVIRONMENT,
        "database_identity": config.ANNEX_DATABASE_IDENTITY,
    }
    assert entry["schedule"] == timedelta(minutes=1)
    assert 0 < entry["options"]["expires"] <= 3600
    assert entry["options"]["queue"] == config.publication_queue_name()


def test_readiness_refuses_foreign_worker_stale_pid_missing_handlers_and_concurrency() -> (
    None
):
    module = "onyx.background.celery.regulatory_annex_readiness"
    assert importlib.util.find_spec(module), (
        "annex readiness does not verify live ownership"
    )
    from onyx.background.celery.regulatory_annex_readiness import validate_worker
    from onyx.regulatory.amendments.annexes.config import (
        analysis_queue_name,
        publication_queue_name,
        source_queue_name,
    )

    worker = "regulatory_annex@fixture"
    queues = {
        worker: [
            {"name": queue}
            for queue in (
                source_queue_name(),
                analysis_queue_name(),
                publication_queue_name(),
            )
        ]
    }
    stats = {worker: {"pid": 71, "pool": {"max-concurrency": 1}}}
    registered = {
        worker: [
            "acquire_amendment_sources",
            "regulatory_amendment_run",
            "publish_annex_change",
            "recover_annex_publications",
            "recover_amendment_sources",
        ]
    }
    validate_worker(worker, 71, queues=queues, stats=stats, registered=registered)
    failures = [
        {"queues": {"regulatory_annex@foreign": queues[worker]}},
        {"queues": {worker: queues[worker] + [{"name": "celery"}]}},
        {"stats": {worker: {"pid": 70, "pool": {"max-concurrency": 1}}}},
        {"stats": {worker: {"pid": 71, "pool": {"max-concurrency": 2}}}},
        {"registered": {worker: registered[worker][:-1]}},
    ]
    for changed in failures:
        with pytest.raises(ValueError):
            validate_worker(
                worker,
                71,
                **{
                    "queues": queues,
                    "stats": stats,
                    "registered": registered,
                    **changed,
                },
            )


def test_analysis_dispatch_preserves_ordinary_worker_until_dev_lane_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE
    from onyx.regulatory.amendments.annexes import config

    monkeypatch.setattr(config, "REGULATORY_ANNEX_WORKER_ENABLED", False)
    assert config.analysis_delivery_queue_name() == REGULATORY_AMENDMENT_QUEUE
    monkeypatch.setattr(config, "REGULATORY_ANNEX_WORKER_ENABLED", True)
    assert config.analysis_delivery_queue_name() == config.analysis_queue_name()


def test_source_worker_refuses_foreign_single_tenant_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    from onyx.background.celery.tasks.regulatory_amendments import sources
    from onyx.regulatory.amendments.annexes import config

    monkeypatch.setattr(sources, "get_current_tenant_id", lambda: "foreign_tenant")
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)

    def unexpected_load(**_kwargs: object) -> None:
        raise AssertionError("foreign tenant reached source loading")

    monkeypatch.setattr(sources, "run_source_package", unexpected_load)
    with pytest.raises(ValueError, match="scope mismatch"):
        sources.acquire_amendment_sources.run(
            package_id=str(uuid4()),
            tenant_id="foreign_tenant",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
