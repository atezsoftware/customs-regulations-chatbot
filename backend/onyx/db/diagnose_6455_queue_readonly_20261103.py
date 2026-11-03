"""Read DEV source queue counts and safe task names without task payload export."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

NAMESPACE = "customs-regulations-dev"
PACKAGE = "d2d0756e-e996-46cf-991b-aec286ed4aab"


def remote(expected: str) -> dict[str, Any]:
    from onyx import __version__
    from onyx.background.celery.versioned_apps.client import app
    from onyx.regulatory.amendments.annexes import config

    assert __version__ == expected
    assert os.environ.get("POSTGRES_DB") == NAMESPACE
    assert config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
    source_queue = config.source_queue_name()
    broker = urlsplit(str(app.conf.broker_url))
    broker_identity = {
        "scheme": broker.scheme,
        "host": broker.hostname,
        "port": broker.port,
        "path": broker.path,
        "key_prefix": app.conf.broker_transport_options.get("global_keyprefix", ""),
    }
    report: dict[str, Any] = {
        "diagnostic": "6455_safe_queue_readonly",
        "revision": __version__,
        "database_identity_sha256": hashlib.sha256(
            config.ANNEX_DATABASE_IDENTITY.encode()
        ).hexdigest(),
        "environment_sha256": hashlib.sha256(
            config.REGULATORY_ANNEX_ENVIRONMENT.encode()
        ).hexdigest(),
        "source_queue": source_queue,
        "broker_identity_sha256": hashlib.sha256(
            json.dumps(broker_identity, sort_keys=True).encode()
        ).hexdigest(),
        "source_worker_enabled": config.REGULATORY_ANNEX_WORKER_ENABLED,
        "application_mutations": 0,
        "task_payloads_exported": False,
        "provider_calls": 0,
    }
    try:
        with app.connection_for_read() as connection:
            connection.ensure_connection(max_retries=0)
            with connection.channel() as channel:
                assert connection.transport.driver_type == "redis"
                priorities = list(channel.priority_steps)
                assert 1 <= len(priorities) <= 10
                report["queue_counts_by_priority"] = [
                    {
                        "priority": priority,
                        "count": channel.client.llen(
                            channel._q_for_pri(source_queue, priority)
                        ),
                    }
                    for priority in priorities
                ]
    except Exception as error:
        report["queue_count_error_class"] = type(error).__name__
    inspect = app.control.inspect(timeout=3)
    for method in ("active_queues", "active", "reserved", "scheduled"):
        try:
            states = getattr(inspect, method)() or {}
            workers = []
            for worker, entries in states.items():
                if not isinstance(entries, list):
                    continue
                if method == "active_queues":
                    workers.append(
                        {
                            "worker": worker,
                            "queues": sorted(
                                entry["name"]
                                for entry in entries
                                if isinstance(entry.get("name"), str)
                            ),
                        }
                    )
                    continue
                tasks = []
                for entry in entries:
                    task = entry.get("request", entry)
                    name = task.get("name") or task.get("type")
                    if not isinstance(name, str) or not re.fullmatch(
                        r"[A-Za-z0-9_.-]{1,200}", name
                    ):
                        continue
                    kwargs = task.get("kwargs")
                    owned = (
                        isinstance(kwargs, dict) and kwargs.get("package_id") == PACKAGE
                    )
                    tasks.append({"name": name, "owned_6455": owned})
                workers.append(
                    {"worker": worker, "count": len(entries), "tasks": tasks}
                )
            report[method] = workers
        except Exception as error:
            report[method + "_error_class"] = type(error).__name__
    return report


def launch(expected: str) -> None:
    pods = json.loads(
        subprocess.check_output(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json"], timeout=30
        )
    )["items"]
    targets: dict[str, tuple[str, str]] = {}
    for pod in pods:
        name = pod["metadata"]["name"]
        label = pod["metadata"].get("labels", {}).get("app", "")
        if pod["status"]["phase"] != "Running" or pod["metadata"].get(
            "deletionTimestamp"
        ):
            continue
        role = (
            "background"
            if label == "dev-customs-regulations-background"
            else "api"
            if "api" in label or "api" in name
            else None
        )
        if role is None or role in targets:
            continue
        statuses = {c["name"]: c for c in pod["status"].get("containerStatuses", [])}
        for container in pod["spec"]["containers"]:
            if container["image"].endswith(
                "customs-regulations-backend-lite-dev:" + expected
            ) and statuses.get(container["name"], {}).get("ready"):
                targets[role] = name, container["name"]
    assert set(targets) == {"api", "background"}
    for role, (pod, container) in sorted(targets.items()):
        command = [
            "kubectl",
            "exec",
            "-i",
            pod,
            "-n",
            NAMESPACE,
            "-c",
            container,
            "--",
            "sh",
            "-ec",
            '. /vault/secrets/config; export PYTHONPATH=/app; exec python - --remote --expected-version "$1"',
            "sh",
            expected,
        ]
        result = subprocess.run(
            command,
            input=Path(__file__).read_text(),
            text=True,
            capture_output=True,
            timeout=90,
        )
        accepted = []
        for line in result.stdout.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if (
                record.get("diagnostic") == "6455_safe_queue_readonly"
                or "diagnostic_error_class" in record
            ):
                accepted.append(record)
        assert len(accepted) == 1
        print(json.dumps({"role": role, "pod": pod, **accepted[0]}))
        if result.returncode:
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.expected_version)
    if not args.remote:
        launch(args.expected_version)
        return
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            logging.disable(logging.CRITICAL)
            report = remote(args.expected_version)
        print(json.dumps(report))
    except Exception as error:
        print(
            json.dumps(
                {
                    "diagnostic_error_class": type(error).__name__,
                    "application_mutations": 0,
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
