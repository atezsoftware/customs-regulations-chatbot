"""Executed on stdin in an old DEV background pod; never purges broker queues."""

import json
import socket
import subprocess
import sys
import time
from typing import Any

from celery import Celery

SUPERVISOR = ["supervisorctl", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
WORKERS = (
    "regulatory_benchmark",
    "user_file_processing",
    "regulatory_indexing",
    "light",
    "csv_generation",
    "monitoring",
    "regulatory_annex",
)
BEATS = ("celery_beat", "celery_beat_regulatory_indexing")


class CutoverRefusal(RuntimeError):
    """A fixed diagnostic that is safe to expose in runner logs."""


def run(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=30
    ).stdout


def drain() -> None:
    lines = run(SUPERVISOR + ["status"]).splitlines()
    processes = {line.split()[0]: line.split()[1:] for line in lines}
    allowed = (
        {f"celery_worker_{name}" for name in WORKERS}
        | set(BEATS)
        | {"log-redirect-handler"}
    )
    if set(processes) - allowed:
        raise CutoverRefusal("unknown_supervisor_program")
    for name in BEATS:
        if processes.get(name, [""])[0] == "RUNNING":
            run(SUPERVISOR + ["stop", name])
    app = Celery("dev_cutover")
    app.config_from_object("onyx.background.celery.configs.base")
    destinations: list[str] = []
    for name in WORKERS:
        process = processes.get(f"celery_worker_{name}")
        if process is None:
            if name == "regulatory_annex":
                continue
            raise CutoverRefusal("missing_old_worker")
        if process[0] in {"STOPPED", "EXITED"} and name == "regulatory_annex":
            continue
        if process[0] != "RUNNING" or process[1] != "pid":
            raise CutoverRefusal("old_worker_not_running")
        destination = f"{name}@{socket.gethostname()}"
        inspect = app.control.inspect(destination=[destination], timeout=10)
        stats = inspect.stats()
        if (
            not stats
            or set(stats) != {destination}
            or stats[destination]["pid"] != int(process[2].rstrip(","))
        ):
            raise CutoverRefusal("old_worker_identity_mismatch")
        queues = inspect.active_queues()
        if not queues or set(queues) != {destination} or not queues[destination]:
            raise CutoverRefusal("old_worker_queues_unknown")
        for queue in queues[destination]:
            result = app.control.cancel_consumer(
                queue["name"], destination=[destination], reply=True, timeout=10
            )
            if not result or any(
                "ok" not in reply.get(destination, {}) for reply in result
            ):
                raise CutoverRefusal("consumer_cancel_unacknowledged")
        destinations.append(destination)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        inspect = app.control.inspect(destination=destinations, timeout=10)
        responses: list[dict[str, Any] | None] = [
            inspect.active(),
            inspect.reserved(),
            inspect.scheduled(),
            inspect.active_queues(),
        ]
        if any(
            not response or set(response) != set(destinations) for response in responses
        ):
            raise CutoverRefusal("drain_inspection_incomplete")
        if all(
            not tasks
            for response in responses
            if response
            for tasks in response.values()
        ):
            for destination in destinations:
                run(SUPERVISOR + ["stop", "celery_worker_" + destination.split("@")[0]])
            print(json.dumps({"drained": True, "workers": len(destinations)}))
            return
        time.sleep(2)
    raise CutoverRefusal("drain_timeout_queues_preserved")


if __name__ == "__main__":
    try:
        drain()
    except CutoverRefusal as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("DEV_CUTOVER_DRAIN_REFUSED", file=sys.stderr)
        sys.exit(1)
