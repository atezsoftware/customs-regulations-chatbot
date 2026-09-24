"""One DEV task; fresh serial interpreters release each file's retained memory."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

PREFIX = "regulatory_maintenance:remaining78-20260924-resume2:"
PREVIOUS = "regulatory_maintenance:remaining78-20260924-resume1:"
ORIGINAL = "regulatory_maintenance:remaining78-20260924:"
MEMORY_CEILING = 3584 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(connection: Connection, key: str) -> Any:
    with connection.cursor() as cursor:
        cursor.execute("SELECT value FROM public.key_value_store WHERE key=%s", (key,))
        row = cursor.fetchone()
        return row[0] if row else None


def save(connection: Connection, key: str, value: dict[str, Any]) -> None:
    if not key.startswith(PREFIX):
        raise ValueError("supervisor may only update its own maintenance receipts")
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.key_value_store(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, Json(value)),
        )


def claim(connection: Connection, plans: list[dict[str, Any]]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.key_value_store(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO NOTHING RETURNING key",
            (
                PREFIX + "claim",
                Json(
                    {
                        "selection_sha256": hashlib.sha256(
                            json.dumps(plans, sort_keys=True).encode()
                        ).hexdigest()
                    }
                ),
            ),
        )
        if cursor.fetchone() is None:
            raise ValueError("run already claimed; inspect before any retry")


def stop_requested(paths: list[Path], read: Callable[[str], Any]) -> bool:
    if any(path.exists() for path in paths):
        return True
    for prefix in (PREFIX, PREVIOUS, ORIGINAL):
        control = read(prefix + "control")
        if isinstance(control, dict) and control.get("stop") is True:
            if prefix == PREFIX or control.get("requested_by") != "agent":
                return True
    return False


def child_result(
    path: Path, file_id: str, code: int, reason: str | None
) -> dict[str, Any]:
    if code == 0 and path.exists():
        result = json.loads(path.read_text())
        if result.get("file_id") != file_id or result.get("state") not in (
            "ready",
            "failed",
        ):
            raise ValueError("child receipt identity differs from selection")
        return result
    return {
        "file_id": file_id,
        "state": "failed",
        "at": now(),
        "error_type": reason or "ChildProcessExited",
        "exit_code": code,
        "detail": "Inspect retained publication manifest before retry; no automatic retry.",
    }


def include_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    ready = result["state"] == "ready"
    state["processed"] += 1
    state["ready"] += int(ready)
    state["failed"] += int(not ready)
    if ready:
        state["indexed_count"] += result.get("indexed_count", 0)
        state["new_indexed_count"] += result.get("new_indexed_count", 0)


def enforce_memory_budget(child: subprocess.Popen[Any], memory: int) -> str | None:
    if memory <= MEMORY_CEILING or child.poll() is not None:
        return None
    try:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    except ProcessLookupError:
        return None
    return "MemoryBudgetExceeded"


def main() -> None:
    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        raise ValueError("server project container required")
    directory = Path(__file__).resolve().parent
    paths = [
        directory / "STOP",
        directory.parent / "dev-remaining78-20260924" / "STOP",
        directory.parent / "dev-remaining78-20260924-resume1" / "STOP",
    ]
    plans = json.loads((directory / "file-plan.json").read_text())
    if len(plans) != 71 or len({item["file_id"] for item in plans}) != 71:
        raise ValueError("reviewed remaining selection changed")
    connection = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ["POSTGRES_PASSWORD"],
        application_name="remaining78-supervisor",
        options="-c statement_timeout=30000",
        connect_timeout=15,
    )
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        if cursor.fetchone() != ("customs-regulations-dev",):
            raise ValueError("DEV database identity mismatch")

    def read(key: str) -> Any:
        return load(connection, key)

    if stop_requested(paths, read):
        raise ValueError("STOP is present; refusing launch")
    prior = read(PREVIOUS + "status")
    if (
        prior.get("pid") != 508
        or prior.get("processed") != 4
        or prior.get("heartbeat_at") != "2026-09-24T06:47:07.784315+00:00"
    ):
        raise ValueError("reviewed interrupted checkpoint changed")
    claim(connection, plans)
    state: dict[str, Any] = {
        "state": "running",
        "selected": len(plans),
        "processed": 0,
        "ready": 0,
        "failed": 0,
        "indexed_count": 0,
        "new_indexed_count": 0,
        "pid": os.getpid(),
    }

    def persist() -> None:
        save(connection, PREFIX + "status", {**state, "heartbeat_at": now()})

    active_child: subprocess.Popen[Any] | None = None
    try:
        for number, item in enumerate(plans):
            if stop_requested(paths, read):
                state["state"] = "stopped"
                break
            result_path = directory / f"result-{number}.json"
            state.update(current_file=item["file_id"], file_started_at=now())
            persist()
            with (directory / f"file-{number}.log").open("x") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(directory / "repair_server.py"),
                        "--apply",
                        "--plan",
                        str(directory / "file-plan.json"),
                        "--file-index",
                        str(number),
                        "--output",
                        str(result_path),
                    ],
                    cwd="/app/onyx/db",
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active_child = child
                state["child_pid"] = child.pid
                last_status = 0.0
                reason = None
                while child.poll() is None:
                    memory = int(Path("/sys/fs/cgroup/memory.current").read_text())
                    state["container_memory_bytes"] = memory
                    reason = enforce_memory_budget(child, memory)
                    if reason:
                        break
                    if time.monotonic() - last_status >= 30:
                        phase = directory / "phase.json"
                        if phase.exists():
                            detail = json.loads(phase.read_text())
                            if detail.get("file_id") == item["file_id"]:
                                state["phase"] = detail.get("phase")
                        persist()
                        last_status = time.monotonic()
                    time.sleep(1)
                result = child_result(
                    result_path, item["file_id"], child.wait(), reason
                )
                active_child = None
            save(connection, PREFIX + "file:" + item["file_id"], result)
            include_result(state, result)
            persist()
        else:
            state["state"] = "complete_with_errors" if state["failed"] else "complete"
    except BaseException:
        if active_child is not None and active_child.poll() is None:
            enforce_memory_budget(active_child, MEMORY_CEILING + 1)
        state["state"] = "operator_error"
        raise
    finally:
        persist()
        connection.close()


if __name__ == "__main__":
    main()
