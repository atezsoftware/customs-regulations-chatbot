"""Temporary DEV-only launcher; remove from git after the approved corpus run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

DATABASE = "customs-regulations-dev"
INDEX = "q8lSz7g2Rvq7739qGJz6jg"
IMAGE = "1b683deb0c7bdedc1af3a952e5b0616e33fd5175"
SELECTION_HASH = "89f9cd193ebb4eea98a636bae51e5683e2759f04094511af4e0e0e29a6680970"
RUN = "dev-baseline-20260923-89f9cd19"
PREFIX = "regulatory_maintenance:" + RUN + ":"
ACTIVE_KEY = "regulatory_maintenance:baseline:server_once"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_record(record: dict[str, Any]) -> int:
    if record.get("database") != DATABASE or record.get("state") != "ready":
        raise ValueError("file did not reach DEV readiness")
    before = record["indexes"]
    after = record.get("after", before)
    if INDEX not in before or set(before) != set(after):
        raise ValueError("physical index inventory changed")
    count = 0
    for index, original in before.items():
        current = after[index]
        if current["state"] not in {"ready", "unindexed"} or current["issues"]:
            raise ValueError("post-apply source issues")
        for key in (
            "source_sha256",
            "vectors_sha256",
            "indexed_count",
            "canonical_count",
        ):
            if current[key] != original[key]:
                raise ValueError("preservation mismatch: " + key)
        if current["binding_count"] != current["indexed_count"]:
            raise ValueError("incomplete temporal bindings")
        count += current["indexed_count"]
    return count


class Receipts:
    def __init__(
        self, identifiers: list[str], persist: Callable[[dict[str, Any]], None]
    ):
        self.selected = set(identifiers)
        if len(self.selected) != len(identifiers):
            raise ValueError("duplicate selection")
        self.processed: set[str] = set()
        self.ready = self.failed = self.indexed = 0
        self.last_progress_at = now()
        self.persist = persist

    def accept(self, record: dict[str, Any]) -> None:
        identifier = record["file_id"]
        if identifier not in self.selected or identifier in self.processed:
            raise ValueError("foreign or duplicate report")
        record = {**record, "observed_at": now()}
        try:
            count = validate_record(record)
        except (ValueError, KeyError, TypeError) as error:
            record.update(state="error", verification_error=str(error))
            count = 0
        self.persist(record)
        self.processed.add(identifier)
        self.ready += int(record["state"] == "ready")
        self.failed += int(record["state"] != "ready")
        self.indexed += count
        self.last_progress_at = now()

    def summary(self) -> dict[str, int]:
        return dict(
            selected=len(self.selected),
            processed=len(self.processed),
            ready=self.ready,
            failed=self.failed,
            indexed_records=self.indexed,
        )


def consume(path: Path, position: int, accept: Callable[[dict[str, Any]], None]) -> int:
    if not path.exists():
        return position
    with path.open() as source:
        source.seek(position)
        while line := source.readline():
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            if "file_id" in record:
                accept(record)
            position = source.tell()
    return position


def terminal_state(code: int, summary: dict[str, int]) -> str:
    if code == 3:
        return "stopped"
    if code not in (0, 2) or summary["processed"] != summary["selected"]:
        return "error"
    return "complete_with_errors" if summary["failed"] else "complete"


class Store:
    """Only task-prefixed operational receipts in the existing DEV key/value table."""

    def __init__(self) -> None:
        from sqlalchemy import text

        from onyx.db.engine.sql_engine import SqlEngine

        SqlEngine.init_engine(pool_size=1, max_overflow=0)
        with self.transaction() as session:
            if session.scalar(text("SELECT current_database()")) != DATABASE:
                raise ValueError("server must use DEV database")

    def transaction(self):
        from onyx.db.engine.sql_engine import get_session_with_tenant

        return get_session_with_tenant(tenant_id="public")

    def claim(self, initial: dict[str, Any]) -> bool:
        from sqlalchemy import text

        with self.transaction() as session:
            if session.scalar(text("SELECT current_database()")) != DATABASE:
                raise ValueError("server must use DEV database")
            inserted = session.scalar(
                text("""INSERT INTO key_value_store(key,value)
                VALUES(:key,CAST(:value AS jsonb)) ON CONFLICT(key) DO NOTHING RETURNING key"""),
                dict(key=ACTIVE_KEY, value=json.dumps(initial)),
            )
            if inserted is None:
                return False
            session.execute(
                text(
                    "INSERT INTO key_value_store(key,value) VALUES(:key,CAST(:value AS jsonb))"
                ),
                dict(key=PREFIX + "status", value=json.dumps(initial)),
            )
            session.commit()
            return True

    def put(self, suffix: str, value: dict[str, Any]) -> None:
        from sqlalchemy import text

        with self.transaction() as session:
            session.execute(
                text("""INSERT INTO key_value_store(key,value) VALUES(:key,CAST(:value AS jsonb))
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value"""),
                dict(key=PREFIX + suffix, value=json.dumps(value)),
            )
            session.commit()

    def claim_supervisor(self) -> dict[str, Any]:
        from sqlalchemy import text

        with self.transaction() as session:
            row = session.scalar(
                text("""UPDATE key_value_store SET value=value || CAST(:change AS jsonb)
                WHERE key=:key AND value->>'state'='launching' RETURNING value"""),
                dict(
                    key=PREFIX + "status",
                    change=json.dumps(
                        dict(
                            state="starting",
                            supervisor_pid=os.getpid(),
                            heartbeat_at=now(),
                        )
                    ),
                ),
            )
            if row is None:
                raise ValueError("supervisor already claimed or launch unavailable")
            session.commit()
            return row

    def get(self, suffix: str) -> Any:
        from sqlalchemy import text

        with self.transaction() as session:
            return session.scalar(
                text("SELECT value FROM key_value_store WHERE key=:key"),
                dict(key=PREFIX + suffix),
            )


def selection(directory: Path) -> list[str]:
    raw = (directory / "file-ids.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SELECTION_HASH:
        raise ValueError("remaining selection digest changed")
    identifiers = json.loads(raw)
    if len(identifiers) != 5712 or len(set(identifiers)) != 5712:
        raise ValueError("remaining selection count changed")
    for identifier in identifiers:
        UUID(identifier)
    return identifiers


def supervise(directory: Path, store: Store) -> None:
    identifiers = selection(directory)
    current = store.claim_supervisor()
    stop = directory / "STOP"
    if stop.exists():
        raise ValueError("server STOP exists")
    os.nice(10)
    book = Receipts(identifiers, lambda row: store.put("file:" + row["file_id"], row))
    command = [
        sys.executable,
        "-u",
        "/app/scripts/prepare_regulatory_publication_baselines.py",
        "--expected-database",
        DATABASE,
        "--expected-index-uuid",
        INDEX,
        "--file-list",
        str(directory / "file-ids.json"),
        "--apply",
        "--workers",
        "4",
        "--stop-file",
        str(stop),
        "--output",
        str(directory / "baseline-report.jsonl"),
        "--progress-file",
        str(directory / "started.jsonl"),
    ]
    with (directory / "worker.log").open("a", buffering=1) as log:
        os.chmod(directory / "worker.log", 0o600)
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd="/app/onyx/db",
        )

        def status(state: str, **extra: Any) -> None:
            store.put(
                "status",
                {
                    **current,
                    **book.summary(),
                    "state": state,
                    "supervisor_pid": os.getpid(),
                    "worker_pid": child.pid,
                    "heartbeat_at": now(),
                    "last_progress_at": book.last_progress_at,
                    **extra,
                },
            )

        cursor = 0
        heartbeat = 0.0
        try:
            while child.poll() is None:
                cursor = consume(
                    directory / "baseline-report.jsonl", cursor, book.accept
                )
                if time.monotonic() - heartbeat >= 15:
                    status("running")
                    if store.get("control") == {"stop": True}:
                        stop.write_text("Requested stop from operator\n")
                    heartbeat = time.monotonic()
                time.sleep(1)
            consume(directory / "baseline-report.jsonl", cursor, book.accept)
            status(
                terminal_state(child.returncode, book.summary()),
                exit_code=child.returncode,
            )
        except BaseException:
            stop.write_text(
                "Supervisor failure: stop between files; inspect receipts before restarting\n"
            )
            try:
                status("error", error="supervisor_failed")
            finally:
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    os.chdir("/app/onyx/db")
    sys.path.insert(0, "/app")
    if os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev":
        raise ValueError("server must use DEV publication environment")
    selection(directory)
    store = Store()
    if args.supervise:
        try:
            supervise(directory, store)
        except BaseException as error:
            current = store.get("status")
            if current and current.get("supervisor_pid") == os.getpid():
                store.put(
                    "status",
                    {
                        **current,
                        "state": "error",
                        "error_type": type(error).__name__,
                        "heartbeat_at": now(),
                    },
                )
            raise
        return
    initial = dict(
        run_id=RUN,
        state="launching",
        started_at=now(),
        heartbeat_at=now(),
        database=DATABASE,
        index_uuid=INDEX,
        image_sha=IMAGE,
        selection_sha256=SELECTION_HASH,
        selected=5712,
        processed=0,
        ready=0,
        failed=0,
        indexed_records=0,
        hostname=socket.gethostname(),
    )
    if not store.claim(initial):
        print(json.dumps({"state": "already_claimed", "run_id": RUN}))
        return
    with (directory / "supervisor.log").open("a") as log:
        os.chmod(directory / "supervisor.log", 0o600)
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--supervise"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd="/app/onyx/db",
        )
    print(json.dumps({"state": "launched", "run_id": RUN, "pid": child.pid}))


if __name__ == "__main__":
    main()
