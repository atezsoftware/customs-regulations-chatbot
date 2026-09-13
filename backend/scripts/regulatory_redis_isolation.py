"""Claim and verify the dedicated logical Redis databases for the DEV deployment."""

from __future__ import annotations

import hashlib
import json
import re
import signal
import socket
import subprocess
import sys
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from redis import Redis

DATABASES = (4, 5, 6)  # application, broker, results
OWNER_KEY = "onyx:deployment-owner"


class IsolationRefusal(RuntimeError):
    """Only fixed operational codes are safe to publish."""


def claim_database(client: Redis, owner: str) -> None:
    result = client.eval(
        """
        local owner = redis.call('GET', KEYS[1])
        if owner then
            if owner == ARGV[1] then return 1 end
            return -1
        end
        if redis.call('DBSIZE') ~= 0 then return -2 end
        redis.call('SET', KEYS[1], ARGV[1])
        return 1
        """,
        1,
        OWNER_KEY,
        owner,
    )
    if result != 1:
        raise IsolationRefusal("owner_mismatch" if result == -1 else "not_empty")


def verify_worker() -> None:
    from onyx.background.celery.versioned_apps.client import app

    process = subprocess.run(
        [
            "supervisorctl",
            "-c",
            "/etc/supervisor/conf.d/supervisord.conf",
            "status",
            "celery_worker_user_file_processing",
        ],
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )
    status = re.fullmatch(
        r"celery_worker_user_file_processing\s+RUNNING\s+pid ([1-9][0-9]*),[^\n]*\n?",
        process.stdout,
    )
    if process.returncode or status is None:
        raise IsolationRefusal("ordinary_worker_not_running")
    hostname = socket.gethostname()
    node = "user_file_processing@" + hostname
    inspector = app.control.inspect(timeout=4, destination=[node])
    stats = (inspector.stats() or {}).get(node, {})
    registered = (inspector.registered() or {}).get(node, [])
    queues = (inspector.active_queues() or {}).get(node, [])
    if stats.get("pid") != int(status[1]):
        raise IsolationRefusal("ordinary_worker_pid_mismatch")
    if not {"process_single_user_file", "index_single_user_file"} <= {
        task.split(" ", 1)[0] for task in registered
    }:
        raise IsolationRefusal("ordinary_worker_handlers_missing")
    if {item["name"] for item in queues} != {
        "user_file_processing",
        "user_file_project_sync",
        "user_file_delete",
        "user_file_port",
    }:
        raise IsolationRefusal("ordinary_worker_queues_mismatch")
    workers = app.control.inspect(timeout=4).ping() or {}
    if node not in workers or any(
        name.rsplit("@", 1)[-1] != hostname for name in workers
    ):
        raise IsolationRefusal("unexpected_broker_worker")


def verify_transports() -> None:
    from celery.backends.redis import RedisBackend

    from onyx.background.celery.versioned_apps.client import app

    for connection in (app.connection_for_read(), app.connection_for_write()):
        with connection:
            if connection.virtual_host.strip("/") != str(DATABASES[1]):
                raise IsolationRefusal("effective_broker_database_mismatch")
    backend = app.backend
    if not isinstance(backend, RedisBackend):
        raise IsolationRefusal("redis_result_backend_required")
    if backend.client.connection_pool.connection_kwargs.get("db") != DATABASES[2]:
        raise IsolationRefusal("effective_result_database_mismatch")
    if backend.task_keyprefix != b"onyx:redis-db:6:celery-task-meta-":
        raise IsolationRefusal("result_notification_scope_mismatch")


def run(mode: str) -> dict[str, str | bool]:
    from redis import Redis

    from onyx.configs import app_configs as config
    from onyx.redis.redis_pool import RedisPool

    if mode not in {"reserve", "verify", "worker"}:
        raise IsolationRefusal("invalid_mode")
    if (
        config.POSTGRES_DB != "customs-regulations-dev"
        or config.REGULATORY_ANNEX_ENVIRONMENT != "dev"
    ):
        raise IsolationRefusal("DEV_environment_required")
    configured = (
        config.REDIS_DB_NUMBER,
        config.REDIS_DB_NUMBER_CELERY,
        config.REDIS_DB_NUMBER_CELERY_RESULT_BACKEND,
    )
    if mode != "reserve" and configured != DATABASES:
        raise IsolationRefusal("runtime_databases_mismatch")
    if mode != "reserve":
        verify_transports()
    owner = (
        "customs-regulations-dev:"
        + hashlib.sha256(
            f"{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}".encode()
        ).hexdigest()
    )
    with ExitStack() as stack:
        clients: list[Redis] = []
        for database in DATABASES:
            pool = RedisPool.create_pool(
                db=database, ssl=config.REDIS_SSL, max_connections=1
            )
            pool.connection_kwargs.update(socket_timeout=3, socket_connect_timeout=3)
            stack.callback(pool.disconnect)
            clients.append(stack.enter_context(Redis(connection_pool=pool)))
        # Inspect all destinations before making any reservation. Never copy or
        # purge queues from the formerly shared broker.
        for database, client in zip(DATABASES, clients, strict=True):
            existing = client.get(OWNER_KEY)
            if existing is not None and existing != owner.encode():
                raise IsolationRefusal("owner_mismatch")
            if mode == "reserve" and existing is None:
                own_id = client.client_id()
                connections = cast(list[dict[str, Any]], client.client_list())
                if any(
                    int(item["db"]) == database and int(item["id"]) != own_id
                    for item in connections
                ):
                    raise IsolationRefusal("database_has_clients")
                if client.dbsize() != 0:
                    raise IsolationRefusal("not_empty")
            elif existing is None:
                raise IsolationRefusal("owner_missing")
        if mode == "reserve":
            for client in clients:
                claim_database(client, owner)
    if mode == "worker":
        verify_worker()
    return {
        "redis_isolation": {
            "reserve": "reserved",
            "verify": "verified",
            "worker": "worker_verified",
        }[mode],
        "migration_required": configured != DATABASES,
    }


def main() -> None:
    def expired(_signum: int, _frame: object) -> None:
        raise IsolationRefusal("probe_timeout")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(45)
    try:
        if len(sys.argv) != 2:
            raise IsolationRefusal("invalid_mode")
        print(json.dumps(run(sys.argv[1]), sort_keys=True), flush=True)
    except Exception as error:
        code = (
            str(error) if isinstance(error, IsolationRefusal) else type(error).__name__
        )
        print("NOT_READY redis_isolation " + code, flush=True)
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
