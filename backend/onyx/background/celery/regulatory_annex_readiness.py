"""Read-only, PID-bound readiness for the actual scoped annex consumer."""

import argparse
import os
import re
import socket
import subprocess
from typing import cast

from onyx.regulatory.amendments.annexes.config import (
    analysis_queue_name,
    publication_queue_name,
    source_queue_name,
)

_REQUIRED_TASKS = frozenset(
    {
        "acquire_amendment_sources",
        "regulatory_amendment_run",
        "publish_annex_change",
        "recover_annex_publications",
        "recover_amendment_sources",
    }
)


def _worker_response(response: object, worker: str) -> object:
    if not isinstance(response, dict) or worker not in response:
        raise ValueError("local annex worker did not respond")
    return cast(dict[str, object], response)[worker]


def validate_worker(
    worker: str, pid: int, *, queues: object, stats: object, registered: object
) -> None:
    records = _worker_response(queues, worker)
    if not isinstance(records, list) or any(
        not isinstance(row, dict) for row in records
    ):
        raise ValueError("invalid annex worker queue response")
    names = {cast(dict[str, object], row).get("name") for row in records}
    if names != {source_queue_name(), analysis_queue_name(), publication_queue_name()}:
        raise ValueError("annex worker must consume its exact scoped queues")
    state = _worker_response(stats, worker)
    if (
        not isinstance(state, dict)
        or cast(dict[str, object], state).get("pid") != pid
        or pid <= 0
    ):
        raise ValueError("annex worker PID does not match the local process")
    pool = cast(dict[str, object], state).get("pool")
    if (
        not isinstance(pool, dict)
        or cast(dict[str, object], pool).get("max-concurrency") != 1
    ):
        raise ValueError("annex worker must have concurrency one")
    tasks = _worker_response(registered, worker)
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError("invalid annex worker task response")
    if not _REQUIRED_TASKS <= set(tasks):
        raise ValueError("annex worker is missing required handlers")


def check_worker(worker: str, pid: int) -> None:
    from onyx.background.celery.versioned_apps.regulatory_annex import app
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.regulatory.publication_reads import public_read_store

    # Standalone readiness has no API/worker startup signal to initialize its pool.
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    public_read_store().observe()
    inspector = app.control.inspect(timeout=5, destination=[worker])
    validate_worker(
        worker,
        pid,
        queues=inspector.active_queues(),
        stats=inspector.stats(),
        registered=inspector.registered(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hostname", default=f"regulatory_annex@{socket.gethostname()}"
    )
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    try:
        if not os.environ.get("REGULATORY_ANNEX_ENVIRONMENT"):
            raise ValueError("explicit stable publication environment is required")
        pid = args.pid
        if pid is None:
            status = subprocess.run(
                [
                    "supervisorctl",
                    "-c",
                    "/etc/supervisor/conf.d/supervisord.conf",
                    "status",
                    "celery_worker_regulatory_annex",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            match = re.fullmatch(
                r"celery_worker_regulatory_annex\s+RUNNING\s+pid ([1-9][0-9]*),[^\n]*\n?",
                status,
            )
            if match is None:
                raise ValueError("annex Supervisor process is not running")
            pid = int(match.group(1))
        check_worker(args.hostname, pid)
    except Exception as error:
        # Credentials, broker URLs and provider configuration never enter this output.
        print(f"NOT_READY annex_worker {type(error).__name__}")
        return 1
    print("READY annex_worker scoped_queues registered_handlers concurrency_one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
