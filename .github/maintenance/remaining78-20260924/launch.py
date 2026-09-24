"""Proposed one-time launcher inside the existing, resource-limited DEV pod."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

directory = Path(__file__).resolve().parent
if not os.environ.get("KUBERNETES_SERVICE_HOST"):
    raise SystemExit("Launch requires the existing DEV project container")
if (directory / "STOP").exists():
    raise SystemExit("STOP is present; refusing launch")
with (directory / "launch.json").open("x") as receipt:
    with (directory / "worker.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(directory / "repair_server.py"),
                "--apply",
                "--plan",
                str(directory / "file-plan.json"),
                "--output",
                str(directory / "results.jsonl"),
            ],
            cwd="/app/onyx/db",
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt.write(json.dumps({"pid": process.pid, "run": "remaining78-20260924"}))
time.sleep(3)
if process.poll() is not None:
    raise SystemExit(
        "Worker exited during launch; inspect worker.log, do not redispatch"
    )
print(
    json.dumps({"state": "launched", "pid": process.pid, "run": "remaining78-20260924"})
)
