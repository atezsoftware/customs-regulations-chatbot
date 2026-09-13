"""Read only operational receipts for two owned DEV files on one observed consumer."""

import json
import os
import re
import socket
from pathlib import Path

EXPECTED_HOST = "test-v1-customs-regulations-background-deployment-57788b682n78h"
UPLOAD_FILE = "32ab4fc6-f7cd-4d51-a34f-1c808760908f"
CANARY_FILE = "cbb3a7ac-46bb-4ca8-907c-16c0c6dfad94"
LOG_PATH = Path("/var/log/onyx/celery_worker_user_file_processing.log")
MAX_BYTES = 16 * 1024 * 1024


def owned_events(content: str) -> list[dict[str, str]]:
    lines = re.sub(r"\x1b\[[0-9;]*m", "", content).splitlines()
    patterns = {
        "upload_started": f"process_user_file_impl - Starting id={UPLOAD_FILE}",
        "upload_missing": f"process_user_file_impl - UserFile not found id={UPLOAD_FILE}",
        "upload_lock_held": f"process_user_file_impl - Lock held, skipping user_file_id={UPLOAD_FILE}",
        "upload_finished": f"process_user_file_impl - Finished id={UPLOAD_FILE}",
        "upload_failed": f"process_user_file_impl - Error processing file id={UPLOAD_FILE}",
        "canary_index_missing": f"index_user_file_impl - user file {CANARY_FILE} is gone or being deleted; skipping",
        "canary_index_finished": f"index_user_file_impl - Indexed id={CANARY_FILE}",
        "canary_index_failed": f"index_user_file_impl - Failed to index user file {CANARY_FILE}",
    }
    events: list[dict[str, str]] = []
    for position, line in enumerate(lines):
        kinds = [kind for kind, pattern in patterns.items() if pattern in line]
        if "Received unregistered task of type 'index_single_user_file'" in line:
            block = [line]
            for following in lines[position + 1 : position + 40]:
                if re.match(r"[A-Z]+\s+[0-9]{2}/[0-9]{2}/[0-9]{4} ", following):
                    break
                block.append(following)
            if any(CANARY_FILE in following for following in block):
                kinds.append("canary_index_unregistered")
        for kind in kinds:
            event = {"kind": kind}
            timestamp = re.search(
                r"[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} [AP]M", line
            )
            task = re.search(
                r"\[(?:process_single_user_file|index_single_user_file)\(([0-9a-f-]{36})\)\]",
                line,
            )
            if timestamp:
                event["timestamp"] = timestamp[0]
            if task:
                event["task_id"] = task[1]
            events.append(event)
    return events


def read_receipts() -> dict[str, object]:
    if socket.gethostname() != EXPECTED_HOST:
        return {"stage": "shared_queue_receipt", "status": "hostname_mismatch"}
    events: list[dict[str, str]] = []
    files = 0
    read_bytes = 0
    truncated = False
    for number in range(11):
        path = Path(str(LOG_PATH) + (f".{number}" if number else ""))
        try:
            with path.open("rb") as source:
                size = os.fstat(source.fileno()).st_size
                source.seek(max(0, size - MAX_BYTES))
                content = source.read(MAX_BYTES)
        except FileNotFoundError:
            break
        files += 1
        read_bytes += len(content)
        truncated = truncated or size > MAX_BYTES
        events.extend(owned_events(content.decode("utf-8", errors="replace")))
    # Deduplicate duplicated handler output; only finite event metadata leaves the pod.
    unique = {json.dumps(event, sort_keys=True): event for event in events}
    return {
        "stage": "shared_queue_receipt",
        "status": "read",
        "hostname_verified": True,
        "database_accessed": False,
        "log_files": files,
        "log_bytes": read_bytes,
        "log_truncated": truncated,
        "events_truncated": len(unique) > 30,
        "events": list(unique.values())[:30],
    }


if __name__ == "__main__":
    import signal

    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError()

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(30)
    try:
        print(json.dumps(read_receipts(), sort_keys=True))
    except Exception:
        print(json.dumps({"stage": "shared_queue_receipt", "status": "read_failed"}))
        raise SystemExit(1) from None
