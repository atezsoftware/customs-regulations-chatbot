import json
import socket
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import pytest
from scripts import regulatory_annex_dev_cutover as cutover


def test_owned_task_log_keeps_events_and_frames_without_payloads() -> None:
    task = "719599a4-cddb-402c-ac52-3c3534581d46"
    file = "32ab4fc6-f7cd-4d51-a34f-1c808760908f"
    logs = f"""INFO Task process_single_user_file[{task}] received
INFO process_user_file_impl - Starting id={file}
ERROR process_user_file_impl - Error processing file id={file} - ValueError
Traceback (most recent call last):
  File "/app/onyx/background/celery/tasks/user_file_processing/tasks.py", line 784, in process_user_file_impl
    secret_expression()
ValueError: private source and credentials
"""
    result = cutover.summarize_markdown_task_log(logs)
    assert result["owned_task_received"] is True
    assert result["owned_task_started"] is True
    assert result["owned_task_failed"] is True
    frames = result["owned_task_frames"]
    assert isinstance(frames, str)
    assert "tasks.py:784:process_user_file_impl" in frames
    assert "private" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_unrelated_task_does_not_supply_owned_evidence() -> None:
    result = cutover.summarize_markdown_task_log(
        "Task process_single_user_file[unrelated] received\n"
        "ERROR process_user_file_impl - Error processing file id=unrelated - ValueError\n"
        '  File "/app/onyx/private.py", line 4, in unrelated\n'
    )
    assert result["owned_task_received"] is False
    assert result["owned_task_failed"] is False
    assert result["owned_task_frames"] == ""


def test_receipt_discovers_consumers_without_hostname_assumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.versioned_apps.client import app

    local = "user_file_processing@owned-dev"
    other = "user-file-import@other"
    task = "719599a4-cddb-402c-ac52-3c3534581d46"
    monkeypatch.setattr(socket, "gethostname", lambda: "owned-dev")
    monkeypatch.setattr("builtins.open", Mock(side_effect=FileNotFoundError))
    inspector = Mock()
    inspector.active_queues.return_value = {
        local: [{"name": "user_file_processing"}],
        other: [{"name": "user_file_processing"}],
        "unrelated-worker": [{"name": "unrelated-queue"}],
    }
    inspector.query_task.return_value = {
        other: {task: ["reserved", {"args": "private"}]}
    }
    inspect_call = Mock(return_value=inspector)
    monkeypatch.setattr(app.control, "inspect", inspect_call)
    result = cutover.read_markdown_task_receipt()
    assert result["normal_worker_responses"] == 3
    assert result["normal_worker_other_consumers"] == 1
    assert result["queue_consumer_names"] == ",".join(sorted([local, other]))
    assert result["owned_task_receivers"] == other + ":reserved"
    assert result["owned_task_log_available"] is False
    assert "private" not in json.dumps(result)
    inspect_call.assert_called_once_with(timeout=3)
    inspector.query_task.assert_called_once_with(task)
    cutover.validate_markdown_worker_report(
        {
            **result,
            "stage": "markdown_worker",
            "database_read_only": True,
        }
    )


def test_rotated_application_log_retains_owned_event_and_time_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "Path"
) -> None:
    from onyx.background.celery.versioned_apps.client import app

    current = tmp_path / "current.log"
    rotated = tmp_path / "rotated.log"
    current.write_text("INFO 09/13/2026 12:00:00 PM unrelated periodic task\n")
    rotated.write_text(
        "INFO 09/13/2026 11:00:00 AM unrelated periodic task\n"
        "INFO 09/13/2026 11:20:22 AM process_user_file_impl - Starting id=32ab4fc6-f7cd-4d51-a34f-1c808760908f\n"
        "WARNING 09/13/2026 11:20:22 AM process_user_file_impl - UserFile not found id=32ab4fc6-f7cd-4d51-a34f-1c808760908f\n"
    )
    paths = {
        "/var/log/onyx/celery_worker_user_file_processing.log": current,
        "/var/log/onyx/celery_worker_user_file_processing.log.1": rotated,
    }

    def log_open(path: str, mode: str) -> "BinaryIO":
        if path not in paths:
            raise FileNotFoundError(path)
        assert mode == "rb"
        return paths[path].open("rb")

    monkeypatch.setattr("builtins.open", log_open)
    inspector = Mock()
    inspector.active_queues.return_value = {}
    inspector.query_task.return_value = {}
    monkeypatch.setattr(app.control, "inspect", Mock(return_value=inspector))
    result = cutover.read_markdown_task_receipt()
    assert result["owned_task_started"] is True
    assert result["owned_task_file_missing"] is True
    assert result["owned_task_received"] is False
    assert result["owned_task_log_files"] == 2
    assert result["owned_task_log_truncated"] is False
    assert result["owned_log_first_timestamp"] == "2026-09-13T11:00:00"
    assert result["owned_log_last_timestamp"] == "2026-09-13T12:00:00"
    cutover.validate_markdown_worker_report(
        {
            **result,
            "stage": "markdown_worker",
            "database_read_only": True,
        }
    )
