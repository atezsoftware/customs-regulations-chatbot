from pathlib import Path

import pytest
from scripts import regulatory_shared_queue_log_probe as probe


def test_owned_receipt_retains_exact_task_and_reason_without_payload() -> None:
    task = "719599a4-cddb-402c-ac52-3c3534581d46"
    content = (
        f"WARNING 09/13/2026 11:20:22 AM tasks.py:782: [process_single_user_file({task})] "
        f"process_user_file_impl - UserFile not found id={probe.UPLOAD_FILE}\n"
        "private unrelated message\n"
    )
    assert probe.owned_events(content) == [
        {
            "kind": "upload_missing",
            "timestamp": "09/13/2026 11:20:22 AM",
            "task_id": task,
        }
    ]


def test_unregistered_canary_event_requires_owned_file_in_message_block() -> None:
    message = "Received unregistered task of type 'index_single_user_file'.\n"
    assert probe.owned_events(message + "unrelated-file") == []
    assert probe.owned_events(message + probe.CANARY_FILE) == [
        {"kind": "canary_index_unregistered"}
    ]
    assert (
        probe.owned_events(message + "INFO 09/13/2026 11:20:22 AM " + probe.CANARY_FILE)
        == []
    )


def test_hostname_mismatch_does_not_read_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "unrelated")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Unexpected filesystem read")

    monkeypatch.setattr(Path, "open", forbidden)
    assert probe.read_receipts()["status"] == "hostname_mismatch"


def test_fixed_log_rotations_keep_only_owned_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "worker.log"
    log.write_text("unrelated private log\n")
    log.with_name("worker.log.1").write_text(
        f"index_user_file_impl - user file {probe.CANARY_FILE} is gone or being deleted; skipping\n"
    )
    monkeypatch.setattr(probe, "LOG_PATH", log)
    monkeypatch.setattr(probe.socket, "gethostname", lambda: probe.EXPECTED_HOST)
    result = probe.read_receipts()
    assert result["log_files"] == 2
    assert result["database_accessed"] is False
    assert result["events"] == [{"kind": "canary_index_missing"}]
