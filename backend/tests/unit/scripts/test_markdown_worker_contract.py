import json
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from scripts import regulatory_annex_dev_cutover as cutover


def replies() -> dict[str, Any]:
    node = "user_file_processing@owned-dev-pod"
    return {
        "registered": {node: ["process_single_user_file", "index_single_user_file"]},
        "active_queues": {
            node: [
                {"name": name}
                for name in [
                    "user_file_processing",
                    "user_file_project_sync",
                    "user_file_delete",
                    "user_file_port",
                ]
            ]
        },
        "stats": {node: {"pid": 23, "pool": {"max-concurrency": 4}}},
        "active": {
            node: [
                {
                    "name": "index_single_user_file",
                    "kwargs": {"secret": "must not escape"},
                }
            ]
        },
        "reserved": {
            node: [{"name": "unrelated-secret-task", "args": ["private body"]}]
        },
    }


def test_current_worker_contract_is_scoped_and_payload_free() -> None:
    result = cutover.summarize_markdown_worker(
        "user_file_processing@owned-dev-pod", 23, replies()
    )
    assert result["status"] == "read"
    assert result["index_registered"] is True
    assert result["queues_match"] is True
    assert result["stats_pid_matches"] is True
    assert result["concurrency"] == 4
    assert result["active_index_count"] == 1
    assert result["reserved_other_count"] == 1
    assert not any(
        value in json.dumps(result)
        for value in ["private", "secret", "kwargs", "owned-dev-pod"]
    )


def test_other_node_response_is_unavailable() -> None:
    result = cutover.summarize_markdown_worker("different-dev-pod", 23, replies())
    assert result["status"] == "unavailable"
    assert result["registered_response"] is False


def test_missing_registration_queue_and_pid_are_visible() -> None:
    data = replies()
    node = "user_file_processing@owned-dev-pod"
    data["registered"][node] = ["process_single_user_file"]
    data["active_queues"][node] = [{"name": "user_file_processing"}]
    result = cutover.summarize_markdown_worker(node, 99, data)
    assert result["index_registered"] is False
    assert result["queues_match"] is False
    assert result["stats_pid_matches"] is False


def test_oversized_worker_reply_is_refused() -> None:
    data = replies()
    node = "user_file_processing@owned-dev-pod"
    data["active"][node] = [{}] * 1001
    with pytest.raises(ValueError):
        cutover.summarize_markdown_worker(node, 23, data)


def test_inspection_targets_only_local_node(monkeypatch: pytest.MonkeyPatch) -> None:
    from onyx.background.celery.versioned_apps.client import app

    data = replies()
    inspector = Mock(**{name + ".return_value": value for name, value in data.items()})
    inspect_call = Mock(return_value=inspector)
    status = Mock(
        return_value=SimpleNamespace(
            stdout="celery_worker_user_file_processing RUNNING pid 23, uptime 0:04:00\n"
        )
    )
    monkeypatch.setattr(socket, "gethostname", lambda: "owned-dev-pod")
    monkeypatch.setattr(app.control, "inspect", inspect_call)
    monkeypatch.setattr(subprocess, "run", status)
    result = cutover.read_markdown_worker()
    inspect_call.assert_called_once_with(
        timeout=3, destination=["user_file_processing@owned-dev-pod"]
    )
    assert status.call_args.args[0][1:3] == [
        "-c",
        "/etc/supervisor/conf.d/supervisord.conf",
    ]
    assert status.call_args.kwargs["timeout"] == 3
    assert result["stats_pid_matches"] is True
    cutover.validate_markdown_worker_report({**result, "database_read_only": True})


def test_embedded_program_refuses_wrong_environment_without_services(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POSTGRES_DB", "wrong-environment")

    def command(args: list[str], **_kwargs: Any) -> str:
        assert (
            args[-3]
            == '. /vault/secrets/config; export PGOPTIONS="-c default_transaction_read_only=on"; exec python -c "$1"'
        )
        return subprocess.run(
            [sys.executable, "-c", args[-1]],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout

    driver = Mock(sha=cutover.MARKDOWN_WORKER_RUNTIME, command=command)
    cutover.diagnose_markdown_worker(driver, "owned-pod", "backend")
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "stage": "markdown_worker",
        "status": "failed",
        "database_read_only": True,
        "failure_type": "ValueError",
    }


@pytest.mark.parametrize(
    "extra",
    [{"raw_error": "private"}, {"active_count": -1}, {"index_registered": "private"}],
)
def test_runner_refuses_unsafe_worker_report(extra: dict[str, Any]) -> None:
    with pytest.raises(cutover.CutoverRefusal):
        cutover.validate_markdown_worker_report(
            {
                "stage": "markdown_worker",
                "status": "read",
                "database_read_only": True,
                **extra,
            }
        )
