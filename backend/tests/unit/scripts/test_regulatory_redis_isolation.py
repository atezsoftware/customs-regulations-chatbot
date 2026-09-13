from unittest.mock import Mock, patch

import pytest
from scripts import regulatory_redis_isolation as isolation


@pytest.mark.parametrize("failure", [None, "pid", "handlers", "queues", "foreign"])
def test_worker_admission_requires_local_process_and_isolated_broker(
    failure: str | None,
) -> None:
    from onyx.background.celery.versioned_apps.client import app

    node = "user_file_processing@owned-dev-pod"
    inspector = Mock()
    inspector.stats.return_value = {node: {"pid": 42 if failure != "pid" else 43}}
    inspector.registered.return_value = {
        node: []
        if failure == "handlers"
        else ["process_single_user_file", "index_single_user_file"]
    }
    inspector.active_queues.return_value = {
        node: []
        if failure == "queues"
        else [
            {"name": name}
            for name in (
                "user_file_processing",
                "user_file_project_sync",
                "user_file_delete",
                "user_file_port",
            )
        ]
    }
    inspector.ping.return_value = {node: {"ok": "pong"}}
    if failure == "foreign":
        inspector.ping.return_value["user_file_processing@foreign-pod"] = {"ok": "pong"}
    with (
        patch.object(app.control, "inspect", return_value=inspector),
        patch.object(isolation.socket, "gethostname", return_value="owned-dev-pod"),
        patch.object(
            isolation.subprocess,
            "run",
            return_value=Mock(
                returncode=0,
                stdout="celery_worker_user_file_processing RUNNING pid 42, uptime 1:00\n",
            ),
        ),
    ):
        if failure is None:
            isolation.verify_worker()
        else:
            with pytest.raises(isolation.IsolationRefusal):
                isolation.verify_worker()
