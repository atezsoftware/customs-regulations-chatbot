"""Uses the task-owned local Redis on port19379; never a configured remote service."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from redis import Redis

PROGRAM = r"""
import asyncio, json, os
from onyx.configs import app_configs as settings
from onyx.background.celery.versioned_apps.client import app
from onyx.background.celery.celery_redis import celery_get_broker_client, celery_get_queue_length
from onyx.redis.redis_pool import get_redis_client, get_raw_redis_client, get_async_redis_connection

key = os.environ["ISOLATION_TEST_KEY"]
value = os.environ["ISOLATION_TEST_VALUE"]
client = get_redis_client(tenant_id="public")
raw = get_raw_redis_client()
broker = celery_get_broker_client(app)
async def async_value():
    connection = await get_async_redis_connection()
    return await connection.get("public:" + key)

before = client.get(key)
result_before = app.backend.get_task_meta(key)["status"]
queue_before = celery_get_queue_length(key, broker)
lock = client.lock(key + ":lock", timeout=60)
locked = lock.acquire(blocking=False)
if os.environ["ISOLATION_TEST_MODE"] == "write":
    client.set(key, value, ex=60)
    app.backend.store_result(key, value, "SUCCESS")
    app.send_task("isolation_fixture", args=[value], task_id=key, queue=key, expires=30)
current = client.get(key)
raw_value = raw.get("public:" + key)
async_result = asyncio.run(async_value())
with app.connection_for_read() as connection:
    with connection.channel() as channel:
        message = channel.basic_get(key)
        consumed = message.payload[0][0] if message else None
        if message:
            message.reject(requeue=True)
        fanout_prefix = channel.keyprefix_fanout
print(json.dumps({
    "databases": [settings.REDIS_DB_NUMBER, settings.REDIS_DB_NUMBER_CELERY, settings.REDIS_DB_NUMBER_CELERY_RESULT_BACKEND],
    "before": before.decode() if before else None,
    "result_before": result_before,
    "queue_before": queue_before,
    "lock_acquired": locked,
    "current": current.decode() if current else None,
    "raw": raw_value.decode() if raw_value else None,
    "async": async_result.decode() if async_result else None,
    "result": app.backend.get_task_meta(key).get("result"),
    "queue": celery_get_queue_length(key, broker),
    "consumed": consumed,
    "fanout_prefix": fanout_prefix,
}))
"""


def process(
    key: str, value: str, *, isolated: bool, mode: str = "write"
) -> dict[str, object]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("REDIS_", "POSTGRES_"))
    }
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
            "REDIS_HOST": "127.0.0.1",
            "REDIS_REPLICA_HOST": "127.0.0.1",
            "REDIS_PORT": "19379",
            "REDIS_PASSWORD": "",
            "REDIS_SSL": "false",
            "USE_REDIS_IAM_AUTH": "false",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_DB": "owned_local_isolation",
            "MULTI_TENANT": "false",
            "ISOLATION_TEST_KEY": key,
            "ISOLATION_TEST_VALUE": value,
            "ISOLATION_TEST_MODE": mode,
        }
    )
    if isolated:
        environment["REDIS_DEPLOYMENT_DATABASES"] = "1,2,3"
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_deployments_isolate_broker_results_locks_and_raw_async_clients() -> None:
    key = "deployment_isolation_" + uuid4().hex
    try:
        first = process(key, "DEV", isolated=True)
        second = process(key, "OTHER", isolated=False)
        repeated = process(key, "DEV", isolated=True, mode="read")
        assert first["databases"] == [1, 2, 3]
        assert second["databases"] == [0, 15, 14]
        assert second["before"] is None
        assert second["result_before"] == "PENDING"
        assert second["queue_before"] == 0
        assert first["lock_acquired"] is True
        assert second["lock_acquired"] is True
        assert repeated["lock_acquired"] is False
        assert repeated["queue"] == second["queue"] == 1
        assert repeated["current"] == repeated["raw"] == repeated["async"] == "DEV"
        assert repeated["result"] == "DEV"
        assert second["result"] == "OTHER"
        assert first["consumed"] == repeated["consumed"] == "DEV"
        assert second["consumed"] == "OTHER"
        assert first["fanout_prefix"] == "/2."
        assert second["fanout_prefix"] == "/15."
    finally:
        for database in (0, 1, 2, 3, 14, 15):
            with Redis(host="127.0.0.1", port=19379, db=database) as client:
                keys = list(client.scan_iter(match="*" + key + "*"))
                if keys:
                    client.delete(*keys)


@pytest.mark.parametrize("value", ["1,2", "1,1,2", "1,-2,3"])
def test_invalid_deployment_database_selection_fails_before_use(value: str) -> None:
    environment = dict(os.environ, REDIS_DEPLOYMENT_DATABASES=value)
    result = subprocess.run(
        [sys.executable, "-c", "from onyx.configs import app_configs"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "REDIS_DEPLOYMENT_DATABASES" in result.stderr


def test_result_notifications_do_not_cross_deployments_with_same_task_id() -> None:
    key = "deployment_isolation_" + uuid4().hex
    try:
        with Redis(host="127.0.0.1", port=19379).pubsub() as listener:
            listener.psubscribe("*" + key)
            assert listener.get_message(timeout=2)["type"] == "psubscribe"
            process(key, "DEV", isolated=True)
            first = listener.get_message(timeout=2)
            process(key, "OTHER", isolated=False)
            second = listener.get_message(timeout=2)
            assert first is not None and second is not None
            assert json.loads(first["data"])["result"] == "DEV"
            assert json.loads(second["data"])["result"] == "OTHER"
            assert first["channel"] != second["channel"]
    finally:
        for database in (0, 1, 2, 3, 14, 15):
            with Redis(host="127.0.0.1", port=19379, db=database) as client:
                keys = list(client.scan_iter(match="*" + key + "*"))
                if keys:
                    client.delete(*keys)
