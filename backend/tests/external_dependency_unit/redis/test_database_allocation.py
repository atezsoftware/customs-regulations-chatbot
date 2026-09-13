import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from redis import ConnectionPool, Redis


def test_allocation_claim_is_atomic_and_rejects_foreign_data() -> None:
    from scripts.regulatory_redis_isolation import claim_database

    with Redis(host="127.0.0.1", port=19379, db=4) as client:
        key = "owned-allocation-test:" + uuid4().hex
        try:
            client.set(key, "foreign")
            with pytest.raises(RuntimeError, match="not_empty"):
                claim_database(client, "owned-dev")
            assert client.get(key) == b"foreign"
        finally:
            client.delete(key)
        try:
            claim_database(client, "owned-dev")
            claim_database(client, "owned-dev")
            with pytest.raises(RuntimeError, match="owner_mismatch"):
                claim_database(client, "other-deployment")
        finally:
            client.delete("onyx:deployment-owner")


@pytest.fixture
def allocation() -> Iterator[None]:
    from scripts import regulatory_redis_isolation as isolation

    from onyx.configs import app_configs as config
    from onyx.redis.redis_pool import RedisPool

    def local_pool(*, db: int, ssl: bool, max_connections: int) -> ConnectionPool:
        assert ssl is False
        return ConnectionPool(
            host="127.0.0.1", port=19379, db=db, max_connections=max_connections
        )

    with (
        patch.object(isolation, "verify_transports"),
        patch.object(RedisPool, "create_pool", side_effect=local_pool),
        patch.object(config, "POSTGRES_DB", "customs-regulations-dev"),
        patch.object(config, "REGULATORY_ANNEX_ENVIRONMENT", "dev"),
        patch.object(config, "REDIS_SSL", False),
        patch.object(config, "REDIS_DB_NUMBER", 0),
        patch.object(config, "REDIS_DB_NUMBER_CELERY", 15),
        patch.object(config, "REDIS_DB_NUMBER_CELERY_RESULT_BACKEND", 14),
    ):
        try:
            yield
        finally:
            for database in (4, 5, 6):
                with Redis(host="127.0.0.1", port=19379, db=database) as client:
                    client.delete("onyx:deployment-owner")


@pytest.mark.usefixtures("allocation")
def test_reservation_rechecks_active_clients_before_claiming() -> None:
    from scripts.regulatory_redis_isolation import run

    with Redis(host="127.0.0.1", port=19379, db=5) as foreign:
        foreign.ping()
        with pytest.raises(RuntimeError, match="database_has_clients"):
            run("reserve")
        with Redis(host="127.0.0.1", port=19379, db=4) as first:
            assert first.dbsize() == 0


@pytest.mark.usefixtures("allocation")
def test_reserved_databases_require_matching_runtime_and_owner() -> None:
    from scripts.regulatory_redis_isolation import run

    from onyx.configs import app_configs as config

    assert run("reserve") == {
        "redis_isolation": "reserved",
        "migration_required": True,
    }
    assert run("reserve")["migration_required"] is True
    with pytest.raises(RuntimeError, match="runtime_databases_mismatch"):
        run("verify")
    with (
        patch.object(config, "REDIS_DB_NUMBER", 4),
        patch.object(config, "REDIS_DB_NUMBER_CELERY", 5),
        patch.object(config, "REDIS_DB_NUMBER_CELERY_RESULT_BACKEND", 6),
    ):
        assert run("verify") == {
            "redis_isolation": "verified",
            "migration_required": False,
        }
        with Redis(host="127.0.0.1", port=19379, db=6) as client:
            client.set("onyx:deployment-owner", "other-deployment")
        with pytest.raises(RuntimeError, match="owner_mismatch"):
            run("reserve")


@pytest.mark.usefixtures("allocation")
def test_other_environment_refused_before_redis_connection() -> None:
    from scripts.regulatory_redis_isolation import run

    from onyx.configs import app_configs as config
    from onyx.redis.redis_pool import RedisPool

    with (
        patch.object(config, "POSTGRES_DB", "other-environment"),
        patch.object(RedisPool, "create_pool") as connect,
    ):
        with pytest.raises(RuntimeError, match="DEV_environment_required"):
            run("reserve")
        connect.assert_not_called()


def test_injected_cli_reserves_and_verifies_effective_production_clients() -> None:
    backend = Path(__file__).resolve().parents[3]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("REDIS_", "POSTGRES_", "CELERY_"))
    }
    environment.update(
        PYTHONPATH=str(backend),
        POSTGRES_DB="customs-regulations-dev",
        POSTGRES_HOST="127.0.0.1",
        REDIS_HOST="127.0.0.1",
        REDIS_REPLICA_HOST="127.0.0.1",
        REDIS_PORT="19379",
        REDIS_PASSWORD="",
        REDIS_SSL="false",
        USE_REDIS_IAM_AUTH="false",
        REGULATORY_ANNEX_ENVIRONMENT="dev",
        MULTI_TENANT="false",
    )
    source = (backend / "scripts/regulatory_redis_isolation.py").read_text()
    try:
        for mode in ("reserve", "verify"):
            if mode == "verify":
                environment["REDIS_DEPLOYMENT_DATABASES"] = "4,5,6"
            result = subprocess.run(
                [sys.executable, "-", mode],
                input=source,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            assert len(result.stdout.splitlines()) == 1
            assert json.loads(result.stdout) == {
                "redis_isolation": "reserved" if mode == "reserve" else "verified",
                "migration_required": mode == "reserve",
            }
    finally:
        for database in (4, 5, 6):
            with Redis(host="127.0.0.1", port=19379, db=database) as client:
                client.delete("onyx:deployment-owner")
