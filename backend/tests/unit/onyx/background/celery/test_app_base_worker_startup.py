from typing import Any

import pytest

from onyx.background.celery.apps import app_base
from onyx.configs.constants import OnyxRedisLocks


def test_secondary_worker_skips_primary_wait_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_base, "CELERY_PRIMARY_WORKER_REQUIRED", False)

    def unexpected_redis_client(**kwargs: Any) -> None:
        raise AssertionError(f"Redis must not be queried when the wait is disabled: {kwargs}")

    monkeypatch.setattr(app_base, "get_redis_client", unexpected_redis_client)

    app_base.on_secondary_worker_init(sender=None)


def test_secondary_worker_checks_primary_lock_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_keys: list[str] = []

    class ReadyRedis:
        def exists(self, key: str) -> bool:
            checked_keys.append(key)
            return True

    monkeypatch.setattr(app_base, "CELERY_PRIMARY_WORKER_REQUIRED", True)
    monkeypatch.setattr(app_base, "get_redis_client", lambda **_kwargs: ReadyRedis())

    app_base.on_secondary_worker_init(sender=None)

    assert checked_keys == [OnyxRedisLocks.PRIMARY_WORKER]
