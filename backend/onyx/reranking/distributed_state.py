from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable

from onyx.cache.factory import get_cache_backend
from onyx.cache.interface import CacheBackend

RERANKER_TEST_ATTESTATION_TTL_SECONDS = 300
_ATTESTATION_KEY_PREFIX = "reranker_test_attestation"
_ATTESTATION_LOCK_PREFIX = "reranker_test_attestation_lock"
_LOCK_TTL_SECONDS = 10
_LOCK_WAIT_SECONDS = 5

CacheFactory = Callable[[str], CacheBackend]


def _default_cache_factory(tenant_id: str) -> CacheBackend:
    return get_cache_backend(tenant_id=tenant_id)


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _attestation_value(*, config_fingerprint: str, config_generation: str) -> bytes:
    return f"{config_generation}:{config_fingerprint}".encode("utf-8")


class DistributedRerankerAttestationStore:
    """Tenant-scoped, shared, short-lived reranker test attestations."""

    def __init__(
        self,
        *,
        cache_factory: CacheFactory = _default_cache_factory,
        attestation_ttl_seconds: int = RERANKER_TEST_ATTESTATION_TTL_SECONDS,
    ) -> None:
        self._cache_factory = cache_factory
        self._attestation_ttl_seconds = attestation_ttl_seconds

    def issue(
        self,
        *,
        tenant_id: str,
        config_fingerprint: str,
        config_generation: str,
    ) -> str:
        token = secrets.token_urlsafe(32)
        digest = _token_digest(token)
        self._cache_factory(tenant_id).set(
            f"{_ATTESTATION_KEY_PREFIX}:{digest}",
            _attestation_value(
                config_fingerprint=config_fingerprint,
                config_generation=config_generation,
            ),
            ex=self._attestation_ttl_seconds,
        )
        return token

    def consume(
        self,
        *,
        tenant_id: str,
        token: str | None,
        config_fingerprint: str,
        config_generation: str,
    ) -> bool:
        if token is None:
            return False
        digest = _token_digest(token)
        cache = self._cache_factory(tenant_id)
        key = f"{_ATTESTATION_KEY_PREFIX}:{digest}"
        expected = _attestation_value(
            config_fingerprint=config_fingerprint,
            config_generation=config_generation,
        )
        lock = cache.lock(
            f"{_ATTESTATION_LOCK_PREFIX}:{digest}", timeout=_LOCK_TTL_SECONDS
        )
        acquired = lock.acquire(blocking=True, blocking_timeout=_LOCK_WAIT_SECONDS)
        if not acquired:
            return False
        try:
            stored = cache.get(key)
            if stored is None or not hmac.compare_digest(stored, expected):
                return False
            cache.delete(key)
            return True
        finally:
            if lock.owned():
                lock.release()


distributed_reranker_attestations = DistributedRerankerAttestationStore()
