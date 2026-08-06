from __future__ import annotations

from onyx.reranking.distributed_state import DistributedRerankerAttestationStore
from tests.unit.fakes import FakeCache


def _store(cache: FakeCache) -> DistributedRerankerAttestationStore:
    return DistributedRerankerAttestationStore(
        cache_factory=lambda _tenant_id: cache,
        attestation_ttl_seconds=30,
    )


def test_attestation_issued_by_one_instance_is_atomically_consumed_by_another() -> None:
    cache = FakeCache()
    worker_a = _store(cache)
    worker_b = _store(cache)

    token = worker_a.issue(
        tenant_id="tenant-a",
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )

    assert worker_b.consume(
        tenant_id="tenant-a",
        token=token,
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )
    assert not worker_a.consume(
        tenant_id="tenant-a",
        token=token,
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )


def test_attestation_is_bound_to_tenant_fingerprint_and_generation() -> None:
    caches = {"tenant-a": FakeCache(), "tenant-b": FakeCache()}
    worker_a = DistributedRerankerAttestationStore(
        cache_factory=lambda tenant_id: caches[tenant_id]
    )
    worker_b = DistributedRerankerAttestationStore(
        cache_factory=lambda tenant_id: caches[tenant_id]
    )
    token = worker_a.issue(
        tenant_id="tenant-a",
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )

    assert not worker_b.consume(
        tenant_id="tenant-b",
        token=token,
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )
    assert not worker_b.consume(
        tenant_id="tenant-a",
        token=token,
        config_fingerprint="fingerprint-b",
        config_generation="generation-a",
    )
    assert not worker_a.consume(
        tenant_id="tenant-a",
        token=token,
        config_fingerprint="fingerprint-a",
        config_generation="generation-b",
    )


def test_attestation_has_bounded_ttl_and_token_is_not_used_as_cache_key() -> None:
    cache = FakeCache()
    store = _store(cache)

    token = store.issue(
        tenant_id="tenant-a",
        config_fingerprint="fingerprint-a",
        config_generation="generation-a",
    )

    assert token not in cache.store
    assert list(cache.expiries.values()) == [30]
