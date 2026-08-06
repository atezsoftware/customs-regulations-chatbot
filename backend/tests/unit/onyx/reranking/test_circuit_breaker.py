from onyx.reranking.circuit_breaker import (
    RerankCircuitBreaker,
    reranker_configuration_fingerprint,
)
from onyx.reranking.models import RerankCircuitKey


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_circuit_state_is_scoped_by_tenant_and_configuration_fingerprint() -> None:
    clock = Clock()
    circuit = RerankCircuitBreaker(failure_threshold=1, clock=clock)
    key = RerankCircuitKey(tenant_id="tenant-a", config_fingerprint="fp-a")

    circuit.record_failure(key)

    assert circuit.is_open(key) is True
    assert (
        circuit.is_open(
            RerankCircuitKey(tenant_id="tenant-b", config_fingerprint="fp-a")
        )
        is False
    )
    assert (
        circuit.is_open(
            RerankCircuitKey(tenant_id="tenant-a", config_fingerprint="fp-b")
        )
        is False
    )


def test_transient_failures_open_only_at_threshold_and_success_resets() -> None:
    clock = Clock()
    circuit = RerankCircuitBreaker(failure_threshold=2, clock=clock)
    key = RerankCircuitKey(tenant_id="tenant", config_fingerprint="fp")

    circuit.record_failure(key)
    assert circuit.is_open(key) is False
    circuit.record_success(key)
    circuit.record_failure(key)
    assert circuit.is_open(key) is False
    circuit.record_failure(key)
    assert circuit.is_open(key) is True


def test_retry_after_is_capped_and_circuit_closes_after_cooldown() -> None:
    clock = Clock()
    circuit = RerankCircuitBreaker(
        failure_threshold=3,
        default_cooldown_seconds=30,
        max_cooldown_seconds=120,
        clock=clock,
    )
    key = RerankCircuitKey(tenant_id="tenant", config_fingerprint="fp")

    circuit.record_failure(key, retry_after_seconds=9_000, immediate=True)
    clock.now += 119
    assert circuit.is_open(key) is True
    clock.now += 1
    assert circuit.is_open(key) is False


def test_invalidate_can_clear_one_configuration_or_whole_tenant() -> None:
    circuit = RerankCircuitBreaker(failure_threshold=1)
    first = RerankCircuitKey(tenant_id="tenant", config_fingerprint="first")
    second = RerankCircuitKey(tenant_id="tenant", config_fingerprint="second")
    other = RerankCircuitKey(tenant_id="other", config_fingerprint="first")
    for key in (first, second, other):
        circuit.record_failure(key)

    circuit.invalidate(tenant_id="tenant", config_fingerprint="first")
    assert circuit.is_open(first) is False
    assert circuit.is_open(second) is True
    circuit.invalidate(tenant_id="tenant")
    assert circuit.is_open(second) is False
    assert circuit.is_open(other) is True


def test_configuration_fingerprint_is_stable_and_secret_free() -> None:
    first = reranker_configuration_fingerprint(model="model", api_key="top-secret")
    second = reranker_configuration_fingerprint(model="model", api_key="top-secret")

    assert first == second
    assert "model" not in first
    assert "top-secret" not in first
