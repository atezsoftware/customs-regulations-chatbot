"""Concurrent costs must preserve an actual final-generation allowance."""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_cost_budget as cost_module
from onyx.asv3.cropped_cost_budget import (
    CroppedCostBudget,
    CroppedRepairBudgetReached,
    CroppedResearchBudgetReached,
    cropped_cache_write_partition_verified,
)
from onyx.asv3.models import RunContext, RunStopped
from onyx.llm.model_response import Usage


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> CroppedCostBudget:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001, 0.0000025)
    )
    return CroppedCostBudget(RunContext(scope={"document_set_id": 15}), "same request")


def state(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = payload["state"]
    assert isinstance(value, dict)
    return value


def usage(
    incoming: int = 100000,
    outgoing: int = 1000,
    cached: int = 0,
    written: int = 0,
) -> Usage:
    return Usage(
        prompt_tokens=incoming,
        completion_tokens=outgoing,
        total_tokens=incoming + outgoing,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
    )


def test_parallel_reservations_count_before_ten_dollar_research_gate(
    budget: CroppedCostBudget,
) -> None:
    budget.record_external_charge(
        "ancillary", known_cost_usd=9.2, uncertain_upper_usd=0.0, context=budget.context
    )

    def prepare(owner: str):
        try:
            return budget.prepare(
                "sol", "openai", 100000, 16384, research=True, owner=owner
            )
        except CroppedResearchBudgetReached:
            return None

    with ThreadPoolExecutor(max_workers=3) as executor:
        tickets = list(executor.map(prepare, ["root", "worker", "third"]))
    assert sum(ticket is not None for ticket in tickets) == 2
    assert budget.research_allowed is False
    for ticket in tickets:
        if ticket is not None:
            budget.complete(ticket, usage())
    final = budget.prepare(
        "sol", "openai", 100000, 128000, research=False, owner="root"
    )
    assert final.phase == "finalization" and final.max_tokens == 128000


def test_uncached_reservation_releases_verified_cache_savings(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "sol", "openai", 100000, 128000, research=True, owner="root"
    )
    budget.complete(ticket, usage(cached=90000))
    assert state(budget.export())["spent_usd"] == pytest.approx(0.039)


def test_unknown_provider_result_keeps_reserve_across_resume(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "sol", "openai", 100000, 128000, research=True, owner="root"
    )
    checkpoint = budget.export()
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(checkpoint)
    assert state(clone.export())["uncertain_usd"] == ticket.reserved_usd
    budget.complete(ticket, None, TimeoutError())
    assert budget.finish_reason == "unpriced_or_inconsistent_provider_usage"


def test_known_provider_rejection_releases_reserve(budget: CroppedCostBudget) -> None:
    class Rejected(Exception):
        status_code = 400

    ticket = budget.prepare(
        "sol", "openai", 100000, 128000, research=True, owner="root"
    )
    budget.complete(ticket, None, Rejected())
    assert state(budget.export())["uncertain_usd"] == 0
    assert budget.finish_reason is None


def test_bad_reported_usage_does_not_zero_cost(budget: CroppedCostBudget) -> None:
    ticket = budget.prepare("sol", "openai", 100, 1000, research=True, owner="root")
    budget.complete(ticket, usage(incoming=ticket.input_upper_tokens + 1))
    assert state(budget.export())["uncertain_usd"] == ticket.reserved_usd


def test_wrong_request_and_changed_checkpoint_rejected(
    budget: CroppedCostBudget,
) -> None:
    checkpoint = budget.export()
    with pytest.raises(ValueError, match="ownership"):
        CroppedCostBudget(budget.context, "other request").restore(checkpoint)
    changed = copy.deepcopy(checkpoint)
    state(changed)["spent_usd"] = 0.1
    with pytest.raises(ValueError, match="changed"):
        budget.restore(changed)


def test_first_finalization_after_ten_dollars_is_not_repair(
    budget: CroppedCostBudget,
) -> None:
    budget.record_external_charge(
        "prior", known_cost_usd=10.0, uncertain_upper_usd=0.25, context=budget.context
    )
    assert not budget.research_allowed and not budget.repair_allowed
    with pytest.raises(CroppedResearchBudgetReached):
        budget.prepare("sol", "openai", 1000, 1000, research=True, owner="root")
    with pytest.raises(CroppedRepairBudgetReached):
        budget.prepare(
            "sol", "openai", 1000, 1000, research=False, owner="root", phase="repair"
        )
    ticket = budget.prepare(
        "sol", "openai", 100000, 128000, research=False, owner="root"
    )
    assert ticket.phase == "finalization" and ticket.max_tokens == 128000
    budget.complete(ticket, usage())
    assert state(budget.export())["spent_usd"] == pytest.approx(0.21)


def test_repair_ten_dollar_gate_includes_parallel_reservations(
    budget: CroppedCostBudget,
) -> None:
    budget.record_external_charge(
        "observed-searches",
        known_cost_usd=9.9,
        uncertain_upper_usd=0.0,
        context=budget.context,
    )
    ticket = budget.prepare(
        "sol", "openai", 1000, 8192, research=False, owner="worker", phase="repair"
    )
    assert ticket.phase == "repair" and ticket.max_tokens == 8192
    with pytest.raises(CroppedRepairBudgetReached):
        budget.prepare(
            "sol", "openai", 1000, 8192, research=False, owner="root", phase="repair"
        )
    assert budget.finish_reason is None
    budget.complete(ticket, usage(incoming=1000, outgoing=100))
    assert budget.repair_allowed and budget.research_allowed


def test_missing_price_hands_research_to_unpriced_first_writing(
    budget: CroppedCostBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_: object) -> tuple[float, float, float]:
        raise RunStopped("Registry price missing")

    monkeypatch.setattr(cost_module, "_rates", missing)
    with pytest.raises(CroppedResearchBudgetReached, match="pricing is unavailable"):
        budget.prepare("unknown", "provider", 1000, 128000, research=True, owner="root")
    first = budget.prepare(
        "unknown", "provider", 1000, 128000, research=False, owner="root"
    )
    assert first.max_tokens == 128000 and not first.price_available
    assert not first.reservation_upper_verified
    assert budget.state_instructions()["recorded_cost_upper_usd"] is None
    budget.complete(first, usage(incoming=1000, outgoing=100))
    row = cast_call(state(budget.export())["calls"])
    assert row["known_cost_usd"] is None
    assert row["retained_reserve_usd"] is None
    assert state(budget.export())["unpriced_call_ids"] == [first.call_id]
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    with pytest.raises(CroppedRepairBudgetReached):
        clone.prepare(
            "unknown",
            "provider",
            1000,
            1000,
            research=False,
            owner="root",
            phase="repair",
        )
    second = clone.prepare(
        "unknown", "provider", 1000, 128000, research=False, owner="root"
    )
    assert second.max_tokens == 128000


def cast_call(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, list) and value and isinstance(value[0], dict)
    return value[0]


def test_external_charge_identity_fence_and_checkpoint_integrity(
    budget: CroppedCostBudget,
) -> None:
    for _ in range(2):
        budget.record_external_charge(
            "same-receipt",
            known_cost_usd=0.1,
            uncertain_upper_usd=0.02,
            context=budget.context,
        )
    assert budget.state_instructions()["recorded_cost_upper_usd"] == pytest.approx(0.12)
    with pytest.raises(ValueError, match="identity changed"):
        budget.record_external_charge(
            "same-receipt",
            known_cost_usd=0.2,
            uncertain_upper_usd=0.02,
            context=budget.context,
        )
    with pytest.raises(ValueError, match="run or scope"):
        budget.record_external_charge(
            "foreign",
            known_cost_usd=0.1,
            uncertain_upper_usd=0.0,
            context=RunContext(scope={"document_set_id": 16}),
        )
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.state_instructions()["recorded_cost_upper_usd"] == pytest.approx(0.12)
    invalid = budget.export()
    invalid["version"] = True
    with pytest.raises(ValueError, match="version"):
        clone.restore(invalid)


def test_absent_requested_output_uses_actual_configured_capacity(
    budget: CroppedCostBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm, "get_model_info", lambda **_: {"max_output_tokens": 128000}
    )
    ticket = budget.prepare("sol", "openai", 1000, None, research=True, owner="root")
    assert ticket.max_tokens == 128000 and ticket.output_bound_known


def test_unverified_write_partition_keeps_explicit_exposure_across_restore(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare("sol", "openai", 10000, 1024, research=True, owner="root")
    budget.complete(
        ticket,
        usage(incoming=10000, outgoing=100, cached=1000),
        cache_write_partition_verified=False,
    )
    assert state(budget.export())["uncertain_usd"] == pytest.approx(9000 * 0.0000005)
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


@pytest.mark.parametrize("written", [0, 10])
def test_actual_raw_cache_write_counter_proves_partition(written: int) -> None:
    assert cropped_cache_write_partition_verified(
        {
            "usage_status": "provider_reported",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 20,
                    "cache_write_tokens": written,
                },
            },
        }
    )


@pytest.mark.parametrize(
    "terminal",
    [
        None,
        {
            "usage_status": "provider_reported",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
        {
            "usage_status": "invalid",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cache_write_tokens": 0},
            },
        },
        {
            "usage_status": "provider_reported",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cache_write_tokens": True},
            },
        },
        {
            "usage_status": "provider_reported",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 1},
            },
        },
        {
            "usage_status": "provider_reported",
            "provider_reported_usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cache_write_tokens": -1},
            },
        },
    ],
)
def test_missing_or_invalid_raw_partition_is_not_normalized_zero_proof(
    terminal: object,
) -> None:
    assert not cropped_cache_write_partition_verified(terminal)


def test_bound_zero_write_contract_does_not_add_speculative_write_exposure(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "sol",
        "openai",
        10000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    budget.complete(
        ticket,
        usage(incoming=10000, outgoing=100),
        cache_write_partition_verified=False,
    )
    assert state(budget.export())["uncertain_usd"] == 0
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


def test_input_upper_over_272k_is_refused_before_reservation(
    budget: CroppedCostBudget,
) -> None:
    with pytest.raises(RunStopped, match="272k"):
        budget.prepare("sol", "openai", 244000, 1000, research=True, owner="root")
    assert state(budget.export())["calls"] == []


def test_optional_null_price_tiers_retain_available_base_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda **_: {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.0000001,
            "cache_creation_input_token_cost": 0.0000025,
            "input_cost_per_token_above_128k_tokens": None,
            "output_cost_per_token_above_200k_tokens": None,
            "input_cost_per_token_above_272k_tokens": 0.000004,
        },
    )
    assert cost_module._rates("sol", "openai", 240000) == (
        0.000002,
        0.00001,
        0.0000001,
        0.0000025,
    )
    assert cost_module._rates("sol", "openai", 273000)[0] == 0.000004


def test_ordinary_healthy_calls_keep_native_output_and_final_flow(
    budget: CroppedCostBudget,
) -> None:
    for _ in range(6):
        ticket = budget.prepare(
            "sol",
            "openai",
            100000,
            128000,
            research=True,
            owner="root",
            enforce=False,
        )
        assert ticket.max_tokens == 128000
        assert ticket.bounded is False
        budget.complete(ticket, usage())
    assert state(budget.export())["spent_usd"] == pytest.approx(1.26)
    assert budget.finish_reason is None
    assert state(budget.export())["bounded"] is False
    ticket = budget.prepare(
        "sol", "openai", 100000, 128000, research=True, owner="root", enforce=True
    )
    assert ticket.max_tokens == 128000
    assert budget.state_instructions()["target_is_hard_cap"] is False
    assert budget.research_allowed


def test_activation_carries_prior_spending_and_inflight_reservations(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "sol",
        "openai",
        100000,
        128000,
        research=True,
        owner="worker",
        enforce=False,
    )
    checkpoint = budget.export()
    assert state(checkpoint)["uncertain_usd"] == ticket.reserved_usd
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(checkpoint)
    assert state(clone.export())["bounded"] is False
    assert clone.state_instructions()["recorded_cost_upper_usd"] == pytest.approx(
        ticket.reserved_usd
    )
    second = budget.prepare(
        "sol", "openai", 100000, 128000, research=True, owner="root", enforce=True
    )
    assert second.reserved_usd > 0
    budget.complete(ticket, usage())
    assert state(budget.export())["spent_usd"] == pytest.approx(0.21)
    assert state(budget.export())["uncertain_usd"] == second.reserved_usd
    final = budget.prepare(
        "sol",
        "openai",
        100000,
        128000,
        research=False,
        owner="root",
        enforce=True,
    )
    assert final.bounded is True
    assert final.max_tokens == 128000


def test_unknown_price_does_not_interrupt_an_ordinary_run(
    budget: CroppedCostBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unknown(*_args: object) -> tuple[float, float, float]:
        raise RunStopped("Registry price missing")

    monkeypatch.setattr(cost_module, "_rates", unknown)
    ticket = budget.prepare(
        "unknown",
        "provider",
        100000,
        128000,
        research=True,
        owner="root",
        enforce=False,
    )
    assert ticket.max_tokens == 128000
    assert ticket.price_available is False
    # An interrupted unknown-price call remains unknown across restore.
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert state(clone.export())["unpriced_call_ids"] == [ticket.call_id]
    budget.complete(ticket, usage())
    assert not budget.research_allowed
    assert state(budget.export())["unpriced_call_ids"] == [ticket.call_id]


def test_actual_first_v8_cache_write_is_priced_without_additive_input_charge(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "gpt-6.1-sol", "openai", 41529, 16384, research=True, owner="root"
    )
    assert ticket.cache_write_rate == 0.0000025
    assert ticket.reserved_usd == pytest.approx(
        ticket.input_upper_tokens * 0.0000025 + ticket.max_tokens * 0.00001
    )
    actual = usage(incoming=31493, outgoing=272, written=31490)
    actual.reasoning_tokens = 0
    budget.complete(ticket, actual)
    saved = state(budget.export())
    assert saved["spent_usd"] == pytest.approx(0.081451)
    assert saved["uncertain_usd"] == 0
    assert saved["unpriced_call_ids"] == []
    assert budget.finish_reason is None


def test_mixed_input_read_write_partitions_and_reasoning_are_counted_once(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=False, owner="root")
    actual = usage(incoming=1000, outgoing=100, cached=300, written=200)
    actual.reasoning_tokens = 75
    budget.complete(ticket, actual)
    assert state(budget.export())["spent_usd"] == pytest.approx(0.00253)
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


@pytest.mark.parametrize(
    "read_rate,write_rate", [(0.0000001, 0.0000025), (0.0000035, 0.0000025)]
)
def test_input_reservation_covers_highest_declared_partition_rate(
    budget: CroppedCostBudget,
    monkeypatch: pytest.MonkeyPatch,
    read_rate: float,
    write_rate: float,
) -> None:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, read_rate, write_rate)
    )
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=False, owner="root")
    assert ticket.reservation_upper_verified
    assert ticket.reserved_usd == pytest.approx(
        ticket.input_upper_tokens * max(0.000002, read_rate, write_rate)
        + ticket.max_tokens * 0.00001
    )
    budget.complete(ticket, usage(incoming=1000, outgoing=100, written=1000))
    spent = state(budget.export())["spent_usd"]
    assert isinstance(spent, (int, float)) and spent <= ticket.reserved_usd


@pytest.mark.parametrize(
    "incoming,cached,written",
    [(100, -1, 0), (100, 0, -1), (100, 101, 0), (100, 60, 41), (-1, 0, 0)],
)
def test_invalid_input_cache_partitions_keep_full_reserve(
    budget: CroppedCostBudget, incoming: int, cached: int, written: int
) -> None:
    ticket = budget.prepare("sol", "openai", 100, 1024, research=True, owner="root")
    actual = usage(incoming=incoming, outgoing=10, cached=cached, written=written)
    actual.cost = 0
    budget.complete(ticket, actual)
    saved = state(budget.export())
    assert saved["spent_usd"] == 0
    assert saved["uncertain_usd"] == ticket.reserved_usd
    assert budget.finish_reason == "unpriced_or_inconsistent_provider_usage"


def test_zero_reported_write_is_not_inferred_from_total_input(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    budget.complete(ticket, usage(incoming=1000, outgoing=100))
    assert state(budget.export())["spent_usd"] == pytest.approx(0.003)


def test_changed_ticket_write_rate_cannot_release_or_reduce_its_reservation(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    with pytest.raises(ValueError, match="reservation changed"):
        budget.complete(
            replace(ticket, cache_write_rate=0),
            usage(incoming=1000, outgoing=100, written=1000),
        )
    assert state(budget.export())["uncertain_usd"] == ticket.reserved_usd
    budget.complete(ticket, usage(incoming=1000, outgoing=100, written=1000))
    assert state(budget.export())["spent_usd"] == pytest.approx(0.0035)
    assert state(budget.export())["uncertain_usd"] == 0


def test_legacy_three_rates_never_assume_a_cache_write_price(
    budget: CroppedCostBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001)
    )
    with pytest.raises(RunStopped, match="unknown prior charges"):
        budget.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    assert state(budget.export())["calls"] == []
    # An ordinary completed call with explicit zero writes is still measurable.
    ordinary = CroppedCostBudget(budget.context, "same request")
    ticket = ordinary.prepare(
        "sol", "openai", 1000, 1024, research=True, owner="root", enforce=False
    )
    assert ticket.cache_write_rate is None and not ticket.reservation_upper_verified
    ordinary.complete(ticket, usage(incoming=1000, outgoing=100))
    assert state(ordinary.export())["spent_usd"] == pytest.approx(0.003)
    # A reported write cannot be priced as zero/base, even with another cost value.
    ticket = ordinary.prepare(
        "sol", "openai", 1000, 1024, research=True, owner="root", enforce=False
    )
    actual = usage(incoming=1000, outgoing=100, written=900)
    actual.cost = 0.25
    ordinary.complete(ticket, actual)
    saved = state(ordinary.export())
    assert saved["spent_usd"] == pytest.approx(0.003)
    assert saved["unpriced_call_ids"] == [ticket.call_id]
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(ordinary.export())
    assert state(clone.export())["unpriced_call_ids"] == [ticket.call_id]
    final = clone.prepare("sol", "openai", 1000, 1024, research=False, owner="root")
    assert final.phase == "finalization"
    assert clone.state_instructions()["recorded_cost_upper_usd"] is None


def test_registry_absent_cache_write_rate_remains_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda **_: {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.0000001,
        },
    )
    assert cost_module._rates("sol", "openai", 1000) == (
        0.000002,
        0.00001,
        0.0000001,
        None,
    )


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "2.5e-6"])
def test_invalid_registry_cache_write_price_is_not_assumed(
    monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda **_: {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.0000001,
            "cache_creation_input_token_cost": invalid,
        },
    )
    with pytest.raises(RunStopped, match="pricing is unavailable"):
        cost_module._rates("sol", "openai", 1000)


def test_registry_cache_write_tier_applies_at_exact_existing_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda **_: {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.0000001,
            "cache_creation_input_token_cost": 0.0000025,
            "input_cost_per_token_above_272k_tokens": 0.000004,
            "cache_creation_input_token_cost_above_272k_tokens": 0.000005,
        },
    )
    assert cost_module._rates("sol", "openai", 272000) == (
        0.000002,
        0.00001,
        0.0000001,
        0.0000025,
    )
    assert cost_module._rates("sol", "openai", 272001) == (
        0.000004,
        0.00001,
        0.0000001,
        0.000005,
    )


@pytest.mark.parametrize(
    "change", ["old_version", "write_rate", "write_count", "totals", "duplicate_call"]
)
def test_rehashed_checkpoint_cannot_change_cache_partition_pricing(
    budget: CroppedCostBudget, change: str
) -> None:
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    budget.complete(ticket, usage(incoming=1000, outgoing=100, cached=300, written=200))
    checkpoint = budget.export()
    saved = state(checkpoint)
    calls = saved["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    call = calls[0]
    if change == "old_version":
        checkpoint["version"] = 2
    elif change == "write_rate":
        call["cache_write_rate"] = None
    elif change == "write_count":
        reported = call["usage"]
        assert isinstance(reported, dict)
        reported["cache_creation_input_tokens"] = 0
    elif change == "totals":
        saved["spent_usd"] = 0
    else:
        calls.append(copy.deepcopy(call))
    checkpoint["state_hash"] = cost_module._hash(saved)
    clone = CroppedCostBudget(budget.context, "same request")
    with pytest.raises(ValueError):
        clone.restore(checkpoint)


def test_actual_v9_final_input_requires_a_proven_disabled_write_reservation(
    budget: CroppedCostBudget,
) -> None:
    prior = budget.prepare(
        "sol", "openai", 1000, 1024, research=True, owner="root", enforce=False
    )
    settled = usage(incoming=1000, outgoing=100)
    settled.cost = 0.9596472
    budget.complete(prior, settled)
    ordinary = budget.prepare(
        "sol", "openai", 129429, 128000, research=False, owner="root"
    )
    assert ordinary.max_tokens == 128000
    budget.complete(ordinary, usage(incoming=129429, outgoing=1000, cached=100000))
    ticket = budget.prepare(
        "sol",
        "openai",
        129429,
        128000,
        research=False,
        owner="root",
        cache_writes_disabled=True,
    )
    assert ticket.input_upper_tokens == 146468
    assert ticket.cache_writes_disabled and ticket.reservation_upper_verified
    assert ticket.max_tokens == 128000
    budget.complete(ticket, usage(incoming=129429, outgoing=1000, cached=100000))
    assert budget.research_allowed


@pytest.mark.parametrize("read_rate", [0.0000001, 0.0000035])
def test_disabled_writes_reserve_maximum_actual_read_or_base_rate(
    budget: CroppedCostBudget,
    monkeypatch: pytest.MonkeyPatch,
    read_rate: float,
) -> None:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, read_rate, 0.000005)
    )
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=False,
        owner="root",
        cache_writes_disabled=True,
    )
    assert ticket.reserved_usd == pytest.approx(
        ticket.input_upper_tokens * max(0.000002, read_rate) + 1024 * 0.00001
    )
    budget.complete(ticket, usage(incoming=1000, outgoing=100, cached=300))
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


def test_missing_write_price_is_allowed_only_for_the_disabled_write_contract(
    budget: CroppedCostBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001)
    )
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    assert ticket.cache_write_rate is None and ticket.reservation_upper_verified
    budget.complete(ticket, usage(incoming=1000, outgoing=100))
    assert state(budget.export())["spent_usd"] == pytest.approx(0.003)
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


def test_disabled_write_missing_usage_keeps_exact_reserve_on_resume(
    budget: CroppedCostBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001)
    )
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    interrupted = budget.export()
    assert state(interrupted)["uncertain_usd"] == ticket.reserved_usd
    assert state(interrupted)["unpriced_call_ids"] == []
    pending = state(interrupted)["interrupted_reservations"]
    assert isinstance(pending, list) and isinstance(pending[0], dict)
    assert pending[0]["cache_writes_disabled"] is True
    assert pending[0]["cache_write_rate"] is None
    assert pending[0]["input_reservation_rate"] == 0.000002
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(interrupted)
    assert state(clone.export())["uncertain_usd"] == ticket.reserved_usd
    assert clone.export() == interrupted
    budget.complete(ticket, None, TimeoutError())
    assert state(budget.export())["uncertain_usd"] == ticket.reserved_usd
    assert state(budget.export())["unpriced_call_ids"] == []
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    assert clone.export() == budget.export()


@pytest.mark.parametrize("write_rate", [0.0000025, None])
def test_disabled_write_violation_is_recorded_and_settled_before_stopping(
    budget: CroppedCostBudget,
    monkeypatch: pytest.MonkeyPatch,
    write_rate: float | None,
) -> None:
    monkeypatch.setattr(
        cost_module,
        "_rates",
        lambda *_: (0.000002, 0.00001, 0.0000001, write_rate),
    )
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    actual = usage(incoming=1000, outgoing=100, written=1000)
    with pytest.raises(RunStopped, match="violated disabled cache writes"):
        budget.complete(ticket, actual)
    saved = state(budget.export())
    assert saved["spent_usd"] == pytest.approx(0.0035 if write_rate else 0)
    assert saved["uncertain_usd"] == (0 if write_rate else ticket.reserved_usd)
    assert saved["unpriced_call_ids"] == ([] if write_rate else [ticket.call_id])
    calls = saved["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    assert calls[0]["cache_writes_disabled"] is True
    assert calls[0]["cache_write_contract_violated"] is True
    assert calls[0]["usage"] == actual.model_dump(mode="json")
    assert budget.finish_reason == "provider_violated_disabled_cache_writes"
    clone = CroppedCostBudget(budget.context, "same request")
    clone.restore(budget.export())
    with pytest.raises(RunStopped, match="violated disabled cache writes"):
        clone.prepare("sol", "openai", 1000, 1024, research=False, owner="root")


def test_changed_disabled_write_ticket_cannot_release_reservation(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    with pytest.raises(ValueError, match="reservation changed"):
        budget.complete(replace(ticket, cache_writes_disabled=True), usage(1000, 100))
    assert state(budget.export())["uncertain_usd"] == ticket.reserved_usd
    budget.complete(ticket, usage(1000, 100))


@pytest.mark.parametrize("change", ["flag", "rate", "reserve", "violation", "version"])
def test_disabled_write_checkpoint_cannot_relabel_its_reservation_proof(
    budget: CroppedCostBudget, change: str
) -> None:
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    budget.complete(ticket, usage(1000, 100))
    checkpoint = budget.export()
    saved = state(checkpoint)
    calls = saved["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    call = calls[0]
    if change == "flag":
        call["cache_writes_disabled"] = False
    elif change == "rate":
        call["input_reservation_rate"] = 0.0000025
    elif change == "reserve":
        call["reserved_usd"] = 0
    elif change == "violation":
        call["cache_write_contract_violated"] = True
    else:
        checkpoint["version"] = 3
    checkpoint["state_hash"] = cost_module._hash(saved)
    clone = CroppedCostBudget(budget.context, "same request")
    with pytest.raises(ValueError):
        clone.restore(checkpoint)


@pytest.mark.parametrize("change", ["flag", "remove", "reserve", "usage", "duplicate"])
def test_interrupted_disabled_write_checkpoint_preserves_actual_declared_contract(
    budget: CroppedCostBudget, change: str
) -> None:
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    checkpoint = budget.export()
    saved = state(checkpoint)
    rows = saved["interrupted_reservations"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    assert row["call_id"] == ticket.call_id
    if change == "flag":
        row["cache_writes_disabled"] = False
    elif change == "remove":
        rows.clear()
    elif change == "reserve":
        row["retained_reserve_usd"] = 0
    elif change == "usage":
        row["usage"] = usage(1000, 100).model_dump(mode="json")
    else:
        rows.append(copy.deepcopy(row))
    checkpoint["state_hash"] = cost_module._hash(saved)
    clone = CroppedCostBudget(budget.context, "same request")
    with pytest.raises(ValueError):
        clone.restore(checkpoint)


def test_measured_disabled_write_contract_failure_cannot_be_removed_on_resume(
    budget: CroppedCostBudget,
) -> None:
    ticket = budget.prepare(
        "sol",
        "openai",
        1000,
        1024,
        research=True,
        owner="root",
        cache_writes_disabled=True,
    )
    with pytest.raises(RunStopped, match="violated disabled cache writes"):
        budget.complete(ticket, usage(1000, 100, written=1000))
    checkpoint = budget.export()
    saved = state(checkpoint)
    saved["finish_reason"] = None
    checkpoint["state_hash"] = cost_module._hash(saved)
    with pytest.raises(ValueError, match="contract stop was removed"):
        CroppedCostBudget(budget.context, "same request").restore(checkpoint)


def legacy_budget(context: RunContext) -> CroppedCostBudget:
    legacy = CroppedCostBudget(context, "same request", repair_usd=5.0)
    ticket = legacy.prepare("sol", "openai", 1000, 1024, research=True, owner="root")
    legacy.complete(ticket, usage(1000, 100))
    legacy.record_external_charge(
        "retained-prior-charges",
        known_cost_usd=5.1,
        uncertain_upper_usd=0.02,
        context=context,
    )
    return legacy


def test_exact_legacy_repair_policy_upgrade_preserves_every_recorded_charge(
    budget: CroppedCostBudget,
) -> None:
    legacy = legacy_budget(budget.context)
    before = legacy.export()
    assert not legacy.repair_allowed
    assert budget.research_usd == budget.repair_usd == 10.0
    assert budget.total_llm_usd == 1.5
    budget.restore(before)
    after = budget.export()
    assert after["repair_usd"] == 10.0
    assert {**before, "repair_usd": 10.0} == after
    assert budget.repair_allowed and budget.research_allowed
    assert budget.state_instructions()["target_is_hard_cap"] is False
    resumed = CroppedCostBudget(budget.context, "same request")
    resumed.restore(after)
    assert resumed.export() == after


@pytest.mark.parametrize(
    "change",
    [
        "research",
        "target",
        "repair",
        "version",
        "pricing",
        "run",
        "request",
        "scope",
        "hash",
        "usage",
        "reserve",
    ],
)
def test_legacy_upgrade_does_not_relax_policy_ownership_or_cost_proofs(
    budget: CroppedCostBudget,
    change: str,
) -> None:
    saved = legacy_budget(budget.context).export()
    if change in {"research", "target", "repair"}:
        saved[
            {
                "research": "research_usd",
                "target": "total_llm_usd",
                "repair": "repair_usd",
            }[change]
        ] = {"research": 20.0, "target": 3.0, "repair": 4.0}[change]
    elif change in {"version", "pricing", "run", "request", "scope", "hash"}:
        field = {
            "version": "version",
            "pricing": "pricing_policy",
            "run": "run_id",
            "request": "request_hash",
            "scope": "scope_hash",
            "hash": "state_hash",
        }[change]
        saved[field] = 4 if change == "version" else "changed"
    else:
        saved_state = state(saved)
        calls = saved_state["calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        if change == "usage":
            supplied_usage = calls[0]["usage"]
            assert isinstance(supplied_usage, dict)
            supplied_usage["prompt_tokens"] = 999999
        else:
            calls[0]["reserved_usd"] = 0.0
        saved["state_hash"] = cost_module._hash(saved_state)
    before = budget.export()
    with pytest.raises(ValueError):
        budget.restore(saved)
    assert budget.export() == before


@pytest.mark.parametrize(
    "custom", [{"repair_usd": 9.0}, {"research_usd": 20.0}, {"total_llm_usd": 3.0}]
)
def test_legacy_upgrade_is_not_applied_to_custom_destination_policy(
    budget: CroppedCostBudget,
    custom: dict[str, float],
) -> None:
    saved = legacy_budget(budget.context).export()
    destination = CroppedCostBudget(budget.context, "same request", **custom)
    with pytest.raises(ValueError, match="policy changed"):
        destination.restore(saved)


def test_legacy_upgrade_preserves_unknown_cost_stop(
    budget: CroppedCostBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = legacy_budget(budget.context)

    def unknown(*_: object) -> tuple[float, float, float, float]:
        raise RunStopped("Unavailable pricing")

    monkeypatch.setattr(cost_module, "_rates", unknown)
    ticket = legacy.prepare(
        "unknown", "provider", 1000, 1024, research=False, owner="root"
    )
    legacy.complete(ticket, None, TimeoutError())
    saved = legacy.export()
    budget.restore(saved)
    assert state(budget.export()) == state(saved)
    assert not budget.repair_allowed and not budget.research_allowed
