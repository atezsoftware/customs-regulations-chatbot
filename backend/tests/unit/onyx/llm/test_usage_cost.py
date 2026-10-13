from unittest.mock import MagicMock

import litellm
import pytest

from onyx.llm import usage_cost
from onyx.llm.litellm_singleton.config import load_model_metadata_enrichments
from onyx.llm.model_response import _usage_from_usage_data
from onyx.llm.native_responses_stream import _reported_response_usage
from onyx.tracing.llm_utils import _build_usage_dict


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, float | None]:
    rates: dict[str, float | None] = {
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 10e-6,
        "cache_read_input_token_cost": 0.2e-6,
        "cache_creation_input_token_cost": 2.5e-6,
        "cache_creation_input_token_cost_above_1hr": 4e-6,
    }
    monkeypatch.setattr(usage_cost.cost_overrides, "get_override", lambda *_args: None)
    monkeypatch.setattr(litellm, "get_model_info", lambda **_kwargs: rates)
    return rates


@pytest.mark.usefixtures("registry")
def test_cache_and_reasoning_are_disjoint_billable_counts() -> None:
    result = usage_cost.price_generation(
        "model",
        "provider",
        {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_read_input_tokens": 300,
            "reasoning_tokens": 120,
        },
        MagicMock(),
    )
    assert result.complete
    assert {line.category: line.tokens for line in result.lines} == {
        "input": 700,
        "cache_read": 300,
        "output": 80,
        "reasoning": 120,
    }
    assert result.known_cost_usd == pytest.approx(0.00346)
    assert sum(line.tokens for line in result.lines) == 1200


def test_provider_details_survive_response_and_trace_conversion() -> None:
    usage = _usage_from_usage_data(
        {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 5},
            "prompt_tokens_details": {"cached_tokens": 4},
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1,
                "ephemeral_1h_input_tokens": 2,
            },
        }
    )
    recorded = _build_usage_dict(usage)
    assert recorded is not None
    assert (
        recorded["reasoning_tokens"] == 5 and recorded["cache_read_input_tokens"] == 4
    )
    assert (
        recorded["cache_creation_5m_tokens"] == 1
        and recorded["cache_creation_1h_tokens"] == 2
    )


@pytest.mark.usefixtures("registry")
def test_no_inferred_reasoning_or_cache_write_lifetime() -> None:
    result = usage_cost.price_generation(
        "model",
        "anthropic",
        {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 20},
        MagicMock(),
    )
    assert result.reasoning_tokens is None
    assert not result.complete
    assert next(line for line in result.lines if line.category == "output").tokens == 50
    assert (
        next(
            line for line in result.lines if line.category == "cache_write_unknown"
        ).cost_usd
        is None
    )


def test_cache_write_lifetimes_and_context_tiers(
    registry: dict[str, float | None],
) -> None:
    registry["input_cost_per_token_above_200k_tokens"] = 4e-6
    result = usage_cost.price_generation(
        "model",
        "provider",
        {
            "input_tokens": 200001,
            "output_tokens": 50,
            "cache_creation_input_tokens": 30,
            "cache_creation_5m_tokens": 10,
            "cache_creation_1h_tokens": 20,
        },
        MagicMock(),
    )
    assert result.complete
    assert {line.category: line.usd_per_million for line in result.lines} == {
        "input": 4,
        "output": 10,
        "cache_write": 2.5,
        "cache_write_1h": 4,
    }


@pytest.mark.parametrize(
    "usage", [None, {}, {"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 7}]
)
def test_unknown_price_or_usage_is_never_a_verified_zero(
    registry: dict[str, float | None], usage: dict[str, int] | None
) -> None:
    registry.clear()
    assert not usage_cost.price_generation(
        "unknown", "provider", usage, MagicMock()
    ).complete


def test_true_free_price_and_invalid_subcounts(
    registry: dict[str, float | None],
) -> None:
    registry.update(input_cost_per_token=0, output_cost_per_token=0)
    free = usage_cost.price_generation(
        "free", "provider", {"input_tokens": 10, "output_tokens": 5}, MagicMock()
    )
    assert free.complete and free.known_cost_usd == 0
    invalid = usage_cost.price_generation(
        "free",
        "provider",
        {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 6},
        MagicMock(),
    )
    assert not invalid.complete


@pytest.fixture
def sol_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage_cost.cost_overrides, "get_override", lambda *_args: None)
    existing = {"supports_vision": True}
    monkeypatch.setattr(litellm, "model_cost", {"gpt-6.1-sol": existing})
    load_model_metadata_enrichments()
    assert litellm.model_cost["gpt-6.1-sol"]["supports_vision"] is True


@pytest.mark.usefixtures("sol_registry")
def test_openai_native_writes_are_priced_without_anthropic_lifetime() -> None:
    usage = _reported_response_usage(
        {
            "input_tokens": 60137,
            "output_tokens": 351,
            "total_tokens": 60488,
            "input_tokens_details": {
                "cached_tokens": 21755,
                "cache_write_tokens": 38379,
            },
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    )
    assert usage is not None
    result = usage_cost.price_generation(
        "gpt-6.1-sol", "openai", _build_usage_dict(usage), MagicMock()
    )
    assert result.complete
    assert result.known_cost_usd == pytest.approx(0.101639)
    assert {line.category: line.tokens for line in result.lines} == {
        "input": 3,
        "cache_read": 21755,
        "cache_write": 38379,
        "output": 351,
    }
    assert sum(line.tokens for line in result.lines) == 60488


@pytest.mark.usefixtures("sol_registry")
@pytest.mark.parametrize("incoming,long_context", [(272000, False), (272001, True)])
@pytest.mark.parametrize("tier,multiplier", [(None, 1), ("flex", 0.5), ("priority", 2)])
def test_openai_context_threshold_prices_the_whole_call(
    incoming: int, long_context: bool, tier: str | None, multiplier: float
) -> None:
    result = usage_cost.price_generation(
        "gpt-6.1-sol",
        "openai",
        {
            "input_tokens": incoming,
            "output_tokens": 200,
            "reasoning_tokens": 120,
            "cache_read_input_tokens": 100000,
            "cache_creation_input_tokens": incoming - 100003,
        },
        MagicMock(),
        {"service_tier": tier},
    )
    input_multiplier = multiplier * (2 if long_context else 1)
    output_rate = 10 * multiplier * (1.5 if long_context else 1)
    assert result.complete
    assert {line.category: line.usd_per_million for line in result.lines} == {
        "input": pytest.approx(2 * input_multiplier),
        "cache_read": pytest.approx(0.1 * input_multiplier),
        "cache_write": pytest.approx(2.5 * input_multiplier),
        "output": pytest.approx(output_rate),
        "reasoning": pytest.approx(output_rate),
    }
    assert result.known_cost_usd == pytest.approx(
        (
            3 * 2 * input_multiplier
            + 100000 * 0.1 * input_multiplier
            + (incoming - 100003) * 2.5 * input_multiplier
            + 200 * output_rate
        )
        / 1000000
    )
    assert sum(line.tokens for line in result.lines) == incoming + 200


def test_openai_missing_write_price_keeps_cost_partial(
    registry: dict[str, float | None],
) -> None:
    registry.pop("cache_creation_input_token_cost")
    result = usage_cost.price_generation(
        "unpriced-openai-model",
        "openai",
        {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 20},
        MagicMock(),
    )
    assert not result.complete
    assert (
        next(line for line in result.lines if line.category == "cache_write").cost_usd
        is None
    )


@pytest.mark.usefixtures("sol_registry")
@pytest.mark.parametrize("tier", ["fast", "ultrafast", "unknown"])
def test_unpriced_openai_tier_is_not_reported_as_standard(tier: str) -> None:
    result = usage_cost.price_generation(
        "gpt-6.1-sol",
        "openai",
        {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 20},
        MagicMock(),
        {"service_tier": tier},
    )
    assert not result.complete
    assert result.source == "unavailable"


@pytest.mark.usefixtures("sol_registry")
def test_reported_service_tier_takes_precedence_over_requested_tier() -> None:
    result = usage_cost.price_generation(
        "gpt-6.1-sol",
        "openai",
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "service_tier": "flex",
        },
        MagicMock(),
        {"service_tier": "priority"},
    )
    assert result.complete
    assert result.known_cost_usd == pytest.approx(0.000355)
