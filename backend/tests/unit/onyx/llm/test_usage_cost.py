from unittest.mock import MagicMock

import litellm
import pytest

from onyx.llm import usage_cost
from onyx.llm.model_response import _usage_from_usage_data
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
        "provider",
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
