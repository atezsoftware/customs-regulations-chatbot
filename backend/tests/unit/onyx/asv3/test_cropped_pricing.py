"""Scoped pricing uses declared rates, exclusive cache partitions and tier metadata."""

import pytest

from onyx.asv3.cropped_cost_budget import _rates
from onyx.asv3.cropped_pricing import model_pricing_info, token_rate
from onyx.asv3.models import RunStopped


def test_selected_sol_enriches_cache_write_and_context_rates_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda **_: {
            "key": "openai/gpt-6.1-sol",
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/gpt-6.1-sol",
        {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.0000001,
            "cache_creation_input_token_cost": 0.0000025,
            "input_cost_per_token_above_272k_tokens": 0.000004,
            "output_cost_per_token_above_272k_tokens": 0.000015,
            "cache_read_input_token_cost_above_272k_tokens": 0.0000002,
            "cache_creation_input_token_cost_above_272k_tokens": 0.000005,
        },
    )
    info = model_pricing_info("gpt-6.1-sol", "openai")
    assert info["cache_creation_input_token_cost"] == 0.0000025
    assert _rates("gpt-6.1-sol", "openai", 272000) == (
        0.000002,
        0.00001,
        0.0000001,
        0.0000025,
    )
    assert _rates("gpt-6.1-sol", "openai", 272001) == (
        0.000004,
        0.000015,
        0.0000002,
        0.000005,
    )


def test_optional_tiers_do_not_fall_back_to_guessed_free_or_standard_rates() -> None:
    info = {
        "input_cost_per_token": 0.000002,
        "input_cost_per_token_flex": 0.000001,
        "input_cost_per_token_priority": None,
        "input_cost_per_token_above_272k_tokens_flex": 0.000002,
    }
    assert token_rate(info, "input_cost_per_token", 272000, "standard") == 0.000002
    assert token_rate(info, "input_cost_per_token", 272000, "flex") == 0.000001
    assert token_rate(info, "input_cost_per_token", 272001, "flex") == 0.000002
    assert token_rate(info, "input_cost_per_token", 1000, "priority") is None
    assert token_rate(info, "input_cost_per_token", 1000, "unrecognized") is None


@pytest.mark.parametrize("rate", [True, -1, float("nan"), float("inf"), "2e-6", None])
def test_invalid_declared_rate_is_unavailable(rate: object) -> None:
    assert (
        token_rate({"input_cost_per_token": rate}, "input_cost_per_token", 1000) is None
    )


def test_missing_registry_price_is_an_error_not_a_zero_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "get_model_info", lambda **_: {})
    with pytest.raises(RunStopped, match="pricing is unavailable"):
        _rates("unknown-selected-model", "unknown", 1000)
