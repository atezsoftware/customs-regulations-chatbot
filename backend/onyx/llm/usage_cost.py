"""Display estimates from captured usage; independent of billing rollups."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from onyx.llm import cost_overrides
from onyx.llm.cost import get_model_price_per_million


class TokenCostLine(BaseModel):
    category: str
    tokens: int
    usd_per_million: float | None
    cost_usd: float | None


class GenerationCost(BaseModel):
    model: str
    provider: str | None
    source: str
    priced_at: datetime
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    lines: list[TokenCostLine] = Field(default_factory=list)
    complete: bool = False
    known_cost_usd: float = 0


class ResponseUsage(BaseModel):
    duration_seconds: float | None = None
    status: Literal["complete", "partial", "unavailable", "running"] = "unavailable"
    total_cost_usd: float | None = None
    known_cost_usd: float | None = None
    currency: Literal["USD"] = "USD"
    calls: int = 0
    unpriced_calls: int = 0
    excluded_service_calls: int = 0
    models: list[GenerationCost] = Field(default_factory=list)


def token_count(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _rate(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value) * 1_000_000
    return None


def price_generation(
    model: str,
    provider: str | None,
    usage: dict[str, Any] | None,
    db_session: Session,
    request_params: dict[str, Any] | None = None,
) -> GenerationCost:
    """Snapshot provider-aware rates once; never reprice history during rendering."""
    result = GenerationCost(
        model=model,
        provider=provider,
        source="unavailable",
        priced_at=datetime.now(timezone.utc),
    )
    if not usage:
        return result
    incoming = token_count(usage, "input_tokens", "prompt_tokens")
    outgoing = token_count(usage, "output_tokens", "completion_tokens")
    cached = token_count(usage, "cache_read_input_tokens") or 0
    created = token_count(usage, "cache_creation_input_tokens") or 0
    reasoning = token_count(usage, "reasoning_tokens")
    result.input_tokens, result.output_tokens, result.reasoning_tokens = (
        incoming,
        outgoing,
        reasoning,
    )
    if incoming is None or outgoing is None:
        return result
    if cached + created > incoming or (reasoning is not None and reasoning > outgoing):
        return result
    rates: dict[str, float | None] = {}
    override = cost_overrides.get_override(db_session, model, provider or "")
    params = request_params or {}
    if override is not None:
        result.source = "admin_override"
        rates = {
            "input": override.input_cost_per_mtok,
            "output": override.output_cost_per_mtok,
            "reasoning": override.output_cost_per_mtok,
            "cache_read": override.cache_read_cost_per_mtok
            if override.cache_read_cost_per_mtok is not None
            else override.input_cost_per_mtok,
        }
    elif provider == "openrouter":
        quote = get_model_price_per_million(model, provider)
        result.source = "provider_catalog"
        rates = {
            "input": quote.input_per_mtok,
            "output": quote.output_per_mtok,
            "reasoning": quote.output_per_mtok,
            "cache_read": quote.cache_per_mtok,
        }
    else:
        import litellm

        try:
            info = litellm.get_model_info(model=model, custom_llm_provider=provider)
        except Exception:
            info = {}
        tier = params.get("service_tier")
        suffix = f"_{tier}" if tier in {"flex", "priority"} else ""
        fields = {
            "input": "input_cost_per_token",
            "output": "output_cost_per_token",
            "cache_read": "cache_read_input_token_cost",
            "cache_write": "cache_creation_input_token_cost",
            "cache_write_1h": "cache_creation_input_token_cost_above_1hr",
            "reasoning": "output_cost_per_reasoning_token",
        }
        for category, field in fields.items():
            selected = field + suffix
            candidates = []
            for key, value in info.items():
                match = re.fullmatch(
                    re.escape(field) + r"_above_(\d+)k_tokens" + re.escape(suffix), key
                )
                if match and incoming > int(match[1]) * 1000 and value is not None:
                    candidates.append((int(match[1]), key))
            if candidates:
                selected = max(candidates)[1]
            rates[category] = _rate(info.get(selected))
        # Standard reasoning is part of completion usage, not an additional token count.
        if not suffix and rates["reasoning"] is None:
            rates["reasoning"] = rates["output"]
        elif suffix:
            rates["reasoning"] = rates["output"]
        if any(rate is not None for rate in rates.values()):
            result.source = "model_registry"

    counts = {"input": incoming - cached - created, "cache_read": cached}
    one_hour = token_count(usage, "cache_creation_1h_tokens")
    five_minute = token_count(usage, "cache_creation_5m_tokens")
    if (
        one_hour is not None
        and five_minute is not None
        and one_hour + five_minute == created
    ):
        counts.update(cache_write=five_minute, cache_write_1h=one_hour)
    else:
        # A missing cache lifetime cannot establish its write rate.
        counts["cache_write_unknown"] = created
    counts["output"] = outgoing - (reasoning or 0)
    if reasoning is not None:
        counts["reasoning"] = reasoning
    for category, tokens in counts.items():
        if not tokens:
            continue
        rate = rates.get(category)
        cost = tokens * rate / 1_000_000 if rate is not None else None
        result.lines.append(
            TokenCostLine(
                category=category, tokens=tokens, usd_per_million=rate, cost_usd=cost
            )
        )
    result.complete = all(line.cost_usd is not None for line in result.lines) and any(
        rate is not None for rate in rates.values()
    )
    result.known_cost_usd = sum(line.cost_usd or 0 for line in result.lines)
    return result
