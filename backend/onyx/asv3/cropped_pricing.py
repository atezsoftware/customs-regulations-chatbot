"""Scoped selected-model pricing compatible with the shipped pricing resolver."""

import math
import re
from typing import Any


def model_pricing_info(model: str, provider: str) -> dict[str, Any]:
    import litellm

    try:
        info = dict(litellm.get_model_info(model=model, custom_llm_provider=provider))
    except Exception:
        info = {}
    for key in (info.get("key"), f"{provider}/{model}", model):
        if not isinstance(key, str):
            continue
        configured = litellm.model_cost.get(key)
        if configured is None:
            continue
        for field, value in configured.items():
            if field.startswith(
                ("input_cost_per_token", "output_cost_per_token", "cache_")
            ):
                info[field] = value
        break
    return info


def token_rate(
    info: dict[str, Any], field: str, incoming: int, service_tier: str | None = None
) -> float | None:
    if service_tier not in {None, "auto", "default", "standard", "flex", "priority"}:
        return None
    suffix = f"_{service_tier}" if service_tier in {"flex", "priority"} else ""
    selected = field + suffix
    thresholds: list[tuple[int, str]] = []
    for key, value in info.items():
        match = re.fullmatch(re.escape(field) + r"_above_(\d+)k_tokens" + suffix, key)
        if match and incoming > int(match[1]) * 1000 and value is not None:
            thresholds.append((int(match[1]), key))
    if thresholds:
        selected = max(thresholds)[1]
    value = info.get(selected)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value)
    return None
