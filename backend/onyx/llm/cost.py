"""LLM cost calculation utilities."""

import threading
from urllib.parse import quote

import httpx
from cachetools import TTLCache
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.configs.app_configs import (
    DEFAULT_IMAGE_COST_CENTS,
    DEFAULT_LLM_INPUT_COST_PER_MTOK,
    DEFAULT_LLM_OUTPUT_COST_PER_MTOK,
)
from onyx.llm import cost_overrides
from onyx.tracing.flows import IMAGE_FLOWS, LLMFlow
from onyx.utils.logger import setup_logger

logger = setup_logger()


class ModelPrice(BaseModel):
    model: str
    provider: str | None
    input_per_mtok: float | None
    output_per_mtok: float | None
    cache_per_mtok: float | None


_OPENROUTER_PRICE_CACHE_TTL_SECONDS = 60 * 60
_OPENROUTER_PRICE_CACHE: TTLCache[str, ModelPrice] = TTLCache(
    maxsize=512,
    ttl=_OPENROUTER_PRICE_CACHE_TTL_SECONDS,
)
_OPENROUTER_PRICE_CACHE_LOCK = threading.Lock()


def _is_openrouter(provider: str | None) -> bool:
    return bool(provider and provider.strip().lower() == "openrouter")


def _openrouter_price_per_million(model: str) -> ModelPrice:
    normalized_model = model.strip()
    if normalized_model.startswith("openrouter/"):
        normalized_model = normalized_model.removeprefix("openrouter/")
    if not normalized_model:
        return ModelPrice(
            model=model,
            provider="openrouter",
            input_per_mtok=None,
            output_per_mtok=None,
            cache_per_mtok=None,
        )

    with _OPENROUTER_PRICE_CACHE_LOCK:
        cached = _OPENROUTER_PRICE_CACHE.get(normalized_model)
    if cached is not None:
        return cached

    price = ModelPrice(
        model=model,
        provider="openrouter",
        input_per_mtok=None,
        output_per_mtok=None,
        cache_per_mtok=None,
    )
    try:
        encoded_model = quote(normalized_model, safe="/:._-")
        response = httpx.get(
            f"https://openrouter.ai/api/v1/model/{encoded_model}",
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        pricing = data.get("pricing", {}) if isinstance(data, dict) else {}

        def _per_million(field: str) -> float | None:
            raw_value = pricing.get(field) if isinstance(pricing, dict) else None
            if raw_value is None:
                return None
            parsed = float(raw_value) * 1_000_000
            return parsed if parsed >= 0 else None

        price = ModelPrice(
            model=model,
            provider="openrouter",
            input_per_mtok=_per_million("prompt"),
            output_per_mtok=_per_million("completion"),
            cache_per_mtok=_per_million("input_cache_read"),
        )
    except Exception:
        logger.debug(
            "OpenRouter pricing lookup failed for model %s",
            normalized_model,
            exc_info=True,
        )

    with _OPENROUTER_PRICE_CACHE_LOCK:
        _OPENROUTER_PRICE_CACHE[normalized_model] = price
    return price


def get_model_price_per_million(
    model: str,
    provider: str | None,
    db_session: Session | None = None,
) -> ModelPrice:
    """Return override-aware USD per million tokens without raising."""
    if db_session is not None:
        try:
            rates = cost_overrides.get_override(db_session, model, provider or "")
        except Exception:
            logger.exception("Override lookup failed for model %s", model)
            rates = None
        if rates is not None:
            return ModelPrice(
                model=model,
                provider=provider,
                input_per_mtok=rates.input_cost_per_mtok,
                output_per_mtok=rates.output_cost_per_mtok,
                cache_per_mtok=rates.cache_read_cost_per_mtok,
            )

    if _is_openrouter(provider):
        openrouter_price = _openrouter_price_per_million(model)
        if (
            openrouter_price.input_per_mtok is not None
            or openrouter_price.output_per_mtok is not None
        ):
            return openrouter_price

    try:
        import litellm

        entry = litellm.get_model_info(model=model, custom_llm_provider=provider)
        input_per_tok = entry.get("input_cost_per_token")
        output_per_tok = entry.get("output_cost_per_token")
        cache_per_tok = entry.get("cache_read_input_token_cost")
        return ModelPrice(
            model=model,
            provider=provider,
            input_per_mtok=(
                float(input_per_tok) * 1_000_000 if input_per_tok is not None else None
            ),
            output_per_mtok=(
                float(output_per_tok) * 1_000_000
                if output_per_tok is not None
                else None
            ),
            cache_per_mtok=(
                float(cache_per_tok) * 1_000_000 if cache_per_tok is not None else None
            ),
        )
    except Exception:
        logger.debug("No price-per-million for model %s (provider %s)", model, provider)
        return ModelPrice(
            model=model,
            provider=provider,
            input_per_mtok=None,
            output_per_mtok=None,
            cache_per_mtok=None,
        )


def _image_cost_cents(model: str, provider: str | None) -> float:
    """Per-image cents from litellm, else DEFAULT_IMAGE_COST_CENTS."""
    try:
        import litellm

        try:
            entry = litellm.get_model_info(model=model, custom_llm_provider=provider)
        except Exception:
            entry = litellm.model_cost.get(model) or {}
        # litellm prices images per-image under either of these keys. Use an
        # explicit None check so a genuinely free (0.0) model is billed 0, not
        # silently bumped to the flat fallback.
        per_image_usd = entry.get("output_cost_per_image")
        if per_image_usd is None:
            per_image_usd = entry.get("input_cost_per_image")
        if per_image_usd is not None:
            return float(per_image_usd) * 100
    except Exception:
        logger.exception("Image price lookup failed for model %s", model)
    return DEFAULT_IMAGE_COST_CENTS


def _override_cost_cents(
    rates: cost_overrides.CostOverrideRates,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
) -> tuple[float, float]:
    """Apply admin per-Mtok rates. Cache reads bill at the admin cache rate when
    set, otherwise at the input rate. Cache cost is folded into the input half."""
    input_per_mtok = rates.input_cost_per_mtok
    output_per_mtok = rates.output_cost_per_mtok
    cache_per_mtok = rates.cache_read_cost_per_mtok
    cache_rate = cache_per_mtok if cache_per_mtok is not None else input_per_mtok
    input_cents = (
        input_tokens / 1_000_000 * input_per_mtok * 100
        + cache_read_tokens / 1_000_000 * cache_rate * 100
    )
    output_cents = output_tokens / 1_000_000 * output_per_mtok * 100
    return input_cents, output_cents


def _price_cost_cents(
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
) -> tuple[float, float]:
    input_rate = price.input_per_mtok or 0.0
    output_rate = price.output_per_mtok or 0.0
    cache_rate = (
        price.cache_per_mtok if price.cache_per_mtok is not None else input_rate
    )
    return (
        (
            input_tokens / 1_000_000 * input_rate
            + cache_read_tokens / 1_000_000 * cache_rate
        )
        * 100,
        output_tokens / 1_000_000 * output_rate * 100,
    )


def compute_cost_cents(
    model: str,
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    flow: LLMFlow | str | None = None,
    image_count: int = 1,
    db_session: Session | None = None,
) -> tuple[float, float]:
    """Return (input_cost_cents, output_cost_cents) for an LLM call.

    Resolution order: image pricing → admin override → litellm → default
    fallback rates (0 unless set). Never raises (usage hot path)."""
    if flow in IMAGE_FLOWS:
        return 0.0, _image_cost_cents(model, provider) * max(image_count, 1)

    if db_session is not None:
        try:
            rates = cost_overrides.get_override(db_session, model, provider or "")
        except Exception:
            logger.exception("Override lookup failed for model %s", model)
            rates = None
        if rates is not None:
            return _override_cost_cents(
                rates, input_tokens, output_tokens, cache_read_tokens
            )

    try:
        import litellm

        # custom_llm_provider is required for non-self-identifying model names
        # (bedrock/vertex/anthropic-plain) — without it litellm raises and we'd
        # record $0 for entire provider classes.
        # input_tokens are non-cached; cache reads are additional prompt tokens
        # billed at the model's (discounted) cache-read rate, never as output.
        prompt_cost_usd, completion_cost_usd = litellm.cost_per_token(
            model=model,
            custom_llm_provider=provider,
            prompt_tokens=input_tokens + cache_read_tokens,
            completion_tokens=output_tokens,
            cache_read_input_tokens=cache_read_tokens,
        )
        input_cents = prompt_cost_usd * 100
        output_cents = completion_cost_usd * 100
        if (
            input_cents + output_cents == 0
            and input_tokens + output_tokens + cache_read_tokens > 0
            and _is_openrouter(provider)
        ):
            resolved_price = get_model_price_per_million(model, provider)
            if any(
                rate is not None and rate > 0
                for rate in (
                    resolved_price.input_per_mtok,
                    resolved_price.output_per_mtok,
                    resolved_price.cache_per_mtok,
                )
            ):
                return _price_cost_cents(
                    resolved_price,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                )
        return input_cents, output_cents
    except Exception:
        # Unpriced model: configurable default rates; debug log distinguishes
        # transient litellm failure from a genuinely unpriced model.
        logger.debug(
            "litellm pricing failed for model %s (provider %s); using default rates",
            model,
            provider,
            exc_info=True,
        )
        resolved_price = get_model_price_per_million(model, provider)
        if (
            resolved_price.input_per_mtok is not None
            or resolved_price.output_per_mtok is not None
        ):
            return _price_cost_cents(
                resolved_price,
                input_tokens,
                output_tokens,
                cache_read_tokens,
            )

        billed_input = input_tokens + cache_read_tokens
        input_cents = billed_input / 1_000_000 * DEFAULT_LLM_INPUT_COST_PER_MTOK * 100
        output_cents = (
            output_tokens / 1_000_000 * DEFAULT_LLM_OUTPUT_COST_PER_MTOK * 100
        )
        if not (DEFAULT_LLM_INPUT_COST_PER_MTOK or DEFAULT_LLM_OUTPUT_COST_PER_MTOK):
            logger.warning(
                "No price for model %s (provider %s); recording 0 cost.",
                model,
                provider,
            )
        return input_cents, output_cents
