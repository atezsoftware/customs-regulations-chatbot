"""Refresh Auto providers from live Google discovery or a recommendation feed.

Vertex connections retain explicit defaults and pinned models while exposing newly
published chat models. Other providers follow the configured recommendation feed.
"""

from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from onyx.cache.factory import get_cache_backend
from onyx.configs.app_configs import AUTO_LLM_CONFIG_URL
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.llm import (
    fetch_auto_mode_providers,
    sync_auto_mode_models,
    sync_vertex_model_configurations,
)
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.auto_update_models import LLMRecommendations
from onyx.llm.well_known_providers.vertex_models import discover_vertex_models
from onyx.server.manage.llm.models import SyncModelEntry
from onyx.server.manage.llm.provider_cache import invalidate_provider_listing_cache
from onyx.utils.logger import setup_logger

logger = setup_logger()

_CACHE_KEY_LAST_UPDATED_AT = "auto_llm_update:last_updated_at"
_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def sync_vertex_models_from_google() -> dict[str, int]:
    """Discover each saved Auto connection without holding a DB lease over HTTP."""
    with get_session_with_current_tenant() as db_session:
        connections = [
            (provider.id, dict(provider.custom_config or {}))
            for provider in fetch_auto_mode_providers(db_session)
            if provider.provider == LlmProviderNames.VERTEX_AI
        ]

    results: dict[str, int] = {}
    for provider_id, custom_config in connections:
        try:
            models = discover_vertex_models(custom_config)
            entries = [SyncModelEntry(**model.model_dump()) for model in models]
            with get_session_with_current_tenant() as db_session:
                added = sync_vertex_model_configurations(
                    db_session, provider_id, custom_config, entries
                )
            if added is not None:
                # Sync can also upgrade visibility/capabilities without adding rows.
                invalidate_provider_listing_cache()
                results[str(provider_id)] = added
        except Exception as exc:
            # One connection's credentials/quota must not block other providers.
            # Never log credential-bearing request data or upstream error bodies.
            logger.warning(
                "Vertex model discovery failed for provider id=%s (%s); retaining existing models",
                provider_id,
                type(exc).__name__,
            )
    return results


def _get_cached_last_updated_at() -> datetime | None:
    try:
        value = get_cache_backend().get(_CACHE_KEY_LAST_UPDATED_AT)
        if value is not None:
            return datetime.fromisoformat(value.decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to get cached last_updated_at: %s", e)
    return None


def _set_cached_last_updated_at(updated_at: datetime) -> None:
    try:
        get_cache_backend().set(
            _CACHE_KEY_LAST_UPDATED_AT,
            updated_at.isoformat(),
            ex=_CACHE_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning("Failed to set cached last_updated_at: %s", e)


def fetch_llm_recommendations_from_github(
    timeout: float = 30.0,
) -> LLMRecommendations | None:
    """Fetch LLM configuration from GitHub.

    Returns:
        GitHubLLMConfig if successful, None on error.
    """
    if not AUTO_LLM_CONFIG_URL:
        logger.debug("AUTO_LLM_CONFIG_URL not configured, skipping fetch")
        return None

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(AUTO_LLM_CONFIG_URL)
            response.raise_for_status()

            data = response.json()
            return LLMRecommendations.model_validate(data)
    except httpx.HTTPError as e:
        logger.error("Failed to fetch LLM config from GitHub: %s", e)
        return None
    except Exception as e:
        logger.error("Error parsing LLM config: %s", e)
        return None


def sync_llm_models_from_github(
    db_session: Session,
    force: bool = False,
) -> dict[str, int]:
    """Sync models from GitHub config to database for all Auto mode providers.

    In Auto mode, EVERYTHING is controlled by GitHub config:
    - Model list
    - Model visibility (is_visible)
    - Default model
    - Fast default model

    Args:
        db_session: Database session
        config: GitHub LLM configuration
        force: If True, skip the updated_at check and force sync

    Returns:
        Dict of provider_name -> number of changes made.
    """
    results: dict[str, int] = {}

    # Get all providers in Auto mode
    auto_providers = [
        provider
        for provider in fetch_auto_mode_providers(db_session)
        if provider.provider != LlmProviderNames.VERTEX_AI
    ]
    if not auto_providers:
        logger.debug("No providers in Auto mode found")
        return {}

    # Fetch config from GitHub
    config = fetch_llm_recommendations_from_github()
    if not config:
        logger.warning("Failed to fetch GitHub config")
        return {}

    # Skip if we've already processed this version (unless forced)
    last_updated_at = _get_cached_last_updated_at()
    if not force and last_updated_at and config.updated_at <= last_updated_at:
        logger.debug("GitHub config unchanged, skipping sync")
        _set_cached_last_updated_at(config.updated_at)
        return {}

    for provider in auto_providers:
        provider_type = provider.provider  # e.g., "openai", "anthropic"

        if provider_type not in config.providers:
            logger.debug(
                "No config for provider type '%s' in GitHub config", provider_type
            )
            continue

        # Sync models - this replaces the model list entirely for Auto mode
        changes = sync_auto_mode_models(
            db_session=db_session,
            provider=provider,
            llm_recommendations=config,
        )

        if changes > 0:
            results[str(provider.id)] = changes
            logger.info(
                "Applied %s model changes to provider '%s'",
                changes,
                provider.name or provider.provider,
            )

    _set_cached_last_updated_at(config.updated_at)
    return results


def reset_cache() -> None:
    """Reset the cache timestamp. Useful for testing."""
    try:
        get_cache_backend().delete(_CACHE_KEY_LAST_UPDATED_AT)
    except Exception as e:
        logger.warning("Failed to reset cache: %s", e)
