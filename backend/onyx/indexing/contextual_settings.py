from typing import Protocol

from onyx.configs.app_configs import ENABLE_CONTEXTUAL_RAG
from onyx.db.models import SearchSettings
from onyx.llm.factory import get_contextual_rag_llm_for_search_settings
from onyx.llm.interfaces import LLM
from shared_configs.configs import MULTI_TENANT


class ContextualSettingsLike(Protocol):
    enable_contextual_rag: bool


class ContextualIndexingConfigurationError(RuntimeError):
    """Contextual indexing is required but has no usable model."""


def effective_contextual_rag_enabled(
    search_settings: ContextualSettingsLike,
    *,
    env_enabled: bool = ENABLE_CONTEXTUAL_RAG,
    multitenant: bool = MULTI_TENANT,
) -> bool:
    """Return the deployment-safe contextual-indexing policy."""

    if multitenant:
        return False
    return bool(search_settings.enable_contextual_rag or env_enabled)


def require_contextual_rag_llm(
    search_settings: SearchSettings,
    *,
    env_enabled: bool = ENABLE_CONTEXTUAL_RAG,
    multitenant: bool = MULTI_TENANT,
) -> LLM | None:
    """Resolve the required model, or fail before any document is indexed."""

    if not effective_contextual_rag_enabled(
        search_settings,
        env_enabled=env_enabled,
        multitenant=multitenant,
    ):
        return None

    llm = get_contextual_rag_llm_for_search_settings(search_settings)
    if llm is None:
        raise ContextualIndexingConfigurationError(
            "Select a contextualization model before indexing"
        )
    return llm
