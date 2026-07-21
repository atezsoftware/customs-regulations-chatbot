"""Selects an `LLMClient` implementation based on environment/config."""

import os

from .base import LLMClient
from .gemini import GeminiLLMClient
from .openrouter import OpenRouterLLMClient


def get_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """
    Build the configured `LLMClient`.

    Provider precedence: explicit `provider` arg > `FS_EXPLORER_LLM_PROVIDER`
    env var > `"gemini"`. Model precedence: explicit `model` arg >
    `FS_EXPLORER_LLM_MODEL` env var > the provider's own default.
    """
    resolved_provider = (
        provider or os.getenv("FS_EXPLORER_LLM_PROVIDER") or "gemini"
    ).lower()
    resolved_model = model or os.getenv("FS_EXPLORER_LLM_MODEL")

    if resolved_provider == "gemini":
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        return GeminiLLMClient(
            api_key=api_key, model=resolved_model, temperature=temperature
        )

    if resolved_provider == "openrouter":
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        return OpenRouterLLMClient(
            api_key=api_key,
            model=resolved_model or os.getenv("OPENROUTER_DEFAULT_MODEL"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM provider: {resolved_provider!r}")
