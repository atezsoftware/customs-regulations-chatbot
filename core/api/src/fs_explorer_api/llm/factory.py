"""Selects an `LLMClient` implementation based on environment/config."""

import os
from typing import AsyncIterator

from .base import ChatTurn, LLMClient, LLMUsage, SchemaT, ThinkingLevel
from .gemini import GeminiLLMClient
from .openrouter import OpenRouterLLMClient
from .profile import (
    LLMProfile,
    LLMRole,
    LLMRoleConfig,
    SUPPORTED_REASONING_EFFORTS,
    load_llm_profile,
)


class RoleConfiguredLLMClient:
    """LLM client that enforces the selected role's default reasoning policy.

    A call may still explicitly lower or raise ``thinking_level`` when a
    measured workflow stage requires it.  Omitting the argument consistently
    applies the immutable profile snapshot captured for this client.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        role: LLMRole,
        config: LLMRoleConfig,
    ) -> None:
        self._client = client
        self.role = role
        self.provider = config.provider
        self.model = config.model
        self.reasoning_effort = config.reasoning_effort

    @property
    def raw_client(self) -> LLMClient:
        """Expose the provider adapter for diagnostics and focused tests."""
        return self._client

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema: type[SchemaT],
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> tuple[SchemaT, LLMUsage]:
        return await self._client.generate_structured(
            history,
            system_prompt,
            schema,
            thinking_level=thinking_level or self.reasoning_effort,
        )

    async def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._client.stream_text(
            history,
            system_prompt,
            thinking_level=thinking_level or self.reasoning_effort,
        ):
            yield chunk

    def last_stream_usage(self) -> LLMUsage | None:
        return self._client.last_stream_usage()


def _build_llm_client(
    *,
    provider: str,
    model: str | None,
    temperature: float | None,
    api_key: str | None,
    enable_reasoning_effort: bool = False,
) -> LLMClient:
    if provider == "gemini":
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        return GeminiLLMClient(api_key=api_key, model=model, temperature=temperature)

    if provider == "openrouter":
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        return OpenRouterLLMClient(
            api_key=api_key,
            model=model or os.getenv("OPENROUTER_DEFAULT_MODEL"),
            temperature=temperature,
            enable_reasoning_effort=enable_reasoning_effort,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    role: LLMRole | None = None,
    reasoning_effort: ThinkingLevel | None = None,
    profile: LLMProfile | None = None,
) -> LLMClient:
    """
    Build the configured `LLMClient`.

    Without ``role``, provider precedence remains explicit argument >
    ``FS_EXPLORER_LLM_PROVIDER`` > ``"gemini"`` and model precedence remains
    explicit argument > ``FS_EXPLORER_LLM_MODEL`` > provider default.

    With ``role``, explicit arguments override the corresponding immutable
    role profile. The returned wrapper automatically applies that role's
    reasoning effort when a call does not provide ``thinking_level``.
    """
    if role is None:
        if profile is not None:
            raise ValueError("profile requires a named LLM role.")
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort requires a named LLM role.")
        resolved_provider = (
            provider or os.getenv("FS_EXPLORER_LLM_PROVIDER") or "gemini"
        ).lower()
        resolved_model = model or os.getenv("FS_EXPLORER_LLM_MODEL")
        return _build_llm_client(
            provider=resolved_provider,
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
            enable_reasoning_effort=False,
        )

    role_config = (profile or load_llm_profile()).for_role(role)
    resolved_provider = (provider or role_config.provider).lower()
    resolved_model = model or role_config.model
    resolved_reasoning = reasoning_effort or role_config.reasoning_effort
    if resolved_reasoning not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of "
            f"{sorted(SUPPORTED_REASONING_EFFORTS)!r}; "
            f"got {resolved_reasoning!r}."
        )

    effective_config = LLMRoleConfig(
        provider=resolved_provider,
        model=resolved_model,
        reasoning_effort=resolved_reasoning,
    )
    client = _build_llm_client(
        provider=resolved_provider,
        model=resolved_model,
        temperature=temperature,
        api_key=api_key,
        enable_reasoning_effort=True,
    )
    return RoleConfiguredLLMClient(
        client,
        role=role,
        config=effective_config,
    )
