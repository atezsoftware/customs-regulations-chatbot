"""Role-specific LLM policy for the multi-agent research workflow.

The legacy agent still uses the global ``FS_EXPLORER_LLM_*`` settings through
``get_llm_client()``.  New orchestration code opts into this policy by asking
the factory for a named role.  Keeping the two configuration paths separate
allows a feature-flagged rollout without silently changing existing chats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast

from .base import ThinkingLevel

LLMRole = Literal["planner", "task", "worker", "final"]

SUPPORTED_LLM_PROVIDERS = frozenset({"gemini", "openrouter"})
SUPPORTED_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


@dataclass(frozen=True, slots=True)
class LLMRoleConfig:
    """Provider policy for one isolated orchestration role."""

    provider: str
    model: str
    reasoning_effort: ThinkingLevel


@dataclass(frozen=True, slots=True)
class LLMProfile:
    """Complete, immutable model policy for one multi-agent run."""

    planner: LLMRoleConfig
    task: LLMRoleConfig
    worker: LLMRoleConfig
    final: LLMRoleConfig

    def for_role(self, role: LLMRole) -> LLMRoleConfig:
        """Return the configuration for ``role`` with an exhaustive check."""
        if role == "planner":
            return self.planner
        if role == "task":
            return self.task
        if role == "worker":
            return self.worker
        if role == "final":
            return self.final
        raise ValueError(f"Unknown LLM role: {role!r}")


DEFAULT_LLM_PROFILE = LLMProfile(
    planner=LLMRoleConfig(
        provider="openrouter",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
    ),
    task=LLMRoleConfig(
        provider="openrouter",
        model="google/gemini-3.6-flash",
        reasoning_effort="medium",
    ),
    worker=LLMRoleConfig(
        provider="openrouter",
        model="google/gemini-3.5-flash-lite",
        reasoning_effort="low",
    ),
    final=LLMRoleConfig(
        provider="openrouter",
        model="google/gemini-3.6-flash",
        reasoning_effort="high",
    ),
)


def _configured_value(
    environment: Mapping[str, str],
    variable: str,
    default: str,
) -> str:
    raw_value = environment.get(variable)
    if raw_value is None:
        return default
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{variable} must not be empty.")
    return value


def _load_role(
    environment: Mapping[str, str],
    role: LLMRole,
    default: LLMRoleConfig,
) -> LLMRoleConfig:
    prefix = f"FS_EXPLORER_{role.upper()}"
    provider = _configured_value(
        environment, f"{prefix}_PROVIDER", default.provider
    ).lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"{prefix}_PROVIDER must be one of "
            f"{sorted(SUPPORTED_LLM_PROVIDERS)!r}; got {provider!r}."
        )

    model = _configured_value(environment, f"{prefix}_MODEL", default.model)
    reasoning = _configured_value(
        environment,
        f"{prefix}_REASONING",
        default.reasoning_effort,
    ).lower()
    if reasoning not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            f"{prefix}_REASONING must be one of "
            f"{sorted(SUPPORTED_REASONING_EFFORTS)!r}; got {reasoning!r}."
        )

    return LLMRoleConfig(
        provider=provider,
        model=model,
        reasoning_effort=cast(ThinkingLevel, reasoning),
    )


def load_llm_profile(
    environment: Mapping[str, str] | None = None,
) -> LLMProfile:
    """Load and validate the role policy from environment variables.

    Each role uses ``FS_EXPLORER_<ROLE>_PROVIDER``,
    ``FS_EXPLORER_<ROLE>_MODEL`` and ``FS_EXPLORER_<ROLE>_REASONING``.  The
    global legacy variables intentionally do not override role defaults.
    """

    source = os.environ if environment is None else environment
    return LLMProfile(
        planner=_load_role(source, "planner", DEFAULT_LLM_PROFILE.planner),
        task=_load_role(source, "task", DEFAULT_LLM_PROFILE.task),
        worker=_load_role(source, "worker", DEFAULT_LLM_PROFILE.worker),
        final=_load_role(source, "final", DEFAULT_LLM_PROFILE.final),
    )
