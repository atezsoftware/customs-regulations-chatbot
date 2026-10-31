"""Resolved, run-local model roles for explicitly selected Fast profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.llm.interfaces import LLM

CompositionRole = Literal[
    "coordinator", "routine", "search", "author", "repair", "verifier"
]
CompositionProfileId = Literal[
    "sol_reference_v1", "flash_sol_v1", "terra_sol_v1", "argon_sonnet_v1", "luna_sol_v1"
]
COMPOSITION_ROLES: tuple[CompositionRole, ...] = (
    "coordinator",
    "routine",
    "search",
    "author",
    "repair",
    "verifier",
)
MODEL_COMPOSITION_SERVICE = "asv3_model_composition"


@dataclass(frozen=True)
class ModelBinding:
    """Internal LLM/callable binding; frozen dataclasses keep these out of wire models."""

    llm: LLM
    token_counter: Callable[[str], int]
    provider_id: int

    def descriptor(self) -> dict[str, JsonValue]:
        config = self.llm.config
        configuration = config.model_dump(mode="json", exclude={"api_key"})
        return {
            "provider_id": self.provider_id,
            "provider": config.model_provider,
            "model": config.model_name,
            "max_input_tokens": config.max_input_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
            "configuration_sha256": hashlib.sha256(
                json.dumps(configuration, sort_keys=True).encode()
            ).hexdigest(),
        }


@dataclass(frozen=True)
class ModelComposition:
    profile_id: CompositionProfileId
    version: int
    role_bindings: Mapping[CompositionRole, ModelBinding]

    def __post_init__(self) -> None:
        if set(self.role_bindings) != set(COMPOSITION_ROLES):
            raise ValueError("A model composition requires every exact role")
        object.__setattr__(
            self, "role_bindings", MappingProxyType(dict(self.role_bindings))
        )

    def binding(self, role: CompositionRole) -> ModelBinding:
        return self.role_bindings[role]

    def effective_search_reasoning_effort(self) -> Literal["low", "off"]:
        config = self.binding("search").llm.config
        return (
            "low"
            if config.model_provider == "openai" and config.model_name == "gpt-6.1-sol"
            else "off"
        )

    def identity(self) -> dict[str, JsonValue]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "reasoning_effort": "low",
            "requested_search_reasoning_effort": "off",
            "effective_search_reasoning_effort": self.effective_search_reasoning_effort(),
            "roles": {
                role: self.binding(role).descriptor() for role in COMPOSITION_ROLES
            },
        }


def get_model_composition(context: RunContext) -> ModelComposition | None:
    value = context.services.get(MODEL_COMPOSITION_SERVICE)
    if value is None:
        return None
    if not isinstance(value, ModelComposition):
        raise ValueError("Invalid resolved model composition")
    return value
