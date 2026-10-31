"""The fixed experiment matrix; model names never come from a role override."""

from dataclasses import dataclass

from onyx.asv3.model_composition.models import CompositionProfileId, CompositionRole

COMPOSITION_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    provider_type: str
    model_name: str


MODEL_SPECS = {
    "sol": ModelSpec("openai", "gpt-6.1-sol"),
    "terra": ModelSpec("openai", "gpt-6.1-terra"),
    "luna": ModelSpec("openai", "gpt-6-luna"),
    "flash": ModelSpec("vertex_ai", "gemini-3.8-flash"),
    "lite": ModelSpec("vertex_ai", "gemini-3.5-flash-lite"),
    "argon": ModelSpec("vertex_ai", "gemini-4-argon"),
    "sonnet": ModelSpec("anthropic", "claude-sonnet-5-5"),
}
COMPOSITION_PROFILES: dict[CompositionProfileId, dict[CompositionRole, str]] = {
    "sol_reference_v1": dict.fromkeys(
        ("coordinator", "routine", "search", "author", "repair", "verifier"), "sol"
    ),
    "flash_sol_v1": {
        "coordinator": "flash",
        "routine": "lite",
        "search": "lite",
        "author": "sol",
        "repair": "sol",
        "verifier": "sol",
    },
    "terra_sol_v1": {
        "coordinator": "terra",
        "routine": "lite",
        "search": "lite",
        "author": "terra",
        "repair": "terra",
        "verifier": "sol",
    },
    "argon_sonnet_v1": {
        "coordinator": "argon",
        "routine": "lite",
        "search": "lite",
        "author": "sonnet",
        "repair": "sonnet",
        "verifier": "argon",
    },
    "luna_sol_v1": {
        "coordinator": "sol",
        "routine": "luna",
        "search": "luna",
        "author": "sol",
        "repair": "sol",
        "verifier": "sonnet",
    },
}
