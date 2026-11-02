"""Versioned role presets; model names never come from a request role override."""

from dataclasses import dataclass

from onyx.asv3.model_composition.models import (
    KOMPOZIT_PROFILE_ID,
    CompositionProfileId,
    CompositionRole,
)

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
    "factorial_v1_sss": {
        "coordinator": "sol",
        "routine": "sol",
        "search": "lite",
        "author": "sol",
        "repair": "terra",
        "verifier": "sol",
    },
    "factorial_v1_ssa": {
        "coordinator": "sol",
        "routine": "sol",
        "search": "lite",
        "author": "sol",
        "repair": "terra",
        "verifier": "argon",
    },
    "factorial_v1_sns": {
        "coordinator": "sol",
        "routine": "sol",
        "search": "lite",
        "author": "sonnet",
        "repair": "terra",
        "verifier": "sol",
    },
    "factorial_v1_sna": {
        "coordinator": "sol",
        "routine": "sol",
        "search": "lite",
        "author": "sonnet",
        "repair": "terra",
        "verifier": "argon",
    },
    "factorial_v1_fss": {
        "coordinator": "flash",
        "routine": "flash",
        "search": "lite",
        "author": "sol",
        "repair": "terra",
        "verifier": "sol",
    },
    "factorial_v1_fsa": {
        "coordinator": "flash",
        "routine": "flash",
        "search": "lite",
        "author": "sol",
        "repair": "terra",
        "verifier": "argon",
    },
    "factorial_v1_fns": {
        "coordinator": "flash",
        "routine": "flash",
        "search": "lite",
        "author": "sonnet",
        "repair": "terra",
        "verifier": "sol",
    },
    "factorial_v1_fna": {
        "coordinator": "flash",
        "routine": "flash",
        "search": "lite",
        "author": "sonnet",
        "repair": "terra",
        "verifier": "argon",
    },
    KOMPOZIT_PROFILE_ID: {
        "coordinator": "flash",
        "routine": "flash",
        "search": "lite",
        "author": "sonnet",
        "repair": "terra",
        "verifier": "argon",
    },
}
