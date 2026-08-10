"""Overrides sent over the wire / stored in the DB

NOTE: these models are used in many places, so have to be
kepy in a separate file to avoid circular imports.
"""

from pydantic import BaseModel


class LLMOverride(BaseModel):
    """Per-request LLM settings that override persona defaults.

    All fields are optional — only the fields that differ from the persona's
    configured LLM need to be supplied. Used both over the wire (API requests)
    and for multi-model comparison, where one override is supplied per model.

    Attributes:
        model_provider: Provider instance name, or the legacy provider-type
            selector for older payloads. When ``None``, resolution uses
            ``model_provider_type`` or the persona default.
        model_provider_type: Provider implementation type paired with
            ``model_provider``. New callers set this so equal display names
            across provider types remain unambiguous; legacy payloads may omit it.
        model_provider_id: Exact provider row ID for callers that must select
            between multiple nameless instances of the same provider type.
        model_version: Specific model version string (e.g. ``"gpt-4o"``).
            When ``None``, the persona's default model is used.
        temperature: Sampling temperature in ``[0, 2]``. When ``None``, the
            persona's default temperature is used.
        display_name: Human-readable label shown in the UI for this model,
            e.g. ``"GPT-4 Turbo"``. Optional; falls back to ``model_version``
            when not set.
    """

    model_provider: str | None = None
    model_provider_type: str | None = None
    model_provider_id: int | None = None
    model_version: str | None = None
    temperature: float | None = None
    display_name: str | None = None

    # This disables the "model_" protected namespace for pydantic
    model_config = {"protected_namespaces": ()}


class PromptOverride(BaseModel):
    system_prompt: str | None = None
    task_prompt: str | None = None
