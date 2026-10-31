"""Preflight every configured role through the ordinary authorized model factory."""

from onyx.asv3.model_composition.models import (
    CompositionProfileId,
    ModelBinding,
    ModelComposition,
)
from onyx.asv3.model_composition.profiles import (
    COMPOSITION_PROFILES,
    COMPOSITION_VERSION,
    MODEL_SPECS,
)
from onyx.asv3.workflow_variant import ASV3_FAST_CROPPED_VARIANT, ASV3_FAST_VARIANT
from onyx.db.asv3_models import (
    check_asv3_secondary_model_cost_limit,
    resolve_asv3_model_provider_id,
)
from onyx.db.models import Persona, User
from onyx.llm.factory import get_llm_for_persona, get_llm_token_counter
from onyx.llm.override_models import LLMOverride


def resolve_model_composition(
    profile_id: CompositionProfileId,
    user: User,
    persona: Persona | None,
    workflow_variant: str,
) -> ModelComposition:
    if workflow_variant not in {ASV3_FAST_VARIANT, ASV3_FAST_CROPPED_VARIANT}:
        raise ValueError("Model composition requires an explicit Fast workflow")
    if profile_id not in COMPOSITION_PROFILES:
        raise ValueError("Unknown Fast model composition profile")
    profile = COMPOSITION_PROFILES[profile_id]
    resolved: dict[str, ModelBinding] = {}
    for alias in dict.fromkeys(profile.values()):
        spec = MODEL_SPECS[alias]
        provider_id = resolve_asv3_model_provider_id(
            spec.provider_type, spec.model_name
        )
        llm = get_llm_for_persona(
            persona,
            user,
            llm_override=LLMOverride(
                model_provider_id=provider_id,
                model_provider_type=spec.provider_type,
                model_version=spec.model_name,
            ),
        )
        config = llm.config
        if (
            config.model_provider != spec.provider_type
            or config.model_name != spec.model_name
            or type(config.max_input_tokens) is not int
            or config.max_input_tokens <= 0
        ):
            raise ValueError("The configured composition model cannot serve its role")
        check_asv3_secondary_model_cost_limit(config.api_key)
        resolved[alias] = ModelBinding(llm, get_llm_token_counter(llm), provider_id)
    return ModelComposition(
        profile_id,
        COMPOSITION_VERSION,
        {role: resolved[alias] for role, alias in profile.items()},
    )
