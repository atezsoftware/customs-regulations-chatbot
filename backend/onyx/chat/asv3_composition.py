"""Fixed product presets and authorized opt-in benchmark model compositions."""

from onyx.asv3.model_composition.models import ModelComposition
from onyx.asv3.model_composition.profiles import KOMPOZIT_PROFILE_ID, MODEL_SPECS
from onyx.asv3.model_composition.resolver import resolve_model_composition
from onyx.asv3.models import ASv3WorkflowSelection
from onyx.asv3.workflow_variant import (
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    ASV3_KOMPOZIT_CROPPED_VARIANT,
    ASV3_KOMPOZIT_VARIANT,
)
from onyx.auth.schemas import UserRole
from onyx.db.asv3_models import resolve_asv3_model_provider_id
from onyx.db.asv3_runs import load_asv3_checkpoint
from onyx.db.models import Persona, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import MessageOrigin, SendMessageRequest


def kompozit_llm_override(
    request: SendMessageRequest, user: User
) -> LLMOverride | None:
    variant = (
        ASV3_KOMPOZIT_VARIANT
        if request.asv3_kompozit
        else ASV3_KOMPOZIT_CROPPED_VARIANT
        if request.asv3_kompozit_cropped
        else None
    )
    if request.asv3_resume_message_id is not None:
        checkpoint = load_asv3_checkpoint(
            message_id=request.asv3_resume_message_id, user_id=user.id
        )
        saved = checkpoint.get("asv3_workflow_variant") if checkpoint else None
        if saved in {ASV3_KOMPOZIT_VARIANT, ASV3_KOMPOZIT_CROPPED_VARIANT}:
            if any(
                (
                    request.asv3_fast,
                    request.asv3_fast_cropped,
                    request.asv3_exp_cropped,
                    request.asv3_guarded_experimental,
                    request.asv3_guardrails_v2,
                    request.asv3_guardrails_v3,
                )
            ):
                raise OnyxError(
                    OnyxErrorCode.INVALID_INPUT,
                    "Kompozit resume cannot select another workflow.",
                )
            if variant is not None and variant != saved:
                raise OnyxError(
                    OnyxErrorCode.INVALID_INPUT,
                    "Resume requires the same Kompozit workflow.",
                )
            variant = saved
        elif variant is not None:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "Kompozit cannot resume another workflow's checkpoint.",
            )
    if variant is None:
        return None
    spec = MODEL_SPECS["flash"]
    return LLMOverride(
        model_provider_id=resolve_asv3_model_provider_id(
            spec.provider_type, spec.model_name
        ),
        model_provider_type=spec.provider_type,
        model_version=spec.model_name,
        display_name="Kompozit Cropped"
        if variant == ASV3_KOMPOZIT_CROPPED_VARIANT
        else "Kompozit",
    )


def resolve_chat_model_composition(
    request: SendMessageRequest,
    user: User,
    persona: Persona,
    workflow: ASv3WorkflowSelection,
    selected_llm: LLM,
) -> ModelComposition | None:
    if workflow.workflow_variant in {
        ASV3_KOMPOZIT_VARIANT,
        ASV3_KOMPOZIT_CROPPED_VARIANT,
    }:
        if request.asv3_model_profile is not None:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "Kompozit uses a fixed server-side model composition.",
            )
        return resolve_model_composition(
            profile_id=KOMPOZIT_PROFILE_ID,
            user=user,
            persona=persona,
            workflow_variant=workflow.workflow_variant,
        )
    if request.asv3_model_profile is None:
        return None
    # Authenticated API-key/PAT chat requests have their origin normalized by the endpoint.
    if user.role != UserRole.ADMIN or request.origin not in {
        MessageOrigin.BENCHMARK,
        MessageOrigin.API,
    }:
        raise OnyxError(
            OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
            "Model composition benchmarks require an administrator.",
        )
    if workflow.workflow_variant not in {ASV3_FAST_VARIANT, ASV3_FAST_CROPPED_VARIANT}:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Model compositions require an explicit Fast ASv3 workflow.",
        )
    composition = resolve_model_composition(
        profile_id=request.asv3_model_profile,
        user=user,
        persona=persona,
        workflow_variant=workflow.workflow_variant,
    )
    expected = composition.binding("coordinator").llm.config
    selected = selected_llm.config
    if (
        selected.model_provider,
        selected.model_name,
        selected.temperature,
        selected.seed,
    ) != (
        expected.model_provider,
        expected.model_name,
        expected.temperature,
        expected.seed,
    ):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Select the composition's coordinator model and default sampling settings.",
        )
    return composition
