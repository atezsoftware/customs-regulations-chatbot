"""Authorized opt-in model compositions for Fast ASv3 benchmark requests."""

from onyx.asv3.model_composition.models import ModelComposition
from onyx.asv3.model_composition.resolver import resolve_model_composition
from onyx.asv3.models import ASv3WorkflowSelection
from onyx.asv3.workflow_variant import ASV3_FAST_CROPPED_VARIANT, ASV3_FAST_VARIANT
from onyx.auth.schemas import UserRole
from onyx.db.models import Persona, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM
from onyx.server.query_and_chat.models import MessageOrigin, SendMessageRequest


def resolve_chat_model_composition(
    request: SendMessageRequest,
    user: User,
    persona: Persona,
    workflow: ASv3WorkflowSelection,
    selected_llm: LLM,
) -> ModelComposition | None:
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
