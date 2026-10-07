"""Authorized model selection for the isolated source-to-answer experiment."""

from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.db.asv3_models import default_asv3_persona
from onyx.db.models import User
from onyx.llm.factory import get_llm_for_persona
from onyx.llm.interfaces import LLM
from onyx.llm.override_models import LLMOverride


def uses_source_answer_model(workflow_variant: str, llm: LLM) -> bool:
    return (
        workflow_variant == ASV3_TUNED_VARIANT
        and llm.config.model_provider == "vertex_ai"
        and llm.config.model_name == "gemini-3.8-flash"
    )


def source_answer_model(user: User) -> LLM:
    return get_llm_for_persona(
        default_asv3_persona(),
        user,
        llm_override=LLMOverride(
            model_provider_type="anthropic", model_version="claude-sonnet-5-5"
        ),
    )
