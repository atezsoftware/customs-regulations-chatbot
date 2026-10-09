"""Fixed model selection for the independent legal-review workflow."""

from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM
from onyx.llm.override_models import LLMOverride

LEGAL_REVIEW_MODEL = "gemini-3.8-flash"


def legal_review_override() -> LLMOverride:
    return LLMOverride(
        model_provider_type="vertex_ai",
        model_version=LEGAL_REVIEW_MODEL,
        temperature=0,
        display_name="Gemini 3.8 Flash",
    )


def require_legal_review_model(llm: LLM) -> None:
    if llm.config.model_name != LEGAL_REVIEW_MODEL:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Hukuki İnceleme requires access to the configured Gemini 3.8 Flash model.",
        )
