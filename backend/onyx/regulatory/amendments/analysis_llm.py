"""The amendment analysis model is pinned, never inherited from the default.

Segmentation, target confirmation and drafting all depend on reliable
structured output over Turkish legal text. Taking whichever model happens to
be the tenant's default silently moves this pipeline onto an unrelated
provider — an OpenRouter entry, for instance — where the same instructions
return well-formed but empty results and every change looks unmatched.
"""

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.llm import fetch_vertex_model_configuration
from onyx.llm.factory import llm_from_provider
from onyx.llm.interfaces import LLM
from onyx.regulatory.labeling.provider import DEFAULT_MODEL
from onyx.server.manage.llm.models import LLMProviderView

AMENDMENT_ANALYSIS_MODEL = DEFAULT_MODEL


def get_amendment_analysis_llm(
    *, temperature: float | None = 0.0, timeout: int | None = None
) -> LLM:
    """Return the pinned Vertex AI analysis model, or explain what is missing."""

    with get_session_with_current_tenant() as db_session:
        model = fetch_vertex_model_configuration(db_session, AMENDMENT_ANALYSIS_MODEL)
        if model is None:
            raise RuntimeError(
                "Amendment analysis requires the Vertex AI model "
                f"{AMENDMENT_ANALYSIS_MODEL}. Configure it on the Gemini (Vertex "
                "AI) provider; the tenant default model is deliberately not used."
            )
        return llm_from_provider(
            model_name=model.name,
            llm_provider=LLMProviderView.from_model(model.llm_provider),
            temperature=temperature,
            timeout=timeout,
        )
