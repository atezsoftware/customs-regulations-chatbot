"""The batch selects an allowed Vertex model, never the tenant chat default.

Segmentation, target confirmation and drafting all depend on reliable
structured output over Turkish legal text. Taking whichever model happens to
be the tenant's default silently moves this pipeline onto an unrelated
provider — an OpenRouter entry, for instance — where the same instructions
return well-formed but empty results and every change looks unmatched.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.llm import fetch_vertex_model_configuration
from onyx.llm.factory import llm_from_provider
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.model_choice import AmendmentAnalysisModel
from onyx.server.manage.llm.models import LLMProviderView

AMENDMENT_ANALYSIS_MODEL = AmendmentAnalysisModel.FLASH.value
_ANALYSIS_MODEL = ContextVar(
    "amendment_analysis_model", default=AmendmentAnalysisModel.FLASH
)


@contextmanager
def use_analysis_model(model: str) -> Iterator[None]:
    token = _ANALYSIS_MODEL.set(AmendmentAnalysisModel(model))
    try:
        yield
    finally:
        _ANALYSIS_MODEL.reset(token)


def get_amendment_analysis_llm(
    *,
    temperature: float | None = 0.0,
    timeout: int | None = None,
    model: AmendmentAnalysisModel | None = None,
) -> LLM:
    """Return the batch's Vertex analysis model, or explain what is missing."""

    selected = (model or _ANALYSIS_MODEL.get()).value
    with get_session_with_current_tenant() as db_session:
        configuration = fetch_vertex_model_configuration(db_session, selected)
        if configuration is None:
            raise RuntimeError(
                "Amendment analysis requires the Vertex AI model "
                f"{selected}. Configure it on the Gemini (Vertex "
                "AI) provider; the tenant default model is deliberately not used."
            )
        return llm_from_provider(
            model_name=configuration.name,
            llm_provider=LLMProviderView.from_model(configuration.llm_provider),
            temperature=temperature,
            timeout=timeout,
        )
