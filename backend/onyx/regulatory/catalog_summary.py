"""Generate a factual discovery overview from the final catalog search hits."""

import json
import re

from sqlalchemy.orm import Session

from onyx.db.llm import can_user_access_llm_provider, fetch_existing_llm_providers
from onyx.llm.constants import LlmProviderNames
from onyx.llm.factory import get_llm_token_counter, llm_from_provider
from onyx.llm.interfaces import LLM
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.utils import llm_response_to_string
from onyx.prompts.catalog_summary import CATALOG_SUMMARY_SYSTEM_PROMPT
from onyx.server.features.regulatory.catalog_models import CatalogSearchPage
from onyx.server.manage.llm.models import LLMProviderView
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response

CATALOG_SUMMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
CATALOG_SUMMARY_MAX_TOKENS = 1800
CATALOG_SUMMARY_TIMEOUT_SECONDS = 90


class CatalogSummaryUnavailable(RuntimeError):
    """A usable source-bound overview could not be produced."""


def get_catalog_summary_llm(db_session: Session) -> LLM:
    """Use only the explicitly configured Ultra model, with provider restrictions."""
    providers = fetch_existing_llm_providers(db_session, flow_type_filter=[])
    for provider in sorted(providers, key=lambda item: item.id):
        if not can_user_access_llm_provider(
            provider, user_group_ids=set(), persona=None, is_admin=True
        ):
            continue
        for model in provider.model_configurations:
            model_name = (
                model.name.removeprefix("openrouter/")
                if provider.provider == LlmProviderNames.OPENROUTER
                else model.name
            )
            if not model.is_visible or model_name != CATALOG_SUMMARY_MODEL:
                continue
            provider_view = LLMProviderView.from_model(provider)
            if model_name != model.name:
                # The LLM factory adds the transport prefix; retain the configured budget.
                provider_view = provider_view.model_copy(
                    update={
                        "model_configurations": [
                            configured.model_copy(update={"name": model_name})
                            if configured.name == model.name
                            else configured
                            for configured in provider_view.model_configurations
                        ]
                    }
                )
            return llm_from_provider(
                model_name,
                provider_view,
                timeout=CATALOG_SUMMARY_TIMEOUT_SECONDS,
                temperature=0,
            )
    raise CatalogSummaryUnavailable(
        "Nemotron özeti için Dil Modelleri ekranında "
        f"{CATALOG_SUMMARY_MODEL} modelini yapılandırıp görünür yapın."
    )


def catalog_summary_messages(
    query: str, search: CatalogSearchPage
) -> list[ChatCompletionMessage]:
    """Keep reference numbers aligned with the returned chunk cards."""
    payload = {
        "query": query,
        "sources": [
            {
                "reference": reference,
                "file_id": str(hit.file.id),
                "filename": hit.file.name,
                "chunk_id": hit.chunk.id,
                "heading_path": hit.chunk.heading_path,
                "text": hit.chunk.text,
                "validity_start_date": hit.chunk.validity_start_date,
                "validity_end_date": hit.chunk.validity_end_date,
                "status": hit.chunk.status,
            }
            for reference, hit in enumerate(search.items, start=1)
        ],
    }
    return [
        SystemMessage(content=CATALOG_SUMMARY_SYSTEM_PROMPT),
        UserMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]


def generate_catalog_summary(
    query: str, search: CatalogSearchPage, db_session: Session
) -> str | None:
    if not search.items:
        return None
    llm = get_catalog_summary_llm(db_session)
    messages = catalog_summary_messages(query, search)
    count_tokens = get_llm_token_counter(llm)
    prompt_tokens = sum(count_tokens(str(message.content)) + 16 for message in messages)
    if prompt_tokens + CATALOG_SUMMARY_MAX_TOKENS > llm.config.max_input_tokens:
        raise CatalogSummaryUnavailable(
            "Bulunan chunkların tamamı yapılandırılan modelin bağlam sınırına sığmadığı "
            "için özet üretilemedi. Arama sonuçlarını inceleyebilirsiniz."
        )
    with llm_generation_span(
        llm=llm, flow=LLMFlow.CATALOG_SEARCH_SUMMARY, input_messages=messages
    ) as span:
        response = llm.invoke(
            messages,
            max_tokens=CATALOG_SUMMARY_MAX_TOKENS,
            timeout_override=CATALOG_SUMMARY_TIMEOUT_SECONDS,
            reasoning_effort=ReasoningEffort.OFF,
            use_streaming=False,
        )
        record_llm_response(span, response)
    summary = llm_response_to_string(response).strip()
    references = [int(value) for value in re.findall(r"\[(\d+)\]", summary)]
    if (
        not summary
        or response.choice.finish_reason == "length"
        or not references
        or any(
            reference < 1 or reference > len(search.items) for reference in references
        )
    ):
        raise CatalogSummaryUnavailable(
            "Nemotron kaynak referansları doğrulanabilen bir özet üretemedi. "
            "Arama sonuçlarını inceleyebilirsiniz."
        )
    return summary
