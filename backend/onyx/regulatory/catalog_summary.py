"""Generate a factual discovery overview from the final catalog search hits."""

import json

from sqlalchemy.orm import Session

from onyx.db.llm import can_user_access_llm_provider, fetch_existing_llm_providers
from onyx.llm.constants import LlmProviderNames
from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.utils import get_max_input_tokens_from_llm_provider
from onyx.prompts.catalog_summary import CATALOG_SUMMARY_SYSTEM_PROMPT
from onyx.server.features.regulatory.catalog_models import CatalogSearchPage
from onyx.server.manage.llm.models import LLMProviderView
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger

CATALOG_SUMMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
CATALOG_SUMMARY_MAX_TOKENS = 1800
CATALOG_SUMMARY_TIMEOUT_SECONDS = 90
CATALOG_SUMMARY_MAX_WORDS = 120
_INVALID_EXCERPTS_MESSAGE = (
    "Nemotron kaynak referansları ve alıntıları doğrulanabilen bir özet üretemedi. "
    "Arama sonuçlarını inceleyebilirsiniz."
)
logger = setup_logger()


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
            return LitellmLLM(
                api_key=provider_view.api_key,
                model_provider=provider_view.provider,
                model_name=model_name,
                max_input_tokens=get_max_input_tokens_from_llm_provider(
                    provider_view, model.name
                ),
                timeout=CATALOG_SUMMARY_TIMEOUT_SECONDS,
                api_base=provider_view.api_base,
                api_version=provider_view.api_version,
                deployment_name=provider_view.deployment_name,
                custom_config=provider_view.custom_config,
                extra_headers=(
                    {"HTTP-Referer": "https://onyx.app", "X-Title": "Onyx"}
                    if provider_view.provider == LlmProviderNames.OPENROUTER
                    else None
                ),
                extra_body=(
                    {"reasoning": {"enabled": False}}
                    if provider_view.provider == LlmProviderNames.OPENROUTER
                    else None
                ),
                model_kwargs={"num_retries": 0, "max_retries": 0},
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


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
        result[key] = value
    return result


def render_catalog_excerpt_summary(content: str, search: CatalogSearchPage) -> str:
    """Bind every displayed quotation to its exact source, without repairing it."""
    try:
        payload: object = json.loads(content, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError:
        raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE) from None
    if not isinstance(payload, dict) or set(payload) != {"excerpts"}:
        raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
    excerpts = payload["excerpts"]
    if not isinstance(excerpts, list) or not 1 <= len(excerpts) <= 3:
        raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
    rendered: list[str] = []
    references: set[int] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, dict) or set(excerpt) != {"reference", "quote"}:
            raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
        reference = excerpt["reference"]
        quote = excerpt["quote"]
        if (
            not isinstance(reference, int)
            or isinstance(reference, bool)
            or not 1 <= reference <= len(search.items)
            or reference in references
            or not isinstance(quote, str)
        ):
            raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
        normalized_quote = " ".join(quote.split())
        source_text = " ".join(search.items[reference - 1].chunk.text.split())
        if not normalized_quote or normalized_quote not in source_text:
            raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
        references.add(reference)
        rendered.append(f"[{reference}] numaralı parça: “{normalized_quote}”")
    result = " ".join(rendered)
    if len(result.split()) > CATALOG_SUMMARY_MAX_WORDS:
        raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
    return result


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
            provider_compatibility_attempts=1,
        )
        record_llm_response(span, response)
    content = response.choice.message.content
    if (
        response.choice.finish_reason != "stop"
        or response.choice.message.tool_calls
        or not isinstance(content, str)
    ):
        logger.warning(
            "Catalog excerpt selection rejected finish_reason=%s has_content=%s",
            response.choice.finish_reason,
            bool(response.choice.message.content),
        )
        raise CatalogSummaryUnavailable(_INVALID_EXCERPTS_MESSAGE)
    return render_catalog_excerpt_summary(content, search)
