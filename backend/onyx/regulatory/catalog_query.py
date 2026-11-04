"""Optional one-call query preparation; explicit lookups bypass generation."""

import json
import re

from sqlalchemy.orm import Session

from onyx.db.llm import can_user_access_llm_provider, fetch_existing_llm_providers
from onyx.llm.constants import LlmProviderNames
from onyx.llm.interfaces import LLM
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.utils import llm_response_to_string
from onyx.prompts.catalog_query import CATALOG_QUERY_SYSTEM_PROMPT
from onyx.regulatory.catalog_query_models import CatalogQueryPreparation
from onyx.server.manage.llm.models import LLMProviderView
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger

QUERY_PREPARATION_MODELS = (
    "deepseek/deepseek-v4.1-flash:floor",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-5.6-luna",
)
QUERY_PREPARATION_MAX_TOKENS = 160
QUERY_PREPARATION_TIMEOUT_SECONDS = 15
_DESCRIPTION_CUES = (
    "şöyle bir",
    "böyle bir",
    "bir şey vardı",
    "birşey vardı",
    "tam adını",
    "adını hatırla",
    "hatırlamıyorum",
    "bulamıyorum",
    "durum şöyle",
    "konu şu",
    "şu durumda",
    "derdimi",
    "şöyle anlat",
    "bir düzenleme vardı",
)
logger = setup_logger()


def catalog_query_needs_preparation(query: str) -> bool:
    """Spend nothing on identifiers, quoted titles, or concise keyword lookups."""
    if re.search(r'"[^"\n]{2,}"|“[^”\n]{2,}”', query):
        return False
    normalized = query.casefold().replace("i\u0307", "i")
    if (
        re.fullmatch(r"\s*\d{3,}\s*", normalized)
        or re.search(r"\b\d{3,}\s+(?:sayılı|numaralı)\b", normalized)
        or re.search(
            r"\b\d{4}[/.-]\d{1,4}\s+(?:sayılı\s+)?(?:tebliğ|genelge|karar)\b",
            normalized,
        )
        or re.search(
            r"\b(?:kanun|tebliğ|genelge|karar|gtip|tarife)\s+(?:no[.:]?\s*)?\d{3,}\b",
            normalized,
        )
    ):
        return False
    if any(cue in normalized for cue in _DESCRIPTION_CUES):
        return True
    words = re.findall(r"\w+", normalized)
    if len(words) <= 12:
        return False
    if len(words) >= 24:
        return True
    return any(
        cue in normalized
        for cue in ("istiyorum", "hatırl", "yaşadım", "olmuştu", "hangi", "nasıl")
    )


def get_catalog_query_llm(db_session: Session) -> LLM | None:
    providers = fetch_existing_llm_providers(db_session, flow_type_filter=[])
    for preferred_model in QUERY_PREPARATION_MODELS:
        for provider in sorted(providers, key=lambda item: item.id):
            if provider.provider != LlmProviderNames.OPENROUTER:
                continue
            if not can_user_access_llm_provider(
                provider, user_group_ids=set(), persona=None, is_admin=True
            ):
                continue
            for model in provider.model_configurations:
                model_name = model.name.removeprefix("openrouter/")
                if not model.is_visible or model_name != preferred_model:
                    continue
                configured = LLMProviderView.from_model(provider)
                # This instance alone disables thinking and retries, including when
                # LiteLLM's model registry does not know this provider model yet.
                return LitellmLLM(
                    api_key=configured.api_key,
                    model_provider=configured.provider,
                    model_name=model_name,
                    max_input_tokens=model.max_input_tokens or 8192,
                    timeout=QUERY_PREPARATION_TIMEOUT_SECONDS,
                    api_base=configured.api_base,
                    api_version=configured.api_version,
                    deployment_name=configured.deployment_name,
                    custom_config=configured.custom_config,
                    extra_headers={
                        "HTTP-Referer": "https://onyx.app",
                        "X-Title": "Onyx",
                    },
                    extra_body={"reasoning": {"enabled": False}},
                    model_kwargs={"num_retries": 0, "max_retries": 0},
                    temperature=0,
                )
    return None


def _valid_prepared_query(original: str, prepared: str) -> bool:
    if (
        not prepared
        or len(prepared) > 2000
        or "\n" in prepared
        or "\r" in prepared
        or "`" in prepared
        or prepared.startswith(("{", "[", "- "))
    ):
        return False
    # Preserve supplied numeric identities; never accept an invented legal number.
    numbers = re.compile(r"\d+(?:[./,-]\d+)*")
    if set(numbers.findall(prepared)) != set(numbers.findall(original)):
        return False
    acronyms = re.findall(r"\b[A-ZÇĞİÖŞÜ]{2,}\b", original)
    return all(acronym in prepared for acronym in acronyms)


def prepare_catalog_search_query(
    query: str, db_session: Session
) -> CatalogQueryPreparation:
    original = CatalogQueryPreparation(query=query, rewritten=False)
    if not catalog_query_needs_preparation(query):
        return original
    try:
        llm = get_catalog_query_llm(db_session)
        if llm is None:
            logger.warning(
                "Catalog query preparation has no configured inexpensive model"
            )
            return original
        payload = json.dumps({"user_request": query}, ensure_ascii=False)
        # Every UTF-8 byte can be a token; reject a configured context too small
        # for the complete request rather than dropping the user's constraints.
        if (
            len((CATALOG_QUERY_SYSTEM_PROMPT + payload).encode("utf-8"))
            + QUERY_PREPARATION_MAX_TOKENS
            + 32
            > llm.config.max_input_tokens
        ):
            return original
        messages: list[ChatCompletionMessage] = [
            SystemMessage(content=CATALOG_QUERY_SYSTEM_PROMPT),
            UserMessage(content=payload),
        ]
        with llm_generation_span(
            llm=llm, flow=LLMFlow.CATALOG_QUERY_PREPARATION, input_messages=messages
        ) as span:
            response = llm.invoke(
                messages,
                max_tokens=QUERY_PREPARATION_MAX_TOKENS,
                timeout_override=QUERY_PREPARATION_TIMEOUT_SECONDS,
                reasoning_effort=ReasoningEffort.OFF,
                use_streaming=False,
                provider_compatibility_attempts=1,
            )
            record_llm_response(span, response)
        prepared = llm_response_to_string(response).strip()
        if response.choice.finish_reason != "stop" or not _valid_prepared_query(
            query, prepared
        ):
            logger.warning(
                "Catalog query preparation rejected incomplete or invalid output"
            )
            return original
        return CatalogQueryPreparation(query=prepared, rewritten=prepared != query)
    except Exception as error:
        logger.warning("Catalog query preparation failed: %s", type(error).__name__)
        return original
