"""Generate a factual discovery overview from the final catalog search hits."""

import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import NoReturn, TypedDict

from sqlalchemy.orm import Session

from onyx.db.llm import can_user_access_llm_provider, fetch_existing_llm_providers
from onyx.llm.constants import LlmProviderNames
from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.utils import get_max_input_tokens_from_llm_provider
from onyx.prompts.catalog_summary import CATALOG_SUMMARY_SYSTEM_PROMPT
from onyx.regulatory.catalog_summary_usage import CatalogSummaryUsage
from onyx.server.features.regulatory.catalog_models import CatalogSearchPage
from onyx.server.manage.llm.models import LLMProviderView
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger

CATALOG_SUMMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
CATALOG_SUMMARY_MAX_TOKENS = 160
CATALOG_SUMMARY_TIMEOUT_SECONDS = 90
CATALOG_SUMMARY_MAX_WORDS = 120
_INVALID_EXCERPTS_MESSAGE = (
    "Nemotron kaynak referansları ve alıntıları doğrulanabilen bir özet üretemedi. "
    "Arama sonuçlarını inceleyebilirsiniz."
)
logger = setup_logger()


class CatalogSummaryUnavailable(RuntimeError):
    """A usable source-bound overview could not be produced."""

    def __init__(
        self,
        detail: str = _INVALID_EXCERPTS_MESSAGE,
        *,
        code: "CatalogSummaryRejectionCode | None" = None,
    ) -> None:
        self.code = code or CatalogSummaryRejectionCode.INVALID_SHAPE
        super().__init__(detail)


class CatalogSummaryRejectionCode(StrEnum):
    INVALID_JSON = "invalid_json"
    INVALID_SHAPE = "invalid_shape"
    UNKNOWN_PASSAGE = "unknown_passage"
    DUPLICATE_PASSAGE = "duplicate_passage"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_PASSAGES = "no_passages"
    INCOMPLETE_GENERATION = "incomplete_generation"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONTEXT_EXCEEDED = "context_exceeded"


class CatalogSummaryPassage(TypedDict):
    passage_id: str
    reference: int
    chunk_id: str
    text: str
    start: int
    end: int
    display_words: int


def _reject(code: CatalogSummaryRejectionCode) -> NoReturn:
    logger.warning("Catalog summary selection rejected code=%s", code.value)
    raise CatalogSummaryUnavailable(code=code)


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
        f"{CATALOG_SUMMARY_MODEL} modelini yapılandırıp görünür yapın.",
        code=CatalogSummaryRejectionCode.MODEL_UNAVAILABLE,
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    segment = text[start:end]
    trimmed_start = start + len(segment) - len(segment.lstrip())
    trimmed_end = end - (len(segment) - len(segment.rstrip()))
    return (trimmed_start, trimmed_end) if trimmed_start < trimmed_end else None


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n+", text):
        if span := _trim_span(text, start, separator.start()):
            spans.append(span)
        start = separator.end()
    if span := _trim_span(text, start, len(text)):
        spans.append(span)
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current = start
    for boundary in re.finditer(
        r"""(?P<stop>[.!?][”"')\]*]*)\s+(?=[A-ZÇĞİÖŞÜ0-9“"(*])""",
        text[start:end],
    ):
        sentence_end = start + boundary.end("stop")
        prefix = text[current:sentence_end].rstrip("”\"')]*")
        last_word = re.search(r"(\w+(?:\.\w+)*)[.!?]$", prefix)
        if last_word:
            token = last_word.group(1)
            if (
                token.isdigit()
                or len(token) == 1
                or token.casefold()
                in {
                    "m",
                    "md",
                    "no",
                    "dr",
                    "av",
                    "prof",
                    "t.c",
                    "bkz",
                    "vb",
                    "vs",
                }
            ):
                continue
        if span := _trim_span(text, current, sentence_end):
            spans.append(span)
        current = start + boundary.end()
    if span := _trim_span(text, current, end):
        spans.append(span)
    return spans


def build_catalog_summary_passages(
    search: CatalogSearchPage,
) -> list[CatalogSummaryPassage]:
    """Offer complete source units with short IDs owned by this search response."""
    passages: list[CatalogSummaryPassage] = []
    for reference, hit in enumerate(search.items, start=1):
        text = hit.chunk.text
        for paragraph_index, (start, end) in enumerate(_paragraph_spans(text), start=1):
            spans = (
                [(start, end)]
                if len(text[start:end].split()) + 3 <= CATALOG_SUMMARY_MAX_WORDS
                else _sentence_spans(text, start, end)
            )
            for unit_index, (unit_start, unit_end) in enumerate(spans, start=1):
                unit = text[unit_start:unit_end]
                display_words = len(unit.split()) + 3
                if display_words > CATALOG_SUMMARY_MAX_WORDS:
                    continue
                passage_id = f"s{reference}p{paragraph_index}"
                if len(spans) > 1:
                    passage_id += f"u{unit_index}"
                passages.append(
                    CatalogSummaryPassage(
                        passage_id=passage_id,
                        reference=reference,
                        chunk_id=hit.chunk.id,
                        text=unit,
                        start=unit_start,
                        end=unit_end,
                        display_words=display_words,
                    )
                )
    return passages


def catalog_summary_messages(
    query: str, search: CatalogSearchPage
) -> list[ChatCompletionMessage]:
    """Send every source character once, with eligible source-owned passage IDs."""
    passages_by_reference: dict[int, list[CatalogSummaryPassage]] = defaultdict(list)
    for passage in build_catalog_summary_passages(search):
        passages_by_reference[passage["reference"]].append(passage)
    sources: list[dict[str, object]] = []
    for reference, hit in enumerate(search.items, start=1):
        text = hit.chunk.text
        segments: list[dict[str, object]] = []
        cursor = 0
        for passage in passages_by_reference[reference]:
            if cursor < passage["start"]:
                segments.append(
                    {"text": text[cursor : passage["start"]], "eligible": False}
                )
            segments.append(
                {
                    "passage_id": passage["passage_id"],
                    "text": passage["text"],
                    "eligible": True,
                    "display_words": passage["display_words"],
                }
            )
            cursor = passage["end"]
        if cursor < len(text) or not segments:
            segments.append({"text": text[cursor:], "eligible": False})
        sources.append(
            {
                "reference": reference,
                "file_id": str(hit.file.id),
                "filename": hit.file.name,
                "chunk_id": hit.chunk.id,
                "heading_path": hit.chunk.heading_path,
                "segments": segments,
                "validity_start_date": hit.chunk.validity_start_date,
                "validity_end_date": hit.chunk.validity_end_date,
                "status": hit.chunk.status,
            }
        )
    payload = {
        "query": query,
        "word_budget": CATALOG_SUMMARY_MAX_WORDS,
        "sources": sources,
    }
    return [
        SystemMessage(content=CATALOG_SUMMARY_SYSTEM_PROMPT),
        UserMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]


def strict_catalog_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _reject(CatalogSummaryRejectionCode.INVALID_JSON)
        result[key] = value
    return result


def render_catalog_passage_summary(content: str, search: CatalogSearchPage) -> str:
    """Resolve selected IDs only against the server's original source passages."""
    try:
        payload: object = json.loads(
            content, object_pairs_hook=strict_catalog_json_object
        )
    except json.JSONDecodeError:
        _reject(CatalogSummaryRejectionCode.INVALID_JSON)
    if not isinstance(payload, dict) or set(payload) != {"passage_ids"}:
        _reject(CatalogSummaryRejectionCode.INVALID_SHAPE)
    selected = payload["passage_ids"]
    if not isinstance(selected, list) or not 1 <= len(selected) <= 3:
        _reject(CatalogSummaryRejectionCode.INVALID_SHAPE)
    passages = {
        passage["passage_id"]: passage
        for passage in build_catalog_summary_passages(search)
    }
    chosen: list[CatalogSummaryPassage] = []
    seen: set[str] = set()
    for passage_id in selected:
        if not isinstance(passage_id, str):
            _reject(CatalogSummaryRejectionCode.INVALID_SHAPE)
        if passage_id in seen:
            _reject(CatalogSummaryRejectionCode.DUPLICATE_PASSAGE)
        if passage_id not in passages:
            _reject(CatalogSummaryRejectionCode.UNKNOWN_PASSAGE)
        seen.add(passage_id)
        chosen.append(passages[passage_id])
    rendered: list[str] = []
    words = 0
    for passage in chosen:
        if words + passage["display_words"] > CATALOG_SUMMARY_MAX_WORDS:
            continue
        rendered.append(f"[{passage['reference']}] numaralı parça: “{passage['text']}”")
        words += passage["display_words"]
    return " ".join(rendered)


def observe_catalog_summary_usage(
    response: ModelResponse,
    observer: Callable[[CatalogSummaryUsage], None] | None,
) -> None:
    if observer is None:
        return
    usage = response.usage
    cost = usage.cost if usage else None
    if cost is not None and (not math.isfinite(cost) or cost < 0):
        cost = None
    event = CatalogSummaryUsage(
        response_id=response.id or None,
        input_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
        provider_cost_usd=cost,
    )
    try:
        observer(event)
    except Exception as error:
        logger.warning(
            "Catalog summary usage observer failed: %s", type(error).__name__
        )


def generate_catalog_summary(
    query: str,
    search: CatalogSearchPage,
    db_session: Session,
    *,
    usage_observer: Callable[[CatalogSummaryUsage], None] | None = None,
    overview: bool = False,
) -> str | None:
    if overview:
        from onyx.regulatory.catalog_overview import generate_catalog_overview

        return generate_catalog_overview(
            query, search, db_session, usage_observer=usage_observer
        )
    if not search.items:
        return None
    if not build_catalog_summary_passages(search):
        _reject(CatalogSummaryRejectionCode.NO_PASSAGES)
    llm = get_catalog_summary_llm(db_session)
    messages = catalog_summary_messages(query, search)
    count_tokens = get_llm_token_counter(llm)
    prompt_tokens = sum(count_tokens(str(message.content)) + 16 for message in messages)
    if prompt_tokens + CATALOG_SUMMARY_MAX_TOKENS > llm.config.max_input_tokens:
        raise CatalogSummaryUnavailable(
            "Bulunan chunkların tamamı yapılandırılan modelin bağlam sınırına sığmadığı "
            "için özet üretilemedi. Arama sonuçlarını inceleyebilirsiniz.",
            code=CatalogSummaryRejectionCode.CONTEXT_EXCEEDED,
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
        observe_catalog_summary_usage(response, usage_observer)
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
        _reject(CatalogSummaryRejectionCode.INCOMPLETE_GENERATION)
    return render_catalog_passage_summary(content, search)
