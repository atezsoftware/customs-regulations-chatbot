"""
An explanation of the search tool found below:

Step 1: Queries
- The LLM will generate some queries based on the chat history for what it thinks are the best things to search for.
This has a pretty generic prompt so it's not perfectly tuned for search but provides breadth and also the LLM can often break up
the query into multiple searches which the other flows do not do. Exp: Compare the sales process between company X and Y can be
broken up into "sales process company X" and "sales process company Y".
- A specifial prompt and history is used to generate another query which is best tuned for a semantic/hybrid search pipeline.
- A small set of keyword emphasized queries are also generated to cover additional breadth. This is important for cases where
the query is short, keyword heavy, or has a lot of model unseen terminology.

Step 2: Recombination
We use a weighted RRF to combine the search results from the queries above. Each query will have a list of search results with
some scores however these are downstream of a normalization step so they cannot easily be compared with one another on an
absolute scale. RRF is a good way to combine these and allows us to give some custom weightings. We also merge document chunks
that are adjacent to provide more continuous context to the LLM.

Step 3: Selection
We pass the recombined results (truncated set) to the LLM to select the most promising ones to read. This is to reduce noise and
reduce downstream chances of hallucination. The LLM at this point also has the entire set of document chunks so it has
information across documents not just per document. This also reduces the number of tokens required for the next step.

Step 4: Expansion
For the selected documents, we pass the main retrieved sections from above (this may be a single chunk or a section comprised of
several consecutive chunks) along with chunks above and below the section to the LLM. The LLM determines how much of the document
it wants to read. This is done in parallel for all selected documents. Reason being that the LLM would not be able to do a good
job of this with all of the documents in the prompt at once. Keeping every LLM decision step as simple as possible is key for
reliable performance.

Step 5: Prompt Building
We construct a response string back to the LLM as the result of the tool call. We also pass relevant richer objects back
so that the rest of the code can persist it, render it in the UI, etc. The response is a json that makes it easy for the LLM to
refer to by using matching keywords to other parts of the prompt and reminders.
"""

import json
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from onyx.chat.emitter import Emitter
from onyx.configs.chat_configs import MAX_CHUNKS_FED_TO_CHAT, MAX_SEARCH_QUERY_LANES
from onyx.configs.constants import DocumentSource, FederatedConnectorSource
from onyx.context.search.federated.slack_search import slack_retrieval
from onyx.context.search.models import (
    BaseFilters,
    ChunkIndexRequest,
    ChunkSearchRequest,
    IndexFilters,
    InferenceChunk,
    InferenceSection,
    PersonaSearchInfo,
    SearchDocsResponse,
)
from onyx.context.search.pipeline import search_pipeline
from onyx.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from onyx.context.search.utils import (
    convert_inference_sections_to_search_docs,
    inference_section_from_single_chunk,
    populate_file_ids_on_sections,
)
from onyx.db.connector import (
    check_connectors_exist,
    check_federated_connectors_exist,
    fetch_unique_document_sources,
)
from onyx.db.document_set import filter_document_set_names_by_user_access
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.federated import (
    get_federated_connector_document_set_mappings_by_document_set_names,
    list_federated_connector_oauth_tokens,
)
from onyx.db.models import SearchSettings, User
from onyx.db.regulatory_chunks import get_visible_regulatory_chunk_ids
from onyx.db.reranking import get_reranker_configuration
from onyx.db.search_settings import get_current_search_settings
from onyx.db.slack_bot import fetch_slack_bots
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.federated_connectors.federated_retrieval import (
    FederatedRetrievalInfo,
    get_federated_retrieval_functions,
)
from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.onyxbot.slack.models import SlackContext
from onyx.regulatory.heading_path import (
    RegulatoryArticleHeading,
    RegulatoryProvisionReference,
    extract_single_regulatory_provision_reference,
    normalize_regulatory_heading_path,
    parse_regulatory_article_heading,
    regulatory_heading_path_matches_reference,
)
from onyx.regulatory.provision_retrieval import (
    RegulatoryProvisionNavigation,
    build_regulatory_provision_navigation,
    expand_selected_regulatory_references,
    expand_selected_regulatory_sections,
    regulatory_provision_navigation_payload,
)
from onyx.reranking.diversity import apply_soft_diversity
from onyx.reranking.service import rerank_chunks
from onyx.secondary_llm_flows.document_filter import (
    select_chunks_for_relevance,
    select_sections_for_expansion,
)
from onyx.secondary_llm_flows.query_expansion import (
    keyword_query_expansion,
    semantic_query_rephrase,
)
from onyx.secondary_llm_flows.source_filter import SearchCycle, decide_search_scope
from onyx.secondary_llm_flows.time_filter import TimeFilter, decide_time_filter
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    SearchToolDocumentsDelta,
    SearchToolFilterDelta,
    SearchToolQueriesDelta,
    SearchToolStart,
)
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatMinimalTextMessage,
    SearchToolOverrideKwargs,
    ToolCallException,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.constants import (
    KEYWORD_QUERY_HYBRID_ALPHA,
    LLM_KEYWORD_QUERY_WEIGHT,
    LLM_NON_CUSTOM_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    MAX_CHUNKS_FOR_RELEVANCE,
    ORIGINAL_QUERY_WEIGHT,
)
from onyx.tools.tool_implementations.search.search_utils import (
    weighted_reciprocal_rank_fusion,
)
from onyx.tools.tool_implementations.utils import (
    convert_inference_sections_to_llm_string,
)
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from onyx.utils.timing import log_function_time
from shared_configs.configs import (
    DOC_EMBEDDING_CONTEXT_SIZE,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
)

logger = setup_logger()

QUERIES_FIELD = "queries"
SEARCH_MODE_FIELD = "search_mode"
COVERAGE_ITEM_FIELD = "coverage_item"
EVIDENCE_TARGET_FIELD = "evidence_target"
SEARCH_MODES = {"hybrid", "keyword", "full_text"}
_REGULATORY_PROVISION_OVERFETCH_FACTOR = 4
_REGULATORY_PROVISION_MAX_CANDIDATES = 128
_REGULATORY_PROVISION_FAMILY_SEED_LIMIT = 4
_REGULATORY_KEYWORD_QUERY_HYBRID_ALPHA = 0.0
_REGULATORY_SEARCH_DESCRIPTION = (
    "Search administrator-indexed regulatory chunks for evidence. You decide "
    "whether a search or materially different retry is useful and when the "
    "evidence is sufficient. Write the focused query and select its retrieval "
    "mode yourself. Independent calls may run in parallel."
)

DecisionT = TypeVar("DecisionT")


class _SharedDecision(Generic[DecisionT]):
    """Compute one decision once while concurrent search forks share the result."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._computing = False
        self._computed = False
        self._value: DecisionT | None = None
        self._error: BaseException | None = None

    def get_or_compute(self, compute: Callable[[], DecisionT]) -> DecisionT:
        with self._condition:
            if self._computed:
                if self._error is not None:
                    raise self._error
                return cast(DecisionT, self._value)
            if self._computing:
                self._condition.wait_for(lambda: self._computed)
                if self._error is not None:
                    raise self._error
                return cast(DecisionT, self._value)
            self._computing = True

        try:
            value = compute()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._computed = True
                self._condition.notify_all()
            raise

        with self._condition:
            self._value = value
            self._computed = True
            self._condition.notify_all()
        return value


def _validate_search_queries(
    llm_kwargs: dict[str, Any],
    *,
    regulatory_chunks_only: bool = True,
) -> list[str]:
    if QUERIES_FIELD not in llm_kwargs:
        example = (
            '{"queries": ["one focused search query"]}'
            if regulatory_chunks_only
            else '{"queries": ["your search query here"]}'
        )
        requirement = (
            "a single focused search query"
            if regulatory_chunks_only
            else "an array of search queries"
        )
        raise ToolCallException(
            message=f"Missing required '{QUERIES_FIELD}' parameter in internal_search tool call",
            llm_facing_message=(
                f"The internal_search tool requires a '{QUERIES_FIELD}' parameter "
                f"containing {requirement}. Please provide the queries like: {example}"
            ),
        )

    raw_queries = llm_kwargs[QUERIES_FIELD]
    if not regulatory_chunks_only:
        return cast(list[str], raw_queries)

    if (
        not isinstance(raw_queries, list)
        or len(raw_queries) != 1
        or not isinstance(raw_queries[0], str)
        or not raw_queries[0].strip()
    ):
        raise ToolCallException(
            message="internal_search requires exactly one non-empty query per call",
            llm_facing_message=(
                "Call internal_search with exactly one focused query. Do not combine "
                "different requested items or independent legal issues in one call. "
                'Use: {"queries": ["one focused search query"]}. Make another tool '
                "call for the next issue."
            ),
        )

    query = raw_queries[0].strip()
    if len(query) > REGULATORY_MAX_SEARCH_QUERY_CHARS:
        raise ToolCallException(
            message=(
                "Regulatory internal_search query exceeds the focused-query "
                f"limit of {REGULATORY_MAX_SEARCH_QUERY_CHARS} characters"
            ),
            llm_facing_message=(
                "This internal_search query is too broad. Rewrite it as one "
                "focused retrieval fragment containing only the source or "
                "mechanism anchor and the operative relationship needed for "
                "this issue, then retry. Do not paste the full user narrative."
            ),
        )

    return [query]


def _validate_search_mode(
    llm_kwargs: dict[str, Any],
    *,
    regulatory_chunks_only: bool = True,
) -> str:
    if not regulatory_chunks_only:
        return "hybrid"

    raw_mode = llm_kwargs.get(SEARCH_MODE_FIELD)
    if not isinstance(raw_mode, str) or raw_mode not in SEARCH_MODES:
        raise ToolCallException(
            message=f"Invalid internal_search mode: {raw_mode}",
            llm_facing_message=(
                "Every internal_search call must explicitly set search_mode to one "
                "of: hybrid, keyword, "
                "full_text. Use keyword for exact codes or identifiers, full_text "
                "for high-coverage lexical matching, and hybrid for conceptual "
                "discovery."
            ),
        )
    return raw_mode


def _validate_coverage_item(llm_kwargs: dict[str, Any]) -> str:
    raw_item = llm_kwargs.get(COVERAGE_ITEM_FIELD)
    if not isinstance(raw_item, str) or not raw_item.strip():
        raise ToolCallException(
            message="internal_search requires a non-empty coverage_item",
            llm_facing_message=(
                "Every internal_search call must set coverage_item to a short "
                "description of the user-facing issue this search informs."
            ),
        )
    return raw_item.strip()


def _validate_evidence_target(llm_kwargs: dict[str, Any]) -> str:
    raw_target = llm_kwargs.get(EVIDENCE_TARGET_FIELD)
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ToolCallException(
            message="internal_search requires a non-empty evidence_target",
            llm_facing_message=(
                "Every internal_search call must set evidence_target to the one "
                "independent legal test this call is trying to resolve within its "
                "coverage_item. State it briefly in your own words."
            ),
        )
    return raw_target.strip()


def _normalize_exact_search_query(query: str) -> str:
    without_quotes = re.sub(r'["“”]', " ", query)
    without_grouping = re.sub(r"[(){}\[\]]", " ", without_quotes)
    without_boolean_syntax = re.sub(
        r"\b(?:AND|OR|NOT)\b", " ", without_grouping, flags=re.IGNORECASE
    )
    return " ".join(without_boolean_syntax.split())


def _prepare_search_query(query: str, _search_mode: str) -> str:
    # Retrieval mode controls matching; Boolean-looking text is otherwise just
    # noisy query content for both lexical and semantic paths.
    return _normalize_exact_search_query(query)


def _add_search_receipt(
    response: str,
    *,
    coverage_item: str,
    evidence_target: str,
) -> str:
    """Add call intent without changing the search result or citation payloads."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Could not add internal_search receipt to non-JSON response")
        return response

    if not isinstance(payload, dict):
        logger.warning("Could not add internal_search receipt to non-object response")
        return response

    response_with_receipt: dict[str, Any] = {
        "receipt": {
            COVERAGE_ITEM_FIELD: coverage_item,
            EVIDENCE_TARGET_FIELD: evidence_target,
        }
    }
    response_with_receipt.update(payload)
    return json.dumps(response_with_receipt, indent=2, ensure_ascii=False)


def _add_regulatory_provision_navigation(
    response: str,
    navigation: RegulatoryProvisionNavigation | None,
) -> str:
    """Add metadata-only discovery leads without changing result citations."""
    if navigation is None:
        return response

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        logger.warning(
            "Could not add regulatory provision navigation to non-JSON response"
        )
        return response
    if not isinstance(payload, dict):
        logger.warning(
            "Could not add regulatory provision navigation to non-object response"
        )
        return response

    payload["regulatory_provision_navigation"] = (
        regulatory_provision_navigation_payload(navigation)
    )
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _can_use_ranked_regulatory_selection(
    sections: list[InferenceSection], *, focused_search: bool
) -> bool:
    """Avoid a redundant selector LLM for bounded, model-written chunk queries."""

    return (
        focused_search
        and bool(sections)
        and all(
            section.center_chunk.regulatory_chunk_id is not None for section in sections
        )
    )


def _reserve_ranked_regulatory_seeds(
    sections: list[InferenceSection], max_total_sections: int
) -> list[InferenceSection]:
    """Leave bounded room for paragraphs adjacent to the best legal hits."""

    if max_total_sections <= 0:
        return []
    seed_budget = max(1, (max_total_sections + 1) // 2)
    return sections[:seed_budget]


def _regulatory_reference_expansion_limit(
    selected_section_count: int, max_total_sections: int
) -> int:
    """Share spare evidence slots between one-hop references and seed families."""

    if max_total_sections <= 0:
        return 0
    bounded_selected_count = min(max(selected_section_count, 0), max_total_sections)
    remaining_slots = max_total_sections - bounded_selected_count
    reference_slots = (remaining_slots + 1) // 2
    return bounded_selected_count + reference_slots


def _backfill_ranked_regulatory_sections(
    selected: list[InferenceSection],
    ranked: list[InferenceSection],
    max_total_sections: int,
) -> list[InferenceSection]:
    """Use remaining ranked hits when a selected provision has no siblings."""

    if max_total_sections <= 0:
        return []

    result = selected[:max_total_sections]
    if len(result) >= max_total_sections:
        return result

    seen = {
        (
            section.center_chunk.document_id,
            section.center_chunk.regulatory_chunk_id
            if section.center_chunk.regulatory_chunk_id is not None
            else section.center_chunk.chunk_id,
        )
        for section in result
    }
    for section in ranked:
        identity = (
            section.center_chunk.document_id,
            section.center_chunk.regulatory_chunk_id
            if section.center_chunk.regulatory_chunk_id is not None
            else section.center_chunk.chunk_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(section)
        if len(result) >= max_total_sections:
            break
    return result


def _rich_response_sections(
    top_sections: list[InferenceSection],
    selected_sections: list[InferenceSection],
    *,
    authoritative_selected: bool,
) -> list[InferenceSection]:
    """Merge rich results while preserving authoritative selected projections."""

    sections_by_identity = {
        (section.center_chunk.document_id, section.center_chunk.chunk_id): section
        for section in top_sections
    }
    for section in selected_sections:
        identity = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        if authoritative_selected:
            sections_by_identity[identity] = section
        else:
            sections_by_identity.setdefault(identity, section)
    return list(sections_by_identity.values())


def _interleave_ranked_chunk_results(
    primary_results: list[InferenceChunk],
    supplemental_results: list[InferenceChunk],
    limit: int,
) -> list[InferenceChunk]:
    """Fairly fold supplemental results into one bounded ranking lane."""

    if limit <= 0:
        return []

    merged_results: list[InferenceChunk] = []
    seen_identities: set[tuple[str, int]] = set()
    for rank in range(max(len(primary_results), len(supplemental_results))):
        for ranked_results in (primary_results, supplemental_results):
            if rank >= len(ranked_results):
                continue
            chunk = ranked_results[rank]
            identity = (chunk.document_id, chunk.chunk_id)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            merged_results.append(chunk)
            if len(merged_results) >= limit:
                return merged_results

    return merged_results


def _reorder_sections_by_chunk_ranking(
    sections: list[InferenceSection], ranked_chunks: list[InferenceChunk]
) -> list[InferenceSection]:
    sections_by_identity = {
        (section.center_chunk.document_id, section.center_chunk.chunk_id): section
        for section in sections
    }
    return [
        sections_by_identity[identity]
        for chunk in ranked_chunks
        if (identity := (chunk.document_id, chunk.chunk_id)) in sections_by_identity
    ]


def _filter_visible_regulatory_sections(
    sections: list[InferenceSection], visible_chunk_ids: set[str]
) -> list[InferenceSection]:
    """Drop stale index hits that PostgreSQL does not expose in the snapshot."""

    return [
        section
        for section in sections
        if section.center_chunk.regulatory_chunk_id in visible_chunk_ids
    ]


def _regulatory_provision_family(
    chunk: InferenceChunk,
) -> tuple[str, ...] | None:
    """Return the current exact structural article identity in a chunk path."""

    if chunk.regulatory_chunk_id is None or not isinstance(chunk.heading_path, list):
        return None

    parsed_path: list[tuple[int, RegulatoryArticleHeading]] = []
    for index, heading in enumerate(chunk.heading_path):
        if not isinstance(heading, str):
            return None
        parsed_heading = parse_regulatory_article_heading(heading)
        if parsed_heading is not None:
            parsed_path.append((index, parsed_heading))
    if not parsed_path:
        return None

    _, current_article = parsed_path[-1]
    normalized_path = normalize_regulatory_heading_path(
        chunk.heading_path,
        article_no=current_article.article_no,
    )
    scope: list[str] = []
    current_family: tuple[str, ...] | None = None
    for heading in normalized_path:
        parsed_heading = parse_regulatory_article_heading(heading)
        if parsed_heading is None:
            scope.append(" ".join(heading.casefold().split()))
            continue
        current_family = (
            *scope,
            parsed_heading.qualifier or "",
            parsed_heading.article_no,
        )
    return current_family


def _diversify_focused_regulatory_retrieval_lanes(
    fused_chunks: list[InferenceChunk],
    ranked_search_results: list[list[InferenceChunk]],
    *,
    max_chunks: int | None,
    focused_search: bool,
    regulatory_chunks_only: bool,
) -> list[InferenceChunk]:
    """Preserve lane and provision-family coverage in a bounded candidate set.

    Reciprocal-rank fusion rewards chunks that appear in both semantic and
    lexical lanes. For a focused legal query, that can fill the whole bounded
    window with one document before a strong lane-specific candidate is seen.
    This deterministic reordering preserves the fused winner, then round-robins
    distinct regulatory documents from a bounded lane-head band. It also keeps
    a few exact structural article families from the strongest represented
    instrument visible when repetitive annex/table chunks precede its operative
    provisions. The scan and output stay capped; ordinary search is untouched.
    """

    if (
        not focused_search
        or not regulatory_chunks_only
        or not fused_chunks
        or max_chunks == 0
    ):
        return fused_chunks

    bounded_fused_chunks = (
        fused_chunks if max_chunks is None else fused_chunks[:max_chunks]
    )
    if not all(chunk.regulatory_chunk_id is not None for chunk in bounded_fused_chunks):
        return fused_chunks

    diversified_chunks = [bounded_fused_chunks[0]]
    seen_chunk_ids = {bounded_fused_chunks[0].unique_id}
    represented_document_ids = {bounded_fused_chunks[0].document_id}

    bounded_chunk_count = len(bounded_fused_chunks)
    diversity_target = min(
        bounded_chunk_count,
        1 + len(ranked_search_results),
    )
    lane_head_limit = max_chunks if max_chunks is not None else MAX_CHUNKS_FED_TO_CHAT
    bounded_ranked_search_results = [
        ranked_lane[:lane_head_limit] for ranked_lane in ranked_search_results
    ]
    lane_offsets = [0] * len(ranked_search_results)
    while len(diversified_chunks) < diversity_target:
        made_progress = False
        for lane_index, ranked_lane in enumerate(bounded_ranked_search_results):
            while lane_offsets[lane_index] < len(ranked_lane):
                lane_candidate = ranked_lane[lane_offsets[lane_index]]
                lane_offsets[lane_index] += 1
                if (
                    lane_candidate.regulatory_chunk_id is None
                    or lane_candidate.unique_id in seen_chunk_ids
                    or lane_candidate.document_id in represented_document_ids
                ):
                    continue
                diversified_chunks.append(lane_candidate)
                seen_chunk_ids.add(lane_candidate.unique_id)
                represented_document_ids.add(lane_candidate.document_id)
                made_progress = True
                break
            if len(diversified_chunks) >= diversity_target:
                break
        if not made_progress:
            break

    # A source can rank first because an annex or table matches strongly while
    # its operative articles sit just beyond the ordinary result window. Search
    # a bounded overfetch pool for distinct structural article families from the
    # first represented document that actually has them. This is structural
    # coverage, not a topical or scenario-specific inference.
    candidate_scan_limit = max(
        max_chunks or 0,
        _REGULATORY_PROVISION_MAX_CANDIDATES,
    )
    candidate_scan = fused_chunks[:candidate_scan_limit]
    represented_documents = list(
        dict.fromkeys(chunk.document_id for chunk in bounded_fused_chunks)
    )
    provision_families_by_document: dict[
        str, list[tuple[tuple[str, ...], InferenceChunk]]
    ] = {}
    for chunk in candidate_scan:
        family = _regulatory_provision_family(chunk)
        if family is None:
            continue
        provision_families_by_document.setdefault(chunk.document_id, []).append(
            (family, chunk)
        )

    provision_source_document = next(
        (
            document_id
            for document_id in represented_documents
            if provision_families_by_document.get(document_id)
        ),
        None,
    )
    if provision_source_document is not None:
        seen_provision_families: set[tuple[str, ...]] = set()
        for chunk in diversified_chunks:
            if chunk.document_id != provision_source_document:
                continue
            existing_family = _regulatory_provision_family(chunk)
            if existing_family is not None:
                seen_provision_families.add(existing_family)
        initial_family_count = len(seen_provision_families)

        for family, chunk in provision_families_by_document[provision_source_document]:
            if len(seen_provision_families) >= (
                _REGULATORY_PROVISION_FAMILY_SEED_LIMIT
            ):
                break
            if family in seen_provision_families:
                continue
            seen_provision_families.add(family)
            if chunk.unique_id in seen_chunk_ids:
                continue
            diversified_chunks.append(chunk)
            seen_chunk_ids.add(chunk.unique_id)
            if max_chunks is not None and len(diversified_chunks) >= max_chunks:
                return diversified_chunks[:max_chunks]
        seeded_family_count = len(seen_provision_families) - initial_family_count
        if seeded_family_count:
            logger.debug(
                "Internal search - promoted %d structural provision families "
                "from the strongest represented regulatory source",
                seeded_family_count,
            )

    if max_chunks is not None and len(diversified_chunks) >= max_chunks:
        return diversified_chunks[:max_chunks]

    for chunk in fused_chunks:
        if chunk.unique_id in seen_chunk_ids:
            continue
        diversified_chunks.append(chunk)
        seen_chunk_ids.add(chunk.unique_id)
        if max_chunks is not None and len(diversified_chunks) >= max_chunks:
            break

    return diversified_chunks


class QueryExpansionAndScope(BaseModel):
    """Result of one search cycle's query expansion + source-scope decision."""

    semantic_query: str | None
    keyword_queries: list[str]
    plan_scope: list[DocumentSource] | None
    time_filter: TimeFilter | None = None


class SearchQueryLane(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    weight: float
    hybrid_alpha: float | None
    high_term_coverage: bool = False


def _query_deduplication_key(query: str) -> str:
    normalized = unicodedata.normalize("NFKD", query).casefold()
    searchable_characters = (
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
        if not unicodedata.category(character).startswith("M")
    )
    return " ".join("".join(searchable_characters).split())


def build_query_lanes(
    *,
    original_query: str | None,
    semantic_query: str | None,
    model_queries: list[str],
    keyword_queries: list[str],
    search_mode: str,
    regulatory_chunks_only: bool,
) -> list[SearchQueryLane]:
    """Build a deterministic, language-preserving bounded retrieval plan."""

    lanes: list[SearchQueryLane] = []
    lane_index_by_query: dict[str, int] = {}

    def add_lane(
        query: str | None,
        *,
        weight: float,
        hybrid_alpha: float | None,
        high_term_coverage: bool = False,
        mode_sensitive_deduplication: bool = False,
    ) -> None:
        if query is None:
            return
        stripped_query = query.strip()
        if not stripped_query:
            return
        deduplication_key = _query_deduplication_key(stripped_query)
        if not deduplication_key:
            return
        if mode_sensitive_deduplication:
            deduplication_key = (
                f"{deduplication_key}\0{hybrid_alpha}\0{high_term_coverage}"
            )
        existing_index = lane_index_by_query.get(deduplication_key)
        if existing_index is not None:
            existing_lane = lanes[existing_index]
            lanes[existing_index] = existing_lane.model_copy(
                update={"weight": existing_lane.weight + weight}
            )
            return
        if len(lanes) >= MAX_SEARCH_QUERY_LANES:
            return
        lane_index_by_query[deduplication_key] = len(lanes)
        lanes.append(
            SearchQueryLane(
                query=stripped_query,
                weight=weight,
                hybrid_alpha=hybrid_alpha,
                high_term_coverage=high_term_coverage,
            )
        )

    base_hybrid_alpha = None if search_mode == "hybrid" else 0.0
    add_lane(
        original_query,
        weight=ORIGINAL_QUERY_WEIGHT,
        hybrid_alpha=base_hybrid_alpha,
        high_term_coverage=search_mode == "full_text",
    )
    if search_mode == "hybrid":
        add_lane(
            semantic_query,
            weight=LLM_SEMANTIC_QUERY_WEIGHT,
            hybrid_alpha=None,
        )
    for model_query in model_queries:
        add_lane(
            model_query,
            weight=LLM_NON_CUSTOM_QUERY_WEIGHT,
            hybrid_alpha=base_hybrid_alpha,
            high_term_coverage=search_mode == "full_text",
        )
    if search_mode == "hybrid":
        keyword_hybrid_alpha = (
            _REGULATORY_KEYWORD_QUERY_HYBRID_ALPHA
            if regulatory_chunks_only
            else KEYWORD_QUERY_HYBRID_ALPHA
        )
        if regulatory_chunks_only:
            for model_query in model_queries:
                if extract_single_regulatory_provision_reference(model_query) is None:
                    continue
                add_lane(
                    model_query,
                    weight=LLM_NON_CUSTOM_QUERY_WEIGHT,
                    hybrid_alpha=_REGULATORY_KEYWORD_QUERY_HYBRID_ALPHA,
                    mode_sensitive_deduplication=True,
                )
        for keyword_query in keyword_queries:
            add_lane(
                keyword_query,
                weight=LLM_KEYWORD_QUERY_WEIGHT,
                hybrid_alpha=keyword_hybrid_alpha,
                mode_sensitive_deduplication=regulatory_chunks_only,
            )
    return lanes


def _build_scope_note(
    scope: list[DocumentSource] | None, queries_run: list[str]
) -> str:
    """Note appended to a scoped search's response: which source(s) it covered
    and the queries that ran, so a repeat can vary terms. "" when unscoped."""
    if not scope:
        return ""
    searched = ", ".join(source.value for source in scope)
    queries_str = "; ".join(queries_run) or "(none)"
    return (
        f"(This internal search covered only: {searched}. Queries run: {queries_str}. "
        "Judge sufficiency from the returned chunk text; make another search only "
        "when a material unresolved point warrants a distinct attempt.)"
    )


def deduplicate_queries(
    queries_with_weights: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Deduplicate queries by case-insensitive comparison and sum weights.

    Args:
        queries_with_weights: List of (query, weight) tuples

    Returns:
        Deduplicated list of (query, weight) tuples with summed weights
    """
    query_map: dict[str, tuple[str, float]] = {}
    for query, weight in queries_with_weights:
        query_lower = query.lower()
        if query_lower in query_map:
            # Sum weights for duplicate queries
            existing_query, existing_weight = query_map[query_lower]
            query_map[query_lower] = (existing_query, existing_weight + weight)
        else:
            # Keep the first occurrence (preserves original casing)
            query_map[query_lower] = (query, weight)
    return list(query_map.values())


def _estimate_section_tokens(
    section: InferenceSection,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
) -> int:
    """Estimate token count for a section using the LLM tokenizer.

    Args:
        section: InferenceSection to estimate tokens for
        token_counter: Function that counts tokens in text
        max_chunks_per_section: Maximum chunks to consider per section (None for all)

    Returns:
        Token count for the section
    """
    # Estimate for metadata (title, source_type, etc.)
    METADATA_TOKEN_ESTIMATE = 75

    # If max_chunks_per_section is specified, only count tokens for selected chunks
    if max_chunks_per_section is not None:
        selected_chunks = select_chunks_for_relevance(section, max_chunks_per_section)
        # Combine content from selected chunks
        combined_content = "\n".join(chunk.content for chunk in selected_chunks)
        content_tokens = token_counter(combined_content)
    else:
        content_tokens = token_counter(section.combined_content)

    return content_tokens + METADATA_TOKEN_ESTIMATE


@log_function_time(print_only=True)
def _trim_sections_by_tokens(
    sections: list[InferenceSection],
    max_tokens: int,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
) -> list[InferenceSection]:
    """Trim sections to fit within a token budget using the LLM tokenizer.

    Args:
        sections: List of InferenceSection objects to trim
        max_tokens: Maximum token budget
        token_counter: Function that counts tokens in text
        max_chunks_per_section: Maximum chunks to consider per section (None for all)

    Returns:
        Trimmed list of sections that fit within the token budget
    """
    if not sections or max_tokens <= 0:
        return sections

    trimmed_sections = []
    total_tokens = 0

    for section in sections:
        section_tokens = _estimate_section_tokens(
            section, token_counter, max_chunks_per_section
        )
        if total_tokens + section_tokens <= max_tokens:
            trimmed_sections.append(section)
            total_tokens += section_tokens
        else:
            break

    logger.debug(
        "Trimmed sections from %s to %s (%s tokens, budget: %s)",
        len(sections),
        len(trimmed_sections),
        total_tokens,
        max_tokens,
    )

    return trimmed_sections


class SearchTool(Tool[SearchToolOverrideKwargs]):
    NAME = "internal_search"
    DISPLAY_NAME = "Internal Search"
    DESCRIPTION = "Search connected applications for information."

    def __init__(
        self,
        tool_id: int,
        emitter: Emitter,
        # Used for ACLs and federated search, anonymous users only see public docs
        user: User,
        # Pre-extracted persona search configuration
        persona_search_info: PersonaSearchInfo,
        llm: LLM,
        document_index: DocumentIndex,
        # Respecting user selections
        user_selected_filters: BaseFilters | None,
        # Vespa metadata filters for overflowing user files.  NOT the raw IDs
        # of the current project/persona — only set when user files couldn't
        # fit in the LLM context and need to be searched via vector DB.
        project_id_filter: int | None,
        persona_id_filter: int | None = None,
        bypass_acl: bool = False,
        # Slack context for federated Slack search (tokens fetched internally)
        slack_context: SlackContext | None = None,
        # Whether to enable Slack federated search
        enable_slack_search: bool = True,
        # Whether to infer source and time filters from the
        # query. When False, only user/persona-selected filters are applied.
        auto_detect_filters: bool = True,
        shared_time_filter_decision: _SharedDecision[TimeFilter | None] | None = None,
        parallel_scope_decision: (
            _SharedDecision[list[DocumentSource] | None] | None
        ) = None,
    ) -> None:
        super().__init__(emitter=emitter)

        self.user = user
        self.persona_search_info = persona_search_info
        self.llm = llm
        self.document_index = document_index
        self.user_selected_filters = user_selected_filters
        self.project_id_filter = project_id_filter
        self.persona_id_filter = persona_id_filter
        self.bypass_acl = bypass_acl
        self.slack_context = slack_context
        self.enable_slack_search = enable_slack_search
        self.auto_detect_filters = auto_detect_filters

        self._search_cycles: list[SearchCycle] = []
        self._cached_expansion: tuple[str | None, list[str]] | None = None
        self._scope_decision_settled = False
        self._time_filter: TimeFilter | None = None
        self._time_filter_computed = False
        self._shared_time_filter_decision = (
            shared_time_filter_decision or _SharedDecision()
        )
        self._parallel_scope_decision = parallel_scope_decision

        self._id = tool_id

    def _fork_for_parallel_call(
        self,
        parallel_scope_decision: _SharedDecision[list[DocumentSource] | None],
    ) -> "SearchTool":
        """Create isolated query state with filter decisions shared safely."""
        return SearchTool(
            tool_id=self._id,
            emitter=self.emitter,
            user=self.user,
            persona_search_info=self.persona_search_info,
            llm=self.llm,
            document_index=self.document_index,
            user_selected_filters=self.user_selected_filters,
            project_id_filter=self.project_id_filter,
            persona_id_filter=self.persona_id_filter,
            bypass_acl=self.bypass_acl,
            slack_context=self.slack_context,
            enable_slack_search=self.enable_slack_search,
            auto_detect_filters=self.auto_detect_filters,
            shared_time_filter_decision=self._shared_time_filter_decision,
            parallel_scope_decision=parallel_scope_decision,
        )

    def fork_for_parallel_calls(self, call_count: int) -> list["SearchTool"]:
        """Create one parallel batch whose source decision is computed once."""

        if call_count <= 0:
            raise ValueError("call_count must be positive")
        parallel_scope_decision: _SharedDecision[list[DocumentSource] | None] = (
            _SharedDecision()
        )
        return [
            self._fork_for_parallel_call(parallel_scope_decision)
            for _ in range(call_count)
        ]

    def fork_for_parallel_call(self) -> "SearchTool":
        """Create one isolated fork for callers that do not manage a batch."""

        return self.fork_for_parallel_calls(1)[0]

    def fork_for_independent_context(
        self,
        *,
        emitter: Emitter | None = None,
    ) -> "SearchTool":
        """Create a fork with isolated decisions and optional output destination."""

        return SearchTool(
            tool_id=self._id,
            emitter=self.emitter if emitter is None else emitter,
            user=self.user,
            persona_search_info=self.persona_search_info,
            llm=self.llm,
            document_index=self.document_index,
            user_selected_filters=self.user_selected_filters,
            project_id_filter=self.project_id_filter,
            persona_id_filter=self.persona_id_filter,
            bypass_acl=self.bypass_acl,
            slack_context=self.slack_context,
            enable_slack_search=self.enable_slack_search,
            auto_detect_filters=self.auto_detect_filters,
        )

    def _prefetch_slack_data(
        self, db_session: Session
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """Pre-fetch Slack access token, bot token, and entity config from DB.

        All DB queries for Slack federated search are performed here in a
        single session, so the parallel search phase needs no DB access.

        Returns:
            (access_token, bot_token, entities) — access_token is None when
            Slack search should be skipped.
        """
        bot_token: str | None = None
        access_token: str | None = None
        entities: dict[str, Any] = {}

        # Case 1: Slack bot context — requires a Slack federated connector
        # linked via the persona's document sets
        if self.slack_context:
            document_set_names = self.persona_search_info.document_set_names
            if not document_set_names:
                logger.debug(
                    "Skipping Slack federated search: no document sets on persona"
                )
                return None, None, {}

            slack_federated_mappings = (
                get_federated_connector_document_set_mappings_by_document_set_names(
                    db_session, document_set_names
                )
            )
            found_slack_connector = False
            for mapping in slack_federated_mappings:
                if (
                    mapping.federated_connector is not None
                    and mapping.federated_connector.source
                    == FederatedConnectorSource.FEDERATED_SLACK
                ):
                    entities = mapping.federated_connector.config or {}
                    found_slack_connector = True
                    logger.debug("Found Slack federated connector config: %s", entities)
                    break

            if not found_slack_connector:
                logger.debug(
                    "Skipping Slack federated search: no Slack federated connector linked to document sets %s",
                    document_set_names,
                )
                return None, None, {}

            try:
                slack_bots = fetch_slack_bots(db_session)
                if not slack_bots:
                    return None, None, {}

                tenant_slack_bot = next(
                    (bot for bot in slack_bots if bot.enabled and bot.user_token),
                    None,
                )
                if not tenant_slack_bot:
                    tenant_slack_bot = next(
                        (bot for bot in slack_bots if bot.enabled), None
                    )

                if tenant_slack_bot:
                    bot_token = (
                        tenant_slack_bot.bot_token.get_value(apply_mask=False)
                        if tenant_slack_bot.bot_token
                        else None
                    )
                    user_token = (
                        tenant_slack_bot.user_token.get_value(apply_mask=False)
                        if tenant_slack_bot.user_token
                        else None
                    )
                    access_token = user_token or bot_token
            except Exception as e:
                logger.warning("Could not fetch Slack bot tokens: %s", e)

        # Case 2: Web user with federated OAuth (if bot context didn't yield a token)
        if not access_token and self.user:
            try:
                federated_oauth_tokens = list_federated_connector_oauth_tokens(
                    db_session, self.user.id
                )
                if not federated_oauth_tokens:
                    return access_token, bot_token, entities

                slack_oauth_token = next(
                    (
                        token
                        for token in federated_oauth_tokens
                        if token.federated_connector.source
                        == FederatedConnectorSource.FEDERATED_SLACK
                    ),
                    None,
                )
                if slack_oauth_token and slack_oauth_token.token:
                    access_token = slack_oauth_token.token.get_value(apply_mask=False)
                    entities = slack_oauth_token.federated_connector.config or {}
            except Exception as e:
                logger.warning("Could not fetch Slack OAuth token: %s", e)

        return access_token, bot_token, entities

    def _run_slack_search(
        self,
        query: str,
        access_token: str,
        bot_token: str | None,
        entities: dict[str, Any],
        search_settings: SearchSettings,
        limit: int,
    ) -> list[InferenceChunk]:
        """Run Slack federated search using pre-fetched tokens and config.

        All DB data is pre-fetched in run() so this method needs no DB session.

        Args:
            query: The user's original search query
            access_token: Slack access token (user or bot)
            bot_token: Slack bot token (for enhanced permissions)
            entities: Federated connector entity config (channel filtering)
            search_settings: Pre-fetched SearchSettings for chunking config

        Returns:
            List of InferenceChunk results from Slack
        """
        try:
            chunk_request = ChunkIndexRequest(
                query=query,
                filters=IndexFilters(access_control_list=None),
            )

            chunks = slack_retrieval(
                query=chunk_request,
                access_token=access_token,
                connector=None,
                entities=entities,
                limit=limit,
                slack_event_context=self.slack_context,
                bot_token=bot_token,
                team_id=None,
                search_settings=search_settings,
            )

            logger.info("Slack federated search returned %s chunks", len(chunks))
            return chunks[:limit]

        except Exception as e:
            logger.error("Slack federated search error: %s", e, exc_info=True)
            return []

    def _run_search_for_query(
        self,
        query: str,
        hybrid_alpha: float | None,
        high_term_coverage: bool,
        num_hits: int,
        acl_filters: list[str] | None,
        embedding_model: EmbeddingModel,
        federated_retrieval_infos: list[FederatedRetrievalInfo],
        effective_filters: BaseFilters | None,
        provision_reference: RegulatoryProvisionReference | None = None,
    ) -> list[InferenceChunk]:
        """Run search pipeline for a single query using pre-fetched data.

        All DB data (ACL filters, embedding model, federated retrieval info)
        is pre-fetched in run() so this method needs no DB session.

        Args:
            query: The search query string
            hybrid_alpha: Hybrid search alpha parameter (None for default)
            num_hits: Maximum number of hits to return
            acl_filters: Pre-fetched ACL filters (None when bypass_acl)
            embedding_model: Pre-fetched embedding model
            federated_retrieval_infos: Pre-fetched federated retrieval functions
            effective_filters: Filters for THIS search, with the per-call source
                scope already applied (computed once in run()).

        Returns:
            List of InferenceChunk results
        """
        candidate_limit = num_hits
        regulatory_candidate_search = bool(
            effective_filters and effective_filters.regulatory_chunks_only
        )
        if regulatory_candidate_search or provision_reference is not None:
            candidate_limit = max(
                num_hits,
                min(
                    num_hits * _REGULATORY_PROVISION_OVERFETCH_FACTOR,
                    _REGULATORY_PROVISION_MAX_CANDIDATES,
                ),
            )

        chunks = search_pipeline(
            chunk_search_request=ChunkSearchRequest(
                query=query,
                hybrid_alpha=hybrid_alpha,
                high_term_coverage=high_term_coverage,
                # For projects, the search scope is the project and has no other limits
                user_selected_filters=(
                    effective_filters if self.project_id_filter is None else None
                ),
                bypass_acl=self.bypass_acl,
                limit=candidate_limit,
            ),
            project_id_filter=self.project_id_filter,
            persona_id_filter=self.persona_id_filter,
            document_index=self.document_index,
            user=self.user,
            persona_search_info=self.persona_search_info,
            acl_filters=acl_filters,
            embedding_model=embedding_model,
            prefetched_federated_retrieval_infos=federated_retrieval_infos,
        )
        if provision_reference is None:
            return chunks[:num_hits]

        structurally_matching = [
            chunk
            for chunk in chunks
            if chunk.heading_path is not None
            and regulatory_heading_path_matches_reference(
                chunk.heading_path,
                provision_reference,
            )
        ]
        logger.info(
            "Internal search - exact provision %s retained %d/%d candidates",
            provision_reference.article_no,
            len(structurally_matching),
            len(chunks),
        )
        return structurally_matching[:num_hits]

    @classmethod
    def is_available(cls, db_session: Session) -> bool:
        """Check if search tool is available.

        Returns False when the vector DB is disabled (search cannot function
        without it). Otherwise, available if ANY of the following exist:
        - Regular connectors (team knowledge)
        - Federated connectors (e.g., Slack)
        - User files (User Knowledge mode)
        """
        from onyx.configs.app_configs import DISABLE_VECTOR_DB
        from onyx.db.connector import check_user_files_exist

        if DISABLE_VECTOR_DB:
            return False

        return (
            check_connectors_exist(db_session)
            or check_federated_connectors_exist(db_session)
            or check_user_files_exist(db_session)
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        if (
            self.user_selected_filters is not None
            and self.user_selected_filters.regulatory_chunks_only
        ):
            return _REGULATORY_SEARCH_DESCRIPTION
        return self.DESCRIPTION

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    """For explicit tool calling"""

    def tool_definition(self) -> dict:
        if not (
            self.user_selected_filters is not None
            and self.user_selected_filters.regulatory_chunks_only
        ):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            QUERIES_FIELD: {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of search queries to execute, typically a "
                                    "single query. Query expansion and filter "
                                    "extraction steps will be run automatically "
                                    "downstream, do not include time or source type "
                                    "scoping details in your query."
                                ),
                            },
                        },
                        "required": [QUERIES_FIELD],
                    },
                },
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        QUERIES_FIELD: {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": REGULATORY_MAX_SEARCH_QUERY_CHARS,
                            },
                            "minItems": 1,
                            "maxItems": 1,
                            "description": (
                                "Exactly one focused search query. Use separate tool calls "
                                "when you judge the issues need distinct retrieval attempts. "
                                "Use the smallest discriminative legal query likely to occur "
                                "in the controlling text. Each call is an independent retrieval "
                                "fragment and does not inherit anchors from another call. If a "
                                "known source, instrument, mechanism, status, provision, code, "
                                "or other identifier disambiguates this fragment, retain that "
                                "identifier verbatim rather than replacing it with an umbrella category. "
                                "Omit unrelated facts and predicted conclusions. Write in the likely indexed source "
                                "language. Use plain "
                                "terms or a natural phrase, not Boolean AND/OR/NOT syntax. "
                                "Bounded same-language semantic and lexical variants are generated automatically. "
                                "Source-type and temporal filter extraction run separately, "
                                "so do not include those scoping details in the query."
                            ),
                        },
                        SEARCH_MODE_FIELD: {
                            "type": "string",
                            "enum": ["hybrid", "keyword", "full_text"],
                            "description": (
                                "Choose the retrieval mode for this evidence target; do not "
                                "default every independent target to one mode. Use keyword "
                                "when a literal provision identifier, code, numeric value, acronym, "
                                "or rare proper name is a sufficient anchor and remaining terms may "
                                "be optional, or when literal alternatives may occur in different "
                                "chunks and any matching alternative is useful. Use full_text only "
                                "when multiple exact anchors or a high "
                                "proportion of the analyzed terms must co-occur lexically in the "
                                "same chunk; keep its query to the terms whose co-occurrence is "
                                "actually required. Short queries are stricter than longer natural-language "
                                "queries. A mechanism name combined only with general legal "
                                "words does not by itself justify a lexical mode. "
                                "Use hybrid when the controlling vocabulary, synonym, source "
                                "label, or provision wording is uncertain."
                            ),
                        },
                    },
                    "required": [
                        QUERIES_FIELD,
                        SEARCH_MODE_FIELD,
                    ],
                },
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolStart(),
            )
        )

    def _decide_search_scope(
        self, decide_args: tuple[Any, ...]
    ) -> list[DocumentSource] | None:
        if self._parallel_scope_decision is None:
            return decide_search_scope(*decide_args)
        return self._parallel_scope_decision.get_or_compute(
            lambda: decide_search_scope(*decide_args)
        )

    def _decide_time_filter(
        self, message_history: list[ChatMinimalTextMessage]
    ) -> TimeFilter | None:
        return self._shared_time_filter_decision.get_or_compute(
            lambda: decide_time_filter(message_history, self.llm)
        )

    @log_function_time(
        func_name="Search tool - query expansion + scope decision",
        print_only=True,
        debug_only=True,
    )
    def _expand_queries_and_decide_scope(
        self,
        skip_query_expansion: bool,
        message_history: list[ChatMinimalTextMessage],
        user_info: str | None,
        memories: list[str],
        decide_args: tuple[Any, ...],
        filter_message_history: list[ChatMinimalTextMessage] | None = None,
    ) -> QueryExpansionAndScope:
        """Expand the query and decide the source/time scope, in parallel when each
        applies.

        Repeat calls reuse the cached expansion instead of re-expanding. A parallel
        batch computes one source decision from its complete query set; later batches
        can still reroute. The conversation-level time decision is computed once per
        turn across every fork. Both auto decisions are gated by
        ``auto_detect_filters``.
        """
        expand_queries = not skip_query_expansion
        decide_scope = self.auto_detect_filters and not self._scope_decision_settled
        decide_time = self.auto_detect_filters and not self._time_filter_computed

        jobs: list[tuple[Callable, tuple]] = []
        scope_job_index: int | None = None
        time_job_index: int | None = None
        if expand_queries:
            expansion_args = (message_history, self.llm, user_info, memories)
            jobs.append((semantic_query_rephrase, expansion_args))
            jobs.append((keyword_query_expansion, expansion_args))
        if decide_scope:
            scope_job_index = len(jobs)
            jobs.append((self._decide_search_scope, (decide_args,)))
        if decide_time:
            time_job_index = len(jobs)
            jobs.append(
                (
                    self._decide_time_filter,
                    (filter_message_history or message_history,),
                )
            )

        results = run_functions_tuples_in_parallel(jobs) if jobs else []

        semantic_query: str | None = None
        keyword_queries: list[str] = []
        if expand_queries:
            semantic_query = results[0]
            keyword_queries = results[1] or []
            self._cached_expansion = (semantic_query, keyword_queries)

        plan_scope: list[DocumentSource] | None = None
        if scope_job_index is not None:
            plan_scope = results[scope_job_index]
            self._scope_decision_settled = plan_scope is None

        if time_job_index is not None:
            self._time_filter = results[time_job_index]
            self._time_filter_computed = True

        return QueryExpansionAndScope(
            semantic_query=semantic_query,
            keyword_queries=keyword_queries,
            plan_scope=plan_scope,
            time_filter=self._time_filter,
        )

    @log_function_time(print_only=True)
    def run(
        self,
        placement: Placement,
        override_kwargs: SearchToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        # Start overall timing
        overall_start_time = time.time()
        regulatory_chunks_only = bool(
            self.user_selected_filters is not None
            and self.user_selected_filters.regulatory_chunks_only
        )
        llm_queries = _validate_search_queries(
            llm_kwargs,
            regulatory_chunks_only=regulatory_chunks_only,
        )
        search_mode = _validate_search_mode(
            llm_kwargs,
            regulatory_chunks_only=regulatory_chunks_only,
        )
        # Preserve backward compatibility with stored/older tool payloads without
        # forcing the answering model to manufacture a server-shaped research plan.
        raw_coverage_item = llm_kwargs.get(COVERAGE_ITEM_FIELD)
        raw_evidence_target = llm_kwargs.get(EVIDENCE_TARGET_FIELD)
        if regulatory_chunks_only:
            coverage_item = (
                raw_coverage_item.strip()
                if isinstance(raw_coverage_item, str) and raw_coverage_item.strip()
                else "Model-directed search"
            )
            evidence_target = (
                raw_evidence_target.strip()
                if isinstance(raw_evidence_target, str) and raw_evidence_target.strip()
                else llm_queries[0].strip()
            )
            llm_queries = [_prepare_search_query(llm_queries[0], search_mode)]
        else:
            coverage_item = ""
            evidence_target = ""

        # Initialize selection timing in case of an early exception.
        document_selection_elapsed = 0.0

        connected_sources: list[DocumentSource] = []

        # Pre-fetch all DB data in a single short-lived session so that
        # parallel search workers need zero DB connections.
        with get_session_with_current_tenant() as db_session:
            # ACL filters
            acl_filters: list[str] | None = (
                None
                if self.bypass_acl
                else build_access_filters_for_user(self.user, db_session)
            )

            # Validate document-set access for user-supplied filters.
            if (
                self.user_selected_filters
                and self.user_selected_filters.document_set
                and not self.bypass_acl
                and self.user
                and not self.user.is_anonymous
            ):
                requested = self.user_selected_filters.document_set
                accessible = filter_document_set_names_by_user_access(
                    db_session=db_session,
                    document_set_names=requested,
                    user=self.user,
                )
                unauthorized = sorted(
                    name for name in requested if name not in accessible
                )
                if unauthorized:
                    raise OnyxError(
                        OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
                        f"User does not have access to document sets: {unauthorized}",
                    )

            # SearchSettings → materialise EmbeddingModel while session is
            # open (forces lazy-load of cloud_provider properties)
            search_settings = get_current_search_settings(db_session)
            if not search_settings:
                raise RuntimeError(
                    "No search settings configured — cannot run internal search"
                )

            embedding_model = EmbeddingModel.from_db_model(
                search_settings=search_settings,
                server_host=MODEL_SERVER_HOST,
                server_port=MODEL_SERVER_PORT,
            )
            reranker_config = get_reranker_configuration(db_session)

            # Federated retrieval functions (non-Slack; Slack is separate)
            if self.project_id_filter is not None:
                # Project mode ignores user filters → no federated sources
                prefetch_source_types = None
            else:
                prefetch_source_types = (
                    list(self.user_selected_filters.source_type)
                    if self.user_selected_filters
                    and self.user_selected_filters.source_type
                    else None
                )
            federated_retrieval_infos = (
                get_federated_retrieval_functions(
                    db_session=db_session,
                    user_id=self.user.id if self.user else None,
                    source_types=prefetch_source_types,
                    document_set_names=self.persona_search_info.document_set_names,
                )
                or []
            )

            # Project mode ignores user filters, so source scoping doesn't apply.
            if self.project_id_filter is None:
                connected_sources = fetch_unique_document_sources(db_session)

            # Slack tokens and entity config — only prefetch when Slack
            # search is enabled or we're in a Slack bot context.
            if self.enable_slack_search or self.slack_context:
                slack_access_token, slack_bot_token, slack_entities = (
                    self._prefetch_slack_data(db_session)
                )
            else:
                slack_access_token, slack_bot_token, slack_entities = (
                    None,
                    None,
                    {},
                )
        # Session is closed here — all parallel work uses plain Python objects only

        # Run semantic and keyword query expansion in parallel (unless skipped)
        # Use message history, memories, and user info from override_kwargs
        message_history = (
            override_kwargs.message_history if override_kwargs.message_history else []
        )
        filter_message_history = (
            override_kwargs.filter_message_history
            if override_kwargs.filter_message_history is not None
            else message_history
        )
        memories = (
            override_kwargs.user_memory_context.as_formatted_list()
            if override_kwargs.user_memory_context
            else []
        )
        user_info = override_kwargs.user_info

        # A persona/user source restriction is the outer bound the decision works within.
        user_source_restriction: list[DocumentSource] | None = (
            list(self.user_selected_filters.source_type)
            if self.user_selected_filters and self.user_selected_filters.source_type
            else None
        )
        if user_source_restriction is not None:
            allowed = set(user_source_restriction)
            candidate_sources = [s for s in connected_sources if s in allowed]
        else:
            candidate_sources = connected_sources

        decide_args = (
            filter_message_history,
            self.llm,
            candidate_sources,
            list(self._search_cycles),
            override_kwargs.filter_queries or llm_queries,
        )
        expansion = self._expand_queries_and_decide_scope(
            skip_query_expansion=(
                override_kwargs.skip_query_expansion or search_mode != "hybrid"
            ),
            message_history=message_history,
            user_info=user_info,
            memories=memories,
            decide_args=decide_args,
            filter_message_history=filter_message_history,
        )
        semantic_query = expansion.semantic_query if search_mode == "hybrid" else None
        keyword_queries = expansion.keyword_queries if search_mode == "hybrid" else []
        plan_scope = expansion.plan_scope

        resolved_scope = (
            plan_scope if plan_scope is not None else user_source_restriction
        )

        logger.info(
            "Internal search - source scope: %s",
            [s.value for s in resolved_scope] if resolved_scope else "all sources",
        )

        # On a repeat call that advanced to a not-yet-searched source, reuse the
        # cached expansion (it is source-agnostic) rather than searching raw queries.
        searched_sources = {
            value for cycle in self._search_cycles for value in cycle.searched_sources
        }
        is_new_filter = bool(resolved_scope) and any(
            source.value not in searched_sources for source in resolved_scope
        )
        if (
            search_mode == "hybrid"
            and override_kwargs.skip_query_expansion
            and is_new_filter
            and self._cached_expansion is not None
        ):
            semantic_query, keyword_queries = self._cached_expansion

        self._search_cycles.append(
            SearchCycle(
                cycle_number=len(self._search_cycles) + 1,
                queries=list(llm_queries),
                searched_sources=(
                    [source.value for source in resolved_scope]
                    if resolved_scope
                    else []
                ),
            )
        )

        # Surface the applied filters (source scope + time window) to the UI. Scope
        # is reported only when it narrows to a strict subset — scoping to all
        # connected sources is equivalent to an unscoped search.
        scopes_all_sources = bool(connected_sources) and set(
            connected_sources
        ).issubset(resolved_scope or [])
        emitted_sources = (
            [source.value for source in resolved_scope]
            if resolved_scope and not scopes_all_sources
            else []
        )
        time_filter = expansion.time_filter
        if emitted_sources or time_filter is not None:
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=SearchToolFilterDelta(
                        sources=emitted_sources,
                        time_filter_start=time_filter.start if time_filter else None,
                        time_filter_end=time_filter.end if time_filter else None,
                    ),
                )
            )

        effective_filters = self.user_selected_filters
        if resolved_scope is not None:
            effective_filters = (
                self.user_selected_filters or BaseFilters()
            ).model_copy(update={"source_type": resolved_scope})
            federated_retrieval_infos = [
                info
                for info in federated_retrieval_infos
                if info.source.to_non_federated_source() in resolved_scope
            ]
            # Disable the Slack federated search when Slack is out of scope.
            if DocumentSource.SLACK not in resolved_scope:
                slack_access_token = None

        # The pipeline composes the lower bound with any persona time floor.
        if time_filter is not None:
            effective_filters = time_filter.apply_to(effective_filters or BaseFilters())
            logger.info(
                "Internal search - time window (%s): %s to %s",
                time_filter.field.value,
                time_filter.start.isoformat() if time_filter.start else "any",
                time_filter.end.isoformat() if time_filter.end else "any",
            )

        focused_regulatory_search = bool(
            effective_filters
            and effective_filters.regulatory_chunks_only
            and (override_kwargs.skip_query_expansion or search_mode != "hybrid")
        )

        canonical_original_query = override_kwargs.original_query or llm_queries[0]
        query_lanes = build_query_lanes(
            original_query=canonical_original_query,
            semantic_query=semantic_query,
            model_queries=llm_queries,
            keyword_queries=keyword_queries,
            search_mode=search_mode,
            regulatory_chunks_only=regulatory_chunks_only,
        )
        queries_run = [lane.query for lane in query_lanes]
        scope_note = _build_scope_note(resolved_scope, queries_run)

        logger.debug("Bounded search query lanes: %s", queries_run)

        # Emit the queries early so the UI can display them immediately
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolQueriesDelta(
                    queries=queries_run,
                ),
            )
        )

        # Run no more than five bounded lanes. Slack is fetched in parallel but
        # folded into the original lane before RRF, so it cannot create lane six.
        search_functions: list[tuple[Callable, tuple]] = []
        for lane in query_lanes:
            provision_reference = (
                extract_single_regulatory_provision_reference(lane.query)
                if focused_regulatory_search and lane.hybrid_alpha == 0.0
                else None
            )
            search_functions.append(
                (
                    self._run_search_for_query,
                    (
                        lane.query,
                        lane.hybrid_alpha,
                        lane.high_term_coverage,
                        override_kwargs.per_lane_num_hits,
                        acl_filters,
                        embedding_model,
                        federated_retrieval_infos,
                        effective_filters,
                        provision_reference,
                    ),
                )
            )

        slack_search_scheduled = False
        if slack_access_token and override_kwargs.original_query:
            slack_search_scheduled = True
            search_functions.append(
                (
                    self._run_slack_search,
                    (
                        override_kwargs.original_query,
                        slack_access_token,
                        slack_bot_token,
                        slack_entities,
                        search_settings,
                        override_kwargs.per_lane_num_hits,
                    ),
                )
            )

        all_search_results = cast(
            list[list[InferenceChunk]],
            run_functions_tuples_in_parallel(search_functions),
        )
        if slack_search_scheduled:
            slack_results = all_search_results.pop()
            if all_search_results:
                all_search_results[0] = _interleave_ranked_chunk_results(
                    all_search_results[0],
                    slack_results,
                    override_kwargs.per_lane_num_hits,
                )

        top_chunks = weighted_reciprocal_rank_fusion(
            ranked_results=all_search_results,
            weights=[lane.weight for lane in query_lanes],
            id_extractor=lambda chunk: chunk.unique_id,
        )
        lexical_first_search_results = [
            results
            for lane, results in zip(query_lanes, all_search_results)
            if lane.hybrid_alpha == 0.0
        ] + [
            results
            for lane, results in zip(query_lanes, all_search_results)
            if lane.hybrid_alpha != 0.0
        ]
        top_chunks = _diversify_focused_regulatory_retrieval_lanes(
            top_chunks,
            lexical_first_search_results,
            max_chunks=override_kwargs.rerank_candidate_limit,
            focused_search=focused_regulatory_search,
            regulatory_chunks_only=bool(
                effective_filters and effective_filters.regulatory_chunks_only
            ),
        )

        fused_candidates = top_chunks[: override_kwargs.rerank_candidate_limit]
        rerank_result = rerank_chunks(
            query=canonical_original_query,
            chunks=fused_candidates,
            config=reranker_config,
        )
        diverse_candidate_chunks = apply_soft_diversity(
            chunks=rerank_result.ordered_chunks,
            scores=rerank_result.scores_by_chunk,
            limit=override_kwargs.rerank_candidate_limit,
        )
        chunks_for_selection = (
            diverse_candidate_chunks
            if rerank_result.used_external
            else rerank_result.ordered_chunks
        )
        candidate_sections = [
            inference_section_from_single_chunk(chunk) for chunk in chunks_for_selection
        ]
        diverse_candidate_sections = [
            inference_section_from_single_chunk(chunk)
            for chunk in diverse_candidate_chunks
        ]
        returned_sections = diverse_candidate_sections[: override_kwargs.num_hits]

        if not candidate_sections:
            logger.info("Search tool - no results found, returning empty response")
            empty_response, _, _ = convert_inference_sections_to_llm_string(
                top_sections=[],
                note=scope_note or None,
            )
            return ToolResponse(
                rich_response=SearchDocsResponse(
                    search_docs=[],
                    citation_mapping={},
                    displayed_docs=None,
                ),
                llm_facing_response=(
                    _add_search_receipt(
                        empty_response,
                        coverage_item=coverage_item,
                        evidence_target=evidence_target,
                    )
                    if regulatory_chunks_only
                    else empty_response
                ),
            )

        # Enrich chunks with `Document.file_id` (Postgres-only metadata not
        # stored in Vespa).
        with get_session_with_current_tenant() as enrichment_session:
            populate_file_ids_on_sections(candidate_sections, enrichment_session)

        secondary_flows_user_query = (
            override_kwargs.original_query
            or semantic_query
            or (llm_queries[0] if llm_queries else "")
        )

        ranked_regulatory_sections: list[InferenceSection] | None = None
        max_selected_sections = override_kwargs.max_llm_chunks
        if rerank_result.used_external:
            selected_sections = candidate_sections[:max_selected_sections]
            if regulatory_chunks_only:
                ranked_regulatory_sections = candidate_sections
            logger.debug(
                "Search tool - using external reranker ordering without a "
                "secondary selector (%s sections)",
                len(selected_sections),
            )
        else:
            token_counter = get_llm_token_counter(self.llm)
            max_tokens_for_selection = (
                override_kwargs.max_llm_chunks * DOC_EMBEDDING_CONTEXT_SIZE
            )
            sections_for_selection = _trim_sections_by_tokens(
                sections=candidate_sections,
                max_tokens=max_tokens_for_selection,
                token_counter=token_counter,
                max_chunks_per_section=MAX_CHUNKS_FOR_RELEVANCE,
            )
            if _can_use_ranked_regulatory_selection(
                sections_for_selection,
                focused_search=focused_regulatory_search,
            ):
                ranked_regulatory_sections = sections_for_selection
                selected_sections = _reserve_ranked_regulatory_seeds(
                    sections_for_selection, max_selected_sections
                )
                logger.debug(
                    "Search tool - using ranked regulatory sections without a "
                    "secondary selector (%s sections)",
                    len(selected_sections),
                )
            else:
                document_selection_start_time = time.time()
                selected_sections, _ = select_sections_for_expansion(
                    sections=sections_for_selection,
                    user_query=secondary_flows_user_query,
                    llm=self.llm,
                    max_chunks_per_section=MAX_CHUNKS_FOR_RELEVANCE,
                )
                document_selection_elapsed = time.time() - document_selection_start_time
                logger.debug(
                    "Search tool - LLM picking documents took %s seconds "
                    "(selected %s sections)",
                    format(document_selection_elapsed, ".3f"),
                    len(selected_sections),
                )

            diverse_selected_chunks = apply_soft_diversity(
                chunks=[section.center_chunk for section in selected_sections],
                scores=rerank_result.scores_by_chunk,
                limit=max_selected_sections,
            )
            selected_sections = _reorder_sections_by_chunk_ranking(
                selected_sections,
                diverse_selected_chunks,
            )

        # Expose a bounded, metadata-only provision outline when multiple real
        # search seeds point to the same regulatory source. Build it before
        # sibling expansion so deterministic neighbors cannot manufacture a
        # dominant source.
        regulatory_navigation: RegulatoryProvisionNavigation | None = None
        provision_as_of_date = (
            effective_filters.as_of_date if effective_filters else None
        )

        # Once the LLM has identified a controlling regulatory hit, include the
        # other paragraphs of that same provision deterministically. This stays
        # within the existing chunk budget and does not open the whole file or
        # invoke another model.
        if regulatory_chunks_only:
            navigation_seed_sections = (
                ranked_regulatory_sections[:max_selected_sections]
                if ranked_regulatory_sections is not None
                else selected_sections
            )
            with get_session_with_current_tenant() as provision_session:
                visible_chunk_ids = get_visible_regulatory_chunk_ids(
                    provision_session,
                    [
                        chunk_id
                        for section in navigation_seed_sections
                        if (chunk_id := section.center_chunk.regulatory_chunk_id)
                        is not None
                    ],
                    as_of_date=provision_as_of_date,
                )
                navigation_seed_sections = _filter_visible_regulatory_sections(
                    navigation_seed_sections,
                    visible_chunk_ids,
                )
                selected_sections = _filter_visible_regulatory_sections(
                    selected_sections,
                    visible_chunk_ids,
                )
                if ranked_regulatory_sections is not None:
                    ranked_regulatory_sections = _filter_visible_regulatory_sections(
                        ranked_regulatory_sections,
                        visible_chunk_ids,
                    )
                regulatory_navigation = build_regulatory_provision_navigation(
                    provision_session,
                    navigation_seed_sections,
                    query=llm_queries[0],
                    as_of_date=provision_as_of_date,
                )
                selected_sections = expand_selected_regulatory_references(
                    provision_session,
                    selected_sections,
                    reference_sections=navigation_seed_sections,
                    query=llm_queries[0],
                    as_of_date=provision_as_of_date,
                    max_total_sections=_regulatory_reference_expansion_limit(
                        len(selected_sections), max_selected_sections
                    ),
                )
                selected_sections = expand_selected_regulatory_sections(
                    provision_session,
                    selected_sections,
                    # evidence_target is reporting-only; retrieval remains driven
                    # exclusively by the model-written query.
                    query=llm_queries[0],
                    as_of_date=provision_as_of_date,
                    max_total_sections=(max_selected_sections),
                )
        if ranked_regulatory_sections is not None:
            selected_sections = _backfill_ranked_regulatory_sections(
                selected_sections,
                ranked_regulatory_sections,
                max_selected_sections,
            )

        search_docs = convert_inference_sections_to_search_docs(
            _rich_response_sections(
                returned_sections,
                selected_sections,
                authoritative_selected=regulatory_chunks_only,
            )[: override_kwargs.num_hits],
            is_internet=False,
        )

        # To show the users, we only pass in the docs that are determined to be good by the LLM
        final_ui_docs = convert_inference_sections_to_search_docs(
            selected_sections[: override_kwargs.num_hits], is_internet=False
        )

        self.emitter.emit(
            Packet(
                placement=placement,
                obj=SearchToolDocumentsDelta(
                    documents=final_ui_docs,
                ),
            )
        )

        (
            docs_str,
            citation_mapping,
            citation_chunk_mapping,
        ) = convert_inference_sections_to_llm_string(
            top_sections=selected_sections,
            citation_start=override_kwargs.starting_citation_num,
            limit=override_kwargs.max_llm_chunks,
            include_document_id=False,
            include_link=override_kwargs.include_link,
            note=scope_note or None,
        )
        docs_str = _add_regulatory_provision_navigation(
            docs_str,
            regulatory_navigation,
        )

        # End overall timing
        overall_elapsed = time.time() - overall_start_time
        logger.debug(
            "Search tool - Total execution time: %s seconds (document selection: %ss)",
            format(overall_elapsed, ".3f"),
            format(document_selection_elapsed, ".3f"),
        )

        llm_facing_response = (
            _add_search_receipt(
                docs_str,
                coverage_item=coverage_item,
                evidence_target=evidence_target,
            )
            if regulatory_chunks_only
            else docs_str
        )

        return ToolResponse(
            # Typically the rich response will give more docs in case it needs to be displayed in the UI
            rich_response=SearchDocsResponse(
                search_docs=search_docs,
                citation_mapping=citation_mapping,
                citation_chunk_mapping=citation_chunk_mapping,
                displayed_docs=final_ui_docs,
            ),
            # The LLM facing response typically includes less docs to cut down on noise and token usage
            llm_facing_response=llm_facing_response,
        )
