import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.chat_utils import create_tool_call_failure_messages
from onyx.chat.citation_processor import (
    CitationMapping,
    CitationMode,
    DynamicCitationProcessor,
)
from onyx.chat.citation_utils import (
    collapse_citations,
    update_citation_processor_from_tool_response,
)
from onyx.chat.emitter import Emitter
from onyx.chat.llm_loop import construct_message_history
from onyx.chat.llm_step import run_llm_step, run_llm_step_pkt_generator
from onyx.chat.models import ChatMessageSimple, LlmStepResult, ToolCallSimple
from onyx.configs.chat_configs import DR_REPORT_LLM_TIMEOUT_S
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.deep_research.dr_mock_tools import (
    RESEARCH_AGENT_TASK_KEY,
    THINK_TOOL_RESPONSE_MESSAGE,
    THINK_TOOL_RESPONSE_TOKEN_COUNT,
    get_research_agent_additional_tool_definitions,
)
from onyx.deep_research.models import (
    CombinedResearchAgentCallResult,
    ResearchAgentCallResult,
)
from onyx.deep_research.utils import (
    check_special_tool_calls,
    create_think_tool_token_processor,
)
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort, ToolChoiceOptions
from onyx.prompts.deep_research.dr_tool_prompts import (
    OPEN_URLS_TOOL_DESCRIPTION,
    OPEN_URLS_TOOL_DESCRIPTION_REASONING,
    WEB_SEARCH_TOOL_DESCRIPTION,
)
from onyx.prompts.deep_research.research_agent import (
    MAX_RESEARCH_CYCLES,
    OPEN_URL_REMINDER_RESEARCH_AGENT,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT_REASONING,
    RESEARCH_REPORT_PROMPT,
    USER_REPORT_QUERY,
)
from onyx.prompts.prompt_utils import get_current_llm_day_time
from onyx.prompts.tool_prompts import INTERNAL_SEARCH_GUIDANCE
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerEvidenceChunk,
    build_candidate_answer_evidence_chunk,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    IntermediateReportCitedDocs,
    IntermediateReportDelta,
    IntermediateReportStart,
    Packet,
    PacketException,
    ResearchAgentStart,
    SectionEnd,
    StreamingType,
)
from onyx.tools.interface import Tool
from onyx.tools.models import ToolCallInfo, ToolCallKickoff, ToolResponse
from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.utils import extract_url_snippet_map
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tools.tool_runner import run_tool_calls
from onyx.tools.utils import (
    compute_all_tool_tokens,
    compute_tool_definition_tokens,
    generate_tools_description,
)
from onyx.tracing.framework.create import function_span
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()


# 30 minute timeout per research agent
RESEARCH_AGENT_TIMEOUT_SECONDS = 30 * 60
RESEARCH_AGENT_TIMEOUT_MESSAGE = "Research Agent timed out after 30 minutes"
# 12 minute timeout before forcing intermediate report generation
RESEARCH_AGENT_FORCE_REPORT_SECONDS = 12 * 60
# May be good to experiment with this, empirically reports of around 5,000 tokens are pretty good.
MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS = 10000
# Focused fragments should remain small enough that a full bounded parent run can
# retain all reports in a 50k context without paying for arbitrary verbosity.
REGULATORY_MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS = 3072
# Allows a complete search -> assess -> search pattern for every research cycle,
# while preventing repeated reasoning-only calls from running until the wall timeout.
MAX_RESEARCH_AGENT_LLM_DECISIONS = MAX_RESEARCH_CYCLES * 2
# Keep a focused research turn broad enough for lower-ranked structural hits while
# avoiding the default 25-chunk payload on every iterative decision.
_REGULATORY_RESEARCH_MAX_LLM_CHUNKS_PER_CALL = 12
_OLDER_SEARCH_DECISION_EXCERPTS_PER_RESPONSE = 2
_OLDER_SEARCH_DECISION_EXCERPT_CHARS = 240
_OLDER_SEARCH_DECISION_VALUE_CHARS = 360
_OLDER_SEARCH_DECISION_TITLE_CHARS = 220
_OLDER_SEARCH_DECISION_TITLE_SOURCE_CHARS = 80
_OLDER_SEARCH_DECISION_TITLE_TERMINAL_PARTS = 2
_REGULATORY_REPORT_EVIDENCE_CALL_ID = "regulatory-report-evidence"
_SEARCH_MODES = frozenset({"hybrid", "keyword", "full_text"})
_REGULATORY_DUPLICATE_SEARCH_FEEDBACK = (
    "This internal_search call was not executed because its normalized query and "
    "search_mode exactly repeat an earlier attempt in this focused research task. "
    "If a material issue remains unresolved, call internal_search with a materially "
    "different query or search_mode; otherwise call generate_report."
)
_REGULATORY_EXTRA_SEARCH_CALLS_FEEDBACK = (
    "Only the first internal_search call from the previous decision was executed; "
    "{count} additional retrieval call(s) were deferred so you can inspect the "
    "result before choosing the next query, mode, or generate_report."
)


@dataclass(frozen=True)
class ResearchAgentRunBudget:
    """Optional per-call bounds; ordinary research keeps the established defaults."""

    max_research_cycles: int
    max_llm_decisions: int
    max_report_tokens: int

    def __post_init__(self) -> None:
        if (
            self.max_research_cycles <= 0
            or self.max_llm_decisions <= 0
            or self.max_report_tokens <= 0
        ):
            raise ValueError("research-agent run budget values must be positive")


_SearchResultIdentity = tuple[str, str, str]
_DecisionInventoryIdentity = tuple[str, ...]
_SearchQueryModeIdentity = tuple[str, str]


class _ResearchAgentState(Protocol):
    def get_tool_calls(self) -> list[ToolCallInfo]: ...

    def add_tool_call(self, tool_call: ToolCallInfo) -> None: ...

    def add_search_docs(self, search_docs: list[SearchDoc]) -> None: ...


class _ResearchAgentOutputGate:
    """Atomically stop a timed-out agent from mutating shared chat output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = True

    def forward(self, action: Callable[[], None]) -> None:
        with self._lock:
            if self._active:
                action()

    def revoke(self) -> None:
        with self._lock:
            self._active = False


class _ResearchAgentEmitter(Emitter):
    def __init__(self, delegate: Emitter, gate: _ResearchAgentOutputGate) -> None:
        self._delegate = delegate
        self._gate = gate

    def emit(self, packet: Packet) -> None:
        self._gate.forward(lambda: self._delegate.emit(packet))

    def revoke(self) -> None:
        self._gate.revoke()


class _ResearchAgentStateView:
    """Gate the three shared-state operations used by a research agent."""

    def __init__(
        self,
        delegate: ChatStateContainer,
        gate: _ResearchAgentOutputGate,
    ) -> None:
        self._delegate = delegate
        self._gate = gate

    def get_tool_calls(self) -> list[ToolCallInfo]:
        return self._delegate.get_tool_calls()

    def add_tool_call(self, tool_call: ToolCallInfo) -> None:
        self._gate.forward(lambda: self._delegate.add_tool_call(tool_call))

    def add_search_docs(self, search_docs: list[SearchDoc]) -> None:
        self._gate.forward(lambda: self._delegate.add_search_docs(search_docs))


def _regulatory_search_llm_chunk_cap(tools: list[Tool]) -> int | None:
    """Bound only the evidence shown to regulatory research decisions."""

    for tool in tools:
        if (
            isinstance(tool, SearchTool)
            and tool.user_selected_filters is not None
            and tool.user_selected_filters.regulatory_chunks_only
        ):
            return _REGULATORY_RESEARCH_MAX_LLM_CHUNKS_PER_CALL
    return None


def _compact_decision_value(value: str, max_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _compact_decision_value_head_tail(value: str, max_chars: int) -> str:
    """Keep both identity-bearing ends of a long navigation value."""

    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    separator = " … "
    head_chars = (max_chars - len(separator)) // 2
    tail_chars = max_chars - len(separator) - head_chars
    return compact[:head_chars].rstrip() + separator + compact[-tail_chars:].lstrip()


def _compact_regulatory_inventory_title(
    title: str,
    *,
    heading_path: list[str] | None,
) -> str:
    """Preserve the source and terminal provision instead of a long path prefix."""

    compact_title = " ".join(title.split())
    if len(compact_title) <= _OLDER_SEARCH_DECISION_TITLE_CHARS:
        return compact_title

    source_identity = compact_title
    title_path: list[str] = []
    if " — " in compact_title:
        source_identity, raw_title_path = compact_title.split(" — ", 1)
        title_path = [part.strip() for part in raw_title_path.split(" > ") if part]
    elif " > " in compact_title:
        title_parts = [part.strip() for part in compact_title.split(" > ") if part]
        if title_parts:
            source_identity = title_parts[0]
            title_path = title_parts[1:]

    terminal_parts = [part for part in (heading_path or title_path) if part.strip()][
        -_OLDER_SEARCH_DECISION_TITLE_TERMINAL_PARTS:
    ]
    if terminal_parts:
        compact_source = _compact_decision_value_head_tail(
            source_identity,
            _OLDER_SEARCH_DECISION_TITLE_SOURCE_CHARS,
        )
        compact_title = f"{compact_source} — … > {' > '.join(terminal_parts)}"

    return _compact_decision_value_head_tail(
        compact_title,
        _OLDER_SEARCH_DECISION_TITLE_CHARS,
    )


def _search_result_metadata(result: dict[str, object]) -> dict[str, object] | None:
    raw_metadata = result.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            parsed_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed_metadata, dict):
            return {str(key): value for key, value in parsed_metadata.items()}
        return None
    if isinstance(raw_metadata, dict):
        return {str(key): value for key, value in raw_metadata.items()}
    return None


def _regulatory_search_result_identity(
    result: dict[str, object],
) -> _SearchResultIdentity | None:
    """Return a stable identity for exact evidence already shown to one agent."""

    metadata = _search_result_metadata(result)
    if metadata is not None:
        regulatory_chunk_id = metadata.get("regulatory_chunk_id")
        if isinstance(regulatory_chunk_id, str) and regulatory_chunk_id.strip():
            content = result.get("content")
            if isinstance(content, str):
                return ("regulatory_chunk_id", regulatory_chunk_id.strip(), content)

    title = result.get("title")
    content = result.get("content")
    if isinstance(title, str) and isinstance(content, str):
        return ("exact_content", title, content)
    return None


def _regulatory_search_query_mode_identity(
    tool_call: ToolCallKickoff,
) -> _SearchQueryModeIdentity | None:
    """Normalize only exact query/mode identity, without semantic deduplication."""

    if tool_call.tool_name != SearchTool.NAME:
        return None
    raw_queries = tool_call.tool_args.get("queries")
    raw_mode = tool_call.tool_args.get("search_mode")
    if (
        not isinstance(raw_queries, list)
        or len(raw_queries) != 1
        or not isinstance(raw_queries[0], str)
        or not raw_queries[0].strip()
        or not isinstance(raw_mode, str)
        or not raw_mode.strip()
    ):
        return None
    return (
        " ".join(raw_queries[0].casefold().split()),
        raw_mode.strip().casefold(),
    )


def _append_regulatory_duplicate_search_feedback(
    history: list[ChatMessageSimple],
    *,
    tool_call: ToolCallKickoff,
    token_counter: Callable[[str], int],
) -> None:
    """Record a rejected call as a provider-valid assistant/tool response pair."""

    tool_call_message = tool_call.to_msg_str()
    tool_call_token_count = token_counter(tool_call_message)
    history.extend(
        [
            ChatMessageSimple(
                message="",
                token_count=tool_call_token_count,
                message_type=MessageType.ASSISTANT,
                tool_calls=[
                    ToolCallSimple(
                        tool_call_id=tool_call.tool_call_id,
                        tool_name=tool_call.tool_name,
                        tool_arguments=tool_call.tool_args,
                        token_count=tool_call_token_count,
                    )
                ],
                image_files=None,
            ),
            ChatMessageSimple(
                message=_REGULATORY_DUPLICATE_SEARCH_FEEDBACK,
                token_count=token_counter(_REGULATORY_DUPLICATE_SEARCH_FEEDBACK),
                message_type=MessageType.TOOL_CALL_RESPONSE,
                tool_call_id=tool_call.tool_call_id,
                image_files=None,
            ),
        ]
    )


def _exact_regulatory_evidence_from_search_response(
    tool_response: ToolResponse,
) -> list[CandidateAnswerEvidenceChunk]:
    """Capture the exact regulatory text exposed to a research-agent decision."""

    try:
        payload = json.loads(tool_response.llm_facing_response)
    except json.JSONDecodeError:
        return []
    search_response = tool_response.rich_response
    if not isinstance(payload, dict) or not isinstance(
        search_response, SearchDocsResponse
    ):
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    search_docs_by_identity = {
        (search_doc.document_id, search_doc.chunk_ind): search_doc
        for search_doc in search_response.search_docs
    }

    evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        result = {str(key): value for key, value in raw_result.items()}
        citation_number = result.get("document")
        content = result.get("content")
        if (
            not isinstance(citation_number, int)
            or citation_number < 1
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue

        document_id = search_response.citation_mapping.get(citation_number)
        chunk_ind = search_response.citation_chunk_mapping.get(citation_number)
        if document_id is None or chunk_ind is None:
            continue
        search_doc = search_docs_by_identity.get((document_id, chunk_ind))
        if search_doc is None:
            continue

        result_metadata = _search_result_metadata(result) or {}
        search_doc_metadata = search_doc.metadata
        result_chunk_id = result_metadata.get("regulatory_chunk_id")
        search_doc_chunk_id = search_doc_metadata.get("regulatory_chunk_id")
        if (
            isinstance(result_chunk_id, str)
            and isinstance(search_doc_chunk_id, str)
            and result_chunk_id.strip() != search_doc_chunk_id.strip()
        ):
            logger.warning(
                "Skipped mismatched regulatory evidence for citation %d",
                citation_number,
            )
            continue

        metadata = {**search_doc_metadata, **result_metadata}
        raw_heading_path = metadata.get("regulatory_heading_path")
        heading = (
            " > ".join(part for part in raw_heading_path if isinstance(part, str))
            if isinstance(raw_heading_path, list)
            and all(isinstance(part, str) for part in raw_heading_path)
            else search_doc.semantic_identifier
        )
        raw_regulatory_chunk_id = result_chunk_id or search_doc_metadata.get(
            "regulatory_chunk_id"
        )
        if isinstance(raw_regulatory_chunk_id, str) and raw_regulatory_chunk_id.strip():
            chunk_identifier = raw_regulatory_chunk_id.strip()
        else:
            chunk_identifier = f"{document_id}:{chunk_ind}"

        evidence_chunks.append(
            build_candidate_answer_evidence_chunk(
                citation_number=citation_number,
                retrieval_number=citation_number,
                chunk_identifier=chunk_identifier,
                heading=heading,
                content=content,
            )
        )
    return evidence_chunks


def _update_regulatory_search_result_novelty(
    response: str,
    *,
    seen_result_identities: set[_SearchResultIdentity],
) -> tuple[int, int] | None:
    """Count newly exposed and exact-repeat chunks in one LLM-facing response."""

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return None

    new_result_count = 0
    repeated_result_count = 0
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            new_result_count += 1
            continue
        identity = _regulatory_search_result_identity(
            {str(key): value for key, value in raw_result.items()}
        )
        if identity is None:
            new_result_count += 1
        elif identity in seen_result_identities:
            repeated_result_count += 1
        else:
            seen_result_identities.add(identity)
            new_result_count += 1
    return new_result_count, repeated_result_count


def _regulatory_search_novelty_reminder(
    novelty: tuple[int, int] | None,
) -> str | None:
    """Expose retrieval novelty without deciding whether research is complete."""

    if novelty is None:
        return None
    new_result_count, repeated_result_count = novelty
    if new_result_count > 0 and repeated_result_count == 0:
        return None
    return (
        "Retrieval novelty (execution metadata, not legal evidence): the latest "
        f"internal search exposed {new_result_count} previously unseen regulatory "
        f"evidence chunk version(s) and {repeated_result_count} exact result(s) "
        "already seen by this research agent. Decide from the unresolved legal "
        "proposition and the actual chunks whether a materially different search "
        "is useful; do not treat overlap or an empty result alone as proof that "
        "research is complete."
    )


def _search_call_batches(
    history: list[ChatMessageSimple],
) -> list[set[str]]:
    response_ids = {
        message.tool_call_id
        for message in history
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
        and message.tool_call_id is not None
        and message.message != _REGULATORY_DUPLICATE_SEARCH_FEEDBACK
    }
    batches: list[set[str]] = []
    for message in history:
        if message.message_type != MessageType.ASSISTANT or not message.tool_calls:
            continue
        completed_call_ids = {
            tool_call.tool_call_id
            for tool_call in message.tool_calls
            if tool_call.tool_name == SearchTool.NAME
            and tool_call.tool_call_id in response_ids
        }
        if completed_call_ids:
            batches.append(completed_call_ids)
    return batches


def _older_search_result_inventory_item(
    result: dict[str, object],
    *,
    include_excerpt: bool,
) -> dict[str, object]:
    inventory_item: dict[str, object] = {}
    document = result.get("document")
    if isinstance(document, (int, str)):
        inventory_item["document"] = document

    metadata = _search_result_metadata(result)
    raw_heading_path = metadata.get("regulatory_heading_path") if metadata else None
    heading_path = (
        [part for part in raw_heading_path if isinstance(part, str) and part.strip()]
        if isinstance(raw_heading_path, list)
        else None
    )

    title = result.get("title")
    if isinstance(title, str) and title.strip():
        inventory_item["title"] = _compact_regulatory_inventory_title(
            title,
            heading_path=heading_path,
        )

    for key in (
        "url",
        "document_identifier",
        "file_name",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            inventory_item[key] = _compact_decision_value(
                value,
                _OLDER_SEARCH_DECISION_VALUE_CHARS,
            )

    authors = result.get("authors")
    if isinstance(authors, list) and all(isinstance(author, str) for author in authors):
        inventory_item["authors"] = [
            _compact_decision_value(author, _OLDER_SEARCH_DECISION_VALUE_CHARS)
            for author in authors
            if isinstance(author, str)
        ]

    if metadata is not None:
        regulatory_chunk_id = metadata.get("regulatory_chunk_id")
        if isinstance(regulatory_chunk_id, str):
            inventory_item["regulatory_chunk_id"] = _compact_decision_value(
                regulatory_chunk_id,
                _OLDER_SEARCH_DECISION_VALUE_CHARS,
            )
        if heading_path is not None:
            inventory_item["regulatory_heading_path"] = [
                _compact_decision_value(
                    part,
                    _OLDER_SEARCH_DECISION_VALUE_CHARS,
                )
                for part in heading_path
            ]

    if include_excerpt:
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            inventory_item["decision_excerpt"] = _compact_decision_value(
                content,
                _OLDER_SEARCH_DECISION_EXCERPT_CHARS,
            )
    return inventory_item


def _decision_inventory_identity(
    result: dict[str, object],
) -> _DecisionInventoryIdentity | None:
    """Prefer canonical chunk identity; otherwise retain exact-result semantics."""

    metadata = _search_result_metadata(result)
    regulatory_chunk_id = metadata.get("regulatory_chunk_id") if metadata else None
    if isinstance(regulatory_chunk_id, str) and regulatory_chunk_id.strip():
        return ("regulatory_chunk_id", regulatory_chunk_id.strip())

    return _regulatory_search_result_identity(result)


def _deduplicated_older_regulatory_results_by_message(
    history: list[ChatMessageSimple],
    *,
    completed_search_call_ids: set[str],
    older_search_call_ids: set[str],
) -> dict[int, tuple[list[dict[str, object]], int]]:
    """Retain every unique result at its newest occurrence in decision history."""

    seen_result_identities: set[_DecisionInventoryIdentity] = set()
    projections: dict[int, tuple[list[dict[str, object]], int]] = {}

    for message_index in range(len(history) - 1, -1, -1):
        message = history[message_index]
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in completed_search_call_ids
        ):
            continue

        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or not all(
            isinstance(result, dict) for result in raw_results
        ):
            continue

        retained_results: list[dict[str, object]] = []
        omitted_duplicate_count = 0
        for raw_result in raw_results:
            assert isinstance(raw_result, dict)
            result = {str(key): value for key, value in raw_result.items()}
            result_identity = _decision_inventory_identity(result)
            if (
                result_identity is not None
                and result_identity in seen_result_identities
            ):
                omitted_duplicate_count += 1
                continue
            retained_results.append(result)
            if result_identity is not None:
                seen_result_identities.add(result_identity)

        if message.tool_call_id in older_search_call_ids:
            projections[message_index] = (
                retained_results,
                omitted_duplicate_count,
            )

    return projections


def _deduplicated_older_regulatory_navigation_by_message(
    history: list[ChatMessageSimple],
    *,
    completed_search_call_ids: set[str],
    older_search_call_ids: set[str],
) -> dict[int, tuple[dict[str, object] | None, int]]:
    """Retain the newest occurrence of each older provision-navigation lead."""

    seen_heading_keys: set[tuple[str, str]] = set()
    projections: dict[int, tuple[dict[str, object] | None, int]] = {}

    for message_index in range(len(history) - 1, -1, -1):
        message = history[message_index]
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in completed_search_call_ids
        ):
            continue

        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_navigation = payload.get("regulatory_provision_navigation")
        if not isinstance(raw_navigation, dict):
            continue

        document_title = raw_navigation.get("document_title")
        raw_headings = raw_navigation.get("headings")
        if (
            not isinstance(document_title, str)
            or not document_title.strip()
            or not isinstance(raw_headings, list)
        ):
            continue

        headings: list[dict[str, object]] = []
        heading_keys: list[tuple[str, str]] = []
        normalized_document_title = " ".join(document_title.split())
        for raw_heading in raw_headings:
            if not isinstance(raw_heading, dict):
                break
            article_key = raw_heading.get("article_key")
            heading_label = raw_heading.get("heading_label")
            if (
                not isinstance(article_key, str)
                or not article_key.strip()
                or not isinstance(heading_label, str)
                or not heading_label.strip()
            ):
                break
            headings.append({str(key): value for key, value in raw_heading.items()})
            heading_keys.append(
                (normalized_document_title, " ".join(article_key.split()))
            )
        else:
            retained_headings: list[dict[str, object]] = []
            omitted_duplicate_count = 0
            for heading, heading_key in zip(headings, heading_keys):
                if heading_key in seen_heading_keys:
                    omitted_duplicate_count += 1
                    continue
                seen_heading_keys.add(heading_key)
                retained_headings.append(heading)

            if (
                message.tool_call_id in older_search_call_ids
                and omitted_duplicate_count
            ):
                if retained_headings:
                    projected_navigation = {
                        str(key): value for key, value in raw_navigation.items()
                    }
                    projected_navigation["headings"] = retained_headings
                else:
                    projected_navigation = None
                projections[message_index] = (
                    projected_navigation,
                    omitted_duplicate_count,
                )

    return projections


def _project_research_history_for_tool_decision(
    history: list[ChatMessageSimple],
    *,
    token_counter: Callable[[str], int],
) -> tuple[list[ChatMessageSimple], int]:
    """Compact older local searches while keeping the newest evidence verbatim."""

    search_batches = _search_call_batches(history)
    completed_search_call_ids = (
        set().union(*search_batches) if search_batches else set()
    )
    if len(completed_search_call_ids) < 2:
        return history, 0

    latest_search_call_ids = search_batches[-1]
    older_search_call_ids = completed_search_call_ids - latest_search_call_ids
    navigation_projections = _deduplicated_older_regulatory_navigation_by_message(
        history,
        completed_search_call_ids=completed_search_call_ids,
        older_search_call_ids=older_search_call_ids,
    )
    result_projections = _deduplicated_older_regulatory_results_by_message(
        history,
        completed_search_call_ids=completed_search_call_ids,
        older_search_call_ids=older_search_call_ids,
    )
    projected_history: list[ChatMessageSimple] = []
    projected_result_count = 0

    for message_index, message in enumerate(history):
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in older_search_call_ids
        ):
            projected_history.append(message)
            continue

        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            projected_history.append(message)
            continue
        if not isinstance(payload, dict):
            projected_history.append(message)
            continue
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            projected_history.append(message)
            continue
        if not all(isinstance(result, dict) for result in raw_results):
            projected_history.append(message)
            continue

        result_projection = result_projections.get(message_index)
        if result_projection is None:
            retained_results = [
                {str(key): value for key, value in result.items()}
                for result in raw_results
                if isinstance(result, dict)
            ]
            omitted_duplicate_result_count = 0
        else:
            retained_results, omitted_duplicate_result_count = result_projection
        inventory = [
            _older_search_result_inventory_item(
                result,
                include_excerpt=(index < _OLDER_SEARCH_DECISION_EXCERPTS_PER_RESPONSE),
            )
            for index, result in enumerate(retained_results)
        ]
        if any(not item for item in inventory):
            projected_history.append(message)
            continue

        payload["results"] = []
        payload["search_result_inventory"] = inventory
        payload.pop("receipt", None)
        payload.pop("note", None)
        navigation_projection = navigation_projections.get(message_index)
        if navigation_projection is not None:
            projected_navigation, _ = navigation_projection
            if projected_navigation is None:
                payload.pop("regulatory_provision_navigation", None)
            else:
                payload["regulatory_provision_navigation"] = projected_navigation
        raw_compaction = payload.get("history_compaction")
        compaction = dict(raw_compaction) if isinstance(raw_compaction, dict) else {}
        compaction.pop("note", None)
        compaction["full_text_results_omitted_for_research_decision"] = len(raw_results)
        if omitted_duplicate_result_count:
            compaction["duplicate_results_omitted_for_research_decision"] = (
                omitted_duplicate_result_count
            )
        if navigation_projection is not None:
            _, omitted_duplicate_count = navigation_projection
            compaction[
                "duplicate_regulatory_navigation_headings_omitted_for_research_decision"
            ] = omitted_duplicate_count
        payload["history_compaction"] = compaction
        projected_message = json.dumps(payload, indent=2, ensure_ascii=False)
        projected_token_count = token_counter(projected_message)
        if projected_token_count >= message.token_count:
            projected_history.append(message)
            continue
        projected_result_count += len(raw_results)
        projected_history.append(
            message.model_copy(
                update={
                    "message": projected_message,
                    "token_count": projected_token_count,
                }
            )
        )

    return projected_history, projected_result_count


def _compact_regulatory_report_history(
    *,
    research_topic: str,
    history: list[ChatMessageSimple],
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk],
    citation_processor: DynamicCitationProcessor,
    token_counter: Callable[[str], int],
) -> list[ChatMessageSimple]:
    """Build an ephemeral exact-evidence view for one regulatory report.

    The canonical history remains untouched for persistence and later decisions.
    Any uncertainty falls back to that history rather than risking lost evidence or
    a citation remap.
    """

    user_messages = [
        message for message in history if message.message_type == MessageType.USER
    ]
    if len(user_messages) != 1 or user_messages[0].message != research_topic:
        return history
    if not exact_evidence_chunks or any(
        chunk.content_truncated for chunk in exact_evidence_chunks
    ):
        return history

    search_calls: dict[str, dict[str, Any]] = {}
    for message in history:
        if message.message_type != MessageType.ASSISTANT or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if tool_call.tool_name != SearchTool.NAME:
                continue
            if tool_call.tool_call_id in search_calls:
                return history
            search_calls[tool_call.tool_call_id] = tool_call.tool_arguments
    if not search_calls:
        return history

    evidence_by_citation: dict[int, CandidateAnswerEvidenceChunk] = {}
    for evidence_chunk in exact_evidence_chunks:
        citation_number = evidence_chunk.citation_number
        if citation_number is None or citation_number in evidence_by_citation:
            return history
        evidence_by_citation[citation_number] = evidence_chunk

    attempts: list[dict[str, object]] = []
    raw_citation_numbers: set[int] = set()
    completed_search_call_ids: set[str] = set()
    for message in history:
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in search_calls
        ):
            continue
        assert message.tool_call_id is not None
        if message.tool_call_id in completed_search_call_ids:
            return history
        completed_search_call_ids.add(message.tool_call_id)
        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            return history
        if not isinstance(payload, dict):
            return history
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or not all(
            isinstance(result, dict) for result in raw_results
        ):
            return history

        result_citations: list[int] = []
        for raw_result in raw_results:
            assert isinstance(raw_result, dict)
            citation_number = raw_result.get("document")
            if not isinstance(citation_number, int) or citation_number < 1:
                return history
            evidence_chunk = evidence_by_citation.get(citation_number)
            raw_content = raw_result.get("content")
            if (
                evidence_chunk is None
                or not isinstance(raw_content, str)
                or raw_content.strip() != evidence_chunk.content
            ):
                return history
            result_metadata = _search_result_metadata(
                {str(key): value for key, value in raw_result.items()}
            )
            raw_chunk_identifier = (
                result_metadata.get("regulatory_chunk_id")
                if result_metadata is not None
                else None
            )
            if (
                not isinstance(raw_chunk_identifier, str)
                or raw_chunk_identifier.strip() != evidence_chunk.chunk_identifier
            ):
                return history
            raw_citation_numbers.add(citation_number)
            result_citations.append(citation_number)

        tool_arguments = search_calls[message.tool_call_id]
        raw_queries = tool_arguments.get("queries")
        if (
            not isinstance(raw_queries, list)
            or len(raw_queries) != 1
            or not isinstance(raw_queries[0], str)
            or not raw_queries[0].strip()
        ):
            return history
        query = raw_queries[0].strip()
        raw_mode = tool_arguments.get("search_mode")
        if not isinstance(raw_mode, str) or raw_mode not in _SEARCH_MODES:
            return history
        receipt = payload.get("receipt")
        if not isinstance(receipt, dict):
            return history
        raw_coverage_item = receipt.get("coverage_item")
        raw_evidence_target = receipt.get("evidence_target")
        if (
            not isinstance(raw_coverage_item, str)
            or not raw_coverage_item.strip()
            or not isinstance(raw_evidence_target, str)
            or not raw_evidence_target.strip()
        ):
            return history
        coverage_item = raw_coverage_item.strip()
        evidence_target = raw_evidence_target.strip()
        attempts.append(
            {
                "query": _compact_decision_value(
                    query, _OLDER_SEARCH_DECISION_VALUE_CHARS
                ),
                "search_mode": raw_mode,
                "coverage_item": _compact_decision_value(
                    coverage_item, _OLDER_SEARCH_DECISION_VALUE_CHARS
                ),
                "evidence_target": _compact_decision_value(
                    evidence_target, _OLDER_SEARCH_DECISION_VALUE_CHARS
                ),
                "status": "results" if raw_results else "zero_results",
                "result_count": len(raw_results),
                "returned_citation_numbers": result_citations,
            }
        )

    if completed_search_call_ids != set(search_calls):
        return history
    if raw_citation_numbers != set(evidence_by_citation):
        return history

    compact_results: list[dict[str, object]] = []
    kept_citation_by_evidence: dict[tuple[str, int, str], int] = {}
    citation_aliases: dict[int, int] = {}
    for citation_number, evidence_chunk in evidence_by_citation.items():
        search_doc = citation_processor.citation_to_doc.get(citation_number)
        if search_doc is None:
            return history
        raw_chunk_identifier = search_doc.metadata.get("regulatory_chunk_id")
        if (
            not isinstance(raw_chunk_identifier, str)
            or raw_chunk_identifier.strip() != evidence_chunk.chunk_identifier
        ):
            return history

        evidence_key = (
            search_doc.document_id,
            search_doc.chunk_ind,
            evidence_chunk.content,
        )
        kept_citation_number = kept_citation_by_evidence.get(evidence_key)
        if kept_citation_number is not None:
            citation_aliases[citation_number] = kept_citation_number
            continue
        kept_citation_by_evidence[evidence_key] = citation_number
        citation_aliases[citation_number] = citation_number

        compact_result: dict[str, object] = {
            "document": citation_number,
            "source": search_doc.semantic_identifier,
            "heading": evidence_chunk.heading,
            "regulatory_chunk_id": evidence_chunk.chunk_identifier,
            "content": evidence_chunk.content,
        }
        validity_start = search_doc.metadata.get("regulatory_validity_start_date")
        validity_end = search_doc.metadata.get("regulatory_validity_end_date")
        if isinstance(validity_start, str) and validity_start:
            compact_result["regulatory_validity_start_date"] = validity_start
        if isinstance(validity_end, str) and validity_end:
            compact_result["regulatory_validity_end_date"] = validity_end
        compact_results.append(compact_result)

    for attempt in attempts:
        raw_attempt_citations = attempt["returned_citation_numbers"]
        assert isinstance(raw_attempt_citations, list)
        attempt["returned_citation_numbers"] = list(
            dict.fromkeys(
                citation_aliases[citation_number]
                for citation_number in raw_attempt_citations
                if isinstance(citation_number, int)
            )
        )

    compact_response = json.dumps(
        {
            "type": "validated_regulatory_search_evidence",
            "usage_note": (
                "The results contain exact regulatory chunk text validated against "
                "the search citation mapping. Search-attempt metadata is execution "
                "context, not legal evidence; zero results do not prove that a rule "
                "is absent. Exact duplicate chunks are shown once without changing "
                "the retained local citation numbers."
            ),
            "search_attempts": attempts,
            "results": compact_results,
        },
        indent=2,
        ensure_ascii=False,
    )
    compact_tool_arguments: dict[str, Any] = {
        "compacted_completed_search_attempts": len(attempts)
    }
    compact_tool_call_text = ToolCallKickoff(
        tool_call_id=_REGULATORY_REPORT_EVIDENCE_CALL_ID,
        tool_name=SearchTool.NAME,
        tool_args=compact_tool_arguments,
        placement=Placement(turn_index=0),
    ).to_msg_str()
    compact_assistant_message = ChatMessageSimple(
        message="",
        token_count=token_counter(compact_tool_call_text),
        message_type=MessageType.ASSISTANT,
        tool_calls=[
            ToolCallSimple(
                tool_call_id=_REGULATORY_REPORT_EVIDENCE_CALL_ID,
                tool_name=SearchTool.NAME,
                tool_arguments=compact_tool_arguments,
                token_count=token_counter(compact_tool_call_text),
            )
        ],
        image_files=None,
    )
    compact_response_message = ChatMessageSimple(
        message=compact_response,
        token_count=token_counter(compact_response),
        message_type=MessageType.TOOL_CALL_RESPONSE,
        tool_call_id=_REGULATORY_REPORT_EVIDENCE_CALL_ID,
        image_files=None,
    )
    compact_history = [
        user_messages[0],
        compact_assistant_message,
        compact_response_message,
    ]
    if sum(message.token_count for message in compact_history) >= sum(
        message.token_count for message in history
    ):
        return history

    logger.info(
        "Compacted regulatory report evidence from %d to %d history tokens "
        "(%d exact duplicate chunk(s) omitted)",
        sum(message.token_count for message in history),
        sum(message.token_count for message in compact_history),
        len(evidence_by_citation) - len(compact_results),
    )
    return compact_history


def _fork_tools_for_independent_research_agent(
    tools: list[Tool],
    *,
    emitter: Emitter | None = None,
) -> list[Tool]:
    forked_tools: list[Tool] = []
    for tool in tools:
        if not isinstance(tool, SearchTool):
            forked_tools.append(tool)
            continue
        forked_tool = tool.fork_for_independent_context(emitter=emitter)
        forked_tools.append(forked_tool)
    return forked_tools


def _build_research_agent_call_result(
    *,
    intermediate_report: str,
    citation_processor: DynamicCitationProcessor,
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk],
) -> ResearchAgentCallResult:
    seen_citations = citation_processor.get_seen_citations()
    deduplicated_chunks: list[CandidateAnswerEvidenceChunk] = []
    chunk_indexes: dict[tuple[str, str], int] = {}
    for evidence_chunk in exact_evidence_chunks:
        evidence_key = (evidence_chunk.chunk_identifier, evidence_chunk.content)
        existing_index = chunk_indexes.get(evidence_key)
        if existing_index is None:
            chunk_indexes[evidence_key] = len(deduplicated_chunks)
            deduplicated_chunks.append(evidence_chunk)
            continue
        if (
            evidence_chunk.citation_number in seen_citations
            and deduplicated_chunks[existing_index].citation_number
            not in seen_citations
        ):
            deduplicated_chunks[existing_index] = evidence_chunk

    normalized_chunks: list[CandidateAnswerEvidenceChunk] = []
    evidence_citation_mapping: CitationMapping = {}
    for evidence_chunk in deduplicated_chunks:
        local_retrieval_number = (
            evidence_chunk.retrieval_number or evidence_chunk.citation_number
        )
        search_doc = (
            citation_processor.citation_to_doc.get(local_retrieval_number)
            if local_retrieval_number is not None
            else None
        )
        if search_doc is None or local_retrieval_number is None:
            normalized_chunks.append(
                evidence_chunk.model_copy(
                    update={"citation_number": None, "retrieval_number": None}
                )
            )
            continue

        raw_chunk_identifier = search_doc.metadata.get("regulatory_chunk_id")
        mapped_chunk_identifier = (
            raw_chunk_identifier.strip()
            if isinstance(raw_chunk_identifier, str) and raw_chunk_identifier.strip()
            else f"{search_doc.document_id}:{search_doc.chunk_ind}"
        )
        if mapped_chunk_identifier != evidence_chunk.chunk_identifier:
            logger.warning(
                "Skipped mismatched exact-evidence citation mapping for %s",
                evidence_chunk.chunk_identifier,
            )
            normalized_chunks.append(
                evidence_chunk.model_copy(
                    update={"citation_number": None, "retrieval_number": None}
                )
            )
            continue

        evidence_citation_mapping[local_retrieval_number] = search_doc
        normalized_chunks.append(
            evidence_chunk.model_copy(
                update={
                    "citation_number": (
                        local_retrieval_number
                        if local_retrieval_number in seen_citations
                        else None
                    ),
                    "retrieval_number": local_retrieval_number,
                }
            )
        )

    return ResearchAgentCallResult(
        intermediate_report=intermediate_report,
        citation_mapping=seen_citations,
        evidence_citation_mapping=evidence_citation_mapping,
        exact_evidence_chunks=normalized_chunks,
    )


def generate_intermediate_report(
    research_topic: str,
    history: list[ChatMessageSimple],
    llm: LLM,
    token_counter: Callable[[str], int],
    citation_processor: DynamicCitationProcessor,
    user_identity: LLMUserIdentity | None,
    emitter: Emitter,
    placement: Placement,
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    max_tokens: int = MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS,
) -> str:
    # NOTE: This step outputs a lot of tokens and has been observed to run for more than 10 minutes in a nontrivial percentage of
    # research tasks. This is also model / inference provider dependent.
    with function_span("generate_intermediate_report") as span:
        span.span_data.input = (
            f"research_topic={research_topic}, history_length={len(history)}"
        )
        # Having the state container here to handle the tokens and not passed through means there is no way to
        # get partial saves of the report. Arguably this is not useful anyway so not going to implement partial saves.
        state_container = ChatStateContainer()
        system_prompt = ChatMessageSimple(
            message=RESEARCH_REPORT_PROMPT,
            token_count=token_counter(RESEARCH_REPORT_PROMPT),
            message_type=MessageType.SYSTEM,
        )

        reminder_str = USER_REPORT_QUERY
        reminder_message = ChatMessageSimple(
            message=reminder_str,
            token_count=token_counter(reminder_str),
            message_type=MessageType.USER,
        )

        research_history = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=history,
            reminder_message=reminder_message,
            context_files=None,
            available_tokens=llm.config.max_input_tokens,
        )

        intermediate_report_generator = run_llm_step_pkt_generator(
            history=research_history,
            tool_definitions=[],
            tool_choice=ToolChoiceOptions.NONE,
            llm=llm,
            placement=placement,
            citation_processor=citation_processor,
            state_container=state_container,
            reasoning_effort=reasoning_effort,
            final_documents=None,
            user_identity=user_identity,
            max_tokens=max_tokens,
            use_existing_tab_index=True,
            is_deep_research=True,
            timeout_override=DR_REPORT_LLM_TIMEOUT_S,
        )

        while True:
            try:
                packet = next(intermediate_report_generator)
                # Translate AgentResponseStart/Delta packets to IntermediateReportStart/Delta
                # Use original placement consistently for all packets
                if isinstance(packet.obj, AgentResponseStart):
                    emitter.emit(
                        Packet(
                            placement=placement,
                            obj=IntermediateReportStart(),
                        )
                    )
                elif isinstance(packet.obj, AgentResponseDelta):
                    emitter.emit(
                        Packet(
                            placement=placement,
                            obj=IntermediateReportDelta(content=packet.obj.content),
                        )
                    )
                else:
                    # Pass through other packet types (e.g., ReasoningStart, ReasoningDelta, etc.)
                    # Also use original placement to keep everything in the same group
                    emitter.emit(
                        Packet(
                            placement=placement,
                            obj=packet.obj,
                        )
                    )
            except StopIteration as e:
                llm_step_result, _ = e.value
                # Use original placement for completion packets
                emitter.emit(
                    Packet(
                        placement=placement,
                        obj=IntermediateReportCitedDocs(
                            cited_docs=list(
                                citation_processor.get_seen_citations().values()
                            )
                        ),
                    )
                )
                emitter.emit(
                    Packet(
                        placement=placement,
                        obj=SectionEnd(),
                    )
                )
                break

        llm_step_result = cast(LlmStepResult, llm_step_result)

        final_report = llm_step_result.answer
        span.span_data.output = final_report if final_report else None
        if final_report is None:
            raise ValueError(
                f"LLM failed to generate a report for research task: {research_topic}"
            )

        return final_report


def run_research_agent_call(
    research_agent_call: ToolCallKickoff,
    parent_tool_call_id: str,
    tools: list[Tool],
    emitter: Emitter,
    state_container: _ResearchAgentState,
    llm: LLM,
    is_reasoning_model: bool,
    token_counter: Callable[[str], int],
    user_identity: LLMUserIdentity | None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    run_budget: ResearchAgentRunBudget | None = None,
) -> ResearchAgentCallResult | None:
    turn_index = research_agent_call.placement.turn_index
    tab_index = research_agent_call.placement.tab_index
    with function_span("research_agent") as span:
        span.span_data.input = str(research_agent_call.tool_args)
        try:
            # Track start time for timeout-based forced report generation
            start_time = time.monotonic()

            # Used to track citations while keeping original citation markers in intermediate reports.
            # KEEP_MARKERS preserves citation markers like [1], [2] in the text unchanged
            # while tracking which documents were cited via get_seen_citations().
            # This allows collapse_citations() to later renumber them in the final report.
            citation_processor = DynamicCitationProcessor(
                citation_mode=CitationMode.KEEP_MARKERS
            )

            research_cycle_count = 0
            llm_cycle_count = 0
            current_tools = tools
            reasoning_cycles = 0
            just_ran_web_search = False
            is_regulatory_research = (
                _regulatory_search_llm_chunk_cap(current_tools) is not None
            )
            seen_regulatory_search_result_identities: set[_SearchResultIdentity] = set()
            attempted_regulatory_search_query_modes: set[_SearchQueryModeIdentity] = (
                set()
            )
            pending_regulatory_search_novelty: tuple[int, int] | None = None
            pending_regulatory_extra_search_calls = 0
            exact_regulatory_evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
            exact_regulatory_evidence_keys: set[tuple[int, str, str]] = set()
            max_research_cycles = (
                run_budget.max_research_cycles
                if run_budget is not None
                else MAX_RESEARCH_CYCLES
            )
            max_llm_decisions = (
                run_budget.max_llm_decisions
                if run_budget is not None
                else MAX_RESEARCH_AGENT_LLM_DECISIONS
            )
            report_max_tokens = (
                run_budget.max_report_tokens
                if run_budget is not None
                else (
                    REGULATORY_MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS
                    if is_regulatory_research
                    else MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS
                )
            )

            # If this fails to parse, we can't run the loop anyway, let this one fail in that case
            research_topic = research_agent_call.tool_args[RESEARCH_AGENT_TASK_KEY]

            emitter.emit(
                Packet(
                    placement=Placement(turn_index=turn_index, tab_index=tab_index),
                    obj=ResearchAgentStart(research_task=research_topic),
                )
            )

            initial_user_message = ChatMessageSimple(
                message=research_topic,
                token_count=token_counter(research_topic),
                message_type=MessageType.USER,
            )
            msg_history: list[ChatMessageSimple] = [initial_user_message]

            citation_mapping: dict[int, str] = {}
            most_recent_reasoning: str | None = None
            llm_decision_count = 0
            while research_cycle_count <= max_research_cycles:
                # Check if we've exceeded the time limit - if so, skip LLM and generate report
                elapsed_seconds = time.monotonic() - start_time
                if elapsed_seconds > RESEARCH_AGENT_FORCE_REPORT_SECONDS:
                    logger.info(
                        "Research agent exceeded %ss (elapsed: %ss), forcing intermediate report generation",
                        RESEARCH_AGENT_FORCE_REPORT_SECONDS,
                        format(elapsed_seconds, ".1f"),
                    )
                    break

                if llm_decision_count >= max_llm_decisions:
                    logger.info(
                        "Research agent reached the %s-call LLM decision limit; "
                        "forcing intermediate report generation",
                        max_llm_decisions,
                    )
                    break

                if research_cycle_count == max_research_cycles:
                    # Auto-generate report on last cycle
                    logger.debug("Auto-generating intermediate report on last cycle.")
                    break

                tools_by_name = {tool.name: tool for tool in current_tools}

                tools_description = generate_tools_description(current_tools)

                # Regulatory research already receives the canonical execution
                # principles above and the complete SearchTool schema. Repeating
                # the normal-chat search chapter on every decision only increases
                # cost and dilutes the focused task.
                internal_search_tip = (
                    INTERNAL_SEARCH_GUIDANCE
                    if not is_regulatory_research
                    and any(isinstance(tool, SearchTool) for tool in current_tools)
                    else ""
                )
                web_search_tip = (
                    WEB_SEARCH_TOOL_DESCRIPTION
                    if any(isinstance(tool, WebSearchTool) for tool in current_tools)
                    else ""
                )
                has_open_url_tool: bool = any(
                    isinstance(tool, OpenURLTool) for tool in current_tools
                )
                open_urls_tip = OPEN_URLS_TOOL_DESCRIPTION if has_open_url_tool else ""
                if is_reasoning_model and open_urls_tip:
                    open_urls_tip = OPEN_URLS_TOOL_DESCRIPTION_REASONING

                system_prompt_template = (
                    RESEARCH_AGENT_PROMPT_REASONING
                    if is_reasoning_model
                    else RESEARCH_AGENT_PROMPT
                )
                system_prompt_str = system_prompt_template.format(
                    available_tools=tools_description,
                    current_datetime=get_current_llm_day_time(full_sentence=False),
                    current_cycle_count=research_cycle_count,
                    optional_internal_search_tool_description=internal_search_tip,
                    optional_web_search_tool_description=web_search_tip,
                    optional_open_url_tool_description=open_urls_tip,
                )

                system_prompt = ChatMessageSimple(
                    message=system_prompt_str,
                    token_count=token_counter(system_prompt_str),
                    message_type=MessageType.SYSTEM,
                )

                reminder_parts: list[str] = []
                # Gate the open_url nudge on the tool actually being available.
                if just_ran_web_search and has_open_url_tool:
                    reminder_parts.append(OPEN_URL_REMINDER_RESEARCH_AGENT)
                novelty_reminder = _regulatory_search_novelty_reminder(
                    pending_regulatory_search_novelty
                )
                if novelty_reminder is not None:
                    reminder_parts.append(novelty_reminder)
                if pending_regulatory_extra_search_calls:
                    reminder_parts.append(
                        _REGULATORY_EXTRA_SEARCH_CALLS_FEEDBACK.format(
                            count=pending_regulatory_extra_search_calls
                        )
                    )
                reminder_text = "\n\n".join(reminder_parts)
                reminder_message = (
                    ChatMessageSimple(
                        message=reminder_text,
                        token_count=token_counter(reminder_text),
                        message_type=MessageType.USER,
                    )
                    if reminder_text
                    else None
                )
                pending_regulatory_search_novelty = None
                pending_regulatory_extra_search_calls = 0

                research_agent_tools = get_research_agent_additional_tool_definitions(
                    include_think_tool=not is_reasoning_model
                )
                tool_token_budget = compute_all_tool_tokens(
                    current_tools, token_counter
                ) + compute_tool_definition_tokens(research_agent_tools, token_counter)

                (
                    decision_history,
                    projected_result_count,
                ) = _project_research_history_for_tool_decision(
                    msg_history,
                    token_counter=token_counter,
                )
                if projected_result_count:
                    logger.info(
                        "Projected %d older local search result(s) for research-agent decision",
                        projected_result_count,
                    )

                constructed_history = construct_message_history(
                    system_prompt=system_prompt,
                    custom_agent_prompt=None,
                    simple_chat_history=decision_history,
                    reminder_message=reminder_message,
                    context_files=None,
                    available_tokens=max(
                        0, llm.config.max_input_tokens - tool_token_budget
                    ),
                )

                # Use think tool processor for non-reasoning models to convert
                # think_tool calls to reasoning content (same as dr_loop.py)
                custom_processor = (
                    create_think_tool_token_processor()
                    if not is_reasoning_model
                    else None
                )

                llm_step_result, has_reasoned = run_llm_step(
                    emitter=emitter,
                    history=constructed_history,
                    tool_definitions=[tool.tool_definition() for tool in current_tools]
                    + research_agent_tools,
                    tool_choice=ToolChoiceOptions.REQUIRED,
                    llm=llm,
                    placement=Placement(
                        turn_index=turn_index,
                        tab_index=tab_index,
                        sub_turn_index=llm_cycle_count + reasoning_cycles,
                    ),
                    citation_processor=None,
                    state_container=None,
                    reasoning_effort=reasoning_effort,
                    final_documents=None,
                    user_identity=user_identity,
                    custom_token_processor=custom_processor,
                    use_existing_tab_index=True,
                    is_deep_research=True,
                    # In case the model is tripped up by the long context and gets into an endless loop of
                    # things like null tokens, we set a max token limit here. The call will likely not be valid
                    # in these situations but it at least allows a chance of recovery. None of the tool calls should
                    # be this long.
                    max_tokens=1000,
                )
                llm_decision_count += 1
                if has_reasoned:
                    reasoning_cycles += 1

                tool_responses: list[ToolResponse] = []
                tool_calls = llm_step_result.tool_calls or []

                # TODO handle the restriction of only 1 tool call type per turn
                # This is a problem right now because of the Placement system not allowing for
                # differentiating sub-tool calls.
                # Filter tool calls to only include the first tool type used
                # This prevents mixing different tool types in the same batch
                if tool_calls:
                    first_tool_type = tool_calls[0].tool_name
                    tool_calls = [
                        tc for tc in tool_calls if tc.tool_name == first_tool_type
                    ]
                    if (
                        is_regulatory_research
                        and first_tool_type == SearchTool.NAME
                        and len(tool_calls) > 1
                    ):
                        pending_regulatory_extra_search_calls = len(tool_calls) - 1
                        tool_calls = tool_calls[:1]

                just_ran_web_search = False

                special_tool_calls = check_special_tool_calls(tool_calls=tool_calls)
                if special_tool_calls.generate_report_tool_call:
                    report_history = (
                        _compact_regulatory_report_history(
                            research_topic=research_topic,
                            history=msg_history,
                            exact_evidence_chunks=exact_regulatory_evidence_chunks,
                            citation_processor=citation_processor,
                            token_counter=token_counter,
                        )
                        if is_regulatory_research
                        else msg_history
                    )
                    final_report = generate_intermediate_report(
                        research_topic=research_topic,
                        history=report_history,
                        llm=llm,
                        token_counter=token_counter,
                        citation_processor=citation_processor,
                        user_identity=user_identity,
                        emitter=emitter,
                        reasoning_effort=reasoning_effort,
                        max_tokens=report_max_tokens,
                        placement=Placement(
                            turn_index=turn_index,
                            tab_index=tab_index,
                        ),
                    )
                    span.span_data.output = final_report if final_report else None
                    return _build_research_agent_call_result(
                        intermediate_report=final_report,
                        citation_processor=citation_processor,
                        exact_evidence_chunks=exact_regulatory_evidence_chunks,
                    )
                elif special_tool_calls.think_tool_call:
                    think_tool_call = special_tool_calls.think_tool_call
                    tool_call_message = think_tool_call.to_msg_str()
                    tool_call_token_count = token_counter(tool_call_message)

                    with function_span("think_tool") as think_span:
                        think_span.span_data.input = str(think_tool_call.tool_args)

                        # Create ASSISTANT message with tool_calls (OpenAI parallel format)
                        think_tool_simple = ToolCallSimple(
                            tool_call_id=think_tool_call.tool_call_id,
                            tool_name=think_tool_call.tool_name,
                            tool_arguments=think_tool_call.tool_args,
                            token_count=tool_call_token_count,
                        )
                        think_assistant_msg = ChatMessageSimple(
                            message="",
                            token_count=tool_call_token_count,
                            message_type=MessageType.ASSISTANT,
                            tool_calls=[think_tool_simple],
                            image_files=None,
                        )
                        msg_history.append(think_assistant_msg)

                        think_tool_response_msg = ChatMessageSimple(
                            message=THINK_TOOL_RESPONSE_MESSAGE,
                            token_count=THINK_TOOL_RESPONSE_TOKEN_COUNT,
                            message_type=MessageType.TOOL_CALL_RESPONSE,
                            tool_call_id=think_tool_call.tool_call_id,
                            image_files=None,
                        )
                        msg_history.append(think_tool_response_msg)
                        think_span.span_data.output = THINK_TOOL_RESPONSE_MESSAGE
                    reasoning_cycles += 1
                    most_recent_reasoning = llm_step_result.reasoning
                    continue
                else:
                    if (
                        is_regulatory_research
                        and tool_calls
                        and tool_calls[0].tool_name == SearchTool.NAME
                    ):
                        search_identity = _regulatory_search_query_mode_identity(
                            tool_calls[0]
                        )
                        if (
                            search_identity is not None
                            and search_identity
                            in attempted_regulatory_search_query_modes
                        ):
                            _append_regulatory_duplicate_search_feedback(
                                msg_history,
                                tool_call=tool_calls[0],
                                token_counter=token_counter,
                            )
                            logger.info(
                                "Skipped exact duplicate regulatory research search"
                            )
                            most_recent_reasoning = None
                            llm_cycle_count += 1
                            continue
                        if search_identity is not None:
                            attempted_regulatory_search_query_modes.add(search_identity)

                    parallel_tool_call_results = run_tool_calls(
                        tool_calls=tool_calls,
                        tools=current_tools,
                        message_history=msg_history,
                        user_memory_context=None,
                        user_info=None,
                        citation_mapping=citation_mapping,
                        next_citation_num=citation_processor.get_next_citation_number(),
                        # Packets currently cannot differentiate between parallel calls in a nested level
                        # so we just cannot show parallel calls in the UI. This should not happen for deep research anyhow.
                        max_concurrent_tools=1,
                        # May be better to not do this step, hard to say, needs to be tested
                        skip_search_query_expansion=False,
                        search_llm_chunks_per_call_cap=(
                            _regulatory_search_llm_chunk_cap(current_tools)
                        ),
                        url_snippet_map=extract_url_snippet_map(
                            [
                                search_doc
                                for tool_call in state_container.get_tool_calls()
                                if tool_call.search_docs
                                for search_doc in tool_call.search_docs
                            ]
                        ),
                    )
                    tool_responses = parallel_tool_call_results.tool_responses
                    citation_mapping = (
                        parallel_tool_call_results.updated_citation_mapping
                    )

                    if tool_calls and not tool_responses:
                        failure_messages = create_tool_call_failure_messages(
                            tool_calls, token_counter
                        )
                        msg_history.extend(failure_messages)

                        # If there is a failure like this, we still increment to avoid potential infinite loops
                        research_cycle_count += 1
                        llm_cycle_count += 1
                        continue

                    # Filter to only responses with valid tool_call references
                    valid_tool_responses = [
                        tr for tr in tool_responses if tr.tool_call is not None
                    ]

                    # Build ONE ASSISTANT message with all tool calls (OpenAI parallel format)
                    if valid_tool_responses:
                        tool_calls_simple: list[ToolCallSimple] = []
                        for tool_response in valid_tool_responses:
                            tc = tool_response.tool_call
                            assert tc is not None  # Already filtered above
                            tool_call_message = tc.to_msg_str()
                            tool_call_token_count = token_counter(tool_call_message)
                            tool_calls_simple.append(
                                ToolCallSimple(
                                    tool_call_id=tc.tool_call_id,
                                    tool_name=tc.tool_name,
                                    tool_arguments=tc.tool_args,
                                    token_count=tool_call_token_count,
                                )
                            )

                        total_tool_call_tokens = sum(
                            tc.token_count for tc in tool_calls_simple
                        )
                        assistant_with_tools = ChatMessageSimple(
                            message="",
                            token_count=total_tool_call_tokens,
                            message_type=MessageType.ASSISTANT,
                            tool_calls=tool_calls_simple,
                            image_files=None,
                        )
                        msg_history.append(assistant_with_tools)

                    # Now add tool call info and TOOL_CALL_RESPONSE messages for each
                    for tool_response in valid_tool_responses:
                        tc = tool_response.tool_call
                        assert tc is not None  # Already filtered above
                        tool_call_tab_index = tc.placement.tab_index

                        tool = tools_by_name.get(tc.tool_name)
                        if not tool:
                            raise ValueError(
                                f"Tool '{tc.tool_name}' not found in tools list"
                            )

                        search_docs = None
                        displayed_docs = None
                        if isinstance(tool_response.rich_response, SearchDocsResponse):
                            search_docs = tool_response.rich_response.search_docs
                            displayed_docs = tool_response.rich_response.displayed_docs

                            # Add ALL search docs to state container for DB persistence
                            if search_docs:
                                state_container.add_search_docs(search_docs)

                            # This is used for the Open URL reminder in the next cycle
                            # only do this if the web search tool yielded results
                            if search_docs and tc.tool_name == WebSearchTool.NAME:
                                just_ran_web_search = True

                        # Makes sure the citation processor is updated with all the possible docs
                        # and citation numbers so that it's populated when passed in to report generation.
                        update_citation_processor_from_tool_response(
                            tool_response=tool_response,
                            citation_processor=citation_processor,
                        )

                        # Research Agent is a top level tool call but the tools called by the research
                        # agent are sub-tool calls.
                        tool_call_info = ToolCallInfo(
                            parent_tool_call_id=parent_tool_call_id,
                            # At the DB save level, there is only a turn index, no sub-turn etc.
                            # This is implied by the parent tool call's turn index and the depth
                            # of the tree traversal.
                            turn_index=llm_cycle_count + reasoning_cycles,
                            tab_index=tool_call_tab_index,
                            tool_name=tc.tool_name,
                            tool_call_id=tc.tool_call_id,
                            tool_id=tool.id,
                            reasoning_tokens=llm_step_result.reasoning
                            or most_recent_reasoning,
                            tool_call_arguments=tc.tool_args,
                            tool_call_response=tool_response.llm_facing_response,
                            search_docs=displayed_docs or search_docs,
                            generated_images=None,
                        )
                        state_container.add_tool_call(tool_call_info)

                        tool_response_message = tool_response.llm_facing_response
                        if is_regulatory_research and tc.tool_name == SearchTool.NAME:
                            novelty = _update_regulatory_search_result_novelty(
                                tool_response_message,
                                seen_result_identities=(
                                    seen_regulatory_search_result_identities
                                ),
                            )
                            if _regulatory_search_novelty_reminder(novelty) is not None:
                                pending_regulatory_search_novelty = novelty
                            for (
                                evidence_chunk
                            ) in _exact_regulatory_evidence_from_search_response(
                                tool_response
                            ):
                                assert evidence_chunk.citation_number is not None
                                evidence_key = (
                                    evidence_chunk.citation_number,
                                    evidence_chunk.chunk_identifier,
                                    evidence_chunk.content,
                                )
                                if evidence_key in exact_regulatory_evidence_keys:
                                    continue
                                exact_regulatory_evidence_keys.add(evidence_key)
                                exact_regulatory_evidence_chunks.append(evidence_chunk)
                        tool_response_token_count = token_counter(tool_response_message)

                        tool_response_msg = ChatMessageSimple(
                            message=tool_response_message,
                            token_count=tool_response_token_count,
                            message_type=MessageType.TOOL_CALL_RESPONSE,
                            tool_call_id=tc.tool_call_id,
                            image_files=None,
                        )
                        msg_history.append(tool_response_msg)

                # If it reached this point, it did not call reasoning, so here we wipe it to not save it to multiple turns
                most_recent_reasoning = None
                llm_cycle_count += 1
                research_cycle_count += 1

            # If we've run out of cycles, just try to generate a report from everything so far
            report_history = (
                _compact_regulatory_report_history(
                    research_topic=research_topic,
                    history=msg_history,
                    exact_evidence_chunks=exact_regulatory_evidence_chunks,
                    citation_processor=citation_processor,
                    token_counter=token_counter,
                )
                if is_regulatory_research
                else msg_history
            )
            final_report = generate_intermediate_report(
                research_topic=research_topic,
                history=report_history,
                llm=llm,
                token_counter=token_counter,
                citation_processor=citation_processor,
                user_identity=user_identity,
                emitter=emitter,
                reasoning_effort=reasoning_effort,
                max_tokens=report_max_tokens,
                placement=Placement(
                    turn_index=turn_index,
                    tab_index=tab_index,
                ),
            )
            span.span_data.output = final_report if final_report else None
            return _build_research_agent_call_result(
                intermediate_report=final_report,
                citation_processor=citation_processor,
                exact_evidence_chunks=exact_regulatory_evidence_chunks,
            )

        except Exception as e:
            logger.error("Error running research agent call: %s", e)
            emitter.emit(
                Packet(
                    placement=Placement(turn_index=turn_index, tab_index=tab_index),
                    obj=PacketException(type=StreamingType.ERROR.value, exception=e),
                )
            )
            return None


def _on_research_agent_timeout(
    index: int,  # noqa: ARG001
    func: Callable[..., Any],  # noqa: ARG001
    args: tuple[Any, ...],
) -> ResearchAgentCallResult:
    """Callback for handling research agent timeouts.

    Returns a ResearchAgentCallResult with the timeout message so the research
    can continue with other agents.
    """
    research_agent_call: ToolCallKickoff = args[0]  # First arg
    agent_emitter = args[3]
    if isinstance(agent_emitter, _ResearchAgentEmitter):
        agent_emitter.revoke()
    research_task = research_agent_call.tool_args.get(
        RESEARCH_AGENT_TASK_KEY, "unknown"
    )
    logger.warning(
        "Research agent timed out after %s seconds for task: %s",
        RESEARCH_AGENT_TIMEOUT_SECONDS,
        research_task,
    )
    return ResearchAgentCallResult(
        intermediate_report=RESEARCH_AGENT_TIMEOUT_MESSAGE,
        citation_mapping={},
    )


def _remap_exact_evidence_chunks(
    result: ResearchAgentCallResult,
    combined_citation_mapping: CitationMapping,
) -> list[CandidateAnswerEvidenceChunk]:
    """Apply the same chunk-identity citation collapse used for agent reports."""

    combined_number_by_chunk = {
        (search_doc.document_id, search_doc.chunk_ind): citation_number
        for citation_number, search_doc in combined_citation_mapping.items()
    }
    remapped: list[CandidateAnswerEvidenceChunk] = []
    for evidence_chunk in result.exact_evidence_chunks:
        old_retrieval_number = (
            evidence_chunk.retrieval_number or evidence_chunk.citation_number
        )
        if old_retrieval_number is None:
            remapped.append(evidence_chunk)
            continue
        search_doc = result.evidence_citation_mapping.get(
            old_retrieval_number
        ) or result.citation_mapping.get(old_retrieval_number)
        if search_doc is None:
            remapped.append(
                evidence_chunk.model_copy(
                    update={"citation_number": None, "retrieval_number": None}
                )
            )
            continue
        new_retrieval_number = combined_number_by_chunk.get(
            (search_doc.document_id, search_doc.chunk_ind)
        )
        if new_retrieval_number is None:
            remapped.append(
                evidence_chunk.model_copy(
                    update={"citation_number": None, "retrieval_number": None}
                )
            )
            continue
        remapped.append(
            evidence_chunk.model_copy(
                update={
                    "citation_number": (
                        new_retrieval_number
                        if evidence_chunk.citation_number is not None
                        and old_retrieval_number in result.citation_mapping
                        else None
                    ),
                    "retrieval_number": new_retrieval_number,
                }
            )
        )
    return remapped


def _merge_citation_namespaces(
    *citation_mappings: CitationMapping,
) -> CitationMapping:
    """Merge citation namespaces without allowing one number to change chunks."""

    merged: CitationMapping = {}
    for citation_mapping in citation_mappings:
        for citation_number, search_doc in citation_mapping.items():
            existing_doc = merged.get(citation_number)
            if existing_doc is not None and (
                existing_doc.document_id,
                existing_doc.chunk_ind,
            ) != (search_doc.document_id, search_doc.chunk_ind):
                raise ValueError(
                    "Conflicting citation mappings for global citation "
                    f"{citation_number}"
                )
            merged[citation_number] = search_doc
    return merged


def _project_citation_mapping_into_namespace(
    local_mapping: CitationMapping,
    global_mapping: CitationMapping,
) -> CitationMapping:
    """Project local chunk identities onto their allocated global numbers."""

    global_number_by_chunk: dict[tuple[str, int], int] = {}
    for citation_number, search_doc in global_mapping.items():
        global_number_by_chunk.setdefault(
            (search_doc.document_id, search_doc.chunk_ind), citation_number
        )

    projected: CitationMapping = {}
    for search_doc in local_mapping.values():
        global_number = global_number_by_chunk.get(
            (search_doc.document_id, search_doc.chunk_ind)
        )
        if global_number is None:
            raise ValueError("Citation chunk was not allocated in the global namespace")
        projected[global_number] = global_mapping[global_number]
    return projected


def run_research_agent_calls(
    research_agent_calls: list[ToolCallKickoff],
    parent_tool_call_ids: list[str],
    tools: list[Tool],
    emitter: Emitter,
    state_container: ChatStateContainer,
    llm: LLM,
    is_reasoning_model: bool,
    token_counter: Callable[[str], int],
    citation_mapping: CitationMapping,
    evidence_citation_mapping: CitationMapping | None = None,
    user_identity: LLMUserIdentity | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    run_budget: ResearchAgentRunBudget | None = None,
) -> CombinedResearchAgentCallResult:
    if len(research_agent_calls) != len(parent_tool_call_ids):
        raise ValueError(
            "research_agent_calls and parent_tool_call_ids must have equal lengths"
        )

    # Run all research agent calls in parallel with timeout. Each worker gets a
    # revocable view of the live output so a Python thread that survives a wall
    # timeout cannot append packets or mutate persisted chat state afterward.
    output_gates = [_ResearchAgentOutputGate() for _ in research_agent_calls]
    agent_emitters = [_ResearchAgentEmitter(emitter, gate) for gate in output_gates]
    independently_forked_tools = [
        _fork_tools_for_independent_research_agent(tools, emitter=agent_emitter)
        for agent_emitter in agent_emitters
    ]
    agent_state_views = [
        _ResearchAgentStateView(state_container, gate) for gate in output_gates
    ]
    functions_with_args = [
        (
            run_research_agent_call,
            (
                research_agent_call,
                parent_tool_call_id,
                agent_tools,
                agent_emitter,
                agent_state_view,
                llm,
                is_reasoning_model,
                token_counter,
                user_identity,
                reasoning_effort,
                run_budget,
            ),
        )
        for (
            research_agent_call,
            parent_tool_call_id,
            agent_tools,
            agent_emitter,
            agent_state_view,
        ) in zip(
            research_agent_calls,
            parent_tool_call_ids,
            independently_forked_tools,
            agent_emitters,
            agent_state_views,
        )
    ]

    try:
        research_agent_call_results = run_functions_tuples_in_parallel(
            functions_with_args,
            allow_failures=False,
            timeout=RESEARCH_AGENT_TIMEOUT_SECONDS,
            timeout_callback=_on_research_agent_timeout,
        )
    finally:
        # Also covers exceptional exits where the parallel helper stops before
        # invoking timeout callbacks for every still-running future.
        for output_gate in output_gates:
            output_gate.revoke()

    updated_citation_mapping = dict(citation_mapping)
    updated_evidence_citation_mapping = dict(evidence_citation_mapping or {})
    combined_citation_namespace = _merge_citation_namespaces(
        updated_citation_mapping,
        updated_evidence_citation_mapping,
    )
    updated_answers: list[str | None] = []
    combined_exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
    combined_exact_evidence_indexes: dict[tuple[str, str], int] = {}

    for result in research_agent_call_results:
        if result is None:
            updated_answers.append(None)
            continue

        # Use collapse_citations to renumber citations in the text and merge mappings.
        # Since we use KEEP_MARKERS mode, the intermediate reports have original citation
        # markers like [1], [2] which need to be renumbered for the combined report.
        updated_answer, combined_citation_namespace = collapse_citations(
            answer_text=result.intermediate_report,
            existing_citation_mapping=combined_citation_namespace,
            new_citation_mapping=result.citation_mapping,
        )
        updated_citation_mapping.update(
            _project_citation_mapping_into_namespace(
                result.citation_mapping,
                combined_citation_namespace,
            )
        )
        _, combined_citation_namespace = collapse_citations(
            answer_text="",
            existing_citation_mapping=combined_citation_namespace,
            new_citation_mapping=result.evidence_citation_mapping,
        )
        updated_evidence_citation_mapping.update(
            _project_citation_mapping_into_namespace(
                result.evidence_citation_mapping,
                combined_citation_namespace,
            )
        )
        updated_answers.append(updated_answer)
        for evidence_chunk in _remap_exact_evidence_chunks(
            result,
            combined_citation_namespace,
        ):
            evidence_key = (
                evidence_chunk.chunk_identifier,
                evidence_chunk.content,
            )
            existing_index = combined_exact_evidence_indexes.get(evidence_key)
            if existing_index is None:
                combined_exact_evidence_indexes[evidence_key] = len(
                    combined_exact_evidence_chunks
                )
                combined_exact_evidence_chunks.append(evidence_chunk)
                continue
            if (
                combined_exact_evidence_chunks[existing_index].citation_number is None
                and evidence_chunk.citation_number is not None
            ):
                combined_exact_evidence_chunks[existing_index] = evidence_chunk

    return CombinedResearchAgentCallResult(
        intermediate_reports=updated_answers,
        citation_mapping=updated_citation_mapping,
        evidence_citation_mapping=updated_evidence_citation_mapping,
        exact_evidence_chunks=combined_exact_evidence_chunks,
    )


if __name__ == "__main__":
    from uuid import uuid4

    from onyx.chat.chat_state import ChatStateContainer
    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from onyx.db.models import User
    from onyx.db.persona import get_default_behavior_persona
    from onyx.llm.factory import get_default_llm, get_llm_token_counter
    from onyx.llm.model_capabilities import model_is_reasoning_model
    from onyx.server.query_and_chat.placement import Placement
    from onyx.tools.models import ToolCallKickoff
    from onyx.tools.tool_constructor import construct_tools

    # === CONFIGURE YOUR RESEARCH PROMPT HERE ===
    RESEARCH_PROMPT = "Your test research task."

    SqlEngine.set_app_name("research_agent_script")
    SqlEngine.init_engine(pool_size=5, max_overflow=5)

    with get_session_with_current_tenant() as db_session:
        llm = get_default_llm()
        token_counter = get_llm_token_counter(llm)
        is_reasoning = model_is_reasoning_model(
            llm.config.model_name, llm.config.model_provider
        )

        persona = get_default_behavior_persona(db_session, eager_load_for_tools=True)
        if persona is None:
            raise ValueError("No default persona found")

        user = db_session.query(User).first()
        if user is None:
            raise ValueError("No users found in database. Please create a user first.")

        emitter_queue: queue.Queue = queue.Queue()
        emitter = Emitter(merged_queue=emitter_queue)
        state_container = ChatStateContainer()

        tool_dict = construct_tools(
            persona=persona,
            db_session=db_session,
            emitter=emitter,
            user=user,
            llm=llm,
        )
        tools = [
            tool
            for tool_list in tool_dict.values()
            for tool in tool_list
            if tool.name != "generate_image"
        ]

        logger.info("Running research agent with prompt: %s", RESEARCH_PROMPT)
        logger.info("LLM: %s/%s", llm.config.model_provider, llm.config.model_name)
        logger.info("Tools: %s", [t.name for t in tools])

        result = run_research_agent_call(
            research_agent_call=ToolCallKickoff(
                tool_name="research_agent",
                tool_args={RESEARCH_AGENT_TASK_KEY: RESEARCH_PROMPT},
                tool_call_id=str(uuid4()),
                placement=Placement(turn_index=0, tab_index=0),
            ),
            parent_tool_call_id=str(uuid4()),
            tools=tools,
            emitter=emitter,
            state_container=state_container,
            llm=llm,
            is_reasoning_model=is_reasoning,
            token_counter=token_counter,
            user_identity=None,
        )

        if result is None:
            logger.error("Research agent returned no result")
        else:
            print("\n" + "=" * 80)
            print("RESEARCH AGENT RESULT")
            print("=" * 80)
            print(result.intermediate_report)
            print("=" * 80)
            print(f"Citations: {result.citation_mapping}")
            print(f"Total packets emitted: {emitter_queue.qsize()}")
