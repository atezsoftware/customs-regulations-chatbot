import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.chat_utils import (
    build_python_chat_files_from_search_docs,
    create_tool_call_failure_messages,
)
from onyx.chat.citation_processor import (
    CitationMapping,
    CitationMode,
    DynamicCitationProcessor,
)
from onyx.chat.citation_utils import (
    canonicalize_search_tool_response_citations,
    extract_citation_order_from_text,
    update_citation_processor_from_tool_response,
)
from onyx.chat.emitter import BufferedEmitter, Emitter
from onyx.chat.llm_step import (
    _looks_like_xml_tool_call_payload,
    extract_tool_calls_from_response_text,
    run_llm_step,
)
from onyx.chat.models import (
    ChatMessageSimple,
    ContextFileMetadata,
    ExtractedContextFiles,
    FileToolMetadata,
    LlmStepResult,
    ToolCallSimple,
)
from onyx.chat.prompt_utils import (
    append_grounding_guidance,
    build_reminder_message,
    build_system_prompt,
    get_default_base_system_prompt,
    process_prompt_template,
)
from onyx.chat.staged_generation import commit_staged_llm_step
from onyx.configs.app_configs import INTEGRATION_TESTS_MODE
from onyx.configs.chat_configs import MAX_LLM_CYCLES
from onyx.configs.constants import DEFAULT_PERSONA_ID, DocumentSource, MessageType
from onyx.configs.model_configs import GEN_AI_INPUT_TOKEN_SAFETY_MARGIN
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.memory import UserMemoryContext, add_memory, update_memory_at_index
from onyx.db.models import Persona
from onyx.llm.constants import LlmProviderNames
from onyx.llm.exceptions import ClassifiedLLMError
from onyx.llm.interfaces import LLM, LLMUserIdentity, ToolChoiceOptions
from onyx.llm.model_capabilities import is_true_openai_model
from onyx.llm.models import ReasoningEffort
from onyx.prompts.chat_prompts import IMAGE_GEN_REMINDER, OPEN_URL_REMINDER
from onyx.prompts.prompt_utils import substitute_user_placeholders
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerClaimIssue,
    CandidateAnswerEvidenceChunk,
    build_candidate_answer_evidence_chunk,
    build_regulatory_review_user_context,
    format_candidate_answer_review,
    format_candidate_resolution_review,
    review_regulatory_candidate_answer,
    review_regulatory_candidate_resolution,
)
from onyx.regulatory.gap_recovery import (
    merge_recovery_citation_mapping,
    recovery_search_docs_by_citation,
    run_single_gap_recovery,
    select_priority_recovery_issue,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    OverallStop,
    Packet,
    ToolCallDebug,
    TopLevelBranching,
)
from onyx.tools.built_in_tools import CITEABLE_TOOLS_NAMES, STOPPING_TOOLS_NAMES
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatFile,
    CustomToolCallSummary,
    MemoryToolResponseSnapshot,
    PythonToolRichResponse,
    ToolCallInfo,
    ToolCallKickoff,
    ToolResponse,
)
from onyx.tools.tool_implementations.images.models import FinalImageGenerationResponse
from onyx.tools.tool_implementations.memory.models import MemoryToolResponse
from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.utils import extract_url_snippet_map
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tools.tool_runner import run_tool_calls
from onyx.tools.utils import compute_all_tool_tokens
from onyx.tracing.framework.create import ChatTraceMetadata, trace
from onyx.utils.logger import setup_logger

logger = setup_logger()

_REGULATORY_UNPLANNED_MAX_LLM_CHUNKS_PER_CALL = 8
_REGULATORY_UNPLANNED_MAX_SEARCH_CALLS = 16
_REGULATORY_MAX_PARALLEL_SEARCH_CALLS = 8
_MAX_EMPTY_FINAL_RESPONSE_RETRIES = 1
_REGULATORY_POST_REVIEW_MAIN_CYCLES = 3
_REGULATORY_MAX_CANDIDATE_REVIEWS = 2
_REGULATORY_PROJECTED_STOP_SYNTHESIS_CYCLES = 1
_REGULATORY_RECONSIDERATION_HISTORY_RESULT_THRESHOLD = 32
_REGULATORY_RECONSIDERATION_UNCITED_RESULTS_PER_SEARCH = 1
_REGULATORY_RECONSIDERATION_PROVISION_NEIGHBOR_DISTANCE = 2
_REGULATORY_TOOL_DECISION_INVENTORY_VALUE_CHARS = 360
_REGULATORY_TOOL_DECISION_EXCERPT_CHARS = 240
_REGULATORY_TOOL_DECISION_EXCERPTS_PER_SEARCH = 2
_REGULATORY_TOOL_DECISION_NAVIGATION_HEADINGS = 16
_REGULATORY_TOOL_DECISION_DETAILED_SEARCH_BATCHES = 1
_REGULATORY_TOOL_DECISION_OLDER_RESULTS_PER_SEARCH = 1
_REGULATORY_TOOL_DECISION_VISIBLE_TOKEN_ALLOWANCE = 1536
_REGULATORY_SYNTHESIS_MAX_OUTPUT_TOKENS = 5632
_REGULATORY_REASONING_TOKEN_RESERVE = {
    ReasoningEffort.OFF: 0,
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.AUTO: 2048,
    ReasoningEffort.MEDIUM: 2048,
    ReasoningEffort.HIGH: 4096,
    ReasoningEffort.XHIGH: 4096,
}


@dataclass(frozen=True)
class SearchEvidenceLedgerEntry:
    """Compact receipt for one model-directed internal-search attempt."""

    query: str
    search_mode: str
    result_count: int
    repeated_result_count: int = 0


def _regulatory_llm_step_max_tokens(
    *,
    complex_regulatory_request: bool,
    tool_choice: ToolChoiceOptions,
    projected_tool_decision_history: bool,
    reasoning_effort: ReasoningEffort,
) -> int | None:
    """Bound provider output reservations without constraining ordinary chat."""

    if not complex_regulatory_request:
        return None
    decision_only = (
        projected_tool_decision_history or tool_choice is ToolChoiceOptions.REQUIRED
    )
    if not decision_only:
        return _REGULATORY_SYNTHESIS_MAX_OUTPUT_TOKENS
    return (
        _REGULATORY_TOOL_DECISION_VISIBLE_TOKEN_ALLOWANCE
        + (_REGULATORY_REASONING_TOKEN_RESERVE[reasoning_effort])
    )


def _commit_canonical_tool_decision_step(
    *,
    projected_tool_decision_history: bool,
    buffered_emitter: BufferedEmitter,
    staged_state: ChatStateContainer,
    staged_citation_processor: DynamicCitationProcessor,
    canonical_citation_processor: DynamicCitationProcessor,
    emitter: Emitter,
    state_container: ChatStateContainer,
    pre_answer_processing_time: float,
) -> DynamicCitationProcessor:
    """Publish only decisions generated from the canonical evidence history."""

    if projected_tool_decision_history:
        return canonical_citation_processor
    commit_staged_llm_step(
        buffered_emitter=buffered_emitter,
        staged_state=staged_state,
        staged_citation_processor=staged_citation_processor,
        emitter=emitter,
        state_container=state_container,
        pre_answer_processing_time=pre_answer_processing_time,
    )
    return staged_citation_processor


def _compact_ledger_value(value: str, max_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _format_search_evidence_ledger(
    ledger: list[SearchEvidenceLedgerEntry],
) -> str | None:
    """Expose recent retrieval attempts without imposing a research plan."""

    if not ledger:
        return None

    lines = [
        "# Internal search attempts",
        "This is an execution receipt, not legal evidence. Use the retrieved chunks "
        "to decide support.",
    ]
    for entry in ledger:
        line = (
            "- query: "
            f"{_compact_ledger_value(entry.query, max_chars=180)}; "
            f"mode: {entry.search_mode}; returned new chunks: {entry.result_count}"
        )
        if entry.repeated_result_count:
            line += f"; exact repeats omitted: {entry.repeated_result_count}"
        lines.append(line)

    lines.append(
        "Do not repeat an equivalent retrieval attempt. Retry an unresolved "
        "point only with a materially different query or mode; otherwise synthesize "
        "the supported conclusion and mark any source gap explicitly."
    )
    return "\n".join(lines)


def _search_tool_call_batches(
    history: list[ChatMessageSimple],
) -> list[set[str]]:
    """Return completed search-call ids grouped by assistant tool batch."""

    response_ids = {
        message.tool_call_id
        for message in history
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
        and message.tool_call_id is not None
    }
    batches: list[set[str]] = []
    for message in history:
        if message.message_type != MessageType.ASSISTANT or not message.tool_calls:
            continue
        search_call_ids = {
            tool_call.tool_call_id
            for tool_call in message.tool_calls
            if tool_call.tool_name == SearchTool.NAME
            and tool_call.tool_call_id in response_ids
        }
        if search_call_ids:
            batches.append(search_call_ids)
    return batches


def _regulatory_history_inventory_item(
    result: dict[str, object],
    citation_mapping: CitationMapping | None = None,
    *,
    include_excerpt: bool,
) -> dict[str, object]:
    """Keep navigation metadata while excluding legal text from an old hit."""

    citation_number = result.get("document")
    if not isinstance(citation_number, int):
        return {}
    inventory_item: dict[str, object] = {"document": citation_number}

    metadata: dict[str, object] = {}
    raw_metadata = result.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            parsed_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            parsed_metadata = None
        if isinstance(parsed_metadata, dict):
            metadata = {
                key: value
                for key, value in parsed_metadata.items()
                if isinstance(key, str)
            }
    elif isinstance(raw_metadata, dict):
        metadata = {
            key: value for key, value in raw_metadata.items() if isinstance(key, str)
        }

    mapped_doc = citation_mapping.get(citation_number) if citation_mapping else None
    mapped_metadata = mapped_doc.metadata if mapped_doc is not None else {}
    regulatory_chunk_id = (
        result.get("regulatory_chunk_id")
        or metadata.get("regulatory_chunk_id")
        or mapped_metadata.get("regulatory_chunk_id")
    )
    if not isinstance(regulatory_chunk_id, str) or not regulatory_chunk_id.strip():
        return {}
    inventory_item["regulatory_chunk_id"] = _compact_ledger_value(
        regulatory_chunk_id,
        max_chars=_REGULATORY_TOOL_DECISION_INVENTORY_VALUE_CHARS,
    )
    raw_heading_path = (
        result.get("heading_path")
        or metadata.get("regulatory_heading_path")
        or mapped_metadata.get("regulatory_heading_path")
    )
    if (
        isinstance(raw_heading_path, list)
        and raw_heading_path
        and all(isinstance(part, str) for part in raw_heading_path)
    ):
        heading_path = [part for part in raw_heading_path if isinstance(part, str)]
        inventory_item["heading"] = _compact_ledger_value(
            " > ".join(heading_path),
            max_chars=_REGULATORY_TOOL_DECISION_INVENTORY_VALUE_CHARS,
        )

    # Structural paths already contain the source root. Repeating a display title
    # for every hit makes multi-query decision prompts grow without adding a new
    # retrieval lead.
    if "heading" not in inventory_item:
        title = result.get("title")
        if isinstance(title, str) and title.strip():
            inventory_item["title"] = _compact_ledger_value(
                title,
                max_chars=_REGULATORY_TOOL_DECISION_INVENTORY_VALUE_CHARS,
            )
    if "title" not in inventory_item and "heading" not in inventory_item:
        return {}

    if include_excerpt:
        content = result.get("content")
        if not isinstance(content, str) or not content.strip():
            return {}
        inventory_item["decision_excerpt"] = _compact_ledger_value(
            content,
            max_chars=_REGULATORY_TOOL_DECISION_EXCERPT_CHARS,
        )
    return inventory_item


def _project_regulatory_navigation_for_tool_decision(
    navigation: dict[str, object],
) -> dict[str, object] | None:
    """Keep the nearest structural leads without copying a whole outline."""

    raw_headings = navigation.get("headings")
    if not isinstance(raw_headings, list):
        return None
    headings = [heading for heading in raw_headings if isinstance(heading, dict)]
    if len(headings) != len(raw_headings):
        return None

    projected = dict(navigation)
    projected["usage_note"] = (
        "Navigation leads only, not legal evidence or proof of absence. Retrieve "
        "the operative text before relying on a material lead."
    )
    projected["headings"] = headings[:_REGULATORY_TOOL_DECISION_NAVIGATION_HEADINGS]
    omitted_heading_count = max(
        0,
        len(headings) - _REGULATORY_TOOL_DECISION_NAVIGATION_HEADINGS,
    )
    if omitted_heading_count:
        projected["headings_omitted_for_tool_decision"] = omitted_heading_count
    return projected


def _project_regulatory_history_for_tool_decision(
    history: list[ChatMessageSimple],
    *,
    token_counter: Callable[[str], int],
    citation_mapping: CitationMapping | None = None,
    priority_citation_numbers: set[int] | None = None,
) -> tuple[list[ChatMessageSimple], int]:
    """Give one search decision bounded excerpts instead of full legal text.

    The returned history is ephemeral. The latest search batch keeps stable
    navigation metadata and bounded excerpts; older searches keep one structural
    representative plus any review-priority citations. Persisted tool responses,
    citation mappings, and canonical synthesis history are never changed.
    """

    search_batches = _search_tool_call_batches(history)
    if not search_batches:
        return history, 0
    completed_search_call_ids = set().union(*search_batches)
    # One focused search is already bounded to eight LLM-visible chunks. A
    # projection at that point would add a decision call before the same full
    # evidence is supplied for synthesis, increasing rather than reducing cost.
    if len(completed_search_call_ids) < 2:
        return history, 0
    detailed_search_call_ids = set().union(
        *search_batches[-_REGULATORY_TOOL_DECISION_DETAILED_SEARCH_BATCHES:]
    )

    projected_history: list[ChatMessageSimple] = []
    seen_inventory_citations: set[int] = set()
    seen_navigation_payloads: set[str] = set()
    priority_citations = priority_citation_numbers or set()
    omitted_result_count = 0
    for message in history:
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id is None
            or message.tool_call_id not in completed_search_call_ids
        ):
            projected_history.append(message)
            continue

        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            return history, 0
        if not isinstance(payload, dict):
            return history, 0
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            return history, 0

        inventory: list[dict[str, object]] = []
        omitted_from_message = 0
        excerpt_count = 0
        detailed_search = message.tool_call_id in detailed_search_call_ids
        for raw_result in raw_results:
            if not isinstance(raw_result, dict) or not all(
                isinstance(key, str) for key in raw_result
            ):
                return history, 0
            result: dict[str, object] = {
                key: value for key, value in raw_result.items() if isinstance(key, str)
            }
            omitted_from_message += 1
            citation_number = result.get("document")
            if (
                isinstance(citation_number, int)
                and citation_number in seen_inventory_citations
            ):
                continue
            priority_result = (
                isinstance(citation_number, int)
                and citation_number in priority_citations
            )
            if (
                not detailed_search
                and not priority_result
                and len(inventory) >= _REGULATORY_TOOL_DECISION_OLDER_RESULTS_PER_SEARCH
            ):
                continue
            inventory_item = _regulatory_history_inventory_item(
                result,
                citation_mapping=citation_mapping,
                include_excerpt=(
                    priority_result
                    or (
                        detailed_search
                        and excerpt_count
                        < _REGULATORY_TOOL_DECISION_EXCERPTS_PER_SEARCH
                    )
                ),
            )
            if not inventory_item:
                # Correctness wins over token savings when a canonical
                # regulatory identity cannot be preserved in the inventory.
                return history, 0
            if isinstance(citation_number, int):
                seen_inventory_citations.add(citation_number)
            inventory.append(inventory_item)
            if "decision_excerpt" in inventory_item:
                excerpt_count += 1

        if omitted_from_message == 0:
            projected_history.append(message)
            continue

        omitted_result_count += omitted_from_message
        payload["results"] = []
        payload["search_result_inventory"] = inventory

        raw_navigation = payload.get("regulatory_provision_navigation")
        if isinstance(raw_navigation, dict):
            projected_navigation = _project_regulatory_navigation_for_tool_decision(
                {
                    key: value
                    for key, value in raw_navigation.items()
                    if isinstance(key, str)
                }
            )
            if projected_navigation is None:
                return history, 0
            navigation_signature = json.dumps(
                projected_navigation,
                ensure_ascii=False,
                sort_keys=True,
            )
            if navigation_signature in seen_navigation_payloads:
                payload.pop("regulatory_provision_navigation", None)
            else:
                seen_navigation_payloads.add(navigation_signature)
                payload["regulatory_provision_navigation"] = projected_navigation

        raw_compaction = payload.get("history_compaction")
        compaction = dict(raw_compaction) if isinstance(raw_compaction, dict) else {}
        compaction["full_text_results_omitted_for_tool_decision"] = omitted_from_message
        compaction["note"] = (
            "This is a search-decision view only. Headings, identifiers, bounded "
            "excerpts from the latest search batch, older search representatives, "
            "and provision navigation are leads for deciding whether another "
            "focused search is material; they are not evidence of absence. Final "
            "synthesis uses the unchanged canonical full evidence history."
        )
        payload["history_compaction"] = compaction
        projected_message = json.dumps(payload, indent=2, ensure_ascii=False)
        projected_history.append(
            message.model_copy(
                update={
                    "message": projected_message,
                    "token_count": token_counter(projected_message),
                }
            )
        )

    return projected_history, omitted_result_count


def _join_search_work_reminders(*reminders: str | None) -> str | None:
    present = [reminder for reminder in reminders if reminder]
    return "\n\n".join(present) if present else None


def _hide_projected_tool_decision_output(
    llm_step_result: LlmStepResult,
    *,
    turn_index: int,
) -> LlmStepResult:
    """Keep projected tool actions while discarding their non-canonical narration."""

    normalized_tool_calls = [
        tool_call.model_copy(
            update={
                "placement": tool_call.placement.model_copy(
                    update={
                        "turn_index": turn_index,
                        "tab_index": tab_index,
                        "sub_turn_index": None,
                    }
                )
            }
        )
        for tab_index, tool_call in enumerate(llm_step_result.tool_calls or [])
    ]
    return llm_step_result.model_copy(
        update={
            "reasoning": None,
            "answer": None,
            "raw_answer": None,
            "tool_calls": normalized_tool_calls or None,
        }
    )


def _build_candidate_answer_evidence_chunks(
    *,
    candidate_answer: str,
    citation_mapping: CitationMapping,
    llm_visible_results_by_citation: dict[int, tuple[str, str]],
) -> list[CandidateAnswerEvidenceChunk]:
    """Expose the exact chunk text placed in the answer model's history."""

    citation_order = extract_citation_order_from_text(candidate_answer)
    cited_numbers = set(citation_order)
    ordered_citation_numbers = [
        citation_number
        for citation_number in citation_order
        if citation_number in llm_visible_results_by_citation
    ]
    ordered_citation_numbers.extend(
        citation_number
        for citation_number in llm_visible_results_by_citation
        if citation_number not in cited_numbers
    )
    evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
    seen_chunk_identities: set[tuple[str, int]] = set()

    for citation_number in ordered_citation_numbers:
        search_doc = citation_mapping.get(citation_number)
        if search_doc is None:
            continue
        chunk_identity = (search_doc.document_id, search_doc.chunk_ind)
        if chunk_identity in seen_chunk_identities:
            continue
        seen_chunk_identities.add(chunk_identity)

        _, llm_visible_content = llm_visible_results_by_citation[citation_number]
        if not llm_visible_content.strip():
            continue

        raw_heading_path = search_doc.metadata.get("regulatory_heading_path")
        heading = (
            " > ".join(raw_heading_path)
            if isinstance(raw_heading_path, list)
            else search_doc.semantic_identifier
        )
        raw_regulatory_chunk_id = search_doc.metadata.get("regulatory_chunk_id")
        chunk_identifier = (
            raw_regulatory_chunk_id
            if isinstance(raw_regulatory_chunk_id, str)
            else f"{search_doc.document_id}:{search_doc.chunk_ind}"
        )
        evidence_chunks.append(
            build_candidate_answer_evidence_chunk(
                document_id=search_doc.document_id,
                chunk_id=search_doc.chunk_ind,
                citation_number=(
                    citation_number if citation_number in cited_numbers else None
                ),
                retrieval_number=citation_number,
                chunk_identifier=chunk_identifier,
                heading=heading,
                content=llm_visible_content,
            )
        )

    return evidence_chunks


def _is_bounded_cited_provision_neighbor(
    search_doc: SearchDoc,
    cited_provision_anchors: list[tuple[str, int, tuple[str, ...]]],
) -> bool:
    """Match only nearby structural relatives of a cited legal provision."""

    raw_heading_path = search_doc.metadata.get("regulatory_heading_path")
    if (
        not isinstance(raw_heading_path, list)
        or not raw_heading_path
        or not all(isinstance(part, str) for part in raw_heading_path)
    ):
        return False
    heading_path = tuple(raw_heading_path)

    for document_id, chunk_ind, cited_heading_path in cited_provision_anchors:
        if search_doc.document_id != document_id:
            continue
        if (
            abs(search_doc.chunk_ind - chunk_ind)
            > _REGULATORY_RECONSIDERATION_PROVISION_NEIGHBOR_DISTANCE
        ):
            continue

        same_heading = heading_path == cited_heading_path
        immediate_parent_or_child = (
            len(heading_path) + 1 == len(cited_heading_path)
            and heading_path == cited_heading_path[:-1]
        ) or (
            len(cited_heading_path) + 1 == len(heading_path)
            and cited_heading_path == heading_path[:-1]
        )
        immediate_siblings = (
            len(heading_path) >= 2
            and len(cited_heading_path) >= 2
            and heading_path[:-1] == cited_heading_path[:-1]
        )
        if same_heading or immediate_parent_or_child or immediate_siblings:
            return True

    return False


def _compact_regulatory_search_history_for_reconsideration(
    history: list[ChatMessageSimple],
    *,
    candidate_answer: str,
    citation_mapping: CitationMapping,
    token_counter: Callable[[str], int],
) -> tuple[list[ChatMessageSimple], set[int] | None, int]:
    """Bound a large rejected-draft history without locking in its first answer.

    This changes only the subsequent model history. Persisted tool responses,
    frontend search results, and citation mappings remain untouched. A result
    omitted here can therefore be retrieved again if the model later decides it
    is material. ``None`` as the retained set means no safe compaction was
    possible, so callers must keep their existing visible-evidence registry.
    """

    requested_citations = set(extract_citation_order_from_text(candidate_answer)) & set(
        citation_mapping
    )
    if not requested_citations:
        return history, None, 0

    search_tool_call_ids = {
        tool_call.tool_call_id
        for message in history
        if message.message_type == MessageType.ASSISTANT and message.tool_calls
        for tool_call in message.tool_calls
        if tool_call.tool_name == SearchTool.NAME
    }
    if not search_tool_call_ids:
        return history, None, 0

    total_valid_results = 0
    for message in history:
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in search_tool_call_ids
        ):
            continue
        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            return history, None, 0
        if not isinstance(payload, dict):
            return history, None, 0
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            return history, None, 0
        total_valid_results += sum(isinstance(result, dict) for result in raw_results)
    if total_valid_results <= _REGULATORY_RECONSIDERATION_HISTORY_RESULT_THRESHOLD:
        return history, None, 0

    cited_provision_anchors: list[tuple[str, int, tuple[str, ...]]] = []
    for citation_number in requested_citations:
        search_doc = citation_mapping.get(citation_number)
        if search_doc is None:
            continue
        raw_heading_path = search_doc.metadata.get("regulatory_heading_path")
        if (
            isinstance(raw_heading_path, list)
            and raw_heading_path
            and all(isinstance(part, str) for part in raw_heading_path)
        ):
            cited_provision_anchors.append(
                (
                    search_doc.document_id,
                    search_doc.chunk_ind,
                    tuple(raw_heading_path),
                )
            )

    compacted_history: list[ChatMessageSimple] = []
    retained_citations: set[int] = set()
    omitted_result_count = 0
    for message in history:
        if (
            message.message_type != MessageType.TOOL_CALL_RESPONSE
            or message.tool_call_id not in search_tool_call_ids
        ):
            compacted_history.append(message)
            continue

        try:
            payload = json.loads(message.message)
        except json.JSONDecodeError:
            compacted_history.append(message)
            continue
        if not isinstance(payload, dict):
            compacted_history.append(message)
            continue
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            compacted_history.append(message)
            continue

        retained_results: list[object] = []
        omitted_inventory: list[dict[str, object]] = []
        omitted_from_message = 0
        uncited_representatives = 0
        for result in raw_results:
            if not isinstance(result, dict):
                retained_results.append(result)
                continue
            citation_number = result.get("document")
            search_doc = (
                citation_mapping.get(citation_number)
                if isinstance(citation_number, int)
                else None
            )
            retain_full_text = (
                isinstance(citation_number, int)
                and citation_number in requested_citations
            ) or (
                search_doc is not None
                and _is_bounded_cited_provision_neighbor(
                    search_doc, cited_provision_anchors
                )
            )
            if (
                not retain_full_text
                and uncited_representatives
                < _REGULATORY_RECONSIDERATION_UNCITED_RESULTS_PER_SEARCH
            ):
                retain_full_text = True
                uncited_representatives += 1
            if retain_full_text and isinstance(citation_number, int):
                retained_results.append(result)
                retained_citations.add(citation_number)
            else:
                omitted_from_message += 1
                inventory_item: dict[str, object] = {}
                if isinstance(citation_number, int):
                    inventory_item["document"] = citation_number
                title = result.get("title")
                if isinstance(title, str):
                    inventory_item["title"] = title
                raw_metadata = result.get("metadata")
                if isinstance(raw_metadata, str):
                    try:
                        result_metadata = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        result_metadata = None
                    if isinstance(result_metadata, dict):
                        regulatory_chunk_id = result_metadata.get("regulatory_chunk_id")
                        if isinstance(regulatory_chunk_id, str):
                            inventory_item["regulatory_chunk_id"] = regulatory_chunk_id
                if inventory_item:
                    omitted_inventory.append(inventory_item)

        if omitted_from_message == 0:
            compacted_history.append(message)
            continue

        omitted_result_count += omitted_from_message
        payload["results"] = retained_results
        if omitted_inventory:
            payload["omitted_result_inventory"] = omitted_inventory
        raw_compaction = payload.get("history_compaction")
        compaction = dict(raw_compaction) if isinstance(raw_compaction, dict) else {}
        compaction["omitted_after_candidate_review"] = omitted_from_message
        compaction["note"] = (
            "Full text remains for candidate-cited provision groups and one "
            "uncited ranked representative from this search. Inventory-only hits "
            "are not legal evidence or evidence of absence; their full chunks "
            "remain available to a materially different retrieval attempt."
        )
        payload["history_compaction"] = compaction
        compacted_message = json.dumps(payload, indent=2, ensure_ascii=False)
        compacted_history.append(
            message.model_copy(
                update={
                    "message": compacted_message,
                    "token_count": token_counter(compacted_message),
                }
            )
        )

    return compacted_history, retained_citations, omitted_result_count


def _search_query_mode_identity(
    tool_call: ToolCallKickoff,
) -> tuple[str, str] | None:
    """Return the retrieval semantics used for exact duplicate prevention."""

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


def _constrain_regulatory_tool_calls(
    tool_calls: list[ToolCallKickoff],
    *,
    search_slots: int | None,
    attempted_query_modes: set[tuple[str, str]] | None = None,
) -> list[ToolCallKickoff]:
    """Enforce hard budgets and exact retrieval-attempt deduplication."""

    blocked_query_modes = attempted_query_modes or set()
    used_query_modes: set[tuple[str, str]] = set()
    constrained: list[ToolCallKickoff] = []
    retained_search_calls = 0

    for tool_call in tool_calls:
        if tool_call.tool_name != SearchTool.NAME:
            constrained.append(tool_call)
            continue

        if search_slots is not None and retained_search_calls >= search_slots:
            continue

        query_mode = _search_query_mode_identity(tool_call)
        if query_mode is not None:
            if query_mode in blocked_query_modes or query_mode in used_query_modes:
                continue
            used_query_modes.add(query_mode)

        constrained.append(tool_call)
        retained_search_calls += 1

    return constrained


def _format_regulatory_tool_call_batch_feedback(
    *,
    requested_search_calls: int,
    executed_search_calls: int,
) -> str | None:
    """Describe unexecuted searches without copying their potentially large payloads."""

    unexecuted_search_calls = requested_search_calls - executed_search_calls
    if unexecuted_search_calls <= 0:
        return None

    return (
        "# Internal-search batch receipt\n"
        f"Requested search calls: {requested_search_calls}; "
        f"executed now: {executed_search_calls}; "
        f"not executed: {unexecuted_search_calls}.\n"
        "Calls not executed were blocked by the per-batch safety capacity or "
        "exact query/mode duplicate protection. They produced no evidence and "
        "must not be treated as completed research. After assessing the returned "
        "evidence, decide whether any material unresolved proposition warrants a "
        "focused, materially distinct search in the next decision turn. Do not "
        "mechanically replay every omitted call."
    )


def _extract_llm_visible_search_results(
    llm_facing_response: str,
) -> list[tuple[int, str, str]]:
    """Read exactly the search excerpts placed in the answer model's history."""

    try:
        payload = json.loads(llm_facing_response)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    excerpts: list[tuple[int, str, str]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        citation_number = result.get("document")
        title = result.get("title")
        content = result.get("content")
        if (
            isinstance(citation_number, int)
            and citation_number >= 1
            and isinstance(title, str)
            and isinstance(content, str)
        ):
            excerpts.append((citation_number, title, content))
    return excerpts


def _merge_gathered_search_docs(
    gathered_documents: list[SearchDoc] | None,
    search_docs: list[SearchDoc],
) -> list[SearchDoc] | None:
    """Preserve first-seen chunks without repeating them in streamed context."""

    if not search_docs:
        return gathered_documents
    if gathered_documents is None:
        return list(search_docs)

    merged_documents = list(gathered_documents)
    seen_keys = {
        (document.document_id, document.chunk_ind) for document in gathered_documents
    }
    for document in search_docs:
        key = (document.document_id, document.chunk_ind)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_documents.append(document)
    return merged_documents


def _compact_repeated_search_results_for_history(
    llm_facing_response: str,
    previously_visible_results_by_citation: dict[int, tuple[str, str]],
) -> tuple[str, int]:
    """Omit only byte-equivalent search evidence already present in this turn.

    The persisted tool response and rich/UI response remain untouched. Citation
    canonicalization runs before this helper, so a matching citation, title, and
    content tuple identifies the same exact chunk payload the model already saw.
    """

    try:
        payload = json.loads(llm_facing_response)
    except json.JSONDecodeError:
        return llm_facing_response, 0
    if not isinstance(payload, dict):
        return llm_facing_response, 0

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return llm_facing_response, 0

    compacted_results: list[object] = []
    repeated_result_count = 0
    for result in raw_results:
        if not isinstance(result, dict):
            compacted_results.append(result)
            continue

        citation_number = result.get("document")
        title = result.get("title")
        content = result.get("content")
        is_exact_repeat = (
            isinstance(citation_number, int)
            and isinstance(title, str)
            and isinstance(content, str)
            and previously_visible_results_by_citation.get(citation_number)
            == (title, content)
        )
        if is_exact_repeat:
            repeated_result_count += 1
        else:
            compacted_results.append(result)

    if repeated_result_count == 0:
        return llm_facing_response, 0

    payload["results"] = compacted_results
    payload["history_compaction"] = {
        "omitted_exact_repeats": repeated_result_count,
        "note": (
            "These exact chunk payloads remain available earlier in this turn's "
            "tool history; no new evidence was removed."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False), repeated_result_count


def _regulatory_search_chunk_cap(enabled: bool) -> int | None:
    """Bound context per autonomous search call without choosing its subject."""

    return _REGULATORY_UNPLANNED_MAX_LLM_CHUNKS_PER_CALL if enabled else None


def _regulatory_search_call_budget(
    complex_regulatory_request: bool,
) -> int | None:
    """Keep one chat turn below the proxy timeout and runaway-tool threshold."""

    if complex_regulatory_request:
        return _REGULATORY_UNPLANNED_MAX_SEARCH_CALLS
    return None


def _effective_regulatory_search_call_budget(
    base_budget: int | None,
    *,
    candidate_was_rejected: bool,
) -> int | None:
    """Keep ordinary research fixed; direct review recovery is accounted separately."""

    _ = candidate_was_rejected
    return base_budget


class EmptyLLMResponseError(ClassifiedLLMError):
    """Raised when the streamed LLM response completes without a usable answer."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        tool_choice: ToolChoiceOptions,
        client_error_msg: str,
        error_code: str = "EMPTY_LLM_RESPONSE",
        is_retryable: bool = True,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(
            client_error_msg=client_error_msg,
            error_code=error_code,
            is_retryable=is_retryable,
        )
        self.provider = provider
        self.model = model
        self.tool_choice = tool_choice
        self.finish_reason = finish_reason


# LiteLLM maps these native policy blocks to content_filter, but gateways may
# forward the provider value unchanged.
_REFUSAL_FINISH_REASONS = {
    "BLOCKLIST",
    "CONTENT_BLOCKED",
    "ERROR_TOXIC",
    "IMAGE_OTHER",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_SAFETY",
    "LANGUAGE",
    "MODEL_ARMOR",
    "OTHER",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "SAFETY",
    "SPII",
    "content_filter",
    "content_filtered",
    "guardrail_intervened",
    "refusal",
    "sensitive",
}


def _build_empty_llm_response_error(
    llm: LLM,
    llm_step_result: LlmStepResult,
    tool_choice: ToolChoiceOptions,
) -> EmptyLLMResponseError:
    provider = llm.config.model_provider
    model = llm.config.model_name
    finish_reason = llm_step_result.finish_reason

    # A refusal/content-filter stop is a deliberate model decision (HTTP 200
    # with no content), not a transport failure — retrying the same request
    # against the same model will not help.
    if finish_reason in _REFUSAL_FINISH_REASONS:
        model_suggestion = (
            " (e.g. Claude Opus 4.8)" if provider == LlmProviderNames.ANTHROPIC else ""
        )
        return EmptyLLMResponseError(
            provider=provider,
            model=model,
            tool_choice=tool_choice,
            client_error_msg=(
                "The selected model declined to respond to this request and "
                f"returned no content (finish_reason={finish_reason}). Try "
                "rephrasing the request or switching to a different model"
                f"{model_suggestion}."
            ),
            error_code="MODEL_REFUSAL",
            is_retryable=False,
            finish_reason=finish_reason,
        )

    # OpenAI quota exhaustion has reached us as a streamed "stop" with zero content.
    # When the stream is completely empty and there is no reasoning/tool output, surface
    # the likely account-level cause instead of a generic tool-calling error.
    if (
        not llm_step_result.reasoning
        and provider == LlmProviderNames.OPENAI
        and is_true_openai_model(provider, model)
    ):
        return EmptyLLMResponseError(
            provider=provider,
            model=model,
            tool_choice=tool_choice,
            client_error_msg=(
                "The selected OpenAI model returned an empty streamed response "
                "before producing any tokens. This commonly happens when the API "
                "key or project has no remaining quota or billing is not enabled. "
                "Verify quota and billing for this key and try again."
            ),
            error_code="BUDGET_EXCEEDED",
            is_retryable=False,
            finish_reason=finish_reason,
        )

    return EmptyLLMResponseError(
        provider=provider,
        model=model,
        tool_choice=tool_choice,
        client_error_msg=(
            "The selected model returned no final answer before the stream "
            "completed. No text or tool calls were received from the upstream "
            "provider."
        ),
        finish_reason=finish_reason,
    )


def _try_fallback_tool_extraction(
    llm_step_result: LlmStepResult,
    tool_choice: ToolChoiceOptions,
    fallback_extraction_attempted: bool,
    tool_defs: list[dict],
    turn_index: int,
) -> tuple[LlmStepResult, bool]:
    """Attempt to extract tool calls from response text as a fallback.

    This is a last resort fallback for low quality LLMs or those that don't have
    tool calling from the serving layer. Also triggers if there's reasoning but
    no answer and no tool calls.

    Args:
        llm_step_result: The result from the LLM step
        tool_choice: The tool choice option used for this step
        fallback_extraction_attempted: Whether fallback extraction was already attempted
        tool_defs: List of tool definitions
        turn_index: The current turn index for placement

    Returns:
        Tuple of (possibly updated LlmStepResult, whether fallback was attempted this call)
    """
    if fallback_extraction_attempted:
        return llm_step_result, False

    no_tool_calls = (
        not llm_step_result.tool_calls or len(llm_step_result.tool_calls) == 0
    )
    reasoning_but_no_answer_or_tools = (
        llm_step_result.reasoning and not llm_step_result.answer and no_tool_calls
    )
    xml_tool_call_text_detected = no_tool_calls and (
        _looks_like_xml_tool_call_payload(llm_step_result.answer)
        or _looks_like_xml_tool_call_payload(llm_step_result.raw_answer)
        or _looks_like_xml_tool_call_payload(llm_step_result.reasoning)
    )
    should_try_fallback = (
        (tool_choice == ToolChoiceOptions.REQUIRED and no_tool_calls)
        or reasoning_but_no_answer_or_tools
        or xml_tool_call_text_detected
    )

    if not should_try_fallback:
        return llm_step_result, False

    # Try to extract from answer first, then fall back to reasoning
    extracted_tool_calls: list[ToolCallKickoff] = []

    if llm_step_result.answer:
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.answer,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if (
        not extracted_tool_calls
        and llm_step_result.raw_answer
        and llm_step_result.raw_answer != llm_step_result.answer
    ):
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.raw_answer,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if not extracted_tool_calls and llm_step_result.reasoning:
        extracted_tool_calls = extract_tool_calls_from_response_text(
            response_text=llm_step_result.reasoning,
            tool_definitions=tool_defs,
            placement=Placement(turn_index=turn_index),
        )
    if extracted_tool_calls:
        logger.info(
            "Extracted %s tool call(s) from response text as fallback",
            len(extracted_tool_calls),
        )
        return (
            LlmStepResult(
                reasoning=llm_step_result.reasoning,
                answer=llm_step_result.answer,
                tool_calls=extracted_tool_calls,
                raw_answer=llm_step_result.raw_answer,
                finish_reason=llm_step_result.finish_reason,
            ),
            True,
        )

    return llm_step_result, True


# Default 6 covers the common search → open_url pattern:
# Cycle 1: Calls web_search for something
# Cycle 2: Calls open_url for some results
# Cycle 3: Calls web_search for some other aspect of the question
# Cycle 4: Calls open_url for some results
# Cycle 5: Maybe call open_url for some additional results or because last set failed
# Cycle 6: No more tools available, forced to answer
# Override via the MAX_LLM_CYCLES env var when running with tool-heavy MCPs
# that legitimately need more turns. Imported from chat_configs.


def _build_context_file_citation_mapping(
    file_metadata: list[ContextFileMetadata],
    starting_citation_num: int = 1,
) -> CitationMapping:
    """Build citation mapping for context files.

    Converts context file metadata into SearchDoc objects that can be cited.
    Citation numbers start from the provided starting number.

    Args:
        file_metadata: List of context file metadata
        starting_citation_num: Starting citation number (default: 1)

    Returns:
        Dictionary mapping citation numbers to SearchDoc objects
    """
    citation_mapping: CitationMapping = {}

    for idx, file_meta in enumerate(file_metadata, start=starting_citation_num):
        search_doc = SearchDoc(
            document_id=file_meta.file_id,
            chunk_ind=0,
            semantic_identifier=file_meta.filename,
            link=None,
            blurb=file_meta.file_content,
            source_type=DocumentSource.FILE,
            boost=1,
            hidden=False,
            metadata={},
            score=0.0,
            match_highlights=[file_meta.file_content],
        )
        citation_mapping[idx] = search_doc

    return citation_mapping


def _build_project_message(
    context_files: ExtractedContextFiles | None,
    token_counter: Callable[[str], int] | None,
) -> list[ChatMessageSimple]:
    """Build messages for context-injected / tool-backed files.

    Returns up to two messages:
    1. The full-text files message (if file_texts is populated).
    2. A lightweight metadata message for files the LLM should access via the
       FileReaderTool (e.g. oversized files that don't fit in context).
    """
    if not context_files:
        return []

    messages: list[ChatMessageSimple] = []
    if context_files.file_texts:
        messages.append(
            _create_context_files_message(context_files, token_counter=None)
        )
    if context_files.file_metadata_for_tool and token_counter:
        messages.append(
            _create_file_tool_metadata_message(
                context_files.file_metadata_for_tool, token_counter
            )
        )
    return messages


def construct_message_history(
    system_prompt: ChatMessageSimple | None,
    custom_agent_prompt: ChatMessageSimple | None,
    simple_chat_history: list[ChatMessageSimple],
    reminder_message: ChatMessageSimple | None,
    context_files: ExtractedContextFiles | None,
    available_tokens: int,
    last_n_user_messages: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
) -> list[ChatMessageSimple]:
    if last_n_user_messages is not None:
        if last_n_user_messages <= 0:
            raise ValueError(
                "filtering chat history by last N user messages must be a value greater than 0"
            )

    # Build the project / file-metadata messages up front so we can use their
    # actual token counts for the budget.
    project_messages = _build_project_message(context_files, token_counter)
    project_messages_tokens = sum(m.token_count for m in project_messages)

    history_token_budget = available_tokens
    history_token_budget -= system_prompt.token_count if system_prompt else 0
    history_token_budget -= (
        custom_agent_prompt.token_count if custom_agent_prompt else 0
    )
    history_token_budget -= project_messages_tokens
    history_token_budget -= reminder_message.token_count if reminder_message else 0

    if history_token_budget < 0:
        raise ValueError("Not enough tokens available to construct message history")

    if system_prompt:
        system_prompt.should_cache = True

    # If no history, build minimal context
    if not simple_chat_history:
        result = [system_prompt] if system_prompt else []
        if custom_agent_prompt:
            result.append(custom_agent_prompt)
        result.extend(project_messages)
        if reminder_message:
            result.append(reminder_message)
        return result

    # If last_n_user_messages is set, filter history to only include the last n user messages
    if last_n_user_messages is not None:
        # Find all user message indices
        user_msg_indices = [
            i
            for i, msg in enumerate(simple_chat_history)
            if msg.message_type == MessageType.USER
        ]

        if not user_msg_indices:
            raise ValueError("No user message found in simple_chat_history")

        # If we have more than n user messages, keep only the last n
        if len(user_msg_indices) > last_n_user_messages:
            # Find the index of the n-th user message from the end
            # For example, if last_n_user_messages=2, we want the 2nd-to-last user message
            nth_user_msg_index = user_msg_indices[-(last_n_user_messages)]
            # Keep everything from that user message onwards
            simple_chat_history = simple_chat_history[nth_user_msg_index:]

    # Find the last USER message in the history
    # The history may contain tool calls and responses after the last user message
    last_user_msg_index = None
    for i in range(len(simple_chat_history) - 1, -1, -1):
        if simple_chat_history[i].message_type == MessageType.USER:
            last_user_msg_index = i
            break

    if last_user_msg_index is None:
        raise ValueError("No user message found in simple_chat_history")

    # Split history into three parts:
    # 1. History before the last user message
    # 2. The last user message
    # 3. Messages after the last user message (tool calls, responses, etc.)
    history_before_last_user = simple_chat_history[:last_user_msg_index]
    last_user_message = simple_chat_history[last_user_msg_index]
    messages_after_last_user = simple_chat_history[last_user_msg_index + 1 :]

    # Calculate tokens needed for the last user message and everything after it
    last_user_tokens = last_user_message.token_count
    after_user_tokens = sum(msg.token_count for msg in messages_after_last_user)

    # Check if we can fit at least the last user message and messages after it
    required_tokens = last_user_tokens + after_user_tokens
    if required_tokens > history_token_budget:
        raise ValueError(
            f"Not enough tokens to include the last user message and subsequent messages. "
            f"Required: {required_tokens}, Available: {history_token_budget}"
        )

    # Calculate remaining budget for history before the last user message
    remaining_budget = history_token_budget - required_tokens

    # Truncate history_before_last_user from the top to fit in remaining budget.
    # Track dropped file messages so we can provide their metadata to the
    # FileReaderTool instead.
    truncated_history_before: list[ChatMessageSimple] = []
    dropped_file_ids: list[str] = []
    current_token_count = 0

    for msg in reversed(history_before_last_user):
        if current_token_count + msg.token_count <= remaining_budget:
            msg.should_cache = True
            truncated_history_before.insert(0, msg)
            current_token_count += msg.token_count
        else:
            # Can't fit this message, stop truncating.
            # This message and everything older is dropped.
            break

    # Collect file_ids from ALL dropped messages (those not in
    # truncated_history_before). The truncation loop above keeps the most
    # recent messages, so the dropped ones are at the start of the original
    # list up to (len(history) - len(kept)).
    num_kept = len(truncated_history_before)
    for msg in history_before_last_user[: len(history_before_last_user) - num_kept]:
        if msg.file_id is not None:
            dropped_file_ids.append(msg.file_id)

    # Also treat "orphaned" metadata entries as dropped -- these are files
    # from messages removed by summary truncation (before convert_chat_history
    # ran), so no ChatMessageSimple was ever tagged with their file_id.
    if all_injected_file_metadata:
        surviving_file_ids = {
            msg.file_id for msg in simple_chat_history if msg.file_id is not None
        }
        for fid in all_injected_file_metadata:
            if fid not in surviving_file_ids and fid not in dropped_file_ids:
                dropped_file_ids.append(fid)

    # Build a forgotten-files metadata message if any file messages were
    # dropped AND we have metadata for them (meaning the FileReaderTool is
    # available). Reserve tokens for this message in the budget.
    forgotten_files_message: ChatMessageSimple | None = None
    if dropped_file_ids and all_injected_file_metadata and token_counter:
        forgotten_meta = [
            all_injected_file_metadata[fid]
            for fid in dropped_file_ids
            if fid in all_injected_file_metadata
        ]
        if forgotten_meta:
            logger.debug(
                "FileReader: building forgotten-files message for %s",
                [(m.file_id, m.filename) for m in forgotten_meta],
            )
            forgotten_files_message = _create_file_tool_metadata_message(
                forgotten_meta, token_counter
            )
            # Shrink the remaining budget. If the metadata message doesn't
            # fit we may need to drop more history messages.
            remaining_budget -= forgotten_files_message.token_count
            while truncated_history_before and current_token_count > remaining_budget:
                evicted = truncated_history_before.pop(0)
                current_token_count -= evicted.token_count
                # If the evicted message is itself a file, add it to the
                # forgotten metadata (it's now dropped too).
                if (
                    evicted.file_id is not None
                    and evicted.file_id in all_injected_file_metadata
                    and evicted.file_id not in {m.file_id for m in forgotten_meta}
                ):
                    forgotten_meta.append(all_injected_file_metadata[evicted.file_id])
                    # Rebuild the message with the new entry
                    forgotten_files_message = _create_file_tool_metadata_message(
                        forgotten_meta, token_counter
                    )

    # Build the final message list according to README ordering:
    # [system], [history_before_last_user], [custom_agent], [context_files],
    # [forgotten_files], [last_user_message], [messages_after_last_user], [reminder]
    result = [system_prompt] if system_prompt else []

    # 1. Add truncated history before last user message
    result.extend(truncated_history_before)

    # 2. Add custom agent prompt (inserted before last user message)
    if custom_agent_prompt:
        result.append(custom_agent_prompt)

    # 3. Add context files / file-metadata messages (inserted before last user message)
    result.extend(project_messages)

    # 4. Add forgotten-files metadata (right before the user's question)
    if forgotten_files_message:
        result.append(forgotten_files_message)

    # 5. Add last user message (with context images attached)
    result.append(last_user_message)

    # 6. Add messages after last user message (tool calls, responses, etc.)
    result.extend(messages_after_last_user)

    # 7. Add reminder message at the very end
    if reminder_message:
        result.append(reminder_message)

    return _drop_orphaned_tool_call_responses(result)


def _drop_orphaned_tool_call_responses(
    messages: list[ChatMessageSimple],
) -> list[ChatMessageSimple]:
    """Drop tool response messages whose tool_call_id is not in prior assistant tool calls.

    This can happen when history truncation drops an ASSISTANT tool-call message but
    leaves a later TOOL_CALL_RESPONSE message in context. Some providers (e.g. Ollama)
    reject such history with an "unexpected tool call id" error.
    """
    known_tool_call_ids: set[str] = set()
    sanitized: list[ChatMessageSimple] = []

    for msg in messages:
        if msg.message_type == MessageType.ASSISTANT and msg.tool_calls:
            for tool_call in msg.tool_calls:
                known_tool_call_ids.add(tool_call.tool_call_id)
            sanitized.append(msg)
            continue

        if msg.message_type == MessageType.TOOL_CALL_RESPONSE:
            if msg.tool_call_id and msg.tool_call_id in known_tool_call_ids:
                sanitized.append(msg)
            else:
                logger.debug(
                    "Dropping orphaned tool response with tool_call_id=%s while constructing message history",
                    msg.tool_call_id,
                )
            continue

        sanitized.append(msg)

    return sanitized


def _create_file_tool_metadata_message(
    file_metadata: list[FileToolMetadata],
    token_counter: Callable[[str], int],
) -> ChatMessageSimple:
    """Build a lightweight metadata-only message listing files available via FileReaderTool.

    Used when files are too large to fit in context and the vector DB is
    disabled, so the LLM must use ``read_file`` to inspect them.
    """
    lines = [
        "You have access to the following files. Use the read_file tool to "
        "read sections of any file. You MUST pass the file_id UUID (not the "
        "filename) to read_file:"
    ]
    for meta in file_metadata:
        lines.append(
            f'- file_id="{meta.file_id}" filename="{meta.filename}" (~{meta.approx_char_count:,} chars)'
        )

    message_content = "\n".join(lines)
    return ChatMessageSimple(
        message=message_content,
        token_count=token_counter(message_content),
        message_type=MessageType.USER,
    )


def _create_context_files_message(
    context_files: ExtractedContextFiles,
    token_counter: Callable[[str], int] | None,  # noqa: ARG001
) -> ChatMessageSimple:
    """Convert context files to a ChatMessageSimple message.

    Format follows the README specification for document representation.
    """
    import json

    # Format as documents JSON as described in README
    documents_list = []
    for idx, file_text in enumerate(context_files.file_texts, start=1):
        title = (
            context_files.file_metadata[idx - 1].filename
            if idx - 1 < len(context_files.file_metadata)
            else None
        )
        entry: dict[str, Any] = {"document": idx}
        if title:
            entry["title"] = title
        entry["contents"] = file_text
        documents_list.append(entry)

    documents_json = json.dumps({"documents": documents_list}, indent=2)
    message_content = f"Here are some documents provided for context, they may not all be relevant:\n{documents_json}"

    # Use pre-calculated token count from context_files
    return ChatMessageSimple(
        message=message_content,
        token_count=context_files.total_token_count,
        message_type=MessageType.USER,
    )


def select_reminder_text(
    *,
    ran_image_gen: bool,
    just_ran_web_search: bool,
    has_open_url_tool: bool,
    out_of_cycles: bool,
    persona_task_prompt: str | None,
    include_citation_reminder: bool,
    include_file_reminder: bool,
    search_ledger_reminder: str | None = None,
) -> str | None:
    """Choose the reminder appended after a tool cycle.

    The open_url nudge is gated on the tool actually being available; otherwise
    the model is told to call a tool it doesn't have and leaks confusing
    "open_url is not available" replies.
    """
    if ran_image_gen:
        reminder = IMAGE_GEN_REMINDER
    elif just_ran_web_search and has_open_url_tool and not out_of_cycles:
        reminder = OPEN_URL_REMINDER
    else:
        reminder = build_reminder_message(
            reminder_text=persona_task_prompt,
            include_citation_reminder=include_citation_reminder,
            include_file_reminder=include_file_reminder,
            is_last_cycle=out_of_cycles,
        )
    if not search_ledger_reminder:
        return reminder
    if not reminder:
        return search_ledger_reminder
    return reminder + "\n\n" + search_ledger_reminder


def run_llm_loop(
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    custom_agent_prompt: str | None,
    context_files: ExtractedContextFiles,
    persona: Persona | None,
    user_memory_context: UserMemoryContext | None,
    llm: LLM,
    token_counter: Callable[[str], int],
    forced_tool_id: int | None = None,
    user_identity: LLMUserIdentity | None = None,
    chat_session_id: str | None = None,
    chat_files: list[ChatFile] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    include_citations: bool = True,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    with trace(
        "run_llm_loop",
        group_id=chat_session_id,
        metadata=ChatTraceMetadata(
            chat_session_id=chat_session_id,
            user_id=user_identity.user_id if user_identity else None,
        ).model_dump(),
    ):
        # Fix some LiteLLM issues,
        from onyx.llm.litellm_singleton.config import (
            initialize_litellm,
        )  # Here for lazy load LiteLLM

        initialize_litellm()

        # Normalize chat_files to a mutable list so we can extend it mid-loop
        # when a search hit carries an attached file the Python tool should
        # see.
        chat_files = list(chat_files or [])

        # Track when the loop starts for calculating time-to-answer
        loop_start_time = time.monotonic()

        # Initialize citation processor for handling citations dynamically
        # When include_citations is True, use HYPERLINK mode to format citations as [[1]](url)
        # When include_citations is False, use REMOVE mode to strip citations from output
        citation_processor = DynamicCitationProcessor(
            citation_mode=(
                CitationMode.HYPERLINK if include_citations else CitationMode.REMOVE
            )
        )

        # Add project file citation mappings if project files are present
        project_citation_mapping: CitationMapping = {}
        if context_files.file_metadata:
            project_citation_mapping = _build_context_file_citation_mapping(
                context_files.file_metadata
            )
            citation_processor.update_citation_mapping(project_citation_mapping)

        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=None,
            tool_calls=None,
            raw_answer=None,
            finish_reason=None,
        )

        # Hold back a margin below max_input_tokens: our tiktoken estimate can
        # undercount the provider's tokenizer and overflow the context window.
        available_tokens = int(
            llm.config.max_input_tokens * (1 - GEN_AI_INPUT_TOKEN_SAFETY_MARGIN)
        )
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO
        # Initialize gathered_documents with project files if present
        gathered_documents: list[SearchDoc] | None = (
            list(project_citation_mapping.values())
            if project_citation_mapping
            else None
        )
        # TODO allow citing of images in Projects. Since attached to the last user message, it has no text associated with it.
        # One future workaround is to include the images as separate user messages with citation information and process those.
        always_cite_documents: bool = bool(
            context_files.use_as_search_filter or context_files.file_texts
        )
        should_cite_documents: bool = False
        ran_image_gen: bool = False
        just_ran_web_search: bool = False
        has_open_url_tool: bool = any(isinstance(tool, OpenURLTool) for tool in tools)
        has_called_search_tool: bool = False
        code_interpreter_file_generated: bool = False
        fallback_extraction_attempted: bool = False
        citation_mapping: dict[int, str] = {}  # Maps citation_num -> document_id/URL
        search_evidence_ledger: list[SearchEvidenceLedgerEntry] = []
        llm_visible_search_results_by_citation: dict[int, tuple[str, str]] = {}
        regulatory_search_calls_attempted = 0
        regulatory_attempted_query_modes: set[tuple[str, str]] = set()
        regulatory_tool_feedback: str | None = None
        candidate_review_feedback: str | None = None
        candidate_answer_review_count = 0
        candidate_review_issues: list[CandidateAnswerClaimIssue] = []
        candidate_review_rejected_at_cycle: int | None = None
        candidate_final_correction_pending = False
        regulatory_research_complete_pending = False
        empty_final_response_retries = 0
        complex_regulatory_request = False
        regulatory_user_message = ""
        regulatory_earlier_user_context: tuple[str, ...] = ()

        is_global_regulatory_chat = (
            persona is not None
            and persona.id == DEFAULT_PERSONA_ID
            and any(isinstance(tool, SearchTool) for tool in tools)
        )
        if is_global_regulatory_chat:
            regulatory_review_user_context = build_regulatory_review_user_context(
                simple_chat_history
            )
            regulatory_user_message = regulatory_review_user_context.current_request
            regulatory_earlier_user_context = (
                regulatory_review_user_context.earlier_user_context
            )
            # The answering model owns the research strategy. This flag only
            # activates deterministic search/context safety bounds; it does not
            # create or require a coverage plan.
            complex_regulatory_request = bool(regulatory_user_message.strip())
        regulatory_search_chunk_cap = _regulatory_search_chunk_cap(
            complex_regulatory_request
        )
        regulatory_search_call_budget = _regulatory_search_call_budget(
            complex_regulatory_request=complex_regulatory_request,
        )

        # Fetch this in a short-lived session so the long-running stream loop does
        # not pin a connection just to keep read state alive.
        with get_session_with_current_tenant() as prompt_db_session:
            default_base_system_prompt: str = get_default_base_system_prompt(
                prompt_db_session
            )
        system_prompt = None
        custom_agent_prompt_msg = None

        # Resolve author-controlled `{{user.<key>}}` placeholders in the
        # agent's prompts against the current user's directory profile (+
        # basic identity) once, before the cycle loop — so every branch below
        # and every token count sees the final text. Never mutate the shared
        # `persona`.
        placeholder_values = (
            user_memory_context.user_info.placeholder_values
            if user_memory_context
            else {}
        )
        custom_agent_prompt = (
            substitute_user_placeholders(custom_agent_prompt, placeholder_values)
            if custom_agent_prompt
            else custom_agent_prompt
        )
        persona_system_prompt = (
            substitute_user_placeholders(persona.system_prompt, placeholder_values)
            if persona and persona.system_prompt
            else None
        )
        persona_task_prompt = (
            substitute_user_placeholders(persona.task_prompt, placeholder_values)
            if persona and persona.task_prompt
            else None
        )

        reasoning_cycles = 0
        maximum_cycle_count = MAX_LLM_CYCLES + (
            (
                _REGULATORY_POST_REVIEW_MAIN_CYCLES
                + _REGULATORY_PROJECTED_STOP_SYNTHESIS_CYCLES
            )
            if complex_regulatory_request
            else 0
        )
        for llm_cycle_count in range(maximum_cycle_count):
            out_of_cycles = (
                llm_cycle_count >= MAX_LLM_CYCLES - 1
                if candidate_review_rejected_at_cycle is None
                else llm_cycle_count
                >= candidate_review_rejected_at_cycle
                + _REGULATORY_POST_REVIEW_MAIN_CYCLES
            )
            effective_regulatory_search_call_budget = (
                _effective_regulatory_search_call_budget(
                    regulatory_search_call_budget,
                    candidate_was_rejected=(
                        candidate_review_rejected_at_cycle is not None
                    ),
                )
            )
            regulatory_search_budget_exhausted = (
                effective_regulatory_search_call_budget is not None
                and regulatory_search_calls_attempted
                >= effective_regulatory_search_call_budget
            )
            final_synthesis_required = (
                out_of_cycles
                or regulatory_search_budget_exhausted
                or candidate_final_correction_pending
                or regulatory_research_complete_pending
            )
            # Handling tool calls based on cycle count and past cycle conditions
            final_tools: list[Tool] = []
            if forced_tool_id:
                # Needs to be just the single one because the "required" currently doesn't have a specified tool, just a binary
                final_tools = [tool for tool in tools if tool.id == forced_tool_id]
                if not final_tools:
                    raise ValueError(f"Tool {forced_tool_id} not found in tools")
                tool_choice = ToolChoiceOptions.REQUIRED
                forced_tool_id = None
            elif final_synthesis_required or ran_image_gen:
                # Last cycle, no tools allowed, just answer!
                tool_choice = ToolChoiceOptions.NONE
                final_tools = []
            else:
                tool_choice = ToolChoiceOptions.AUTO
                final_tools = tools

            # Handling the system prompt and custom agent prompt
            # The section below calculates the available tokens for history a bit more accurately
            # now that project files are loaded in.
            persona_datetime_aware = persona.datetime_aware if persona else True
            cite_documents = should_cite_documents or always_cite_documents
            if persona and persona.replace_base_system_prompt:
                # Handles the case where user has checked off the "Replace base system prompt" checkbox
                processed_system_prompt = (
                    append_grounding_guidance(
                        process_prompt_template(
                            persona_system_prompt,
                            datetime_aware=persona_datetime_aware,
                            append_datetime_if_aware=True,
                            should_cite_documents=cite_documents,
                        )
                    )
                    if persona_system_prompt
                    else None
                )
                system_prompt = (
                    ChatMessageSimple(
                        message=processed_system_prompt,
                        token_count=token_counter(processed_system_prompt),
                        message_type=MessageType.SYSTEM,
                    )
                    if processed_system_prompt
                    else None
                )
                custom_agent_prompt_msg = None
            else:
                # If it's an empty string, we assume the user does not want to include it as an empty System message
                if default_base_system_prompt:
                    prompt_memory_context = (
                        user_memory_context
                        if inject_memories_in_prompt
                        else (
                            user_memory_context.without_memories()
                            if user_memory_context
                            else None
                        )
                    )
                    system_prompt_str = build_system_prompt(
                        base_system_prompt=default_base_system_prompt,
                        datetime_aware=persona_datetime_aware,
                        user_memory_context=prompt_memory_context,
                        tools=tools,
                        should_cite_documents=cite_documents,
                    )
                    system_prompt = ChatMessageSimple(
                        message=system_prompt_str,
                        token_count=token_counter(system_prompt_str),
                        message_type=MessageType.SYSTEM,
                    )
                    processed_custom_agent_prompt = (
                        process_prompt_template(
                            custom_agent_prompt,
                            datetime_aware=persona_datetime_aware,
                            append_datetime_if_aware=False,
                            should_cite_documents=cite_documents,
                        )
                        if custom_agent_prompt
                        else None
                    )
                    custom_agent_prompt_msg = (
                        ChatMessageSimple(
                            message=processed_custom_agent_prompt,
                            token_count=token_counter(processed_custom_agent_prompt),
                            message_type=MessageType.USER,
                        )
                        if processed_custom_agent_prompt
                        else None
                    )
                else:
                    # If there is a custom agent prompt, it replaces the system prompt when the default system prompt is empty
                    processed_custom_agent_prompt = (
                        append_grounding_guidance(
                            process_prompt_template(
                                custom_agent_prompt,
                                datetime_aware=persona_datetime_aware,
                                append_datetime_if_aware=True,
                                should_cite_documents=cite_documents,
                            )
                        )
                        if custom_agent_prompt
                        else None
                    )
                    system_prompt = (
                        ChatMessageSimple(
                            message=processed_custom_agent_prompt,
                            token_count=token_counter(processed_custom_agent_prompt),
                            message_type=MessageType.SYSTEM,
                        )
                        if processed_custom_agent_prompt
                        else None
                    )
                    custom_agent_prompt_msg = None

            processed_task_prompt = (
                process_prompt_template(
                    persona_task_prompt,
                    datetime_aware=persona_datetime_aware,
                    append_datetime_if_aware=False,
                    should_cite_documents=cite_documents,
                )
                if persona_task_prompt
                else None
            )
            history_for_llm_step = simple_chat_history
            projected_tool_decision_history = False
            if (
                complex_regulatory_request
                and has_called_search_tool
                and tool_choice is ToolChoiceOptions.AUTO
                and llm_cycle_count < maximum_cycle_count - 1
            ):
                (
                    history_for_llm_step,
                    tool_decision_omitted_result_count,
                ) = _project_regulatory_history_for_tool_decision(
                    simple_chat_history,
                    token_counter=token_counter,
                    citation_mapping=citation_processor.citation_to_doc,
                    priority_citation_numbers={
                        citation_number
                        for issue in candidate_review_issues
                        for citation_number in issue.related_citation_numbers
                    },
                )
                projected_tool_decision_history = tool_decision_omitted_result_count > 0
                if projected_tool_decision_history:
                    logger.info(
                        "Projected %d regulatory result(s) to bounded inventory "
                        "for one autonomous tool decision",
                        tool_decision_omitted_result_count,
                    )

            tool_decision_reminder = (
                "# Retrieval decision turn\n"
                "Decide only whether another materially distinct internal search "
                "is useful. If it is, call the tool with the focused query and mode "
                "you choose. If the evidence is sufficient, return only "
                "RESEARCH_COMPLETE. A materially relevant provision visible only as "
                "a navigation lead, or an explicit source/provision target whose "
                "returned headings belong to other instruments, remains unresolved "
                "for this decision; choose whether a different focused attempt is "
                "useful. Do not draft the report in this turn; the "
                "unchanged full evidence history is supplied to the next synthesis."
                if projected_tool_decision_history
                else None
            )
            reminder_message_text = select_reminder_text(
                ran_image_gen=ran_image_gen,
                just_ran_web_search=just_ran_web_search,
                has_open_url_tool=has_open_url_tool,
                out_of_cycles=final_synthesis_required,
                persona_task_prompt=processed_task_prompt,
                include_citation_reminder=should_cite_documents
                or always_cite_documents,
                include_file_reminder=code_interpreter_file_generated,
                search_ledger_reminder=_join_search_work_reminders(
                    regulatory_tool_feedback,
                    _format_search_evidence_ledger(search_evidence_ledger),
                    candidate_review_feedback,
                    tool_decision_reminder,
                ),
            )
            regulatory_tool_feedback = None

            reminder_msg = (
                ChatMessageSimple(
                    message=reminder_message_text,
                    token_count=token_counter(reminder_message_text),
                    message_type=MessageType.USER_REMINDER,
                )
                if reminder_message_text
                else None
            )

            tool_token_budget = compute_all_tool_tokens(final_tools, token_counter)
            truncated_message_history = construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=custom_agent_prompt_msg,
                simple_chat_history=history_for_llm_step,
                reminder_message=reminder_msg,
                context_files=context_files,
                available_tokens=max(0, available_tokens - tool_token_budget),
                token_counter=token_counter,
                all_injected_file_metadata=all_injected_file_metadata,
            )

            # This calls the LLM, yields packets (reasoning, answers, etc.) and returns the result
            # It also pre-processes the tool calls in preparation for running them
            tool_defs = [tool.tool_definition() for tool in final_tools]

            # Calculate total processing time from loop start until now
            # This measures how long the user waits before the answer starts streaming
            pre_answer_processing_time = time.monotonic() - loop_start_time

            # A response only needs hidden staging after the model has actually
            # used indexed evidence and is therefore eligible for review. Direct
            # conversational answers retain the established incremental stream.
            should_stage_regulatory_step = (
                complex_regulatory_request and has_called_search_tool
            )
            buffered_step_emitter = (
                BufferedEmitter() if should_stage_regulatory_step else None
            )
            staged_state_container = (
                ChatStateContainer()
                if should_stage_regulatory_step
                else state_container
            )
            staged_citation_processor = (
                citation_processor.fork()
                if should_stage_regulatory_step
                else citation_processor
            )
            llm_step_result, has_reasoned = run_llm_step(
                emitter=buffered_step_emitter or emitter,
                history=truncated_message_history,
                tool_definitions=tool_defs,
                tool_choice=tool_choice,
                llm=llm,
                placement=Placement(turn_index=llm_cycle_count + reasoning_cycles),
                citation_processor=staged_citation_processor,
                state_container=staged_state_container,
                # The rich docs representation is passed in so that when yielding the answer, it has the final document set.
                final_documents=gathered_documents,
                user_identity=user_identity,
                pre_answer_processing_time=pre_answer_processing_time,
                reasoning_effort=reasoning_effort,
                max_tokens=_regulatory_llm_step_max_tokens(
                    complex_regulatory_request=complex_regulatory_request,
                    tool_choice=tool_choice,
                    projected_tool_decision_history=projected_tool_decision_history,
                    reasoning_effort=reasoning_effort,
                ),
            )
            if has_reasoned and not projected_tool_decision_history:
                reasoning_cycles += 1

            # Some providers emit tool payloads as text instead of native calls.
            llm_step_result, attempted = _try_fallback_tool_extraction(
                llm_step_result=llm_step_result,
                tool_choice=tool_choice,
                fallback_extraction_attempted=fallback_extraction_attempted,
                tool_defs=tool_defs,
                turn_index=llm_cycle_count + reasoning_cycles,
            )
            if attempted:
                fallback_extraction_attempted = True

            if projected_tool_decision_history and llm_step_result.tool_calls:
                # The search actions remain authoritative, but any narration or
                # reasoning came from an evidence-reduced view and must never be
                # streamed or persisted as part of the user's answer.
                llm_step_result = _hide_projected_tool_decision_output(
                    llm_step_result,
                    turn_index=llm_cycle_count + reasoning_cycles,
                )

            # A no-tool answer made from the metadata-only decision projection
            # means "research is complete", not "publish this text". Re-run the
            # normal synthesis once with tools disabled and the unchanged full
            # evidence history so older chunks remain available for claims and
            # citations. The model still owns the stop decision.
            if (
                projected_tool_decision_history
                and not (llm_step_result.tool_calls or [])
                and bool((llm_step_result.answer or "").strip())
            ):
                regulatory_research_complete_pending = True
                logger.info(
                    "Regulatory model stopped retrieval from projected history; "
                    "scheduling full-evidence synthesis"
                )
                continue

            if regulatory_research_complete_pending and bool(
                (llm_step_result.answer or "").strip()
            ):
                # The current non-projected, tools-disabled step is the full
                # synthesis requested above. A reviewer rejection may therefore
                # reopen the ordinary bounded AUTO decision on the next cycle.
                regulatory_research_complete_pending = False

            if buffered_step_emitter is None:
                # Preserve the established incremental behavior outside global
                # regulatory chat, where no candidate can be rejected.
                state_container.set_citation_mapping(citation_processor.citation_to_doc)

            # Run the LLM selected tools, there is some more logic here than a simple execution
            # each tool might have custom logic here
            tool_responses: list[ToolResponse] = []
            llm_history_response_by_tool_call_id: dict[str, str] = {}
            raw_tool_calls = llm_step_result.tool_calls or []
            if (
                tool_choice is ToolChoiceOptions.NONE
                and not raw_tool_calls
                and not llm_step_result.answer
                and not out_of_cycles
                and empty_final_response_retries < _MAX_EMPTY_FINAL_RESPONSE_RETRIES
                and llm_step_result.finish_reason not in _REFUSAL_FINISH_REASONS
            ):
                empty_final_response_retries += 1
                logger.warning(
                    "Final regulatory synthesis returned empty; retrying once"
                )
                continue
            search_slots = (
                min(
                    _REGULATORY_MAX_PARALLEL_SEARCH_CALLS,
                    max(
                        0,
                        effective_regulatory_search_call_budget
                        - regulatory_search_calls_attempted,
                    ),
                )
                if effective_regulatory_search_call_budget is not None
                else None
            )
            tool_calls = _constrain_regulatory_tool_calls(
                raw_tool_calls,
                search_slots=search_slots,
                attempted_query_modes=regulatory_attempted_query_modes,
            )
            requested_search_calls = sum(
                tool_call.tool_name == SearchTool.NAME for tool_call in raw_tool_calls
            )
            executed_search_calls = sum(
                tool_call.tool_name == SearchTool.NAME for tool_call in tool_calls
            )
            regulatory_tool_feedback = _format_regulatory_tool_call_batch_feedback(
                requested_search_calls=requested_search_calls,
                executed_search_calls=executed_search_calls,
            )
            if len(tool_calls) != len(raw_tool_calls):
                logger.warning(
                    "Dropped %d duplicate or over-budget regulatory tool call(s)",
                    len(raw_tool_calls) - len(tool_calls),
                )
            if raw_tool_calls and not tool_calls:
                # Let the model choose a materially different query/mode or answer
                # on the next bounded cycle. A rejected duplicate must not force
                # synthesis.
                if regulatory_tool_feedback is None:
                    regulatory_tool_feedback = (
                        "The last tool batch was skipped because it contained no "
                        "executable call. Decide whether a materially different "
                        "attempt is useful; otherwise answer now."
                    )
                continue
            if tool_calls:
                if projected_tool_decision_history:
                    llm_step_result = _hide_projected_tool_decision_output(
                        llm_step_result.model_copy(update={"tool_calls": tool_calls}),
                        turn_index=llm_cycle_count + reasoning_cycles,
                    )
                    tool_calls = llm_step_result.tool_calls or []
                else:
                    llm_step_result.tool_calls = tool_calls
                for tool_call in tool_calls:
                    if tool_call.tool_name != SearchTool.NAME:
                        continue
                    regulatory_search_calls_attempted += 1
                    query_mode = _search_query_mode_identity(tool_call)
                    if query_mode is not None:
                        regulatory_attempted_query_modes.add(query_mode)

            if buffered_step_emitter is not None and not tool_calls:
                candidate_answer_for_review = (
                    llm_step_result.raw_answer or llm_step_result.answer or ""
                )
                direct_recovery_messages: list[ChatMessageSimple] = []
                if (
                    candidate_answer_review_count < _REGULATORY_MAX_CANDIDATE_REVIEWS
                    and has_called_search_tool
                    and candidate_answer_for_review.strip()
                ):
                    candidate_answer_review_count += 1
                    candidate_evidence_chunks = _build_candidate_answer_evidence_chunks(
                        candidate_answer=candidate_answer_for_review,
                        citation_mapping=(staged_citation_processor.citation_to_doc),
                        llm_visible_results_by_citation=(
                            llm_visible_search_results_by_citation
                        ),
                    )
                    if candidate_answer_review_count == 1:
                        candidate_review = review_regulatory_candidate_answer(
                            llm,
                            user_request=regulatory_user_message,
                            earlier_user_context=regulatory_earlier_user_context,
                            candidate_answer=candidate_answer_for_review,
                            evidence_chunks=candidate_evidence_chunks,
                        )
                        candidate_review_issues = list(
                            candidate_review.advisory_claim_issues
                        )
                        candidate_review_feedback = format_candidate_answer_review(
                            candidate_review
                        )
                        review_kind = "evidence"
                    else:
                        candidate_review = review_regulatory_candidate_resolution(
                            llm,
                            candidate_answer=candidate_answer_for_review,
                            prior_issues=candidate_review_issues,
                            evidence_chunks=candidate_evidence_chunks,
                        )
                        candidate_review_feedback = format_candidate_resolution_review(
                            candidate_review
                        )
                        review_kind = "resolution"
                    if (
                        candidate_answer_review_count == 1
                        and candidate_review.needs_reconsideration
                    ):
                        recovery_issue = select_priority_recovery_issue(
                            candidate_review
                        )
                        recovery_search_tool = next(
                            (
                                tool
                                for tool in tools
                                if isinstance(tool, SearchTool)
                                and tool.user_selected_filters is not None
                                and tool.user_selected_filters.regulatory_chunks_only
                            ),
                            None,
                        )
                        if (
                            recovery_issue is not None
                            and recovery_search_tool is not None
                        ):
                            recovery_query = recovery_issue.recovery_query
                            if recovery_query is None:
                                raise ValueError(
                                    "selected recovery issue has no recovery query"
                                )
                            recovery_placement = Placement(
                                turn_index=llm_cycle_count + reasoning_cycles,
                                tab_index=0,
                            )
                            recovery_call = ToolCallKickoff(
                                tool_call_id=(
                                    "regulatory-gap-recovery-"
                                    f"{llm_cycle_count + reasoning_cycles}"
                                ),
                                tool_name=SearchTool.NAME,
                                tool_args={
                                    "queries": [recovery_query],
                                    "search_mode": "hybrid",
                                },
                                placement=recovery_placement,
                            )
                            try:
                                recovery_response = run_single_gap_recovery(
                                    search_tool=recovery_search_tool,
                                    issue=recovery_issue,
                                    starting_citation_num=(
                                        citation_processor.get_next_citation_number()
                                    ),
                                    placement=recovery_placement,
                                )
                                recovered_docs = recovery_search_docs_by_citation(
                                    recovery_response
                                )
                                merged_docs = merge_recovery_citation_mapping(
                                    citation_processor.citation_to_doc,
                                    recovered_docs,
                                )
                                citation_processor.update_citation_mapping(merged_docs)
                                staged_citation_processor.update_citation_mapping(
                                    merged_docs
                                )

                                rich_response = recovery_response.rich_response
                                if isinstance(rich_response, SearchDocsResponse):
                                    for (
                                        citation_number,
                                        document_id,
                                    ) in rich_response.citation_mapping.items():
                                        existing_document_id = citation_mapping.get(
                                            citation_number
                                        )
                                        if (
                                            existing_document_id is not None
                                            and existing_document_id != document_id
                                        ):
                                            raise ValueError(
                                                "recovery attempted to reassign "
                                                f"citation {citation_number}"
                                            )
                                        citation_mapping[citation_number] = document_id
                                    state_container.add_search_docs(
                                        rich_response.search_docs
                                    )
                                    gathered_documents = _merge_gathered_search_docs(
                                        gathered_documents,
                                        rich_response.search_docs,
                                    )
                                    state_container.add_tool_call(
                                        ToolCallInfo(
                                            parent_tool_call_id=None,
                                            turn_index=(recovery_placement.turn_index),
                                            tab_index=0,
                                            tool_name=SearchTool.NAME,
                                            tool_call_id=recovery_call.tool_call_id,
                                            tool_id=recovery_search_tool.id,
                                            reasoning_tokens=None,
                                            tool_call_arguments=(
                                                recovery_call.tool_args
                                            ),
                                            tool_call_response=(
                                                recovery_response.llm_facing_response
                                            ),
                                            search_docs=(
                                                rich_response.displayed_docs
                                                or rich_response.search_docs
                                            ),
                                            generated_images=None,
                                        )
                                    )

                                visible_recovery_results = (
                                    _extract_llm_visible_search_results(
                                        recovery_response.llm_facing_response
                                    )
                                )
                                for (
                                    citation_number,
                                    title,
                                    content,
                                ) in visible_recovery_results:
                                    llm_visible_search_results_by_citation[
                                        citation_number
                                    ] = (title, content)
                                search_evidence_ledger.append(
                                    SearchEvidenceLedgerEntry(
                                        query=recovery_query,
                                        search_mode="hybrid",
                                        result_count=len(visible_recovery_results),
                                    )
                                )

                                recovery_call_message = recovery_call.to_msg_str()
                                direct_recovery_messages.extend(
                                    [
                                        ChatMessageSimple(
                                            message="",
                                            token_count=token_counter(
                                                recovery_call_message
                                            ),
                                            message_type=MessageType.ASSISTANT,
                                            tool_calls=[
                                                ToolCallSimple(
                                                    tool_call_id=(
                                                        recovery_call.tool_call_id
                                                    ),
                                                    tool_name=SearchTool.NAME,
                                                    tool_arguments=(
                                                        recovery_call.tool_args
                                                    ),
                                                    token_count=token_counter(
                                                        recovery_call_message
                                                    ),
                                                )
                                            ],
                                        ),
                                        ChatMessageSimple(
                                            message=(
                                                recovery_response.llm_facing_response
                                            ),
                                            token_count=token_counter(
                                                recovery_response.llm_facing_response
                                            ),
                                            message_type=(
                                                MessageType.TOOL_CALL_RESPONSE
                                            ),
                                            tool_call_id=(recovery_call.tool_call_id),
                                        ),
                                    ]
                                )
                            except Exception:
                                logger.exception(
                                    "Single regulatory citation-gap search failed; "
                                    "continuing with bounded correction"
                                )
                    logger.info(
                        "Regulatory candidate review kind=%s pass=%d completed=%s "
                        "needs_reconsideration=%s issues=%d error=%s",
                        review_kind,
                        candidate_answer_review_count,
                        candidate_review.completed,
                        candidate_review.needs_reconsideration,
                        len(candidate_review.advisory_claim_issues),
                        candidate_review.review_error,
                    )
                    if candidate_review_feedback is not None:
                        if candidate_review_rejected_at_cycle is None:
                            candidate_review_rejected_at_cycle = llm_cycle_count
                        # A reviewed draft gets one direct server-selected search at
                        # most. All later correction passes run with tools disabled.
                        candidate_final_correction_pending = True
                        simple_chat_history.append(
                            ChatMessageSimple(
                                message=candidate_answer_for_review,
                                token_count=token_counter(candidate_answer_for_review),
                                message_type=MessageType.ASSISTANT,
                            )
                        )
                        simple_chat_history.extend(direct_recovery_messages)
                        continue

                commit_staged_llm_step(
                    buffered_emitter=buffered_step_emitter,
                    staged_state=staged_state_container,
                    staged_citation_processor=staged_citation_processor,
                    emitter=emitter,
                    state_container=state_container,
                    pre_answer_processing_time=(time.monotonic() - loop_start_time),
                )
                citation_processor = staged_citation_processor
                break

            if buffered_step_emitter is not None:
                citation_processor = _commit_canonical_tool_decision_step(
                    projected_tool_decision_history=projected_tool_decision_history,
                    buffered_emitter=buffered_step_emitter,
                    staged_state=staged_state_container,
                    staged_citation_processor=staged_citation_processor,
                    canonical_citation_processor=citation_processor,
                    emitter=emitter,
                    state_container=state_container,
                    pre_answer_processing_time=(time.monotonic() - loop_start_time),
                )

            if INTEGRATION_TESTS_MODE and tool_calls:
                for tool_call in tool_calls:
                    emitter.emit(
                        Packet(
                            placement=tool_call.placement,
                            obj=ToolCallDebug(
                                tool_call_id=tool_call.tool_call_id,
                                tool_name=tool_call.tool_name,
                                tool_args=tool_call.tool_args,
                            ),
                        )
                    )

            if len(tool_calls) > 1:
                emitter.emit(
                    Packet(
                        placement=Placement(
                            turn_index=tool_calls[0].placement.turn_index
                        ),
                        obj=TopLevelBranching(num_parallel_branches=len(tool_calls)),
                    )
                )

            # Quick note for why citation_mapping and citation_processors are both needed:
            # 1. Tools return lightweight string mappings, not SearchDoc objects
            # 2. The SearchDoc resolution is deliberately deferred to llm_loop.py
            # 3. The citation_processor operates on SearchDoc objects and can't provide a complete reverse URL lookup for
            # in-flight citations
            # It can be cleaned up but not super trivial or worthwhile right now
            just_ran_web_search = False
            parallel_tool_call_results = run_tool_calls(
                tool_calls=tool_calls,
                tools=final_tools,
                message_history=truncated_message_history,
                user_memory_context=user_memory_context,
                user_info=None,  # TODO, this is part of memories right now, might want to separate it out
                citation_mapping=citation_mapping,
                next_citation_num=citation_processor.get_next_citation_number(),
                max_concurrent_tools=None,
                skip_search_query_expansion=has_called_search_tool,
                chat_files=chat_files,
                url_snippet_map=extract_url_snippet_map(gathered_documents or []),
                inject_memories_in_prompt=inject_memories_in_prompt,
                search_llm_chunks_per_call_cap=regulatory_search_chunk_cap,
            )
            tool_responses = parallel_tool_call_results.tool_responses
            citation_mapping = parallel_tool_call_results.updated_citation_mapping

            # Failure case, give something reasonable to the LLM to try again
            if tool_calls and not tool_responses:
                failure_messages = create_tool_call_failure_messages(
                    tool_calls, token_counter
                )
                simple_chat_history.extend(failure_messages)
                continue

            for tool_response in tool_responses:
                # Extract tool_call from the response (set by run_tool_calls)
                if tool_response.tool_call is None:
                    raise ValueError("Tool response missing tool_call reference")

                tool_call = tool_response.tool_call
                tab_index = tool_call.placement.tab_index

                canonicalize_search_tool_response_citations(
                    tool_response,
                    citation_processor.citation_to_doc,
                )
                llm_history_response = tool_response.llm_facing_response
                repeated_result_count = 0
                if tool_call.tool_name == SearchTool.NAME:
                    (
                        llm_history_response,
                        repeated_result_count,
                    ) = _compact_repeated_search_results_for_history(
                        tool_response.llm_facing_response,
                        llm_visible_search_results_by_citation,
                    )
                llm_history_response_by_tool_call_id[tool_call.tool_call_id] = (
                    llm_history_response
                )

                # Track if search tool was called (for skipping query expansion on subsequent calls)
                if tool_call.tool_name == SearchTool.NAME:
                    has_called_search_tool = True

                # Track if code interpreter generated files with download links
                if (
                    tool_call.tool_name == PythonTool.NAME
                    and not code_interpreter_file_generated
                ):
                    try:
                        parsed = json.loads(tool_response.llm_facing_response)
                        if parsed.get("generated_files"):
                            code_interpreter_file_generated = True
                    except (json.JSONDecodeError, AttributeError):
                        pass

                tools_by_name = {tool.name: tool for tool in final_tools}

                # Add the results to the chat history. Even though tools may run in parallel,
                # LLM APIs require linear history, so results are added sequentially.
                # Get the tool object to retrieve tool_id
                tool = tools_by_name.get(tool_call.tool_name)
                if not tool:
                    raise ValueError(
                        f"Tool '{tool_call.tool_name}' not found in tools list"
                    )

                # Extract search_docs if this is a search tool response
                search_docs = None
                displayed_docs = None
                if isinstance(tool_response.rich_response, SearchDocsResponse):
                    search_docs = tool_response.rich_response.search_docs
                    displayed_docs = tool_response.rich_response.displayed_docs

                    if tool_call.tool_name == SearchTool.NAME:
                        raw_queries = tool_call.tool_args.get("queries")
                        query = (
                            raw_queries[0]
                            if isinstance(raw_queries, list)
                            and raw_queries
                            and isinstance(raw_queries[0], str)
                            else "unspecified query"
                        )
                        raw_search_mode = tool_call.tool_args.get("search_mode")
                        search_mode = (
                            raw_search_mode
                            if isinstance(raw_search_mode, str)
                            else "unspecified"
                        )
                        llm_visible_results = _extract_llm_visible_search_results(
                            llm_history_response
                        )
                        for (
                            citation_number,
                            title,
                            content,
                        ) in llm_visible_results:
                            llm_visible_search_results_by_citation[citation_number] = (
                                title,
                                content,
                            )
                        search_evidence_ledger.append(
                            SearchEvidenceLedgerEntry(
                                query=query,
                                search_mode=search_mode,
                                result_count=len(llm_visible_results),
                                repeated_result_count=repeated_result_count,
                            )
                        )
                    # Add ALL search docs to state container for DB persistence
                    if search_docs:
                        state_container.add_search_docs(search_docs)

                    gathered_documents = _merge_gathered_search_docs(
                        gathered_documents,
                        search_docs,
                    )

                    # This is used for the Open URL reminder in the next cycle
                    # only do this if the web search tool yielded results
                    if search_docs and tool_call.tool_name == WebSearchTool.NAME:
                        just_ran_web_search = True

                    # Stage any raw source files attached to these hits into
                    # the session's chat_files so the next Python tool call
                    # sees them already uploaded under their display names.
                    if search_docs:
                        staged = build_python_chat_files_from_search_docs(
                            search_docs=search_docs,
                        )
                        if staged:
                            existing_filenames = {cf.filename for cf in chat_files}
                            chat_files.extend(
                                cf
                                for cf in staged
                                if cf.filename not in existing_filenames
                            )

                # Extract generated_images if this is an image generation tool response
                generated_images = None
                if isinstance(
                    tool_response.rich_response, FinalImageGenerationResponse
                ):
                    generated_images = tool_response.rich_response.generated_images

                # Extract generated_files if this is a code interpreter response
                generated_files = None
                if isinstance(tool_response.rich_response, PythonToolRichResponse):
                    generated_files = (
                        tool_response.rich_response.generated_files or None
                    )

                # Persist memory if this is a memory tool response
                memory_snapshot: MemoryToolResponseSnapshot | None = None
                if isinstance(tool_response.rich_response, MemoryToolResponse):
                    persisted_memory_id: int | None = None
                    if user_memory_context and user_memory_context.user_id:
                        if tool_response.rich_response.index_to_replace is not None:
                            persisted_memory_id = update_memory_at_index(
                                user_id=user_memory_context.user_id,
                                index=tool_response.rich_response.index_to_replace,
                                new_text=tool_response.rich_response.memory_text,
                            )
                        else:
                            persisted_memory_id = add_memory(
                                user_id=user_memory_context.user_id,
                                memory_text=tool_response.rich_response.memory_text,
                            )
                    operation: Literal["add", "update"] = (
                        "update"
                        if tool_response.rich_response.index_to_replace is not None
                        else "add"
                    )
                    memory_snapshot = MemoryToolResponseSnapshot(
                        memory_text=tool_response.rich_response.memory_text,
                        operation=operation,
                        memory_id=persisted_memory_id,
                        index=tool_response.rich_response.index_to_replace,
                    )

                if memory_snapshot:
                    saved_response = json.dumps(memory_snapshot.model_dump())
                elif isinstance(tool_response.rich_response, CustomToolCallSummary):
                    saved_response = json.dumps(
                        tool_response.rich_response.model_dump()
                    )
                elif isinstance(tool_response.rich_response, str):
                    saved_response = tool_response.rich_response
                else:
                    saved_response = tool_response.llm_facing_response

                tool_call_info = ToolCallInfo(
                    parent_tool_call_id=None,  # Top-level tool calls are attached to the chat message
                    turn_index=llm_cycle_count + reasoning_cycles,
                    tab_index=tab_index,
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    tool_id=tool.id,
                    reasoning_tokens=llm_step_result.reasoning,  # All tool calls from this loop share the same reasoning
                    tool_call_arguments=tool_call.tool_args,
                    tool_call_response=saved_response,
                    search_docs=displayed_docs or search_docs,
                    generated_images=generated_images,
                    generated_files=generated_files,
                )
                # Add to state container for partial save support
                state_container.add_tool_call(tool_call_info)

                # Update citation processor if this was a search tool
                update_citation_processor_from_tool_response(
                    tool_response, citation_processor
                )

            # After processing all tool responses for this turn, add messages to history
            # using OpenAI parallel tool calling format:
            # 1. ONE ASSISTANT message with tool_calls array
            # 2. N TOOL_CALL_RESPONSE messages (one per tool call)
            if tool_responses:
                # Filter to only responses with valid tool_call references
                valid_tool_responses = [
                    tr for tr in tool_responses if tr.tool_call is not None
                ]

                # Build ToolCallSimple list for all tool calls in this turn
                tool_calls_simple: list[ToolCallSimple] = []
                for tool_response in valid_tool_responses:
                    tc = tool_response.tool_call
                    assert (
                        tc is not None
                    )  # Already filtered above, this is just for typing purposes

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

                # Create ONE ASSISTANT message with all tool calls for this turn
                total_tool_call_tokens = sum(tc.token_count for tc in tool_calls_simple)
                assistant_with_tools = ChatMessageSimple(
                    message="",  # No text content when making tool calls
                    token_count=total_tool_call_tokens,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=tool_calls_simple,
                    image_files=None,
                )
                simple_chat_history.append(assistant_with_tools)

                # Add TOOL_CALL_RESPONSE messages for each tool call
                for tool_response in valid_tool_responses:
                    tc = tool_response.tool_call
                    assert tc is not None  # Already filtered above

                    tool_response_message = llm_history_response_by_tool_call_id.get(
                        tc.tool_call_id, tool_response.llm_facing_response
                    )
                    tool_response_token_count = token_counter(tool_response_message)

                    tool_response_msg = ChatMessageSimple(
                        message=tool_response_message,
                        token_count=tool_response_token_count,
                        message_type=MessageType.TOOL_CALL_RESPONSE,
                        tool_call_id=tc.tool_call_id,
                        image_files=None,
                    )
                    simple_chat_history.append(tool_response_msg)

            # If no tool calls, then it must have answered, wrap up
            if not llm_step_result.tool_calls or len(llm_step_result.tool_calls) == 0:
                break

            # Certain tools do not allow further actions, force the LLM wrap up on the next cycle
            if any(
                tool.tool_name in STOPPING_TOOLS_NAMES
                for tool in llm_step_result.tool_calls
            ):
                ran_image_gen = True

            if llm_step_result.tool_calls and any(
                tool.tool_name in CITEABLE_TOOLS_NAMES
                for tool in llm_step_result.tool_calls
            ):
                # As long as 1 tool with citeable documents is called at any point, we ask the LLM to try to cite
                should_cite_documents = True

        if not llm_step_result.answer and not llm_step_result.tool_calls:
            raise _build_empty_llm_response_error(
                llm=llm,
                llm_step_result=llm_step_result,
                tool_choice=tool_choice,
            )

        if not llm_step_result.answer:
            raise RuntimeError(
                "The LLM did not return a final answer after tool execution. "
                "Typically this indicates invalid tool-call output, a model/provider mismatch, "
                "or serving API misconfiguration."
            )

        emitter.emit(
            Packet(
                placement=Placement(
                    turn_index=llm_cycle_count  # ty: ignore[possibly-unresolved-reference]
                    + reasoning_cycles
                ),
                obj=OverallStop(type="stop"),
            )
        )
