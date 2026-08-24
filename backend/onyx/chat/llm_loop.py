import json
import re
import time
from collections.abc import Callable, Sequence
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
    synchronize_lightweight_citation_mapping,
    update_citation_processor_from_tool_response,
)
from onyx.chat.emitter import BufferedEmitter, Emitter
from onyx.chat.empty_response import (
    REFUSAL_FINISH_REASONS,
    build_empty_llm_response_error,
)
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
from onyx.configs.app_configs import (
    INTEGRATION_TESTS_MODE,
    QUERY_EMBEDDING_CACHE_ENABLED,
)
from onyx.configs.chat_configs import MAX_LLM_CYCLES
from onyx.configs.constants import DocumentSource, MessageType
from onyx.configs.model_configs import GEN_AI_INPUT_TOKEN_SAFETY_MARGIN
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.context.search.utils import prime_query_embedding_cache
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.memory import UserMemoryContext, add_memory, update_memory_at_index
from onyx.db.models import Persona
from onyx.llm.interfaces import LLM, LLMUserIdentity, ToolChoiceOptions
from onyx.llm.models import ReasoningEffort
from onyx.prompts.chat_prompts import IMAGE_GEN_REMINDER, OPEN_URL_REMINDER
from onyx.prompts.prompt_utils import substitute_user_placeholders
from onyx.regulatory.candidate_answer_review import (
    MAX_REGULATORY_CLAIM_ISSUES,
    CandidateAnswerClaimIssue,
    CandidateAnswerEvidenceChunk,
    CandidateAnswerReviewResult,
    build_candidate_answer_evidence_chunk,
    build_regulatory_review_llm,
    build_regulatory_review_user_context,
    format_candidate_answer_review,
    format_candidate_resolution_review,
    format_candidate_review_regression_guard,
    review_regulatory_candidate_answer_with_fallback,
    review_regulatory_candidate_matrix_closure_with_fallback,
    review_regulatory_candidate_resolution_with_fallback,
)
from onyx.regulatory.coverage_plan import (
    RegulatoryCoverageItem,
    RegulatoryCoveragePlan,
    build_regulatory_coverage_plan,
    format_regulatory_coverage_plan,
)
from onyx.regulatory.evidence_matrix import (
    EvidenceCoverageStatus,
    RegulatoryEvidenceMatrix,
    RegulatoryEvidenceMatrixRow,
    RegulatoryNavigationLead,
    build_regulatory_evidence_matrix,
    evidence_matrix_recovery_queries,
    format_regulatory_evidence_matrix,
)
from onyx.regulatory.gap_recovery import (
    merge_recovery_citation_mapping,
    recovery_search_docs_by_citation,
    run_batched_gap_recovery,
    select_priority_recovery_issues,
)
from onyx.regulatory.navigation_recovery import (
    select_regulatory_navigation_recovery_leads,
)
from onyx.regulatory.workflow_profile import (
    STANDARD_REGULATORY_WORKFLOW,
    get_regulatory_workflow_profile,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    OverallStop,
    Packet,
    ToolCallDebug,
    TopLevelBranching,
)
from onyx.tools.built_in_tools import CITEABLE_TOOLS_NAMES, STOPPING_TOOLS_NAMES
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
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
from onyx.tools.tool_implementations.search.search_tool import (
    SearchTool,
    _prepare_search_query,
)
from onyx.tools.tool_implementations.web_search.utils import extract_url_snippet_map
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tools.tool_runner import run_tool_calls
from onyx.tools.utils import compute_all_tool_tokens
from onyx.tracing.framework.create import ChatTraceMetadata, trace
from onyx.utils.logger import setup_logger

logger = setup_logger()

_REGULATORY_MAX_PARALLEL_SEARCH_CALLS = 32
_REGULATORY_MAX_CONCURRENT_SEARCH_TOOLS = 8
_REGULATORY_SEARCH_LLM_CHUNKS_PER_CALL = 10
_REGULATORY_BOOTSTRAP_COVERAGE_CYCLES = 1
_REGULATORY_AUTONOMOUS_RESEARCH_CYCLES = 1
_MAX_EMPTY_FINAL_RESPONSE_RETRIES = 1
_FAST_REGULATORY_ABSENCE_RECOVERY_REMINDER = (
    "# Source-gap recovery correction\n"
    "The previous draft claimed that requested legal text was unavailable. "
    "That draft was withheld and is not authoritative. Re-audit all exact "
    "evidence below, including newly retrieved evidence and sibling provisions, "
    "before producing a complete replacement answer. A retrieval miss never "
    "proves that the database lacks the source. If the exact controlling text "
    "still cannot be established after these attempts, say only that it was not "
    "reached in the searches performed (Turkish: 'uygulanan aramalarda "
    "ulaşılamadı') and identify the unresolved proposition. Do not say or imply "
    "that the database/index has no data, and do not emit blank placeholders."
)

_REGULATORY_SOURCE_GAP_PATTERNS = (
    re.compile(r"\bkaynak\s+boşlu", re.IGNORECASE),
    re.compile(
        r"\b(?:arama\s+sonuç|indeks|veri\s*taban|kaynak|kayıt|tam\s+metin|"
        r"metin|hük|madde)\w*\b.{0,140}\b(?:erişemed|erişilemed|bulunamad|"
        r"ulaşılamad|getirilemed|tespit\s+edilemed|teyit\s+edilemed|"
        r"teyit\s+edemiyorum|yer\s+almıyor|mevcut\s+değil|yok)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:erişemed|erişilemed|bulunamad|ulaşılamad|getirilemed)\w*\b"
        r".{0,100}\b(?:kaynak|kayıt|metin|hüküm|madde|indeks|veri\s*taban)\w*\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:source\s+gap|could\s+not\s+(?:find|retrieve|verify)|"
        r"not\s+(?:found|retrieved)\s+in\s+(?:the\s+)?(?:database|index))\b",
        re.IGNORECASE,
    ),
)

_UNVERIFIED_DATABASE_ABSENCE_PATTERN = re.compile(
    r"\b(?:veri\s*taban|indeks)\w*\b[^.!?\n:]{0,160}"
    r"\b(?:(?:kayıt|veri|metin)\s+)?(?:yok|bulunamadı|yer\s+almıyor|"
    r"mevcut\s+değil)\b",
    re.IGNORECASE,
)
_BLANK_PLACEHOLDER_PATTERN = re.compile(r"`?\[?\s*_{3,}\s*\]?`?")
_SOURCE_LOCATION_PATTERN = re.compile(
    r"\b(?:(?:arama\s+sonuçları|kaynak|indeks|veri\s*tabanı)"
    r"(?:nda|nde|ta|te|da|de)|(?:in\s+the\s+)?(?:database|index))\b",
    re.IGNORECASE,
)


def _draft_claims_regulatory_source_gap(candidate_answer: str) -> bool:
    """Detect an explicit claim that requested indexed evidence was unavailable."""

    normalized_answer = " ".join(candidate_answer.split())
    if not normalized_answer:
        return False
    return any(
        pattern.search(normalized_answer) is not None
        for pattern in _REGULATORY_SOURCE_GAP_PATTERNS
    )


def _qualify_fast_regulatory_source_gap_answer(candidate_answer: str) -> str:
    """Remove claims that confuse a retrieval miss with database absence."""

    qualified = _UNVERIFIED_DATABASE_ABSENCE_PATTERN.sub(
        "Uygulanan aramalarda ilgili metne ulaşılamadı",
        candidate_answer,
    )
    qualified = re.sub(
        r"\bkaynak\s+boşluğu\b",
        "Arama kapsamı notu",
        qualified,
        flags=re.IGNORECASE,
    )
    qualified = re.sub(
        r"\b(?:tam\s+metni\s+)?getirilemedi\b",
        "uygulanan aramalarda tam metne ulaşılamadı",
        qualified,
        flags=re.IGNORECASE,
    )
    qualified = _BLANK_PLACEHOLDER_PATTERN.sub(
        " uygulanan aramalarda ulaşılamadı",
        qualified,
    )
    sentence_parts = re.split(r"(?<=[.!?])(?=\s|$)|\n", qualified)
    for index, sentence in enumerate(sentence_parts):
        if not _draft_claims_regulatory_source_gap(sentence):
            continue
        english_gap = re.search(
            r"\b(?:source\s+gap|could\s+not|not\s+(?:found|retrieved))\b",
            sentence,
            re.IGNORECASE,
        )
        search_qualifier = (
            "in the searches performed" if english_gap else "uygulanan aramalarda"
        )
        sentence = _SOURCE_LOCATION_PATTERN.sub(search_qualifier, sentence)
        if search_qualifier not in sentence.casefold():
            leading_space = sentence[: len(sentence) - len(sentence.lstrip())]
            sentence = (
                leading_space + search_qualifier.capitalize() + " " + sentence.lstrip()
            )
        sentence_parts[index] = sentence
    return "".join(sentence_parts)


def _prime_fast_regulatory_query_embeddings(
    tool_calls: Sequence[ToolCallKickoff],
) -> None:
    """Batch/cache validated V2 plan queries without changing their semantics."""

    if not QUERY_EMBEDDING_CACHE_ENABLED:
        return
    queries: list[str] = []
    seen_queries: set[str] = set()
    for call in tool_calls:
        if (
            not call.tool_call_id.startswith("regulatory-coverage-0-")
            or call.tool_args.get("search_mode") != "hybrid"
        ):
            continue
        raw_queries = call.tool_args.get("queries")
        if (
            not isinstance(raw_queries, list)
            or len(raw_queries) != 1
            or not isinstance(raw_queries[0], str)
            or not raw_queries[0].strip()
        ):
            continue
        query = _prepare_search_query(raw_queries[0], "hybrid")
        if query and query not in seen_queries:
            seen_queries.add(query)
            queries.append(query)
    if not queries:
        return
    try:
        with get_session_with_current_tenant() as db_session:
            prime_query_embedding_cache(
                queries,
                db_session=db_session,
                max_workers=_REGULATORY_MAX_CONCURRENT_SEARCH_TOOLS,
            )
    except Exception:
        # SearchTool retains its established per-query embedding and lexical
        # fail-soft paths; cache priming must never block retrieval.
        logger.exception(
            "Fast regulatory embedding batch prime failed; continuing per query"
        )


_REGULATORY_POST_REVIEW_MAIN_CYCLES = 3
# One full independent audit followed by one issue-resolution audit. Further
# audits of the same fixed evidence create a rewrite loop rather than new proof.
_REGULATORY_MAX_CANDIDATE_REVIEWS = 2
_REGULATORY_MATRIX_EVIDENCE_PER_RESEARCH_TARGET = 3
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
_REGULATORY_REASONING_TOKEN_RESERVE = {
    ReasoningEffort.OFF: 0,
    ReasoningEffort.LOW: 4096,
    ReasoningEffort.AUTO: 8192,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 12288,
    ReasoningEffort.XHIGH: 12288,
}


@dataclass(frozen=True)
class SearchEvidenceLedgerEntry:
    """Compact receipt for one model-directed internal-search attempt."""

    query: str
    search_mode: str
    result_count: int
    repeated_result_count: int = 0


def _build_regulatory_coverage_tool_calls(
    plan: RegulatoryCoveragePlan | None,
    *,
    turn_index: int,
    max_calls: int | None = _REGULATORY_MAX_PARALLEL_SEARCH_CALLS,
    include_auxiliary_searches: bool = True,
    include_lexical_fallbacks: bool = True,
) -> list[ToolCallKickoff]:
    """Allocate retrieval fairly from a request-derived, source-neutral plan."""

    if plan is None:
        return []

    def has_capacity(current_count: int) -> bool:
        return max_calls is None or current_count < max_calls

    def bounded(value: str, max_chars: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= max_chars:
            return normalized
        truncated = normalized[: max_chars + 1]
        last_space = truncated.rfind(" ")
        if last_space >= max_chars // 2:
            truncated = truncated[:last_space]
        return truncated[:max_chars].rstrip(" ,;:")

    def query_for(
        item: RegulatoryCoverageItem,
        *,
        evidence_dimension: str | None = None,
    ) -> str:
        """Build the smallest source-shaped query for one atomic evidence row.

        The full coverage item and factual branches remain available as tool-call
        provenance. Repeating them in the retrieval query dilutes the independently
        searchable target and makes broad scenario terms dominate exact provisions.
        """

        parts: list[str] = []
        if evidence_dimension:
            parts.append(bounded(evidence_dimension, 280))
        else:
            parts.append(bounded(item.completion_test, 280))
        return "\n".join(parts)[:REGULATORY_MAX_SEARCH_QUERY_CHARS].rstrip()

    atomic_rows: list[tuple[str, str, str, str, list[str]]] = []
    row_identities: set[str] = set()

    def atomic_dimensions(
        item: RegulatoryCoverageItem,
    ) -> list[tuple[str, str]]:
        """Pair each independent evidence row with one bounded lexical query.

        Structured model output can occasionally violate the parallel-list
        contract. In that case the source-neutral evidence dimensions are the
        authoritative omission ledger; ignoring unmatched query alternatives is
        safer than silently dropping independent legal outcomes.
        """

        if item.evidence_dimensions:
            retrieval_queries = (
                item.retrieval_queries
                if len(item.retrieval_queries) == len(item.evidence_dimensions)
                else item.evidence_dimensions
            )
            return list(zip(retrieval_queries, item.evidence_dimensions, strict=True))
        if item.retrieval_queries:
            return [(query, query) for query in item.retrieval_queries]
        return [(item.completion_test, item.completion_test)]

    # Allocate the first attempt for every independent row before spending
    # budget on a complementary lexical interpretation of an earlier row.
    dimension_index = 0
    while has_capacity(len(atomic_rows)):
        added = False
        for item in plan.coverage_items:
            dimensions = atomic_dimensions(item)
            if dimension_index >= len(dimensions):
                continue
            retrieval_query, evidence_dimension = dimensions[dimension_index]
            hybrid_query = query_for(item, evidence_dimension=evidence_dimension)
            keyword_query = query_for(item, evidence_dimension=retrieval_query)
            identity = " ".join(hybrid_query.casefold().split())
            if hybrid_query.strip() and identity not in row_identities:
                row_identities.add(identity)
                atomic_rows.append(
                    (
                        hybrid_query,
                        keyword_query,
                        item.research_question,
                        evidence_dimension,
                        list(item.source_anchors),
                    )
                )
                added = True
            if not has_capacity(len(atomic_rows)):
                break
        if not added:
            break
        dimension_index += 1

    obligation_rows: list[tuple[str, str, str, list[str]]] = []
    obligation_identities: set[str] = set()
    obligation_index = 0
    while has_capacity(len(obligation_rows)):
        added = False
        for item in plan.coverage_items:
            if obligation_index >= len(item.request_anchor_groups):
                continue
            anchor_group = item.request_anchor_groups[obligation_index]
            query = bounded("; ".join(anchor_group), 220)
            identity = " ".join(query.casefold().split())
            if query and identity not in obligation_identities:
                obligation_identities.add(identity)
                obligation_rows.append(
                    (
                        query,
                        item.research_question,
                        "Request-grounded obligation: " + bounded(query, 220),
                        list(item.source_anchors),
                    )
                )
                added = True
            if not has_capacity(len(obligation_rows)):
                break
        if not added:
            break
        obligation_index += 1

    branch_rows: list[tuple[str, str, str, list[str]]] = []
    branch_identities: set[str] = set()
    branch_index = 0
    while has_capacity(len(branch_rows)):
        added = False
        for item in plan.coverage_items:
            if branch_index >= len(item.material_factual_branches):
                continue
            branch = item.material_factual_branches[branch_index]
            # A factual branch may be an upstream prerequisite governed by a
            # different instrument than the downstream coverage item. Keep the
            # supplied source anchors as provenance, but do not force an
            # unproven source relationship into the lexical query.
            query = bounded(branch, 220)
            identity = " ".join(query.casefold().split())
            if query and identity not in branch_identities:
                branch_identities.add(identity)
                branch_rows.append(
                    (
                        query,
                        item.research_question,
                        "Material factual branch: " + bounded(branch, 220),
                        list(item.source_anchors),
                    )
                )
                added = True
            if not has_capacity(len(branch_rows)):
                break
        if not added:
            break
        branch_index += 1

    source_anchors: list[str] = []
    source_anchor_identities: set[str] = set()
    for item in plan.coverage_items:
        for source_anchor in item.source_anchors:
            bounded_anchor = bounded(source_anchor, 120)
            identity = " ".join(bounded_anchor.casefold().split())
            if not bounded_anchor or identity in source_anchor_identities:
                continue
            source_anchor_identities.add(identity)
            source_anchors.append(bounded_anchor)

    context_atom_rows: list[tuple[str, str, str, list[str]]] = []
    for atom in plan.request_context_atoms:
        bounded_atom = bounded(atom, 220)
        if not bounded_atom:
            continue
        if source_anchors:
            # The source name routes a scenario fact toward the right document,
            # but hard-scoping it would also assert that the fact belongs to the
            # named subsection and suppress semantically related provisions.
            context_atom_rows.extend(
                (
                    bounded(f"{source_anchor} {bounded_atom}", 280),
                    "Request-supplied scenario context",
                    "Request context atom: " + bounded_atom,
                    [],
                )
                for source_anchor in source_anchors
            )
        else:
            context_atom_rows.append(
                (
                    bounded_atom,
                    "Request-supplied scenario context",
                    "Request context atom: " + bounded_atom,
                    [],
                )
            )
    calls: list[tuple[str, str, str, str, list[str]]] = []
    identities: set[tuple[str, str]] = set()

    def append(
        query: str,
        mode: str,
        *,
        coverage_item: str,
        evidence_target: str,
        source_anchors: list[str],
    ) -> None:
        if not has_capacity(len(calls)):
            return
        identity = (" ".join(query.casefold().split()), mode)
        if not query.strip() or identity in identities:
            return
        identities.add(identity)
        calls.append(
            (query, mode, coverage_item, evidence_target, list(source_anchors))
        )

    # Allocate one hybrid attempt fairly across all independent evidence rows
    # before spending capacity on exact request phrases or lexical fallbacks.
    # Hybrid retrieval combines the existing BM25 index with the compatible
    # query embedding configured for this physical index.
    for (
        hybrid_query,
        _keyword_query,
        coverage_item,
        evidence_target,
        source_anchors,
    ) in atomic_rows:
        append(
            hybrid_query,
            "hybrid",
            coverage_item=coverage_item,
            evidence_target=evidence_target,
            source_anchors=source_anchors,
        )
    if include_auxiliary_searches:
        for rows in (branch_rows, context_atom_rows, obligation_rows):
            for query, coverage_item, evidence_target, source_anchors in rows:
                append(
                    query,
                    "hybrid",
                    coverage_item=coverage_item,
                    evidence_target=evidence_target,
                    source_anchors=source_anchors,
                )
    # Allocate remaining recall-first keyword fallbacks fairly across planner rows.
    if include_lexical_fallbacks:
        for (
            _hybrid_query,
            keyword_query,
            coverage_item,
            evidence_target,
            source_anchors,
        ) in atomic_rows:
            append(
                keyword_query,
                "keyword",
                coverage_item=coverage_item,
                evidence_target=evidence_target,
                source_anchors=source_anchors,
            )
    return [
        ToolCallKickoff(
            tool_call_id=f"regulatory-coverage-{turn_index}-{query_index}",
            tool_name=SearchTool.NAME,
            tool_args={
                "queries": [query],
                "search_mode": mode,
                "coverage_item": coverage_item,
                "evidence_target": evidence_target,
                "source_anchors": source_anchors,
            },
            placement=Placement(turn_index=turn_index, tab_index=query_index),
        )
        for query_index, (
            query,
            mode,
            coverage_item,
            evidence_target,
            source_anchors,
        ) in enumerate(calls)
    ]


def _build_regulatory_navigation_recovery_tool_calls(
    leads: Sequence[RegulatoryNavigationLead],
    *,
    turn_index: int,
) -> list[ToolCallKickoff]:
    """Retrieve exact text for a bounded set of model-selected outline leads."""

    calls: list[ToolCallKickoff] = []
    seen_queries: set[str] = set()
    for lead in leads:
        query = " ".join(f"{lead.document_title} {lead.heading_label}".split())[
            :REGULATORY_MAX_SEARCH_QUERY_CHARS
        ].rstrip()
        identity = query.casefold()
        if not query or identity in seen_queries:
            continue
        seen_queries.add(identity)
        calls.append(
            ToolCallKickoff(
                tool_call_id=(
                    f"regulatory-navigation-recovery-{turn_index}-{len(calls)}"
                ),
                tool_name=SearchTool.NAME,
                tool_args={
                    "queries": [query],
                    "search_mode": "hybrid",
                    "coverage_item": "Request-grounded source-outline recovery",
                    "evidence_target": (
                        "Source-outline exact-text recovery: " + lead.heading_label
                    ),
                    "source_anchors": [lead.document_title],
                },
                placement=Placement(turn_index=turn_index, tab_index=len(calls)),
            )
        )
    return calls


def _unattempted_regulatory_navigation_leads(
    leads: Sequence[RegulatoryNavigationLead],
    attempted_identities: set[tuple[str, str]],
) -> list[RegulatoryNavigationLead]:
    """Keep unattempted source-outline entries in stable discovery order."""

    return [
        lead
        for lead in leads
        if (lead.document_title, lead.article_key) not in attempted_identities
    ]


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
        return None
    return (
        _REGULATORY_TOOL_DECISION_VISIBLE_TOKEN_ALLOWANCE
        + (_REGULATORY_REASONING_TOKEN_RESERVE[reasoning_effort])
    )


def _should_schedule_regulatory_candidate_correction(
    review_count: int,
    review: CandidateAnswerReviewResult,
    *,
    max_reviews: int = _REGULATORY_MAX_CANDIDATE_REVIEWS,
) -> bool:
    """Never replace the last reviewed draft with an unreviewed full rewrite."""

    return review.needs_reconsideration and review_count < max_reviews


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
    research_targets_by_citation: dict[int, list[str]] | None = None,
    coverage_items_by_citation: dict[int, list[str]] | None = None,
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
                research_target="\n".join(
                    (research_targets_by_citation or {}).get(citation_number, [])
                ),
                coverage_item="\n".join(
                    (coverage_items_by_citation or {}).get(citation_number, [])
                ),
            )
        )

    return evidence_chunks


def _select_regulatory_closure_evidence(
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
    *,
    candidate_answer: str,
    evidence_matrix: RegulatoryEvidenceMatrix | None,
    priority_citation_numbers: set[int] | None = None,
) -> list[CandidateAnswerEvidenceChunk]:
    """Keep all unique evidence, ordering closure-critical chunks first."""

    protected_numbers = set(extract_citation_order_from_text(candidate_answer))
    protected_numbers.update(priority_citation_numbers or ())
    if evidence_matrix is not None:
        protected_numbers.update(
            document_number
            for row in evidence_matrix.rows
            for document_number in row.document_numbers
        )

    selected: list[CandidateAnswerEvidenceChunk] = []
    selected_numbers: set[int] = set()

    def append(chunk: CandidateAnswerEvidenceChunk) -> None:
        if chunk.retrieval_number is None or chunk.retrieval_number in selected_numbers:
            return
        selected.append(chunk)
        selected_numbers.add(chunk.retrieval_number)

    for chunk in evidence_chunks:
        if chunk.retrieval_number in protected_numbers:
            append(chunk)

    for chunk in evidence_chunks:
        append(chunk)

    return selected


def _select_regulatory_matrix_input_evidence(
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
) -> list[CandidateAnswerEvidenceChunk]:
    """Keep the strongest bounded evidence for each retrieval probe."""

    def lexical_score(chunk: CandidateAnswerEvidenceChunk, target: str) -> int:
        target_terms = {
            term
            for term in re.findall(r"[^\W_]+", target.casefold())
            if len(term) >= 4 or term.isdigit()
        }
        evidence_terms = set(
            re.findall(
                r"[^\W_]+",
                f"{chunk.heading} {chunk.content}".casefold(),
            )
        )
        score = 0
        for target_term in target_terms:
            if target_term in evidence_terms:
                score += 3
            elif not target_term.isdigit() and any(
                len(evidence_term) >= 4 and evidence_term[:4] == target_term[:4]
                for evidence_term in evidence_terms
            ):
                score += 2
        return score

    chunks_by_target: dict[str, list[CandidateAnswerEvidenceChunk]] = {}
    for chunk in evidence_chunks:
        targets = list(
            dict.fromkeys(
                target.strip()
                for target in chunk.research_target.splitlines()
                if target.strip()
            )
        ) or [chunk.coverage_item.strip()]
        for target in targets:
            if target:
                chunks_by_target.setdefault(target, []).append(chunk)

    selected: list[CandidateAnswerEvidenceChunk] = []
    selected_numbers: set[int] = set()
    for target, target_chunks in chunks_by_target.items():
        ranked = sorted(
            target_chunks,
            key=lambda chunk: (
                -lexical_score(chunk, target),
                chunk.retrieval_number or 0,
                chunk.chunk_identifier,
            ),
        )
        target_selection = list(
            dict.fromkeys(
                [
                    ranked[0],
                    target_chunks[0],
                    target_chunks[-1],
                    *ranked,
                ]
            )
        )[:_REGULATORY_MATRIX_EVIDENCE_PER_RESEARCH_TARGET]
        for chunk in target_selection:
            if (
                chunk.retrieval_number is None
                or chunk.retrieval_number in selected_numbers
            ):
                continue
            selected.append(chunk)
            selected_numbers.add(chunk.retrieval_number)
    return selected


def _select_regulatory_matrix_review_evidence(
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
    matrix: RegulatoryEvidenceMatrix,
) -> list[CandidateAnswerEvidenceChunk]:
    """Keep exact chunks named by matrix rows for a focused closure audit."""

    matrix_document_numbers = {
        document_number
        for row in matrix.rows
        for document_number in row.document_numbers
    }
    return [
        chunk
        for chunk in evidence_chunks
        if chunk.retrieval_number in matrix_document_numbers
    ]


def _build_regulatory_matrix_citation_issues(
    candidate_answer: str,
    matrix: RegulatoryEvidenceMatrix,
) -> list[CandidateAnswerClaimIssue]:
    """Reject supported matrix rows whose exact evidence is absent from citations."""

    cited_document_numbers = set(extract_citation_order_from_text(candidate_answer))
    uncited_rows = []
    for row_index, row in enumerate(matrix.rows):
        if row.status is not EvidenceCoverageStatus.SUPPORTED:
            continue
        if cited_document_numbers.intersection(row.document_numbers):
            continue
        uncited_rows.append((row_index, row))

    # A model-generated matrix can contain several independently supported
    # propositions for one request target. Select missing-citation issues fairly
    # across target IDs so an early verbose target cannot hide every later one.
    target_issue_counts: dict[str, int] = {}
    ordered_rows: list[RegulatoryEvidenceMatrixRow] = []
    pending_rows = list(uncited_rows)
    while pending_rows:
        selected_index, (_, selected_row) = min(
            enumerate(pending_rows),
            key=lambda item: (
                min(
                    (
                        target_issue_counts.get(target_id, 0)
                        for target_id in item[1][1].target_ids
                    ),
                    default=target_issue_counts.get("", 0),
                ),
                item[1][0],
            ),
        )
        pending_rows.pop(selected_index)
        ordered_rows.append(selected_row)
        for target_id in selected_row.target_ids or [""]:
            target_issue_counts[target_id] = target_issue_counts.get(target_id, 0) + 1

    issues: list[CandidateAnswerClaimIssue] = []
    for row in ordered_rows:
        issues.append(
            CandidateAnswerClaimIssue(
                claim_reference=row.target[:280].rstrip(),
                advisory_feedback=(
                    "This supported evidence-matrix row has no inline citation to "
                    "any of its exact source chunks. State the material supported "
                    "result and cite the directly entailing chunk; do not replace "
                    "available evidence with a source-gap disclaimer."
                ),
                related_citation_numbers=row.document_numbers,
            )
        )
        if len(issues) == MAX_REGULATORY_CLAIM_ISSUES:
            break
    return issues


def _merge_candidate_review_issues(
    *issue_groups: Sequence[CandidateAnswerClaimIssue],
) -> list[CandidateAnswerClaimIssue]:
    """Combine independent gates while keeping one bounded issue per defect."""

    merged: list[CandidateAnswerClaimIssue] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for issue in (issue for group in issue_groups for issue in group):
        identity = (
            " ".join(issue.claim_reference.casefold().split()),
            tuple(issue.related_citation_numbers),
        )
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(issue)
        if len(merged) == MAX_REGULATORY_CLAIM_ISSUES:
            break
    return merged


def _merge_candidate_review_verdicts(
    *reviews: CandidateAnswerReviewResult,
) -> CandidateAnswerReviewResult:
    """Require every completed independent audit to agree that no issue remains."""

    merged_issues = _merge_candidate_review_issues(
        *(review.advisory_claim_issues for review in reviews)
    )
    if merged_issues:
        return CandidateAnswerReviewResult(
            needs_reconsideration=True,
            advisory_claim_issues=merged_issues,
        )
    if any(review.completed for review in reviews):
        return CandidateAnswerReviewResult(needs_reconsideration=False)
    if reviews:
        return reviews[0]
    return CandidateAnswerReviewResult(needs_reconsideration=False)


def _build_regulatory_synthesis_history(
    *,
    current_request: str,
    earlier_user_context: Sequence[str],
    visible_results_by_citation: dict[int, tuple[str, str]],
    research_targets_by_citation: dict[int, list[str]] | None = None,
    token_counter: Callable[[str], int],
    prior_candidate_answer: str | None = None,
    coverage_contract: str | None = None,
    evidence_matrix: str | None = None,
    priority_citation_numbers: set[int] | None = None,
    max_history_tokens: int | None = None,
) -> list[ChatMessageSimple]:
    """Build a tool-free final context from canonical evidence, not tool transcripts."""

    request_payload = json.dumps(
        {
            "current_request": current_request,
            "earlier_user_context": list(earlier_user_context),
            "coverage_contract": coverage_contract,
            "evidence_matrix": evidence_matrix,
        },
        ensure_ascii=False,
    )
    history = [
        ChatMessageSimple(
            message=request_payload,
            token_count=token_counter(request_payload),
            message_type=MessageType.USER,
        )
    ]
    if prior_candidate_answer and prior_candidate_answer.strip():
        bounded_candidate = prior_candidate_answer.strip()[:36_000].rstrip()
        history.append(
            ChatMessageSimple(
                message=bounded_candidate,
                token_count=token_counter(bounded_candidate),
                message_type=MessageType.ASSISTANT,
            )
        )

    def compact_research_target(target: str) -> str:
        normalized_target = " ".join(target.split())
        if normalized_target.startswith("Specific evidence target:"):
            sentence_end = normalized_target.find(". ")
            if sentence_end >= 0:
                normalized_target = normalized_target[: sentence_end + 1]
        return normalized_target[:480].rstrip()

    evidence_results: list[dict[str, object]] = []
    research_target_ids: dict[str, str] = {}
    research_target_labels: dict[str, str] = {}
    research_target_documents: dict[str, list[int]] = {}
    priority_citations = priority_citation_numbers or set()
    ordered_visible_results = sorted(
        visible_results_by_citation.items(),
        key=lambda item: (item[0] not in priority_citations, item[0]),
    )
    for citation_number, (title, content) in ordered_visible_results:
        if not content.strip():
            continue
        result_target_ids: list[str] = []
        for research_target in (research_targets_by_citation or {}).get(
            citation_number, []
        )[:6]:
            target_id = research_target_ids.get(research_target)
            if target_id is None:
                target_id = f"T{len(research_target_ids) + 1}"
                research_target_ids[research_target] = target_id
                research_target_labels[research_target] = compact_research_target(
                    research_target
                )
                research_target_documents[research_target] = []
            result_target_ids.append(target_id)
            target_documents = research_target_documents[research_target]
            if citation_number not in target_documents:
                target_documents.append(citation_number)
        evidence_results.append(
            {
                "document": citation_number,
                "title": title[:600].rstrip(),
                "content": content[:1_800].rstrip(),
                "research_target_ids": result_target_ids,
            }
        )
    coverage_target_index = [
        {
            "target_id": target_id,
            "target": research_target_labels[research_target],
            "documents": research_target_documents[research_target],
        }
        for research_target, target_id in research_target_ids.items()
    ]

    def serialize_evidence_payload(
        results: list[dict[str, object]],
        *,
        physically_compacted: bool,
    ) -> str:
        return json.dumps(
            {
                "type": "canonical_regulatory_evidence_for_final_synthesis",
                "usage_note": (
                    "This is the exact bounded evidence selected during research. "
                    "The document integers are the only valid citation numbers. "
                    "research_target_ids link each result to the final "
                    "coverage_target_index; target labels are provenance and closure "
                    "obligations, never legal evidence. Silently close every applicable "
                    "target as: exact source and operative text, fact-to-rule application, "
                    "supported result, inline citation. Otherwise state the precise source "
                    "gap. Treat every request-grounded supported evidence-matrix row as an "
                    "atomic drafting requirement: preserve every material limitation and "
                    "relationship established by its cited text. Do not replace an available "
                    "exact rule with an uncertainty note merely because applying one of its "
                    "branches depends on a fact not supplied; state supported alternatives "
                    "conditionally. Keep distinct any propositions that the request or exact "
                    "evidence distinguishes, and do not apply a rule beyond the facts that "
                    "establish its scope. "
                    "A supplied ancestor heading or lead-in is a negative scope limit: "
                    "never detach descendant text from its actor, status or class, stage, "
                    "exception, or condition; it is not positive support alone. "
                    "A heading, neighboring rule, duty, or breach does not establish an "
                    "unstated sanction. Do not call tools or cite an absent number. "
                    + (
                        "Some long chunk bodies were shortened only to fit the answering "
                        "model's physical token context; every retrieved document remains "
                        "represented. Priority evidence uses plan-target-aligned extracts "
                        "before any shortening is permitted. Treat each compaction marker "
                        "as a physical-context boundary, never as evidence that omitted "
                        "text or the source is absent. "
                        if physically_compacted
                        else ""
                    )
                    + (
                        "This is a correction pass. Return a complete replacement answer in "
                        "the visible response, even though the prior draft appears above. "
                        "Preserve supported rows and citations; change only propositions "
                        "implicated by the review or newly retrieved exact evidence. Remove "
                        "or accurately qualify a proposition whose complete trigger is not "
                        "established. "
                        if prior_candidate_answer and prior_candidate_answer.strip()
                        else ""
                    )
                ),
                "results": results,
                "coverage_target_index_note": (
                    "This compact index appears last so it can be used as a final closure "
                    "check. A target is supported only by exact content in its listed "
                    "documents; the target label itself is not legal evidence."
                ),
                "coverage_target_index": coverage_target_index,
            },
            ensure_ascii=False,
        )

    evidence_payload = serialize_evidence_payload(
        evidence_results,
        physically_compacted=False,
    )
    max_evidence_tokens = (
        max_history_tokens - sum(message.token_count for message in history)
        if max_history_tokens is not None
        else None
    )
    if max_evidence_tokens is not None and max_evidence_tokens <= 0:
        raise ValueError(
            "The mandatory regulatory synthesis context exceeds the model's "
            "physical token context."
        )
    if (
        max_evidence_tokens is not None
        and token_counter(evidence_payload) > max_evidence_tokens
        and evidence_results
    ):
        original_contents = [str(result["content"]) for result in evidence_results]
        compaction_marker = "\n...[physical-context compaction]...\n"
        target_extract_marker = "\n...[target-aligned physical-context extract]...\n"

        def compact_content(content: str, max_chars: int) -> str:
            if len(content) <= max_chars:
                return content
            remaining_chars = max_chars - len(compaction_marker)
            if remaining_chars <= 1:
                return content[:max_chars]
            prefix_chars = remaining_chars // 2
            suffix_chars = remaining_chars - prefix_chars
            return content[:prefix_chars] + compaction_marker + content[-suffix_chars:]

        def compact_priority_content(
            content: str,
            max_chars: int,
            citation_number: int,
        ) -> str:
            if len(content) <= max_chars:
                return content
            targets = (research_targets_by_citation or {}).get(citation_number, [])
            lowered_content = content.casefold()
            matched_span: tuple[int, int] | None = None
            for target in targets:
                words = target.split()
                for width in range(len(words), 0, -1):
                    if matched_span is not None:
                        break
                    for start_word in range(0, len(words) - width + 1):
                        phrase = " ".join(
                            words[start_word : start_word + width]
                        ).strip()
                        if len(phrase) < 6:
                            continue
                        match_start = lowered_content.find(phrase.casefold())
                        if match_start >= 0:
                            matched_span = (match_start, match_start + len(phrase))
                            break
                if matched_span is not None:
                    break
            if matched_span is None:
                return compact_content(content, max_chars)

            marker_tokens = len(target_extract_marker) * 2
            excerpt_chars = max_chars - marker_tokens
            if excerpt_chars <= 0:
                return compact_content(content, max_chars)
            match_start, match_end = matched_span
            match_center = (match_start + match_end) // 2
            excerpt_start = max(0, match_center - excerpt_chars // 2)
            excerpt_end = min(len(content), excerpt_start + excerpt_chars)
            excerpt_start = max(0, excerpt_end - excerpt_chars)
            prefix_marker = target_extract_marker if excerpt_start > 0 else ""
            suffix_marker = target_extract_marker if excerpt_end < len(content) else ""
            excerpt = prefix_marker + content[excerpt_start:excerpt_end] + suffix_marker
            return excerpt[:max_chars]

        compacted_payload: str | None = None
        for compact_priority in (False, True):
            low = max(len(compaction_marker), len(target_extract_marker) * 2) + 2
            high = max(len(content) for content in original_contents)
            while low <= high:
                content_char_budget = (low + high) // 2
                compacted_results: list[dict[str, object]] = []
                for result, content in zip(
                    evidence_results, original_contents, strict=True
                ):
                    citation_number = int(result["document"])
                    is_priority = citation_number in priority_citations
                    compacted_content = (
                        compact_priority_content(
                            content,
                            content_char_budget,
                            citation_number,
                        )
                        if is_priority and compact_priority
                        else (
                            content
                            if is_priority
                            else compact_content(content, content_char_budget)
                        )
                    )
                    compacted_results.append({**result, "content": compacted_content})
                candidate_payload = serialize_evidence_payload(
                    compacted_results,
                    physically_compacted=True,
                )
                if token_counter(candidate_payload) <= max_evidence_tokens:
                    compacted_payload = candidate_payload
                    low = content_char_budget + 1
                else:
                    high = content_char_budget - 1
            if compacted_payload is not None:
                break
        if compacted_payload is None:
            raise ValueError(
                "The complete regulatory evidence inventory cannot fit the model's "
                "physical token context even after count-blind content compaction."
            )
        evidence_payload = compacted_payload
    history.append(
        ChatMessageSimple(
            message=evidence_payload,
            token_count=token_counter(evidence_payload),
            message_type=MessageType.USER_REMINDER,
        )
    )
    return history


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


def _build_fast_regulatory_absence_recovery_tool_calls(
    plan: RegulatoryCoveragePlan | None,
    *,
    turn_index: int,
    attempted_query_modes: set[tuple[str, str]],
) -> list[ToolCallKickoff]:
    """Build the one V2 recall pass from deferred, query-distinct plan probes."""

    all_plan_calls = _build_regulatory_coverage_tool_calls(
        plan,
        turn_index=turn_index,
        max_calls=None,
        include_auxiliary_searches=True,
        include_lexical_fallbacks=True,
    )
    deferred_calls = _constrain_regulatory_tool_calls(
        all_plan_calls,
        search_slots=None,
        attempted_query_modes=attempted_query_modes,
    )
    return [
        call.model_copy(
            update={
                "tool_call_id": (
                    f"regulatory-absence-recovery-{turn_index}-{call_index}"
                ),
                "placement": Placement(
                    turn_index=turn_index,
                    tab_index=call_index,
                ),
            }
        )
        for call_index, call in enumerate(deferred_calls)
    ]


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


def _regulatory_outline_result_matches_requested_lead(
    title: str,
    tool_args: dict[str, Any],
) -> bool:
    """Identify the exact operative result requested from a selected outline lead."""

    if tool_args.get("coverage_item") != "Request-grounded source-outline recovery":
        return False
    raw_target = tool_args.get("evidence_target")
    raw_source_anchors = tool_args.get("source_anchors")
    if not isinstance(raw_target, str) or not isinstance(raw_source_anchors, list):
        return False
    target_prefix = "Source-outline exact-text recovery:"
    if not raw_target.startswith(target_prefix):
        return False

    def normalized(value: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())

    normalized_title = normalized(title)
    normalized_document_title = normalized(title.split(" — ", 1)[0])
    normalized_target = normalized(raw_target.removeprefix(target_prefix))
    if len(normalized_target) < 8 or normalized_target not in normalized_title:
        return False
    return any(
        isinstance(source_anchor, str)
        and len(normalized(source_anchor)) >= 4
        and normalized(source_anchor) == normalized_document_title
        for source_anchor in raw_source_anchors
    )


def _extract_regulatory_navigation_leads(
    llm_facing_response: str,
    *,
    research_target: str,
) -> list[RegulatoryNavigationLead]:
    """Extract metadata-only source-outline leads from one search response."""

    try:
        payload = json.loads(llm_facing_response)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    navigation = payload.get("regulatory_provision_navigation")
    if not isinstance(navigation, dict):
        return []
    document_title = navigation.get("document_title")
    raw_headings = navigation.get("headings")
    if not isinstance(document_title, str) or not isinstance(raw_headings, list):
        return []

    leads: list[RegulatoryNavigationLead] = []
    for heading in raw_headings:
        if not isinstance(heading, dict):
            continue
        article_key = heading.get("article_key")
        heading_label = heading.get("heading_label")
        if not isinstance(article_key, str) or not article_key.strip():
            continue
        if not isinstance(heading_label, str) or not heading_label.strip():
            continue
        try:
            leads.append(
                RegulatoryNavigationLead(
                    document_title=document_title,
                    article_key=article_key,
                    heading_label=heading_label,
                    research_targets=[research_target],
                )
            )
        except ValueError:
            continue
    return leads


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


def _regulatory_search_chunk_cap(
    enabled: bool,
    *,
    chunks_per_call: int = _REGULATORY_SEARCH_LLM_CHUNKS_PER_CALL,
) -> int | None:
    """Use a modestly wider window for structure-fragmented regulatory text."""

    return chunks_per_call if enabled else None


def _regulatory_search_call_budget(
    complex_regulatory_request: bool,
) -> int | None:
    """Retain the standard cycle-bounded Onyx tool budget."""

    _ = complex_regulatory_request
    return None


def _effective_regulatory_search_call_budget(
    base_budget: int | None,
    *,
    candidate_was_rejected: bool,
) -> int | None:
    """Keep ordinary research fixed; direct review recovery is accounted separately."""

    _ = candidate_was_rejected
    return base_budget


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
        regulatory_research_targets_by_citation: dict[int, list[str]] = {}
        regulatory_coverage_items_by_citation: dict[int, list[str]] = {}
        regulatory_synthesis_priority_citations: set[int] = set()
        regulatory_navigation_leads_by_identity: dict[
            tuple[str, str], RegulatoryNavigationLead
        ] = {}

        def record_regulatory_research_target(
            citation_number: int,
            target: str,
            *,
            coverage_item: str = "",
        ) -> None:
            normalized_target = " ".join(target.split())
            if not normalized_target:
                return
            targets = regulatory_research_targets_by_citation.setdefault(
                citation_number, []
            )
            if normalized_target not in targets:
                # Focused facet provenance is more useful for omission control than
                # the broad seed query that may have returned the same chunk first.
                if normalized_target.startswith("Specific evidence target:"):
                    targets.insert(0, normalized_target[:900].rstrip())
                else:
                    targets.append(normalized_target[:900].rstrip())
                del targets[6:]
            normalized_coverage_item = " ".join(coverage_item.split())
            if not normalized_coverage_item:
                return
            coverage_items = regulatory_coverage_items_by_citation.setdefault(
                citation_number, []
            )
            if normalized_coverage_item not in coverage_items:
                coverage_items.append(normalized_coverage_item[:900].rstrip())
                del coverage_items[4:]

        def record_regulatory_navigation_leads(
            leads: Sequence[RegulatoryNavigationLead],
        ) -> None:
            for lead in leads:
                identity = (lead.document_title, lead.article_key)
                previous = regulatory_navigation_leads_by_identity.get(identity)
                if previous is None:
                    regulatory_navigation_leads_by_identity[identity] = lead
                    continue
                regulatory_navigation_leads_by_identity[identity] = previous.model_copy(
                    update={
                        "research_targets": list(
                            dict.fromkeys(
                                [
                                    *previous.research_targets,
                                    *lead.research_targets,
                                ]
                            )
                        )[:6]
                    }
                )

        regulatory_search_calls_attempted = 0
        regulatory_attempted_query_modes: set[tuple[str, str]] = set()
        regulatory_tool_feedback: str | None = None
        candidate_review_feedback: str | None = None
        candidate_answer_review_count = 0
        candidate_review_issues: list[CandidateAnswerClaimIssue] = []
        candidate_review_issue_history: list[CandidateAnswerClaimIssue] = []
        candidate_review_rejected_at_cycle: int | None = None
        candidate_final_correction_pending = False
        regulatory_research_complete_pending = False
        fast_absence_recovery_attempted = False
        fast_absence_correction_pending = False
        fast_absence_recovery_feedback: str | None = None
        empty_final_response_retries = 0
        complex_regulatory_request = False
        regulatory_user_message = ""
        regulatory_earlier_user_context: tuple[str, ...] = ()
        regulatory_review_llm = llm
        regulatory_coverage_plan: RegulatoryCoveragePlan | None = None
        regulatory_coverage_reminder: str | None = None
        regulatory_evidence_matrix: RegulatoryEvidenceMatrix | None = None
        regulatory_evidence_matrix_reminder: str | None = None
        regulatory_evidence_matrix_ready = False
        regulatory_evidence_matrix_recovery_started = False
        regulatory_navigation_recovery_started = False
        regulatory_navigation_recovery_ready = False
        regulatory_attempted_navigation_lead_identities: set[tuple[str, str]] = set()
        pending_regulatory_coverage_tool_calls: list[ToolCallKickoff] = []

        regulatory_search_tool = next(
            (
                tool
                for tool in tools
                if isinstance(tool, SearchTool)
                and tool.user_selected_filters is not None
                and tool.user_selected_filters.regulatory_chunks_only
            ),
            None,
        )
        is_regulatory_search_chat = regulatory_search_tool is not None
        regulatory_workflow_profile = STANDARD_REGULATORY_WORKFLOW
        if regulatory_search_tool is not None:
            regulatory_workflow_profile = get_regulatory_workflow_profile(
                regulatory_search_tool.user_selected_filters.regulatory_workflow_mode
            )
        if is_regulatory_search_chat:
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
            regulatory_review_llm = build_regulatory_review_llm(llm)
            regulatory_coverage_plan = build_regulatory_coverage_plan(
                regulatory_review_llm,
                user_request=regulatory_user_message,
            )
            regulatory_coverage_reminder = format_regulatory_coverage_plan(
                regulatory_coverage_plan
            )
            pending_regulatory_coverage_tool_calls = (
                _build_regulatory_coverage_tool_calls(
                    regulatory_coverage_plan,
                    turn_index=0,
                    max_calls=regulatory_workflow_profile.max_parallel_search_calls,
                    include_auxiliary_searches=(
                        regulatory_workflow_profile.include_auxiliary_searches
                    ),
                    include_lexical_fallbacks=(
                        regulatory_workflow_profile.include_lexical_fallbacks
                    ),
                )
            )
            regulatory_navigation_recovery_ready = (
                not regulatory_workflow_profile.use_navigation_recovery
            )
            regulatory_evidence_matrix_ready = (
                not regulatory_workflow_profile.use_evidence_matrix
            )
        regulatory_bootstrap_searches_completed = not bool(
            pending_regulatory_coverage_tool_calls
        )
        # Without a bootstrap batch, the established first model/tool cycle is
        # already the autonomous Onyx research turn and needs no duplicate.
        regulatory_autonomous_research_completed = not bool(
            pending_regulatory_coverage_tool_calls
        )

        def run_full_regulatory_candidate_review(
            candidate_answer: str,
            evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
        ) -> CandidateAnswerReviewResult:
            return review_regulatory_candidate_answer_with_fallback(
                regulatory_review_llm,
                llm,
                user_request=regulatory_user_message,
                earlier_user_context=regulatory_earlier_user_context,
                coverage_contract=regulatory_coverage_reminder,
                evidence_matrix=regulatory_evidence_matrix_reminder,
                candidate_answer=candidate_answer,
                evidence_chunks=evidence_chunks,
            )

        regulatory_search_chunk_cap = _regulatory_search_chunk_cap(
            complex_regulatory_request,
            chunks_per_call=regulatory_workflow_profile.search_chunks_per_call,
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
                regulatory_workflow_profile.post_review_cycles
                + _REGULATORY_PROJECTED_STOP_SYNTHESIS_CYCLES
                + _REGULATORY_BOOTSTRAP_COVERAGE_CYCLES
                + _REGULATORY_AUTONOMOUS_RESEARCH_CYCLES
            )
            if complex_regulatory_request
            else 0
        )
        for llm_cycle_count in range(maximum_cycle_count):
            out_of_cycles = (
                llm_cycle_count
                >= MAX_LLM_CYCLES
                + _REGULATORY_BOOTSTRAP_COVERAGE_CYCLES
                + _REGULATORY_AUTONOMOUS_RESEARCH_CYCLES
                - 1
                if candidate_review_rejected_at_cycle is None
                else llm_cycle_count
                >= candidate_review_rejected_at_cycle
                + regulatory_workflow_profile.post_review_cycles
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
            autonomous_regulatory_research_turn = (
                complex_regulatory_request
                and has_called_search_tool
                and regulatory_bootstrap_searches_completed
                and not regulatory_autonomous_research_completed
                and not final_synthesis_required
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

            if (
                complex_regulatory_request
                and has_called_search_tool
                and regulatory_autonomous_research_completed
                and not regulatory_navigation_recovery_ready
            ):
                if not regulatory_navigation_recovery_started:
                    available_navigation_leads = (
                        _unattempted_regulatory_navigation_leads(
                            list(regulatory_navigation_leads_by_identity.values()),
                            regulatory_attempted_navigation_lead_identities,
                        )
                    )
                    selected_navigation_leads = (
                        select_regulatory_navigation_recovery_leads(
                            regulatory_review_llm,
                            user_request=regulatory_user_message,
                            coverage_contract=regulatory_coverage_reminder,
                            navigation_leads=available_navigation_leads,
                        )
                    )
                    regulatory_attempted_navigation_lead_identities.update(
                        (lead.document_title, lead.article_key)
                        for lead in selected_navigation_leads
                    )
                    pending_regulatory_coverage_tool_calls = (
                        _build_regulatory_navigation_recovery_tool_calls(
                            selected_navigation_leads,
                            turn_index=llm_cycle_count,
                        )
                    )
                    regulatory_navigation_recovery_started = True
                    logger.info(
                        "Regulatory source-outline recovery selected=%d available=%d",
                        len(pending_regulatory_coverage_tool_calls),
                        len(regulatory_navigation_leads_by_identity),
                    )
                    if not pending_regulatory_coverage_tool_calls:
                        regulatory_navigation_recovery_ready = True
                elif not pending_regulatory_coverage_tool_calls:
                    regulatory_navigation_recovery_ready = True

            if (
                complex_regulatory_request
                and has_called_search_tool
                and regulatory_navigation_recovery_ready
                and not regulatory_evidence_matrix_ready
            ):
                matrix_evidence_chunks = _build_candidate_answer_evidence_chunks(
                    candidate_answer="",
                    citation_mapping=citation_processor.citation_to_doc,
                    llm_visible_results_by_citation=(
                        llm_visible_search_results_by_citation
                    ),
                    research_targets_by_citation=(
                        regulatory_research_targets_by_citation
                    ),
                    coverage_items_by_citation=(regulatory_coverage_items_by_citation),
                )
                matrix_evidence_chunks = _select_regulatory_matrix_input_evidence(
                    matrix_evidence_chunks
                )
                logger.info(
                    "Regulatory evidence matrix input evidence=%d",
                    len(matrix_evidence_chunks),
                )
                independent_evidence_matrix = build_regulatory_evidence_matrix(
                    regulatory_review_llm,
                    user_request=regulatory_user_message,
                    coverage_contract=regulatory_coverage_reminder,
                    evidence_chunks=matrix_evidence_chunks,
                    navigation_leads=list(
                        regulatory_navigation_leads_by_identity.values()
                    ),
                    prior_matrix=(
                        regulatory_evidence_matrix
                        if regulatory_evidence_matrix_recovery_started
                        else None
                    ),
                )
                regulatory_evidence_matrix = independent_evidence_matrix
                regulatory_evidence_matrix_reminder = format_regulatory_evidence_matrix(
                    regulatory_evidence_matrix
                )
                recovery_queries = (
                    evidence_matrix_recovery_queries(
                        regulatory_evidence_matrix,
                        limit=regulatory_workflow_profile.max_parallel_search_calls,
                    )
                    if not regulatory_evidence_matrix_recovery_started
                    else []
                )
                if recovery_queries and tool_choice is ToolChoiceOptions.AUTO:
                    pending_regulatory_coverage_tool_calls = [
                        ToolCallKickoff(
                            tool_call_id=(
                                "regulatory-matrix-recovery-"
                                f"{llm_cycle_count}-{query_index}"
                            ),
                            tool_name=SearchTool.NAME,
                            tool_args={
                                "queries": [query],
                                "search_mode": "hybrid",
                                "coverage_item": (
                                    "Claim-source evidence matrix open row"
                                ),
                                "evidence_target": (
                                    "Evidence matrix recovery: " + query
                                ),
                                "source_anchors": [],
                            },
                            placement=Placement(
                                turn_index=llm_cycle_count,
                                tab_index=query_index,
                            ),
                        )
                        for query_index, query in enumerate(recovery_queries)
                    ]
                    regulatory_evidence_matrix_recovery_started = True
                    logger.info(
                        "Regulatory evidence matrix rows=%d open_recovery_queries=%d",
                        (
                            len(regulatory_evidence_matrix.rows)
                            if regulatory_evidence_matrix is not None
                            else 0
                        ),
                        len(recovery_queries),
                    )
                else:
                    regulatory_evidence_matrix_ready = True
                    logger.info(
                        "Regulatory evidence matrix ready rows=%d recovery_completed=%s",
                        (
                            len(regulatory_evidence_matrix.rows)
                            if regulatory_evidence_matrix is not None
                            else 0
                        ),
                        regulatory_evidence_matrix_recovery_started,
                    )

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
            isolated_synthesis_inputs: dict[str, Any] | None = None
            isolated_synthesis_evidence_count = 0
            if (
                final_synthesis_required
                and complex_regulatory_request
                and llm_visible_search_results_by_citation
            ):
                prior_candidate_answer = next(
                    (
                        message.message
                        for message in reversed(simple_chat_history)
                        if message.message_type == MessageType.ASSISTANT
                        and message.message.strip()
                    ),
                    None,
                )
                complete_synthesis_evidence = _build_candidate_answer_evidence_chunks(
                    candidate_answer=prior_candidate_answer or "",
                    citation_mapping=citation_processor.citation_to_doc,
                    llm_visible_results_by_citation=(
                        llm_visible_search_results_by_citation
                    ),
                    research_targets_by_citation=(
                        regulatory_research_targets_by_citation
                    ),
                    coverage_items_by_citation=(regulatory_coverage_items_by_citation),
                )
                selected_synthesis_evidence = _select_regulatory_closure_evidence(
                    complete_synthesis_evidence,
                    candidate_answer=prior_candidate_answer or "",
                    evidence_matrix=regulatory_evidence_matrix,
                    priority_citation_numbers=(regulatory_synthesis_priority_citations),
                )
                selected_synthesis_numbers = {
                    chunk.retrieval_number
                    for chunk in selected_synthesis_evidence
                    if chunk.retrieval_number is not None
                }
                synthesis_visible_results = {
                    citation_number: result
                    for citation_number, result in (
                        llm_visible_search_results_by_citation.items()
                    )
                    if citation_number in selected_synthesis_numbers
                }
                isolated_synthesis_inputs = {
                    "current_request": regulatory_user_message,
                    "earlier_user_context": regulatory_earlier_user_context,
                    "visible_results_by_citation": synthesis_visible_results,
                    "research_targets_by_citation": (
                        regulatory_research_targets_by_citation
                    ),
                    "token_counter": token_counter,
                    "prior_candidate_answer": (
                        prior_candidate_answer
                        if candidate_final_correction_pending
                        or fast_absence_correction_pending
                        else None
                    ),
                    "coverage_contract": regulatory_coverage_reminder,
                    "evidence_matrix": regulatory_evidence_matrix_reminder,
                    "priority_citation_numbers": (
                        regulatory_synthesis_priority_citations
                    ),
                }
                isolated_synthesis_evidence_count = len(synthesis_visible_results)
            if (
                complex_regulatory_request
                and has_called_search_tool
                and tool_choice is ToolChoiceOptions.AUTO
                and llm_cycle_count < maximum_cycle_count - 1
                and (
                    regulatory_evidence_matrix_ready
                    or sum(message.token_count for message in simple_chat_history)
                    > available_tokens
                )
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
                    regulatory_coverage_reminder,
                    regulatory_evidence_matrix_reminder,
                    regulatory_tool_feedback,
                    _format_search_evidence_ledger(search_evidence_ledger),
                    candidate_review_feedback,
                    fast_absence_recovery_feedback,
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
            if (
                isolated_synthesis_inputs is not None
                and not projected_tool_decision_history
            ):
                project_message_tokens = sum(
                    message.token_count
                    for message in _build_project_message(context_files, token_counter)
                )
                mandatory_external_tokens = (
                    (system_prompt.token_count if system_prompt else 0)
                    + (
                        custom_agent_prompt_msg.token_count
                        if custom_agent_prompt_msg
                        else 0
                    )
                    + project_message_tokens
                    + (reminder_msg.token_count if reminder_msg else 0)
                    + tool_token_budget
                )
                synthesis_history_budget = available_tokens - mandatory_external_tokens
                if synthesis_history_budget <= 0:
                    raise ValueError(
                        "Mandatory prompts exceed the answering model's physical "
                        "token context before regulatory evidence is added."
                    )
                history_for_llm_step = _build_regulatory_synthesis_history(
                    **isolated_synthesis_inputs,
                    max_history_tokens=synthesis_history_budget,
                )
                logger.info(
                    "Built isolated regulatory synthesis context evidence=%d/%d "
                    "history_budget=%d prior_candidate=%s",
                    isolated_synthesis_evidence_count,
                    len(llm_visible_search_results_by_citation),
                    synthesis_history_budget,
                    bool(isolated_synthesis_inputs["prior_candidate_answer"]),
                )
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
            step_reasoning_effort = (
                ReasoningEffort.OFF
                if final_synthesis_required and complex_regulatory_request
                else reasoning_effort
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
            if (
                pending_regulatory_coverage_tool_calls
                and tool_choice is ToolChoiceOptions.AUTO
            ):
                llm_step_result = LlmStepResult(
                    reasoning=None,
                    answer=None,
                    tool_calls=pending_regulatory_coverage_tool_calls,
                    raw_answer=None,
                    finish_reason="tool_calls",
                )
                pending_regulatory_coverage_tool_calls = []
                has_reasoned = False
            else:
                llm_step_result, has_reasoned = run_llm_step(
                    emitter=buffered_step_emitter or emitter,
                    history=truncated_message_history,
                    tool_definitions=tool_defs,
                    tool_choice=tool_choice,
                    llm=llm,
                    placement=Placement(turn_index=llm_cycle_count + reasoning_cycles),
                    citation_processor=staged_citation_processor,
                    state_container=staged_state_container,
                    # Rich docs let answer packets expose the final document set.
                    final_documents=gathered_documents,
                    user_identity=user_identity,
                    pre_answer_processing_time=pre_answer_processing_time,
                    reasoning_effort=step_reasoning_effort,
                    max_tokens=_regulatory_llm_step_max_tokens(
                        complex_regulatory_request=complex_regulatory_request,
                        tool_choice=tool_choice,
                        projected_tool_decision_history=(
                            projected_tool_decision_history
                        ),
                        reasoning_effort=step_reasoning_effort,
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

            if autonomous_regulatory_research_turn:
                regulatory_autonomous_research_completed = True
                if not (llm_step_result.tool_calls or []):
                    logger.info(
                        "Regulatory autonomous research turn requested no additional "
                        "tools; continuing with evidence closure"
                    )
                    continue

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
                and llm_step_result.finish_reason not in REFUSAL_FINISH_REASONS
            ):
                empty_final_response_retries += 1
                logger.warning(
                    "Final regulatory synthesis returned empty; retrying once"
                )
                continue
            search_slots = (
                min(
                    regulatory_workflow_profile.max_parallel_search_calls,
                    max(
                        0,
                        effective_regulatory_search_call_budget
                        - regulatory_search_calls_attempted,
                    ),
                )
                if effective_regulatory_search_call_budget is not None
                else regulatory_workflow_profile.max_parallel_search_calls
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
                accepted_answer_override: str | None = None
                if (
                    regulatory_workflow_profile.mode == "fast"
                    and has_called_search_tool
                    and candidate_answer_for_review.strip()
                    and not fast_absence_recovery_attempted
                    and _draft_claims_regulatory_source_gap(candidate_answer_for_review)
                ):
                    fast_absence_recovery_attempted = True
                    fast_absence_correction_pending = True
                    fast_absence_recovery_feedback = (
                        _FAST_REGULATORY_ABSENCE_RECOVERY_REMINDER
                    )
                    pending_regulatory_coverage_tool_calls = (
                        _build_fast_regulatory_absence_recovery_tool_calls(
                            regulatory_coverage_plan,
                            turn_index=llm_cycle_count + reasoning_cycles + 1,
                            attempted_query_modes=regulatory_attempted_query_modes,
                        )
                    )
                    simple_chat_history.append(
                        ChatMessageSimple(
                            message=candidate_answer_for_review,
                            token_count=token_counter(candidate_answer_for_review),
                            message_type=MessageType.ASSISTANT,
                        )
                    )
                    if pending_regulatory_coverage_tool_calls:
                        logger.info(
                            "Fast regulatory source-gap draft withheld; scheduling "
                            "%d deferred plan search(es)",
                            len(pending_regulatory_coverage_tool_calls),
                        )
                    else:
                        regulatory_research_complete_pending = True
                        logger.info(
                            "Fast regulatory source-gap draft withheld; no distinct "
                            "plan search remained, scheduling evidence re-synthesis"
                        )
                    continue
                if (
                    candidate_answer_review_count
                    < regulatory_workflow_profile.max_candidate_reviews
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
                        research_targets_by_citation=(
                            regulatory_research_targets_by_citation
                        ),
                        coverage_items_by_citation=(
                            regulatory_coverage_items_by_citation
                        ),
                    )
                    candidate_evidence_chunks = _select_regulatory_closure_evidence(
                        candidate_evidence_chunks,
                        candidate_answer=candidate_answer_for_review,
                        evidence_matrix=regulatory_evidence_matrix,
                        priority_citation_numbers=(
                            regulatory_synthesis_priority_citations
                        ),
                    )
                    prior_candidate_review_issues = _merge_candidate_review_issues(
                        candidate_review_issue_history,
                        candidate_review_issues,
                    )
                    if candidate_answer_review_count == 1 or not (
                        prior_candidate_review_issues
                    ):
                        candidate_review = run_full_regulatory_candidate_review(
                            candidate_answer_for_review,
                            candidate_evidence_chunks,
                        )
                        review_kind = "evidence"
                    else:
                        # Recheck the bounded union of all earlier issues on every
                        # correction pass. This prevents a later edit from undoing
                        # an already resolved row without repeating the much larger
                        # full-evidence audit payload.
                        resolution_review = (
                            review_regulatory_candidate_resolution_with_fallback(
                                regulatory_review_llm,
                                llm,
                                candidate_answer=candidate_answer_for_review,
                                prior_issues=prior_candidate_review_issues,
                                evidence_chunks=candidate_evidence_chunks,
                            )
                        )
                        independent_review = run_full_regulatory_candidate_review(
                            candidate_answer_for_review,
                            candidate_evidence_chunks,
                        )
                        candidate_review = _merge_candidate_review_verdicts(
                            resolution_review,
                            independent_review,
                        )
                        review_kind = "resolution+independent-evidence"
                    matrix_closure_review = None
                    structural_matrix_issues: list[CandidateAnswerClaimIssue] = []
                    if regulatory_evidence_matrix is not None:
                        structural_matrix_issues = (
                            _build_regulatory_matrix_citation_issues(
                                candidate_answer_for_review,
                                regulatory_evidence_matrix,
                            )
                        )
                        focused_matrix_evidence = (
                            _select_regulatory_matrix_review_evidence(
                                candidate_evidence_chunks,
                                regulatory_evidence_matrix,
                            )
                        )
                        if (
                            regulatory_evidence_matrix_reminder is not None
                            and focused_matrix_evidence
                        ):
                            matrix_closure_review = review_regulatory_candidate_matrix_closure_with_fallback(
                                regulatory_review_llm,
                                llm,
                                user_request=regulatory_user_message,
                                candidate_answer=candidate_answer_for_review,
                                evidence_chunks=focused_matrix_evidence,
                                evidence_matrix=(regulatory_evidence_matrix_reminder),
                            )
                            review_kind += "+matrix"
                    merged_review_issues = _merge_candidate_review_issues(
                        structural_matrix_issues,
                        (
                            matrix_closure_review.advisory_claim_issues
                            if matrix_closure_review is not None
                            else []
                        ),
                        candidate_review.advisory_claim_issues,
                    )
                    if merged_review_issues:
                        candidate_review = CandidateAnswerReviewResult(
                            needs_reconsideration=True,
                            advisory_claim_issues=merged_review_issues,
                        )
                    elif (
                        matrix_closure_review is not None
                        and matrix_closure_review.completed
                    ):
                        candidate_review = CandidateAnswerReviewResult(
                            needs_reconsideration=False
                        )
                    candidate_review_issues = list(
                        candidate_review.advisory_claim_issues
                    )
                    regulatory_synthesis_priority_citations.update(
                        citation_number
                        for issue in candidate_review_issues
                        for citation_number in issue.related_citation_numbers
                    )
                    current_review_feedback = (
                        format_candidate_answer_review(candidate_review)
                        if review_kind.startswith("evidence")
                        or "+matrix" in review_kind
                        else format_candidate_resolution_review(candidate_review)
                    )
                    regression_guard = (
                        format_candidate_review_regression_guard(
                            candidate_review_issue_history,
                            candidate_review_issues,
                        )
                        if current_review_feedback is not None
                        else None
                    )
                    candidate_review_feedback = (
                        "\n\n".join(
                            part
                            for part in (current_review_feedback, regression_guard)
                            if part
                        )
                        or None
                    )
                    candidate_review_issue_history.extend(candidate_review_issues)
                    if (
                        candidate_answer_review_count == 1
                        and candidate_review.needs_reconsideration
                    ):
                        recovery_issues = select_priority_recovery_issues(
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
                        if recovery_issues and recovery_search_tool is not None:
                            recovery_queries = [
                                issue.recovery_query
                                for issue in recovery_issues
                                if issue.recovery_query is not None
                            ]
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
                                    "queries": recovery_queries,
                                    "search_mode": "hybrid",
                                },
                                placement=recovery_placement,
                            )
                            try:
                                recovery_response = run_batched_gap_recovery(
                                    search_tool=recovery_search_tool,
                                    issues=recovery_issues,
                                    starting_citation_num=(
                                        citation_processor.get_next_citation_number()
                                    ),
                                    placement=recovery_placement,
                                )
                                canonicalize_search_tool_response_citations(
                                    recovery_response,
                                    citation_processor.citation_to_doc,
                                    reserved_citation_numbers=citation_mapping,
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
                                raw_recovery_coverage_item = (
                                    recovery_call.tool_args.get("coverage_item")
                                )
                                recovery_coverage_item = (
                                    raw_recovery_coverage_item
                                    if isinstance(raw_recovery_coverage_item, str)
                                    else "Candidate answer evidence recovery"
                                )
                                for (
                                    citation_number,
                                    title,
                                    content,
                                ) in visible_recovery_results:
                                    regulatory_synthesis_priority_citations.add(
                                        citation_number
                                    )
                                    llm_visible_search_results_by_citation[
                                        citation_number
                                    ] = (title, content)
                                    record_regulatory_research_target(
                                        citation_number,
                                        "\n".join(recovery_queries),
                                        coverage_item=recovery_coverage_item,
                                    )
                                search_evidence_ledger.append(
                                    SearchEvidenceLedgerEntry(
                                        query="\n".join(recovery_queries),
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
                                    "Batched regulatory citation-gap search failed; "
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
                    if candidate_review_feedback is not None and (
                        _should_schedule_regulatory_candidate_correction(
                            candidate_answer_review_count,
                            candidate_review,
                            max_reviews=(
                                regulatory_workflow_profile.max_candidate_reviews
                            ),
                        )
                    ):
                        if candidate_review_rejected_at_cycle is None:
                            candidate_review_rejected_at_cycle = llm_cycle_count
                        # A reviewed draft gets one bounded, server-selected retrieval
                        # call at most. All later correction passes run with tools disabled.
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
                    if candidate_review_feedback is not None:
                        logger.warning(
                            "Regulatory final reviewed candidate retains %d issue(s); "
                            "publishing the reviewed draft instead of scheduling an "
                            "unreviewed full rewrite",
                            len(candidate_review.advisory_claim_issues),
                        )

                if fast_absence_recovery_attempted:
                    publication_candidate = (
                        llm_step_result.answer or candidate_answer_for_review
                    )
                    qualified_candidate = _qualify_fast_regulatory_source_gap_answer(
                        publication_candidate
                    )
                    if qualified_candidate != publication_candidate:
                        accepted_answer_override = qualified_candidate
                        llm_step_result = llm_step_result.model_copy(
                            update={
                                "answer": qualified_candidate,
                                "raw_answer": qualified_candidate,
                            }
                        )

                commit_staged_llm_step(
                    buffered_emitter=buffered_step_emitter,
                    staged_state=staged_state_container,
                    staged_citation_processor=staged_citation_processor,
                    emitter=emitter,
                    state_container=state_container,
                    pre_answer_processing_time=(time.monotonic() - loop_start_time),
                    answer_override=accepted_answer_override,
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
            if regulatory_workflow_profile.mode == "fast":
                _prime_fast_regulatory_query_embeddings(tool_calls)
            parallel_tool_call_results = run_tool_calls(
                tool_calls=tool_calls,
                tools=final_tools,
                message_history=truncated_message_history,
                user_memory_context=user_memory_context,
                user_info=None,  # TODO, this is part of memories right now, might want to separate it out
                citation_mapping=citation_mapping,
                next_citation_num=citation_processor.get_next_citation_number(),
                max_concurrent_tools=None,
                max_parallel_workers=(
                    regulatory_workflow_profile.max_concurrent_search_tools
                    if complex_regulatory_request
                    else None
                ),
                skip_search_query_expansion=False,
                chat_files=chat_files,
                url_snippet_map=extract_url_snippet_map(gathered_documents or []),
                inject_memories_in_prompt=inject_memories_in_prompt,
                search_llm_chunks_per_call_cap=regulatory_search_chunk_cap,
            )
            tool_responses = parallel_tool_call_results.tool_responses
            citation_mapping = parallel_tool_call_results.updated_citation_mapping

            absence_recovery_batch = any(
                tool_call.tool_call_id.startswith("regulatory-absence-recovery-")
                for tool_call in tool_calls
            )
            if absence_recovery_batch:
                # The recovery attempt is consumed even when every backend call
                # fails. The next cycle must be one tools-disabled replacement
                # synthesis, never an unbounded autonomous retry loop.
                regulatory_research_complete_pending = True

            # Failure case, give something reasonable to the LLM to try again
            if tool_calls and not tool_responses:
                failure_messages = create_tool_call_failure_messages(
                    tool_calls, token_counter
                )
                simple_chat_history.extend(failure_messages)
                continue

            if any(
                tool_call.tool_call_id.startswith("regulatory-coverage-0-")
                for tool_call in tool_calls
            ):
                regulatory_bootstrap_searches_completed = True
                if regulatory_workflow_profile.direct_synthesis_after_plan_search:
                    regulatory_autonomous_research_completed = True
                    regulatory_research_complete_pending = True
            for tool_response in tool_responses:
                # Extract tool_call from the response (set by run_tool_calls)
                if tool_response.tool_call is None:
                    raise ValueError("Tool response missing tool_call reference")

                tool_call = tool_response.tool_call
                tab_index = tool_call.placement.tab_index

                raw_citation_numbers = (
                    set(tool_response.rich_response.citation_mapping)
                    if isinstance(tool_response.rich_response, SearchDocsResponse)
                    else set()
                )
                canonicalize_search_tool_response_citations(
                    tool_response,
                    citation_processor.citation_to_doc,
                    reserved_citation_numbers=(
                        set(citation_mapping) - raw_citation_numbers
                    ),
                )
                synchronize_lightweight_citation_mapping(
                    tool_response,
                    citation_mapping,
                    raw_citation_numbers,
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

                # Track whether indexed evidence is available for review and synthesis.
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
                        raw_coverage_item = tool_call.tool_args.get("coverage_item")
                        coverage_item = (
                            raw_coverage_item
                            if isinstance(raw_coverage_item, str)
                            else ""
                        )
                        record_regulatory_navigation_leads(
                            _extract_regulatory_navigation_leads(
                                llm_history_response,
                                research_target=query,
                            )
                        )
                        for (
                            citation_number,
                            title,
                            content,
                        ) in llm_visible_results:
                            if absence_recovery_batch:
                                # Exact excerpts retrieved specifically to repair a
                                # withheld source-gap draft must survive any later
                                # physical-context compaction verbatim.
                                regulatory_synthesis_priority_citations.add(
                                    citation_number
                                )
                            llm_visible_search_results_by_citation[citation_number] = (
                                title,
                                content,
                            )
                            record_regulatory_research_target(
                                citation_number,
                                query,
                                coverage_item=coverage_item,
                            )
                            if _regulatory_outline_result_matches_requested_lead(
                                title,
                                tool_call.tool_args,
                            ):
                                regulatory_synthesis_priority_citations.add(
                                    citation_number
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
            raise build_empty_llm_response_error(
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
