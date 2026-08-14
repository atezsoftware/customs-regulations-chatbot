# TODO: Notes for potential extensions and future improvements:
# 1. Allow tools that aren't search specific tools
# 2. Use user provided custom prompts
# 3. Save the plan for replay

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMapping, DynamicCitationProcessor
from onyx.chat.citation_utils import extract_citation_order_from_text
from onyx.chat.emitter import BufferedEmitter, Emitter
from onyx.chat.empty_response import (
    OUTPUT_TOKEN_LIMIT_FINISH_REASONS,
    REFUSAL_FINISH_REASONS,
    EmptyLLMResponseError,
    build_empty_llm_response_error,
)
from onyx.chat.llm_loop import (
    _build_regulatory_coverage_tool_calls,
    construct_message_history,
)
from onyx.chat.llm_step import run_llm_step, run_llm_step_pkt_generator
from onyx.chat.models import (
    ChatMessageSimple,
    FileToolMetadata,
    LlmStepResult,
    ToolCallSimple,
)
from onyx.chat.staged_generation import commit_staged_llm_step
from onyx.configs.chat_configs import (
    DR_REPORT_LLM_TIMEOUT_S,
    SKIP_DEEP_RESEARCH_CLARIFICATION,
)
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDocsResponse
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.tools import get_tool_by_name
from onyx.deep_research.dr_mock_tools import (
    MAX_RESEARCH_AGENT_TASK_CHARS,
    RESEARCH_AGENT_TASK_KEY,
    RESEARCH_AGENT_TOOL_NAME,
    THINK_TOOL_RESPONSE_MESSAGE,
    THINK_TOOL_RESPONSE_TOKEN_COUNT,
    get_clarification_tool_definitions,
    get_orchestrator_tools,
)
from onyx.deep_research.utils import (
    check_special_tool_calls,
    create_think_tool_token_processor,
)
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.model_capabilities import model_is_reasoning_model
from onyx.llm.models import ReasoningEffort, ToolChoiceOptions
from onyx.prompts.deep_research.orchestration_layer import (
    CLARIFICATION_PROMPT,
    FINAL_REPORT_PROMPT,
    FIRST_CYCLE_REMINDER,
    FIRST_CYCLE_REMINDER_TOKENS,
    INTERNAL_SEARCH_CLARIFICATION_GUIDANCE,
    INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE,
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_PROMPT_REASONING,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_PLAN_REMINDER,
    USER_FINAL_REPORT_QUERY,
)
from onyx.prompts.prompt_utils import get_current_llm_day_time
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerEvidenceChunk,
    build_regulatory_review_llm,
    build_regulatory_review_user_context,
    format_candidate_answer_review,
    format_candidate_correction_evidence,
    format_candidate_resolution_review,
    review_regulatory_candidate_answer,
    review_regulatory_candidate_resolution,
)
from onyx.regulatory.coverage_plan import (
    RegulatoryCoveragePlan,
    build_regulatory_coverage_plan,
    format_regulatory_coverage_plan,
)
from onyx.regulatory.gap_recovery import (
    exact_recovery_evidence_chunks,
    merge_recovery_citation_mapping,
    merge_recovery_evidence_chunks,
    recovery_search_docs_by_citation,
    run_single_gap_recovery,
    select_priority_recovery_issue,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    DeepResearchPlanDelta,
    DeepResearchPlanStart,
    OverallStop,
    Packet,
    SectionEnd,
    TopLevelBranching,
)
from onyx.tools.fake_tools.research_agent import (
    _exact_regulatory_evidence_from_search_response,
    run_research_agent_calls,
)
from onyx.tools.interface import Tool
from onyx.tools.models import ToolCallInfo, ToolCallKickoff
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_runner import run_tool_calls
from onyx.tracing.framework.create import ChatTraceMetadata, function_span, trace
from onyx.utils.logger import setup_logger
from onyx.utils.timing import log_function_time

logger = setup_logger()

MAX_USER_MESSAGES_FOR_CONTEXT = 5
MAX_CLARIFICATION_TOKENS = 1024
MAX_RESEARCH_PLAN_TOKENS = 2048
MAX_FINAL_REPORT_TOKENS = 20000
MAX_REGULATORY_FINAL_REPORT_TOKENS = MAX_FINAL_REPORT_TOKENS
MAX_PARALLEL_RESEARCH_AGENT_CALLS = 3
MAX_TOTAL_RESEARCH_AGENT_CALLS = 12
MAX_REGULATORY_RETRY_REPORTS = 6
MAX_REGULATORY_RETRY_REPORT_CHARS = 8_000
MAX_REGULATORY_RETRY_EVIDENCE_CHARS = 24_000
_REGULATORY_COVERAGE_SEARCH_BATCH_SIZE = 4
_REGULATORY_COVERAGE_SEARCH_CHUNKS_PER_CALL = 6
_REGULATORY_COVERAGE_EVIDENCE_CHUNKS = 48
_REGULATORY_SOURCE_GAP_FALLBACK = (
    "Mevcut aramada elde edilen doğrulanabilir kaynak parçaları, istenen hukuki "
    "sonuca güvenilir biçimde ulaşmak için yeterli değildir. Bu nedenle "
    "desteklenmeyen bir hukuki sonuç sunmuyorum. Değerlendirme için uygulanacak "
    "tarih ve hukuk ile ilgili düzenlemelerin tam ve yürürlükteki metinlerinin "
    "doğrulanması gerekir."
)

# 30 minute timeout before forcing final report generation
# NOTE: The overall execution may be much longer still because it could run a research cycle at minute 29
# and that runs for another nearly 30 minutes.
DEEP_RESEARCH_FORCE_REPORT_SECONDS = 30 * 60

# Might be something like (this gives a lot of leeway for change but typically the models don't do this):
# 0. Research topics 1-3
# 1. Think
# 2. Research topics 4-5
# 3. Think
# 4. Research topics 6 + something new or different from the plan
# 5. Think
# 6. Research, possibly something new or different from the plan
# 7. Think
# 8. Generate report
MAX_ORCHESTRATOR_CYCLES = 8

# Similar but without the 4 thinking tool calls
MAX_ORCHESTRATOR_CYCLES_REASONING = 4


@dataclass(frozen=True)
class _StagedFinalReportAttempt:
    emitter: BufferedEmitter
    state: ChatStateContainer
    citation_processor: DynamicCitationProcessor
    result: LlmStepResult
    has_reasoned: bool


def _final_report_attempt_is_usable(result: LlmStepResult) -> bool:
    candidate = result.raw_answer or result.answer or ""
    return bool(
        result.answer is not None
        and candidate.strip()
        and result.finish_reason not in OUTPUT_TOKEN_LIMIT_FINISH_REASONS
        and result.finish_reason not in REFUSAL_FINISH_REASONS
    )


def _run_staged_final_report_attempt(
    *,
    history: list[ChatMessageSimple],
    llm: LLM,
    base_citation_processor: DynamicCitationProcessor,
    final_documents: list,
    user_identity: LLMUserIdentity | None,
    turn_index: int,
    reasoning_effort: ReasoningEffort,
    max_tokens: int,
    pre_answer_processing_time: float | None,
) -> _StagedFinalReportAttempt:
    staged_emitter = BufferedEmitter()
    staged_state = ChatStateContainer()
    staged_citation_processor = base_citation_processor.fork()
    result, has_reasoned = run_llm_step(
        emitter=staged_emitter,
        history=history,
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.NONE,
        llm=llm,
        reasoning_effort=reasoning_effort,
        placement=Placement(turn_index=turn_index),
        citation_processor=staged_citation_processor,
        state_container=staged_state,
        final_documents=final_documents,
        user_identity=user_identity,
        max_tokens=max_tokens,
        is_deep_research=True,
        pre_answer_processing_time=pre_answer_processing_time,
        timeout_override=DR_REPORT_LLM_TIMEOUT_S,
    )
    return _StagedFinalReportAttempt(
        emitter=staged_emitter,
        state=staged_state,
        citation_processor=staged_citation_processor,
        result=result,
        has_reasoned=has_reasoned,
    )


def _generate_usable_final_report_attempt(
    *,
    history: list[ChatMessageSimple],
    llm: LLM,
    base_citation_processor: DynamicCitationProcessor,
    final_documents: list,
    user_identity: LLMUserIdentity | None,
    turn_index: int,
    max_tokens: int,
    pre_answer_processing_time: float | None,
    retry_history: list[ChatMessageSimple] | None = None,
) -> _StagedFinalReportAttempt:
    attempts = (
        (history, ReasoningEffort.LOW),
        (retry_history or history, ReasoningEffort.OFF),
    )
    terminal_attempt: _StagedFinalReportAttempt | None = None
    for attempt_number, (attempt_history, attempt_effort) in enumerate(
        attempts, start=1
    ):
        terminal_attempt = _run_staged_final_report_attempt(
            history=attempt_history,
            llm=llm,
            base_citation_processor=base_citation_processor,
            final_documents=final_documents,
            user_identity=user_identity,
            turn_index=turn_index,
            reasoning_effort=attempt_effort,
            max_tokens=max_tokens,
            pre_answer_processing_time=pre_answer_processing_time,
        )
        if _final_report_attempt_is_usable(terminal_attempt.result):
            return terminal_attempt

        error = build_empty_llm_response_error(
            llm=llm,
            llm_step_result=terminal_attempt.result,
            tool_choice=ToolChoiceOptions.NONE,
        )
        logger.warning(
            "Deep-research final report attempt unusable: provider=%s model=%s "
            "attempt=%s effort=%s finish_reason=%s has_reasoning=%s "
            "has_raw_answer=%s",
            llm.config.model_provider,
            llm.config.model_name,
            attempt_number,
            attempt_effort.value,
            terminal_attempt.result.finish_reason,
            bool(terminal_attempt.result.reasoning),
            bool((terminal_attempt.result.raw_answer or "").strip()),
        )
        if not error.is_retryable:
            raise error

    if terminal_attempt is None:
        raise RuntimeError("Final report attempt schedule must not be empty")
    raise build_empty_llm_response_error(
        llm=llm,
        llm_step_result=terminal_attempt.result,
        tool_choice=ToolChoiceOptions.NONE,
    )


def _orchestrator_cycle_schedule(max_decision_cycles: int) -> range:
    """Reserve one forced-report pass after every advertised decision cycle."""

    if max_decision_cycles <= 0:
        raise ValueError("max_decision_cycles must be positive")
    return range(max_decision_cycles + 1)


def _custom_agent_prompt_message(
    custom_agent_prompt: str | None,
    token_counter: Callable[[str], int],
) -> ChatMessageSimple | None:
    if not custom_agent_prompt:
        return None
    return ChatMessageSimple(
        message=custom_agent_prompt,
        token_count=token_counter(custom_agent_prompt),
        message_type=MessageType.USER,
    )


def _deep_research_search_tools(tools: Sequence[Tool]) -> list[Tool]:
    """Keep Deep Research restricted to the built-in internal search type."""

    return [tool for tool in tools if isinstance(tool, SearchTool)]


def _round_robin_coverage_evidence(
    evidence_by_call: Sequence[Sequence[CandidateAnswerEvidenceChunk]],
    *,
    limit: int = _REGULATORY_COVERAGE_EVIDENCE_CHUNKS,
) -> list[CandidateAnswerEvidenceChunk]:
    """Retain evidence from every atomic target before taking deeper ranks."""

    selected: list[CandidateAnswerEvidenceChunk] = []
    seen: set[tuple[str, str]] = set()
    max_depth = max((len(chunks) for chunks in evidence_by_call), default=0)
    for depth in range(max_depth):
        for chunks in evidence_by_call:
            if depth >= len(chunks):
                continue
            chunk = chunks[depth]
            identity = (chunk.chunk_identifier, chunk.content)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(chunk)
            if len(selected) >= limit:
                return selected
    return selected


def _densify_regulatory_evidence_citations(
    citation_mapping: CitationMapping,
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
) -> tuple[CitationMapping, list[CandidateAnswerEvidenceChunk]]:
    """Expose one contiguous citation namespace for the bounded evidence matrix."""

    dense_citation_mapping: CitationMapping = {}
    dense_evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
    dense_number_by_document: dict[tuple[str, int], int] = {}

    for evidence_chunk in evidence_chunks:
        source_number = (
            evidence_chunk.retrieval_number or evidence_chunk.citation_number
        )
        search_doc = (
            citation_mapping.get(source_number)
            if source_number is not None
            else None
        )
        if search_doc is None:
            logger.warning(
                "Dropped regulatory evidence without a matching citation mapping: %s",
                evidence_chunk.chunk_identifier,
            )
            continue

        document_identity = (search_doc.document_id, search_doc.chunk_ind)
        dense_number = dense_number_by_document.get(document_identity)
        if dense_number is None:
            dense_number = len(dense_citation_mapping) + 1
            dense_number_by_document[document_identity] = dense_number
            dense_citation_mapping[dense_number] = search_doc

        dense_evidence_chunks.append(
            evidence_chunk.model_copy(
                update={
                    "citation_number": dense_number,
                    "retrieval_number": dense_number,
                }
            )
        )

    return dense_citation_mapping, dense_evidence_chunks


def _run_regulatory_coverage_searches(
    *,
    coverage_plan: RegulatoryCoveragePlan,
    tools: list[Tool],
    history: list[ChatMessageSimple],
    state_container: ChatStateContainer,
    turn_index: int,
) -> tuple[CitationMapping, list[CandidateAnswerEvidenceChunk]]:
    """Execute request-derived atomic retrieval without lossy sub-agent summaries."""

    search_tool = next(
        (
            tool
            for tool in tools
            if isinstance(tool, SearchTool)
            and tool.user_selected_filters is not None
            and tool.user_selected_filters.regulatory_chunks_only
        ),
        None,
    )
    if search_tool is None:
        return {}, []

    coverage_calls = _build_regulatory_coverage_tool_calls(
        coverage_plan,
        turn_index=turn_index,
    )
    if not coverage_calls:
        return {}, []

    lightweight_citation_mapping: dict[int, str] = {}
    citation_mapping: CitationMapping = {}
    evidence_by_call: list[list[CandidateAnswerEvidenceChunk]] = []

    for batch_start in range(
        0,
        len(coverage_calls),
        _REGULATORY_COVERAGE_SEARCH_BATCH_SIZE,
    ):
        batch = coverage_calls[
            batch_start : batch_start + _REGULATORY_COVERAGE_SEARCH_BATCH_SIZE
        ]
        next_citation_number = max(lightweight_citation_mapping, default=0) + 1
        batch_result = run_tool_calls(
            tool_calls=batch,
            tools=tools,
            message_history=history,
            user_memory_context=None,
            user_info=None,
            citation_mapping=lightweight_citation_mapping,
            next_citation_num=next_citation_number,
            skip_search_query_expansion=True,
            inject_memories_in_prompt=False,
            search_llm_chunks_per_call_cap=(
                _REGULATORY_COVERAGE_SEARCH_CHUNKS_PER_CALL
            ),
        )
        lightweight_citation_mapping = batch_result.updated_citation_mapping

        for tool_response in batch_result.tool_responses:
            tool_call = tool_response.tool_call
            if tool_call is None or not isinstance(
                tool_response.rich_response, SearchDocsResponse
            ):
                continue
            search_response = tool_response.rich_response
            state_container.add_search_docs(search_response.search_docs)
            state_container.add_tool_call(
                ToolCallInfo(
                    parent_tool_call_id=None,
                    turn_index=tool_call.placement.turn_index,
                    tab_index=tool_call.placement.tab_index,
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    tool_id=search_tool.id,
                    reasoning_tokens=None,
                    tool_call_arguments=tool_call.tool_args,
                    tool_call_response=tool_response.llm_facing_response,
                    search_docs=search_response.search_docs,
                    generated_images=None,
                )
            )

            search_docs_by_identity = {
                (search_doc.document_id, search_doc.chunk_ind): search_doc
                for search_doc in search_response.search_docs
            }
            for (
                citation_number,
                document_id,
            ) in search_response.citation_mapping.items():
                chunk_ind = search_response.citation_chunk_mapping.get(citation_number)
                if chunk_ind is None:
                    continue
                search_doc = search_docs_by_identity.get((document_id, chunk_ind))
                if search_doc is not None:
                    citation_mapping[citation_number] = search_doc
            evidence_by_call.append(
                [
                    evidence_chunk.model_copy(
                        update={
                            "research_target": (
                                "Specific evidence target: "
                                + str(
                                    tool_call.tool_args.get(
                                        "evidence_target", ""
                                    )
                                ).strip()
                                + ". Coverage item: "
                                + str(
                                    tool_call.tool_args.get(
                                        "coverage_item", ""
                                    )
                                ).strip()
                            )[:900].rstrip()
                        }
                    )
                    for evidence_chunk in (
                        _exact_regulatory_evidence_from_search_response(tool_response)
                    )
                ]
            )

    selected_evidence = _round_robin_coverage_evidence(evidence_by_call)
    return _densify_regulatory_evidence_citations(
        citation_mapping,
        selected_evidence,
    )


def _candidate_review_evidence(
    candidate_answer: str,
    evidence_chunks: list[CandidateAnswerEvidenceChunk],
) -> list[CandidateAnswerEvidenceChunk]:
    """Mark only citations that the current hidden candidate actually uses."""

    cited_numbers = set(extract_citation_order_from_text(candidate_answer))
    reviewed_evidence: list[CandidateAnswerEvidenceChunk] = []
    for evidence_chunk in evidence_chunks:
        available_number = (
            evidence_chunk.citation_number or evidence_chunk.retrieval_number
        )
        reviewed_evidence.append(
            evidence_chunk.model_copy(
                update={
                    "citation_number": (
                        available_number if available_number in cited_numbers else None
                    )
                }
            )
        )
    return reviewed_evidence


def _bounded_research_agent_batch(
    tool_calls: list[ToolCallKickoff],
    *,
    remaining_call_budget: int | None = None,
) -> tuple[list[ToolCallKickoff], list[str]]:
    """Select one bounded parallel batch while preserving model-chosen order."""

    batch_limit = MAX_PARALLEL_RESEARCH_AGENT_CALLS
    if remaining_call_budget is not None:
        batch_limit = min(batch_limit, max(0, remaining_call_budget))
    research_agent_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call.tool_name == RESEARCH_AGENT_TOOL_NAME
    ][:batch_limit]
    return (
        research_agent_calls,
        [tool_call.tool_call_id for tool_call in research_agent_calls],
    )


def _regulatory_research_task_rejection(
    tool_call: ToolCallKickoff,
) -> str | None:
    """Reject malformed or over-broad regulatory delegation without truncation."""

    raw_task = tool_call.tool_args.get(RESEARCH_AGENT_TASK_KEY)
    if not isinstance(raw_task, str) or not raw_task.strip():
        return (
            "This research_agent call was not run because its task is missing or "
            "empty. Retry with one focused research fragment in one or two "
            "descriptive sentences."
        )
    if len(raw_task.strip()) <= MAX_RESEARCH_AGENT_TASK_CHARS:
        return None
    return (
        "This research_agent call was not run because its task is too broad. "
        "Rewrite it as one focused research fragment using only the facts, "
        "jurisdiction, source or mechanism, and operative relationship needed "
        "to resolve that fragment. Do not copy the full user narrative."
    )


def _append_rejected_research_agent_feedback(
    history: list[ChatMessageSimple],
    rejected_calls: list[tuple[ToolCallKickoff, str]],
    token_counter: Callable[[str], int],
) -> None:
    """Pair every rejected tool call with a provider-valid retry response."""

    if not rejected_calls:
        return
    tool_calls_simple: list[ToolCallSimple] = []
    for tool_call, _ in rejected_calls:
        tool_call_message = tool_call.to_msg_str()
        tool_calls_simple.append(
            ToolCallSimple(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_arguments=tool_call.tool_args,
                token_count=token_counter(tool_call_message),
            )
        )
    history.append(
        ChatMessageSimple(
            message="",
            token_count=sum(call.token_count for call in tool_calls_simple),
            message_type=MessageType.ASSISTANT,
            tool_calls=tool_calls_simple,
            image_files=None,
        )
    )
    for tool_call, feedback in rejected_calls:
        history.append(
            ChatMessageSimple(
                message=feedback,
                token_count=token_counter(feedback),
                message_type=MessageType.TOOL_CALL_RESPONSE,
                tool_call_id=tool_call.tool_call_id,
                image_files=None,
            )
        )


def _append_unrun_research_agent_feedback(
    history: list[ChatMessageSimple],
    *,
    unrun_call_count: int,
    total_budget_exhausted: bool,
    token_counter: Callable[[str], int],
) -> None:
    """Tell the next decision about omitted calls without retaining their tasks."""

    if unrun_call_count <= 0:
        return
    if total_budget_exhausted:
        feedback = (
            f"{unrun_call_count} proposed research_agent call(s) were not run "
            "because the total research-agent call budget is exhausted. They "
            "produced no evidence; do not treat their topics as researched. "
            "Preserve any material unresolved gap in the report."
        )
    else:
        feedback = (
            f"{unrun_call_count} proposed research_agent call(s) were not run "
            "because this decision exceeded the parallel limit. They produced no "
            "evidence. After reviewing the executed results, reassess any material "
            "unresolved gap and emit a needed focused fragment again if useful."
        )
    history.append(
        ChatMessageSimple(
            message=feedback,
            token_count=token_counter(feedback),
            message_type=MessageType.USER_REMINDER,
            image_files=None,
        )
    )


def _merge_correction_citation_mapping(
    citation_mapping: CitationMapping,
    evidence_citation_mapping: CitationMapping,
) -> CitationMapping:
    """Build one correction namespace without permitting citation reassignment."""

    merged = dict(citation_mapping)
    for citation_number, search_doc in evidence_citation_mapping.items():
        existing_doc = merged.get(citation_number)
        if existing_doc is not None and (
            existing_doc.document_id,
            existing_doc.chunk_ind,
        ) != (search_doc.document_id, search_doc.chunk_ind):
            raise ValueError(
                "Conflicting correction citation mapping for citation "
                f"{citation_number}"
            )
        merged[citation_number] = search_doc
    return merged


def _bounded_correction_citation_mapping(
    formatted_evidence: str,
    correction_citation_mapping: CitationMapping,
) -> CitationMapping:
    """Expose only source numbers included in the bounded correction payload."""

    try:
        payload = json.loads(formatted_evidence)
    except (TypeError, json.JSONDecodeError):
        logger.warning("Could not parse bounded regulatory correction evidence")
        return {}

    if not isinstance(payload, dict):
        return {}
    evidence_chunks = payload.get("evidence_chunks")
    if not isinstance(evidence_chunks, list):
        return {}

    bounded_numbers: set[int] = set()
    for evidence_chunk in evidence_chunks:
        if not isinstance(evidence_chunk, dict):
            continue
        citation_number = evidence_chunk.get("citation_number")
        if type(citation_number) is int and citation_number > 0:
            bounded_numbers.add(citation_number)
    missing_numbers = bounded_numbers - correction_citation_mapping.keys()
    if missing_numbers:
        logger.warning(
            "Bounded regulatory correction evidence referenced unavailable source "
            "numbers: %s",
            sorted(missing_numbers),
        )
    return {
        citation_number: correction_citation_mapping[citation_number]
        for citation_number in sorted(bounded_numbers)
        if citation_number in correction_citation_mapping
    }


def _stage_regulatory_source_gap_fallback(
    turn_index: int,
) -> tuple[
    BufferedEmitter,
    ChatStateContainer,
    DynamicCitationProcessor,
    LlmStepResult,
]:
    """Create a deterministic citation-free answer after a confirmed final defect."""

    fallback_emitter = BufferedEmitter()
    fallback_state = ChatStateContainer()
    fallback_citation_processor = DynamicCitationProcessor()
    fallback_state.set_answer_tokens(_REGULATORY_SOURCE_GAP_FALLBACK)
    fallback_state.set_pre_answer_processing_time(0)
    placement = Placement(turn_index=turn_index)
    fallback_emitter.emit(
        Packet(
            placement=placement,
            obj=AgentResponseStart(
                final_documents=[],
                pre_answer_processing_seconds=0,
            ),
        )
    )
    fallback_emitter.emit(
        Packet(
            placement=placement,
            obj=AgentResponseDelta(content=_REGULATORY_SOURCE_GAP_FALLBACK),
        )
    )
    return (
        fallback_emitter,
        fallback_state,
        fallback_citation_processor,
        LlmStepResult(
            reasoning=None,
            answer=_REGULATORY_SOURCE_GAP_FALLBACK,
            raw_answer=_REGULATORY_SOURCE_GAP_FALLBACK,
            tool_calls=None,
            finish_reason="stop",
        ),
    )


def _format_earlier_user_context_for_correction(
    earlier_user_context: tuple[str, ...],
    *,
    max_chars: int = 12_000,
) -> str:
    """Keep recent user facts bounded and distinct from current deliverables."""

    selected_reversed: list[str] = []
    for message in reversed(earlier_user_context):
        candidate_reversed = [*selected_reversed, message]
        projected = json.dumps(
            list(reversed(candidate_reversed)),
            ensure_ascii=False,
        )
        if len(projected) > max_chars:
            break
        selected_reversed.append(message)
    if not selected_reversed and earlier_user_context:
        latest = earlier_user_context[-1]
        omission = "\n[... earlier user context omitted ...]\n"
        retained_chars = max_chars - len(omission) - 4
        leading_chars = retained_chars // 2
        trailing_chars = retained_chars - leading_chars
        selected_reversed.append(
            latest[:leading_chars] + omission + latest[-trailing_chars:]
        )
    if not selected_reversed:
        return ""
    return json.dumps(
        {
            "usage_note": (
                "Reference facts and antecedents only; these are not additional "
                "deliverables. The current user request controls any conflict."
            ),
            "messages": list(reversed(selected_reversed)),
        },
        ensure_ascii=False,
    )


def _compact_regulatory_final_retry_history(
    *,
    history: list[ChatMessageSimple],
    research_plan: str,
    state_container: ChatStateContainer,
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk],
    system_prompt: ChatMessageSimple,
    custom_agent_prompt_message: ChatMessageSimple | None,
    token_counter: Callable[[str], int],
    available_tokens: int,
) -> list[ChatMessageSimple] | None:
    """Flatten research evidence so the retry has no tool-call protocol history."""
    reports = [
        tool_call.tool_call_response.strip()[:MAX_REGULATORY_RETRY_REPORT_CHARS]
        for tool_call in state_container.get_tool_calls()
        if tool_call.tool_name == RESEARCH_AGENT_TOOL_NAME
        and tool_call.tool_call_response.strip()
    ][:MAX_REGULATORY_RETRY_REPORTS]
    exact_evidence = (
        format_candidate_correction_evidence(exact_evidence_chunks)[
            :MAX_REGULATORY_RETRY_EVIDENCE_CHARS
        ]
        if exact_evidence_chunks
        else ""
    )
    if not reports and not exact_evidence:
        return None

    review_context = build_regulatory_review_user_context(history)
    report_sections = "\n\n".join(
        f"## Completed research report {index}\n{report}"
        for index, report in enumerate(reports, start=1)
    )
    retry_payload = (
        "# Current request\n"
        f"{review_context.current_request}\n\n"
        "# Research plan\n"
        f"{research_plan}\n\n"
        "# Completed research\n"
        f"{report_sections or 'No narrative research report was produced.'}\n\n"
        "# Exact retrieved evidence\n"
        f"{exact_evidence or 'No exact evidence chunk was retrieved.'}\n\n"
        "Produce the final answer now. Use only the material above, preserve valid "
        "citation numbers, explicitly identify unsupported points, and do not call tools."
    )
    flattened_history = [
        ChatMessageSimple(
            message=retry_payload,
            token_count=token_counter(retry_payload),
            message_type=MessageType.USER,
        )
    ]
    return construct_message_history(
        system_prompt=system_prompt,
        custom_agent_prompt=custom_agent_prompt_message,
        simple_chat_history=flattened_history,
        reminder_message=None,
        context_files=None,
        available_tokens=available_tokens,
    )


def generate_final_report(
    history: list[ChatMessageSimple],
    research_plan: str,
    llm: LLM,
    token_counter: Callable[[str], int],
    state_container: ChatStateContainer,
    emitter: Emitter,
    turn_index: int,
    citation_mapping: CitationMapping,
    is_regulatory_research: bool,
    user_identity: LLMUserIdentity | None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    saved_reasoning: str | None = None,
    pre_answer_processing_time: float | None = None,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] | None = None,
    evidence_citation_mapping: CitationMapping | None = None,
    recovery_tools: Sequence[Tool] | None = None,
    custom_agent_prompt: str | None = None,
) -> bool:
    """Generate the final research report.

    Returns:
        bool: True if reasoning occurred during report generation (turn_index was incremented),
              False otherwise.
    """
    with function_span("generate_report") as span:
        span.span_data.input = (
            f"history_length={len(history)}, turn_index={turn_index}, "
            f"requested_reasoning_effort={reasoning_effort.value}"
        )
        final_report_prompt = FINAL_REPORT_PROMPT.format(
            current_datetime=get_current_llm_day_time(full_sentence=False),
        )
        system_prompt = ChatMessageSimple(
            message=final_report_prompt,
            token_count=token_counter(final_report_prompt),
            message_type=MessageType.SYSTEM,
        )
        custom_agent_prompt_message = _custom_agent_prompt_message(
            custom_agent_prompt,
            token_counter,
        )
        final_reminder = USER_FINAL_REPORT_QUERY.format(research_plan=research_plan)
        reminder_message = ChatMessageSimple(
            message=final_reminder,
            token_count=token_counter(final_reminder),
            message_type=MessageType.USER_REMINDER,
        )
        final_report_history = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent_prompt_message,
            simple_chat_history=history,
            reminder_message=reminder_message,
            context_files=None,
            available_tokens=llm.config.max_input_tokens,
            all_injected_file_metadata=all_injected_file_metadata,
        )

        citation_processor = DynamicCitationProcessor()
        citation_processor.update_citation_mapping(citation_mapping)
        available_exact_evidence_chunks = list(exact_evidence_chunks or [])
        correction_citation_mapping = _merge_correction_citation_mapping(
            citation_mapping,
            evidence_citation_mapping or {},
        )

        # Only passing in the cited documents as the whole list would be too long
        final_documents = list(citation_processor.citation_to_doc.values())

        staged_started_at = time.monotonic()
        compact_retry_history = (
            _compact_regulatory_final_retry_history(
                history=history,
                research_plan=research_plan,
                state_container=state_container,
                exact_evidence_chunks=list(exact_evidence_chunks or []),
                system_prompt=system_prompt,
                custom_agent_prompt_message=custom_agent_prompt_message,
                token_counter=token_counter,
                available_tokens=llm.config.max_input_tokens,
            )
            if is_regulatory_research
            else None
        )
        try:
            initial_attempt = _generate_usable_final_report_attempt(
                history=final_report_history,
                llm=llm,
                base_citation_processor=citation_processor,
                final_documents=final_documents,
                user_identity=user_identity,
                turn_index=turn_index,
                max_tokens=(
                    MAX_REGULATORY_FINAL_REPORT_TOKENS
                    if is_regulatory_research
                    else MAX_FINAL_REPORT_TOKENS
                ),
                pre_answer_processing_time=pre_answer_processing_time,
                retry_history=compact_retry_history,
            )
        except EmptyLLMResponseError:
            if not is_regulatory_research:
                raise
            logger.warning(
                "Regulatory final synthesis attempts were empty; publishing a "
                "bounded source-gap response"
            )
            (
                fallback_emitter,
                fallback_state,
                fallback_citation_processor,
                fallback_result,
            ) = _stage_regulatory_source_gap_fallback(turn_index)
            commit_staged_llm_step(
                buffered_emitter=fallback_emitter,
                staged_state=fallback_state,
                staged_citation_processor=fallback_citation_processor,
                emitter=emitter,
                state_container=state_container,
                pre_answer_processing_time=(
                    (pre_answer_processing_time or 0.0)
                    + time.monotonic()
                    - staged_started_at
                ),
                final_documents_from_emitted_citations=True,
            )
            if saved_reasoning:
                state_container.set_reasoning_tokens(saved_reasoning)
            span.span_data.output = fallback_result.answer
            return False

        if not is_regulatory_research:
            total_pre_answer_processing_time = (
                (pre_answer_processing_time or 0.0)
                + time.monotonic()
                - staged_started_at
            )
            commit_staged_llm_step(
                buffered_emitter=initial_attempt.emitter,
                staged_state=initial_attempt.state,
                staged_citation_processor=initial_attempt.citation_processor,
                emitter=emitter,
                state_container=state_container,
                pre_answer_processing_time=total_pre_answer_processing_time,
            )
            if saved_reasoning:
                state_container.set_reasoning_tokens(saved_reasoning)
            span.span_data.output = initial_attempt.result.answer
            return initial_attempt.has_reasoned

        initial_emitter = initial_attempt.emitter
        initial_state = initial_attempt.state
        initial_citation_processor = initial_attempt.citation_processor
        initial_result = initial_attempt.result
        initial_has_reasoned = initial_attempt.has_reasoned
        initial_answer = initial_result.answer
        initial_candidate = initial_result.raw_answer or initial_answer or ""

        accepted_emitter = initial_emitter
        accepted_state = initial_state
        accepted_citation_processor = initial_citation_processor
        accepted_result = initial_result
        accepted_has_reasoned = initial_has_reasoned

        review_user_context = build_regulatory_review_user_context(history)
        user_request = review_user_context.current_request
        earlier_user_context = review_user_context.earlier_user_context
        formatted_earlier_user_context = _format_earlier_user_context_for_correction(
            earlier_user_context
        )
        earlier_user_context_section = (
            "\n\n# Earlier user context (reference only)\n"
            f"{formatted_earlier_user_context}\n"
            "Only the current user request defines this answer's deliverables.\n\n"
            if formatted_earlier_user_context
            else ""
        )
        review_evidence = _candidate_review_evidence(
            initial_candidate,
            available_exact_evidence_chunks,
        )
        review_llm = build_regulatory_review_llm(llm)

        if user_request.strip():
            candidate_review = review_regulatory_candidate_answer(
                review_llm,
                user_request=user_request,
                earlier_user_context=earlier_user_context,
                candidate_answer=initial_candidate,
                evidence_chunks=review_evidence,
                coverage_contract=research_plan,
            )
            recovery_issue = select_priority_recovery_issue(candidate_review)
            internal_recovery_tool = next(
                (
                    tool
                    for tool in recovery_tools or []
                    if isinstance(tool, SearchTool)
                    and tool.user_selected_filters is not None
                    and tool.user_selected_filters.regulatory_chunks_only
                ),
                None,
            )
            if recovery_issue is not None and internal_recovery_tool is not None:
                recovery_placement = Placement(turn_index=turn_index, tab_index=0)
                starting_citation_num = (
                    max(
                        [
                            0,
                            *citation_mapping.keys(),
                            *(evidence_citation_mapping or {}).keys(),
                        ]
                    )
                    + 1
                )
                try:
                    recovery_response = run_single_gap_recovery(
                        search_tool=internal_recovery_tool,
                        issue=recovery_issue,
                        starting_citation_num=starting_citation_num,
                        placement=recovery_placement,
                    )
                    recovered_docs = recovery_search_docs_by_citation(recovery_response)
                    recovered_evidence = exact_recovery_evidence_chunks(
                        recovery_response
                    )
                    citation_mapping = merge_recovery_citation_mapping(
                        citation_mapping,
                        recovered_docs,
                    )
                    evidence_citation_mapping = merge_recovery_citation_mapping(
                        evidence_citation_mapping or {},
                        recovered_docs,
                    )
                    available_exact_evidence_chunks = merge_recovery_evidence_chunks(
                        available_exact_evidence_chunks,
                        recovered_evidence,
                    )
                    review_evidence = _candidate_review_evidence(
                        initial_candidate,
                        available_exact_evidence_chunks,
                    )
                    correction_citation_mapping = _merge_correction_citation_mapping(
                        citation_mapping,
                        evidence_citation_mapping,
                    )

                    rich_response = recovery_response.rich_response
                    if isinstance(rich_response, SearchDocsResponse):
                        state_container.add_search_docs(rich_response.search_docs)
                        recovery_call = ToolCallKickoff(
                            tool_call_id=f"regulatory-gap-recovery-{turn_index}",
                            tool_name=SearchTool.NAME,
                            tool_args={
                                "queries": [recovery_issue.recovery_query],
                                "search_mode": "hybrid",
                            },
                            placement=recovery_placement,
                        )
                        state_container.add_tool_call(
                            ToolCallInfo(
                                parent_tool_call_id=None,
                                turn_index=turn_index,
                                tab_index=0,
                                tool_name=SearchTool.NAME,
                                tool_call_id=recovery_call.tool_call_id,
                                tool_id=internal_recovery_tool.id,
                                reasoning_tokens=None,
                                tool_call_arguments=recovery_call.tool_args,
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
                except Exception:
                    logger.exception(
                        "Single regulatory evidence-gap search failed; "
                        "continuing with the existing correction path"
                    )
            review_feedback = format_candidate_answer_review(candidate_review)
            if candidate_review.needs_reconsideration and review_feedback is not None:
                correction_evidence = format_candidate_correction_evidence(
                    review_evidence,
                )
                bounded_correction_citation_mapping = (
                    _bounded_correction_citation_mapping(
                        correction_evidence,
                        correction_citation_mapping,
                    )
                )
                correction_reminder = (
                    f"{review_feedback}\n\n"
                    f"{earlier_user_context_section}"
                    "# Exact evidence available for this correction\n"
                    f"{correction_evidence}\n\n"
                    "No additional retrieval is available in this bounded correction "
                    "pass. Produce one corrected final report. Preserve supported "
                    "analysis and citation numbers; resolve each material concern from "
                    "the exact evidence, or expressly qualify the conclusion or source "
                    "gap."
                )
                correction_history = [
                    system_prompt,
                    *(
                        [custom_agent_prompt_message]
                        if custom_agent_prompt_message is not None
                        else []
                    ),
                    ChatMessageSimple(
                        message=user_request,
                        token_count=token_counter(user_request),
                        message_type=MessageType.USER,
                    ),
                    ChatMessageSimple(
                        message=initial_candidate,
                        token_count=token_counter(initial_candidate),
                        message_type=MessageType.ASSISTANT,
                    ),
                    ChatMessageSimple(
                        message=correction_reminder,
                        token_count=token_counter(correction_reminder),
                        message_type=MessageType.USER_REMINDER,
                    ),
                ]
                correction_base_citation_processor = DynamicCitationProcessor()
                correction_base_citation_processor.update_citation_mapping(
                    bounded_correction_citation_mapping
                )
                correction_final_documents = list(
                    bounded_correction_citation_mapping.values()
                )
                correction_attempt = _generate_usable_final_report_attempt(
                    history=correction_history,
                    llm=llm,
                    base_citation_processor=correction_base_citation_processor,
                    final_documents=correction_final_documents,
                    user_identity=user_identity,
                    turn_index=turn_index,
                    max_tokens=MAX_REGULATORY_FINAL_REPORT_TOKENS,
                    pre_answer_processing_time=0,
                )
                correction_candidate = (
                    correction_attempt.result.raw_answer
                    or correction_attempt.result.answer
                    or ""
                )
                correction_review_evidence = _candidate_review_evidence(
                    correction_candidate,
                    available_exact_evidence_chunks,
                )
                try:
                    resolution_review = review_regulatory_candidate_resolution(
                        review_llm,
                        candidate_answer=correction_candidate,
                        prior_issues=candidate_review.advisory_claim_issues,
                        evidence_chunks=correction_review_evidence,
                    )
                except Exception:
                    logger.exception(
                        "Regulatory correction review failed; publishing a "
                        "citation-free source-gap response"
                    )
                    (
                        accepted_emitter,
                        accepted_state,
                        accepted_citation_processor,
                        accepted_result,
                    ) = _stage_regulatory_source_gap_fallback(turn_index)
                    accepted_has_reasoned = False
                else:
                    resolution_feedback = format_candidate_resolution_review(
                        resolution_review
                    )
                    if resolution_feedback is None:
                        accepted_emitter = correction_attempt.emitter
                        accepted_state = correction_attempt.state
                        accepted_citation_processor = (
                            correction_attempt.citation_processor
                        )
                        accepted_result = correction_attempt.result
                        accepted_has_reasoned = correction_attempt.has_reasoned
                    else:
                        final_correction_reminder = (
                            f"{resolution_feedback}\n\n"
                            f"{earlier_user_context_section}"
                            "# Exact evidence available for this final correction\n"
                            f"{correction_evidence}\n\n"
                            "No additional retrieval or correction pass is available. "
                            "Produce one final corrected report using only the exact "
                            "evidence above. Resolve every remaining issue, or remove "
                            "or accurately qualify the affected proposition and state "
                            "the precise source gap."
                        )
                        final_correction_history = [
                            system_prompt,
                            *(
                                [custom_agent_prompt_message]
                                if custom_agent_prompt_message is not None
                                else []
                            ),
                            ChatMessageSimple(
                                message=user_request,
                                token_count=token_counter(user_request),
                                message_type=MessageType.USER,
                            ),
                            ChatMessageSimple(
                                message=correction_candidate,
                                token_count=token_counter(correction_candidate),
                                message_type=MessageType.ASSISTANT,
                            ),
                            ChatMessageSimple(
                                message=final_correction_reminder,
                                token_count=token_counter(final_correction_reminder),
                                message_type=MessageType.USER_REMINDER,
                            ),
                        ]
                        final_correction_base_citation_processor = (
                            DynamicCitationProcessor()
                        )
                        final_correction_base_citation_processor.update_citation_mapping(
                            bounded_correction_citation_mapping
                        )
                        final_correction_attempt = (
                            _generate_usable_final_report_attempt(
                                history=final_correction_history,
                                llm=llm,
                                base_citation_processor=(
                                    final_correction_base_citation_processor
                                ),
                                final_documents=correction_final_documents,
                                user_identity=user_identity,
                                turn_index=turn_index,
                                max_tokens=MAX_REGULATORY_FINAL_REPORT_TOKENS,
                                pre_answer_processing_time=0,
                            )
                        )
                        final_correction_candidate = (
                            final_correction_attempt.result.raw_answer
                            or final_correction_attempt.result.answer
                            or ""
                        )
                        try:
                            final_resolution_review = (
                                review_regulatory_candidate_resolution(
                                    review_llm,
                                    candidate_answer=final_correction_candidate,
                                    prior_issues=(
                                        resolution_review.advisory_claim_issues
                                    ),
                                    evidence_chunks=_candidate_review_evidence(
                                        final_correction_candidate,
                                        available_exact_evidence_chunks,
                                    ),
                                )
                            )
                        except Exception:
                            logger.exception(
                                "Final regulatory correction review failed; publishing "
                                "a citation-free source-gap response"
                            )
                            (
                                accepted_emitter,
                                accepted_state,
                                accepted_citation_processor,
                                accepted_result,
                            ) = _stage_regulatory_source_gap_fallback(turn_index)
                            accepted_has_reasoned = False
                        else:
                            final_resolution_feedback = (
                                format_candidate_resolution_review(
                                    final_resolution_review
                                )
                            )
                            if final_resolution_feedback is None:
                                accepted_emitter = final_correction_attempt.emitter
                                accepted_state = final_correction_attempt.state
                                accepted_citation_processor = (
                                    final_correction_attempt.citation_processor
                                )
                                accepted_result = final_correction_attempt.result
                                accepted_has_reasoned = (
                                    final_correction_attempt.has_reasoned
                                )
                            else:
                                logger.warning(
                                    "Final regulatory correction remained materially "
                                    "unresolved; publishing a citation-free source-gap "
                                    "response"
                                )
                                (
                                    accepted_emitter,
                                    accepted_state,
                                    accepted_citation_processor,
                                    accepted_result,
                                ) = _stage_regulatory_source_gap_fallback(turn_index)
                                accepted_has_reasoned = False
        else:
            logger.warning(
                "Skipped regulatory deep-research review because exact evidence "
                "or the user request was unavailable"
            )

        total_pre_answer_processing_time = (
            (pre_answer_processing_time or 0.0) + time.monotonic() - staged_started_at
        )
        commit_staged_llm_step(
            buffered_emitter=accepted_emitter,
            staged_state=accepted_state,
            staged_citation_processor=accepted_citation_processor,
            emitter=emitter,
            state_container=state_container,
            pre_answer_processing_time=total_pre_answer_processing_time,
            final_documents_from_emitted_citations=True,
        )
        if saved_reasoning:
            state_container.set_reasoning_tokens(saved_reasoning)

        span.span_data.output = accepted_result.answer
        return accepted_has_reasoned


def _get_research_agent_tool_id() -> int:
    with get_session_with_current_tenant() as db_session:
        return get_tool_by_name(
            tool_name=RESEARCH_AGENT_TOOL_NAME,
            db_session=db_session,
        ).id


@log_function_time(print_only=True)
def run_deep_research_llm_loop(
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    custom_agent_prompt: str | None,
    llm: LLM,
    token_counter: Callable[[str], int],
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    skip_clarification: bool = False,
    user_identity: LLMUserIdentity | None = None,
    chat_session_id: str | None = None,
    all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
) -> None:
    with trace(
        "run_deep_research_llm_loop",
        group_id=chat_session_id,
        metadata=ChatTraceMetadata(
            chat_session_id=chat_session_id,
            user_id=user_identity.user_id if user_identity else None,
        ).model_dump(),
    ):
        # Here for lazy load LiteLLM
        from onyx.llm.litellm_singleton.config import initialize_litellm

        # An approximate limit. In extreme cases it may still fail but this should allow deep research
        # to work in most cases.
        if llm.config.max_input_tokens < 50000:
            raise RuntimeError(
                "Cannot run Deep Research with an LLM that has less than 50,000 max input tokens"
            )

        initialize_litellm()

        # Track processing start time for tool duration calculation
        processing_start_time = time.monotonic()

        available_tokens = llm.config.max_input_tokens
        custom_agent_prompt_msg = _custom_agent_prompt_message(
            custom_agent_prompt,
            token_counter,
        )

        llm_step_result: LlmStepResult | None = None

        # This deployment's research boundary is the indexed internal corpus.
        # Keep the allowlist here as well as in the global tool registry so a
        # future registry change cannot silently grant web access to research.
        allowed_tools = _deep_research_search_tools(tools)
        include_internal_search_tunings = bool(allowed_tools)
        is_regulatory_deep_research = any(
            isinstance(tool, SearchTool)
            and tool.user_selected_filters is not None
            and tool.user_selected_filters.regulatory_chunks_only
            for tool in allowed_tools
        )
        regulatory_coverage_plan: RegulatoryCoveragePlan | None = None
        regulatory_coverage_contract: str | None = None
        if is_regulatory_deep_research:
            regulatory_review_context = build_regulatory_review_user_context(
                simple_chat_history
            )
            regulatory_review_llm = build_regulatory_review_llm(llm)
            regulatory_coverage_plan = build_regulatory_coverage_plan(
                regulatory_review_llm,
                user_request=regulatory_review_context.current_request,
            )
            regulatory_coverage_contract = format_regulatory_coverage_plan(
                regulatory_coverage_plan
            )
        exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] = []
        exact_evidence_indexes: dict[tuple[str, str], int] = {}
        initial_regulatory_citation_mapping: CitationMapping = {}
        orchestrator_start_turn_index = 1

        #########################################################
        # CLARIFICATION STEP (optional)
        #########################################################
        internal_search_clarification_guidance = (
            INTERNAL_SEARCH_CLARIFICATION_GUIDANCE
            if include_internal_search_tunings
            else ""
        )
        if not SKIP_DEEP_RESEARCH_CLARIFICATION and not skip_clarification:
            with function_span("clarification_step") as span:
                clarification_prompt = CLARIFICATION_PROMPT.format(
                    current_datetime=get_current_llm_day_time(full_sentence=False),
                    internal_search_clarification_guidance=internal_search_clarification_guidance,
                )
                system_prompt = ChatMessageSimple(
                    message=clarification_prompt,
                    token_count=300,  # Skips the exact token count but has enough leeway
                    message_type=MessageType.SYSTEM,
                )

                truncated_message_history = construct_message_history(
                    system_prompt=system_prompt,
                    custom_agent_prompt=custom_agent_prompt_msg,
                    simple_chat_history=simple_chat_history,
                    reminder_message=None,
                    context_files=None,
                    available_tokens=available_tokens,
                    last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT,
                    all_injected_file_metadata=all_injected_file_metadata,
                )

                # Calculate tool processing duration for clarification step
                # (used if the LLM emits a clarification question instead of calling tools)
                clarification_tool_duration = time.monotonic() - processing_start_time
                llm_step_result, _ = run_llm_step(
                    emitter=emitter,
                    history=truncated_message_history,
                    tool_definitions=get_clarification_tool_definitions(),
                    tool_choice=ToolChoiceOptions.AUTO,
                    llm=llm,
                    reasoning_effort=reasoning_effort,
                    placement=Placement(turn_index=0),
                    # No citations in this step, it should just pass through all
                    # tokens directly so initialized as an empty citation processor
                    citation_processor=None,
                    state_container=state_container,
                    final_documents=None,
                    user_identity=user_identity,
                    max_tokens=(
                        MAX_CLARIFICATION_TOKENS
                        if is_regulatory_deep_research
                        else None
                    ),
                    is_deep_research=True,
                    pre_answer_processing_time=clarification_tool_duration,
                )

                if not llm_step_result.tool_calls:
                    # Mark this turn as a clarification question
                    state_container.set_is_clarification(True)
                    span.span_data.output = "clarification_required"

                    emitter.emit(
                        Packet(
                            placement=Placement(turn_index=0),
                            obj=OverallStop(type="stop"),
                        )
                    )

                    # If a clarification is asked, we need to end this turn and wait on user input
                    return

        if regulatory_coverage_contract is not None:
            simple_chat_history.append(
                ChatMessageSimple(
                    message=regulatory_coverage_contract,
                    token_count=token_counter(regulatory_coverage_contract),
                    message_type=MessageType.USER_REMINDER,
                )
            )

        if (
            is_regulatory_deep_research
            and regulatory_coverage_plan is not None
            and regulatory_coverage_contract is not None
        ):
            (
                initial_regulatory_citation_mapping,
                exact_evidence_chunks,
            ) = _run_regulatory_coverage_searches(
                coverage_plan=regulatory_coverage_plan,
                tools=allowed_tools,
                history=simple_chat_history,
                state_container=state_container,
                turn_index=1,
            )
            exact_evidence_indexes = {
                (evidence.chunk_identifier, evidence.content): index
                for index, evidence in enumerate(exact_evidence_chunks)
            }
            if exact_evidence_chunks:
                evidence_matrix = (
                    "# Retrieved exact evidence matrix\n"
                    + format_candidate_correction_evidence(exact_evidence_chunks)
                    + "\n\nThis is bootstrap evidence, not proof that every material "
                    "coverage row is closed. Use the research phase to resolve "
                    "missing, indirect, incomplete, or conflicting rows."
                )
                simple_chat_history.append(
                    ChatMessageSimple(
                        message=evidence_matrix,
                        token_count=token_counter(evidence_matrix),
                        message_type=MessageType.USER_REMINDER,
                    )
                )

        #########################################################
        # RESEARCH PLAN STEP
        #########################################################
        with function_span("research_plan_step") as span:
            system_prompt = ChatMessageSimple(
                message=RESEARCH_PLAN_PROMPT.format(
                    current_datetime=get_current_llm_day_time(full_sentence=False)
                ),
                token_count=300,
                message_type=MessageType.SYSTEM,
            )
            # Note this is fine to use a USER message type here as it can just be interpretered as a
            # user's message directly to the LLM.
            reminder_message = ChatMessageSimple(
                message=RESEARCH_PLAN_REMINDER,
                token_count=100,
                message_type=MessageType.USER,
            )
            truncated_message_history = construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=custom_agent_prompt_msg,
                simple_chat_history=simple_chat_history + [reminder_message],
                reminder_message=None,
                context_files=None,
                available_tokens=available_tokens,
                last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT + 1,
                all_injected_file_metadata=all_injected_file_metadata,
            )

            research_plan_generator = run_llm_step_pkt_generator(
                history=truncated_message_history,
                tool_definitions=[],
                tool_choice=ToolChoiceOptions.NONE,
                llm=llm,
                reasoning_effort=reasoning_effort,
                placement=Placement(turn_index=0),
                citation_processor=None,
                state_container=state_container,
                final_documents=None,
                user_identity=user_identity,
                max_tokens=(
                    MAX_RESEARCH_PLAN_TOKENS if is_regulatory_deep_research else None
                ),
                is_deep_research=True,
            )

            while True:
                try:
                    packet = next(research_plan_generator)
                    # Translate AgentResponseStart/Delta packets to DeepResearchPlanStart/Delta
                    # The LLM response from this prompt is the research plan
                    if isinstance(packet.obj, AgentResponseStart):
                        emitter.emit(
                            Packet(
                                placement=packet.placement,
                                obj=DeepResearchPlanStart(),
                            )
                        )
                    elif isinstance(packet.obj, AgentResponseDelta):
                        emitter.emit(
                            Packet(
                                placement=packet.placement,
                                obj=DeepResearchPlanDelta(content=packet.obj.content),
                            )
                        )
                    else:
                        # Pass through other packet types (e.g., ReasoningStart, ReasoningDelta, etc.)
                        emitter.emit(packet)
                except StopIteration as e:
                    llm_step_result, reasoned = e.value
                    emitter.emit(
                        Packet(
                            # Marks the last turn end which should be the plan generation
                            placement=Placement(
                                turn_index=1 if reasoned else 0,
                            ),
                            obj=SectionEnd(),
                        )
                    )
                    if reasoned:
                        orchestrator_start_turn_index += 1
                    break
            llm_step_result = cast(LlmStepResult, llm_step_result)

            research_plan = llm_step_result.answer
            if research_plan is None:
                raise RuntimeError("Deep Research failed to generate a research plan")
            if regulatory_coverage_contract is not None:
                research_plan = (
                    f"{regulatory_coverage_contract}\n\n"
                    "# Adaptive research plan\n"
                    f"{research_plan}"
                )
            span.span_data.output = research_plan if research_plan else None

        #########################################################
        # RESEARCH EXECUTION STEP
        #########################################################
        with function_span("research_execution_step") as span:
            is_reasoning_model = model_is_reasoning_model(
                llm.config.model_name, llm.config.model_provider
            )

            max_orchestrator_cycles = (
                MAX_ORCHESTRATOR_CYCLES
                if not is_reasoning_model
                else MAX_ORCHESTRATOR_CYCLES_REASONING
            )

            orchestrator_prompt_template = (
                ORCHESTRATOR_PROMPT
                if not is_reasoning_model
                else ORCHESTRATOR_PROMPT_REASONING
            )

            internal_search_research_task_guidance = (
                INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE
                if include_internal_search_tunings
                else ""
            )
            token_count_prompt = orchestrator_prompt_template.format(
                current_datetime=get_current_llm_day_time(full_sentence=False),
                current_cycle_count=1,
                max_cycles=max_orchestrator_cycles,
                research_plan=research_plan,
                internal_search_research_task_guidance=internal_search_research_task_guidance,
            )
            orchestration_tokens = token_counter(token_count_prompt)

            reasoning_cycles = 0
            most_recent_reasoning: str | None = None
            citation_mapping: CitationMapping = dict(
                initial_regulatory_citation_mapping
            )
            evidence_citation_mapping: CitationMapping = dict(
                initial_regulatory_citation_mapping
            )
            research_agent_calls_started = 0
            final_turn_index: int = orchestrator_start_turn_index  # Track the final turn_index for stop packet
            for cycle in _orchestrator_cycle_schedule(max_orchestrator_cycles):
                # The extra schedule entry guarantees a report after all advertised
                # decision cycles if the model did not choose to report earlier.
                elapsed_seconds = time.monotonic() - processing_start_time
                timed_out = elapsed_seconds > DEEP_RESEARCH_FORCE_REPORT_SECONDS
                is_forced_report_cycle = cycle == max_orchestrator_cycles

                if timed_out or is_forced_report_cycle:
                    if timed_out:
                        logger.info(
                            "Deep research exceeded %ss (elapsed: %ss), forcing final report generation",
                            DEEP_RESEARCH_FORCE_REPORT_SECONDS,
                            format(elapsed_seconds, ".1f"),
                        )
                    report_turn_index = (
                        orchestrator_start_turn_index + cycle + reasoning_cycles
                    )
                    report_reasoned = generate_final_report(
                        history=simple_chat_history,
                        research_plan=research_plan,
                        llm=llm,
                        token_counter=token_counter,
                        state_container=state_container,
                        emitter=emitter,
                        turn_index=report_turn_index,
                        citation_mapping=citation_mapping,
                        is_regulatory_research=is_regulatory_deep_research,
                        user_identity=user_identity,
                        reasoning_effort=reasoning_effort,
                        pre_answer_processing_time=elapsed_seconds,
                        all_injected_file_metadata=all_injected_file_metadata,
                        exact_evidence_chunks=(
                            exact_evidence_chunks
                            if is_regulatory_deep_research
                            else None
                        ),
                        evidence_citation_mapping=(
                            evidence_citation_mapping
                            if is_regulatory_deep_research
                            else None
                        ),
                        recovery_tools=allowed_tools,
                        custom_agent_prompt=custom_agent_prompt,
                    )
                    final_turn_index = report_turn_index + (1 if report_reasoned else 0)
                    break

                if cycle == 1:
                    first_cycle_reminder_message = ChatMessageSimple(
                        message=FIRST_CYCLE_REMINDER,
                        token_count=FIRST_CYCLE_REMINDER_TOKENS,
                        message_type=MessageType.USER_REMINDER,
                    )
                else:
                    first_cycle_reminder_message = None

                orchestrator_prompt = orchestrator_prompt_template.format(
                    current_datetime=get_current_llm_day_time(full_sentence=False),
                    current_cycle_count=cycle,
                    max_cycles=max_orchestrator_cycles,
                    research_plan=research_plan,
                    internal_search_research_task_guidance=internal_search_research_task_guidance,
                )

                system_prompt = ChatMessageSimple(
                    message=orchestrator_prompt,
                    token_count=orchestration_tokens,
                    message_type=MessageType.SYSTEM,
                )

                truncated_message_history = construct_message_history(
                    system_prompt=system_prompt,
                    custom_agent_prompt=custom_agent_prompt_msg,
                    simple_chat_history=simple_chat_history,
                    reminder_message=first_cycle_reminder_message,
                    context_files=None,
                    available_tokens=available_tokens,
                    last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT,
                    all_injected_file_metadata=all_injected_file_metadata,
                )

                # Use think tool processor for non-reasoning models to convert
                # think_tool calls to reasoning content
                custom_processor = (
                    create_think_tool_token_processor()
                    if not is_reasoning_model
                    else None
                )

                llm_step_result, has_reasoned = run_llm_step(
                    emitter=emitter,
                    history=truncated_message_history,
                    tool_definitions=get_orchestrator_tools(
                        include_think_tool=not is_reasoning_model
                    ),
                    tool_choice=ToolChoiceOptions.REQUIRED,
                    llm=llm,
                    reasoning_effort=reasoning_effort,
                    placement=Placement(
                        turn_index=orchestrator_start_turn_index
                        + cycle
                        + reasoning_cycles
                    ),
                    # No citations in this step, it should just pass through all
                    # tokens directly so initialized as an empty citation processor
                    citation_processor=DynamicCitationProcessor(),
                    state_container=state_container,
                    final_documents=None,
                    user_identity=user_identity,
                    custom_token_processor=custom_processor,
                    is_deep_research=True,
                    # Even for the reasoning tool, this should be plenty
                    # The generation here should never be very long as it's just the tool calls.
                    # This prevents timeouts where the model gets into an endless loop of null or bad tokens.
                    max_tokens=1024,
                )
                if has_reasoned:
                    reasoning_cycles += 1

                tool_calls = llm_step_result.tool_calls or []

                if not tool_calls and cycle == 0:
                    raise RuntimeError(
                        "Deep Research failed to generate any research tasks for the agents."
                    )

                if not tool_calls:
                    # Basically hope that this is an infrequent occurence and hopefully multiple research
                    # cycles have already ran
                    logger.warning("No tool calls found, this should not happen.")
                    report_turn_index = (
                        orchestrator_start_turn_index + cycle + reasoning_cycles
                    )
                    report_reasoned = generate_final_report(
                        history=simple_chat_history,
                        research_plan=research_plan,
                        llm=llm,
                        token_counter=token_counter,
                        state_container=state_container,
                        emitter=emitter,
                        turn_index=report_turn_index,
                        citation_mapping=citation_mapping,
                        is_regulatory_research=is_regulatory_deep_research,
                        user_identity=user_identity,
                        reasoning_effort=reasoning_effort,
                        pre_answer_processing_time=time.monotonic()
                        - processing_start_time,
                        all_injected_file_metadata=all_injected_file_metadata,
                        exact_evidence_chunks=(
                            exact_evidence_chunks
                            if is_regulatory_deep_research
                            else None
                        ),
                        evidence_citation_mapping=(
                            evidence_citation_mapping
                            if is_regulatory_deep_research
                            else None
                        ),
                        recovery_tools=allowed_tools,
                        custom_agent_prompt=custom_agent_prompt,
                    )
                    final_turn_index = report_turn_index + (1 if report_reasoned else 0)
                    break

                special_tool_calls = check_special_tool_calls(tool_calls=tool_calls)

                if special_tool_calls.generate_report_tool_call:
                    report_turn_index = special_tool_calls.generate_report_tool_call.placement.turn_index
                    report_reasoned = generate_final_report(
                        history=simple_chat_history,
                        research_plan=research_plan,
                        llm=llm,
                        token_counter=token_counter,
                        state_container=state_container,
                        emitter=emitter,
                        turn_index=report_turn_index,
                        citation_mapping=citation_mapping,
                        is_regulatory_research=is_regulatory_deep_research,
                        user_identity=user_identity,
                        reasoning_effort=reasoning_effort,
                        saved_reasoning=most_recent_reasoning,
                        pre_answer_processing_time=time.monotonic()
                        - processing_start_time,
                        all_injected_file_metadata=all_injected_file_metadata,
                        exact_evidence_chunks=(
                            exact_evidence_chunks
                            if is_regulatory_deep_research
                            else None
                        ),
                        evidence_citation_mapping=(
                            evidence_citation_mapping
                            if is_regulatory_deep_research
                            else None
                        ),
                        recovery_tools=allowed_tools,
                        custom_agent_prompt=custom_agent_prompt,
                    )
                    final_turn_index = report_turn_index + (1 if report_reasoned else 0)
                    break
                elif special_tool_calls.think_tool_call:
                    think_tool_call = special_tool_calls.think_tool_call
                    # Only process the THINK_TOOL and skip all other tool calls
                    # This will not actually get saved to the db as a tool call but we'll attach it to the tool(s) called after
                    # it as if it were just a reasoning model doing it. In the chat history, because it happens in 2 steps,
                    # we will show it as a separate message.
                    # NOTE: This does not need to increment the reasoning cycles because the custom token processor causes
                    # the LLM step to handle this
                    with function_span("think_tool") as span:
                        span.span_data.input = str(think_tool_call.tool_args)
                        most_recent_reasoning = state_container.reasoning_tokens
                        tool_call_message = think_tool_call.to_msg_str()
                        tool_call_token_count = token_counter(tool_call_message)

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
                        simple_chat_history.append(think_assistant_msg)

                        think_tool_response_msg = ChatMessageSimple(
                            message=THINK_TOOL_RESPONSE_MESSAGE,
                            token_count=THINK_TOOL_RESPONSE_TOKEN_COUNT,
                            message_type=MessageType.TOOL_CALL_RESPONSE,
                            tool_call_id=think_tool_call.tool_call_id,
                            image_files=None,
                        )
                        simple_chat_history.append(think_tool_response_msg)
                        span.span_data.output = THINK_TOOL_RESPONSE_MESSAGE
                    continue
                else:
                    for tool_call in tool_calls:
                        if tool_call.tool_name != RESEARCH_AGENT_TOOL_NAME:
                            logger.warning(
                                "Unexpected tool call: %s", tool_call.tool_name
                            )
                    remaining_research_agent_call_budget = (
                        MAX_TOTAL_RESEARCH_AGENT_CALLS - research_agent_calls_started
                        if is_regulatory_deep_research
                        else None
                    )
                    bounded_research_agent_calls, _ = _bounded_research_agent_batch(
                        tool_calls,
                        remaining_call_budget=remaining_research_agent_call_budget,
                    )
                    total_research_agent_calls = sum(
                        tool_call.tool_name == RESEARCH_AGENT_TOOL_NAME
                        for tool_call in tool_calls
                    )
                    unrun_research_agent_call_count = max(
                        0,
                        total_research_agent_calls - len(bounded_research_agent_calls),
                    )

                    rejected_research_agent_calls: list[
                        tuple[ToolCallKickoff, str]
                    ] = []
                    research_agent_calls: list[ToolCallKickoff] = []
                    for research_agent_call in bounded_research_agent_calls:
                        rejection = (
                            _regulatory_research_task_rejection(research_agent_call)
                            if is_regulatory_deep_research
                            else None
                        )
                        if rejection is None:
                            research_agent_calls.append(research_agent_call)
                        else:
                            rejected_research_agent_calls.append(
                                (research_agent_call, rejection)
                            )

                    _append_rejected_research_agent_feedback(
                        simple_chat_history,
                        rejected_research_agent_calls,
                        token_counter,
                    )
                    if rejected_research_agent_calls:
                        logger.warning(
                            "Rejected %d malformed or over-broad regulatory "
                            "research-agent task(s)",
                            len(rejected_research_agent_calls),
                        )

                    total_research_agent_budget_exhausted = bool(
                        is_regulatory_deep_research
                        and research_agent_calls_started + len(research_agent_calls)
                        >= MAX_TOTAL_RESEARCH_AGENT_CALLS
                    )
                    if unrun_research_agent_call_count:
                        if total_research_agent_budget_exhausted:
                            logger.warning(
                                "Dropped %d research-agent call(s) because the "
                                "regulatory turn safety budget is exhausted",
                                unrun_research_agent_call_count,
                            )
                        else:
                            logger.warning(
                                "Deferred %d over-limit research-agent call(s); the "
                                "orchestrator can reassess them in the next cycle",
                                unrun_research_agent_call_count,
                            )

                    if not research_agent_calls:
                        _append_unrun_research_agent_feedback(
                            simple_chat_history,
                            unrun_call_count=unrun_research_agent_call_count,
                            total_budget_exhausted=(
                                total_research_agent_budget_exhausted
                            ),
                            token_counter=token_counter,
                        )
                        if rejected_research_agent_calls:
                            most_recent_reasoning = None
                            continue
                        logger.warning(
                            "No research agent tool calls found, this should not happen."
                        )
                        report_turn_index = (
                            orchestrator_start_turn_index + cycle + reasoning_cycles
                        )
                        report_reasoned = generate_final_report(
                            history=simple_chat_history,
                            research_plan=research_plan,
                            llm=llm,
                            token_counter=token_counter,
                            state_container=state_container,
                            emitter=emitter,
                            turn_index=report_turn_index,
                            citation_mapping=citation_mapping,
                            is_regulatory_research=is_regulatory_deep_research,
                            user_identity=user_identity,
                            reasoning_effort=reasoning_effort,
                            pre_answer_processing_time=time.monotonic()
                            - processing_start_time,
                            all_injected_file_metadata=all_injected_file_metadata,
                            exact_evidence_chunks=(
                                exact_evidence_chunks
                                if is_regulatory_deep_research
                                else None
                            ),
                            evidence_citation_mapping=(
                                evidence_citation_mapping
                                if is_regulatory_deep_research
                                else None
                            ),
                            recovery_tools=allowed_tools,
                            custom_agent_prompt=custom_agent_prompt,
                        )
                        final_turn_index = report_turn_index + (
                            1 if report_reasoned else 0
                        )
                        break

                    parent_tool_call_ids = [
                        tool_call.tool_call_id for tool_call in research_agent_calls
                    ]

                    if is_regulatory_deep_research:
                        research_agent_calls_started += len(research_agent_calls)

                    if len(research_agent_calls) > 1:
                        emitter.emit(
                            Packet(
                                placement=Placement(
                                    turn_index=research_agent_calls[
                                        0
                                    ].placement.turn_index
                                ),
                                obj=TopLevelBranching(
                                    num_parallel_branches=len(research_agent_calls)
                                ),
                            )
                        )

                    research_results = run_research_agent_calls(
                        # The tool calls here contain the placement information
                        research_agent_calls=research_agent_calls,
                        parent_tool_call_ids=parent_tool_call_ids,
                        tools=allowed_tools,
                        emitter=emitter,
                        state_container=state_container,
                        llm=llm,
                        is_reasoning_model=is_reasoning_model,
                        token_counter=token_counter,
                        citation_mapping=citation_mapping,
                        evidence_citation_mapping=evidence_citation_mapping,
                        user_identity=user_identity,
                        # Session override wins in sub-agents. AUTO keeps the tuned LOW default.
                        reasoning_effort=(
                            reasoning_effort
                            if reasoning_effort is not ReasoningEffort.AUTO
                            else ReasoningEffort.LOW
                        ),
                    )

                    citation_mapping = research_results.citation_mapping
                    evidence_citation_mapping = (
                        research_results.evidence_citation_mapping
                    )
                    if is_regulatory_deep_research:
                        for evidence_chunk in research_results.exact_evidence_chunks:
                            evidence_key = (
                                evidence_chunk.chunk_identifier,
                                evidence_chunk.content,
                            )
                            existing_index = exact_evidence_indexes.get(evidence_key)
                            if existing_index is None:
                                exact_evidence_indexes[evidence_key] = len(
                                    exact_evidence_chunks
                                )
                                exact_evidence_chunks.append(evidence_chunk)
                                continue
                            if (
                                exact_evidence_chunks[existing_index].citation_number
                                is None
                                and evidence_chunk.citation_number is not None
                            ):
                                exact_evidence_chunks[existing_index] = evidence_chunk

                    # Build ONE ASSISTANT message with all tool calls (OpenAI parallel format)
                    tool_calls_simple: list[ToolCallSimple] = []
                    for current_tool_call in research_agent_calls:
                        tool_call_message = current_tool_call.to_msg_str()
                        tool_call_token_count = token_counter(tool_call_message)
                        tool_calls_simple.append(
                            ToolCallSimple(
                                tool_call_id=current_tool_call.tool_call_id,
                                tool_name=current_tool_call.tool_name,
                                tool_arguments=current_tool_call.tool_args,
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
                    simple_chat_history.append(assistant_with_tools)

                    # Now add TOOL_CALL_RESPONSE messages and tool call info for each result
                    research_agent_tool_id = _get_research_agent_tool_id()
                    for tab_index, report in enumerate(
                        research_results.intermediate_reports
                    ):
                        if report is None:
                            # Every tool_use id in the preceding assistant message must have a
                            # matching TOOL_CALL_RESPONSE or strict providers (e.g. AWS Bedrock
                            # Converse) reject the next request with 400 "Expected toolResult
                            # blocks at messages.N.content for the following Ids: ...". Emit a
                            # synthetic failure response so the invariant holds and the LLM
                            # knows the call failed.
                            logger.error(
                                "Research agent call at tab_index %s failed; emitting synthetic failure response",
                                tab_index,
                            )
                            failed_tool_call = research_agent_calls[tab_index]
                            failure_message = "Research agent call failed. Try a different approach or continue without this result."
                            simple_chat_history.append(
                                ChatMessageSimple(
                                    message=failure_message,
                                    token_count=token_counter(failure_message),
                                    message_type=MessageType.TOOL_CALL_RESPONSE,
                                    tool_call_id=failed_tool_call.tool_call_id,
                                    image_files=None,
                                )
                            )
                            continue

                        current_tool_call = research_agent_calls[tab_index]
                        tool_call_info = ToolCallInfo(
                            parent_tool_call_id=None,
                            turn_index=orchestrator_start_turn_index
                            + cycle
                            + reasoning_cycles,
                            tab_index=tab_index,
                            tool_name=current_tool_call.tool_name,
                            tool_call_id=current_tool_call.tool_call_id,
                            tool_id=research_agent_tool_id,
                            reasoning_tokens=llm_step_result.reasoning
                            or most_recent_reasoning,
                            tool_call_arguments=current_tool_call.tool_args,
                            tool_call_response=report,
                            search_docs=None,  # Intermediate docs are not saved/shown
                            generated_images=None,
                        )
                        state_container.add_tool_call(tool_call_info)

                        tool_call_response_msg = ChatMessageSimple(
                            message=report,
                            token_count=token_counter(report),
                            message_type=MessageType.TOOL_CALL_RESPONSE,
                            tool_call_id=current_tool_call.tool_call_id,
                            image_files=None,
                        )
                        simple_chat_history.append(tool_call_response_msg)

                    _append_unrun_research_agent_feedback(
                        simple_chat_history,
                        unrun_call_count=unrun_research_agent_call_count,
                        total_budget_exhausted=total_research_agent_budget_exhausted,
                        token_counter=token_counter,
                    )

                # If it reached this point, it did not call reasoning, so here we wipe it to not save it to multiple turns
                most_recent_reasoning = None

        emitter.emit(
            Packet(
                placement=Placement(turn_index=final_turn_index),
                obj=OverallStop(type="stop"),
            )
        )
