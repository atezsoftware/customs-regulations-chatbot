"""Isolated ASv3 chat entry point, durable research and citation publication."""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from functools import wraps
from typing import ParamSpec, cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.answer_model import source_answer_model, uses_source_answer_model
from onyx.asv3.authority import cited_lower_statute_gap, native_named_authority_gap
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.external_tools import build_external_specs
from onyx.asv3.harness import Harness
from onyx.asv3.jev_answer_review import ReviewEvidence, review_and_repair_answer
from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    related_source_reviews_enabled,
)
from onyx.asv3.llm_adapter import (
    LanguageProfile,
    ResearchModel,
)
from onyx.asv3.models import (
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.parallel_checkpoint_writer import ParallelCheckpointWriter
from onyx.asv3.parallel_execution import (
    ParallelExecutionSlots,
    parallel_execution_enabled,
)
from onyx.asv3.progress import (
    ProgressEvent,
    ProgressReporter,
    localized_notifications,
    official_corpus_source_name,
    report_source_deliveries,
)
from onyx.asv3.publication_gaps import combine_source_publication_gaps
from onyx.asv3.question_research import QuestionResearch
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import ResearchState, build_research_specs
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.asv3.scenario import initial_questions
from onyx.asv3.search_adapter import build_search_adapter
from onyx.asv3.serial_experimental_session import (
    SERIAL_SESSION_POLICY,
    SerialExperimentalSession,
    accepted_serial_memory_state,
    merge_serial_session_memory,
    validate_serial_session_answer,
)
from onyx.asv3.session_research import (
    retain_session_research,
    session_research_checkpoint,
)
from onyx.asv3.shared_reads import SharedReads
from onyx.asv3.source_tools import build_source_specs
from onyx.asv3.supplemental_tools import (
    ScenarioState,
    build_supplemental_specs,
    public_narration_valid,
)
from onyx.asv3.workers import WorkerPool
from onyx.asv3.workflow_variant import (
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_GUARDRAILS_V2_VARIANT,
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_VARIANT,
    checkpoint_variant_fields,
    validate_asv3_variant_resume,
)
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.chat.stop_signal_checker import is_connected
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters, SearchDoc
from onyx.context.search.retrieval.query_embedding_scope import (
    ParallelQueryEmbeddingScope,
)
from onyx.db.asv3_corpus import bind_pc_corpus_scope
from onyx.db.asv3_runs import (
    load_asv3_checkpoint,
    load_asv3_session_checkpoint,
    save_asv3_checkpoint,
)
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_PARALLEL_PROMPT_VERSION,
    EXPERIMENTAL_PROMPT_VERSION,
)
from onyx.prompts.asv3.research import (
    PROMPT_VERSION,
    VERIFICATION_PROMPT,
)
from onyx.prompts.asv3.tuned import TUNED_PROMPT_VERSION
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
    SectionEnd,
)
from onyx.tools.interface import Tool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.framework.create import ChatTraceMetadata, ensure_trace

logger = logging.getLogger(__name__)
P = ParamSpec("P")


def _verification_context(context: RunContext) -> RunContext:
    if not parallel_execution_enabled(context):
        return context
    isolated = copy.copy(context)
    isolated.services = dict(context.services)
    return isolated


def _presentation_search_doc(item: EvidenceItem) -> SearchDoc | None:
    if item.search_doc is None:
        return None
    document = item.search_doc.model_copy(deep=True)
    name = official_corpus_source_name(item)
    if name:
        document.metadata["asv3_source_display_name"] = name
    return document


def _presentation_search_docs(
    ledger: EvidenceLedger, numbers: Iterable[int]
) -> dict[int, SearchDoc]:
    documents: dict[int, SearchDoc] = {}
    for number in numbers:
        item = ledger.get(number)
        if item is not None:
            document = _presentation_search_doc(item)
            if document is not None:
                documents[number] = document
    return documents


def _trace_asv3(function: Callable[P, None]) -> Callable[P, None]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        llm = cast(LLM, kwargs["llm"])
        identity = cast(LLMUserIdentity | None, kwargs.get("user_identity"))
        with ensure_trace(
            "run_asv3_loop",
            group_id=str(kwargs["chat_session_id"]),
            metadata=ChatTraceMetadata(
                chat_session_id=str(kwargs["chat_session_id"]),
                user_id=identity.user_id
                if identity
                else str(cast(User, kwargs["user"]).id),
                user_message_id=cast(int, kwargs["user_message_id"]),
                assistant_message_id=cast(int, kwargs["assistant_message_id"]),
                model_name=llm.config.model_name,
            ).model_dump(),
        ):
            function(*args, **kwargs)

    return wrapped


def _evidence_record(
    ledger: EvidenceLedger,
    draft: str,
    max_chars: int = 180000,
    *,
    preferred_numbers: list[int] | None = None,
    include_witness_spans: bool = False,
    include_supplemental_originals: bool = False,
    required_numbers: list[int] | None = None,
) -> str:
    numbers = list(extract_citation_numbers(draft))
    required = tuple(
        n
        for n in dict.fromkeys([*numbers, *(required_numbers or [])])
        if ledger.get(n) is not None
    )
    numbers = list(required)
    if preferred_numbers:
        numbers = list(dict.fromkeys([*required, *preferred_numbers]))
    elif numbers:
        source_ids = {
            item.source_id for n in required if (item := ledger.get(n)) is not None
        }
        numbers.extend(
            n
            for item in ledger.summaries()
            if isinstance(n := item["citation"], int)
            and item["source_id"] in source_ids
        )
    else:
        numbers = [
            n for item in ledger.summaries() if isinstance(n := item["citation"], int)
        ]
    if include_supplemental_originals:
        numbers = list(dict.fromkeys([*numbers, *ledger.citation_numbers()]))
    return ledger.serialize_records(
        numbers,
        required=required,
        max_chars=max_chars,
        include_witness_spans=include_witness_spans,
    )


@_trace_asv3
def run_asv3_loop(
    *,
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    llm: LLM,
    token_counter: Callable[[str], int],
    user: User,
    filters: BaseFilters | None,
    document_set_names_override: list[str] | None,
    user_identity: LLMUserIdentity | None,
    chat_session_id: UUID,
    user_message_id: int,
    assistant_message_id: int,
    reasoning_effort: ReasoningEffort,
    include_citations: bool,
    cache: CacheBackend,
    research_profile: str = "deep",
    parallel_research: bool = False,
    workflow_variant: str = ASV3_STANDARD_VARIANT,
    research_llm: LLM | None = None,
    repair_llm: LLM | None = None,
    resume_message_id: int | None = None,
    custom_agent_prompt: str | None = None,
    allow_external: bool = False,
    user_memory_context: UserMemoryContext | None = None,
    user_info: str | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    start = time.monotonic()
    question = next(
        (
            m.message
            for m in reversed(simple_chat_history)
            if m.message_type == MessageType.USER
        ),
        "",
    )
    guarded_experimental = workflow_variant == ASV3_GUARDED_EXPERIMENTAL_VARIANT
    context = RunContext(
        language="und",
        timeout_seconds=1800 if guarded_experimental else float("inf"),
        research_reserve_seconds=30 if guarded_experimental else 0,
        budget=(
            SharedBudget(max_tools=24, max_decisions=32)
            if guarded_experimental
            else SharedBudget(unlimited_execution=True)
        ),
        cancelled=lambda: not is_connected(chat_session_id, cache),
        services={
            "lean_native_mode": True,
            **(
                {
                    "provider_max_attempts": 2,
                    "provider_compatibility_attempts": 1,
                }
                if guarded_experimental
                else {}
            ),
        },
    )
    if custom_agent_prompt:
        context.services["assistant_instructions"] = custom_agent_prompt
    scope = IndexFilters(
        **(filters.model_dump() if filters else {}),
        access_control_list=[],
        regulatory_workflow_mode=filters.regulatory_workflow_mode
        if filters
        else "standard",
        regulatory_label_search_enabled=filters.regulatory_label_search_enabled
        if filters
        else False,
        regulatory_label_run_ids=filters.regulatory_label_run_ids if filters else (),
    )
    if document_set_names_override:
        scope.forced_document_set = document_set_names_override
    scope = bind_pc_corpus_scope(user=user, filters=scope)
    context.scope = scope.model_dump(mode="json")
    previous = (
        load_asv3_checkpoint(message_id=resume_message_id, user_id=user.id)
        if resume_message_id is not None
        else None
    )
    if resume_message_id is not None and (
        previous is None
        or previous.get("scope") != context.scope
        or previous.get("request") != question
    ):
        raise ValueError("ASv3 resume requires the same authorized scope and question")
    validate_asv3_variant_resume(workflow_variant, previous)
    if workflow_variant == ASV3_TUNED_VARIANT and (
        research_profile != "normal" or parallel_research
    ):
        raise ValueError(
            "Tuned ASv3 requires the normal profile without parallel research"
        )
    if previous is not None:
        profile = LanguageProfile.model_validate(previous.get("public_profile"))
        profile.requires_sources = True
        context.run_id = str(previous["run_id"])
    else:
        profile = LanguageProfile(
            language="und",
            notifications=localized_notifications("und"),
            requires_sources=True,
        )
    if previous is not None:
        research_profile = str(
            previous.get("research_profile")
            or ("deep" if "question_research" in previous else "normal")
        )
        saved_parallel = previous.get("parallel_research", False)
        if type(saved_parallel) is not bool:
            raise ValueError("Invalid saved ASv3 parallel research mode")
        parallel_research = saved_parallel
    if research_profile not in {"normal", "deep", "experimental"}:
        raise ValueError("Unknown ASv3 research profile")
    if parallel_research and research_profile != "experimental":
        raise ValueError("Parallel research requires the Experimental profile")
    if (
        parallel_research
        and previous is not None
        and previous.get("parallel_research_policy") != SERIAL_SESSION_POLICY
    ):
        raise ValueError("Saved parallel research uses a different session policy")
    if research_profile == "experimental" or workflow_variant == ASV3_TUNED_VARIANT:
        research_llm = None
    answer_llm = None
    if uses_source_answer_model(workflow_variant, llm):
        # Resolve visibility and user access before starting paid research.
        answer_llm = source_answer_model(user)
        llm = llm.with_temperature(0.1)
        context.services["explicit_research_temperature"] = True
    elif (
        workflow_variant == ASV3_TUNED_VARIANT
        and llm.config.model_provider == "vertex_ai"
        and llm.config.model_name.startswith("gemini-")
        and llm.config.temperature == 0.1
    ):
        llm = llm.with_temperature(llm.config.temperature)
        context.services["explicit_research_temperature"] = True
    context.language = profile.language
    context.services["research_profile"] = research_profile
    context.services["asv3_legacy_search_payload"] = workflow_variant in {
        ASV3_STANDARD_VARIANT,
        ASV3_TUNED_VARIANT,
    }
    context.services["experimental_parallel"] = parallel_research
    if workflow_variant in {
        ASV3_TUNED_VARIANT,
        ASV3_GUARDED_EXPERIMENTAL_VARIANT,
        ASV3_GUARDRAILS_V2_VARIANT,
    }:
        context.services["asv3_workflow_variant"] = workflow_variant
    context.services["independent_question_mode"] = (
        research_profile == "deep" or parallel_research
    )
    if parallel_research:
        context.services["parallel_execution_slots"] = ParallelExecutionSlots()
        context.services["parallel_query_embeddings"] = ParallelQueryEmbeddingScope()
        context.services["scenario_request"] = question
    context.corpus_only = not (allow_external and profile.external_requested)
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    parallel_answers = (
        ParallelAnswerReceipts(context, question, user_id=str(user.id))
        if parallel_research
        else None
    )
    source_reviews = (
        LegalSourceReviews(context, question)
        if related_source_reviews_enabled(context)
        else None
    )
    if source_reviews is not None:
        context.services["legal_source_reviews"] = source_reviews
    authority_requirements = (
        AuthorityRequirements(
            context,
            question,
            syntactic_reference_binding=parallel_research,
        )
        if research_profile == "experimental"
        else None
    )
    if authority_requirements is not None:
        context.services["authority_requirements"] = authority_requirements

    def named_authority_gap(
        answer: str,
        model_call_id: str | None,
        run_context: RunContext | None = None,
    ) -> dict[str, JsonValue] | None:
        if answer_llm is not None:
            # This policy accepts a supplied passage establishing the actual claim.
            return None
        gap = native_named_authority_gap(
            answer,
            ledger,
            strict_reference_boundaries=(
                research_profile == "experimental"
                or workflow_variant == ASV3_TUNED_VARIANT
            ),
            syntactic_reference_binding=(
                parallel_research or workflow_variant == ASV3_TUNED_VARIANT
            ),
        )
        if authority_requirements is not None:
            return authority_requirements.publication_gap(
                answer, model_call_id, run_context or context, ledger, native_gap=gap
            )
        return gap

    registry = CapabilityRegistry()
    broker = CorpusBroker(
        user,
        scope,
        vision_llm=research_llm or llm,
        allow_numbered_title_fallback=(
            research_profile == "experimental" or workflow_variant == ASV3_TUNED_VARIANT
        ),
    )
    if parallel_research or workflow_variant == ASV3_TUNED_VARIANT:

        def shared_producer(caller: RunContext) -> RunContext:
            return RunContext(
                run_id=caller.run_id,
                language=caller.language,
                scope=caller.scope,
                services=caller.services,
                budget=caller.budget,
                deadline=caller.deadline,
                research_deadline=caller.research_deadline,
                depth=caller.depth,
                max_depth=caller.max_depth,
                corpus_only=caller.corpus_only,
                cancelled=context.is_cancelled,
            )

        context.services["shared_reads"] = SharedReads(
            fence=broker.shared_read_fence, producer_context=shared_producer
        )
    context.services["legal_source_navigation_acquire"] = (
        broker.related_sources_for_evidence
    )
    context.services["legal_source_navigation"] = broker.related_source_navigation
    history = "\n".join(
        f"{m.message_type.value}: {m.message}"
        for m in simple_chat_history[-8:]
        if m.message_type in (MessageType.USER, MessageType.ASSISTANT)
    )
    model = ResearchModel(
        llm,
        context,
        user_identity=user_identity,
        reasoning_effort=reasoning_effort,
        history=history,
        token_counter=token_counter,
        lean_native_mode=True,
        research_llm=None if guarded_experimental else research_llm,
        answer_llm=answer_llm,
    )
    if previous is not None:
        model.restore_native_sampling(previous)
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is not None and (
        research_llm is not None
        or research_profile == "experimental"
        or workflow_variant == ASV3_TUNED_VARIANT
    ):
        search = search.fork_for_independent_context()
        search.llm = research_llm or llm

    def search_history(child: RunContext) -> list[ChatMessageSimple]:
        private = child.services.get("search_message_history")
        return (
            cast(list[ChatMessageSimple], private)
            if isinstance(private, list)
            and all(isinstance(m, ChatMessageSimple) for m in private)
            else list(simple_chat_history)
        )

    adapter = build_search_adapter(
        search,
        question,
        broker,
        message_history=search_history,
        user_memory_context=user_memory_context,
        user_info=user_info,
        inject_memories_in_prompt=inject_memories_in_prompt,
        user_identity=user_identity,
    )
    broker.search_adapter = adapter
    context.services["prepare_search_batch"] = adapter.prepare_batch
    scenarios = ScenarioState(initial_questions(question), frozen=True)
    research_state = ResearchState(
        initial_questions(question),
        context,
        require_need_bindings=False,
    )
    context.services.update(scenario_state=scenarios, research_state=research_state)
    outcome_map = OutcomeMap(
        initial_questions(question),
        context,
        factual_context=question + "\n" + history,
        detailed_fact_errors=research_profile == "experimental",
        reuse_retained_conditions=workflow_variant == ASV3_TUNED_VARIANT,
    )
    context.services["outcome_map"] = outcome_map
    emitted: list[dict[str, JsonValue]] = []
    checkpoint_lock = threading.RLock()
    progress_lock = threading.RLock()
    root_control_lock = threading.RLock()
    root_owner_thread = threading.get_ident()
    parallel_root_control: dict[str, JsonValue] | None = None
    checkpoint_sequence = 0
    checkpoint_writer: ParallelCheckpointWriter | None = None
    harness: Harness | None = None
    workers: WorkerPool | None = None
    question_research: QuestionResearch | None = None
    final_published = False
    publication_status = OutcomeStatus.PARTIAL
    publication_stop_reason: str | None = None
    clarification: str | None = None
    partial: str | None = None
    first_decision = True
    standalone_answer_call = False

    def emit_progress(event: ProgressEvent) -> None:
        phase, status = event.phase, event.status
        if (
            parallel_research
            and not final_published
            and event.task_id is not None
            and phase in {"final", "completed"}
        ):
            phase = "tools"
        elif phase == "completed" and not final_published:
            phase, status = "final", "running"
        words = profile.notifications.get(phase, profile.notifications["tools"])
        packet = ASv3Progress.model_validate(
            dict(
                run_id=event.run_id,
                event_id=event.event_id,
                sequence=event.sequence,
                language=event.language,
                phase=phase,
                status=status,
                title=event.title if event.public_narration else words[0],
                message=event.message if event.public_narration else words[1],
                task_id=event.task_id,
                parent_task_id=event.parent_task_id,
                active_tasks=event.active_workers,
                completed_tasks=event.completed_workers,
            )
        )
        with progress_lock if parallel_research else checkpoint_lock:
            emitted.append(packet.model_dump(mode="json"))
            if len(emitted) > 100:
                latest = {
                    str(row.get("task_id") or "coordinator"): row for row in emitted
                }
                retained = {
                    str(row["event_id"]): row
                    for row in [emitted[0], *latest.values(), *emitted[-40:]]
                }
                emitted[:] = sorted(
                    retained.values(), key=lambda row: int(str(row["sequence"]))
                )
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=packet))

    progress = ProgressReporter(context.run_id, context.language, emit_progress)
    context.services["progress"] = progress

    def on_decision(decision: Decision) -> None:
        nonlocal first_decision, standalone_answer_call
        if checkpoint_writer is not None:
            checkpoint_writer.raise_if_failed()
        standalone_answer_call = (
            len(decision.calls) == 1 and decision.calls[0].name == "submit_answer"
        )
        for call in decision.calls:
            language = call.arguments.get("_language")
            if profile.language == "und" and isinstance(language, str):
                try:
                    candidate = LanguageProfile(
                        language=language,
                        notifications=localized_notifications(language),
                        requires_sources=profile.requires_sources,
                        external_requested=profile.external_requested,
                    )
                except ValueError:
                    continue
                if candidate.language != profile.language:
                    profile.language, profile.notifications = (
                        candidate.language,
                        candidate.notifications,
                    )
                context.language = progress.language = candidate.language
            notifications = call.arguments.get("_notifications")
            if isinstance(notifications, dict):
                for phase, pair in notifications.items():
                    if (
                        phase in profile.notifications
                        and isinstance(pair, list)
                        and len(pair) == 2
                        and all(isinstance(word, str) and word.strip() for word in pair)
                    ):
                        title, message = map(str, pair)
                        if public_narration_valid(title, message, context):
                            profile.notifications[phase] = [title[:240], message[:1600]]
            if first_decision and call.arguments.get("_external_requested") is True:
                profile.external_requested = True
        context.corpus_only = not (allow_external and profile.external_requested)
        context.services["native_citation_label"] = profile.notifications[
            "native_citation"
        ][0]
        state_container.set_stop_notice(profile.notifications["cancelled"][1])
        first_decision = False
        report_source_deliveries(
            ledger,
            model.last_call_id,
            context,
            progress,
            profile.notifications["tools"],
        )

    def persist_checkpoint(snapshot: dict[str, JsonValue]) -> None:
        with graph_step(
            "asv3.checkpoint_persistence", {"sequence": snapshot.get("sequence")}
        ) as persistence:
            started = time.monotonic()
            save_asv3_checkpoint(
                message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
            )
            persistence.output_value = {"persist_seconds": time.monotonic() - started}

    def publish_root_control() -> None:
        nonlocal parallel_root_control
        if threading.get_ident() != root_owner_thread or harness is None:
            raise ValueError("Root checkpoint control requires its semantic owner")
        control = harness.snapshot_control()
        control.update(
            native_coordinator_sampling=model.native_sampling_snapshot(),
            scope=copy.deepcopy(context.scope),
            public_profile=profile.model_dump(mode="json"),
            publication_status=publication_status.value,
            publication_stop_reason=publication_stop_reason,
            final_publication_gap=control["publication_gap"],
        )
        with root_control_lock:
            parallel_root_control = control

    def capture_parallel_checkpoint() -> dict[str, JsonValue]:
        nonlocal checkpoint_sequence
        with graph_step(
            "asv3.checkpoint_capture", {"deferred_aggregate": True}
        ) as capture:
            lock_started = time.monotonic()
            with checkpoint_lock:
                lock_wait_seconds = time.monotonic() - lock_started
                with root_control_lock:
                    if parallel_root_control is None:
                        raise ValueError("Parallel checkpoint control is missing")
                    control = parallel_root_control
                snapshot = copy.deepcopy(control)
                with progress_lock:
                    progress_snapshot = list(emitted)
                augmentation_started = time.monotonic()
                worker_capture = (
                    workers.capture_checkpoint_state() if workers is not None else None
                )
                child_control_capture_seconds = time.monotonic() - augmentation_started
                snapshot["research_state"] = research_state.export()
                checkpoint_sequence += 1
                snapshot.update(
                    sequence=checkpoint_sequence,
                    **checkpoint_variant_fields(workflow_variant),
                    prompt_version=TUNED_PROMPT_VERSION
                    if workflow_variant == ASV3_TUNED_VARIANT
                    else EXPERIMENTAL_PARALLEL_PROMPT_VERSION
                    if parallel_research
                    else EXPERIMENTAL_PROMPT_VERSION
                    if research_profile == "experimental"
                    else PROMPT_VERSION,
                    research_profile=research_profile,
                    parallel_research=parallel_research,
                    execution_mode="native",
                    native_coordinator_sampling=snapshot["native_coordinator_sampling"],
                    scope=snapshot["scope"],
                    progress=progress_snapshot,
                    progress_state=progress.export(),
                    public_profile=snapshot["public_profile"],
                    workers={},
                    question_research=question_research.export()
                    if question_research
                    else {},
                    outcome_map=outcome_map.export(),
                    publication_status=snapshot["publication_status"],
                    publication_stop_reason=snapshot["publication_stop_reason"],
                    final_publication_gap=snapshot["final_publication_gap"],
                    scenario=scenarios.snapshot(),
                    question_message_id=user_message_id,
                    session_research=session_research_checkpoint(context, question),
                )
                if source_reviews is not None:
                    snapshot["legal_source_reviews"] = source_reviews.export()
                if authority_requirements is not None:
                    snapshot["authority_requirements"] = authority_requirements.export()
                receipts = (
                    parallel_answers.export() if parallel_answers is not None else {}
                )
                # The immutable child controls precede this single canonical ledger cut.
                ledger_started = time.monotonic()
                snapshot["evidence"] = ledger.export()
                ledger_export_seconds = time.monotonic() - ledger_started
                materialization_started = time.monotonic()
                if workers is not None and worker_capture is not None:
                    snapshot["workers"] = workers.materialize_checkpoint_state(
                        worker_capture, snapshot["evidence"]
                    )
                child_materialization_seconds = (
                    time.monotonic() - materialization_started
                )
                if parallel_answers is not None:
                    snapshot["parallel_research_policy"] = SERIAL_SESSION_POLICY
                    snapshot["parallel_answers"] = receipts
                    accepted_states: list[dict[str, JsonValue]] = []
                    receipt_rows = receipts.get("receipts", [])
                    questions_state = snapshot["question_research"]
                    assignment_rows = (
                        questions_state.get("assignments", [])
                        if isinstance(questions_state, dict)
                        else []
                    )
                    if not isinstance(assignment_rows, list):
                        raise ValueError("Invalid checkpoint assignments")
                    if (
                        question_research is not None
                        and workers is not None
                        and isinstance(receipt_rows, list)
                    ):
                        for assignment in assignment_rows:
                            if not isinstance(assignment, dict):
                                raise ValueError("Invalid checkpoint assignment")
                            task_id = str(assignment["task_id"])
                            receipt = next(
                                (
                                    row
                                    for row in receipt_rows
                                    if isinstance(row, dict)
                                    and row.get("task_id") == task_id
                                ),
                                None,
                            )
                            if receipt is None:
                                continue
                            worker_rows = snapshot["workers"]
                            task_rows = (
                                worker_rows.get("tasks", [])
                                if isinstance(worker_rows, dict)
                                else []
                            )
                            if not isinstance(task_rows, list):
                                raise ValueError("Invalid checkpoint worker tasks")
                            worker_row = next(
                                (
                                    row
                                    for row in task_rows
                                    if isinstance(row, dict)
                                    and row.get("task_id") == task_id
                                ),
                                None,
                            )
                            wrapper = (
                                worker_row.get("child_checkpoint")
                                if worker_row is not None
                                else None
                            )
                            serial_snapshot = (
                                wrapper.get("snapshot")
                                if isinstance(wrapper, dict)
                                else None
                            )
                            if not isinstance(serial_snapshot, dict):
                                raise ValueError(
                                    "Sealed serial memory checkpoint is missing"
                                )
                            accepted_states.append(
                                accepted_serial_memory_state(
                                    serial_snapshot,
                                    assignment,
                                    receipt,
                                    context,
                                    question,
                                    history,
                                )
                            )
                    memory = snapshot["session_research"]
                    if not isinstance(memory, dict):
                        raise ValueError("Serial session memory checkpoint is missing")
                    snapshot["session_research"] = merge_serial_session_memory(
                        memory, accepted_states
                    )
                snapshot["budget"] = context.budget.snapshot()
                augmentation_seconds = time.monotonic() - augmentation_started
            capture.output_value = {
                "sequence": snapshot["sequence"],
                "capture_lock_wait_seconds": lock_wait_seconds,
                "augmentation_seconds": augmentation_seconds,
                "child_control_capture_seconds": child_control_capture_seconds,
                "ledger_export_seconds": ledger_export_seconds,
                "child_materialization_seconds": child_materialization_seconds,
                "deferred_aggregate": True,
            }
        return snapshot

    def notify_checkpoint(*, durable: bool = False) -> None:
        if checkpoint_writer is None:
            raise ValueError("Parallel checkpoint writer is missing")
        context.check_active()
        revision = checkpoint_writer.notify()
        if durable:
            with graph_step(
                "asv3.checkpoint_barrier", {"revision": revision}
            ) as barrier:
                started = time.monotonic()
                checkpoint_writer.flush(revision)
                barrier.output_value = {"flush_seconds": time.monotonic() - started}

    def checkpoint(
        snapshot: dict[str, JsonValue],
        *,
        durable: bool = False,
        root_snapshot_seconds: float | None = None,
    ) -> None:
        nonlocal checkpoint_sequence
        if parallel_research:
            if threading.get_ident() == root_owner_thread:
                publish_root_control()
            notify_checkpoint(durable=durable)
            return
        if checkpoint_writer is not None:
            checkpoint_writer.raise_if_failed()
        capture = (
            graph_step("asv3.checkpoint_capture", {})
            if parallel_research and durable
            else nullcontext()
        )
        with capture as capture_step:
            revision: int | None = None
            lock_started = time.monotonic()
            with checkpoint_lock:
                lock_wait_seconds = time.monotonic() - lock_started
                context.check_active()
                if parallel_research:
                    if harness is None:
                        raise ValueError("Parallel capture requires its root harness")
                    snapshot_started = time.monotonic()
                    snapshot = harness.snapshot()
                    root_snapshot_seconds = time.monotonic() - snapshot_started
                augmentation_started = time.monotonic()
                checkpoint_sequence += 1
                snapshot.update(
                    sequence=checkpoint_sequence,
                    **checkpoint_variant_fields(workflow_variant),
                    prompt_version=TUNED_PROMPT_VERSION
                    if workflow_variant == ASV3_TUNED_VARIANT
                    else EXPERIMENTAL_PARALLEL_PROMPT_VERSION
                    if parallel_research
                    else EXPERIMENTAL_PROMPT_VERSION
                    if research_profile == "experimental"
                    else PROMPT_VERSION,
                    research_profile=research_profile,
                    parallel_research=parallel_research,
                    execution_mode="native",
                    native_coordinator_sampling=model.native_sampling_snapshot(),
                    scope=context.scope,
                    progress=list(emitted),
                    progress_state=progress.export(),
                    public_profile=profile.model_dump(mode="json"),
                    workers=workers.export() if workers else {},
                    question_research=question_research.export()
                    if question_research
                    else {},
                    outcome_map=outcome_map.export(),
                    publication_status=publication_status.value,
                    publication_stop_reason=publication_stop_reason,
                    final_publication_gap=harness.publication_gap.model_dump(
                        mode="json"
                    )
                    if harness and harness.publication_gap
                    else None,
                    scenario=scenarios.snapshot(),
                    question_message_id=user_message_id,
                    session_research=session_research_checkpoint(context, question),
                )
                if source_reviews is not None:
                    snapshot["legal_source_reviews"] = source_reviews.export()
                if authority_requirements is not None:
                    snapshot["authority_requirements"] = authority_requirements.export()
                guardrails_v2_review = context.services.get("guardrails_v2_review")
                if isinstance(guardrails_v2_review, dict):
                    snapshot["guardrails_v2_review"] = cast(
                        dict[str, JsonValue], copy.deepcopy(guardrails_v2_review)
                    )
                if parallel_answers is not None:
                    snapshot["parallel_research_policy"] = SERIAL_SESSION_POLICY
                    receipts = parallel_answers.export()
                    snapshot["parallel_answers"] = receipts
                    accepted_states: list[dict[str, JsonValue]] = []
                    receipt_rows = receipts.get("receipts", [])
                    if (
                        question_research is not None
                        and workers is not None
                        and isinstance(receipt_rows, list)
                    ):
                        for assignment in question_research.assignments:
                            task_id = str(assignment["task_id"])
                            receipt = next(
                                (
                                    row
                                    for row in receipt_rows
                                    if isinstance(row, dict)
                                    and row.get("task_id") == task_id
                                ),
                                None,
                            )
                            if receipt is None:
                                continue
                            serial_snapshot = workers.checkpoint(task_id)
                            if serial_snapshot is None:
                                raise ValueError(
                                    "Sealed serial memory checkpoint is missing"
                                )
                            accepted_states.append(
                                accepted_serial_memory_state(
                                    serial_snapshot,
                                    assignment,
                                    receipt,
                                    context,
                                    question,
                                    history,
                                )
                            )
                    memory = snapshot["session_research"]
                    if not isinstance(memory, dict):
                        raise ValueError("Serial session memory checkpoint is missing")
                    snapshot["session_research"] = merge_serial_session_memory(
                        memory, accepted_states
                    )
                    # Child acceptance may occur after the caller captured its root snapshot.
                    snapshot["evidence"] = ledger.export()
                augmentation_seconds = time.monotonic() - augmentation_started
                submit_started = time.monotonic()
                if checkpoint_writer is not None:
                    revision = checkpoint_writer.submit(snapshot)
                else:
                    save_asv3_checkpoint(
                        message_id=assistant_message_id,
                        user_id=user.id,
                        snapshot=snapshot,
                    )
                submit_seconds = time.monotonic() - submit_started
            flush_started = time.monotonic()
            if durable and checkpoint_writer is not None:
                checkpoint_writer.flush(revision)
            if capture_step is not None:
                capture_step.output_value = {
                    "sequence": snapshot.get("sequence"),
                    "revision": revision,
                    "root_snapshot_seconds": root_snapshot_seconds,
                    "capture_lock_wait_seconds": lock_wait_seconds,
                    "augmentation_seconds": augmentation_seconds,
                    "submit_seconds": submit_seconds,
                    "flush_seconds": time.monotonic() - flush_started,
                    "durable_barrier": durable,
                }

    def root_checkpoint(*, durable: bool = False) -> None:
        if harness is None:
            raise ValueError("Root checkpoint requires its harness")
        if parallel_research:
            checkpoint({}, durable=durable)
            return
        started = time.monotonic()
        snapshot = harness.snapshot()
        checkpoint(
            snapshot,
            durable=durable,
            root_snapshot_seconds=time.monotonic() - started,
        )

    def record(receipt: ToolReceipt) -> None:
        if checkpoint_writer is not None:
            checkpoint_writer.raise_if_failed()
        for number in receipt.evidence_ids:
            item = ledger.get(number)
            if item and item.search_doc is None and item.metadata.get("source_sha256"):
                ledger.add(
                    [
                        broker.attach_native_citation(
                            item, number, assistant_message_id, context
                        )
                    ],
                    context,
                )
        state_container.add_search_docs(
            list(_presentation_search_docs(ledger, receipt.evidence_ids).values())
        )

    def verify(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        child = _verification_context(child)
        numbers = args.get("evidence_numbers", args.get("citations", []))
        anchors = (
            " ".join(f"[{n}]" for n in numbers) if isinstance(numbers, list) else ""
        )
        verifier = ResearchModel(
            research_llm or llm,
            child,
            user_identity=user_identity,
            reasoning_effort=reasoning_effort,
            token_counter=token_counter,
        )
        child.consume_research_decision()
        report = verifier.invoke_verification(
            VERIFICATION_PROMPT,
            json.dumps(
                {
                    "language": child.language,
                    "scenario": question,
                    "claim": str(args.get("claim", "")),
                    "evidence": _evidence_record(
                        ledger,
                        anchors,
                        preferred_numbers=[n for n in numbers if isinstance(n, int)]
                        if isinstance(numbers, list)
                        else [],
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=2500,
            consume_budget=False,
        ).model_dump(mode="json")
        return ToolOutcome(
            status=OutcomeStatus.FOUND
            if report.get("status") == "supported"
            else OutcomeStatus.PARTIAL,
            summary="Claim checked against selected original evidence",
            data=report,
        )

    context.services["verify_claim"] = verify

    def remember_session_originals(active_harness: Harness) -> None:
        memory = context.services.get("session_research")
        numbers = (
            memory.get("reused_evidence_numbers", [])
            if isinstance(memory, dict)
            else []
        )
        if isinstance(numbers, list):
            for number in numbers:
                item = ledger.get(number) if type(number) is int else None
                if item is not None:
                    if item.search_doc is None and item.metadata.get("source_sha256"):
                        ledger.add(
                            [
                                broker.attach_native_citation(
                                    item, number, assistant_message_id, context
                                )
                            ],
                            context,
                        )
                    active_harness.evidence_working_set.remember(
                        number, 0, len(item.text)
                    )

    def researcher(
        task: str, child: RunContext, updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        if parallel_research and child.services.get("independent_question") is True:

            def serial_capabilities(local: RunContext) -> list[ToolSpec]:
                serial_broker = CorpusBroker(
                    user, scope, vision_llm=llm, allow_numbered_title_fallback=True
                )
                local.services["search_message_history"] = [
                    *simple_chat_history,
                    ChatMessageSimple(
                        message=task,
                        token_count=token_counter(task),
                        message_type=MessageType.USER,
                    ),
                ]
                serial_search = (
                    search.fork_for_independent_context()
                    if search is not None
                    else None
                )
                if serial_search is not None:
                    serial_search.llm = llm
                serial_adapter = build_search_adapter(
                    serial_search,
                    task,
                    serial_broker,
                    message_history=search_history,
                    user_memory_context=user_memory_context,
                    user_info=user_info,
                    inject_memories_in_prompt=inject_memories_in_prompt,
                    user_identity=user_identity,
                )
                serial_broker.search_adapter = serial_adapter
                local.services["prepare_search_batch"] = serial_adapter.prepare_batch
                local.services["legal_source_navigation_acquire"] = (
                    serial_broker.related_sources_for_evidence
                )
                local.services["legal_source_navigation"] = (
                    serial_broker.related_source_navigation
                )
                return (
                    build_corpus_specs(
                        serial_broker,
                        require_search_targets=True,
                        source_identity_guidance=True,
                    )
                    + build_source_specs(serial_broker)
                    + build_sandbox_specs(serial_broker)
                    + external_specs
                    + build_supplemental_specs()
                )

            def serial_checkpoint(snapshot: dict[str, JsonValue]) -> None:
                callback = child.services.get("record_child_checkpoint_control")
                if not callable(callback) or harness is None:
                    raise ValueError(
                        "Serial question requires its task-bound checkpoint writer"
                    )
                callback(snapshot)
                serial_state = snapshot.get("serial_experimental_session")
                root_checkpoint(
                    durable=isinstance(serial_state, dict)
                    and serial_state.get("accepted") is not None
                )

            session = SerialExperimentalSession(
                outer_context=child,
                request=task,
                scenario_request=question,
                history=history,
                ledger=ledger,
                llm=llm,
                reasoning_effort=reasoning_effort,
                token_counter=token_counter,
                capability_factory=serial_capabilities,
                verify=lambda local, args: verify(args, local),
                user_identity=user_identity,
                on_receipt=record,
                checkpoint_callback=serial_checkpoint,
                deferred_checkpoint_control=True,
                progress=progress,
                allow_external=allow_external,
                notifications=profile.notifications,
            )
            saved_session = child.services.get("previous_child_checkpoint")
            if isinstance(saved_session, dict):
                session.restore(saved_session)
            serial_result = session.run()
            if serial_result.summary and "serial_session_state" in serial_result.data:
                if question_research is None or parallel_answers is None:
                    raise ValueError(
                        "Serial answer requires its outer assignment and receipts"
                    )
                call_id = session.model.last_call_id or ""
                child.services["last_model_call_id"] = call_id
                assignment = question_research.assignment(
                    str(child.services["task_id"])
                )
                receipt_id = parallel_answers.seal(
                    child,
                    assignment=assignment,
                    answer=serial_result.summary,
                    status=serial_result.status,
                    model_call_id=call_id,
                    ledger=ledger,
                    validate_body=lambda: session.validate_accepted(
                        serial_result.summary, call_id, serial_result.status
                    ),
                    source_state=session.source_state(),
                )
                serial_result.data["parallel_answer_receipt"] = receipt_id
            return serial_result
        if parallel_research:
            child.services["experimental_parallel"] = False
            child.services["independent_question_mode"] = False
        child.services["scenario_state"] = ScenarioState([task], frozen=True)
        independent = child.services.get("independent_question") is True
        if independent:
            child.services.pop("submitted_answer", None)
            child.services.pop("independent_answers", None)
            child.services.pop("independent_evidence_numbers", None)
            child.services["research_state"] = ResearchState(
                [task], child, require_need_bindings=False
            )
        child.services["search_message_history"] = [
            ChatMessageSimple(
                message=question,
                token_count=token_counter(question),
                message_type=MessageType.USER,
            ),
            ChatMessageSimple(
                message=task,
                token_count=token_counter(task),
                message_type=MessageType.USER,
            ),
        ]
        researcher_model = ResearchModel(
            llm,
            child,
            user_identity=user_identity,
            reasoning_effort=reasoning_effort,
            history=history,
            updates=updates,
            token_counter=token_counter,
            lean_native_mode=True,
            research_llm=research_llm,
        )
        child_registry = (
            CapabilityRegistry(
                common_specs
                + build_research_specs(
                    cast(ResearchState, child.services["research_state"]), ledger
                )
            )
            if independent
            else registry
        )
        child.services["registry"] = child_registry
        child_partial: str | None = None

        def submit_child_partial(
            args: dict[str, JsonValue], _context: RunContext
        ) -> ToolOutcome:
            nonlocal child_partial
            candidate = str(args["answer"])
            gap = (
                source_publication_gap(
                    candidate, researcher_model.last_call_id, run_context=child
                )
                if extract_citation_numbers(candidate)
                else None
            )
            authority_gap = named_authority_gap(
                candidate, researcher_model.last_call_id, child
            )
            if gap is None and authority_gap is not None:
                gap = ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="An unresolved answer cannot assert an unsupported statutory result.",
                    data=authority_gap,
                )
            child_harness.last_draft = candidate
            child_harness.publication_gap = gap
            if gap is not None:
                return gap
            child_partial = candidate
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Supported partial answer and precise source gap retained.",
            )

        if independent:

            def submit_child_answer(
                args: dict[str, JsonValue], _context: RunContext
            ) -> ToolOutcome:
                candidate = (
                    str(args["answer"])
                    if parallel_research
                    else str(args["answer"]).strip()
                )
                gap = source_publication_gap(
                    candidate, researcher_model.last_call_id, run_context=child
                )
                child_harness.last_draft = candidate
                child_harness.publication_gap = gap
                if gap is not None:
                    return gap
                child.services["submitted_answer"] = candidate
                return ToolOutcome(
                    status=OutcomeStatus.FOUND,
                    summary="Complete supported question answer retained.",
                )

            child_registry.register(
                ToolSpec(
                    name="submit_answer",
                    description="Finish this question with its complete original-supported answer; coverage metadata may accompany this same final decision.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "minLength": 1},
                            "basis": {"type": "string", "enum": ["originals"]},
                        },
                        "required": ["answer", "basis"],
                        "additionalProperties": False,
                    },
                    handler=submit_child_answer,
                    parallel_safe=False,
                    consumes_tool_budget=False,
                )
            )
            child_registry.register(
                ToolSpec(
                    name="submit_partial_answer",
                    description="Finish this question with supported parts and its precise unresolved source gap. Preserve available detail and original citations. Missing source text is not proof that no rule exists.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Complete supported answer and precise source gaps. Use recorded global [n] citations beside supported legal claims; (n) is not a citation. Preserve legal article/paragraph numbering.",
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    handler=submit_child_partial,
                    parallel_safe=False,
                    consumes_tool_budget=False,
                )
            )

        def child_checkpoint(snapshot: dict[str, JsonValue]) -> None:
            callback = child.services.get("record_child_checkpoint")
            if not callable(callback) or harness is None:
                raise ValueError(
                    "Parallel child requires its task-bound checkpoint writer"
                )
            callback(snapshot)
            root_checkpoint()

        child_harness = Harness(
            request=task,
            context=child,
            registry=child_registry,
            decide=researcher_model.decide,
            evidence=ledger,
            on_receipt=record,
            checkpoint=child_checkpoint if parallel_research and independent else None,
            progress=progress,
            report_terminal=False,
            max_workers=2,
            adaptive_tool_parallelism=parallel_research and independent,
            draft_guard=(
                lambda answer: source_publication_gap(
                    answer, researcher_model.last_call_id, run_context=child
                )
            )
            if independent
            else None,
            partial_submission=lambda: child_partial,
        )
        if independent:
            remember_session_originals(child_harness)
            for spec in build_core_specs(
                child_registry, ledger, child_harness.snapshot
            ):
                child_registry.register(spec)
        result = child_harness.run()
        data: dict[str, JsonValue] = {
            "evidence_numbers": sorted(
                {n for receipt in result.receipts for n in receipt.evidence_ids}
            ),
            "questions": result.questions,
            "facts": result.facts,
        }
        if parallel_answers is not None and independent and result.answer:
            if question_research is None:
                raise ValueError(
                    "Parallel answer requires its root assignment registry"
                )
            assignment = question_research.assignment(str(child.services["task_id"]))
            receipt_id = parallel_answers.seal(
                child,
                assignment=assignment,
                answer=result.answer,
                status=result.status,
                model_call_id=researcher_model.last_call_id or "",
                ledger=ledger,
                validate_body=lambda: source_publication_gap(
                    result.answer or "",
                    researcher_model.last_call_id,
                    run_context=child,
                    requires_sources=bool(
                        extract_citation_numbers(result.answer or "")
                    ),
                ),
                source_state={"owner": str(child.services["task_id"])},
            )
            data["parallel_answer_receipt"] = receipt_id
        return ToolOutcome(
            status=result.status,
            summary=result.answer or ""
            if independent
            else (result.answer or "Research incomplete")[:12000],
            data=data,
        )

    workers = WorkerPool(
        context,
        researcher,
        progress=progress,
        max_workers=2 if parallel_research else 4,
    )
    question_research = QuestionResearch(
        context, workers, initial_questions(question), host_assembly=parallel_research
    )
    context.services["registry"] = registry
    external_specs = build_external_specs(
        tools,
        read_allowlist=tuple(
            name.strip()
            for name in os.getenv("ASV3_EXTERNAL_READ_TOOLS", "").split(",")
            if name.strip()
        ),
    )
    external_names = {spec.name for spec in external_specs}
    common_specs = (
        build_corpus_specs(
            broker,
            require_search_targets=True,
            named_provision_reads=workflow_variant == ASV3_TUNED_VARIANT,
        )
        + build_source_specs(broker)
        + build_sandbox_specs(broker)
        + external_specs
        + build_supplemental_specs()
    )
    for spec in (
        common_specs
        + (workers.tool_specs() if not parallel_research else [])
        + (
            question_research.tool_specs()
            if research_profile == "deep" or parallel_research
            else []
        )
        + build_research_specs(research_state, ledger)
    ):
        registry.register(spec)
    model.pending_tasks = lambda: [
        task.model_dump(mode="json") for task in workers.list()
    ]
    context.services["wait_for_task_change"] = workers.wait_for_change

    def ask_user(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        nonlocal clarification
        if child.depth or question_research.answers:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Preserve completed independent answers; retain a precise missing fact beside its conditional outcome.",
            )
        text = str(args["question"]).strip()
        if not public_narration_valid("Clarification", text, context):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Ask a concrete user fact without internal details.",
            )
        clarification = text
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="User clarification requested"
        )

    registry.register(
        ToolSpec(
            name="ask_user",
            description="Ask a missing user fact that materially changes the outcome; publish the question and end this turn. Use source tools for missing legal text. Call on its own.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 1, "maxLength": 1600}
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            handler=ask_user,
            parallel_safe=False,
            consumes_tool_budget=False,
        )
    )

    def source_publication_gap(
        answer: str,
        model_call_id: str | None,
        *,
        requires_sources: bool | None = None,
        run_context: RunContext | None = None,
    ) -> ToolOutcome | None:
        numbers = extract_citation_numbers(answer)
        unknown = sorted(set(numbers) - ledger.citation_mapping().keys())
        if unknown:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Use recorded original citation numbers.",
                data={"unknown_citations": unknown},
            )
        delivered = ledger.completely_delivered(model_call_id or "")
        undelivered = [n for n in numbers if n not in delivered]
        if undelivered:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Read cited originals that have not reached the model.",
                data={"undelivered_citations": undelivered},
            )
        if (
            profile.requires_sources if requires_sources is None else requires_sources
        ) and not numbers:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="A legal answer needs recorded original citations. Retrieve the operative source or disclose the precise gap using submit_partial_answer.",
                data={"missing": "original legal evidence"},
            )
        tuned = context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        gaps: list[ToolOutcome] = []
        authority_gap = named_authority_gap(answer, model_call_id, run_context)
        if authority_gap is not None:
            gap = ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Read and cite the named governing original beside its actual legal assertion; a lower source's reference does not supply that original.",
                data=authority_gap,
            )
            if not tuned:
                return gap
            gaps.append(gap)
        if tuned and answer_llm is None:
            reference_gap = cited_lower_statute_gap(answer, ledger, delivered)
            if reference_gap is not None:
                gaps.append(
                    ToolOutcome(
                        status=OutcomeStatus.PARTIAL,
                        summary="Read the governing provisions explicitly referenced by the cited lower originals before assessing their legal effect.",
                        data=reference_gap,
                    )
                )
        if source_reviews is not None:
            review_gap = source_reviews.publication_gap(
                answer, model_call_id or "", run_context or context, ledger
            )
            if review_gap is not None:
                if not tuned:
                    return review_gap
                gaps.append(review_gap)
        return combine_source_publication_gaps(gaps)

    def publication_guard(
        answer: str,
        *,
        host_gap_assembly: bool = False,
        requires_sources: bool | None = None,
    ) -> ToolOutcome | None:
        if question_research.assignments or question_research.answers:
            gap = question_research.preservation_gap(answer)
            if gap is not None:
                return gap
        if parallel_answers is not None and question_research.answers:
            for item in question_research.answers:
                gap = validate_parallel_answer(item)
                if gap is not None:
                    return gap
            return None
        if (
            host_gap_assembly
            and not extract_citation_numbers(answer)
            and question_research.answers
            and all(
                item.get("status") != OutcomeStatus.FOUND.value
                for item in question_research.answers
            )
        ):
            authority_gap = named_authority_gap(answer, model.last_call_id)
            if authority_gap is not None:
                return ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="An unresolved answer cannot assert an unsupported statutory result.",
                    data=authority_gap,
                )
            return None
        return source_publication_gap(
            answer, model.last_call_id, requires_sources=requires_sources
        )

    question_research.repair_guard = lambda answer: source_publication_gap(
        answer, model.last_call_id
    )

    def validate_parallel_answer(item: dict[str, JsonValue]) -> ToolOutcome | None:
        if parallel_answers is None:
            raise ValueError("Parallel body validation requires its host receipts")
        task_id = str(item["task_id"])
        assignment = question_research.assignment(task_id)
        for key in (
            "question_id",
            "question",
            "answer_title",
            "parent_question_ids",
            "outcome_ids",
        ):
            if item.get(key) != assignment.get(key):
                raise ValueError("Parallel answer assignment changed")
        body = str(item["answer"])
        if item.get("host_gap") is True:
            if (
                item.get("parallel_answer_receipt") is not None
                or body != question_research.incomplete_answer(context.language)
                or item.get("status") != OutcomeStatus.PARTIAL.value
            ):
                raise ValueError("Unaccepted parallel body is not a precise host gap")
            return None
        receipt_id = item.get("parallel_answer_receipt")
        if not isinstance(receipt_id, str):
            raise ValueError("Parallel body is missing its accepted receipt")
        owner = context.child()
        owner.services["task_id"] = task_id
        owner.services["assignment_id"] = assignment["question_id"]
        owner.services["task_outcome_ids"] = assignment.get("outcome_ids", [])
        accepted_call = parallel_answers.model_call_id(receipt_id)
        if workers is None:
            raise ValueError("Serial publication requires its owned checkpoint")
        serial_snapshot = workers.checkpoint(task_id)
        if serial_snapshot is None:
            raise ValueError("Serial publication checkpoint is missing")
        serial_state, serial_gap = validate_serial_session_answer(
            snapshot=serial_snapshot,
            outer_context=owner,
            request=str(assignment["question"]),
            scenario_request=question,
            history=history,
            ledger=ledger,
            body=body,
            call_id=accepted_call,
            status=OutcomeStatus(str(item["status"])),
        )
        parallel_answers.verify(
            context,
            receipt_id=receipt_id,
            task_id=task_id,
            assignment=assignment,
            answer=body,
            status=OutcomeStatus(str(item["status"])),
            ledger=ledger,
            validate_body=lambda: serial_gap,
            source_state=serial_state,
        )
        return None

    if parallel_answers is not None:
        question_research.accepted_answer_guard = validate_parallel_answer

    def submit_answer(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        if child.depth or not standalone_answer_call:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Only the coordinator may submit a complete answer, on its own.",
            )
        if question_research.assignments or question_research.answers:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Arrange the complete independent answer bodies using assemble_answers.",
            )
        candidate = str(args["answer"]).strip()
        if not candidate:
            return ToolOutcome(
                status=OutcomeStatus.INVALID, summary="Supply an answer."
            )
        # The model selects this candidate's support type; it does not change run policy.
        gap = publication_guard(
            candidate, requires_sources=args["basis"] == "originals"
        )
        if harness is not None:
            harness.last_draft = candidate
            harness.publication_gap = gap
        if gap is not None:
            return gap
        context.services["submitted_answer"] = candidate
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Complete answer submitted"
        )

    registry.register(
        ToolSpec(
            name="submit_answer",
            description="Publish a complete answer and end this turn, on its own. basis=conversation only for social dialogue with no legal claims; scenario only for supplied facts or arithmetic with no legal effects; originals for legal answers supported by fully delivered original citations. Otherwise research or ask_user.",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "minLength": 1},
                    "basis": {
                        "type": "string",
                        "enum": ["conversation", "scenario", "originals"],
                    },
                },
                "required": ["answer", "basis"],
                "additionalProperties": False,
            },
            handler=submit_answer,
            parallel_safe=False,
            consumes_tool_budget=False,
        )
    )

    def check_assembly(answer: str) -> ToolOutcome | None:
        gap = publication_guard(answer, host_gap_assembly=True)
        if harness is not None:
            harness.publication_gap = gap
            if gap is not None:
                harness.last_draft = answer
        return gap

    question_research.publish_guard = check_assembly

    def submit_partial(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        nonlocal partial
        if child.depth:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Return available originals and precise gaps to the coordinator; only it can publish.",
            )
        candidate = str(args["answer"]).strip()
        gap = publication_guard(
            candidate,
            requires_sources=bool(
                extract_citation_numbers(candidate)
                or question_research.assignments
                or question_research.answers
            ),
        )
        if gap is None:
            authority_gap = named_authority_gap(candidate, model.last_call_id)
            if authority_gap is not None:
                gap = ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="A precise missing-original notice cannot assert an unsupported statutory result.",
                    data=authority_gap,
                )
        if (
            research_profile == "experimental" or workflow_variant == ASV3_TUNED_VARIANT
        ) and harness is not None:
            harness.last_draft = candidate
            harness.publication_gap = gap
        if gap is not None:
            return gap
        partial = candidate
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Precise supported partial answer submitted",
        )

    registry.register(
        ToolSpec(
            name="submit_partial_answer",
            description="End this turn with supported parts and a precise unresolved source gap. Cite each supported legal assertion. Do not turn missing evidence into a claim that no law exists. Call on its own.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string", "minLength": 1}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            handler=submit_partial,
            parallel_safe=False,
            consumes_tool_budget=False,
        )
    )
    harness = Harness(
        request=question,
        context=context,
        registry=registry,
        decide=model.decide,
        evidence=ledger,
        on_receipt=record,
        checkpoint=checkpoint,
        checkpoint_snapshot=(lambda: {}) if parallel_research else None,
        progress=progress,
        draft_guard=publication_guard,
        partial_submission=lambda: clarification or partial,
        report_terminal=False,
        on_decision=on_decision,
    )
    for spec in build_core_specs(registry, ledger, harness.snapshot):
        registry.register(spec)

    def revalidate(items: list[EvidenceItem], *, resuming: bool = False) -> None:
        canonical: list[EvidenceItem] = []
        for item in items:
            if item.metadata.get("external"):
                if (
                    resuming
                    or context.corpus_only
                    or item.metadata.get("external_tool_name") not in external_names
                ):
                    raise PermissionError(
                        "External evidence requires fresh authorized retrieval"
                    )
            else:
                canonical.append(item)
        context.check_active()
        broker.revalidate_evidence(canonical, context)

    try:
        if previous is None:
            retain_session_research(
                load_asv3_session_checkpoint(
                    chat_session_id=chat_session_id,
                    user_message_id=user_message_id,
                    user_id=user.id,
                ),
                context,
                ledger,
                broker.revalidate_evidence,
            )
            remember_session_originals(harness)
        if previous is not None:
            remembered = previous.get("session_research")
            if isinstance(remembered, dict):
                context.services["session_research"] = remembered
            harness.restore(previous)
            harness.publication_gap = None
            revalidate(
                [item for n in ledger.citation_mapping() if (item := ledger.get(n))],
                resuming=True,
            )
            saved_outcomes = previous.get("outcome_map")
            if isinstance(saved_outcomes, dict):
                outcome_map.restore(saved_outcomes, ledger)
            saved_reviews = previous.get("legal_source_reviews")
            if source_reviews is not None and isinstance(saved_reviews, dict):
                source_reviews.restore(saved_reviews, context, question, ledger)
            saved_requirements = previous.get("authority_requirements")
            if authority_requirements is not None and isinstance(
                saved_requirements, dict
            ):
                authority_requirements.restore(
                    saved_requirements,
                    context,
                    question,
                    allow_legacy_upgrade=parallel_research,
                    ledger=ledger,
                    retained_answer=harness.last_draft,
                )
            if parallel_answers is not None:
                saved_parallel = previous.get("parallel_answers")
                if not isinstance(saved_parallel, dict):
                    raise ValueError(
                        "Parallel checkpoint is missing its accepted answer receipts"
                    )
                parallel_answers.restore(saved_parallel, context, question, ledger)
            question_research.restore(previous.get("question_research"))
            worker_state = previous.get("workers")
            if isinstance(worker_state, dict):
                workers.restore(worker_state)
            progress_state = previous.get("progress_state")
            if isinstance(progress_state, dict):
                progress.restore(progress_state)
            saved_sequence = previous.get("sequence", 0)
            if not isinstance(saved_sequence, int):
                raise ValueError("Invalid checkpoint sequence")
            checkpoint_sequence = saved_sequence
            if parallel_answers is not None and question_research.answers:
                question_research.assemble_retained_answers()
        if parallel_research:
            publish_root_control()
            checkpoint_writer = ParallelCheckpointWriter(
                persist_checkpoint, capture=capture_parallel_checkpoint
            )
        result = harness.run()
        context.check_active()
        if parallel_research and workers is not None:
            workers.close()
        if workflow_variant == ASV3_TUNED_VARIANT and not (
            clarification or partial or result.answer
        ):
            publication_status = result.status
            publication_stop_reason = result.stop_reason
            root_checkpoint(durable=True)
            failure = localized_notifications(
                profile.language if profile.language != "und" else "en"
            )["failed"][1]
            raise OnyxError(OnyxErrorCode.LLM_PROVIDER_ERROR, failure)
        final = (
            clarification
            or partial
            or result.answer
            or profile.notifications["failed"][1]
        )
        if not parallel_research:
            final = final.strip()
        if (
            workflow_variant == ASV3_GUARDRAILS_V2_VARIANT
            and result.answer
            and not clarification
            and not partial
        ):
            cited_first = list(extract_citation_numbers(final))
            review_numbers = list(
                dict.fromkeys([*cited_first, *ledger.citation_numbers()])
            )
            review_evidence: list[ReviewEvidence] = []
            for number in review_numbers:
                item = ledger.get(number)
                if item is None or item.search_doc is None:
                    continue
                review_evidence.append(
                    ReviewEvidence(
                        citation=number,
                        source_id=item.source_id,
                        text=item.text,
                        metadata=item.metadata,
                    )
                )
                if len(review_evidence) == 40:
                    break
            review_outcome = review_and_repair_answer(
                question=question,
                candidate_answer=final,
                evidence=review_evidence,
                repair_llm=repair_llm,
                user_identity=user_identity,
            )
            final = review_outcome.answer
            context.services["guardrails_v2_review"] = review_outcome.model_dump(
                mode="json", exclude={"answer"}
            )
        publication_status = (
            result.status
            if result.answer and not clarification and not partial
            else OutcomeStatus.PARTIAL
        )
        publication_stop_reason = (
            "clarification_requested"
            if clarification
            else "native_partial_published"
            if partial
            else "native_answer_published"
            if result.answer
            else result.stop_reason
        )
        progress.report("final")
        allowed = _presentation_search_docs(ledger, ledger.citation_mapping())
        numbers = extract_citation_numbers(final)
        if set(numbers) - allowed.keys():
            raise ValueError("ASv3 final answer contains unrecorded citation targets")
        revalidate([item for n in numbers if (item := ledger.get(n))])
        state_container.add_search_docs(list(allowed.values()))
        state_container.set_pre_answer_processing_time(time.monotonic() - start)
        ledger.include(numbers)
        root_checkpoint(durable=True)
        processor = DynamicCitationProcessor(
            citation_mode=CitationMode.HYPERLINK
            if include_citations
            else CitationMode.REMOVE
        )
        processor.update_citation_mapping(allowed)
        emitter.emit(
            Packet(
                placement=Placement(turn_index=0),
                obj=AgentResponseStart(
                    final_documents=list(allowed.values()),
                    pre_answer_processing_seconds=time.monotonic() - start,
                ),
            )
        )
        answer_parts: list[str] = []
        for token in (final, None):
            for part in processor.process_token(token):
                context.check_active()
                if isinstance(part, CitationInfo):
                    doc = allowed.get(part.citation_number)
                    if doc and (
                        doc.metadata.get("regulatory_chunk_id")
                        or doc.metadata.get("asv3_native_locator")
                    ):
                        part.preview_url = f"/api/asv3/citation/{assistant_message_id}/{part.citation_number}"
                    state_container.add_emitted_citation(part.citation_number)
                    emitter.emit(Packet(placement=Placement(turn_index=0), obj=part))
                else:
                    answer_parts.append(part)
                    emitter.emit(
                        Packet(
                            placement=Placement(turn_index=0),
                            obj=AgentResponseDelta(content=part),
                        )
                    )
        state_container.set_citation_mapping(processor.citation_to_doc)
        state_container.set_answer_tokens("".join(answer_parts))
        final_published = True
        progress.report("completed", status="completed")
        root_checkpoint(durable=True)
        if checkpoint_writer is not None:
            checkpoint_writer.close()
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=SectionEnd()))
    except RunStopped as error:
        logger.info("ASv3 stopped before publication: %s", str(error))
        progress.report(
            "cancelled" if context.is_cancelled() else "interrupted", status="failed"
        )
        raise
    except Exception:
        logger.exception("ASv3 native research failed")
        if workflow_variant != ASV3_TUNED_VARIANT or publication_stop_reason is None:
            publication_stop_reason = "native_runtime_error"
        progress.report("failed", status="failed")
        raise
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            if workers:
                workers.close()
        finally:
            teardown_error = sys.exc_info()[0] is not None
            if checkpoint_writer is not None:
                try:
                    checkpoint_writer.close()
                except BaseException:
                    if not primary_error and not teardown_error:
                        raise
                    logger.exception(
                        "Parallel checkpoint persistence failed during teardown"
                    )
