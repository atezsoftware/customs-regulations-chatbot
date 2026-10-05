"""Isolated ASv3 chat entry point, durable research and citation publication."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from functools import wraps
from typing import ParamSpec, cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.external_tools import build_external_specs
from onyx.asv3.harness import Harness
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
from onyx.asv3.progress import (
    ProgressEvent,
    ProgressReporter,
    localized_notifications,
    official_corpus_source_name,
    report_source_deliveries,
)
from onyx.asv3.question_research import QuestionResearch
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import ResearchState, build_research_specs
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.asv3.scenario import initial_questions
from onyx.asv3.search_adapter import build_search_adapter
from onyx.asv3.session_research import (
    retain_session_research,
    session_research_checkpoint,
)
from onyx.asv3.source_tools import build_source_specs
from onyx.asv3.supplemental_tools import (
    ScenarioState,
    build_supplemental_specs,
    public_narration_valid,
)
from onyx.asv3.workers import WorkerPool
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.chat.stop_signal_checker import is_connected
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters, SearchDoc
from onyx.db.asv3_corpus import bind_pc_corpus_scope
from onyx.db.asv3_runs import (
    load_asv3_checkpoint,
    load_asv3_session_checkpoint,
    save_asv3_checkpoint,
)
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.asv3.research import (
    PROMPT_VERSION,
    VERIFICATION_PROMPT,
)
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
from onyx.tracing.framework.create import ChatTraceMetadata, ensure_trace

logger = logging.getLogger(__name__)
P = ParamSpec("P")


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
    context = RunContext(
        language="und",
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
        cancelled=lambda: not is_connected(chat_session_id, cache),
        services={"lean_native_mode": True},
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
    if research_profile not in {"normal", "deep"}:
        raise ValueError("Unknown ASv3 research profile")
    context.language = profile.language
    context.services["research_profile"] = research_profile
    context.services["independent_question_mode"] = research_profile == "deep"
    context.corpus_only = not (allow_external and profile.external_requested)
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    registry = CapabilityRegistry()
    broker = CorpusBroker(user, scope, vision_llm=llm)
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
    )
    if previous is not None:
        model.restore_native_sampling(previous)
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)

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
    emitted: list[dict[str, JsonValue]] = []
    checkpoint_lock = threading.RLock()
    checkpoint_sequence = 0
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
        if phase == "completed" and not final_published:
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
        with checkpoint_lock:
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

    def checkpoint(snapshot: dict[str, JsonValue]) -> None:
        nonlocal checkpoint_sequence
        with checkpoint_lock:
            context.check_active()
            checkpoint_sequence += 1
            snapshot.update(
                sequence=checkpoint_sequence,
                prompt_version=PROMPT_VERSION,
                research_profile=research_profile,
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
                publication_status=publication_status.value,
                publication_stop_reason=publication_stop_reason,
                final_publication_gap=harness.publication_gap.model_dump(mode="json")
                if harness and harness.publication_gap
                else None,
                scenario=scenarios.snapshot(),
                question_message_id=user_message_id,
                session_research=session_research_checkpoint(context, question),
            )
            save_asv3_checkpoint(
                message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
            )

    def record(receipt: ToolReceipt) -> None:
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
        numbers = args.get("evidence_numbers", args.get("citations", []))
        anchors = (
            " ".join(f"[{n}]" for n in numbers) if isinstance(numbers, list) else ""
        )
        verifier = ResearchModel(
            llm,
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
        child.services["scenario_state"] = ScenarioState([task], frozen=True)
        independent = child.services.get("independent_question") is True
        if independent:
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
                source_publication_gap(candidate, researcher_model.last_call_id)
                if extract_citation_numbers(candidate)
                else None
            )
            authority_gap = native_named_authority_gap(candidate, ledger)
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
        child_harness = Harness(
            request=task,
            context=child,
            registry=child_registry,
            decide=researcher_model.decide,
            evidence=ledger,
            on_receipt=record,
            progress=progress,
            report_terminal=False,
            max_workers=2,
            draft_guard=(
                lambda answer: source_publication_gap(
                    answer, researcher_model.last_call_id
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
        return ToolOutcome(
            status=result.status,
            summary=result.answer or ""
            if independent
            else (result.answer or "Research incomplete")[:12000],
            data={
                "evidence_numbers": sorted(
                    {n for receipt in result.receipts for n in receipt.evidence_ids}
                ),
                "questions": result.questions,
                "facts": result.facts,
            },
        )

    workers = WorkerPool(context, researcher, progress=progress)
    question_research = QuestionResearch(context, workers, initial_questions(question))
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
        build_corpus_specs(broker, require_search_targets=True)
        + build_source_specs(broker)
        + build_sandbox_specs(broker)
        + external_specs
        + build_supplemental_specs()
    )
    for spec in (
        common_specs
        + workers.tool_specs()
        + (question_research.tool_specs() if research_profile == "deep" else [])
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
        authority_gap = native_named_authority_gap(answer, ledger)
        if authority_gap is not None:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Read and cite the named governing original beside its actual legal assertion; a lower source's reference does not supply that original.",
                data=authority_gap,
            )
        return None

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
        if (
            host_gap_assembly
            and not extract_citation_numbers(answer)
            and question_research.answers
            and all(
                item.get("status") != OutcomeStatus.FOUND.value
                for item in question_research.answers
            )
        ):
            authority_gap = native_named_authority_gap(answer, ledger)
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
            authority_gap = native_named_authority_gap(candidate, ledger)
            if authority_gap is not None:
                gap = ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="A precise missing-original notice cannot assert an unsupported statutory result.",
                    data=authority_gap,
                )
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
            question_research.restore(previous.get("question_research"))
            harness.restore(previous)
            harness.publication_gap = None
            revalidate(
                [item for n in ledger.citation_mapping() if (item := ledger.get(n))],
                resuming=True,
            )
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
        result = harness.run()
        context.check_active()
        final = (
            clarification
            or partial
            or result.answer
            or profile.notifications["failed"][1]
        ).strip()
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
        checkpoint(harness.snapshot())
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
        checkpoint(harness.snapshot())
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=SectionEnd()))
    except RunStopped as error:
        logger.info("ASv3 stopped before publication: %s", str(error))
        progress.report(
            "cancelled" if context.is_cancelled() else "interrupted", status="failed"
        )
        raise
    except Exception:
        logger.exception("ASv3 native research failed")
        publication_stop_reason = "native_runtime_error"
        progress.report("failed", status="failed")
        raise
    finally:
        if workers:
            workers.close()
