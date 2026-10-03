"""Isolated ASv3 chat entry point, durable research and citation publication."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import authority_obligations, unresolved_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.external_tools import build_external_specs
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import (
    LanguageProfile,
    ResearchModel,
    VerificationResult,
    parse_json_object,
)
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.progress import ProgressEvent, ProgressReporter
from onyx.asv3.publication import publication_gap, question_inventory
from onyx.asv3.quotations import unmatched_quoted_terms
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import ResearchState, build_research_specs
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.asv3.scenario import initial_questions
from onyx.asv3.search_adapter import build_search_adapter
from onyx.asv3.source_tools import build_source_specs
from onyx.asv3.supplemental_tools import ScenarioState, build_supplemental_specs
from onyx.asv3.workers import WorkerPool
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.chat.stop_signal_checker import is_connected
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.asv3_corpus import bind_pc_corpus_scope
from onyx.db.asv3_runs import load_asv3_checkpoint, save_asv3_checkpoint
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.asv3.research import (
    FINAL_PROMPT,
    LANGUAGE_PROMPT,
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
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import ChatTraceMetadata, ensure_trace

logger = logging.getLogger(__name__)
P = ParamSpec("P")


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


_LANGUAGE_INSTRUCTION = (
    LANGUAGE_PROMPT
    + """
Also return notifications: a JSON object with started, tools, worker, final, completed,
failed, cancelled, interrupted, resume, native_citation keys. Each value is [short title, one natural sentence], ALL in the
question's requested language. Make them specific to this user's topic: what source,
condition, time limit or alternative is being checked. Never mention tool names,
functions, file paths, SQL, code, model internals or reasoning. Do not assert unverified
findings or pretend a particular document has already been read. 'final' means preparing
the answer, 'completed' means answer ready, 'failed' means the research remains incomplete.
'failed' and 'interrupted' must state that research could not be completed, without
inventing a technical cause, timeout, provider failure or any diagnosis not in this input.
'resume' offers to continue research,
and 'native_citation' labels an excerpt extracted from an original file (derived evidence).
"""
)


def _evidence_record(
    ledger: EvidenceLedger,
    draft: str,
    max_chars: int = 180000,
    *,
    preferred_numbers: list[int] | None = None,
) -> str:
    numbers = list(extract_citation_numbers(draft))
    required = tuple(n for n in numbers if ledger.get(n) is not None)
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
    return ledger.serialize_records(numbers, required=required, max_chars=max_chars)


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
            message.message
            for message in reversed(simple_chat_history)
            if message.message_type == MessageType.USER
        ),
        "",
    )
    context = RunContext(
        timeout_seconds=float("inf"),
        cancelled=lambda: not is_connected(chat_session_id, cache),
    )
    if custom_agent_prompt:
        context.services["assistant_instructions"] = custom_agent_prompt
    scope = IndexFilters(
        **(filters.model_dump() if filters else {}), access_control_list=[]
    )
    if document_set_names_override:
        scope.forced_document_set = document_set_names_override
    scope = bind_pc_corpus_scope(user=user, filters=scope)
    context.scope = scope.model_dump(mode="json")
    history = "\n".join(
        f"{message.message_type.value}: {message.message}"
        for message in simple_chat_history[-8:]
        if message.message_type in (MessageType.USER, MessageType.ASSISTANT)
    )
    model = ResearchModel(
        llm,
        context,
        user_identity=user_identity,
        reasoning_effort=reasoning_effort,
        history=history,
        token_counter=token_counter,
    )
    previous: dict[str, JsonValue] | None = None
    if resume_message_id is not None:
        previous = load_asv3_checkpoint(message_id=resume_message_id, user_id=user.id)
        if (
            previous is None
            or previous.get("scope") != context.scope
            or previous.get("request") != question
        ):
            raise ValueError(
                "ASv3 resume requires the same authorized scope and question"
            )
        profile = LanguageProfile.model_validate(previous.get("public_profile"))
    else:
        profile = LanguageProfile.model_validate(
            parse_json_object(
                model.invoke_text(
                    _LANGUAGE_INSTRUCTION,
                    question,
                    LLMFlow.ASV3_LANGUAGE,
                    max_tokens=1800,
                )
            )
        )
    required_phases = {
        "started",
        "tools",
        "worker",
        "final",
        "completed",
        "failed",
        "cancelled",
    }
    if not required_phases.issubset(profile.notifications) or any(
        len(value) != 2 for value in profile.notifications.values()
    ):
        raise ValueError("Incomplete ASv3 language profile")
    context.language = profile.language
    state_container.set_stop_notice(profile.notifications["cancelled"][1])
    context.services["native_citation_label"] = profile.notifications.get(
        "native_citation", ["", ""]
    )[0]
    # Both application consent and explicit user intent are required.
    context.corpus_only = not (allow_external and profile.external_requested)
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    registry = CapabilityRegistry()
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    broker = CorpusBroker(user, scope, vision_llm=llm)

    def search_history(child: RunContext) -> list[ChatMessageSimple]:
        private = child.services.get("search_message_history")
        if isinstance(private, list) and all(
            isinstance(item, ChatMessageSimple) for item in private
        ):
            return cast(list[ChatMessageSimple], private)
        return list(simple_chat_history)

    broker.search_adapter = build_search_adapter(
        search,
        question,
        broker,
        message_history=search_history,
        user_memory_context=user_memory_context,
        user_info=user_info,
        inject_memories_in_prompt=inject_memories_in_prompt,
        user_identity=user_identity,
    )
    if previous is not None:
        context.run_id = str(previous["run_id"])
    scenarios = ScenarioState(initial_questions(question), frozen=True)
    context.services["scenario_state"] = scenarios
    research_state = ResearchState(initial_questions(question), context)
    context.services["research_state"] = research_state
    emitted: list[dict[str, JsonValue]] = []
    checkpoint_lock = threading.RLock()
    checkpoint_sequence = 0
    harness: Harness | None = None
    workers: WorkerPool | None = None
    final_published = False
    latest_review: VerificationResult | None = None
    publication_status = OutcomeStatus.PARTIAL
    publication_stop_reason: str | None = None
    final_publication_gap: ToolOutcome | None = None
    approved_draft: str | None = None
    guard_cache: dict[
        str, tuple[ToolOutcome | None, VerificationResult, str | None, list[str]]
    ] = {}
    approved_review: VerificationResult | None = None
    approved_call_id: str | None = None
    approved_questions: list[str] = []

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
                title=event.title
                if event.public_narration
                or phase == "research"
                or (
                    phase == "worker"
                    and event.task_id
                    and workers
                    and any(
                        task.task_id == event.task_id and task.public_title
                        for task in workers.list()
                    )
                )
                else words[0],
                message=event.message
                if event.public_narration
                or phase == "research"
                or (
                    phase == "worker"
                    and event.task_id
                    and workers
                    and any(
                        task.task_id == event.task_id and task.public_message
                        for task in workers.list()
                    )
                )
                else words[1],
                task_id=event.task_id,
                parent_task_id=event.parent_task_id,
                active_tasks=event.active_workers,
                completed_tasks=event.completed_workers,
            )
        )
        with checkpoint_lock:
            emitted.append(packet.model_dump(mode="json"))
            if len(emitted) > 100:
                latest: dict[str, dict[str, JsonValue]] = {}
                for saved in emitted:
                    latest[str(saved.get("task_id") or "coordinator")] = saved
                retained = list(
                    {
                        str(saved["event_id"]): saved
                        for saved in [emitted[0], *latest.values(), *emitted[-40:]]
                    }.values()
                )
                emitted[:] = sorted(
                    retained,
                    key=lambda saved: (
                        saved["sequence"] if isinstance(saved["sequence"], int) else 0
                    ),
                )
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=packet))

    progress = ProgressReporter(context.run_id, context.language, emit_progress)
    context.services["progress"] = progress

    def checkpoint(snapshot: dict[str, JsonValue]) -> None:
        nonlocal checkpoint_sequence
        with checkpoint_lock:
            context.check_active()
            checkpoint_sequence += 1
            snapshot.update(
                sequence=checkpoint_sequence,
                prompt_version=PROMPT_VERSION,
                scope=context.scope,
                progress=list(emitted),
                progress_state=progress.export(),
                public_profile=profile.model_dump(mode="json"),
                workers=workers.export() if workers else {},
                publication_review=latest_review.model_dump(mode="json")
                if latest_review
                else None,
                publication_status=publication_status.value,
                publication_stop_reason=publication_stop_reason,
                final_publication_gap=final_publication_gap.model_dump(mode="json")
                if final_publication_gap
                else None,
                draft_approval={
                    "text_hash": hashlib.sha256(approved_draft.encode()).hexdigest(),
                    "verification_call_id": approved_call_id,
                    "questions": approved_questions,
                }
                if approved_draft is not None
                else None,
                scenario=scenarios.snapshot(),
                question_message_id=user_message_id,
            )
            save_asv3_checkpoint(
                message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
            )

    def record(receipt: ToolReceipt) -> None:
        for number in receipt.evidence_ids:
            item = ledger.get(number)
            if item and item.search_doc is None and item.metadata.get("source_sha256"):
                anchored = broker.attach_native_citation(
                    item, number, assistant_message_id, context
                )
                ledger.add([anchored], context)
        docs = [ledger.get(number) for number in receipt.evidence_ids]
        state_container.add_search_docs(
            [item.search_doc for item in docs if item and item.search_doc]
        )

    def verify(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        numbers = args.get("evidence_numbers", args.get("citations", []))
        claim = str(args.get("claim", ""))
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
                    "claim": claim,
                    "evidence": _evidence_record(ledger, anchors),
                    "available_evidence": ledger.summaries(max_chars=6000),
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
            summary="Original-source assessment format failed; no claim was approved"
            if report.get("format_error")
            else "Claim checked against original evidence",
            data=report,
        )

    context.services["verify_claim"] = verify

    def researcher(
        task: str, child: RunContext, updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        local_scenario = ScenarioState([task], frozen=True)
        child.services["scenario_state"] = local_scenario
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
            history=question,
            updates=updates,
            token_counter=token_counter,
        )
        result = Harness(
            request=task,
            context=child,
            registry=registry,
            decide=researcher_model.decide,
            evidence=ledger,
            on_receipt=record,
            progress=progress,
            report_terminal=False,
            max_workers=2,
        ).run()
        return ToolOutcome(
            status=result.status,
            summary=(result.answer or "Research incomplete")[:12000],
            data={
                "evidence_numbers": sorted(
                    {n for receipt in result.receipts for n in receipt.evidence_ids}
                ),
                "questions": result.questions,
                "facts": result.facts,
            },
        )

    workers = WorkerPool(context, researcher, progress=progress)
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
    registry_specs = (
        build_corpus_specs(broker)
        + build_source_specs(broker)
        + build_sandbox_specs(broker)
        + workers.tool_specs()
        + build_supplemental_specs()
        + build_research_specs(research_state, ledger)
        + external_specs
    )
    for spec in registry_specs:
        registry.register(spec)
    model.pending_tasks = lambda: [
        task.model_dump(mode="json") for task in workers.list()
    ]

    context.services["wait_for_task_change"] = workers.wait_for_change

    def review_answer(
        draft: str,
        *,
        research: bool,
        preservation_reference: str | None = None,
        previous_review: VerificationResult | None = None,
    ) -> VerificationResult:
        nonlocal latest_review
        if harness is None:
            raise RuntimeError("Research state is not ready for verification")
        view = harness.view()
        assertion_units = assertion_inventory(draft)
        evidence = _evidence_record(
            ledger,
            draft + ("\n" + preservation_reference if preservation_reference else ""),
            max_chars=max(8000, min(180000, (llm.config.max_input_tokens - 18000) * 2)),
            preferred_numbers=research_state.preferred_citations(),
        )
        if research:
            context.consume_research_decision()
        latest_review = model.invoke_verification(
            VERIFICATION_PROMPT,
            json.dumps(
                {
                    "language": context.language,
                    "scenario": question,
                    "questions": question_inventory(view.questions),
                    "claim": draft,
                    "assertion_units": assertion_units,
                    "preservation_reference": {
                        "draft": preservation_reference,
                        "previous_review": previous_review.model_dump(mode="json")
                        if previous_review
                        else None,
                    }
                    if preservation_reference
                    else None,
                    "evidence": evidence,
                    "available_evidence": ledger.summaries(max_chars=6000),
                    "authority_obligations": authority_obligations(draft, ledger),
                    "research_state": research_state.view(max_chars=24000),
                    "unmatched_quoted_terms": unmatched_quoted_terms(
                        draft, question, ledger
                    ),
                    "require_sources": profile.requires_sources,
                    "pending_tasks": model.pending_tasks(),
                },
                ensure_ascii=False,
            ),
            max_tokens=max(
                6000,
                min(
                    16000,
                    3000
                    + 220 * len(assertion_units)
                    + 130
                    * sum(len(unit["evidence_numbers"]) for unit in assertion_units)
                    + 250 * len(view.questions),
                ),
            ),
            consume_budget=not research,
        )
        checkpoint(harness.snapshot())
        return latest_review

    def draft_guard(draft: str) -> ToolOutcome | None:
        nonlocal approved_draft, approved_review, approved_call_id, approved_questions
        approved_draft = None
        approved_review = None
        approved_call_id = None
        approved_questions = []
        if profile.requires_sources and not ledger.citation_mapping():
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="No original citable evidence is recorded. Source access failures do not justify legal conclusions from memory.",
                data={
                    "missing": "original legal evidence",
                    "available_tasks": model.pending_tasks(),
                },
            )
        missing_authority = (
            unresolved_authority_gap(draft, ledger)
            if profile.requires_sources
            else None
        )
        if missing_authority is not None:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="An explicitly used statutory basis still needs its original provision.",
                data=missing_authority,
            )
        assert harness is not None
        questions = harness.view().questions
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    draft,
                    questions,
                    scenarios.snapshot(),
                    ledger.authority_metadata(),
                    research_state.export(),
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = guard_cache.get(fingerprint)
        if cached is not None:
            gap, review, call_id, _ = cached
            if gap is None:
                approved_draft, approved_review, approved_call_id = (
                    draft,
                    review,
                    call_id,
                )
                approved_questions = list(questions)
            return gap
        review = review_answer(draft, research=True)
        call_id = model.last_call_id
        gap = publication_gap(
            draft,
            review,
            questions,
            ledger,
            require_sources=profile.requires_sources,
            verification_call_id=call_id,
            require_direct_authority=True,
            scenario=question,
            require_quotation_checks=True,
            require_assertion_checks=True,
            research_state=research_state,
        )
        if gap is None:
            approved_draft, approved_review, approved_call_id = draft, review, call_id
            approved_questions = list(questions)
        if review.format_error is None:
            guard_cache[fingerprint] = (gap, review, call_id, list(questions))
            if len(guard_cache) > 8:
                guard_cache.pop(next(iter(guard_cache)))
        return gap

    harness = Harness(
        request=question,
        context=context,
        registry=registry,
        decide=model.decide,
        evidence=ledger,
        progress=progress,
        on_receipt=record,
        checkpoint=checkpoint,
        report_terminal=False,
        draft_guard=draft_guard,
    )
    for spec in build_core_specs(
        registry,
        ledger,
        lambda: {
            "scenario": scenarios.snapshot(),
            "workers": [task.model_dump(mode="json") for task in workers.list()],
            "receipts": [
                receipt.model_dump(mode="json") for receipt in harness.receipts[-20:]
            ],
        },
    ):
        registry.register(spec)
    try:

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
                    context.check_active()
                else:
                    canonical.append(item)
            broker.revalidate_evidence(canonical, context)

        if previous is not None:
            context.run_id = str(previous["run_id"])
            previous["language"] = context.language
            harness.restore(previous)
            # A durable snapshot is not authorization for current access/publication.
            revalidate(
                [
                    item
                    for n in ledger.citation_mapping()
                    if (item := ledger.get(n)) is not None
                ],
                resuming=True,
            )
            worker_state = previous.get("workers")
            if isinstance(worker_state, dict):
                workers.restore(worker_state)
            scenario = previous.get("scenario")
            if isinstance(scenario, dict):
                scenarios.record(
                    [str(q) for q in scenario.get("questions", [])],
                    [str(f) for f in scenario.get("facts", [])],
                )
            progress.run_id = context.run_id
            progress_state = previous.get("progress_state")
            if isinstance(progress_state, dict):
                progress.restore(progress_state)
            sequence = previous.get("sequence", 0)
            if not isinstance(sequence, int):
                raise ValueError("Invalid checkpoint sequence")
            checkpoint_sequence = sequence
        result = harness.run()
        context.check_active()
        # Finish admitted research before constructing the immutable synthesis input.
        # A shared decision stop is not evidence that in-flight source reads finished.
        if result.status != OutcomeStatus.FOUND:
            workers.settle()
        checkpoint(harness.snapshot())
        draft = result.answer or harness.last_draft or ""
        complete = result.status == OutcomeStatus.FOUND
        if not draft:
            draft = json.dumps(
                {
                    "status": result.status.value,
                    "scenario": scenarios.snapshot(),
                    "warning": "Research is incomplete. State the exact unresolved questions.",
                    "last_review": latest_review.model_dump(mode="json")
                    if latest_review
                    else None,
                },
                ensure_ascii=False,
            )
        progress.report("final")
        if complete:
            if (
                approved_draft != draft
                or approved_review is None
                or approved_call_id is None
                or approved_questions != harness.view().questions
            ):
                raise ValueError(
                    "Verified draft approval does not match the final answer"
                )
            final, final_review = draft, approved_review
            # The verifier saw this exact text and its complete original evidence.
            final_gap = publication_gap(
                final,
                final_review,
                approved_questions,
                ledger,
                require_sources=profile.requires_sources,
                verification_call_id=approved_call_id,
                require_direct_authority=True,
                scenario=question,
                require_quotation_checks=True,
                require_assertion_checks=True,
                research_state=research_state,
            )
        else:
            previous_review = latest_review
            evidence = _evidence_record(
                ledger,
                draft,
                max_chars=max(
                    8000, min(180000, (llm.config.max_input_tokens - 18000) * 2)
                ),
                preferred_numbers=research_state.preferred_citations(),
            )
            final = model.invoke_text(
                FINAL_PROMPT,
                json.dumps(
                    {
                        "language": context.language,
                        "question": question,
                        "scenario": scenarios.snapshot(),
                        "draft": draft,
                        "review": latest_review.model_dump(mode="json")
                        if latest_review
                        else None,
                        "evidence": evidence,
                        "research_status": result.status.value,
                        "research_state": research_state.view(max_chars=24000),
                        "assistant_instructions": custom_agent_prompt or "",
                    },
                    ensure_ascii=False,
                ),
                LLMFlow.ASV3_FINAL,
                max_tokens=9000,
            )
            harness.last_draft = final
            checkpoint(harness.snapshot())
            # Review the actual published wording, not just the coordinator's draft.
            final_review = review_answer(
                final,
                research=False,
                preservation_reference=draft
                if result.answer or harness.last_draft
                else None,
                previous_review=previous_review,
            )
            final_gap = publication_gap(
                final,
                final_review,
                harness.view().questions,
                ledger,
                require_sources=profile.requires_sources,
                allow_explicit_gaps=not complete,
                verification_call_id=model.last_call_id,
                require_direct_authority=final_review.status == "supported",
                scenario=question,
                require_quotation_checks=True,
                require_assertion_checks=True,
                research_state=research_state,
            )
        if (
            final_gap is not None
            and context.budget.snapshot()["decisions"]
            < context.budget.limits["decisions"] - 4
        ):
            # The final review feeds the same research loop, with its exact candidate.
            harness.last_draft = final
            harness.publication_gap = final_gap
            checkpoint(harness.snapshot())
            context.services["final_repair"] = True
            try:
                repaired = harness.run()
            finally:
                context.services.pop("final_repair", None)
            workers.settle()
            if (
                repaired.status == OutcomeStatus.FOUND
                and approved_draft == repaired.answer
                and approved_review is not None
            ):
                final, final_review = approved_draft, approved_review
                complete = True
                final_gap = publication_gap(
                    final,
                    final_review,
                    harness.view().questions,
                    ledger,
                    require_sources=profile.requires_sources,
                    verification_call_id=approved_call_id,
                    require_direct_authority=True,
                    scenario=question,
                    require_quotation_checks=True,
                    require_assertion_checks=True,
                    research_state=research_state,
                )
            else:
                # Keep the repaired candidate and only publish supported portions.
                draft = harness.last_draft or final
                if (
                    context.budget.snapshot()["decisions"]
                    <= context.budget.limits["decisions"] - 4
                ):
                    final = model.invoke_text(
                        FINAL_PROMPT,
                        json.dumps(
                            {
                                "language": context.language,
                                "question": question,
                                "scenario": scenarios.snapshot(),
                                "draft": draft,
                                "review": latest_review.model_dump(mode="json")
                                if latest_review
                                else None,
                                "research_state": research_state.view(max_chars=24000),
                                "evidence": _evidence_record(
                                    ledger,
                                    draft,
                                    preferred_numbers=research_state.preferred_citations(),
                                ),
                                "research_status": "incomplete",
                                "assistant_instructions": custom_agent_prompt or "",
                            },
                            ensure_ascii=False,
                        ),
                        LLMFlow.ASV3_FINAL,
                        max_tokens=9000,
                    )
                    harness.last_draft = final
                    final_review = review_answer(
                        final,
                        research=False,
                        preservation_reference=draft,
                        previous_review=latest_review,
                    )
                    final_gap = publication_gap(
                        final,
                        final_review,
                        harness.view().questions,
                        ledger,
                        require_sources=profile.requires_sources,
                        allow_explicit_gaps=True,
                        verification_call_id=model.last_call_id,
                        require_direct_authority=final_review.status == "supported",
                        scenario=question,
                        require_quotation_checks=True,
                        require_assertion_checks=True,
                        research_state=research_state,
                    )
                complete = False
        # Research allocation and publication completeness are separate outcomes.
        # A reviewed finalization can satisfy every obligation after research stops.
        if final_gap is None and not complete and final_review.status == "supported":
            complete = (
                publication_gap(
                    final,
                    final_review,
                    harness.view().questions,
                    ledger,
                    require_sources=profile.requires_sources,
                    verification_call_id=model.last_call_id,
                    require_direct_authority=True,
                    scenario=question,
                    require_quotation_checks=True,
                    require_assertion_checks=True,
                    research_state=research_state,
                )
                is None
            )
        final_publication_gap = final_gap
        publication_stop_reason = (
            "publication_guard_rejected"
            if final_gap is not None
            else "verified_draft_ready"
            if complete
            else result.stop_reason or "incomplete_research"
        )
        # Retain the exact rejection before any safe limitation notice replaces it.
        with graph_step("asv3.publication_guard") as step:
            step.output_value = {
                "stop_reason": result.stop_reason,
                "publication_stop_reason": publication_stop_reason,
                "publication_gap": final_gap.model_dump(mode="json")
                if final_gap
                else None,
                "verification_call_id": approved_call_id
                if complete
                else model.last_call_id,
                "answer_hash": hashlib.sha256(final.encode()).hexdigest(),
            }
        checkpoint(harness.snapshot())
        if final_gap is not None:
            # A source-free limitation notice is safe even when synthesis failed.
            # Reviewer explanations are not substituted for unsupported legal rules.
            final = profile.notifications["failed"][1]
            complete = False
        elif final_review.status != "supported":
            complete = False
        publication_status = OutcomeStatus.FOUND if complete else OutcomeStatus.PARTIAL
        # Unknown citation IDs never reach the UI as plausible evidence links.
        allowed = ledger.citation_mapping()
        unknown = set(extract_citation_numbers(final)) - allowed.keys()
        if unknown:
            raise ValueError("ASv3 final answer contains unrecorded citation targets")
        revalidate([item for n in allowed if (item := ledger.get(n)) is not None])
        state_container.add_search_docs(list(allowed.values()))
        state_container.set_pre_answer_processing_time(time.monotonic() - start)
        ledger.include(extract_citation_numbers(final))
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
        if complete:
            publication_stop_reason = "verified_draft_published"
        progress.report(
            "completed" if complete else "failed",
            status="completed" if complete else "failed",
        )
        checkpoint(harness.snapshot())
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=SectionEnd()))
    except RunStopped as error:
        logger.info("ASv3 stopped before publication: %s", str(error))
        if context.is_cancelled():
            progress.report("cancelled", status="cancelled")
        else:
            progress.report("failed", status="failed")
        # Stop preserves the already durable record; no late write or answer is accepted.
        raise
    except Exception as error:
        publication_stop_reason = f"runtime_error:{type(error).__name__}"
        publication_status = OutcomeStatus.PARTIAL
        if harness is not None and not context.is_cancelled():
            try:
                checkpoint(harness.snapshot())
            except RunStopped:
                pass
            except Exception:
                logger.exception("Could not persist ASv3 terminal diagnostics")
        progress.report("failed", status="failed")
        logger.exception("ASv3 run failed")
        raise
    finally:
        workers.close()
