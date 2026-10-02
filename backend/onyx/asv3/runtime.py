"""Isolated ASv3 chat entry point, durable research and citation publication."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.external_tools import build_external_specs
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import LanguageProfile, ResearchModel, parse_json_object
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.progress import ProgressEvent, ProgressReporter
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.sandbox import build_sandbox_specs
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
from onyx.db.asv3_runs import load_asv3_checkpoint, save_asv3_checkpoint
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.asv3.research import (
    FINAL_PROMPT,
    LANGUAGE_PROMPT,
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
from onyx.tracing.flows import LLMFlow

logger = logging.getLogger(__name__)

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
'interrupted' describes an unexpected interruption, 'resume' offers to continue research,
and 'native_citation' labels an excerpt extracted from an original file (derived evidence).
"""
)


def _evidence_record(
    ledger: EvidenceLedger, draft: str, max_chars: int = 180000
) -> str:
    numbers = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", draft)))
    # Full original cited provisions precede additional context, never silent clipping.
    numbers += [
        n
        for item in ledger.summaries()
        if isinstance(n := item["citation"], int) and n not in numbers
    ]
    records: list[dict[str, JsonValue]] = []
    used = 0
    for number in numbers:
        item = ledger.get(number)
        if item is None:
            continue
        available = max(0, max_chars - used)
        text = item.text[:available]
        records.append(
            {
                "citation": number,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text": text,
                "truncated": len(text) != len(item.text),
                "citable": item.search_doc is not None,
                "metadata": item.metadata,
            }
        )
        used += len(text)
        if used >= max_chars:
            break
    return json.dumps(records, ensure_ascii=False)


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
        timeout_seconds=900,
        research_reserve_seconds=180,
        cancelled=lambda: not is_connected(chat_session_id, cache),
    )
    scope = IndexFilters(
        **(filters.model_dump() if filters else {}), access_control_list=[]
    )
    if document_set_names_override:
        scope.forced_document_set = document_set_names_override
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
    profile = LanguageProfile.model_validate(
        parse_json_object(
            model.invoke_text(
                _LANGUAGE_INSTRUCTION, question, LLMFlow.ASV3_LANGUAGE, max_tokens=1800
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
    registry = CapabilityRegistry()
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    broker = CorpusBroker(user, scope, vision_llm=llm)
    broker.search_adapter = build_search_adapter(search, question, broker)
    scenarios = ScenarioState()
    scenarios.record([question], [])
    context.services["scenario_state"] = scenarios
    emitted: list[dict[str, JsonValue]] = []
    checkpoint_lock = threading.RLock()
    checkpoint_sequence = 0
    harness: Harness | None = None
    workers: WorkerPool | None = None
    final_published = False

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
                title=event.title if phase == "research" else words[0],
                message=event.message if phase == "research" else words[1],
                task_id=event.task_id,
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
                scope=context.scope,
                progress=list(emitted),
                progress_state=progress.export(),
                public_profile=profile.model_dump(mode="json"),
                workers=workers.export() if workers else {},
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
        child.budget.consume_research_decision()
        report = parse_json_object(
            verifier.invoke_text(
                VERIFICATION_PROMPT,
                json.dumps(
                    {
                        "language": child.language,
                        "scenario": question,
                        "claim": claim,
                        "evidence": _evidence_record(ledger, anchors),
                    },
                    ensure_ascii=False,
                ),
                LLMFlow.ASV3_VERIFICATION,
                max_tokens=2500,
                consume_budget=False,
            )
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Claim checked against original evidence",
            data=report,
        )

    context.services["verify_claim"] = verify

    def researcher(
        task: str, child: RunContext, updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        local_scenario = ScenarioState()
        local_scenario.record([task], [])
        child.services["scenario_state"] = local_scenario
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
        + external_specs
    )
    for spec in registry_specs:
        registry.register(spec)
    model.pending_tasks = lambda: [
        task.model_dump(mode="json") for task in workers.list()
    ]
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

        if resume_message_id is not None:
            previous = load_asv3_checkpoint(
                message_id=resume_message_id, user_id=user.id
            )
            if (
                previous is None
                or previous.get("scope") != context.scope
                or previous.get("request") != question
            ):
                raise ValueError(
                    "ASv3 resume requires the same authorized scope and question"
                )
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
        draft = result.answer or ""
        if not draft:
            draft = json.dumps(
                {
                    "status": result.status.value,
                    "scenario": scenarios.snapshot(),
                    "warning": "Research is incomplete. State the exact unresolved questions.",
                },
                ensure_ascii=False,
            )
        progress.report("final")
        evidence = _evidence_record(
            ledger,
            draft,
            max_chars=max(8000, min(180000, (llm.config.max_input_tokens - 18000) * 2)),
        )
        review = model.invoke_text(
            VERIFICATION_PROMPT,
            json.dumps(
                {
                    "language": context.language,
                    "scenario": question,
                    "claim": draft,
                    "evidence": evidence,
                },
                ensure_ascii=False,
            ),
            LLMFlow.ASV3_VERIFICATION,
            max_tokens=3000,
        )
        final = model.invoke_text(
            FINAL_PROMPT,
            json.dumps(
                {
                    "language": context.language,
                    "question": question,
                    "scenario": scenarios.snapshot(),
                    "draft": draft,
                    "review": review,
                    "evidence": evidence,
                    "research_status": result.status.value,
                    "assistant_instructions": custom_agent_prompt or "",
                },
                ensure_ascii=False,
            ),
            LLMFlow.ASV3_FINAL,
            max_tokens=9000,
        )
        # Unknown citation IDs never reach the UI as plausible evidence links.
        allowed = ledger.citation_mapping()
        unknown = {int(n) for n in re.findall(r"\[(\d+)\]", final)} - allowed.keys()
        if unknown:
            raise ValueError("ASv3 final answer contains unrecorded citation targets")
        revalidate([item for n in allowed if (item := ledger.get(n)) is not None])
        state_container.add_search_docs(list(allowed.values()))
        state_container.set_pre_answer_processing_time(time.monotonic() - start)
        ledger.include(int(n) for n in re.findall(r"\[(\d+)\]", final))
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
                    if doc and isinstance(
                        doc.metadata.get("asv3_citation_preview_url"), str
                    ):
                        part.preview_url = str(
                            doc.metadata["asv3_citation_preview_url"]
                        )
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
    except RunStopped:
        if context.is_cancelled():
            progress.report("cancelled", status="cancelled")
        else:
            progress.report("failed", status="failed")
        # Stop preserves the already durable record; no late write or answer is accepted.
        raise
    except Exception:
        progress.report("failed", status="failed")
        logger.exception("ASv3 run failed")
        raise
    finally:
        workers.close()
