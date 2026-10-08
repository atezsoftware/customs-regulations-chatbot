"""Independent PC-only chat entry point using the existing citation protocol."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, SharedBudget
from onyx.asv3.progress import ProgressEvent, ProgressReporter, localized_notifications
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.search_adapter import build_search_adapter
from onyx.asv3.shared_reads import SharedReads
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.chat.stop_signal_checker import is_connected
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.asv3_runs import save_asv3_checkpoint
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.db.supersearch import bind_supersearch_pc_scope
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_composite.models import SourceAction, WorkflowPolicy, WorkflowResult
from onyx.legal_composite.search import CompositeSearchTool
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.supersearch.prompts import PROMPT_VERSION
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
    SectionEnd,
)
from onyx.supersearch.acquisition import SupersearchAcquirer, corpus_specs
from onyx.supersearch.corpus import SupersearchCorpusBroker
from onyx.supersearch.dependencies import SupersearchDependencyExpander
from onyx.supersearch.engine import SupersearchEngine
from onyx.supersearch.gateway import SelectedModelGateway
from onyx.tools.interface import Tool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tracing.framework.create import ChatTraceMetadata, ensure_trace


def _progress_reporter(
    context: RunContext, emitter: Emitter, emitted: list[JsonValue]
) -> ProgressReporter:
    def emit_progress(event: ProgressEvent) -> None:
        packet = ASv3Progress.model_validate(
            {
                "workflow": "supersearch",
                "run_id": event.run_id,
                "event_id": event.event_id,
                "sequence": event.sequence,
                "language": event.language,
                "phase": event.phase,
                "status": event.status,
                "title": event.title,
                "message": event.message,
                "task_id": event.task_id,
                "parent_task_id": event.parent_task_id,
                "active_tasks": event.active_workers,
                "completed_tasks": event.completed_workers,
            }
        )
        emitted.append(packet.model_dump(mode="json"))
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=packet))

    return ProgressReporter(context.run_id, "tr", emit_progress)


def _publish_answer(
    *,
    result: WorkflowResult,
    ledger: EvidenceLedger,
    broker: SupersearchCorpusBroker,
    context: RunContext,
    emitter: Emitter,
    state_container: ChatStateContainer,
    assistant_message_id: int,
    include_citations: bool,
    elapsed: float,
    before_emit: Callable[[], None],
) -> None:
    context.check_active()
    assert result.answer is not None
    final = result.answer.strip()
    numbers = extract_citation_numbers(final)
    originals = [item for number in numbers if (item := ledger.get(number)) is not None]
    broker.revalidate_evidence(originals, context)
    mapping = {
        number: document
        for number, document in ledger.citation_mapping().items()
        if number in numbers
    }
    if set(numbers) - mapping.keys():
        raise ValueError("A Supersearch citation has no authorized canonical original")
    ledger.include(numbers)
    before_emit()
    state_container.add_search_docs(list(mapping.values()))
    state_container.set_pre_answer_processing_time(elapsed)
    processor = DynamicCitationProcessor(
        citation_mode=CitationMode.HYPERLINK
        if include_citations
        else CitationMode.REMOVE
    )
    processor.update_citation_mapping(mapping)
    emitter.emit(
        Packet(
            placement=Placement(turn_index=0),
            obj=AgentResponseStart(
                final_documents=list(mapping.values()),
                pre_answer_processing_seconds=elapsed,
            ),
        )
    )
    parts: list[str] = []
    for token in (final, None):
        for part in processor.process_token(token):
            context.check_active()
            if isinstance(part, CitationInfo):
                part.preview_url = (
                    f"/api/asv3/citation/{assistant_message_id}/{part.citation_number}"
                )
                state_container.add_emitted_citation(part.citation_number)
                emitter.emit(Packet(placement=Placement(turn_index=0), obj=part))
            else:
                parts.append(part)
                emitter.emit(
                    Packet(
                        placement=Placement(turn_index=0),
                        obj=AgentResponseDelta(content=part),
                    )
                )
    state_container.set_citation_mapping(processor.citation_to_doc)
    state_container.set_answer_tokens("".join(parts))
    emitter.emit(Packet(placement=Placement(turn_index=0), obj=SectionEnd()))


def run_supersearch_loop(
    *,
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    llm: LLM,
    user: User,
    chat_session_id: UUID,
    user_message_id: int,
    assistant_message_id: int,
    cache: CacheBackend,
    filters: BaseFilters | None = None,
    document_set_names_override: list[str] | None = None,
    user_identity: LLMUserIdentity | None = None,
    token_counter: Callable[[str], int] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    include_citations: bool = True,
    custom_agent_prompt: str | None = None,
    user_memory_context: UserMemoryContext | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    metadata = ChatTraceMetadata(
        chat_session_id=str(chat_session_id),
        user_id=str(user.id),
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        model_name=llm.config.model_name,
    ).model_dump()
    with ensure_trace("supersearch", group_id=str(chat_session_id), metadata=metadata):
        _run_supersearch_loop(
            emitter=emitter,
            state_container=state_container,
            simple_chat_history=simple_chat_history,
            tools=tools,
            llm=llm,
            user=user,
            chat_session_id=chat_session_id,
            assistant_message_id=assistant_message_id,
            cache=cache,
            filters=filters,
            document_set_names_override=document_set_names_override,
            user_identity=user_identity,
            token_counter=token_counter,
            reasoning_effort=reasoning_effort,
            include_citations=include_citations,
            custom_agent_prompt=custom_agent_prompt,
            user_memory_context=user_memory_context,
            inject_memories_in_prompt=inject_memories_in_prompt,
        )


def _run_supersearch_loop(
    *,
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    llm: LLM,
    user: User,
    chat_session_id: UUID,
    assistant_message_id: int,
    cache: CacheBackend,
    filters: BaseFilters | None,
    document_set_names_override: list[str] | None,
    user_identity: LLMUserIdentity | None,
    token_counter: Callable[[str], int] | None,
    reasoning_effort: ReasoningEffort,
    include_citations: bool,
    custom_agent_prompt: str | None,
    user_memory_context: UserMemoryContext | None,
    inject_memories_in_prompt: bool,
) -> None:
    started = time.monotonic()
    scope = IndexFilters(
        **(
            filters.model_dump(
                exclude={
                    "access_control_list",
                    "regulatory_workflow_mode",
                    "regulatory_label_search_enabled",
                    "regulatory_label_run_ids",
                }
            )
            if filters
            else {}
        ),
        access_control_list=[],
        regulatory_workflow_mode=filters.regulatory_workflow_mode
        if filters
        else "standard",
        regulatory_label_search_enabled=filters.regulatory_label_search_enabled
        if filters
        else False,
        regulatory_label_run_ids=filters.regulatory_label_run_ids if filters else (),
    )
    scope = bind_supersearch_pc_scope(
        user=user,
        filters=scope,
        document_set_names_override=document_set_names_override,
    )
    context = RunContext(
        language="tr",
        scope=scope.model_dump(mode="json"),
        timeout_seconds=math.inf,
        budget=SharedBudget(
            max_tools=sys.maxsize,
            max_decisions=sys.maxsize,
            max_evidence_bytes=32_000_000,
            max_inflight_tools=12,
            max_inflight_models=1,
            max_inflight_sources=4,
            final_decision_reserve=0,
            coordinator_decision_reserve=0,
            unlimited_execution=True,
        ),
        cancelled=lambda: not is_connected(chat_session_id, cache),
        corpus_only=True,
    )
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    broker = SupersearchCorpusBroker(user, scope, vision_llm=llm)
    context.services["shared_reads"] = SharedReads(
        fence=broker.shared_read_fence, producer_context=lambda caller: caller.child()
    )
    question = next(
        (
            message.message
            for message in reversed(simple_chat_history)
            if message.message_type == MessageType.USER
        ),
        "",
    )
    history = "\n".join(
        f"{message.message_type.value}: {message.message}"
        for message in simple_chat_history[-8:]
        if message.message_type in {MessageType.USER, MessageType.ASSISTANT}
    )
    emitted: list[JsonValue] = []
    progress = _progress_reporter(context, emitter, emitted)
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is not None:
        search = CompositeSearchTool.from_fork(search.fork_for_independent_context())
        search.auto_detect_filters = False
        search.enable_slack_search = False
        search.bypass_acl = False
        search.llm = llm
    broker.search_adapter = build_search_adapter(
        search,
        question,
        broker,
        message_history=lambda _context: list(simple_chat_history),
        user_memory_context=user_memory_context,
        inject_memories_in_prompt=inject_memories_in_prompt,
        user_identity=user_identity,
    )
    registry = CapabilityRegistry(corpus_specs(broker))
    # Only concurrency is bounded. Work ends on satisfied needs, a genuine source
    # frontier with no progress, transport failure, or user cancellation.
    policy = WorkflowPolicy(
        timeout_seconds=math.inf,
        max_cost_usd=math.inf,
        max_tools=sys.maxsize,
        max_search_calls=sys.maxsize,
        max_model_calls=sys.maxsize,
        max_parallel_tools=4,
    )
    acquirer = SupersearchAcquirer(registry, context, ledger, policy)
    dependencies = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    gateway = SelectedModelGateway(
        llm=llm,
        ledger=ledger,
        context=context,
        user_identity=user_identity,
        token_counter=token_counter,
        reasoning_effort=reasoning_effort,
    )

    def report(phase: str, language: str) -> None:
        context.language = progress.language = language
        state_container.set_stop_notice(
            localized_notifications(language)["cancelled"][1]
        )
        names = {
            "tools": (
                "Supersearch: PC külliyatında paralel arama",
                "İlgili özgün kaynakları ve bağlı hükümleri birlikte okuyorum.",
            ),
            "final": (
                "Supersearch: yanıt hazırlanıyor",
                "Okunan PC kaynaklarından koşulları ve dayanakları koruyarak yanıtı hazırlıyorum.",
            ),
            "review": (
                "Supersearch: yanıt doğrulanıyor",
                "Hukuki sonuçları, verilen olguları ve atıfları özgün kaynaklarla denetliyorum.",
            ),
        }
        title, message = names.get(
            phase, ("Supersearch", "PC Külliyatı kaynakları inceleniyor.")
        )
        progress.report(
            "verification" if phase == "review" else phase, title=title, message=message
        )

    def batch_progress(
        actions: list[SourceAction], pending: int, completed: int
    ) -> None:
        progress.report(
            "tools",
            # An unscoped terminal event closes the entire frontend run.
            status="running",
            title="Supersearch: özgün kaynaklar",
            message=f"{completed}/{len(actions)} kaynak işlemi tamamlandı.",
            active_workers=min(pending, 4),
            completed_workers=completed,
        )

    acquirer.on_batch_progress = batch_progress
    engine = SupersearchEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        check_active=context.check_active,
        dependency_expander=dependencies,
        report=report,
    )
    report("tools", "tr")
    result = engine.run(question, history, custom_agent_prompt)
    snapshot: dict[str, JsonValue] = {
        "run_id": context.run_id,
        "sequence": 1,
        "request": question,
        "scope": context.scope,
        "evidence": ledger.export(),
        "asv3_workflow_variant": "supersearch",
        "prompt_version": PROMPT_VERSION,
        "publication_status": result.status,
        "supersearch": result.model_dump(mode="json"),
        "authority_dependencies": [
            edge.model_dump(mode="json") for edge in engine.dependencies
        ],
        "source_receipts": engine.receipts,
        "acquisition_counts": {
            "searches": acquirer.search_calls,
            **context.budget.snapshot(),
        },
        "processing_seconds": time.monotonic() - started,
        "progress": emitted,
    }
    if result.answer is None or result.status == "cancelled":
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )
        progress.report(
            "cancelled" if result.status == "cancelled" else "failed", status="failed"
        )
        raise OnyxError(
            OnyxErrorCode.LLM_PROVIDER_ERROR,
            "Supersearch özgün kaynaklarla doğrulanmış bir yanıt üretemedi. Kaynak kapsamı veya model yanıtı tamamlanamadı.",
        )

    def persist_before_emit() -> None:
        snapshot["evidence"] = ledger.export()
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )

    _publish_answer(
        result=result,
        ledger=ledger,
        broker=broker,
        context=context,
        emitter=emitter,
        state_container=state_container,
        assistant_message_id=assistant_message_id,
        include_citations=include_citations,
        elapsed=time.monotonic() - started,
        before_emit=persist_before_emit,
    )
    progress.report(
        "completed",
        status="completed",
        title="Supersearch tamamlandı",
        message="PC Külliyatı kaynaklarına dayalı yanıt hazır.",
    )
    snapshot.update(
        sequence=2,
        evidence=ledger.export(),
        progress=emitted,
        processing_seconds=time.monotonic() - started,
    )
    save_asv3_checkpoint(
        message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
    )
