"""Independent chat entry point; existing workflows never enter this module."""

from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.asv3.progress import ProgressEvent, ProgressReporter, localized_notifications
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.search_adapter import build_search_adapter
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.chat.stop_signal_checker import is_connected
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.asv3_corpus import bind_pc_corpus_scope
from onyx.db.asv3_runs import save_asv3_checkpoint
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import WorkflowPolicy
from onyx.legal_composite.prompts import PROMPT_VERSION
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
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


def run_legal_composite_loop(
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
    research_llm: LLM | None = None,
    token_counter: Callable[[str], int] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    include_citations: bool = True,
    custom_agent_prompt: str | None = None,
    user_memory_context: UserMemoryContext | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    policy = WorkflowPolicy()
    started = time.monotonic()

    def cancelled() -> bool:
        return not is_connected(chat_session_id, cache)

    context = RunContext(
        language="und",
        timeout_seconds=policy.timeout_seconds,
        research_reserve_seconds=policy.finalization_reserve_seconds,
        budget=SharedBudget(
            max_tools=policy.max_tools,
            max_decisions=policy.max_model_calls,
            max_evidence_bytes=500_000,
            max_inflight_tools=policy.max_parallel_tools,
            max_inflight_models=1,
            final_decision_reserve=2,
        ),
        cancelled=cancelled,
        corpus_only=True,
    )
    question = next(
        (
            message.message
            for message in reversed(simple_chat_history)
            if message.message_type == MessageType.USER
        ),
        "",
    )
    scope = IndexFilters(
        **(filters.model_dump(exclude={"access_control_list"}) if filters else {}),
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
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    broker = CorpusBroker(user, scope)
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is not None:
        search = search.fork_for_independent_context()
        search.auto_detect_filters = False
        search.enable_slack_search = False
        search.bypass_acl = False
        search.llm = research_llm or llm
    adapter = build_search_adapter(
        search,
        question,
        broker,
        message_history=lambda _context: list(simple_chat_history),
        user_memory_context=user_memory_context,
        inject_memories_in_prompt=inject_memories_in_prompt,
        user_identity=user_identity,
    )
    broker.search_adapter = adapter
    registry = CapabilityRegistry(
        build_corpus_specs(
            broker, require_search_targets=True, source_identity_guidance=True
        )
    )
    for spec in build_core_specs(registry, ledger, state_provider=lambda: {}):
        if spec.name == "read_evidence":
            registry.register(spec)
    acquirer = CanonicalAcquirer(registry, context, ledger, policy)
    budget = WorkflowBudget(policy, deadline=context.deadline)
    try:
        with get_session_with_current_tenant() as price_session:
            gateway = BudgetedGateway(
                selected_llm=llm,
                research_llm=research_llm or llm,
                budget=budget,
                ledger=ledger,
                db_session=price_session,
                user_identity=user_identity,
                check_active=context.check_active,
                token_counter=token_counter,
                reasoning_effort=reasoning_effort,
                run_id=context.run_id,
                scope=context.scope,
            )
    except RunStopped as error:
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "The selected model cannot fit the Legal Composite generation budget; select a priced, lower-cost model.",
        ) from error
    if search is not None:
        search.llm = gateway.research_proxy()
    history = "\n".join(
        f"{message.message_type.value}: {message.message}"
        for message in simple_chat_history[-8:]
        if message.message_type in {MessageType.USER, MessageType.ASSISTANT}
    )

    emitted: list[JsonValue] = []

    def emit_progress(event: ProgressEvent) -> None:
        words = localized_notifications(context.language)
        title, message = words.get(event.phase, words["tools"])
        packet = ASv3Progress.model_validate(
            dict(
                run_id=event.run_id,
                event_id=event.event_id,
                sequence=event.sequence,
                language=context.language,
                phase=event.phase,
                status=event.status,
                title=title,
                message=message,
                task_id=None,
                parent_task_id=None,
                active_tasks=0,
                completed_tasks=0,
            )
        )
        emitted.append(packet.model_dump(mode="json"))
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=packet))

    progress = ProgressReporter(context.run_id, "und", emit_progress)

    def report(phase: str, language: str) -> None:
        context.language = progress.language = language
        state_container.set_stop_notice(
            localized_notifications(language)["cancelled"][1]
        )
        progress.report(phase)

    def research_available() -> bool:
        context.check_active()
        return (
            time.monotonic() < context.research_deadline and budget.research_available()
        )

    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=policy,
        check_active=context.check_active,
        research_available=research_available,
        report=report,
    )
    metadata = ChatTraceMetadata(
        chat_session_id=str(chat_session_id),
        user_id=str(user.id),
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        model_name=llm.config.model_name,
    ).model_dump()
    with ensure_trace(
        "legal_composite", group_id=str(chat_session_id), metadata=metadata
    ):
        result = engine.run(question, history, custom_agent_prompt)
        snapshot: dict[str, JsonValue] = {
            "run_id": context.run_id,
            "sequence": 1,
            "request": question,
            "scope": context.scope,
            "evidence": ledger.export(),
            "asv3_workflow_variant": "legal_composite",
            "prompt_version": PROMPT_VERSION,
            "publication_status": result.status,
            "legal_composite": result.model_dump(mode="json"),
            "legal_composite_budget": budget.snapshot(),
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
            words = localized_notifications(context.language)
            progress.report(
                "cancelled" if result.status == "cancelled" else "failed",
                status="failed",
            )
            raise OnyxError(OnyxErrorCode.LLM_PROVIDER_ERROR, words["failed"][1])
        context.check_active()
        final = result.answer.strip()
        if result.status == "partial":
            prefix = (
                "Kısmi yanıt — bazı belirleyici noktalar doğrulanamadı.\n\n"
                if context.language.startswith("tr")
                else "Partial answer — some decisive points remain unverified.\n\n"
            )
            final = prefix + final
        numbers = extract_citation_numbers(final)
        items = [item for number in numbers if (item := ledger.get(number)) is not None]
        broker.revalidate_evidence(items, context)
        allowed = {
            number: document
            for number, document in ledger.citation_mapping().items()
            if number in numbers
        }
        if set(numbers) - allowed.keys():
            raise ValueError("A final citation has no authorized original target")
        ledger.include(numbers)
        snapshot["evidence"] = ledger.export()
        snapshot["processing_seconds"] = time.monotonic() - started
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )
        state_container.add_search_docs(list(allowed.values()))
        elapsed = time.monotonic() - started
        state_container.set_pre_answer_processing_time(elapsed)
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
                    pre_answer_processing_seconds=elapsed,
                ),
            )
        )
        parts: list[str] = []
        for token in (final, None):
            for part in processor.process_token(token):
                context.check_active()
                if isinstance(part, CitationInfo):
                    part.preview_url = f"/api/asv3/citation/{assistant_message_id}/{part.citation_number}"
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
        progress.report("completed", status="completed")
        snapshot.update(
            sequence=2, progress=emitted, processing_seconds=time.monotonic() - started
        )
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )
