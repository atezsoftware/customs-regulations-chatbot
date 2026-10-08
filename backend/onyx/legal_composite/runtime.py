"""Independent chat entry point; existing workflows never enter this module."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
)
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
from onyx.db.legal_composite_sources import (
    SourceKind,
    SourceLaneCatalogue,
    load_source_lane_catalogue,
)
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import SourceAction, WorkflowPolicy
from onyx.legal_composite.prompts import PROMPT_VERSION
from onyx.legal_composite.providers import build_source_selector
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.legal_composite.search import CompositeSearchTool
from onyx.legal_composite.source_lanes import build_lane_broker
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
from onyx.tracing.answer_graph import graph_step
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
        _run_legal_composite_loop(
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
            research_llm=research_llm,
            token_counter=token_counter,
            reasoning_effort=reasoning_effort,
            include_citations=include_citations,
            custom_agent_prompt=custom_agent_prompt,
            user_memory_context=user_memory_context,
            inject_memories_in_prompt=inject_memories_in_prompt,
        )


def _progress_reporter(
    context: RunContext, emitter: Emitter, emitted: list[JsonValue]
) -> ProgressReporter:
    def emit_progress(event: ProgressEvent) -> None:
        words = localized_notifications(event.language)
        title, message = (
            (event.title, event.message)
            if event.public_narration
            else words.get(event.phase, words["tools"])
        )
        packet = ASv3Progress.model_validate(
            dict(
                run_id=event.run_id,
                event_id=event.event_id,
                sequence=event.sequence,
                language=event.language,
                phase=event.phase,
                status=event.status,
                title=title,
                message=message,
                task_id=event.task_id,
                parent_task_id=event.parent_task_id,
                active_tasks=event.active_workers,
                completed_tasks=event.completed_workers,
            )
        )
        emitted.append(packet.model_dump(mode="json"))
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=packet))

    return ProgressReporter(context.run_id, "tr", emit_progress)


def _load_source_catalogue(
    *, user: User, scope: IndexFilters, context: RunContext, progress: ProgressReporter
) -> SourceLaneCatalogue:
    started = time.monotonic()
    processed_sources = 0
    progress_updates = 0
    status = "running"
    progress.report(
        "tools",
        title="Kaynak türleri doğrulanıyor",
        message="Arama kollarını özgün kaynak açılışlarından hazırlıyorum.",
    )
    with graph_step("legal_composite.source_catalogue", {"opening_workers": 4}) as step:

        def on_progress(processed: int, has_more: bool) -> None:
            nonlocal processed_sources, progress_updates
            context.check_research_active()
            processed_sources = processed
            progress_updates += 1
            progress.report(
                "tools",
                title="Kaynak türleri doğrulanıyor",
                message=(
                    f"{processed} erişilebilir kaynağın özgün açılışı incelendi; "
                    f"{time.monotonic() - started:.1f} saniye geçti."
                    + (" Diğer kaynaklar inceleniyor." if has_more else "")
                ),
            )

        try:
            context.check_research_active()
            with get_session_with_current_tenant() as inventory_session:
                catalogue = load_source_lane_catalogue(
                    inventory_session,
                    user=user,
                    filters=scope,
                    check_active=context.check_research_active,
                    opening_workers=4,
                    on_progress=on_progress,
                )
            context.check_research_active()
            status = "completed" if catalogue.complete else "partial"
            processed_sources = len(catalogue.records)
            step.output_value = {
                "source_count": processed_sources,
                "uncertain_source_count": sum(
                    row.uncertain for row in catalogue.records
                ),
                "inventory_complete": catalogue.complete,
            }
        except RunStopped:
            status = "cancelled" if context.is_cancelled() else "failed"
            progress.report(status, status="failed")
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            step.output_value = {
                **(step.output_value or {}),
                "status": status,
                "processed_source_count": processed_sources,
                "progress_updates": progress_updates,
                "elapsed_seconds": time.monotonic() - started,
            }
    progress.report(
        "tools",
        status="completed",
        title="Kaynak türleri hazır"
        if catalogue.complete
        else "Kaynak türleri kısmen doğrulandı",
        message=(
            f"{processed_sources} erişilebilir kaynak incelendi; "
            "doğrulanmayan kaynak türleri belirsiz tutuluyor."
        ),
    )
    return catalogue


def _run_legal_composite_loop(
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
    research_llm: LLM | None,
    token_counter: Callable[[str], int] | None,
    reasoning_effort: ReasoningEffort,
    include_citations: bool,
    custom_agent_prompt: str | None,
    user_memory_context: UserMemoryContext | None,
    inject_memories_in_prompt: bool,
) -> None:
    policy = WorkflowPolicy(
        timeout_seconds=math.inf,
        max_cost_usd=math.inf,
        max_context_tokens=128_000,
        final_output_tokens=8_192,
        max_input_tokens=2_000_000,
        max_output_tokens=256_000,
        max_model_calls=32,
        max_tools=512,
        max_parallel_tools=len(SourceKind),
        max_search_calls=384,
        selection_reserve_seconds=12,
    )
    started = time.monotonic()

    def cancelled() -> bool:
        return not is_connected(chat_session_id, cache)

    context = RunContext(
        language="und",
        timeout_seconds=policy.timeout_seconds,
        research_reserve_seconds=(
            policy.finalization_reserve_seconds + policy.selection_reserve_seconds
        ),
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
    emitted: list[JsonValue] = []
    progress = _progress_reporter(context, emitter, emitted)
    catalogue = _load_source_catalogue(
        user=user, scope=scope, context=context, progress=progress
    )
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is not None:
        search = CompositeSearchTool.from_fork(search.fork_for_independent_context())
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

    def build_lane_registry(kind: SourceKind) -> CapabilityRegistry:
        if not catalogue.source_ids(kind):

            def empty_lane(
                _arguments: dict[str, JsonValue], child: RunContext
            ) -> ToolOutcome:
                child.check_research_active()
                return ToolOutcome(
                    status=OutcomeStatus.NOT_FOUND
                    if catalogue.complete
                    else OutcomeStatus.UNAVAILABLE,
                    summary="No authorized source is classified in this lane; this is not proof that no applicable law exists.",
                    data={**catalogue.provenance(), "source_kind": kind.value},
                )

            return CapabilityRegistry(
                [
                    spec.model_copy(update={"handler": empty_lane})
                    for spec in build_corpus_specs(
                        broker,
                        require_search_targets=True,
                        source_identity_guidance=True,
                    )
                ]
            )
        lane = build_lane_broker(broker, catalogue, kind)
        lane.search_adapter = lane.guard_search_adapter(
            build_search_adapter(
                search,
                question,
                lane,
                message_history=lambda _context: list(simple_chat_history),
                user_memory_context=user_memory_context,
                inject_memories_in_prompt=inject_memories_in_prompt,
                user_identity=user_identity,
            )
        )
        scoped_registry = CapabilityRegistry(
            build_corpus_specs(
                lane, require_search_targets=True, source_identity_guidance=True
            )
        )
        for spec in build_core_specs(
            scoped_registry, ledger, state_provider=lambda: {}
        ):
            if spec.name == "read_evidence":
                scoped_registry.register(spec)
        return scoped_registry

    router = SourceLaneRouter(catalogue, build_lane_registry)
    acquirer = CanonicalAcquirer(
        registry,
        context,
        ledger,
        policy,
        registry_for_action=router.registry,
        expand_actions=router.expand,
        lane_inventory=router.inventory(),
    )
    budget = WorkflowBudget(policy, deadline=context.deadline)
    budget.retain_selection_time(policy.selection_reserve_seconds)
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
            selector = build_source_selector(
                session=price_session,
                user=user,
                gateway=gateway,
                budget=budget,
                ledger=ledger,
                check_active=context.check_active,
                token_counter=token_counter,
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

    def report_batch(actions: list[SourceAction], pending: int, completed: int) -> None:
        names = {
            SourceKind.CONSTITUTION: "Anayasa",
            SourceKind.STATUTE: "Kanun",
            SourceKind.TREATY: "Uluslararası antlaşma",
            SourceKind.PRESIDENTIAL_DECREE: "Cumhurbaşkanlığı kararnamesi",
            SourceKind.REGULATION: "Yönetmelik",
            SourceKind.COMMUNIQUE: "Tebliğ",
            SourceKind.CIRCULAR: "Genelge",
            SourceKind.JUDICIAL_DECISION: "Yargı kararı",
            SourceKind.EXECUTIVE_DECISION: "İdari karar",
            SourceKind.PRIVATE_RULING: "Özelge",
            SourceKind.OTHER: "Diğer kaynak",
            SourceKind.UNKNOWN: "Türü doğrulanmamış kaynak",
        }
        kinds = list(dict.fromkeys(action.source_kind for action in actions))
        turkish = context.language.startswith("tr")
        labels = [
            names[kind] if turkish else kind.value.replace("_", " ")
            for kind in kinds
            if kind is not None
        ]
        parallel = len(actions) > 1
        title = (
            ("Paralel kaynak araştırması" if parallel else "Kaynak araştırması")
            if turkish
            else ("Parallel source research" if parallel else "Source research")
        )
        message = (
            (
                f"{', '.join(labels)}: {completed}/{len(actions)} işlem tamamlandı; "
                f"en fazla {policy.max_parallel_tools} arama aynı anda yürütülüyor."
            )
            if turkish
            else (
                f"{', '.join(labels)}: {completed}/{len(actions)} actions complete; "
                f"up to {policy.max_parallel_tools} searches run concurrently."
            )
        )
        progress.report(
            "tools",
            status="running" if pending else "completed",
            active_workers=min(pending, policy.max_parallel_tools),
            completed_workers=completed,
            title=title,
            message=message,
        )

    acquirer.on_batch_progress = report_batch

    def report(phase: str, language: str) -> None:
        context.language = progress.language = language
        state_container.set_stop_notice(
            localized_notifications(language)["cancelled"][1]
        )
        if phase == "selection":
            progress.report(
                "tools",
                title="Kaynak uygunluğu denetleniyor"
                if language.startswith("tr")
                else "Checking source relevance",
                message="Bulunan özgün kaynakları sorularınıza göre birlikte değerlendiriyorum."
                if language.startswith("tr")
                else "Evaluating the original sources together against your questions.",
            )
        else:
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
        selector=selector,
    )
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
        "source_lane_inventory": router.inventory(),
        "source_selection": engine.selection.model_dump(mode="json")
        if engine.selection is not None
        else None,
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
    progress.report("completed", status="completed")
    snapshot.update(
        sequence=2, progress=emitted, processing_seconds=time.monotonic() - started
    )
    save_asv3_checkpoint(
        message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
    )
