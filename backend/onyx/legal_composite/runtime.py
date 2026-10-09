"""Independent chat entry point; existing workflows never enter this module."""

from __future__ import annotations

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
from onyx.db.legal_composite_catalogue import (
    CatalogueTimings,
    load_prepared_source_lane_catalogue,
)
from onyx.db.legal_composite_sources import (
    SourceKind,
)
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.cancellation import PollingCancellation
from onyx.legal_composite.dependencies import CompositeDependencyExpander
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import SourceAction, WorkflowPolicy
from onyx.legal_composite.prompts import PROMPT_VERSION
from onyx.legal_composite.providers import build_answer_reviewer, build_source_selector
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.legal_composite.search import CompositeSearchTool
from onyx.legal_composite.settled_budget import SettledUsageBudget
from onyx.legal_composite.shared_work import SharedCanonicalCenters
from onyx.legal_composite.source_lanes import build_lane_broker
from onyx.llm.factory import get_llm
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
        timeout_seconds=420,
        finalization_reserve_seconds=140,
        max_cost_usd=1.50,
        max_call_seconds=75,
        max_context_tokens=192_000,
        final_output_tokens=16_384,
        max_input_tokens=2_000_000,
        max_output_tokens=128_000,
        max_model_calls=48,
        max_tools=192,
        max_parallel_tools=len(SourceKind),
        max_search_calls=96,
        max_research_rounds=3,
        selection_reserve_seconds=12,
    )
    started = time.monotonic()

    def cancelled() -> bool:
        return not is_connected(chat_session_id, cache)

    cancellation_probe = PollingCancellation(cancelled)
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
        cancelled=cancellation_probe,
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
    progress.report(
        "tools",
        title="Paralel kaynak araştırması hazırlanıyor",
        message="Kayıtlı kaynak türleri yükleniyor; on iki türün aramaları paralel başlayacak.",
    )
    context.check_research_active()
    catalogue_timings = CatalogueTimings()
    with graph_step("legal_composite.prepared_catalogue", {}) as catalogue_step:
        with get_session_with_current_tenant() as session:
            catalogue = load_prepared_source_lane_catalogue(
                session,
                user=user,
                filters=scope,
                check_active=context.check_research_active,
                timings=catalogue_timings,
            )
        catalogue_step.summary = catalogue_timings.safe_summary()
        catalogue_step.output_value = {
            **catalogue.provenance(),
            "runtime_opening_reads": 0,
            "runtime_classifications": 0,
            "catalogue_timings": catalogue_timings.snapshot(),
        }
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is not None:
        search = CompositeSearchTool.from_fork(search.fork_for_independent_context())
        search.enable_shared_prepared_work()
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

    shared_centers = SharedCanonicalCenters()

    def build_lane_registry(kind: SourceKind) -> CapabilityRegistry:
        if not catalogue.source_ids(kind):

            def empty_lane(
                _arguments: dict[str, JsonValue], child: RunContext
            ) -> ToolOutcome:
                child.check_active()
                return ToolOutcome(
                    status=OutcomeStatus.NOT_FOUND,
                    summary="No accessible prepared sources in this source-kind lane.",
                    data={
                        "source_kind": kind.value,
                        "lane_source_count": 0,
                        "source_type_preparation_complete": catalogue.complete,
                        "corpus_absence_verified": False,
                    },
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
        lane = build_lane_broker(broker, catalogue, kind, shared_centers)
        lane_search = (
            search.fork_for_independent_context() if search is not None else None
        )
        if lane_search is not None:
            lane_search.configure_prepared_source_lane(
                kind,
                lane.record_search,
                context.check_research_active,
            )
        lane.search_adapter = lane.guard_search_adapter(
            build_search_adapter(
                lane_search,
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
        coalesce_progress=True,
        prioritize_observed_reads=True,
        registry_for_action=router.registry,
        expand_actions=router.expand,
        lane_inventory=router.inventory(),
        source_kinds={str(row.source_id): row.kind for row in catalogue.records},
        host_registry=CapabilityRegistry(
            build_corpus_specs(
                broker, require_search_targets=True, source_identity_guidance=True
            )
        ),
    )
    dependency_expander = CompositeDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={str(row.source_id): row.kind for row in catalogue.records},
    )
    budget = SettledUsageBudget(policy, deadline=context.deadline)
    budget.retain_selection_time(policy.selection_reserve_seconds)
    try:
        with get_session_with_current_tenant() as price_session:
            reviewer = build_answer_reviewer(
                session=price_session,
                user=user,
                budget=budget,
                ledger=ledger,
                check_active=context.check_active,
                token_counter=token_counter,
                run_id=context.run_id,
                scope=context.scope,
            )
            if reviewer is None:
                raise RunStopped("No authorized answer reviewer is configured")
            config = reviewer.config
            typed_research_llm = get_llm(
                provider=config.model_provider,
                model=config.model_name,
                max_input_tokens=config.max_input_tokens,
                deployment_name=None,
                api_key=config.api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=1,
            )
            gateway = BudgetedGateway(
                max_parallel_generations=4,
                share_draft_context=True,
                preserve_research_finalization_on_timeout=True,
                selected_llm=llm,
                research_llm=typed_research_llm,
                budget=budget,
                ledger=ledger,
                db_session=price_session,
                user_identity=user_identity,
                check_active=context.check_active,
                token_counter=token_counter,
                reasoning_effort=ReasoningEffort.LOW
                if reasoning_effort is ReasoningEffort.AUTO
                else reasoning_effort,
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
            "Legal Composite requires an authorized answer reviewer and a priced model that fits the generation budget.",
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
        dependency_batch = (
            any(action.tool == "dependency_related_sources" for action in actions)
            or not labels
        )
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
        if dependency_batch:
            title = (
                "Kaynak atıfları ve bağlı hükümler araştırılıyor"
                if turkish
                else "Researching source references and dependent authority"
            )
            message = (
                f"Özgün kaynak ilişkileri: {completed}/{len(actions)} işlem tamamlandı."
                if turkish
                else f"Original source relationships: {completed}/{len(actions)} actions complete."
            )
        elapsed = int(time.monotonic() - started)
        message += (
            f" Toplam geçen süre: {elapsed} saniye."
            if turkish
            else f" Total elapsed: {elapsed} seconds."
        )
        progress.report(
            "tools",
            status="running",
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
        if phase in {"review", "repair"}:
            turkish = language.startswith("tr")
            progress.report(
                phase,
                title=(
                    "Cevap kaynaklarla denetleniyor"
                    if phase == "review"
                    else "Eksik bölüm düzeltiliyor"
                )
                if turkish
                else (
                    "Checking the answer against sources"
                    if phase == "review"
                    else "Repairing the affected section"
                ),
                message=(
                    "Sorularınızın kapsamını, koşulları, istisnaları ve kaynakların cevabı desteklemesini denetliyorum."
                    if phase == "review"
                    else "Denetimde belirlenen eksikleri giderip değişen bölümleri yeniden kontrol ediyorum."
                )
                if turkish
                else (
                    "Checking issue coverage, conditions, exceptions and original source support."
                    if phase == "review"
                    else "Repairing identified gaps and rechecking the changed sections."
                ),
            )
        elif phase == "reading":
            progress.report(
                "reading",
                title="Kaynak hükümleri okunuyor"
                if language.startswith("tr")
                else "Reading source provisions",
                message="Bulunan özgün hükümleri her sorunuz için değerlendirip koşulları, süreleri ve gerekli diğer kaynakları belirliyorum."
                if language.startswith("tr")
                else "Evaluating the original provisions for each question to identify conditions, deadlines and other required sources.",
            )
        elif phase == "selection":
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
        dependency_expander=dependency_expander,
        source_kinds={str(row.source_id): row.kind for row in catalogue.records},
        reviewer=reviewer,
        evidence_context=context,
        use_numbered_reading_supports=True,
        allow_terminal_observed_reads=True,
    )
    result = engine.run(question, history, custom_agent_prompt)
    snapshot: dict[str, JsonValue] = {
        "run_id": context.run_id,
        "sequence": 1,
        "request": question,
        "scope": context.scope,
        "evidence": ledger.export(),
        "authority_dependencies": [
            edge.model_dump(mode="json") for edge in getattr(engine, "dependencies", [])
        ],
        "asv3_workflow_variant": "legal_composite",
        "prompt_version": PROMPT_VERSION,
        "publication_status": result.status,
        "legal_composite": result.model_dump(mode="json"),
        "legal_composite_budget": budget.snapshot(),
        "source_requirements": engine.requirements.export(),
        "protocol_defects": engine.protocol_defects,
        "answer_reviewer": getattr(reviewer, "mode", "configured_decisions"),
        "cost_scope": "Generation and Decisions calls only; embedding/reranker costs require separate reconciliation",
        "source_lane_inventory": router.inventory(),
        "source_selection": engine.selection.model_dump(mode="json")
        if engine.selection is not None
        else None,
        "acquisition_counts": {
            "searches": acquirer.search_calls,
            **context.budget.snapshot(),
        },
        "cancellation_observations": cancellation_probe.snapshot(),
        "processing_seconds": time.monotonic() - started,
        "progress": emitted,
    }
    if result.answer is None or result.status == "cancelled":
        words = localized_notifications(context.language)
        progress.report(
            "cancelled" if result.status == "cancelled" else "failed",
            status="failed",
        )
        snapshot["progress"] = emitted
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
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
    snapshot["cancellation_observations"] = cancellation_probe.snapshot()
    save_asv3_checkpoint(
        message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
    )
