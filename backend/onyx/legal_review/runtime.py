"""Independent chat adapter using existing ownership, citations and stop fences."""

from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, SharedBudget
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
from onyx.db.legal_review_providers import resolve_legal_review_decision
from onyx.db.memory import UserMemoryContext
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_review.acquisition import SourceAcquirer
from onyx.legal_review.decisions import DecisionsReviewer
from onyx.legal_review.engine import LegalReviewEngine
from onyx.legal_review.gateway import GeminiGateway, MeteredLLM, UsageMeter
from onyx.legal_review.models import DecisionProviderConfig, WorkflowPolicy
from onyx.legal_review.provider import require_legal_review_model
from onyx.legal_review.search import DiscoverySearchTool
from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import ReasoningEffort
from onyx.prompts.legal_review.prompts import PROMPT_VERSION
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


def run_legal_review_loop(
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
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    include_citations: bool = True,
    custom_agent_prompt: str | None = None,
    user_memory_context: UserMemoryContext | None = None,
    inject_memories_in_prompt: bool = True,
) -> None:
    del research_llm, custom_agent_prompt
    require_legal_review_model(llm)
    review_provider = resolve_legal_review_decision(user)
    if review_provider is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Legal Review requires an accessible configured OpenAI gpt-6-luna provider for the Decisions API.",
        )
    metadata = ChatTraceMetadata(
        chat_session_id=str(chat_session_id),
        user_id=str(user.id),
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        model_name=llm.config.model_name,
    ).model_dump()
    with ensure_trace("legal_review", group_id=str(chat_session_id), metadata=metadata):
        _run(
            emitter=emitter,
            state_container=state_container,
            simple_chat_history=simple_chat_history,
            tools=tools,
            llm=llm,
            review_provider=review_provider,
            user=user,
            chat_session_id=chat_session_id,
            assistant_message_id=assistant_message_id,
            cache=cache,
            filters=filters,
            document_set_names_override=document_set_names_override,
            user_identity=user_identity,
            token_counter=token_counter or get_llm_token_counter(llm),
            reasoning_effort=reasoning_effort,
            include_citations=include_citations,
            user_memory_context=user_memory_context,
            inject_memories_in_prompt=inject_memories_in_prompt,
        )


def _run(
    *,
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    llm: LLM,
    review_provider: DecisionProviderConfig,
    user: User,
    chat_session_id: UUID,
    assistant_message_id: int,
    cache: CacheBackend,
    filters: BaseFilters | None,
    document_set_names_override: list[str] | None,
    user_identity: LLMUserIdentity | None,
    token_counter: Callable[[str], int],
    reasoning_effort: ReasoningEffort,
    include_citations: bool,
    user_memory_context: UserMemoryContext | None,
    inject_memories_in_prompt: bool,
) -> None:
    started = time.monotonic()
    policy = WorkflowPolicy()
    context = RunContext(
        timeout_seconds=policy.timeout_seconds,
        research_reserve_seconds=policy.finalization_reserve_seconds,
        cancelled=lambda: not is_connected(chat_session_id, cache),
        budget=SharedBudget(
            max_tools=policy.max_tools,
            max_decisions=policy.max_model_calls,
            max_evidence_bytes=policy.max_evidence_bytes,
            max_inflight_tools=policy.max_parallel_tools,
            max_inflight_models=1,
            final_decision_reserve=8,
            coordinator_decision_reserve=0,
        ),
        services={"provider_max_attempts": 1, "provider_compatibility_attempts": 1},
        corpus_only=True,
    )
    scope = IndexFilters(
        **(filters.model_dump(exclude={"access_control_list"}) if filters else {}),
        access_control_list=[],
    )
    if document_set_names_override:
        scope.forced_document_set = document_set_names_override
    scope = bind_pc_corpus_scope(user=user, filters=scope)
    context.scope = scope.model_dump(mode="json")
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    broker = CorpusBroker(user, scope)
    meter = UsageMeter(policy)
    selected = MeteredLLM(llm, context, meter)
    question = next(
        (
            message.message
            for message in reversed(simple_chat_history)
            if message.message_type == MessageType.USER
        ),
        "",
    )
    search = next((tool for tool in tools if isinstance(tool, SearchTool)), None)
    if search is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "Legal Review requires indexed search."
        )
    search = DiscoverySearchTool.from_fork(search.fork_for_independent_context())
    search.llm = selected
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
            broker,
            named_provision_reads=False,
            require_search_targets=True,
            source_identity_guidance=True,
        )
    )
    for spec in build_core_specs(registry, ledger, state_provider=lambda: {}):
        if spec.name == "read_evidence":
            registry.register(spec)
    acquirer = SourceAcquirer(
        registry=registry,
        context=context,
        ledger=ledger,
        policy=policy,
        search_adapter=adapter,
    )
    progress: list[JsonValue] = []
    state_container.set_stop_notice("Hukuki inceleme kullanıcı tarafından durduruldu.")

    def report(phase: str, language: str) -> None:
        titles = {
            "planning": "Sorular ve inceleme boyutları belirleniyor",
            "tools": "Özgün kaynaklar araştırılıyor",
            "reading": "Kaynak koşulları ve kanıt boşlukları değerlendiriliyor",
            "review": "Kaynak ve cevap kontrolü yapılıyor",
            "final": "Bütünleşik cevap hazırlanıyor",
            "repair": "Belirlenen eksikler bir kez düzeltiliyor",
            "completed": "Hukuki inceleme tamamlandı",
            "failed": "Hukuki inceleme tamamlanamadı",
            "cancelled": "Hukuki inceleme durduruldu",
        }
        event = ASv3Progress(
            workflow="legal_review",
            run_id=context.run_id,
            event_id=str(uuid4()),
            sequence=len(progress) + 1,
            language=language,
            phase=phase,
            status="completed"
            if phase == "completed"
            else "cancelled"
            if phase == "cancelled"
            else "failed"
            if phase == "failed"
            else "running",
            title=titles.get(phase, "Hukuki inceleme sürüyor"),
        )
        progress.append(event.model_dump(mode="json"))
        emitter.emit(Packet(placement=Placement(turn_index=0), obj=event))

    def admit_review() -> None:
        context.check_active()
        meter.check()
        context.budget.consume("decisions")

    engine = LegalReviewEngine(
        gateway=GeminiGateway(
            llm=selected,
            context=context,
            policy=policy,
            token_counter=token_counter,
            reasoning_effort=reasoning_effort,
        ),
        acquirer=acquirer,
        reviewer=DecisionsReviewer(
            api_key=review_provider.api_key.get_secret_value(),
            before_request=admit_review,
            check_active=context.check_active,
        ),
        ledger=ledger,
        context=context,
        policy=policy,
        report=report,
        record_review_usage=meter.record,
    )
    history = "\n".join(
        f"{message.message_type.value}: {message.message}"
        for message in simple_chat_history
        if message.message_type in {MessageType.USER, MessageType.ASSISTANT}
    )
    result = engine.run(question, history)
    snapshot: dict[str, JsonValue] = {
        "run_id": context.run_id,
        "sequence": 1,
        "request": question,
        "scope": context.scope,
        "asv3_workflow_variant": "legal_review",
        "review_provider": review_provider.model_dump(mode="json"),
        "prompt_version": PROMPT_VERSION,
        "publication_status": result.status,
        "legal_review": result.model_dump(mode="json"),
        "evidence": ledger.export(),
        "progress": progress,
        "source_operations": acquirer.receipts,
        "requirement_history": [
            row.model_dump(mode="json") for row in engine.requirement_history.values()
        ],
        "policy": policy.model_dump(mode="json"),
        "budget": {**context.budget.snapshot(), **meter.snapshot()},
        "retrieval_width": {
            "per_lane_hits": 256,
            "rerank_candidates": 384,
            "model_chunks": 50,
        },
        "processing_seconds": time.monotonic() - started,
        "usage_scope": "Gemini Flash and OpenAI Decisions token usage; embedding/reranker calls retain their separate provider traces",
    }
    if result.answer is None:
        report(
            "cancelled" if result.status == "cancelled" else "failed", context.language
        )
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )
        raise OnyxError(
            OnyxErrorCode.LLM_PROVIDER_ERROR,
            "Legal Review could not verify a publishable answer. "
            + (result.gaps[-1] if result.gaps else "Review incomplete."),
        )
    final = result.answer
    if result.status == "partial":
        prefix = (
            "Kısmi yanıt — bazı belirleyici noktalar doğrulanamadı."
            if context.language.startswith("tr")
            else "Partial answer — some decisive points could not be verified."
        )
        final = prefix + "\n\n" + final
    try:
        context.check_active()
        numbers = extract_citation_numbers(final)
        items = [item for number in numbers if (item := ledger.get(number)) is not None]
        broker.revalidate_evidence(items, context)
        mapping = {
            number: doc
            for number, doc in ledger.citation_mapping().items()
            if number in numbers
        }
        if set(numbers) - mapping.keys():
            raise OnyxError(
                OnyxErrorCode.VALIDATION_ERROR,
                "An answer citation has no original target.",
            )
    except Exception as error:
        failed_status = "cancelled" if context.is_cancelled() else "unavailable"
        report(
            "cancelled" if failed_status == "cancelled" else "failed", context.language
        )
        snapshot["publication_status"] = failed_status
        snapshot["legal_review"] = result.model_copy(
            update={
                "status": failed_status,
                "answer": None,
                "gaps": [*result.gaps, str(error)],
            }
        ).model_dump(mode="json")
        save_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
        )
        raise
    ledger.include(numbers)
    snapshot["evidence"] = ledger.export()
    save_asv3_checkpoint(
        message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
    )
    state_container.add_search_docs(list(mapping.values()))
    state_container.set_pre_answer_processing_time(time.monotonic() - started)
    processor = DynamicCitationProcessor(
        citation_mode=CitationMode.HYPERLINK
        if include_citations
        else CitationMode.REMOVE
    )
    processor.update_citation_mapping(mapping)
    state_container.set_citation_mapping(processor.citation_to_doc)
    emitter.emit(
        Packet(
            placement=Placement(turn_index=0),
            obj=AgentResponseStart(
                final_documents=list(mapping.values()),
                pre_answer_processing_seconds=time.monotonic() - started,
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
                state_container.set_answer_tokens("".join(parts))
                emitter.emit(
                    Packet(
                        placement=Placement(turn_index=0),
                        obj=AgentResponseDelta(content=part),
                    )
                )
    state_container.set_citation_mapping(processor.citation_to_doc)
    state_container.set_answer_tokens("".join(parts))
    emitter.emit(Packet(placement=Placement(turn_index=0), obj=SectionEnd()))
    report("completed", context.language)
    snapshot.update(
        sequence=2, progress=progress, processing_seconds=time.monotonic() - started
    )
    save_asv3_checkpoint(
        message_id=assistant_message_id, user_id=user.id, snapshot=snapshot
    )
