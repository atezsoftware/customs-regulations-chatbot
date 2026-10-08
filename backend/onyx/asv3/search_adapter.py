"""Isolate the established search pipeline and retain its original source text."""

import json
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from uuid import uuid4

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.llm_adapter import model_slot, provider_retry_delay
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
)
from onyx.asv3.parallel_execution import parallel_execution_enabled
from onyx.asv3.workflow_variant import (
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_TUNED_VARIANT,
)
from onyx.chat.emitter import NullEmitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDocsResponse
from onyx.context.search.retrieval.parallel_retrieval_scope import (
    experimental_parallel_retrieval,
)
from onyx.context.search.retrieval.query_embedding_scope import (
    ParallelQueryEmbeddingScope,
    experimental_parallel_query_embeddings,
)
from onyx.db.asv3_candidate_inventory import asv3_source_inventory_scope
from onyx.db.asv3_corpus import bind_pc_corpus_scope
from onyx.db.memory import UserMemoryContext
from onyx.llm.interfaces import LLM, LLMConfig, LLMUserIdentity
from onyx.llm.model_response import ModelResponse, ModelResponseStream
from onyx.llm.models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions
from onyx.regulatory.structured_llm import is_retryable_provider_error
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tools.models import (
    ChatMinimalTextMessage,
    SearchToolRetrievalOverrides,
    ToolCallKickoff,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_runner import run_tool_calls

_TUNED_RETRIEVAL_OVERRIDES = SearchToolRetrievalOverrides(
    per_lane_num_hits=256,
    rerank_candidate_limit=384,
    regulatory_rerank_candidate_limit=384,
    max_llm_chunks=50,
    preserve_source_diversity=True,
    reuse_diversity_comparisons=True,
)

_GUARDED_RETRIEVAL_OVERRIDES = SearchToolRetrievalOverrides(
    per_lane_num_hits=192,
    rerank_candidate_limit=256,
    regulatory_rerank_candidate_limit=256,
    max_llm_chunks=32,
    preserve_source_diversity=True,
    reuse_diversity_comparisons=True,
    guarded_decisions_advisory=True,
)


def guarded_retrieval_overrides(
    workflow_variant: str | None,
) -> SearchToolRetrievalOverrides | None:
    if workflow_variant == ASV3_GUARDED_EXPERIMENTAL_VARIANT:
        return _GUARDED_RETRIEVAL_OVERRIDES
    if workflow_variant == ASV3_TUNED_VARIANT:
        return _TUNED_RETRIEVAL_OVERRIDES
    return None


class ScopedSearchLLM(LLM):
    """Retain the selected model while accounting for secondary search generations."""

    def __init__(
        self, selected: LLM, context: RunContext, user_identity: LLMUserIdentity | None
    ) -> None:
        self.selected = selected
        self.context = context
        self.user_identity = user_identity

    @property
    def config(self) -> LLMConfig:
        return self.selected.config

    def timeout(self, requested: int | None) -> int | None:
        self.context.check_research_active()
        if self.context.research_deadline == float("inf"):
            return requested
        remaining = max(1, int(self.context.research_deadline - time.monotonic()))
        return remaining if requested is None else max(1, min(requested, remaining))

    def wait_to_retry(self, error: Exception, attempt: int) -> None:
        self.context.check_research_active()
        delay = provider_retry_delay(error, attempt)
        if delay >= self.context.research_deadline - time.monotonic():
            raise RunStopped(
                "Provider retry exceeds the remaining search deadline"
            ) from error
        retry_at = time.monotonic() + delay
        while time.monotonic() < retry_at:
            self.context.check_research_active()
            time.sleep(min(0.05, max(0, retry_at - time.monotonic())))

    def provider_max_attempts(self) -> int:
        configured = self.context.services.get("provider_max_attempts", 3)
        if isinstance(configured, int) and not isinstance(configured, bool):
            return max(1, configured)
        return 3

    def provider_compatibility_attempts(
        self, requested: int | None
    ) -> int | None:
        if requested is not None:
            return requested
        configured = self.context.services.get("provider_compatibility_attempts")
        if isinstance(configured, int) and not isinstance(configured, bool):
            return max(1, configured)
        return None

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        use_streaming: bool = True,
        provider_compatibility_attempts: int | None = None,
    ) -> ModelResponse:
        max_attempts = self.provider_max_attempts()
        compatibility_attempts = self.provider_compatibility_attempts(
            provider_compatibility_attempts
        )
        for attempt in range(max_attempts):
            try:
                with model_slot(self.context, research=True):
                    self.context.consume_research_decision()
                    result = self.selected.invoke(
                        prompt=prompt,
                        tools=tools,
                        tool_choice=tool_choice,
                        structured_response_format=structured_response_format,
                        timeout_override=self.timeout(timeout_override),
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        user_identity=user_identity or self.user_identity,
                        use_streaming=use_streaming,
                        provider_compatibility_attempts=compatibility_attempts,
                    )
                    self.context.check_research_active()
                    return result
            except Exception as error:
                if attempt == max_attempts - 1 or not is_retryable_provider_error(error):
                    raise
                self.wait_to_retry(error, attempt)
        raise AssertionError("Selected-provider retry loop did not terminate")

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
    ) -> Iterator[ModelResponseStream]:
        max_attempts = self.provider_max_attempts()
        for attempt in range(max_attempts):
            emitted = False
            try:
                with model_slot(self.context, research=True):
                    self.context.consume_research_decision()
                    for part in self.selected.stream(
                        prompt=prompt,
                        tools=tools,
                        tool_choice=tool_choice,
                        structured_response_format=structured_response_format,
                        timeout_override=self.timeout(timeout_override),
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        user_identity=user_identity or self.user_identity,
                    ):
                        self.context.check_research_active()
                        emitted = True
                        yield part
                    self.context.check_research_active()
                    return
            except Exception as error:
                if (
                    emitted
                    or attempt == max_attempts - 1
                    or not is_retryable_provider_error(error)
                ):
                    raise
                self.wait_to_retry(error, attempt)


class ScopedSearchAdapter:
    def __init__(
        self,
        search: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
        prepare: Callable[[list[CapabilityCall], RunContext], dict[str, SearchTool]],
    ) -> None:
        self._search = search
        self._prepare = prepare

    def __call__(self, args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        return self._search(args, context)

    def prepare_batch(
        self, calls: list[CapabilityCall], context: RunContext
    ) -> dict[str, SearchTool]:
        """Allocate per-call search state without running any model or source I/O."""
        return self._prepare(calls, context)


def build_search_adapter(
    tool: SearchTool | None,
    original_query: str,
    broker: CorpusBroker,
    *,
    message_history: Callable[[RunContext], list[ChatMessageSimple]],
    user_memory_context: UserMemoryContext | None = None,
    user_info: str | None = None,
    inject_memories_in_prompt: bool = True,
    user_identity: LLMUserIdentity | None = None,
) -> ScopedSearchAdapter:
    def prepare_batch(
        calls: list[CapabilityCall], context: RunContext
    ) -> dict[str, SearchTool]:
        context.check_active()
        if tool is None:
            return {}
        eligible: list[CapabilityCall] = []
        queries: list[str] = []
        for call in calls:
            query, mode = call.arguments.get("query"), call.arguments.get("mode")
            if (
                call.name != "search_corpus"
                or call.argument_error is not None
                or not isinstance(query, str)
                or not query.strip()
                or len(query) > REGULATORY_MAX_SEARCH_QUERY_CHARS
                or mode not in ("hybrid", "keyword", "full_text")
            ):
                continue
            eligible.append(call)
            queries.append(query.strip())
        if not eligible:
            return {}
        history = message_history(context)
        user_messages = [
            message.message
            for message in history
            if message.message_type == MessageType.USER
        ]
        if not user_messages:
            return {}
        shared_history = [
            ChatMinimalTextMessage(
                message=message.message, message_type=message.message_type
            )
            for message in history
            if message.message_type in {MessageType.USER, MessageType.ASSISTANT}
        ]
        expansion_history = [
            *shared_history,
            ChatMinimalTextMessage(
                message=(
                    "Current model-selected search queries:\n" + "\n".join(queries)
                ),
                message_type=MessageType.USER,
            ),
        ]
        scoped = tool.fork_for_independent_context(emitter=NullEmitter())
        scoped.user_selected_filters = broker.filters.model_copy(deep=True)
        forks = scoped.fork_for_query_batch(
            len(eligible),
            message_history=expansion_history,
            filter_message_history=shared_history,
            queries=queries,
        )
        return {call.call_id: fork for call, fork in zip(eligible, forks, strict=True)}

    def search_in_scope(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        if tool is None:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Internal search is not configured",
            )
        query, mode = args.get("query"), args.get("mode")
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > REGULATORY_MAX_SEARCH_QUERY_CHARS
            or mode not in ("hybrid", "keyword", "full_text")
        ):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Search requires a bounded non-empty query and an explicit hybrid, keyword or full_text mode",
            )
        history = message_history(context)
        if not any(message.message_type == MessageType.USER for message in history):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Search requires the actual conversation or researcher user/task history",
            )
        prepared = context.services.get("search_batch_tool")
        isolated = (
            prepared
            if isinstance(prepared, SearchTool)
            else tool.fork_for_independent_context(emitter=NullEmitter())
        )
        # Every index lane must use the same captured authorization as direct reads.
        isolated.user_selected_filters = bind_pc_corpus_scope(
            user=broker.user, filters=broker.filters.model_copy(deep=True)
        )
        isolated.llm = ScopedSearchLLM(tool.llm, context, user_identity)
        tool_args: dict[str, JsonValue] = {"queries": [query], "search_mode": mode}
        for field in (
            "coverage_item",
            "evidence_target",
            "source_anchors",
            "label_hint",
            "expand_query",
        ):
            if field in args:
                tool_args[field] = args[field]
        embedding_scope = context.services.get("parallel_query_embeddings")
        scoped_embeddings = (
            experimental_parallel_query_embeddings(
                embedding_scope, check_active=context.check_research_active
            )
            if parallel_execution_enabled(context)
            and isinstance(embedding_scope, ParallelQueryEmbeddingScope)
            else nullcontext()
        )
        scoped_retrieval = (
            experimental_parallel_retrieval(check_active=context.check_research_active)
            if parallel_execution_enabled(context)
            else nullcontext()
        )
        with scoped_embeddings, scoped_retrieval:
            batch = run_tool_calls(
                tool_calls=[
                    ToolCallKickoff(
                        tool_call_id=str(uuid4()),
                        tool_name=SearchTool.NAME,
                        tool_args=tool_args,
                        placement=Placement(turn_index=len(history)),
                    )
                ],
                tools=[isolated],
                message_history=history,
                user_memory_context=user_memory_context,
                user_info=user_info,
                citation_mapping={},
                next_citation_num=1,
                inject_memories_in_prompt=inject_memories_in_prompt,
                tool_execution_timeout_seconds=None,
                search_rerank_context=original_query,
                search_retrieval_overrides=guarded_retrieval_overrides(
                    context.services.get("asv3_workflow_variant")
                    if isinstance(
                        context.services.get("asv3_workflow_variant"), str
                    )
                    else None
                ),
            )
        context.check_active()
        if len(batch.tool_responses) != 1:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="The scoped search invocation returned no tool response; this is not proof of corpus absence",
            )
        response = batch.tool_responses[0]
        rich = response.rich_response
        if not isinstance(rich, SearchDocsResponse):
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Search did not return a usable source mapping; this is not proof of corpus absence",
                data={"search_response": response.llm_facing_response},
            )
        payload = json.loads(response.llm_facing_response)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        evidence: list[EvidenceItem] = []
        seen_evidence: set[tuple[str, str | None, str]] = set()
        unmapped_results = 0
        docs = {(doc.document_id, doc.chunk_ind): doc for doc in rich.search_docs}
        mapped_docs = []
        for result in results:
            if not isinstance(result, dict):
                unmapped_results += 1
                continue
            number, text = result.get("document"), result.get("content")
            if (
                not isinstance(number, int)
                or not isinstance(text, str)
                or not text.strip()
            ):
                unmapped_results += 1
                continue
            doc_id = rich.citation_mapping.get(number)
            chunk = rich.citation_chunk_mapping.get(number)
            if doc_id is None or chunk is None:
                unmapped_results += 1
                continue
            doc = docs.get((doc_id, chunk))
            if doc is None:
                unmapped_results += 1
                continue
            mapped_docs.append(doc)
        hydrated_results = broker.hydrate_search_centers(mapped_docs, context)
        incomplete_closures = 0
        unhydrated_centers: list[JsonValue] = []
        for doc in mapped_docs:
            hydrated = hydrated_results.get((doc.document_id, doc.chunk_ind), [])
            if not hydrated:
                unmapped_results += 1
                unhydrated_centers.append(
                    {
                        "source_id": doc.document_id,
                        "canonical_chunk_id": doc.metadata.get("regulatory_chunk_id"),
                        "projection_ordinal": doc.chunk_ind,
                        "instruction": "The center was not delivered as original evidence. Read its authorized source directly or report the exact gap; do not infer absence.",
                    }
                )
            elif any(
                item.metadata.get("article_closure_complete") is False
                for item in hydrated
            ):
                incomplete_closures += 1
            for item in hydrated:
                if item.identity not in seen_evidence:
                    seen_evidence.add(item.identity)
                    evidence.append(item)
        status = (
            OutcomeStatus.PARTIAL
            if evidence and unmapped_results
            else OutcomeStatus.FOUND
            if evidence
            else OutcomeStatus.UNAVAILABLE
            if results
            else OutcomeStatus.NOT_FOUND
        )
        return ToolOutcome(
            status=status,
            summary="Exact retrieved original text. Local context is bounded, not a complete provision; choose further reading when the claim needs it.",
            data={
                "source_count": len(evidence),
                "retrieved_result_count": len(results),
                "mapped_result_count": len(mapped_docs),
                "hydrated_center_count": len(mapped_docs) - len(unhydrated_centers),
                "retained_evidence_count": len(evidence),
                "unmapped_result_count": unmapped_results,
                "incomplete_closure_count": incomplete_closures,
                "context_policy": "harness_controlled",
                "unhydrated_centers": unhydrated_centers,
                "query": query,
                "mode": mode,
                "original_question": original_query,
                "search_receipt": {
                    key: value for key, value in payload.items() if key != "results"
                }
                if isinstance(payload, dict)
                else {},
            },
            evidence=evidence,
        )

    def search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        scope_key = json.dumps(
            {
                "run_id": context.run_id,
                "user_id": str(broker.user.id),
                "scope": context.scope,
                "filters": broker.filters.model_dump(mode="json"),
            },
            sort_keys=True,
        )
        with asv3_source_inventory_scope(
            scope_key=scope_key, check_active=context.check_active
        ):
            return search_in_scope(args, context)

    return ScopedSearchAdapter(search, prepare_batch)
