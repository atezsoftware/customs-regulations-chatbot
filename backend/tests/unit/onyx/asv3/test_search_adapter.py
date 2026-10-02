"""Run the real V2 search dispatcher and SearchTool with external boundaries faked."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import date
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
)
from onyx.asv3.search_adapter import ScopedSearchLLM, build_search_adapter
from onyx.chat.emitter import NullEmitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import (
    BaseFilters,
    ChunkSearchRequest,
    IndexFilters,
    InferenceChunk,
)
from onyx.context.search.pipeline import search_pipeline
from onyx.db.memory import UserInfo, UserMemoryContext
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import UserMessage
from onyx.tools.tool_implementations.search.search_tool import SearchTool

MODULE = "onyx.tools.tool_implementations.search.search_tool"


def user_message(text: str) -> ChatMessageSimple:
    return ChatMessageSimple(
        message=text, token_count=10, message_type=MessageType.USER
    )


@contextmanager
def search_boundaries(
    chunks: list[InferenceChunk] | None = None,
) -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    pipeline = MagicMock(return_value=chunks or [])
    scope = MagicMock(return_value=[DocumentSource.USER_FILE])
    time = MagicMock(return_value=None)
    with ExitStack() as stack:
        for name, result in {
            "build_access_filters_for_user": [],
            "get_current_search_settings": MagicMock(),
            "get_reranker_configuration": RerankerRuntimeConfig(
                enabled=False,
                provider_type=None,
                model_name=None,
                api_key=None,
                configuration_generation="test",
            ),
            "get_federated_retrieval_functions": [],
            "fetch_unique_document_sources": [
                DocumentSource.USER_FILE,
                DocumentSource.GITHUB,
            ],
        }.items():
            stack.enter_context(patch(f"{MODULE}.{name}", return_value=result))
        stack.enter_context(patch(f"{MODULE}.get_session_with_current_tenant"))
        stack.enter_context(patch(f"{MODULE}.EmbeddingModel"))
        stack.enter_context(patch(f"{MODULE}.search_pipeline", pipeline))
        stack.enter_context(patch(f"{MODULE}.decide_search_scope", scope))
        stack.enter_context(patch(f"{MODULE}.decide_time_filter", time))
        stack.enter_context(
            patch(
                "onyx.asv3.search_adapter.bind_pc_corpus_scope",
                side_effect=lambda **kwargs: kwargs["filters"],
            )
        )
        stack.enter_context(patch(f"{MODULE}.populate_file_ids_on_sections"))
        stack.enter_context(
            patch(
                f"{MODULE}.get_llm_token_counter", return_value=lambda text: len(text)
            )
        )
        stack.enter_context(
            patch(
                f"{MODULE}.get_visible_regulatory_chunk_ids",
                side_effect=lambda _session, identifiers, **_kwargs: set(identifiers),
            )
        )
        for name, result in {
            "get_regulatory_provision_heading_source": None,
            "get_bounded_same_provision_siblings": [],
            "get_bounded_referenced_provisions": [],
            "get_bounded_adjacent_provisions": [],
            "get_bounded_source_lexical_matches": [],
        }.items():
            stack.enter_context(
                patch(
                    f"onyx.regulatory.provision_retrieval.{name}", return_value=result
                )
            )
        yield pipeline, scope, time


def tool_and_broker() -> tuple[SearchTool, CorpusBroker, MagicMock]:
    llm = MagicMock(spec=LLM)

    def invoke(**kwargs: Any) -> ModelResponse:
        return ModelResponse(
            id=str(uuid4()),
            created="0",
            choice=Choice(
                message=Message(
                    content=(
                        "standalone semantic query"
                        if kwargs["max_tokens"] == 512
                        else "lexical variant"
                    )
                )
            ),
        )

    llm.invoke.side_effect = invoke
    tool = SearchTool(
        tool_id=1,
        emitter=NullEmitter(),
        user=MagicMock(is_anonymous=False),
        persona_search_info=MagicMock(document_set_names=[]),
        llm=llm,
        document_index=MagicMock(),
        user_selected_filters=BaseFilters(),
        project_id_filter=None,
        enable_slack_search=False,
        auto_detect_filters=True,
    )
    broker = CorpusBroker(
        tool.user,
        IndexFilters(
            access_control_list=["user:authorized"],
            source_type=[DocumentSource.USER_FILE],
            regulatory_chunks_only=True,
            forced_document_set=["PC Külliyatı"],
            tenant_id="tenant-under-test",
            asv3_document_set_id=15,
        ),
    )
    return tool, broker, llm


def test_hybrid_executes_real_query_expansion_with_actual_filter_history_and_memory() -> (
    None
):
    tool, broker, llm = tool_and_broker()
    memory = UserMemoryContext(
        user_info=UserInfo(name="Researcher"), memories=("scoped-memory",)
    )
    actual_request = "Compare repair rules in PC Külliyatı as of 2025."
    adapter = build_search_adapter(
        tool,
        actual_request,
        broker,
        message_history=lambda _: [user_message(actual_request)],
        user_memory_context=memory,
        user_info="actual-user-profile",
    )
    with search_boundaries() as (pipeline, scope, time):
        outcome = adapter(
            {"query": "model-written repair query", "mode": "hybrid"}, RunContext()
        )

    assert outcome.status == OutcomeStatus.NOT_FOUND
    assert llm.invoke.call_count == 2  # Real semantic and keyword helpers both ran.
    for call in llm.invoke.call_args_list:
        prompt = str(call.kwargs["prompt"])
        assert "model-written repair query" in prompt
        assert "scoped-memory" in prompt
        assert "actual-user-profile" in prompt
    filter_history, _, candidate_sources, _, filter_queries = scope.call_args.args
    assert filter_history[0].message == actual_request
    assert candidate_sources == [DocumentSource.USER_FILE]
    assert filter_queries == ["model-written repair query"]
    assert time.call_args.args[0][0].message == actual_request
    assert pipeline.call_count >= 2
    for call in pipeline.call_args_list:
        filters = call.kwargs["chunk_search_request"].user_selected_filters
        assert filters.forced_document_set == ["PC Külliyatı"]
        assert filters.asv3_document_set_id == 15
        assert filters.tenant_id == "tenant-under-test"
    assert (
        tool.user_selected_filters == BaseFilters()
    )  # Original V2 instance untouched.


@pytest.mark.parametrize(
    "mode,high_coverage", [("keyword", False), ("full_text", True)]
)
def test_explicit_lexical_modes_reach_real_pipeline_without_semantic_expansion(
    mode: str, high_coverage: bool
) -> None:
    tool, broker, llm = tool_and_broker()
    adapter = build_search_adapter(
        tool,
        "user question",
        broker,
        message_history=lambda _: [user_message("user question")],
    )
    with search_boundaries() as (pipeline, _, _):
        outcome = adapter({"query": "unique identifier", "mode": mode}, RunContext())
    assert outcome.status == OutcomeStatus.NOT_FOUND
    llm.invoke.assert_not_called()
    assert pipeline.call_count == 1
    request = pipeline.call_args.kwargs["chunk_search_request"]
    assert request.query == "unique identifier"
    assert request.hybrid_alpha == 0.0
    assert request.high_term_coverage is high_coverage


def test_researcher_task_and_evidence_target_reach_search_receipt_without_invented_query() -> (
    None
):
    tool, broker, llm = tool_and_broker()
    context = RunContext(
        services={
            "search_message_history": [user_message("actual child research task")]
        }
    )
    adapter = build_search_adapter(
        tool,
        "root question",
        broker,
        message_history=lambda ctx: cast(
            list[ChatMessageSimple], ctx.services["search_message_history"]
        ),
    )
    args: dict[str, JsonValue] = {
        "query": "model-written query",
        "mode": "hybrid",
        "coverage_item": "unresolved repair charge basis",
        "evidence_target": "operative valuation formula",
        "source_anchors": ["Named legal instrument"],
    }
    with search_boundaries() as (pipeline, scope, _):
        outcome = adapter(args, context)
    llm.invoke.assert_not_called()  # V2 focused-target semantics preserved.
    assert scope.call_args.args[0][0].message == "actual child research task"
    assert (
        pipeline.call_args.kwargs["chunk_search_request"].query == "model-written query"
    )
    receipt = outcome.data["search_receipt"]
    assert isinstance(receipt, dict)
    assert receipt["receipt"] == {
        "coverage_item": args["coverage_item"],
        "evidence_target": args["evidence_target"],
    }
    assert "user_file" in str(receipt["note"])


@pytest.mark.parametrize(
    "args",
    [
        {"query": "query"},
        {"query": "query", "mode": "semantic"},
        {"query": " ", "mode": "keyword"},
    ],
)
def test_missing_or_unsupported_mode_is_invalid_without_search(
    args: dict[str, JsonValue],
) -> None:
    tool, broker, _ = tool_and_broker()
    adapter = build_search_adapter(
        tool, "request", broker, message_history=lambda _: [user_message("request")]
    )
    with search_boundaries() as (pipeline, _, _):
        assert adapter(args, RunContext()).status == OutcomeStatus.INVALID
    pipeline.assert_not_called()


def test_missing_real_history_is_invalid_and_schema_requires_method_selection() -> None:
    tool, broker, _ = tool_and_broker()
    adapter = build_search_adapter(
        tool, "question cannot replace history", broker, message_history=lambda _: []
    )
    assert (
        adapter({"query": "query", "mode": "hybrid"}, RunContext()).status
        == OutcomeStatus.INVALID
    )
    spec = next(
        spec for spec in build_corpus_specs(broker) if spec.name == "search_corpus"
    )
    assert spec.parameters["required"] == ["query", "mode"]


def test_real_pipeline_retains_mandatory_pc_fence_and_pin_before_retrieval() -> None:
    tool, broker, _ = tool_and_broker()
    broker.filters.as_of_date = date(2025, 1, 1)
    with (
        patch(
            "onyx.context.search.pipeline.search_chunks", return_value=[]
        ) as retrieve,
        patch("onyx.context.search.pipeline.MULTI_TENANT", True),
        patch(
            "onyx.context.search.pipeline.get_current_tenant_id",
            return_value="tenant-under-test",
        ),
        patch(
            "onyx.context.search.pipeline.fetch_ee_implementation_or_noop",
            return_value=lambda **kwargs: kwargs["chunks"],
        ),
    ):
        search_pipeline(
            chunk_search_request=ChunkSearchRequest(
                query="query", user_selected_filters=broker.filters
            ),
            document_index=tool.document_index,
            user=broker.user,
            persona_search_info=None,
            acl_filters=["user:authorized"],
            prefetched_federated_retrieval_infos=[],
        )
    final = retrieve.call_args.kwargs["query_request"].filters
    assert final.forced_document_set == ["PC Külliyatı"]
    assert final.asv3_document_set_id == 15
    assert final.source_type == [DocumentSource.USER_FILE]
    assert final.access_control_list == ["user:authorized"]
    assert final.tenant_id == "tenant-under-test"
    assert final.as_of_date == date(2025, 1, 1)


@pytest.mark.parametrize(
    "bypass,source,forced",
    [
        (True, DocumentSource.USER_FILE, ["PC Külliyatı"]),
        (False, DocumentSource.GITHUB, ["PC Külliyatı"]),
        (False, DocumentSource.USER_FILE, None),
    ],
)
def test_pinned_scope_cannot_widen_before_index_retrieval(
    bypass: bool, source: DocumentSource, forced: list[str] | None
) -> None:
    tool, broker, _ = tool_and_broker()
    broker.filters.source_type = [source]
    broker.filters.forced_document_set = forced
    with patch("onyx.context.search.pipeline.search_chunks") as retrieve:
        with pytest.raises(OnyxError, match="authorized PC corpus"):
            search_pipeline(
                chunk_search_request=ChunkSearchRequest(
                    query="query",
                    user_selected_filters=broker.filters,
                    bypass_acl=bypass,
                ),
                document_index=tool.document_index,
                user=broker.user,
                persona_search_info=None,
                acl_filters=[],
                prefetched_federated_retrieval_infos=[],
            )
    retrieve.assert_not_called()


def test_project_path_retains_pc_pin_without_changing_unpinned_v2_behavior() -> None:
    tool, broker, _ = tool_and_broker()
    tool.project_id_filter = 42
    with patch(f"{MODULE}.search_pipeline", return_value=[]) as pipeline:
        for filters in [broker.filters, BaseFilters()]:
            tool._run_search_for_query(
                query="query",
                hybrid_alpha=0.0,
                high_term_coverage=False,
                num_hits=10,
                acl_filters=[],
                embedding_model=MagicMock(),
                federated_retrieval_infos=[],
                effective_filters=filters,
            )
    assert (
        pipeline.call_args_list[0].kwargs["chunk_search_request"].user_selected_filters
        == broker.filters
    )
    assert (
        pipeline.call_args_list[1].kwargs["chunk_search_request"].user_selected_filters
        is None
    )


def test_secondary_search_llm_preserves_selected_config_identity_and_budget() -> None:
    _, _, selected = tool_and_broker()
    context = RunContext(timeout_seconds=10)
    identity = LLMUserIdentity(user_id="actual-user", session_id="actual-session")
    scoped = ScopedSearchLLM(selected, context, identity)
    assert scoped.config is selected.config
    scoped.invoke(
        [UserMessage(content="actual prompt")], max_tokens=512, timeout_override=99
    )
    assert selected.invoke.call_args.kwargs["user_identity"] == identity
    assert 1 <= selected.invoke.call_args.kwargs["timeout_override"] <= 10
    assert context.budget.snapshot()["decisions"] == 1
    assert context.budget.model_slots.acquire(blocking=False)
    context.budget.model_slots.release()
    context.cancel()
    with pytest.raises(RunStopped):
        scoped.invoke([UserMessage(content="cancelled")], max_tokens=512)
    assert selected.invoke.call_count == 1


def test_secondary_search_retry_is_budgeted_and_final_reserve_is_not_spent() -> None:
    _, _, selected = tool_and_broker()
    response = selected.invoke(max_tokens=512)
    selected.invoke.reset_mock()
    selected.invoke.side_effect = [RuntimeError("retryable failure"), response]
    context = RunContext(budget=SharedBudget(max_decisions=4, final_decision_reserve=2))
    scoped = ScopedSearchLLM(selected, context, None)
    with (
        patch(
            "onyx.asv3.search_adapter.is_retryable_provider_error", return_value=True
        ),
        patch("onyx.asv3.search_adapter.provider_retry_delay", return_value=0),
    ):
        assert scoped.invoke([UserMessage(content="retry")], max_tokens=512) == response
    assert selected.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 2
    with pytest.raises(RunStopped, match="finalization reserve"):
        scoped.invoke([UserMessage(content="reserve protected")], max_tokens=512)
    assert selected.invoke.call_count == 2


def test_real_search_results_are_canonically_hydrated_once_without_payload_text_duplication() -> (
    None
):
    tool, broker, _ = tool_and_broker()
    source_id = str(uuid4())
    chunks = [
        InferenceChunk(
            document_id=source_id,
            chunk_id=n,
            content=f"canonical paragraph {n}",
            source_type=DocumentSource.USER_FILE,
            semantic_identifier="Named source",
            title="Named source",
            boost=1,
            score=0.9,
            hidden=False,
            metadata={},
            match_highlights=[],
            doc_summary="",
            chunk_context="",
            updated_at=None,
            image_file_id=None,
            source_links=None,
            section_continuation=False,
            blurb="source paragraph",
            file_id=source_id,
            regulatory_chunk_id=f"rc-{n}",
            heading_path=["MADDE 142"],
        )
        for n in (1, 2)
    ]
    originals = [
        EvidenceItem(
            source_id=source_id, chunk_id=f"rc-{n}", text=f"canonical paragraph {n}"
        )
        for n in (1, 2)
    ]
    adapter = build_search_adapter(
        tool,
        "actual request",
        broker,
        message_history=lambda _: [user_message("actual request")],
    )
    with (
        search_boundaries(chunks),
        patch.object(
            broker,
            "hydrate_search_results",
            return_value={(source_id, n): originals for n in (1, 2)},
        ) as hydrate,
    ):
        outcome = adapter(
            {"query": "named repair mechanism", "mode": "keyword"}, RunContext()
        )
    assert outcome.status == OutcomeStatus.FOUND
    assert hydrate.call_count == 1
    assert [item.identity for item in outcome.evidence] == [
        item.identity for item in originals
    ]
    assert "canonical paragraph" not in str(outcome.data)
