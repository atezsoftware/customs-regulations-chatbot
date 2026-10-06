from datetime import date
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime, search_adapter, serial_experimental_session
from onyx.asv3.harness import Harness
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.parallel_execution import ParallelExecutionSlots
from onyx.asv3.search_adapter import build_search_adapter
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.context.search.retrieval.query_embedding_scope import (
    ParallelQueryEmbeddingScope,
    QueryEmbeddingScopeBinding,
    current_parallel_query_embedding_scope,
)
from tests.unit.onyx.asv3.test_experimental_parallel_runtime import script_two_children
from tests.unit.onyx.asv3.test_search_adapter import (
    search_boundaries,
    tool_and_broker,
    user_message,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize(
    "profile,parallel,diagnostics,owner,depth,expected_bound",
    [
        ("experimental", True, False, None, 0, True),
        ("experimental", False, True, "owned-question", 0, True),
        ("normal", False, False, None, 0, False),
        ("deep", False, False, None, 0, False),
        ("experimental", False, False, None, 0, False),
        ("experimental", False, 1, "owned-question", 0, False),
        ("experimental", False, True, None, 0, False),
        ("experimental", False, True, "owned-question", 1, False),
    ],
)
def test_actual_dispatch_binds_only_owned_parallel_scope_and_preserves_originals(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    parallel: bool,
    diagnostics: bool | int,
    owner: str | None,
    depth: int,
    expected_bound: bool,
) -> None:
    tool, broker, selected = tool_and_broker()
    broker.filters.as_of_date = date(2025, 1, 1)
    source_id = str(uuid4())
    original = EvidenceItem(
        source_id=source_id,
        chunk_id="canonical-one",
        text="The complete operative condition and exception. İĞŞ 東京.",
    )
    continuation = EvidenceItem(
        source_id=source_id,
        chunk_id="canonical-two",
        text="The complete continuation and later procedural consequence.",
    )
    chunk = InferenceChunk(
        document_id=source_id,
        chunk_id=0,
        content="Retrieval projection, never the canonical original",
        source_type=DocumentSource.USER_FILE,
        semantic_identifier="Verified source",
        title="Verified source",
        boost=1,
        score=0.9,
        blurb="Navigation blurb",
        source_links={},
        image_file_id=None,
        section_continuation=False,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        file_id=source_id,
        regulatory_chunk_id=original.chunk_id,
        heading_path=["MADDE 9"],
    )
    adapter = build_search_adapter(
        tool,
        "full fixed scenario",
        broker,
        message_history=lambda _: [user_message("full fixed scenario")],
    )
    scope = ParallelQueryEmbeddingScope()
    services: dict[str, object] = {
        "lean_native_mode": True,
        "research_profile": profile,
        "experimental_parallel": parallel,
        "serial_session_diagnostics": diagnostics,
        "parallel_query_embeddings": scope,
    }
    if owner is not None:
        services["task_id"] = owner
    context = RunContext(depth=depth, services=services)
    baseline = RunContext(
        depth=depth,
        services={
            k: v for k, v in services.items() if k != "parallel_query_embeddings"
        },
    )
    dispatch = search_adapter.run_tool_calls
    bindings: list[QueryEmbeddingScopeBinding | None] = []
    arguments_seen: list[dict[str, Any]] = []

    def capture(**arguments: Any) -> Any:
        bindings.append(current_parallel_query_embedding_scope())
        arguments_seen.append(arguments)
        return dispatch(**arguments)

    monkeypatch.setattr(search_adapter, "run_tool_calls", capture)
    arguments: dict[str, JsonValue] = {
        "query": "exact independent anchor",
        "mode": "keyword",
        "coverage_item": "one independent outcome",
        "evidence_target": "actual operative condition and continuation",
        "expand_query": False,
    }
    with (
        search_boundaries([chunk]) as (pipeline, _scope_decision, _time_decision),
        patch.object(
            broker,
            "hydrate_search_centers",
            side_effect=lambda docs, _ctx: {
                (doc.document_id, doc.chunk_ind): [original, continuation]
                for doc in docs
            },
        ) as hydrate,
    ):
        unchanged = adapter(arguments, baseline)
        actual = adapter(arguments, context)

    assert bindings[0] is None
    if expected_bound:
        binding = bindings[1]
        assert binding is not None and binding.scope is scope
        assert binding.check_active == context.check_research_active
    else:
        assert bindings[1] is None
    assert current_parallel_query_embedding_scope() is None
    assert actual.status == unchanged.status == OutcomeStatus.FOUND
    assert actual.evidence == unchanged.evidence == [original, continuation]
    assert actual.data == unchanged.data
    assert hydrate.call_count == pipeline.call_count == 2
    assert selected.invoke.call_count == 0
    for invocation in pipeline.call_args_list:
        request = invocation.kwargs["chunk_search_request"]
        assert request.query == arguments["query"]
        assert request.hybrid_alpha == 0.0
        filters = request.user_selected_filters
        assert filters.tenant_id == "tenant-under-test"
        assert filters.access_control_list == ["user:authorized"]
        assert filters.forced_document_set == ["PC Külliyatı"]
        assert filters.asv3_document_set_id == 15
        assert filters.as_of_date == date(2025, 1, 1)
    for invocation in arguments_seen:
        assert invocation["tool_calls"][0].tool_args == {
            "queries": [arguments["query"]],
            "search_mode": arguments["mode"],
            "coverage_item": arguments["coverage_item"],
            "evidence_target": arguments["evidence_target"],
            "expand_query": False,
        }
        assert invocation["search_rerank_context"] == "full fixed scenario"


def test_dispatch_failure_clears_hosted_embedding_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool, broker, _selected = tool_and_broker()
    adapter = build_search_adapter(
        tool,
        "fixed scenario",
        broker,
        message_history=lambda _: [user_message("fixed scenario")],
    )
    scope = ParallelQueryEmbeddingScope()
    context = RunContext(
        services={
            "lean_native_mode": True,
            "research_profile": "experimental",
            "experimental_parallel": False,
            "serial_session_diagnostics": True,
            "task_id": "owned-question",
            "parallel_query_embeddings": scope,
        }
    )

    def fail(**_arguments: Any) -> Any:
        binding = current_parallel_query_embedding_scope()
        assert binding is not None and binding.scope is scope
        raise RuntimeError("synthetic dispatcher failure")

    monkeypatch.setattr(search_adapter, "run_tool_calls", fail)
    with search_boundaries():
        with pytest.raises(RuntimeError, match="synthetic dispatcher failure"):
            adapter({"query": "exact anchor", "mode": "keyword"}, context)
    assert current_parallel_query_embedding_scope() is None
    assert context.services["parallel_query_embeddings"] is scope


def test_real_runtime_children_share_exact_embedding_and_canonical_io_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[RunContext] = []
    actual_harness = runtime.Harness

    def capture(**arguments: Any) -> Harness:
        context = arguments["context"]
        assert isinstance(context, RunContext)
        contexts.append(context)
        return actual_harness(**arguments)

    monkeypatch.setattr(runtime, "Harness", capture)
    monkeypatch.setattr(serial_experimental_session, "Harness", capture)
    (
        kwargs,
        selected,
        secondary,
        checkpoints,
        _queue,
        _fence,
        read,
        bodies,
        _requests,
    ) = script_two_children(monkeypatch)
    selected.config = selected.config.model_copy(
        update={"model_provider": "openai", "model_name": "gpt-6-luna"}
    )
    runtime.run_asv3_loop(**kwargs)

    root, *children = contexts
    assert len(children) == 2
    scope = root.services["parallel_query_embeddings"]
    io_slots = root.services["parallel_execution_slots"]
    assert isinstance(scope, ParallelQueryEmbeddingScope)
    assert isinstance(io_slots, ParallelExecutionSlots)
    for child in children:
        assert child.services["parallel_query_embeddings"] is scope
        assert child.services["parallel_execution_slots"] is io_slots
        assert child.services["experimental_parallel"] is False
        assert child.services["serial_session_diagnostics"] is True
        assert child.budget.tool_slots is root.budget.tool_slots
        assert child.budget.model_slots is root.budget.model_slots
        assert child.budget.source_slots is root.budget.source_slots
    assert selected.invoke.call_count == 5
    secondary.invoke.assert_not_called()
    assert read.call_count == 1
    assert checkpoints[-1]["publication_status"] == "found"
    assert {
        row["answer"] for row in checkpoints[-1]["parallel_answers"]["receipts"]
    } == (set(bodies.values()))
