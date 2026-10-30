"""Fast acquisition opt-in preserves search inputs, originals and scope cleanup."""

import copy
import json
from datetime import date
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3 import search_adapter
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, RunStopped
from onyx.asv3.search_adapter import build_search_adapter
from onyx.context.search.retrieval.parallel_retrieval_scope import (
    check_parallel_retrieval_active,
    experimental_parallel_retrieval,
    parallel_retrieval_enabled,
)
from onyx.context.search.retrieval.query_embedding_scope import (
    ParallelQueryEmbeddingScope,
    current_parallel_query_embedding_scope,
    experimental_parallel_query_embeddings,
)
from tests.unit.onyx.asv3.test_search_adapter import (
    search_boundaries,
    tool_and_broker,
    user_message,
)
from tests.unit.onyx.regulatory.labeling.test_search_overlay import chunk

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "full_text"])
def test_fast_tek_keeps_queries_filters_ranked_originals_and_receipts_identical(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    tool, broker, _selected = tool_and_broker()
    broker.filters.as_of_date = date(2025, 1, 1)
    source_id = str(uuid4())
    canonical = [
        EvidenceItem(
            source_id=source_id,
            chunk_id="operative",
            text="A AND B; exception C. İĞŞ 東京.",
        ),
        EvidenceItem(
            source_id=source_id,
            chunk_id="continuation",
            text="Complete later procedure D.",
        ),
    ]
    retrieved = chunk("1")
    retrieved.document_id = source_id
    retrieved.regulatory_chunk_id = canonical[0].chunk_id
    adapter = build_search_adapter(
        tool,
        "full fixed scenario",
        broker,
        message_history=lambda _: [user_message("full fixed scenario")],
    )
    embedding_scope = ParallelQueryEmbeddingScope(cache_queries=False)
    baseline = RunContext(services={"research_profile": "normal"})
    fast = RunContext(
        services={
            "research_profile": "normal",
            "asv3_fast": True,
            "asv3_fast_retrieval": True,
            "parallel_query_embeddings": embedding_scope,
        }
    )
    dispatched: list[dict[str, Any]] = []
    actual_dispatch = search_adapter.run_tool_calls

    def capture(**arguments: Any) -> Any:
        binding = current_parallel_query_embedding_scope()
        if len(dispatched) == 0:
            assert not parallel_retrieval_enabled() and binding is None
        else:
            assert parallel_retrieval_enabled()
            assert binding is not None and binding.scope is embedding_scope
            assert binding.check_active == fast.check_research_active
        dispatched.append(arguments)
        return actual_dispatch(**arguments)

    monkeypatch.setattr(search_adapter, "run_tool_calls", capture)
    arguments: dict[str, JsonValue] = {
        "query": "exact original anchor",
        "mode": mode,
        "coverage_item": "one requested outcome",
        "evidence_target": "operative condition and complete continuation",
    }
    before = copy.deepcopy(arguments)
    with (
        search_boundaries([retrieved]) as (pipeline, _, _),
        patch.object(
            broker,
            "hydrate_search_centers",
            side_effect=lambda docs, _ctx: {
                (doc.document_id, doc.chunk_ind): canonical for doc in docs
            },
        ),
    ):
        old = adapter(arguments, baseline)
        baseline_requests = [
            call.kwargs["chunk_search_request"].model_dump(
                mode="json", serialize_as_any=True
            )
            for call in pipeline.call_args_list
        ]
        pipeline.reset_mock()
        actual = adapter(arguments, fast)
        fast_requests = [
            call.kwargs["chunk_search_request"].model_dump(
                mode="json", serialize_as_any=True
            )
            for call in pipeline.call_args_list
        ]
    assert actual.status == old.status == OutcomeStatus.FOUND
    assert actual.evidence == old.evidence == canonical
    assert actual.data == old.data
    assert arguments == before
    assert sorted(
        map(lambda row: json.dumps(row, sort_keys=True), fast_requests)
    ) == sorted(map(lambda row: json.dumps(row, sort_keys=True), baseline_requests))
    assert len(dispatched) == 2
    for call in dispatched:
        assert call["tool_calls"][0].tool_args == {
            "queries": [arguments["query"]],
            "search_mode": mode,
            "coverage_item": arguments["coverage_item"],
            "evidence_target": arguments["evidence_target"],
        }
        assert call["search_rerank_context"] == "full fixed scenario"
        assert call["search_retrieval_overrides"] is None
    for request in fast_requests:
        filters = request["user_selected_filters"]
        assert filters["tenant_id"] == "tenant-under-test"
        assert filters["access_control_list"] == ["user:authorized"]
        assert filters["forced_document_set"] == ["PC Külliyatı"]
        assert filters["asv3_document_set_id"] == 15
        assert filters["as_of_date"] == "2025-01-01"
    assert not parallel_retrieval_enabled()
    assert current_parallel_query_embedding_scope() is None


@pytest.mark.parametrize(
    "services,expected",
    [
        ({}, False),
        ({"asv3_fast": True}, False),
        ({"asv3_fast": True, "asv3_exp_cropped": True}, False),
        ({"asv3_fast_retrieval": False}, False),
        ({"asv3_fast_retrieval": 1}, False),
        ({"asv3_fast_retrieval": "true"}, False),
        ({"asv3_fast_retrieval": True}, True),
        (
            {"asv3_fast": True, "asv3_exp_cropped": True, "asv3_fast_retrieval": True},
            True,
        ),
        ({"research_profile": "experimental", "experimental_parallel": True}, True),
    ],
)
def test_only_exact_fast_retrieval_or_existing_parallel_opt_in_binds_scopes(
    monkeypatch: pytest.MonkeyPatch, services: dict[str, object], expected: bool
) -> None:
    tool, broker, _selected = tool_and_broker()
    adapter = build_search_adapter(
        tool, "scenario", broker, message_history=lambda _: [user_message("scenario")]
    )
    scope = ParallelQueryEmbeddingScope(cache_queries=False)
    context = RunContext(
        services={
            "research_profile": "normal",
            "parallel_query_embeddings": scope,
            **services,
        }
    )
    dispatch = search_adapter.run_tool_calls

    def capture(**arguments: Any) -> Any:
        binding = current_parallel_query_embedding_scope()
        assert parallel_retrieval_enabled() is expected
        if expected:
            assert binding is not None and binding.scope is scope
        else:
            assert binding is None
        return dispatch(**arguments)

    monkeypatch.setattr(search_adapter, "run_tool_calls", capture)
    with search_boundaries():
        assert (
            adapter({"query": "exact anchor", "mode": "keyword"}, context).status
            is OutcomeStatus.NOT_FOUND
        )
    assert not parallel_retrieval_enabled()
    assert current_parallel_query_embedding_scope() is None


@pytest.mark.parametrize("failure", ["dispatch", "cancel_before", "cancel_during"])
def test_fast_scope_restores_outer_bindings_on_failure_or_cancellation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tool, broker, selected = tool_and_broker()
    adapter = build_search_adapter(
        tool, "scenario", broker, message_history=lambda _: [user_message("scenario")]
    )
    inner, outer = (
        ParallelQueryEmbeddingScope(cache_queries=False),
        ParallelQueryEmbeddingScope(),
    )
    context = RunContext(
        services={
            "research_profile": "normal",
            "asv3_fast_retrieval": True,
            "parallel_query_embeddings": inner,
        }
    )
    calls: list[str] = []
    outer_checks: list[str] = []

    def fail(**_arguments: Any) -> Any:
        binding = current_parallel_query_embedding_scope()
        assert binding is not None and binding.scope is inner
        assert parallel_retrieval_enabled()
        calls.append("dispatch")
        if failure == "cancel_during":
            context.cancel()
            context.check_research_active()
        raise RuntimeError("synthetic dispatch failure")

    monkeypatch.setattr(search_adapter, "run_tool_calls", fail)
    with (
        experimental_parallel_query_embeddings(outer, check_active=lambda: None),
        experimental_parallel_retrieval(
            check_active=lambda: outer_checks.append("outer")
        ),
        search_boundaries(),
    ):
        outer_binding = current_parallel_query_embedding_scope()
        if failure == "cancel_before":
            context.cancel()
        with pytest.raises(
            RunStopped if failure.startswith("cancel") else RuntimeError
        ):
            adapter({"query": "exact anchor", "mode": "keyword"}, context)
        assert current_parallel_query_embedding_scope() is outer_binding
        check_parallel_retrieval_active()
        assert outer_checks == ["outer", "outer"]
    assert calls == ([] if failure == "cancel_before" else ["dispatch"])
    selected.invoke.assert_not_called()
    assert not parallel_retrieval_enabled()
    assert current_parallel_query_embedding_scope() is None
