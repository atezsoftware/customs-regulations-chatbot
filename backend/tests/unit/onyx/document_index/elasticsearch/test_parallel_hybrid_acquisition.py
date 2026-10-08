from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.context.search.retrieval.parallel_retrieval_scope import (
    experimental_parallel_retrieval,
    parallel_retrieval_enabled,
)
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


def body(normalizer: str) -> dict[str, Any]:
    return {
        "_onyx_hybrid_fusion": {
            "subqueries": [
                {"knn": {"field": "content_vector", "query_vector": [0.1, 0.2]}},
                {"match": {"content": "İĞŞ 東京 operative condition"}},
            ],
            "weights": [0.3, 0.7],
            "filters": [
                {"terms": {"access_control_list": ["authorized-user"]}},
                {"range": {"validity_start_date": {"lte": "2026-01-02"}}},
                {"term": {"document_set": "captured-corpus"}},
            ],
            "rank_window_size": 9,
            "normalizer": normalizer,
        },
        "timeout": "50s",
        "_source": {"excludes": ["content_vector", "title_vector"]},
        "highlight": {"fields": {"content": {}}},
        "explain": True,
        "size": 4,
    }


def response(vector: bool, *, timed_out: bool = False) -> dict[str, Any]:
    return {
        "took": 13 if vector else 19,
        "timed_out": timed_out,
        "hits": {
            "hits": (
                [
                    {
                        "_id": "shared",
                        "_score": 8.0,
                        "highlight": {"content": ["vector"]},
                    },
                    {"_id": "vector-only", "_score": 8.0, "_source": {"content": "v"}},
                    {"_id": "low", "_score": 0.0, "_source": {"content": "low"}},
                ]
                if vector
                else [
                    {
                        "_id": "shared",
                        "_score": 8.0,
                        "highlight": {"content": ["lexical"]},
                    },
                    {"_id": "lexical-only", "_score": 8.0, "_source": {"content": "l"}},
                    {"_id": "low", "_score": 0.0, "_source": {"content": "low"}},
                ]
            )
        },
    }


def client() -> tuple[ElasticsearchIndexClient, MagicMock]:
    selected = ElasticsearchIndexClient.__new__(ElasticsearchIndexClient)
    selected._index_name = "captured-physical-index"
    physical = MagicMock()
    selected._client = physical
    return selected, physical


@pytest.mark.parametrize("normalizer", ["minmax", "zscore"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_two_requests_overlap_and_exact_ordered_fusion_matches_legacy(
    normalizer: str, timed_out: bool
) -> None:
    selected, physical = client()
    selected_body = body(normalizer)
    unchanged_body = copy.deepcopy(selected_body)
    physical.search.side_effect = lambda **kwargs: response(
        "knn" in kwargs, timed_out=timed_out and "knn" not in kwargs
    )
    baseline = selected._search_hybrid_fusion(selected_body)
    baseline_requests = physical.search.call_args_list
    assert ["knn" in call.kwargs for call in baseline_requests] == [True, False]

    rendezvous = Barrier(2, timeout=5)
    observed_tenants: list[str | None] = []

    def acquire(**kwargs: Any) -> dict[str, Any]:
        assert parallel_retrieval_enabled()
        observed_tenants.append(CURRENT_TENANT_ID_CONTEXTVAR.get())
        rendezvous.wait()
        return response("knn" in kwargs, timed_out=timed_out and "knn" not in kwargs)

    physical.reset_mock()
    physical.search.side_effect = acquire
    tenant = CURRENT_TENANT_ID_CONTEXTVAR.set("owned-test-tenant")
    try:
        with experimental_parallel_retrieval(check_active=lambda: None):
            actual = selected._search_hybrid_fusion(selected_body)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(tenant)

    assert json.dumps(actual, ensure_ascii=False) == json.dumps(
        baseline, ensure_ascii=False
    )
    assert actual["hits"]["hits"][0]["highlight"] == {"content": ["lexical"]}
    assert actual["took"] == 32
    assert actual["timed_out"] is timed_out
    assert observed_tenants == ["owned-test-tenant", "owned-test-tenant"]
    assert selected_body == unchanged_body
    assert physical.search.call_count == 2
    actual_requests = sorted(
        (call.kwargs for call in physical.search.call_args_list),
        key=lambda request: "knn" not in request,
    )
    assert actual_requests == [call.kwargs for call in baseline_requests]
    assert not parallel_retrieval_enabled()


def test_all_requests_join_and_first_declared_error_wins() -> None:
    selected, physical = client()
    rendezvous = Barrier(2, timeout=5)
    lexical_finished = Event()
    vector_error = RuntimeError("vector failure")
    lexical_error = ValueError("lexical failure")

    def acquire(**kwargs: Any) -> dict[str, Any]:
        rendezvous.wait()
        if "knn" in kwargs:
            assert lexical_finished.wait(5)
            raise vector_error
        lexical_finished.set()
        raise lexical_error

    physical.search.side_effect = acquire
    with pytest.raises(RuntimeError) as raised:
        with experimental_parallel_retrieval(check_active=lambda: None):
            selected._search_hybrid_fusion(body("minmax"))
    assert raised.value is vector_error
    assert lexical_finished.is_set()
    assert physical.search.call_count == 2
    assert not parallel_retrieval_enabled()


def test_cancelled_scope_does_not_start_provider_and_resets_binding() -> None:
    selected, physical = client()
    failure = RuntimeError("cancelled by owner")

    def check() -> None:
        raise failure

    with pytest.raises(RuntimeError) as raised:
        with experimental_parallel_retrieval(check_active=check):
            selected._search_hybrid_fusion(body("minmax"))
    assert raised.value is failure
    physical.search.assert_not_called()
    assert not parallel_retrieval_enabled()


def test_nested_binding_restores_the_outer_scope() -> None:
    calls: list[str] = []
    with experimental_parallel_retrieval(check_active=lambda: calls.append("outer")):
        with experimental_parallel_retrieval(
            check_active=lambda: calls.append("inner")
        ):
            assert parallel_retrieval_enabled()
        assert parallel_retrieval_enabled()
    assert calls == ["outer", "inner"]
    assert not parallel_retrieval_enabled()


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("native_response", [False, True])
def test_transport_timing_is_scoped_and_preserves_raw_receipts(
    monkeypatch: pytest.MonkeyPatch, parallel: bool, native_response: bool
) -> None:
    @dataclass
    class Step:
        operation: str
        input_value: Any
        output_value: Any = None
        summary: str | None = None
        node_id: str | None = None

    captured: list[Step] = []

    @contextmanager
    def capture(operation: str, input_value: Any) -> Iterator[Step]:
        step = Step(operation, input_value)
        captured.append(step)
        yield step

    monkeypatch.setattr("onyx.tracing.answer_graph.graph_step", capture)
    monkeypatch.setattr(
        "onyx.document_index.elasticsearch.client.time.perf_counter", lambda: 10.0
    )
    selected, physical = client()
    selected_body = body("minmax")
    original_body = copy.deepcopy(selected_body)

    def acquire(**kwargs: Any) -> Any:
        raw = response("knn" in kwargs)
        raw["additional_provider_field"] = {"opaque": "preserved"}
        return (
            SimpleNamespace(body=raw, meta=SimpleNamespace(duration=0.125))
            if native_response
            else raw
        )

    physical.search.side_effect = acquire
    with (
        experimental_parallel_retrieval(check_active=lambda: None)
        if parallel
        else nullcontext()
    ):
        result = selected._search_hybrid_fusion(selected_body)

    assert result["took"] == 32
    assert selected_body == original_body
    queries = [step for step in captured if step.operation != "search.fusion"]
    assert len(queries) == 2
    for step in queries:
        assert step.output_value["additional_provider_field"] == {"opaque": "preserved"}
        assert len(step.output_value["hits"]["hits"]) == 3
        assert step.input_value["body"]["size"] == 9
        assert step.input_value["body"]["_source"] == selected_body["_source"]
        if not parallel:
            assert step.summary is None
            continue
        assert step.summary is not None and len(step.summary) <= 160
        assert "search_call_seconds=0" in step.summary
        assert "server_took_ms=" in step.summary
        assert ("native_http_seconds=0.125" in step.summary) is native_response
        assert "authorized-user" not in step.summary
        assert "operative condition" not in step.summary


@pytest.mark.parametrize("duration", [None, True, -1, float("nan"), float("inf")])
def test_unavailable_native_duration_does_not_fail_parallel_search(
    monkeypatch: pytest.MonkeyPatch, duration: Any
) -> None:
    captured: list[Any] = []

    @contextmanager
    def capture(operation: str, _input_value: Any) -> Iterator[Any]:
        step = SimpleNamespace(
            operation=operation, node_id=None, summary=None, output_value=None
        )
        captured.append(step)
        yield step

    monkeypatch.setattr("onyx.tracing.answer_graph.graph_step", capture)
    selected, physical = client()
    physical.search.side_effect = lambda **kwargs: SimpleNamespace(
        body=response("knn" in kwargs), meta=SimpleNamespace(duration=duration)
    )
    with experimental_parallel_retrieval(check_active=lambda: None):
        result = selected._search_hybrid_fusion(body("minmax"))
    assert result["took"] == 32
    for step in captured:
        if step.operation != "search.fusion":
            assert step.summary is not None
            assert "native_http_seconds" not in step.summary
