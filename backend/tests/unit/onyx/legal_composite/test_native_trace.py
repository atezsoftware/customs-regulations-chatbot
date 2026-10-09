"""Native capture preserves queries, candidates, scopes and borrowed resource ownership."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from elasticsearch import Elasticsearch

from onyx.context.search.retrieval.parallel_retrieval_scope import (
    experimental_parallel_retrieval,
)
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
    ElasticsearchIndexPair,
)
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationScope,
    ReadObservation,
)
from onyx.legal_composite import native_trace
from onyx.tracing.framework.scope import Scope
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace


class CaptureStep:
    def __init__(
        self, operation: str, value: Any, records: list[dict[str, Any]]
    ) -> None:
        self.operation = operation
        self.input_value = value
        self.output_value: Any = None
        self.summary: str | None = None
        self.node_id = None
        self.records = records
        self.trace = Scope.get_current_trace()
        self.span = Scope.get_current_span()

    def __enter__(self) -> CaptureStep:
        return self

    def __exit__(self, *args: Any) -> None:
        if self.trace is not None:
            self.records.append(
                {
                    "trace": self.trace,
                    "span": self.span,
                    "operation": self.operation,
                    "input": self.input_value,
                    "output": self.output_value,
                }
            )


def responses() -> list[dict[str, Any]]:
    def hit(identity: str, score: float) -> dict[str, Any]:
        return {
            "_id": identity,
            "_score": score,
            "_source": {
                "document_id": f"doc-{identity}",
                "regulatory_chunk_id": f"chunk-{identity}",
                "chunk_index": 0,
                "content": f"Synthetic indexed passage {identity}",
            },
        }

    return [
        {"took": 3, "hits": {"hits": [hit("a", 3.0), hit("b", 1.0)]}},
        {"took": 4, "hits": {"hits": [hit("b", 4.0), hit("c", 2.0)]}},
    ]


def fusion_body(normalizer: str) -> dict[str, Any]:
    return {
        "_onyx_hybrid_fusion": {
            "subqueries": [
                {"knn": {"field": "content_vector", "query_vector": [0.1]}},
                {"match": {"content": "synthetic query"}},
            ],
            "weights": [0.5, 0.5],
            "filters": [
                {"term": {"hidden": False}},
                {"terms": {"access_control_list": ["owned-user"]}},
                {"terms": {"document_id": ["doc-a", "doc-b", "doc-c"]}},
            ],
            "rank_window_size": 2,
            "normalizer": normalizer,
        },
        "timeout": "50s",
        "_source": {"excludes": ["content_vector"]},
        "size": 3,
    }


def client(sdk: Any) -> ElasticsearchIndexClient:
    instance = ElasticsearchIndexClient.__new__(ElasticsearchIndexClient)
    instance._index_name = "owned-index"
    instance._client = sdk
    instance._emit_metrics = False
    return instance


@pytest.mark.parametrize("normalizer", ["minmax", "zscore"])
def test_compact_native_fusion_is_exact_and_retains_every_raw_candidate(
    normalizer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = MagicMock()
    sdk.search.side_effect = copy.deepcopy(responses())
    original = client(sdk)
    body = fusion_body(normalizer)
    baseline = original._search_hybrid_fusion(copy.deepcopy(body))
    baseline_calls = sdk.search.call_args_list
    sdk.search.reset_mock()
    sdk.search.side_effect = copy.deepcopy(responses())
    owner_trace = cast(Trace, object())
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(
        native_trace,
        "graph_step",
        lambda operation, value: CaptureStep(operation, value, records),
    )
    borrowed = native_trace._BorrowedIndexClient(original)
    capture = native_trace._NATIVE_CAPTURE.set(
        native_trace._NativeCapture(owner_trace, datetime.date(2030, 1, 1), None)
    )
    trace = Scope.set_current_trace(None)
    try:
        actual = borrowed._search_hybrid_fusion(copy.deepcopy(body))
        assert Scope.get_current_trace() is None
    finally:
        Scope.reset_current_trace(trace)
        native_trace._NATIVE_CAPTURE.reset(capture)
    assert actual == baseline
    assert sdk.search.call_args_list == baseline_calls
    assert len(records) == 2
    assert [record["operation"] for record in records] == [
        "search.vector",
        "search.bm25",
    ]
    receipts = [
        receipt
        for record in records
        for receipt in record["output"]["candidate_receipts"]
    ]
    assert [receipt["candidate_id"] for receipt in receipts] == ["a", "b", "b", "c"]
    assert {receipt["document_id"] for receipt in receipts} == {
        "doc-a",
        "doc-b",
        "doc-c",
    }
    assert (
        receipts[0]["index_content_sha256"]
        == hashlib.sha256(b"Synthetic indexed passage a").hexdigest()
    )
    assert all("content" not in receipt for receipt in receipts)
    assert records[0]["input"]["sdk_keyword_arguments"] == baseline_calls[0].kwargs
    assert records[0]["input"]["as_of_date"] == "2030-01-01"


@pytest.mark.parametrize("response_kind", ["dict", "sdk"])
def test_sdk_call_is_suppressed_and_response_identity_is_preserved(
    response_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = MagicMock()
    body = responses()[0]
    raw: Any = (
        body
        if response_kind == "dict"
        else SimpleNamespace(body=body, meta=SimpleNamespace(duration=0.123))
    )
    scope_marker: ContextVar[str] = ContextVar("synthetic_fence", default="unset")
    owner_trace = cast(Trace, object())
    records: list[dict[str, Any]] = []

    def execute(**_kwargs: Any) -> Any:
        assert Scope.get_current_trace() is None
        assert scope_marker.get() == "fenced"
        return raw

    sdk.search.side_effect = execute
    monkeypatch.setattr(
        native_trace, "graph_step", lambda op, value: CaptureStep(op, value, records)
    )
    recorder = native_trace._NativeResponseRecorder(cast(Elasticsearch, sdk))
    binding = native_trace._NATIVE_CAPTURE.set(
        native_trace._NativeCapture(owner_trace, None, None)
    )
    trace = Scope.set_current_trace(None)
    fence = scope_marker.set("fenced")
    try:
        kwargs = {"index": "owned-index", "query": {"match": {"content": "synthetic"}}}
        assert recorder.search(**kwargs) is raw
        sdk.search.assert_called_once_with(**kwargs)
        assert Scope.get_current_trace() is None
        assert scope_marker.get() == "fenced"
    finally:
        scope_marker.reset(fence)
        Scope.reset_current_trace(trace)
        native_trace._NATIVE_CAPTURE.reset(binding)
    assert records[0]["trace"] is owner_trace
    assert records[0]["output"]["native_http_seconds"] == (
        None if response_kind == "dict" else 0.123
    )


def test_index_facade_preserves_exact_arguments_and_restores_scope_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = MagicMock()
    delegate = client(sdk)
    borrowed = native_trace._BorrowedIndexClient(delegate)
    trace = cast(Trace, object())
    span = cast(Span[Any], object())
    error = RuntimeError("synthetic native failure")
    body = fusion_body("minmax")

    def native_search(self: ElasticsearchIndexClient, **kwargs: Any) -> Any:
        assert self is borrowed
        assert kwargs["body"] is body
        assert kwargs["as_of_date"] == datetime.date(2030, 1, 1)
        assert Scope.get_current_trace() is None
        assert Scope.get_current_span() is None
        binding = native_trace._NATIVE_CAPTURE.get()
        assert binding is not None
        assert binding.trace is trace
        assert binding.span is span
        raise error

    monkeypatch.setattr(ElasticsearchIndexClient, "search", native_search)
    token = Scope.set_current_trace(trace)
    span_token = Scope.set_current_span(span)
    try:
        with pytest.raises(RuntimeError) as raised:
            borrowed.search(
                body=body,
                normalization_method="minmax",
                as_of_date=datetime.date(2030, 1, 1),
            )
        assert raised.value is error
        assert Scope.get_current_trace() is trace
        assert Scope.get_current_span() is span
        assert native_trace._NATIVE_CAPTURE.get() is None
    finally:
        Scope.reset_current_span(span_token)
        Scope.reset_current_trace(token)


def test_facades_do_not_mutate_or_close_primary_secondary_or_borrowed_transport() -> (
    None
):
    sdk = MagicMock()
    original_client = client(sdk)
    original = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    original._client = original_client
    original._index_name = "owned-index"
    secondary = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    secondary._client = client(MagicMock())
    pair = ElasticsearchIndexPair.__new__(ElasticsearchIndexPair)
    pair._primary = original
    pair._secondary = secondary
    wrapped = native_trace.compact_native_index(pair)
    assert isinstance(wrapped, ElasticsearchIndexPair)
    assert wrapped is not pair
    assert wrapped._primary is not original
    assert wrapped._secondary is secondary
    assert original._client is original_client
    assert pair._primary is original
    assert isinstance(wrapped._primary._client, native_trace._BorrowedIndexClient)
    borrowed = wrapped._primary._client
    borrowed.close()
    borrowed.__del__()
    borrowed.__exit__(None, None, None)
    sdk.close.assert_not_called()


def test_concurrent_ordinary_trace_keeps_full_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onyx.tracing.answer_graph as graph

    records: list[dict[str, Any]] = []
    lock = Lock()
    barrier = Barrier(2)
    ordinary_trace = cast(Trace, SimpleNamespace(trace_id="ordinary"))
    composite_trace = cast(Trace, SimpleNamespace(trace_id="composite"))

    def step(operation: str, value: Any = None, **_kwargs: Any) -> CaptureStep:
        with lock:
            return CaptureStep(operation, value, records)

    monkeypatch.setattr(graph, "graph_step", step)
    monkeypatch.setattr(native_trace, "graph_step", step)

    def run(composite: bool) -> Any:
        sdk = MagicMock()
        incoming = iter(copy.deepcopy(responses()))

        def execute(**kwargs: Any) -> Any:
            if "knn" in kwargs:
                barrier.wait(timeout=5)
            assert (Scope.get_current_trace() is None) == composite
            return next(incoming)

        sdk.search.side_effect = execute
        original = client(sdk)
        token = Scope.set_current_trace(
            composite_trace if composite else ordinary_trace
        )
        capture = native_trace._NATIVE_CAPTURE.set(
            native_trace._NativeCapture(composite_trace, None, None)
            if composite
            else None
        )
        suppressed = Scope.set_current_trace(None) if composite else None
        try:
            selected = (
                native_trace._BorrowedIndexClient(original) if composite else original
            )
            return selected._search_hybrid_fusion(fusion_body("minmax"))
        finally:
            if suppressed is not None:
                Scope.reset_current_trace(suppressed)
            native_trace._NATIVE_CAPTURE.reset(capture)
            Scope.reset_current_trace(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        ordinary = executor.submit(run, False)
        composite = executor.submit(run, True)
        assert ordinary.result() == composite.result()
    ordinary_nodes = [record for record in records if record["trace"] is ordinary_trace]
    composite_nodes = [
        record for record in records if record["trace"] is composite_trace
    ]
    assert len(ordinary_nodes) == 3
    assert ordinary_nodes[0]["output"]["hits"]["hits"][0]["_source"]["content"]
    assert len(composite_nodes) == 2
    assert all(
        record["output"]["capture_mode"] == native_trace._CAPTURE_MODE
        for record in composite_nodes
    )


@pytest.mark.parametrize("sdk_fails", [False, True])
def test_capture_failure_preserves_response_or_native_exception(
    sdk_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = MagicMock()
    response = responses()[0]
    error = RuntimeError("synthetic SDK failure")
    sdk.search.side_effect = error if sdk_fails else None
    sdk.search.return_value = response
    recorder = native_trace._NativeResponseRecorder(cast(Elasticsearch, sdk))

    def broken_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("synthetic capture failure")

    monkeypatch.setattr(native_trace, "graph_step", broken_capture)
    owner_trace = cast(Trace, object())
    capture = native_trace._NATIVE_CAPTURE.set(
        native_trace._NativeCapture(owner_trace, None, None)
    )
    token = Scope.set_current_trace(None)
    try:
        if sdk_fails:
            with pytest.raises(RuntimeError) as raised:
                recorder.search(index="owned-index", query={"match_all": {}})
            assert raised.value is error
        else:
            assert (
                recorder.search(index="owned-index", query={"match_all": {}})
                is response
            )
        assert Scope.get_current_trace() is None
    finally:
        Scope.reset_current_trace(token)
        native_trace._NATIVE_CAPTURE.reset(capture)


def test_failed_sdk_call_records_failure_without_exception_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = MagicMock()
    error = RuntimeError("synthetic private diagnostic")
    sdk.search.side_effect = error
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(
        native_trace, "graph_step", lambda op, value: CaptureStep(op, value, records)
    )
    capture = native_trace._NATIVE_CAPTURE.set(
        native_trace._NativeCapture(cast(Trace, object()), None, None)
    )
    token = Scope.set_current_trace(None)
    try:
        with pytest.raises(RuntimeError) as raised:
            native_trace._NativeResponseRecorder(cast(Elasticsearch, sdk)).search(
                index="owned-index", knn={"field": "content_vector"}
            )
        assert raised.value is error
        assert Scope.get_current_trace() is None
    finally:
        Scope.reset_current_trace(token)
        native_trace._NATIVE_CAPTURE.reset(capture)
    assert records[0]["output"]["native_status"] == "failed"
    assert records[0]["output"]["native_error_type"] == "RuntimeError"
    assert records[0]["output"]["candidate_receipts"] == []
    assert "synthetic private diagnostic" not in repr(records)


def test_result_capture_failure_keeps_exact_result_and_restores_parent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result: list[Any] = []
    owner_trace = cast(Trace, object())
    parent_span = cast(Span[Any], object())
    borrowed = native_trace._BorrowedIndexClient(client(MagicMock()))

    def execute(_self: Any, **_kwargs: Any) -> Any:
        assert Scope.get_current_trace() is None
        assert Scope.get_current_span() is None
        return result

    def broken_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("synthetic result capture failure")

    monkeypatch.setattr(ElasticsearchIndexClient, "search", execute)
    monkeypatch.setattr(native_trace, "graph_step", broken_capture)
    trace = Scope.set_current_trace(owner_trace)
    span = Scope.set_current_span(parent_span)
    try:
        assert borrowed.search({"query": {"match_all": {}}}, None) is result
        assert Scope.get_current_trace() is owner_trace
        assert Scope.get_current_span() is parent_span
        assert native_trace._NATIVE_CAPTURE.get() is None
    finally:
        Scope.reset_current_span(span)
        Scope.reset_current_trace(trace)


def test_parallel_native_lanes_keep_fence_and_disabled_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = MagicMock()
    originals = responses()
    scope_marker: ContextVar[str] = ContextVar(
        "synthetic_parallel_fence", default="unset"
    )
    barrier = Barrier(2)
    records: list[dict[str, Any]] = []
    owner_trace = cast(Trace, SimpleNamespace(trace_id="parallel-composite"))
    checks: list[str] = []

    def execute(**kwargs: Any) -> Any:
        assert Scope.get_current_trace() is None
        assert scope_marker.get() == "fenced"
        barrier.wait(timeout=5)
        return copy.deepcopy(originals[0 if "knn" in kwargs else 1])

    sdk.search.side_effect = execute
    monkeypatch.setattr(
        native_trace, "graph_step", lambda op, value: CaptureStep(op, value, records)
    )
    capture = native_trace._NATIVE_CAPTURE.set(
        native_trace._NativeCapture(owner_trace, None, None)
    )
    trace = Scope.set_current_trace(None)
    fence = scope_marker.set("fenced")
    try:
        with experimental_parallel_retrieval(
            check_active=lambda: checks.append("checked")
        ):
            result = native_trace._BorrowedIndexClient(
                client(sdk)
            )._search_hybrid_fusion(fusion_body("minmax"))
        assert [hit["_id"] for hit in result["hits"]["hits"]] == ["a", "b", "c"]
        assert Scope.get_current_trace() is None
        assert scope_marker.get() == "fenced"
    finally:
        scope_marker.reset(fence)
        Scope.reset_current_trace(trace)
        native_trace._NATIVE_CAPTURE.reset(capture)
    assert len(records) == 2
    assert all(record["trace"] is owner_trace for record in records)
    assert len(checks) >= 5


def test_real_search_preserves_temporal_projection_and_canonical_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onyx.db.regulatory_public_reads as public_reads
    import onyx.regulatory.publication_reads as publication_reads

    file_id = UUID("12345678-1234-5678-1234-567812345678")
    source = {
        "document_id": str(file_id),
        "chunk_index": 0,
        "content": "Synthetic canonical passage",
        "regulatory_chunk_id": "synthetic-chunk",
        "source_type": "user_file",
        "public": False,
        "access_control_list": ["owned-user"],
        "global_boost": 0,
        "semantic_identifier": "Synthetic title",
        "blurb": "Synthetic passage",
        "doc_summary": "",
        "chunk_context": "",
    }
    response = {
        "took": 1,
        "hits": {"hits": [{"_id": "owned-hit", "_score": 2.0, "_source": source}]},
    }
    snapshot = PublicationIndexSnapshot(
        index_name="owned-index",
        index_uuid="owned-index-uuid",
        search_settings_id=9,
        model_provider="synthetic",
        model_name="synthetic-encoder",
        vector_dimension=1,
        embedding_config_sha256="a" * 64,
        multitenant=False,
    )
    observation = ReadObservation(
        scope=PublicationScope(
            tenant_id="owned-tenant",
            environment="test",
            database_identity="synthetic-db",
        ),
        committed_epoch=12,
    )
    temporal_calls: list[dict[str, Any]] = []
    filter_observations: list[ReadObservation] = []

    def filter_read(
        observed: ReadObservation, hits: Any, *_args: Any, **_kwargs: Any
    ) -> Any:
        filter_observations.append(observed)
        return hits

    def temporal_bindings(files: tuple[UUID, ...], **kwargs: Any) -> Any:
        assert files == (file_id,)
        temporal_calls.append(kwargs)
        return {
            file_id: [
                SimpleNamespace(
                    projection=SimpleNamespace(
                        ordinal=0, source_json=json.dumps(source)
                    ),
                    semantic_position=9,
                )
            ]
        }

    monkeypatch.setattr(
        publication_reads, "observe_publication_read", lambda: observation
    )
    monkeypatch.setattr(publication_reads, "filter_publication_read", filter_read)
    monkeypatch.setattr(
        public_reads, "current_qualified_file_ids", lambda _files: frozenset({file_id})
    )
    monkeypatch.setattr(public_reads, "query_temporal_bindings", temporal_bindings)
    monkeypatch.setattr(
        public_reads,
        "current_canonical_positions",
        lambda _ids: {"synthetic-chunk": 19},
    )
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(
        native_trace, "graph_step", lambda op, value: CaptureStep(op, value, records)
    )
    sdk = MagicMock()
    parent_span = cast(Span[Any], SimpleNamespace(span_id="owned-parent-span"))

    def execute(**_kwargs: Any) -> Any:
        assert Scope.get_current_trace() is None
        assert Scope.get_current_span() is None
        return response

    sdk.search.side_effect = execute
    sdk.indices.get.return_value = {
        "owned-index": {"settings": {"index": {"uuid": "owned-index-uuid"}}}
    }
    original = client(sdk)
    body = {"query": {"match": {"content": "synthetic"}}, "size": 100}
    as_of_date = datetime.date(2030, 1, 1)
    baseline = original.search(
        body, None, as_of_date=as_of_date, publication_index=snapshot
    )
    baseline_calls = sdk.search.call_args_list
    sdk.search.reset_mock()
    token = Scope.set_current_trace(
        cast(Trace, SimpleNamespace(trace_id="fenced-composite"))
    )
    span_token = Scope.set_current_span(parent_span)
    try:
        actual = native_trace._BorrowedIndexClient(original).search(
            body, None, as_of_date=as_of_date, publication_index=snapshot
        )
        assert native_trace._NATIVE_CAPTURE.get() is None
        assert Scope.get_current_span() is parent_span
    finally:
        Scope.reset_current_span(span_token)
        Scope.reset_current_trace(token)
    assert actual == baseline
    assert actual[0].document_chunk.content == source["content"]
    assert actual[0].document_chunk.publication_index is snapshot
    assert actual[0].document_chunk.publication_observation is observation
    assert actual[0].document_chunk.semantic_position == 9
    assert sdk.search.call_args_list == baseline_calls
    assert len(temporal_calls) == 2
    assert all(
        call["index"] is snapshot and call["as_of_date"] == as_of_date
        for call in temporal_calls
    )
    assert all(item is observation for item in filter_observations)
    assert (
        records[-1]["output"]["candidate_receipts"][0]["publication_observation"]
        is observation
    )
    assert records[-1]["input"]["publication_index"] is snapshot
    assert all(record["span"] is parent_span for record in records)
