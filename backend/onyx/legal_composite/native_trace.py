"""Compact native-search capture on borrowed, Legal Composite-only index facades."""

from __future__ import annotations

import copy
import datetime
import hashlib
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from elasticsearch import Elasticsearch

from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient, SearchHit
from onyx.document_index.elasticsearch.constants import ElasticsearchSearchType
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
    ElasticsearchIndexPair,
)
from onyx.document_index.elasticsearch.schema import DocumentChunkWithoutVectors
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.framework.scope import Scope
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace
from onyx.utils.logger import setup_logger

_CAPTURE_MODE = "legal_composite_compact_native_v1"
logger = setup_logger()


@dataclass(frozen=True)
class _NativeCapture:
    trace: Trace | None
    as_of_date: datetime.date | None
    publication_index: PublicationIndexSnapshot | None
    span: Span[Any] | None = None


_NATIVE_CAPTURE: ContextVar[_NativeCapture | None] = ContextVar(
    "legal_composite_native_capture", default=None
)


def _candidate_receipts(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    hits = response.get("hits", {}).get("hits", [])
    receipts: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        content = source.get("content")
        receipts.append(
            {
                "candidate_id": hit.get("_id"),
                "document_id": source.get("document_id"),
                "regulatory_chunk_id": source.get("regulatory_chunk_id"),
                "chunk_index": source.get("chunk_index"),
                "score": hit.get("_score"),
                "index_content_sha256": hashlib.sha256(content.encode()).hexdigest()
                if isinstance(content, str)
                else None,
            }
        )
    return receipts


class _NativeResponseRecorder:
    def __init__(self, delegate: Elasticsearch) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def search(self, *args: Any, **kwargs: Any) -> Any:
        binding = _NATIVE_CAPTURE.get()
        started = time.perf_counter()
        # SDK execution remains in the suppressed native scope, including SDK hooks.
        try:
            response = self._delegate.search(*args, **kwargs)
        except Exception as error:
            self._capture(
                binding, args, kwargs, started, error_type=type(error).__name__
            )
            raise
        self._capture(binding, args, kwargs, started, response=response)
        return response

    def _capture(
        self,
        binding: _NativeCapture | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        started: float,
        *,
        response: Any = None,
        error_type: str | None = None,
    ) -> None:
        seconds = time.perf_counter() - started
        if binding is None or binding.trace is None:
            return
        operation = "search.vector" if "knn" in kwargs else "search.bm25"
        token = Scope.set_current_trace(binding.trace)
        span_token = Scope.set_current_span(binding.span)
        try:
            body = (
                response
                if isinstance(response, dict)
                else getattr(response, "body", {})
            )
            with graph_step(
                operation,
                {
                    "capture_mode": _CAPTURE_MODE,
                    "sdk_arguments": args,
                    "sdk_keyword_arguments": kwargs,
                    "as_of_date": binding.as_of_date.isoformat()
                    if binding.as_of_date is not None
                    else None,
                    "publication_index": binding.publication_index,
                },
            ) as step:
                step.summary = (
                    f"search_call_seconds={seconds:.6g}; "
                    f"native_http_seconds={getattr(getattr(response, 'meta', None), 'duration', None)}; "
                    f"server_took_ms={body.get('took')}; compact_native"
                )[:160]
                step.output_value = {
                    "capture_mode": _CAPTURE_MODE,
                    "native_call_seconds": seconds,
                    "native_http_seconds": getattr(
                        getattr(response, "meta", None), "duration", None
                    ),
                    "server_took_ms": body.get("took"),
                    "timed_out": body.get("timed_out"),
                    "candidate_receipts": _candidate_receipts(body),
                    "canonical_evidence": False,
                    "native_status": "failed" if error_type else "complete",
                    "native_error_type": error_type,
                }
        except Exception as capture_error:
            # Capture failure cannot change an SDK response or mask its exception.
            logger.warning(
                "Legal Composite compact native capture failed (%s)",
                type(capture_error).__name__,
            )
        finally:
            Scope.reset_current_span(span_token)
            Scope.reset_current_trace(token)


class _BorrowedIndexClient(ElasticsearchIndexClient):
    def __init__(self, delegate: ElasticsearchIndexClient) -> None:
        self._delegate = delegate
        self._client = cast(
            Elasticsearch, _NativeResponseRecorder(delegate.publication_client())
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def close(self) -> None:
        # The caller retains ownership of the original client and its transport.
        return None

    def __del__(self) -> None:
        return None

    def search(
        self,
        body: dict[str, Any],
        normalization_method: str | None,
        search_type: ElasticsearchSearchType = ElasticsearchSearchType.UNKNOWN,
        as_of_date: datetime.date | None = None,
        publication_index: PublicationIndexSnapshot | None = None,
    ) -> list[SearchHit[DocumentChunkWithoutVectors]]:
        binding = _NativeCapture(
            trace=Scope.get_current_trace(),
            as_of_date=as_of_date,
            publication_index=publication_index,
            span=Scope.get_current_span(),
        )
        capture_token = _NATIVE_CAPTURE.set(binding)
        trace_token = Scope.set_current_trace(None)
        span_token = Scope.set_current_span(None)
        try:
            result = super().search(
                body=body,
                normalization_method=normalization_method,
                search_type=search_type,
                as_of_date=as_of_date,
                publication_index=publication_index,
            )
        finally:
            Scope.reset_current_span(span_token)
            Scope.reset_current_trace(trace_token)
            _NATIVE_CAPTURE.reset(capture_token)
        if binding.trace is not None:
            try:
                with graph_step(
                    "search.fusion"
                    if "_onyx_hybrid_fusion" in body
                    else "search.native_result",
                    {
                        "capture_mode": _CAPTURE_MODE,
                        "body": body,
                        "normalization_method": normalization_method,
                        "search_type": search_type,
                        "as_of_date": as_of_date.isoformat() if as_of_date else None,
                        "publication_index": publication_index,
                    },
                ) as step:
                    step.output_value = {
                        "capture_mode": _CAPTURE_MODE,
                        "candidate_receipts": [
                            {
                                "document_id": hit.document_chunk.document_id,
                                "regulatory_chunk_id": hit.document_chunk.regulatory_chunk_id,
                                "chunk_index": hit.document_chunk.chunk_index,
                                "score": hit.score,
                                "index_content_sha256": hashlib.sha256(
                                    hit.document_chunk.content.encode()
                                ).hexdigest(),
                                "publication_observation": hit.document_chunk.publication_observation,
                                "publication_index": hit.document_chunk.publication_index,
                            }
                            for hit in result
                        ],
                        "canonical_evidence": False,
                    }
            except Exception as capture_error:
                logger.warning(
                    "Legal Composite compact result capture failed (%s)",
                    type(capture_error).__name__,
                )
        return result


def compact_native_index(index: DocumentIndex) -> DocumentIndex:
    """Install capture facades on owned index copies without modifying shared resources."""
    if isinstance(index, ElasticsearchIndexPair):
        owned_pair = copy.copy(index)
        owned_pair._primary = cast(
            ElasticsearchDocumentIndex, compact_native_index(index._primary)
        )
        return owned_pair
    if isinstance(index, ElasticsearchDocumentIndex):
        if isinstance(index._client, _BorrowedIndexClient):
            return index
        owned_index = copy.copy(index)
        owned_index._client = _BorrowedIndexClient(index._client)
        return owned_index
    return index
