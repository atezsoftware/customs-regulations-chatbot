from contextlib import nullcontext
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import pytest

from onyx.indexing.indexing_pipeline import (
    ContextualEnrichmentError,
    add_chunk_summaries,
    add_contextual_summaries,
    index_doc_batch,
)
from onyx.indexing.models import DocAwareChunk
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LLMRateLimitError, LLMTimeoutError


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        id="contextual-test",
        created="2026-01-01T00:00:00Z",
        choice=Choice(message=Message(content=content)),
    )


def _chunk(chunk_id: int = 0) -> DocAwareChunk:
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.content = f"chunk-{chunk_id}"
    chunk.chunk_context = ""
    chunk.doc_summary = ""
    chunk.contextual_rag_reserved_tokens = 128
    chunk.source_document.id = "document-1"
    chunk.source_document.get_text_content.return_value = "document text"
    return cast(DocAwareChunk, chunk)


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2]
    tokenizer.decode.return_value = "document text"
    return tokenizer


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.config.max_input_tokens = 16_000
    llm.config.model_name = "contextual-model"
    llm.config.model_provider = "provider"
    return llm


def _common_patches() -> tuple[Any, Any]:
    return (
        patch(
            "onyx.llm.prompt_cache.processor.process_with_prompt_cache",
            return_value=(UserMessage(content="processed prompt"), None),
        ),
        patch(
            "onyx.indexing.indexing_pipeline.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
    )


def test_chunk_context_retries_transient_failures_with_bounded_backoff() -> None:
    chunk = _chunk()
    llm = _llm()
    llm.invoke.side_effect = [
        LLMTimeoutError("timeout"),
        LLMRateLimitError("rate limited"),
        _response("generated context"),
    ]
    prompt_cache_patch, span_patch = _common_patches()

    with (
        prompt_cache_patch,
        span_patch,
        patch("onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_MAX_RETRIES", 2),
        patch(
            "onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_LLM_TIMEOUT_SECONDS",
            17,
        ),
        patch(
            "onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_RETRY_BASE_SECONDS",
            0.25,
        ),
        patch(
            "onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_RETRY_MAX_SECONDS",
            1.0,
        ),
        patch("onyx.indexing.indexing_pipeline.time.sleep") as sleep,
    ):
        add_chunk_summaries(
            [chunk],
            llm,
            _tokenizer(),
            trunc_doc_chunk_tokens=100,
            doc_tokens=[1, 2],
            raise_on_failure=True,
        )

    assert chunk.chunk_context == "generated context"
    assert llm.invoke.call_count == 3
    assert all(
        invoke_call.kwargs["timeout_override"] == 17
        and invoke_call.kwargs["use_streaming"] is False
        for invoke_call in llm.invoke.call_args_list
    )
    assert sleep.call_args_list == [call(0.25), call(0.5)]


def test_strict_chunk_context_failure_propagates_after_retry_budget() -> None:
    chunk = _chunk()
    llm = _llm()
    llm.invoke.side_effect = LLMTimeoutError("timeout")
    prompt_cache_patch, span_patch = _common_patches()

    with (
        prompt_cache_patch,
        span_patch,
        patch("onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_MAX_RETRIES", 1),
        patch("onyx.indexing.indexing_pipeline.time.sleep"),
        pytest.raises(ContextualEnrichmentError, match="chunk enrichment failed"),
    ):
        add_chunk_summaries(
            [chunk],
            llm,
            _tokenizer(),
            trunc_doc_chunk_tokens=100,
            doc_tokens=[1, 2],
            raise_on_failure=True,
        )

    assert llm.invoke.call_count == 2
    assert chunk.chunk_context == ""


def test_best_effort_chunk_context_preserves_existing_soft_failure_semantics() -> None:
    chunk = _chunk()
    llm = _llm()
    llm.invoke.side_effect = LLMRateLimitError("rate limited")
    prompt_cache_patch, span_patch = _common_patches()

    with (
        prompt_cache_patch,
        span_patch,
        patch("onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_MAX_RETRIES", 0),
    ):
        add_chunk_summaries(
            [chunk],
            llm,
            _tokenizer(),
            trunc_doc_chunk_tokens=100,
            doc_tokens=[1, 2],
        )

    assert llm.invoke.call_count == 1
    assert chunk.chunk_context == ""


def test_chunk_context_uses_configured_worker_bound() -> None:
    chunks = [_chunk(chunk_id) for chunk_id in range(20)]
    llm = _llm()

    with (
        patch("onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_MAX_WORKERS", 3),
        patch(
            "onyx.indexing.indexing_pipeline.run_functions_tuples_in_parallel"
        ) as run_parallel,
    ):
        add_chunk_summaries(
            chunks,
            llm,
            _tokenizer(),
            trunc_doc_chunk_tokens=100,
            doc_tokens=[1, 2],
        )

    assert len(run_parallel.call_args.kwargs["functions_with_args"]) == 20
    assert run_parallel.call_args.kwargs["max_workers"] == 3


def test_contextual_enrichment_filters_budget_ineligible_chunks_per_document() -> None:
    eligible = _chunk(1)
    ineligible = _chunk(2)
    ineligible.contextual_rag_reserved_tokens = 0

    with (
        patch(
            "onyx.indexing.indexing_pipeline._contextual_prompt_content_budget",
            return_value=100,
        ),
        patch(
            "onyx.indexing.indexing_pipeline.add_document_summaries",
            return_value=[1, 2],
        ) as add_document,
        patch("onyx.indexing.indexing_pipeline.add_chunk_summaries") as add_chunks,
        patch("onyx.indexing.indexing_pipeline.USE_DOCUMENT_SUMMARY", True),
        patch("onyx.indexing.indexing_pipeline.USE_CHUNK_SUMMARY", True),
    ):
        result = add_contextual_summaries(
            [ineligible, eligible],
            _llm(),
            _tokenizer(),
            chunk_token_limit=512,
        )

    assert result == [ineligible, eligible]
    assert add_document.call_args.args[0] == [eligible]
    assert add_chunks.call_args.args[0] == [eligible]


def test_contextual_call_uses_process_wide_request_gate() -> None:
    chunk = _chunk()
    llm = _llm()
    llm.invoke.return_value = _response("generated context")
    request_slots = MagicMock()
    prompt_cache_patch, span_patch = _common_patches()

    with (
        prompt_cache_patch,
        span_patch,
        patch(
            "onyx.indexing.indexing_pipeline._CONTEXTUAL_RAG_REQUEST_SLOTS",
            request_slots,
        ),
    ):
        add_chunk_summaries(
            [chunk],
            llm,
            _tokenizer(),
            trunc_doc_chunk_tokens=100,
            doc_tokens=[1, 2],
            raise_on_failure=True,
        )

    request_slots.__enter__.assert_called_once_with()
    request_slots.__exit__.assert_called_once()


def test_generic_context_failure_aborts_before_embedding() -> None:
    document = MagicMock()
    document.id = "ordinary-document"
    document.sections = []
    document.get_total_char_length.return_value = 100

    chunk = _chunk()
    chunk.regulatory_chunk_id = None
    adapter = MagicMock()
    context = MagicMock()
    context.updatable_docs = [document]
    adapter.prepare.return_value = context
    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]
    chunker.chunk_token_limit = 512
    llm = _llm()

    with (
        patch(
            "onyx.indexing.indexing_pipeline.process_image_sections",
            return_value=[document],
        ),
        patch(
            "onyx.indexing.indexing_pipeline._apply_document_ingestion_hook",
            return_value=[document],
        ),
        patch(
            "onyx.indexing.indexing_pipeline.get_tokenizer", return_value=_tokenizer()
        ),
        patch(
            "onyx.indexing.indexing_pipeline.add_contextual_summaries",
            side_effect=ContextualEnrichmentError("ordinary contextual failure"),
        ) as enrich,
        patch("onyx.indexing.indexing_pipeline.embed_and_stream") as embed,
        pytest.raises(ContextualEnrichmentError, match="ordinary contextual failure"),
    ):
        index_doc_batch(
            document_batch=[document],
            chunker=chunker,
            embedder=MagicMock(),
            document_indices=[],
            request_id=None,
            tenant_id="public",
            adapter=adapter,
            enable_contextual_rag=True,
            llm=llm,
        )

    assert enrich.call_args.kwargs["raise_on_failure"] is True
    embed.assert_not_called()
