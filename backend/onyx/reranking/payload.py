from collections.abc import Sequence

from onyx.context.search.models import InferenceChunk
from onyx.reranking.constants import RERANK_TOKEN_SAFETY_MARGIN_PERCENT
from onyx.reranking.models import RerankPayloadLimits, SerializedRerankCandidates


def estimate_text_tokens(text: str) -> int:
    """Return a conservative tokenizer-independent request budget.

    A provider can tokenize every UTF-8 byte independently, so character averages
    are unsafe for non-ASCII text. Count each encoded byte as a potential token and
    reserve an additional margin for tokenizer-specific normalization/markers.
    """
    utf8_bytes = len(text.encode("utf-8"))
    safety_margin = (utf8_bytes * RERANK_TOKEN_SAFETY_MARGIN_PERCENT + 99) // 100
    return utf8_bytes + safety_margin


def _within_limits(text: str, *, max_bytes: int, max_tokens: int) -> bool:
    return (
        len(text.encode("utf-8")) <= max_bytes
        and estimate_text_tokens(text) <= max_tokens
    )


def _canonical_source(chunk: InferenceChunk) -> str:
    if chunk.source_links:
        return min(chunk.source_links.items())[1]
    return f"{chunk.source_type.value}:{chunk.document_id}"


def _render_candidate(
    chunk: InferenceChunk,
    *,
    summary: str,
    context: str,
    body: str,
) -> str:
    lines = [
        f"Title: {chunk.title or chunk.semantic_identifier}",
        f"Canonical source: {_canonical_source(chunk)}",
    ]
    if chunk.heading_path:
        lines.append(f"Heading path: {' > '.join(chunk.heading_path)}")
    if summary:
        lines.append(f"Document summary: {summary}")
    if context:
        lines.append(f"Chunk context: {context}")
    lines.append(f"Body:\n{body}")
    return "\n".join(lines)


def _fit_candidate(
    chunk: InferenceChunk, *, max_bytes: int, max_tokens: int
) -> str | None:
    summary = chunk.doc_summary
    context = chunk.chunk_context
    body = chunk.content or chunk.blurb
    candidate = _render_candidate(chunk, summary=summary, context=context, body=body)
    if _within_limits(candidate, max_bytes=max_bytes, max_tokens=max_tokens):
        return candidate

    # Document summaries are repeated across chunks, so they are sacrificed first.
    summary = ""
    candidate = _render_candidate(chunk, summary=summary, context=context, body=body)
    if _within_limits(candidate, max_bytes=max_bytes, max_tokens=max_tokens):
        return candidate

    context = ""
    candidate = _render_candidate(chunk, summary=summary, context=context, body=body)
    if _within_limits(candidate, max_bytes=max_bytes, max_tokens=max_tokens):
        return candidate

    empty_body = _render_candidate(chunk, summary="", context="", body="")
    if not _within_limits(empty_body, max_bytes=max_bytes, max_tokens=max_tokens):
        return None

    low = 0
    high = len(body)
    while low < high:
        midpoint = (low + high + 1) // 2
        shortened = _render_candidate(
            chunk, summary="", context="", body=body[:midpoint]
        )
        if _within_limits(shortened, max_bytes=max_bytes, max_tokens=max_tokens):
            low = midpoint
        else:
            high = midpoint - 1
    return _render_candidate(chunk, summary="", context="", body=body[:low])


def serialize_rerank_candidates(
    chunks: Sequence[InferenceChunk],
    *,
    limits: RerankPayloadLimits | None = None,
) -> SerializedRerankCandidates:
    effective_limits = limits or RerankPayloadLimits()
    documents: list[str] = []
    submitted_chunks: list[InferenceChunk] = []
    total_bytes = 0
    total_tokens = 0

    for chunk in chunks[: effective_limits.max_candidates]:
        remaining_bytes = effective_limits.max_total_bytes - total_bytes
        remaining_tokens = effective_limits.max_total_tokens - total_tokens
        document = _fit_candidate(
            chunk,
            max_bytes=min(effective_limits.max_document_bytes, remaining_bytes),
            max_tokens=min(effective_limits.max_document_tokens, remaining_tokens),
        )
        if document is None:
            break
        document_bytes = len(document.encode("utf-8"))
        document_tokens = estimate_text_tokens(document)
        documents.append(document)
        submitted_chunks.append(chunk)
        total_bytes += document_bytes
        total_tokens += document_tokens

    return SerializedRerankCandidates(
        documents=documents,
        submitted_chunks=submitted_chunks,
        unsent_chunks=list(chunks[len(submitted_chunks) :]),
        utf8_bytes=total_bytes,
        estimated_tokens=total_tokens,
    )
