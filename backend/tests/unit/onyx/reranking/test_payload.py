from datetime import datetime, timezone

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.reranking.models import RerankPayloadLimits
from onyx.reranking.payload import estimate_text_tokens, serialize_rerank_candidates


def _chunk(
    *,
    document_id: str = "doc",
    chunk_id: int = 0,
    body: str = "Madde 5 uyarınca karar 06.08.2026 tarihinde verildi.",
    doc_summary: str = "Belge özeti " * 20,
    chunk_context: str = "Yerel bağlam " * 20,
) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=chunk_id,
        blurb=body[:20],
        content=body,
        source_links={0: "https://mevzuat.example/kanun/5"},
        image_file_id=None,
        section_continuation=False,
        document_id=document_id,
        source_type=DocumentSource.WEB,
        semantic_identifier=f"{document_id}-semantic",
        title="Kanun Başlığı",
        boost=0,
        score=1.0,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary=doc_summary,
        chunk_context=chunk_context,
        updated_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        heading_path=["Kısım I", "Madde 5"],
    )


def test_serializes_document_context_as_labeled_text() -> None:
    payload = serialize_rerank_candidates([_chunk()])

    assert payload.documents == [
        "Title: Kanun Başlığı\n"
        "Canonical source: https://mevzuat.example/kanun/5\n"
        "Heading path: Kısım I > Madde 5\n"
        f"Document summary: {'Belge özeti ' * 20}\n"
        f"Chunk context: {'Yerel bağlam ' * 20}\n"
        "Body:\nMadde 5 uyarınca karar 06.08.2026 tarihinde verildi."
    ]
    assert payload.submitted_chunks == [_chunk()]
    assert payload.unsent_chunks == []


def test_non_ascii_token_estimate_uses_utf8_upper_bound_with_margin() -> None:
    text = "İ" * 100
    utf8_bytes = len(text.encode("utf-8"))

    assert estimate_text_tokens(text) >= utf8_bytes + (utf8_bytes + 9) // 10


def test_per_document_bound_drops_summary_before_context_or_body() -> None:
    chunk = _chunk(doc_summary="S" * 400, chunk_context="context-kept")
    baseline = serialize_rerank_candidates(
        [chunk],
        limits=RerankPayloadLimits(
            max_candidates=1,
            max_document_bytes=10_000,
            max_document_tokens=10_000,
            max_total_bytes=10_000,
            max_total_tokens=10_000,
        ),
    ).documents[0]
    bounded = serialize_rerank_candidates(
        [chunk],
        limits=RerankPayloadLimits(
            max_candidates=1,
            max_document_bytes=len(baseline.encode("utf-8")) - 250,
            max_document_tokens=10_000,
            max_total_bytes=10_000,
            max_total_tokens=10_000,
        ),
    ).documents[0]

    assert "Document summary:" not in bounded
    assert "Chunk context: context-kept" in bounded
    assert "Body:\nMadde 5" in bounded


def test_payload_respects_count_byte_and_token_bounds_and_keeps_unsent_tail() -> None:
    chunks = [_chunk(document_id=f"d{index}") for index in range(4)]
    one_document = serialize_rerank_candidates(
        [chunks[0]],
        limits=RerankPayloadLimits(
            max_candidates=4,
            max_document_bytes=10_000,
            max_document_tokens=10_000,
            max_total_bytes=10_000,
            max_total_tokens=10_000,
        ),
    )
    limits = RerankPayloadLimits(
        max_candidates=3,
        max_document_bytes=10_000,
        max_document_tokens=10_000,
        max_total_bytes=one_document.utf8_bytes * 2,
        max_total_tokens=one_document.estimated_tokens * 2,
    )

    payload = serialize_rerank_candidates(chunks, limits=limits)

    assert len(payload.documents) == 2
    assert payload.utf8_bytes <= limits.max_total_bytes
    assert payload.estimated_tokens <= limits.max_total_tokens
    assert payload.submitted_chunks == chunks[:2]
    assert payload.unsent_chunks == chunks[2:]
    assert sum(estimate_text_tokens(document) for document in payload.documents) == (
        payload.estimated_tokens
    )


def test_oversized_body_is_truncated_without_losing_legal_heading() -> None:
    chunk = _chunk(body="esas " * 500, doc_summary="", chunk_context="")
    payload = serialize_rerank_candidates(
        [chunk],
        limits=RerankPayloadLimits(
            max_candidates=1,
            max_document_bytes=300,
            max_document_tokens=1_000,
            max_total_bytes=300,
            max_total_tokens=1_000,
        ),
    )

    assert len(payload.documents) == 1
    assert "Heading path: Kısım I > Madde 5" in payload.documents[0]
    assert "Body:\nesas" in payload.documents[0]
    assert payload.utf8_bytes <= 300
    assert payload.estimated_tokens <= 1_000
