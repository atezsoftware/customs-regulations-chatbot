from unittest.mock import MagicMock

from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.regulatory.indexing import RegulatoryIndexingChunker


class _CharacterTokenizer:
    """Predictable tokenizer for testing adaptive embedding budgets."""

    def encode(self, text: str) -> list[int]:
        return list(range((len(text) + 1) // 2))


class _OneCharacterPerTokenTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def test_contextual_regulatory_chunking_reserves_embedding_space() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_CharacterTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=True,
    )
    text = "Madde 1 - Genel hükümler\n\n" + "\n\n".join(
        f"({index}) " + "hukuki düzenleme koşulu ve sonucu. " * 12
        for index in range(1, 16)
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        text,
        source_path="file-id",
        source_file="regulation.txt",
    )

    assert len(chunked.chunks) > 1
    assert reserved == [DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS] * len(chunked.chunks)
    assert max(len(chunker.tokenizer.encode(item.text)) for item in chunked.chunks) <= (
        chunker.chunk_token_limit - DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
    )


def test_unsplittable_oversized_legal_line_keeps_text_and_skips_context() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_CharacterTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=True,
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        "Madde 1 - " + "kesintisiz hüküm metni " * 240,
        source_path="file-id",
        source_file="regulation.txt",
    )

    assert len(chunked.chunks) == 1
    assert reserved == [0]
    assert "kesintisiz hüküm metni" in chunked.chunks[0].text


def test_minimum_structural_chunk_uses_remaining_embedding_capacity() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_OneCharacterPerTokenTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=True,
    )
    text = "\n\n".join(
        f"MADDE {index} - " + "hukuki kapsam ve koşul. " * 18 for index in range(1, 4)
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        text,
        source_path="file-id",
        source_file="regulation.txt",
    )

    largest_chunk_tokens = max(
        len(chunker.tokenizer.encode(item.text)) for item in chunked.chunks
    )
    assert len(chunked.chunks) > 1
    assert 312 < largest_chunk_tokens < chunker.chunk_token_limit
    assert min(reserved) == chunker.chunk_token_limit - largest_chunk_tokens
    assert all(value > 0 for value in reserved)


def test_single_chunk_document_does_not_spend_contextual_llm_call() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_CharacterTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=True,
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        "Madde 1 - Kısa ve kendi bağlamını taşıyan hüküm.",
        source_path="file-id",
        source_file="regulation.txt",
    )

    assert len(chunked.chunks) == 1
    assert reserved == [0]


def test_non_contextual_regulatory_chunking_preserves_existing_size() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_CharacterTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=False,
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        "Madde 1 - " + "uzun hüküm. " * 120,
        source_path="file-id",
        source_file="regulation.txt",
    )

    assert chunker._regulatory_chunker.max_chunk_chars == 2400
    assert reserved == [0] * len(chunked.chunks)


def test_one_oversized_chunk_does_not_disable_context_for_eligible_siblings() -> None:
    chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=_OneCharacterPerTokenTokenizer(),  # type: ignore[arg-type]
        enable_contextual_rag=True,
    )
    text = (
        "MADDE 1 - "
        + "kesintisiz" * 80
        + "\n\nMADDE 2 - Kısa fakat başka hükümlerle birlikte bağlam gerektiren kural."
        + "\n\nMADDE 3 - İkinci kısa kural ve sonucu."
    )

    chunked, reserved = chunker._chunk_with_embedding_budget(
        text,
        source_path="file-id",
        source_file="regulation.txt",
    )

    assert len(chunked.chunks) > 1
    assert len(reserved) == len(chunked.chunks)
    assert 0 in reserved
    assert any(value >= 32 for value in reserved)
