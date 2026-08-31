from datetime import date
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.regulatory_chunks import (
    delete_hierarchical_aggregates_referencing_chunk,
    replace_indexed_chunks_for_file,
    supersede_hierarchical_aggregates_referencing_chunk,
)
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.chunker import RegulatoryChunker
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
        tokenizer=cast(BaseTokenizer, _CharacterTokenizer()),
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
        tokenizer=cast(BaseTokenizer, _CharacterTokenizer()),
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
        tokenizer=cast(BaseTokenizer, _OneCharacterPerTokenTokenizer()),
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
        tokenizer=cast(BaseTokenizer, _CharacterTokenizer()),
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
        tokenizer=cast(BaseTokenizer, _CharacterTokenizer()),
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
        tokenizer=cast(BaseTokenizer, _OneCharacterPerTokenTokenizer()),
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


def test_persisted_aggregate_metadata_resolves_atomic_regulatory_chunk_ids() -> None:
    chunked = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        """ULUSLARARASI SÖZLEŞME

MADDE 1 - Bu sözleşmede aşağıdaki veriler işlenir:

a) Data 1.

b) Data 2.
""",
        source_file="sozlesme.md",
    )
    db_session = MagicMock(spec=Session)
    db_session.execute.return_value.all.return_value = []

    rows = replace_indexed_chunks_for_file(
        db_session,
        UUID("00000000-0000-0000-0000-000000000123"),
        chunked.chunks,
    )

    assert len(rows) == 3
    assert [row.projection_ordinal for row in rows] == [0, 1, 2]
    assert rows[0].chunk_metadata["chunk_variant"] == "atomic"
    assert rows[1].chunk_metadata["chunk_variant"] == "atomic"
    assert rows[2].chunk_metadata["chunk_variant"] == "hierarchical_aggregate"
    assert rows[2].chunk_metadata["source_regulatory_chunk_ids"] == [
        rows[0].id,
        rows[1].id,
    ]


def test_atomic_mutation_deletes_only_aggregates_that_reference_it() -> None:
    affected = MagicMock()
    affected.chunk_metadata = {
        "source_regulatory_chunk_ids": ["source-chunk", "sibling-chunk"]
    }
    unaffected = MagicMock()
    unaffected.chunk_metadata = {"source_regulatory_chunk_ids": ["different-chunk"]}
    db_session = MagicMock(spec=Session)
    db_session.scalars.return_value.all.return_value = [affected, unaffected]

    deleted_count = delete_hierarchical_aggregates_referencing_chunk(
        db_session,
        user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        source_chunk_id="source-chunk",
    )

    assert deleted_count == 1
    db_session.delete.assert_called_once_with(affected)


def test_amendment_supersedes_only_aggregates_that_reference_old_chunk() -> None:
    affected = MagicMock()
    affected.chunk_metadata = {
        "source_regulatory_chunk_ids": ["source-chunk", "sibling-chunk"]
    }
    affected.status = "active"
    affected.validity_start_date = date(2020, 1, 1)
    unaffected = MagicMock()
    unaffected.chunk_metadata = {"source_regulatory_chunk_ids": ["different-chunk"]}
    db_session = MagicMock(spec=Session)
    db_session.scalars.return_value.all.return_value = [affected, unaffected]

    superseded = supersede_hierarchical_aggregates_referencing_chunk(
        db_session,
        user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        source_chunk_id="source-chunk",
        superseded_at=date(2026, 8, 15),
    )

    assert superseded == [affected]
    assert affected.status == "superseded"
    assert affected.validity_end_date == date(2026, 8, 15)
    db_session.add.assert_called_once_with(affected)
    db_session.delete.assert_not_called()
