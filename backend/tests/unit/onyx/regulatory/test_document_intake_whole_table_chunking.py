"""Larger source tables must preserve rows without changing prose or retries."""

import hashlib
import re
from unittest.mock import MagicMock

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.document_intake.chunking import (
    IntakeChunkMetadata,
    IntakeRegulatoryChunker,
)
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
)
from onyx.regulatory.indexing import RegulatoryIndexingChunker
from tests.unit.onyx.regulatory.test_document_intake_table_budget import _TARIFF_ROW
from tests.unit.onyx.regulatory.whole_table_policy_helpers import (
    _CharacterTokenizer,
)


def test_table_over_preferred_capacity_splits_only_between_complete_rows() -> None:
    rows = ["| Kod | Ürün | Koşul |", "| --- | --- | --- |"] + [
        f"| {index} | " + "özgün " * 45 + "| %50 sınırı |" for index in range(9)
    ]
    source = "\n".join(rows)
    chunker = IntakeRegulatoryChunker(
        tokenizer=_CharacterTokenizer(),
        content_token_limit=312,
        table_content_token_limit=1848,
        chunking_version="4",
    )
    chunks = chunker.chunk_text(source).chunks
    atomic = [chunk for chunk in chunks if chunk.metadata.chunk_variant == "atomic"]
    assert len(atomic) > 1
    assert [line for chunk in atomic for line in chunk.text.splitlines()] == rows
    assert all(len(chunk.text) <= 1848 for chunk in atomic)
    for chunk in atomic:
        assert (
            source[chunk.metadata.source_start_char : chunk.metadata.source_end_char]
            == chunk.text
        )
    with pytest.raises(ValueError, match="table row exceeds"):
        chunker.chunk_text("| " + "a" * 1900 + "<br>koşul |")


def test_v4_preserves_legacy_caption_and_safe_single_cell_continuation() -> None:
    tokenizer = _CharacterTokenizer()
    captioned = "Ek bilgiler:\n| Özgün koşul ve adres |"
    legacy = IntakeRegulatoryChunker(tokenizer=tokenizer, content_token_limit=312)
    current = IntakeRegulatoryChunker(
        tokenizer=tokenizer,
        content_token_limit=312,
        chunking_version="4",
        table_content_token_limit=1848,
    )
    assert legacy.chunk_text(captioned) == current.chunk_text(captioned)
    source = "| " + "<br>".join([("Özgün belge sunulur. " * 9).strip()] * 20) + " |"
    for chunker in (legacy, current):
        result = chunker.chunk_text(source)
        atomic = [
            chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
        ]
        assert len(atomic) > 1
        assert re.sub(r"\s+", "", "".join(chunk.text for chunk in atomic)) == re.sub(
            r"\s+", "", source
        )
        for index, chunk in enumerate(atomic):
            assert isinstance(chunk.metadata, IntakeChunkMetadata)
            row = chunk.metadata.intake_table_row
            assert row is not None
            assert row.fragment_index == index and row.fragment_count == len(atomic)
            assert row.row_sha256 == hashlib.sha256(source.encode()).hexdigest()
            assert (
                source[row.fragment_source_start_char : row.fragment_source_end_char]
                == chunk.text
            )


def test_whole_table_budget_keeps_shared_rule_and_prose_budget_separate() -> None:
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    rows = ["| Kod | Ürün | Ortak kural |", "| --- | --- | --- |"] + [
        f"| {index} | İzole elektrik iletkenleri ve aksamı | Kullanılan girdilerin kıymeti fabrika çıkış fiyatının %50'sini geçemez |"
        for index in range(10)
    ]
    table = "\n".join(rows)
    source = table + "\n\n" + "Başvuru sahibi özgün belgeyi ibraz eder. " * 80
    writer = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=tokenizer,
        enable_contextual_rag=True,
    )
    result, reserves = writer._chunk_with_embedding_budget(
        source,
        source_path="source",
        source_file="annex.md",
        intake_version="4",
        preparation_policy=IntakePreparationPolicy(),
    )
    atomic = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert len([chunk for chunk in atomic if chunk.text == table]) == 1
    assert re.sub(r"\s+", "", "".join(chunk.text for chunk in atomic)) == re.sub(
        r"\s+", "", source
    )
    for chunk, reserve in zip(result.chunks, reserves, strict=True):
        limit = 2048 if chunk.metadata.chunk_type == "table" else 512
        assert len(tokenizer.encode(chunk.text)) + reserve <= limit
        if chunk.metadata.chunk_type != "table":
            assert len(tokenizer.encode(chunk.text)) <= 312


def test_whole_table_keeps_its_original_article_heading_within_table_budget() -> None:
    source = "# Kaynak belge\n\n## MADDE 1 — Başvuru\n\n" + _TARIFF_ROW
    result = IntakeRegulatoryChunker(
        tokenizer=TiktokenTokenizer("text-embedding-3-large"),
        content_token_limit=312,
        chunking_version="4",
        table_content_token_limit=1848,
    ).chunk_text(source)
    atomic = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert len(atomic) == 1
    assert atomic[0].text == "## MADDE 1 — Başvuru\n" + _TARIFF_ROW
    assert atomic[0].metadata.chunk_type == "table"
    assert atomic[0].metadata.article_no == "1"
