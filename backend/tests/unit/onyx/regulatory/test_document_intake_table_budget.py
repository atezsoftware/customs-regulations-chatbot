"""Whole tariff rows can borrow context capacity without losing column relations."""

import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit_budget as edit_budget
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.natural_language_processing.utils import BaseTokenizer, TiktokenTokenizer
from onyx.regulatory.contextual import fit_context_fields_to_embedding_budget
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker
from onyx.regulatory.indexing import RegulatoryIndexingChunker

# Exact converted source row from RG 20251220-3-1.pdf, physical page 55.
_TARIFF_ROW = (
    "| 8544 ila 8548 | İzole tel, kablo ve diğer izole elektrik iletkenleri, fiber "
    "optik kablolar<br>Kömür elektrotlar, kömür fırçalar, lamba kömürleri, batarya "
    "kömürleri ve grafitten veya başka karbondan yapılma elektrik işlerinde "
    "kullanılan türden diğer eşya<br>Herhangi bir eşyadan mamul elektrik "
    "izolatörleri<br>Elektrikli makineler için izole edici teçhizat, tertibat veya "
    "ekipman, izole edici eşya ile kaplı temel metalden yapılma elektrik tesisatı "
    "ve aksamı<br>Primer hücrelerin, primer pillerin ve elektrik akümülatörlerin "
    "döküntü ve hurdaları; kullanılmış primer hücreler, kullanılmış primer piller "
    "ve kullanılmış elektrik akümülatörleri; makinelerin veya aparatların bu "
    "Fasıl altında başka yerde bulunmayan ve düzenlenmeyen elektrik aksamı | "
    "Kullanılan tüm girdilerin kıymetinin ürünün fabrika çıkış fiyatının %50'sini "
    "geçmeyen imalat |"
)
_TABLE_START = "| Kod | Tanım | Kural |\n| --- | --- | --- |"


def test_real_annex_row_keeps_all_columns_and_uses_remaining_context_capacity() -> None:
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    assert len(tokenizer.encode(_TARIFF_ROW)) == 314
    expected = f"{_TABLE_START}\n{_TARIFF_ROW}\n| 8549 | Diğer eşya | İmalat |"
    source = f"# EK II\n\n{expected}"
    adapter = RegulatoryIndexingChunker(
        db_session=MagicMock(), tokenizer=tokenizer, enable_contextual_rag=True
    )
    chunked, reserves = adapter._chunk_with_embedding_budget(
        source, source_path="annex", source_file="20251220-3-1.pdf", intake_version="3"
    )
    atomic = [
        chunk for chunk in chunked.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert re.sub(r"\s+", "", "".join(chunk.text for chunk in atomic)) == re.sub(
        r"\s+", "", expected
    )
    preserved = [
        index for index, chunk in enumerate(chunked.chunks) if chunk.text == _TARIFF_ROW
    ]
    assert len(preserved) == 1
    assert reserves[preserved[0]] == 198
    row = chunked.chunks[preserved[0]]
    assert (
        source[row.metadata.source_start_char : row.metadata.source_end_char]
        == _TARIFF_ROW
    )
    assert row.metadata.chunk_type == "table"
    assert row.metadata.table_index is not None
    for chunk, reserve in zip(chunked.chunks, reserves):
        assert len(tokenizer.encode(chunk.text)) + reserve <= 512
        summary, context = fit_context_fields_to_embedding_budget(
            title_prefix="",
            content=chunk.text,
            metadata_suffix="",
            doc_summary="Bu ek menşe kurallarını düzenler. " * 100,
            chunk_context="Tarife kodu için satırın tüm ürün tanımları aynı yüzde sınırına tabidir. "
            * 100,
            tokenizer=tokenizer,
            embedding_token_limit=512,
        )
        assert context.strip()
        assert len(tokenizer.encode(summary + chunk.text + context)) <= 512


class _CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.mark.parametrize("row_tokens", [313, 480, 481])
def test_special_row_budget_keeps_minimum_context_and_never_joins_siblings(
    row_tokens: int,
) -> None:
    row = "| code | " + "x" * (row_tokens - len("| code |  |")) + " |"
    expected = f"{_TABLE_START}\n{row}\n| z | z | z |"
    adapter = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=cast(BaseTokenizer, _CharacterTokenizer()),
        enable_contextual_rag=True,
    )
    if row_tokens > 480:
        with pytest.raises(ValueError, match="table row exceeds"):
            adapter._chunk_with_embedding_budget(
                expected, source_path="test", source_file="test.md", intake_version="3"
            )
        return
    chunked, reserves = adapter._chunk_with_embedding_budget(
        expected, source_path="test", source_file="test.md", intake_version="3"
    )
    index = next(
        index for index, chunk in enumerate(chunked.chunks) if chunk.text == row
    )
    assert reserves[index] == 512 - row_tokens
    assert all(len(chunk.text) <= 480 for chunk in chunked.chunks)


@pytest.mark.parametrize("version", ["1", "2"])
def test_old_intake_profiles_keep_their_original_whole_row_rejection(
    version: str,
) -> None:
    adapter = RegulatoryIndexingChunker(
        db_session=MagicMock(),
        tokenizer=TiktokenTokenizer("text-embedding-3-large"),
        enable_contextual_rag=True,
    )
    with pytest.raises(ValueError, match="table row exceeds"):
        adapter._chunk_with_embedding_budget(
            f"{_TABLE_START}\n{_TARIFF_ROW}",
            source_path="test",
            source_file="test.md",
            intake_version=version,
        )


def test_previously_fitting_rows_and_single_cell_continuations_are_unchanged() -> None:
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    for source in (
        f"{_TABLE_START}\n| 8544 | Kablo | İmalat |",
        "| " + "<br>".join([("Özgün belge sunulur. " * 10).strip()] * 5) + " |",
    ):
        original = IntakeRegulatoryChunker(
            tokenizer=tokenizer, content_token_limit=312
        ).chunk_text(source)
        adaptive = IntakeRegulatoryChunker(
            tokenizer=tokenizer,
            content_token_limit=312,
            whole_table_row_token_limit=480,
        ).chunk_text(source)
        assert original == adaptive


def test_extra_row_capacity_never_detaches_a_pending_article_heading() -> None:
    source = f"# Kaynak belge\n\n## MADDE 1 — Başvuru\n\n{_TARIFF_ROW}"
    for whole_row_limit in (312, 480):
        with pytest.raises(ValueError, match="table row exceeds"):
            IntakeRegulatoryChunker(
                tokenizer=TiktokenTokenizer("text-embedding-3-large"),
                content_token_limit=312,
                whole_table_row_token_limit=whole_row_limit,
            ).chunk_text(source)


@pytest.mark.parametrize(
    ("version", "chunk_type", "variant", "text", "accepted"),
    [
        ("3", "table", "atomic", _TARIFF_ROW, True),
        ("2", "table", "atomic", _TARIFF_ROW, False),
        ("3", "free_text", "atomic", _TARIFF_ROW, False),
        ("3", "table", "hierarchical_aggregate", _TARIFF_ROW, False),
        ("3", "table", "atomic", _TARIFF_ROW + "\n| a | b |", False),
        ("3", "table", "atomic", "| code | " + "özgün " * 300 + "|", False),
    ],
)
def test_review_edits_share_the_same_narrow_whole_row_budget(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    chunk_type: str,
    variant: str,
    text: str,
    accepted: bool,
) -> None:
    settings = SimpleNamespace(provider_type=None, model_name="text-embedding-3-large")
    monkeypatch.setattr(
        edit_budget, "get_current_search_settings", lambda _session: settings
    )
    monkeypatch.setattr(
        edit_budget, "effective_contextual_rag_enabled", lambda _settings: True
    )
    monkeypatch.setattr(
        edit_budget,
        "_embedding_tokenizer",
        lambda _settings: TiktokenTokenizer("text-embedding-3-large"),
    )
    generation = edit_budget.compute_regulatory_chunk_generation_hash(
        embedding_provider=None,
        embedding_model_name=settings.model_name,
        enable_contextual_rag=True,
        intake_chunking_version=version,
    )
    file_id = uuid4()
    file = cast(UserFile, SimpleNamespace(regulatory_chunk_generation_hash=generation))
    row = RegulatoryChunk(
        id="tariff-row",
        user_file_id=file_id,
        text=text,
        chunk_type=chunk_type,
        chunk_metadata={"chunk_variant": variant, "intake_chunking_version": version},
    )
    if accepted:
        edit_budget.validate_intake_edit_budgets(
            MagicMock(spec=Session), {file_id: file}, {file_id: [row]}
        )
    else:
        with pytest.raises(ValueError, match="content limit"):
            edit_budget.validate_intake_edit_budgets(
                MagicMock(spec=Session), {file_id: file}, {file_id: [row]}
            )
