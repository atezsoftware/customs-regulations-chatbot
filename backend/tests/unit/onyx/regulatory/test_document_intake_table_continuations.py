"""Source-cell continuations retain the real row instead of inventing table rows."""

import hashlib
import json
import re
from dataclasses import replace

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.chunker import (
    RegulatoryChunk,
    RegulatoryChunker,
    _markdown_to_blocks,
)
from onyx.regulatory.document_intake.chunking import (
    IntakeChunkMetadata,
    IntakeRegulatoryChunker,
)

# EK-3 approval certificate, 20160421-4-1.pdf page 3, original field 10.
REAL_ROW = (
    "| 10. Taşınmasına izin verilen tehlikeli mallar:<br>"
    "Araç, No. 7'deki araç kodlarına atanmış olan tehlikeli malların taşınması için aranan koşulları karşılamaktadır.<br>"
    "10.1 EX/II ise □ uyumluluk grubu J de dahil olmak üzere Sınıf 1 maddeleri<br>"
    "veya EX/III aracı ise ³ □ uyumluluk grubu J hariç olmak üzere Sınıf 1 maddeleri<br>"
    "10.2 Tanker/tüplü gaz tankeri için ³<br>"
    "□ yalnızca tank kodu ve No. 9'da belirtilen özel hükümler kapsamında izin verilen maddeler taşınabilir ⁵<br>"
    "ya da<br>"
    "□ yalnızca aşağıdaki maddeler taşınabilir (Sınıf, BM No. ve gerekliyse ambalajlama grubu ile tam sevkiyat ismi):<br>"
    "Yalnızca, gövde, conta, teçhizat ve (varsa) koruyucu astar malzemeleriyle tehlikeli tepkimeye girmeye meyilli olmayan maddeler taşınabilir. |"
)

FRONT_TABLE = "\n".join(
    [
        "| 1. Sertifika No. | 2. Araç üreticisi: | 3. Araç Tanımlama No | 4. Tescil numarası (varsa): |",
        "| :--- | :--- | :--- | :--- |",
        "| 5. Taşıyıcının, operatörün veya araç sahibinin adı ve iş adresi: |",
        "| 6. Araçla ilgili açıklamalar: ¹ |",
        "| 7. ADR 9.1.1.2 kapsamındaki araç kodu (kodları): ²<br>EX/II EX/III FL OX AT MPÜB |",
        "| 8. Dayanıklı fren sistemi: ³<br>□ Yok<br>□ ADR 9.2.3.1.2 uyarınca etkinlik_tonluk ⁴ bir taşıma ünitesinin toplam kütlesi için yeterlidir. |",
        "| 9. Sabit tanklar/tüplü gaz tankeriyle ilgili açıklama(varsa):<br>9.1 Tank üreticisi:<br>9.2 Tank/tüplü gaz tankeri onay numarası:<br>9.3 Tank üreticisinin seri numarası/Tüplü gaz tankerinin elemanlarının tanımlaması:<br>9.4. Üretim yılı:<br>9.5 ADR 4.3.3.1 veya 4.3.4.1 uyarınca tank kodu:<br>9.6 ADR 6.8.4 uyarınca TC ve TE özel hükmü (varsa) ⁶: |",
        REAL_ROW,
        "| 11. Açıklamalar/Remarks: (Bir açıklama yazılacaksa hem Türkçe hem de İngilizce, Fransızca veya Almanca yazılacaktır) (b) |",
        "| 12. Son geçerlilik tarihi: | Düzenleyen kurum mührü<br><br>Yer, Tarih, İmza |",
    ]
)


@pytest.fixture
def tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer("text-embedding-3-large")


def _atomic(chunks: list[RegulatoryChunk]) -> list[RegulatoryChunk]:
    return [chunk for chunk in chunks if chunk.metadata.chunk_variant == "atomic"]


def _fragments(chunks: list[RegulatoryChunk]) -> list[RegulatoryChunk]:
    return [
        chunk
        for chunk in _atomic(chunks)
        if isinstance(chunk.metadata, IntakeChunkMetadata)
        and chunk.metadata.intake_table_row is not None
    ]


def test_real_single_cell_row_keeps_every_character_and_original_grid_identity(
    tokenizer: TiktokenTokenizer,
) -> None:
    assert len(REAL_ROW) == 742
    assert len(tokenizer.encode(REAL_ROW)) == 318
    assert hashlib.sha256(REAL_ROW.encode()).hexdigest() == (
        "02cfc42ab0c3386e325f052fc6bff2d13629a9b4bf4b896f3c3b5c605028e572"
    )
    source = "# EK-3\n\n" + FRONT_TABLE
    chunker = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    )
    result = chunker.chunk_text(source, source_path="certificate", source_file="ek3.md")
    fragments = _fragments(result.chunks)
    assert len(fragments) == 2
    assert "".join(chunk.text for chunk in fragments) == REAL_ROW
    assert fragments[1].text.startswith("<br>10.2 Tanker/")
    assert [len(tokenizer.encode(chunk.text)) for chunk in fragments] == [137, 181]
    assert re.sub(
        r"\s+", "", "".join(chunk.text for chunk in _atomic(result.chunks))
    ) == (re.sub(r"\s+", "", FRONT_TABLE))
    for index, chunk in enumerate(fragments):
        assert isinstance(chunk.metadata, IntakeChunkMetadata)
        row = chunk.metadata.intake_table_row
        assert row is not None
        assert row.table_index == 1
        assert row.row_index == 6
        assert row.cell_index == 0
        assert (row.fragment_index, row.fragment_count) == (index, 2)
        assert source[row.row_source_start_char : row.row_source_end_char] == REAL_ROW
        assert (
            source[row.fragment_source_start_char : row.fragment_source_end_char]
            == chunk.text
        )
        assert chunk.metadata.source_start_char == row.fragment_source_start_char
        assert chunk.metadata.source_end_char == row.fragment_source_end_char
        stored = chunk.metadata.to_storage_dict()
        assert (
            json.loads(json.dumps(stored["intake_table_row"], ensure_ascii=False))
            == (stored["intake_table_row"])
        )
        assert (
            stored["intake_table_row"]["row_sha256"]
            == hashlib.sha256(REAL_ROW.encode()).hexdigest()
        )
    assert all(len(tokenizer.encode(chunk.text)) <= 312 for chunk in result.chunks)
    assert result == chunker.chunk_text(
        source, source_path="certificate", source_file="ek3.md"
    )


def test_continuation_measures_final_article_prefix_and_preserves_source_spans(
    tokenizer: TiktokenTokenizer,
) -> None:
    heading = "## MADDE 1 — Başvuru açıklamaları"
    source = heading + "\n\n" + REAL_ROW
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    ).chunk_text(source)
    fragments = _fragments(result.chunks)
    assert len(fragments) == 2
    assert fragments[0].text.startswith(heading + "\n| 10.")
    assert all(len(tokenizer.encode(chunk.text)) <= 312 for chunk in result.chunks)
    assert re.sub(
        r"\s+", "", "".join(chunk.text for chunk in _atomic(result.chunks))
    ) == (re.sub(r"\s+", "", source))
    for chunk in fragments:
        assert isinstance(chunk.metadata, IntakeChunkMetadata)
        row = chunk.metadata.intake_table_row
        assert row is not None
        assert (
            source[row.fragment_source_start_char : row.fragment_source_end_char]
            in chunk.text
        )
        assert row.row_source_start_char == source.index(REAL_ROW)


def test_identical_oversized_rows_keep_distinct_offsets_indices_and_chunk_ids(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = "# EK-3\n\n" + REAL_ROW + "\n" + REAL_ROW
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    ).chunk_text(source, source_path="repeated-certificate")
    fragments = _fragments(result.chunks)
    assert len(fragments) == 4
    assert len({chunk.metadata.chunk_id for chunk in fragments}) == 4
    starts = [source.index(REAL_ROW), source.rindex(REAL_ROW)]
    for row_index in [0, 1]:
        row_fragments = fragments[row_index * 2 : row_index * 2 + 2]
        assert "".join(chunk.text for chunk in row_fragments) == REAL_ROW
        for fragment_index, chunk in enumerate(row_fragments):
            assert isinstance(chunk.metadata, IntakeChunkMetadata)
            row = chunk.metadata.intake_table_row
            assert row is not None
            assert row.row_index == row_index
            assert row.row_source_start_char == starts[row_index]
            assert row.row_source_end_char == starts[row_index] + len(REAL_ROW)
            assert row.fragment_index == fragment_index
            assert row.fragment_count == 2
            assert (
                source[row.fragment_source_start_char : row.fragment_source_end_char]
                == chunk.text
            )
    assert all(
        "intake_table_row" not in chunk.metadata.to_storage_dict()
        for chunk in result.chunks
        if chunk.metadata.chunk_variant != "atomic"
    )


def test_tight_heading_budget_changes_only_the_existing_source_break(
    tokenizer: TiktokenTokenizer,
) -> None:
    heading = "## MADDE 1 — " + "Açıklama " * 50
    heading = heading.rstrip()
    assert len(tokenizer.encode(heading + "\n" + REAL_ROW[:324])) > 312
    assert len(tokenizer.encode(heading + "\n" + REAL_ROW[:164])) <= 312
    source = heading + "\n\n" + REAL_ROW
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    ).chunk_text(source)
    fragments = _fragments(result.chunks)
    assert len(fragments) == 2
    assert fragments[0].text.startswith(heading + "\n| 10.")
    assert fragments[1].text.startswith("<br>10.1 EX/II")
    assert all(len(tokenizer.encode(chunk.text)) <= 312 for chunk in result.chunks)
    assert re.sub(
        r"\s+", "", "".join(chunk.text for chunk in _atomic(result.chunks))
    ) == re.sub(r"\s+", "", source)


def test_generic_actual_cell_breaks_support_more_than_two_exact_parts(
    tokenizer: TiktokenTokenizer,
) -> None:
    segment = "İlgili belgeler eksiksiz olarak yetkili makamca saklanır. " * 5
    row = "| " + "<br>".join([segment.rstrip()] * 4) + " |"
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=150, chunking_version="3"
    ).chunk_text("# EK-1\n\n" + row)
    fragments = _fragments(result.chunks)
    assert len(fragments) == 4
    assert "".join(chunk.text for chunk in fragments) == row
    assert all(len(tokenizer.encode(chunk.text)) <= 150 for chunk in result.chunks)
    for index, chunk in enumerate(fragments):
        assert isinstance(chunk.metadata, IntakeChunkMetadata)
        assert chunk.metadata.intake_table_row is not None
        assert chunk.metadata.intake_table_row.fragment_index == index
        assert chunk.metadata.intake_table_row.fragment_count == 4


@pytest.mark.parametrize("version", ["1", "2"])
def test_legacy_profiles_keep_their_explicit_oversized_row_failure(
    tokenizer: TiktokenTokenizer, version: str
) -> None:
    with pytest.raises(ValueError, match="table row exceeds"):
        IntakeRegulatoryChunker(
            tokenizer=tokenizer, content_token_limit=312, chunking_version=version
        ).chunk_text("# EK-3\n\n" + REAL_ROW)


@pytest.mark.parametrize(
    "row",
    [
        "| Identité | " + REAL_ROW[2:-2] + " |",
        "| Deux \\\\| " + REAL_ROW[2:-2] + " |",
        "| " + "özgün " * 200 + "|",
        "| " + "özgün " * 200 + "<br>son |",
        "| `" + "özgün " * 200 + "<br>son` |",
        "| " + "özgün " * 200 + "\\<br>son |",
    ],
)
def test_unsafe_or_multicolumn_rows_still_fail_without_fabricated_boundaries(
    tokenizer: TiktokenTokenizer, row: str
) -> None:
    with pytest.raises(ValueError, match="table row exceeds"):
        IntakeRegulatoryChunker(
            tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
        ).chunk_text("# EK-1\n\n" + row)


def test_fitting_rows_keep_legacy_projection_and_do_not_claim_continuation(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = "# EK-1\n\n| Alan | Değer |\n| --- | --- |\n| 1 | kısa<br>not |"
    old = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="2"
    ).chunk_text(source)
    new = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    ).chunk_text(source)
    assert new == old
    assert all(
        "intake_table_row" not in chunk.metadata.to_storage_dict()
        for chunk in new.chunks
    )


def test_derived_aggregate_does_not_inherit_one_fragment_row_identity(
    tokenizer: TiktokenTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "# EK-3\n\n" + REAL_ROW
    chunker = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=312, chunking_version="3"
    )
    original = chunker.chunk_text(source)
    fragment = _fragments(original.chunks)[0]
    aggregate = replace(
        fragment,
        text="Türetilmiş görünüm",
        metadata=replace(fragment.metadata, chunk_variant="hierarchical_aggregate"),
    )
    monkeypatch.setattr(
        RegulatoryChunker,
        "_build_hierarchical_aggregate_chunks",
        lambda *_args, **_kwargs: [aggregate],
    )
    result = chunker._build_hierarchical_aggregate_chunks(
        atomic_chunks=original.chunks,
        structure=original.structure,
        blocks=_markdown_to_blocks(source),
        metadata=original.metadata,
    )
    assert len(result) == 1
    assert "intake_table_row" not in result[0].metadata.to_storage_dict()
