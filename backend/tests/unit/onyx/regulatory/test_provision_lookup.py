from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.regulatory.provision_lookup import (
    ProvisionLookupResult,
    ProvisionRequest,
    lookup_provision,
)

SOURCE = "Karayolu Dışında Kullanılan Hareketli Makinaların İthalat Denetimi Tebliği"


def chunk(
    number: int = 0,
    *,
    source: str = SOURCE,
    file: str = "file-a",
    heading: str = "MADDE 1",
    text: str = "MADDE 1- (1) Amaç.\n(2) İkinci fıkra.",
    path: list[str] | None = None,
) -> InferenceChunk:
    return InferenceChunk(
        document_id=file,
        chunk_id=number,
        content=text,
        source_type=DocumentSource.USER_FILE,
        semantic_identifier=source,
        title=source,
        boost=1,
        score=1,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        image_file_id=None,
        source_links={},
        section_continuation=False,
        blurb=text,
        regulatory_chunk_id=f"rc-{file}-{number}",
        heading_path=path or [source, heading],
    )


def run_lookup(
    rows: list[InferenceChunk], **kwargs: Any
) -> tuple[ProvisionLookupResult, MagicMock, IndexFilters]:
    index = MagicMock()
    index.keyword_retrieval.return_value = rows
    filters = IndexFilters(
        access_control_list=["user:1"],
        tenant_id="tenant-a",
        document_set=["Corpus"],
        as_of_date=date(2026, 9, 24),
    )
    result = lookup_provision(
        ProvisionRequest(source=SOURCE, article_number="1", **kwargs),
        document_index=index,
        filters=filters,
    )
    return result, index, filters


def test_paragraph_is_found_and_existing_scope_is_preserved() -> None:
    result, index, filters = run_lookup([chunk()], paragraph=2)
    assert result.status == "found"
    assert result.chunks[0].content.endswith("(2) İkinci fıkra.")
    applied = index.keyword_retrieval.call_args.kwargs["filters"]
    assert applied.access_control_list == filters.access_control_list
    assert applied.tenant_id == filters.tenant_id
    assert applied.document_set == filters.document_set
    assert applied.as_of_date == filters.as_of_date
    assert applied.regulatory_chunks_only
    index.semantic_retrieval.assert_not_called()


def test_same_title_different_files_is_ambiguous_not_first_hit() -> None:
    result, _, _ = run_lookup([chunk(file="year-2025"), chunk(file="year-2026")])
    assert result.status == "ambiguous_source"
    assert not result.chunks
    assert len(result.sources) == 2


@pytest.mark.parametrize(
    "heading", ["EK MADDE 1", "GEÇİCİ MADDE 1", "MÜKERRER MADDE 1", "EK-1"]
)
def test_regular_article_does_not_accept_other_namespace(heading: str) -> None:
    result, _, _ = run_lookup(
        [chunk(heading=heading, text=heading + "- İlgisiz metin.")]
    )
    assert result.status != "found"


def test_scalar_paragraph_metadata_does_not_prove_requested_paragraph() -> None:
    row = chunk(text="MADDE 1- a) Liste öğesi\n2) Alt öğe")
    row.metadata["paragraph_no"] = "2"
    result, _, _ = run_lookup([row], paragraph=2)
    assert result.status == "partial"


def test_foreign_source_body_reference_is_not_source_identity() -> None:
    result, _, _ = run_lookup(
        [chunk(source="Başka bir tebliğ", text=SOURCE + "\nMADDE 1- Atıf")]
    )
    assert result.status == "not_found_in_scope"
    assert not result.chunks


def test_source_limit_never_claims_complete_article() -> None:
    result, _, _ = run_lookup([chunk(n) for n in range(129)])
    assert result.status == "partial"
    assert len(result.chunks) <= 24


def test_result_order_uses_structural_position_and_canonical_dedup() -> None:
    a = chunk(1, text="MADDE 1- (1) Başlangıç.")
    b = chunk(2, text="(2) Devam.", path=[SOURCE, "MADDE 1", "(2) Devam."])
    result, _, _ = run_lookup([b, a, a])
    assert [r.chunk_id for r in result.chunks] == [1, 2]


def test_explicit_date_cannot_override_existing_scope_date() -> None:
    result, index, _ = run_lookup([chunk()], as_of_date=date(2025, 1, 1))
    assert result.status == "invalid_reference"
    index.keyword_retrieval.assert_not_called()


def test_legacy_heading_is_resolved_only_with_complete_source_boundaries() -> None:
    index = MagicMock()
    stale = chunk(heading="MADDE 99", text="MADDE 1- (1) Başlangıç.\n(2) Hedef.")
    next_article = chunk(1, heading="MADDE 99", text="MADDE 2- Başka madde.")
    index.keyword_retrieval.side_effect = [[], [stale, next_article]]
    result = lookup_provision(
        ProvisionRequest(source=SOURCE, article_number="1", paragraph=2),
        document_index=index,
        filters=IndexFilters(access_control_list=["user:1"]),
    )
    assert result.status == "found"
    assert [row.chunk_id for row in result.chunks] == [0]
    assert result.chunks[0].heading_path == stale.heading_path  # No metadata rewrite.


def test_official_number_does_not_override_conflicting_named_title() -> None:
    index = MagicMock()
    index.keyword_retrieval.return_value = [
        chunk(source="5434 sayılı Tamamen Farklı Kanunu")
    ]
    result = lookup_provision(
        ProvisionRequest(
            source="5434 sayılı Türkiye Cumhuriyeti Emekli Sandığı Kanunu",
            article_number="1",
        ),
        document_index=index,
        filters=IndexFilters(access_control_list=["user:1"]),
    )
    assert result.status == "not_found_in_scope"


def test_split_paragraph_retains_continuation_until_next_paragraph() -> None:
    first = chunk(0, text="MADDE 1- (1) Başlangıç.\n(2) İlk yarı")
    second = chunk(1, text="ikinci yarı.\n(3) Sonraki fıkra.")
    result, _, _ = run_lookup([first, second], paragraph=2)
    assert result.status == "found"
    assert [row.chunk_id for row in result.chunks] == [0, 1]


def test_stale_continuation_is_resolved_even_when_first_heading_hit_exists() -> None:
    first = chunk(0, text="MADDE 1- (1) Başlangıç.\n(2) İlk yarı")
    second = chunk(1, heading="MADDE 99", text="ikinci yarı.\n(3) Sonraki fıkra.")
    index = MagicMock()
    index.keyword_retrieval.side_effect = [[first], [first, second]]
    result = lookup_provision(
        ProvisionRequest(source=SOURCE, article_number="1", paragraph=2),
        document_index=index,
        filters=IndexFilters(access_control_list=["user:1"]),
    )
    assert result.status == "found"
    assert [row.chunk_id for row in result.chunks] == [0, 1]


def test_previous_paragraph_clauses_do_not_hide_next_paragraph() -> None:
    result, _, _ = run_lookup(
        [chunk(text="MADDE 1- (1) İlk.\na) bent\nb) bent\n(2) Hedef.")], paragraph=2
    )
    assert result.status == "found"


def test_markdown_article_heading_does_not_hide_first_paragraph() -> None:
    result, _, _ = run_lookup(
        [chunk(text="**MADDE 1 -** (1) Amaç ve kapsam.")], paragraph=1
    )
    assert result.status == "found"


@pytest.mark.parametrize(
    "text",
    [
        "MADDE 1- (1) Belgeler.\na) Başvuru.\n    (1) Kimlik.\n    (2) Adres.",
        "MADDE 1- (1) Açıklama.\n“\n(2) Başka maddeden alıntı.\n”",
    ],
)
def test_nested_or_quoted_number_is_not_a_top_level_paragraph(text: str) -> None:
    result, _, _ = run_lookup([chunk(text=text)], paragraph=2)
    assert result.status == "partial"


def test_following_annex_is_not_part_of_the_last_article() -> None:
    first = chunk(0, text="MADDE 1- (1) Amaç.")
    annex = chunk(1, heading="EK-1", text="EK-1\nBaşvuru formu\n(2) Form alanı.")
    index = MagicMock()
    index.keyword_retrieval.side_effect = [[first], [first, annex]]
    result = lookup_provision(
        ProvisionRequest(source=SOURCE, article_number="1", paragraph=2),
        document_index=index,
        filters=IndexFilters(access_control_list=["user:1"]),
    )
    assert result.status == "partial"
    assert [row.chunk_id for row in result.chunks] == [0]


def test_clause_in_next_paragraph_cannot_match_requested_paragraph() -> None:
    result, _, _ = run_lookup(
        [
            chunk(
                path=[SOURCE, "MADDE 1", "(2) Hedef"],
                text="(2) Bent yok.\n(3) Başka fıkra\na) Yanlış bent.",
            )
        ],
        paragraph=2,
        clause="a",
    )
    assert result.status != "found"


def test_clause_without_paragraph_is_ambiguous_when_repeated() -> None:
    result, _, _ = run_lookup(
        [chunk(text="MADDE 1- (1) İlk.\na) Birinci\n(2) İkinci\na) İkinci")], clause="a"
    )
    assert result.status == "ambiguous_unit"
