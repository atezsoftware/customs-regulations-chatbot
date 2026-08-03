import json
from dataclasses import FrozenInstanceError
from datetime import date
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pytest import MonkeyPatch

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.db.enums import RegulatoryChunkStatus
from onyx.db.regulatory_chunks import (
    RegulatoryChunkProjection,
    RegulatoryProvisionHeadingCandidate,
    RegulatoryProvisionHeadingSource,
)
from onyx.regulatory import provision_retrieval

FILE_ID = UUID("00000000-0000-0000-0000-000000000061")


def _chunk(
    chunk_id: int,
    regulatory_chunk_id: str | None,
    *,
    document_id: str = str(FILE_ID),
    content: str | None = None,
) -> InferenceChunk:
    chunk_content = content if content is not None else f"seed-{chunk_id}"
    return InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=chunk_content,
        source_type=DocumentSource.USER_FILE,
        semantic_identifier="Tebliğ — MADDE 61",
        title="Tebliğ",
        boost=1,
        score=0.9,
        hidden=False,
        metadata={},
        match_highlights=["seed"],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        image_file_id=None,
        source_links={},
        section_continuation=False,
        blurb=chunk_content,
        file_id="stored-file-id",
        regulatory_chunk_id=regulatory_chunk_id,
        heading_path=["MADDE 61"] if regulatory_chunk_id else None,
    )


def _projection(
    chunk_id: str,
    projection_index: int,
    *,
    text: str,
    expansion_priority: int = 1,
    status: str = RegulatoryChunkStatus.ACTIVE.value,
) -> RegulatoryChunkProjection:
    return RegulatoryChunkProjection(
        regulatory_chunk_id=chunk_id,
        user_file_id=FILE_ID,
        projection_index=projection_index,
        position=projection_index,
        text=text,
        heading_path=("Tebliğ", "MADDE 61", f"({projection_index})"),
        article_no="61",
        status=status,
        validity_start_date=date(2024, 1, 1),
        validity_end_date=None,
        expansion_priority=expansion_priority,
    )


def test_selected_provision_siblings_become_exact_citable_sections(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(10, "seed-id")
    projections = [
        _projection("sibling-before", 9, text="Birinci fıkra"),
        _projection("seed-id", 10, text="Merkez fıkra"),
        _projection("sibling-after", 11, text="İkinci fıkra"),
    ]
    lookup = MagicMock(return_value=projections)
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        lookup,
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [inference_section_from_single_chunk(seed)],
        query="ikinci fıkra",
        as_of_date=date(2026, 1, 1),
        max_total_sections=3,
    )

    assert [section.center_chunk.regulatory_chunk_id for section in expanded] == [
        "seed-id",
        "sibling-after",
        "sibling-before",
    ]
    sibling = expanded[1].center_chunk
    assert sibling.chunk_id == 11
    assert sibling.content == "İkinci fıkra"
    assert sibling.heading_path == ["Tebliğ", "MADDE 61", "(11)"]
    assert sibling.semantic_identifier == "Tebliğ — Tebliğ > MADDE 61 > (11)"
    assert sibling.file_id == "stored-file-id"
    assert expanded[1].combined_content == "İkinci fıkra"
    lookup.assert_called_once()


def test_same_provision_expansion_prioritizes_distinctive_exact_query_anchor(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(10, "seed-id")
    projections = [
        _projection("seed-id", 10, text="Teminat indirimi koşulları"),
        _projection("generic-neighbor", 11, text="Kapsamlı teminat tutarı"),
        _projection("exact-rate", 12, text="İndirim oranı yüzde 30 olarak uygulanır"),
    ]
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        MagicMock(return_value=projections),
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [inference_section_from_single_chunk(seed)],
        query="indirilmiş kapsamlı teminat yüzde 30",
        as_of_date=None,
        max_total_sections=2,
    )

    assert [section.center_chunk.regulatory_chunk_id for section in expanded] == [
        "seed-id",
        "exact-rate",
    ]


def test_multi_seed_expansion_uses_best_queue_head_before_seed_order(
    monkeypatch: MonkeyPatch,
) -> None:
    first_seed = _chunk(2, "paragraph-two")
    second_seed = _chunk(3, "paragraph-three")
    projections = [
        _projection("weak-paragraph-one", 1, text="Genel kefil yükümlülüğü"),
        _projection("paragraph-two", 2, text="İbra edilmeme bildirimi"),
        _projection("paragraph-three", 3, text="Ödeme talebi süresi"),
        _projection(
            "consequence-paragraph-four",
            4,
            text="Süresinde bildirim yapılmazsa kefil yükümlülüklerinden kurtulur",
        ),
    ]
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        MagicMock(return_value=projections),
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [
            inference_section_from_single_chunk(first_seed),
            inference_section_from_single_chunk(second_seed),
        ],
        query="bildirim yapılmazsa kefil yükümlülüklerinden kurtulur",
        as_of_date=None,
        max_total_sections=3,
    )

    assert [section.center_chunk.regulatory_chunk_id for section in expanded] == [
        "paragraph-two",
        "paragraph-three",
        "consequence-paragraph-four",
    ]


def test_same_provision_expansion_repairs_selected_legacy_seed_heading(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(454, "legacy-heading")
    seed.heading_path = ["Convention", "Guarantee chapter", "2. Amount"]
    projection = RegulatoryChunkProjection(
        regulatory_chunk_id="legacy-heading",
        user_file_id=FILE_ID,
        projection_index=454,
        position=454,
        text="2. Comprehensive guarantee amount",
        heading_path=("Convention", "Guarantee chapter", "2. Amount"),
        article_no="75",
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=None,
        validity_end_date=None,
        chunk_type="numbered_section",
    )
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        MagicMock(return_value=[projection]),
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [inference_section_from_single_chunk(seed)],
        query="guarantee amount",
        as_of_date=None,
        max_total_sections=2,
    )

    assert expanded[0].center_chunk.heading_path == [
        "Convention",
        "Guarantee chapter",
        "MADDE 75",
        "2. Amount",
    ]
    assert expanded[0].center_chunk.content == projection.text


def test_same_provision_repairs_legacy_seed_even_when_budget_is_already_full(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(454, "legacy-heading")
    seed.heading_path = ["Convention", "Guarantee chapter", "2. Amount"]
    projection = RegulatoryChunkProjection(
        regulatory_chunk_id="legacy-heading",
        user_file_id=FILE_ID,
        projection_index=454,
        position=454,
        text="2. Comprehensive guarantee amount",
        heading_path=("Convention", "Guarantee chapter", "2. Amount"),
        article_no="75",
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=None,
        validity_end_date=None,
        chunk_type="numbered_section",
    )
    lookup = MagicMock(return_value=[projection])
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        lookup,
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [inference_section_from_single_chunk(seed)],
        query="guarantee amount",
        as_of_date=None,
        max_total_sections=1,
    )

    lookup.assert_called_once()
    assert "MADDE 75" in expanded[0].center_chunk.heading_path


def test_expansion_keeps_independent_seeds_before_bounded_siblings(
    monkeypatch: MonkeyPatch,
) -> None:
    first = _chunk(10, "first-seed")
    second = _chunk(20, "second-seed")
    projections = [
        _projection("first-extra", 9, text="First extra"),
        _projection("first-seed", 10, text="First seed"),
        _projection("second-seed", 20, text="Second seed"),
        _projection("second-extra", 21, text="Second extra"),
    ]
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        MagicMock(return_value=projections),
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [
            inference_section_from_single_chunk(first),
            inference_section_from_single_chunk(second),
        ],
        query="",
        as_of_date=None,
        max_total_sections=3,
    )

    assert [section.center_chunk.regulatory_chunk_id for section in expanded[:2]] == [
        "first-seed",
        "second-seed",
    ]
    assert len(expanded) == 3


def test_expansion_consumes_structural_context_before_ordinary_siblings(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(10, "seed-id")
    projections = [
        _projection("ordinary-earlier", 1, text="Earlier ordinary sibling"),
        _projection(
            "structural-parent",
            9,
            text="Operative parent text",
            expansion_priority=0,
        ),
        _projection("seed-id", 10, text="Seed"),
    ]
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        MagicMock(return_value=projections),
    )

    expanded = provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [inference_section_from_single_chunk(seed)],
        query="focused proposition",
        as_of_date=None,
        max_total_sections=2,
    )

    assert [section.center_chunk.regulatory_chunk_id for section in expanded] == [
        "seed-id",
        "structural-parent",
    ]


def test_explicit_result_reference_adds_exact_citable_chunk_without_recursing(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(
        20,
        "seed-id",
        content="Bu açıklama Madde 4A hükmüne atıf yapar.",
    )
    referenced = RegulatoryChunkProjection(
        regulatory_chunk_id="article-4a",
        user_file_id=FILE_ID,
        projection_index=4,
        position=4,
        text="Madde 9'a ayrıca atıf yapan doğrudan hüküm.",
        heading_path=("Düzenleme", "MADDE 4A", "(1)"),
        article_no="4A",
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=None,
        validity_end_date=None,
    )
    lookup = MagicMock(return_value=[referenced])
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_referenced_provisions",
        lookup,
    )
    section = inference_section_from_single_chunk(seed)

    expanded = provision_retrieval.expand_selected_regulatory_references(
        MagicMock(),
        [section],
        reference_sections=[section],
        query="ilişkili hüküm",
        as_of_date=None,
        max_total_sections=3,
    )

    assert [item.center_chunk.regulatory_chunk_id for item in expanded] == [
        "seed-id",
        "article-4a",
    ]
    assert expanded[1].center_chunk.heading_path == [
        "Düzenleme",
        "MADDE 4A",
        "(1)",
    ]
    assert expanded[1].combined_content == referenced.text
    lookup.assert_called_once()
    references = lookup.call_args.args[2]
    assert [(item.article_no, item.qualifier) for item in references] == [("4A", None)]
    assert lookup.call_args.kwargs["query"] == "ilişkili hüküm"


def test_reference_slots_prefer_focused_chunks_without_increasing_budget(
    monkeypatch: MonkeyPatch,
) -> None:
    seed = _chunk(
        20,
        "seed-id",
        content="Bu açıklama Madde 67 hükmüne atıf yapar.",
    )
    projections = [
        RegulatoryChunkProjection(
            regulatory_chunk_id="article-heading",
            user_file_id=FILE_ID,
            projection_index=1,
            position=1,
            text="Madde 67 İznin askıya alınması",
            heading_path=("Düzenleme", "MADDE 67"),
            article_no="67",
            status=RegulatoryChunkStatus.ACTIVE.value,
            validity_start_date=None,
            validity_end_date=None,
        ),
        RegulatoryChunkProjection(
            regulatory_chunk_id="generic-intro",
            user_file_id=FILE_ID,
            projection_index=2,
            position=2,
            text="Yetkili makam aşağıdaki durumlarda işlem yapar.",
            heading_path=("Düzenleme", "MADDE 67", "(1)"),
            article_no="67",
            status=RegulatoryChunkStatus.ACTIVE.value,
            validity_start_date=None,
            validity_end_date=None,
        ),
        RegulatoryChunkProjection(
            regulatory_chunk_id="material-condition",
            user_file_id=FILE_ID,
            projection_index=3,
            position=3,
            text="Mali yeterlilik koşulu için iyileştirme süresi verilir.",
            heading_path=("Düzenleme", "MADDE 67", "(1)", "(b)"),
            article_no="67",
            status=RegulatoryChunkStatus.ACTIVE.value,
            validity_start_date=None,
            validity_end_date=None,
        ),
        RegulatoryChunkProjection(
            regulatory_chunk_id="material-deadline",
            user_file_id=FILE_ID,
            projection_index=4,
            position=4,
            text="İyileştirme süresi 30 güne kadar uzatılabilir.",
            heading_path=("Düzenleme", "MADDE 67", "(2)"),
            article_no="67",
            status=RegulatoryChunkStatus.ACTIVE.value,
            validity_start_date=None,
            validity_end_date=None,
        ),
    ]
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_referenced_provisions",
        MagicMock(return_value=projections),
    )
    section = inference_section_from_single_chunk(seed)

    expanded = provision_retrieval.expand_selected_regulatory_references(
        MagicMock(),
        [section],
        reference_sections=[section],
        query="mali yeterlilik koşulu iyileştirme süresi 30 gün",
        as_of_date=None,
        max_total_sections=3,
    )

    assert [item.center_chunk.regulatory_chunk_id for item in expanded] == [
        "seed-id",
        "material-condition",
        "material-deadline",
    ]


def test_generic_sections_do_not_query_regulatory_database(
    monkeypatch: MonkeyPatch,
) -> None:
    generic = inference_section_from_single_chunk(_chunk(1, None, document_id="doc"))
    lookup = MagicMock()
    monkeypatch.setattr(
        provision_retrieval,
        "get_bounded_same_provision_siblings",
        lookup,
    )

    assert provision_retrieval.expand_selected_regulatory_sections(
        MagicMock(),
        [generic],
        query="anything",
        as_of_date=None,
        max_total_sections=5,
    ) == [generic]
    lookup.assert_not_called()


def _heading_candidate(
    position: int,
    heading_path: tuple[str, ...],
    *,
    status: str = RegulatoryChunkStatus.ACTIVE.value,
    validity_start_date: date | None = None,
    validity_end_date: date | None = None,
    article_title: str | None = None,
) -> RegulatoryProvisionHeadingCandidate:
    return RegulatoryProvisionHeadingCandidate(
        user_file_id=FILE_ID,
        position=position,
        heading_path=heading_path,
        status=status,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
        article_title=article_title,
    )


def _heading_source(
    candidates: list[RegulatoryProvisionHeadingCandidate],
    *,
    seed_positions: tuple[int, ...] = (10, 11),
) -> RegulatoryProvisionHeadingSource:
    return RegulatoryProvisionHeadingSource(
        user_file_id=FILE_ID,
        document_title="Genel Düzenleme",
        seed_positions=seed_positions,
        candidates=tuple(candidates),
    )


def test_navigation_deduplicates_bare_and_titled_article_headings() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(9, ("Genel Düzenleme", "MADDE 10")),
                _heading_candidate(
                    10,
                    ("Genel Düzenleme", "MADDE 10 - Bildirim yükümlülüğü"),
                ),
            ]
        ),
        as_of_date=None,
    )

    assert navigation is not None
    assert navigation.entries == (
        provision_retrieval.RegulatoryProvisionNavigationEntry(
            article_key="madde:10",
            heading_label="MADDE 10 — Bildirim yükümlülüğü",
        ),
    )


def test_navigation_keeps_same_article_number_in_annex_as_distinct_lead() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(1, ("Genel Düzenleme", "MADDE 1 - Amaç")),
                _heading_candidate(
                    40,
                    ("Genel Düzenleme", "EK IV", "MADDE 1 - İşlemler"),
                ),
            ],
            seed_positions=(1, 40),
        ),
        as_of_date=None,
    )

    assert navigation is not None
    assert [
        (entry.article_key, entry.heading_label) for entry in navigation.entries
    ] == [
        ("madde:1", "MADDE 1 — Amaç"),
        ("ek-iv::madde:1", "EK IV > MADDE 1 — İşlemler"),
    ]


def test_unknown_article_keeps_short_descendant_as_discovery_hint() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(9, ("Genel Düzenleme", "MADDE 10A")),
                _heading_candidate(
                    10,
                    (
                        "Genel Düzenleme",
                        "MADDE 10A",
                        "(1)",
                        "Sınır ötesi aktarım yasağı ve uygulanacak istisnalar",
                    ),
                ),
                _heading_candidate(11, ("Genel Düzenleme", "MADDE 11")),
            ]
        ),
        as_of_date=None,
    )

    assert navigation is not None
    assert navigation.entries[0] == (
        provision_retrieval.RegulatoryProvisionNavigationEntry(
            article_key="madde:10a",
            heading_label=(
                "MADDE 10A — Sınır ötesi aktarım yasağı ve uygulanacak istisnalar"
            ),
        )
    )


def test_article_title_metadata_outranks_descendant_hint() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(
                    9,
                    ("Genel Düzenleme", "MADDE 12", "Yakın fakat genel ipucu"),
                ),
                _heading_candidate(
                    10,
                    ("Genel Düzenleme", "MADDE 12"),
                    article_title="Denetim ve bildirim usulü",
                ),
            ]
        ),
        as_of_date=None,
    )

    assert navigation is not None
    assert navigation.entries == (
        provision_retrieval.RegulatoryProvisionNavigationEntry(
            article_key="madde:12",
            heading_label="MADDE 12 — Denetim ve bildirim usulü",
        ),
    )


def test_navigation_is_capped_at_12_and_prefers_seed_proximity() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(
                    position,
                    ("Genel Düzenleme", f"MADDE {position}"),
                )
                for position in range(1, 61)
            ],
            seed_positions=(30, 31),
        ),
        as_of_date=None,
        max_headings=100,
    )

    assert navigation is not None
    assert len(navigation.entries) == 12
    article_keys = {entry.article_key for entry in navigation.entries}
    assert {"madde:30", "madde:31"}.issubset(article_keys)
    assert "madde:60" not in article_keys
    assert [entry.article_key for entry in navigation.entries[:4]] == [
        "madde:30",
        "madde:31",
        "madde:29",
        "madde:32",
    ]


def test_navigation_prioritizes_result_references_after_query_overlap() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(
                    position,
                    (
                        "Genel Düzenleme",
                        (
                            "MADDE 4A - Sınır ötesi aktarım yasağı"
                            if position == 4
                            else f"MADDE {position}"
                        ),
                    ),
                )
                for position in range(1, 21)
            ],
            seed_positions=(19, 20),
        ),
        as_of_date=None,
        max_headings=4,
        focused_query="sınır ötesi aktarım yasağı",
        result_references=(
            provision_retrieval.RegulatoryProvisionReference(
                article_no="2",
                qualifier=None,
            ),
        ),
    )

    assert navigation is not None
    assert [entry.article_key for entry in navigation.entries] == [
        "madde:4a",
        "madde:2",
        "madde:19",
        "madde:20",
    ]


def test_navigation_query_overlap_promotes_distant_heading_inside_cap() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(
                    position,
                    (
                        "Genel Düzenleme",
                        (
                            "MADDE 4A - Tehlikeli atık ihracat yasağı"
                            if position == 30
                            else f"MADDE {position}"
                        ),
                    ),
                )
                for position in range(1, 31)
            ],
            seed_positions=(1, 2),
        ),
        as_of_date=None,
        max_headings=3,
        focused_query="4A tehlikeli atık ihracat yasağı",
    )

    assert navigation is not None
    assert navigation.entries[0].article_key == "madde:4a"
    assert len(navigation.entries) == 3


def test_navigation_explicit_query_reference_stays_highest_priority() -> None:
    navigation = provision_retrieval.select_regulatory_provision_navigation(
        _heading_source(
            [
                _heading_candidate(2, ("Genel Düzenleme", "MADDE 2")),
                _heading_candidate(
                    30,
                    (
                        "Genel Düzenleme",
                        "MADDE 13 - Tehlikeli atık ihracat yasağı",
                    ),
                ),
            ],
            seed_positions=(30,),
        ),
        as_of_date=None,
        focused_query="tehlikeli atık ihracat yasağı",
        query_references=(
            provision_retrieval.RegulatoryProvisionReference(
                article_no="2",
                qualifier=None,
            ),
        ),
    )

    assert navigation is not None
    assert [entry.article_key for entry in navigation.entries] == [
        "madde:2",
        "madde:13",
    ]


def test_navigation_rechecks_current_and_historical_visibility() -> None:
    source = _heading_source(
        [
            _heading_candidate(
                1,
                ("Genel Düzenleme", "MADDE 1"),
                status=RegulatoryChunkStatus.SUPERSEDED.value,
                validity_end_date=date(2025, 1, 1),
            ),
            _heading_candidate(
                2,
                ("Genel Düzenleme", "MADDE 2"),
                validity_start_date=date(2025, 1, 1),
            ),
        ],
        seed_positions=(1, 2),
    )

    current = provision_retrieval.select_regulatory_provision_navigation(
        source,
        as_of_date=None,
    )
    historical = provision_retrieval.select_regulatory_provision_navigation(
        source,
        as_of_date=date(2024, 12, 31),
    )

    assert current is not None
    assert [entry.article_key for entry in current.entries] == ["madde:2"]
    assert historical is not None
    assert [entry.article_key for entry in historical.entries] == ["madde:1"]


def test_build_navigation_skips_database_for_non_regulatory_results(
    monkeypatch: MonkeyPatch,
) -> None:
    generic = inference_section_from_single_chunk(_chunk(1, None, document_id="doc"))
    lookup = MagicMock()
    monkeypatch.setattr(
        provision_retrieval,
        "get_regulatory_provision_heading_source",
        lookup,
    )

    assert (
        provision_retrieval.build_regulatory_provision_navigation(
            MagicMock(),
            [generic],
            as_of_date=None,
        )
        is None
    )
    lookup.assert_not_called()


def test_build_navigation_keeps_query_references_separate_from_result_text(
    monkeypatch: MonkeyPatch,
) -> None:
    source = _heading_source(
        [
            _heading_candidate(2, ("Genel Düzenleme", "MADDE 2")),
            _heading_candidate(13, ("Genel Düzenleme", "MADDE 13")),
        ],
        seed_positions=(13,),
    )
    monkeypatch.setattr(
        provision_retrieval,
        "get_regulatory_provision_heading_source",
        MagicMock(return_value=source),
    )
    section = inference_section_from_single_chunk(
        _chunk(13, "seed-13", content="Sonuç metni MADDE 13'e atıf yapar.")
    )

    navigation = provision_retrieval.build_regulatory_provision_navigation(
        MagicMock(),
        [section],
        query="MADDE 2 kapsamında uygulanacak koşullar",
        as_of_date=None,
    )

    assert navigation is not None
    assert [entry.article_key for entry in navigation.entries] == [
        "madde:2",
        "madde:13",
    ]


def test_navigation_payload_is_json_safe_and_marks_leads_as_non_evidence() -> None:
    navigation = provision_retrieval.RegulatoryProvisionNavigation(
        document_title="Düzenleme",
        entries=(
            provision_retrieval.RegulatoryProvisionNavigationEntry(
                article_key="madde:4a",
                heading_label="MADDE 4A - Yükümlülük",
            ),
        ),
    )

    payload = provision_retrieval.regulatory_provision_navigation_payload(navigation)

    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert "not evidence" in payload["usage_note"]
    assert "not evidence that a provision is absent" in payload["usage_note"]
    with pytest.raises(FrozenInstanceError):
        setattr(navigation.entries[0], "heading_label", "changed")
