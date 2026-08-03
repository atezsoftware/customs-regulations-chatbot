import pytest

from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_distinctive_source_hint,
    extract_regulatory_instrument_source_hint,
    extract_regulatory_provision_references,
    extract_regulatory_provision_source_hint,
    extract_regulatory_source_hint,
    extract_single_regulatory_provision_reference,
    normalize_regulatory_heading_path,
    parse_regulatory_article_heading,
    regulatory_heading_path_matches_reference,
    regulatory_provision_heading_phrases,
    regulatory_query_scope_heading_phrases,
)


def test_removes_stale_prior_article_scope() -> None:
    path = [
        "Document title",
        "SECOND PART",
        "MADDE 4 - Earlier rule",
        "Amendment heading",
        "MADDE 9 - Current rule",
        "Second paragraph",
    ]

    assert normalize_regulatory_heading_path(path, article_no="9") == [
        "Document title",
        "SECOND PART",
        "MADDE 9 - Current rule",
        "Second paragraph",
    ]


def test_preserves_current_article_subheadings() -> None:
    path = [
        "Document title",
        "MADDE 4 - Current rule",
        "Special subsection",
        "First paragraph",
    ]

    assert normalize_regulatory_heading_path(path, article_no="4") == path


def test_removes_truncated_nested_unit_leaked_before_current_article() -> None:
    path = [
        "Document title",
        "Authorization chapter",
        "6. A long paragraph from the prior provision was truncated...",
        "MADDE 65 - Current rule",
        "1. Current paragraph",
    ]

    assert normalize_regulatory_heading_path(
        path,
        article_no="65",
        chunk_type="paragraph",
        paragraph_no="1",
    ) == [
        "Document title",
        "Authorization chapter",
        "MADDE 65 - Current rule",
        "1. Current paragraph",
    ]


def test_preserves_short_numbered_scope_before_current_article() -> None:
    path = [
        "Document title",
        "6. Authorization decisions",
        "MADDE 65 - Current rule",
        "1. Current paragraph",
    ]

    assert (
        normalize_regulatory_heading_path(
            path,
            article_no="65",
            chunk_type="paragraph",
            paragraph_no="1",
        )
        == path
    )


def test_removes_exact_boundary_sentence_leaked_before_current_article() -> None:
    leaked_body = (
        "İzin vermeye yetkili gümrük makamı, yeniden değerlendirme sunucunu "
        "izin sahibine bildirir."
    )
    assert len(leaked_body) == 90
    path = [
        "Document title",
        "Authorization chapter",
        f"2. {leaked_body}",
        "MADDE 67 - Suspension",
    ]

    assert normalize_regulatory_heading_path(
        path,
        article_no="67",
        chunk_type="article",
    ) == [
        "Document title",
        "Authorization chapter",
        "MADDE 67 - Suspension",
    ]


def test_preserves_exact_boundary_numbered_scope_without_terminal_sentence() -> None:
    scope_body = "Authorization scope: " + "x" * 68 + ":"
    assert len(scope_body) == 90
    path = [
        "Document title",
        f"6. {scope_body}",
        "MADDE 67 - Suspension",
    ]

    assert (
        normalize_regulatory_heading_path(
            path,
            article_no="67",
            chunk_type="article",
        )
        == path
    )


def test_preserves_paths_without_authoritative_article() -> None:
    path = ["Document title", "ANNEX VII", "Table row"]

    assert normalize_regulatory_heading_path(path, article_no=None) == path


def test_inserts_authoritative_legacy_article_before_numbered_unit() -> None:
    path = [
        "Document title",
        "Guarantee chapter",
        "2. Comprehensive guarantee amount",
    ]

    assert normalize_regulatory_heading_path(
        path,
        article_no="75",
        chunk_type="numbered_section",
    ) == [
        "Document title",
        "Guarantee chapter",
        "MADDE 75",
        "2. Comprehensive guarantee amount",
    ]


def test_appends_authoritative_legacy_article_for_article_chunk() -> None:
    path = ["Document title", "Guarantee chapter"]

    assert normalize_regulatory_heading_path(
        path,
        article_no="MUKERRER 2",
        chunk_type="article",
    ) == ["Document title", "Guarantee chapter", "MÜKERRER MADDE 2"]


def test_inserts_authoritative_article_before_nested_unit_suffix() -> None:
    path = [
        "Document title",
        "Guarantee chapter",
        "2. Comprehensive guarantee amount",
        "(b) Thirty-percent reduction",
    ]

    assert normalize_regulatory_heading_path(
        path,
        article_no="75",
        chunk_type="clause",
        clause_label="b",
    ) == [
        "Document title",
        "Guarantee chapter",
        "MADDE 75",
        "2. Comprehensive guarantee amount",
        "(b) Thirty-percent reduction",
    ]


def test_preserves_enclosing_numbered_scope_above_synthesized_article() -> None:
    path = [
        "Document title",
        "2. GUARANTEES",
        "(b) Thirty-percent reduction",
    ]

    assert normalize_regulatory_heading_path(
        path,
        article_no="75",
        chunk_type="clause",
        clause_label="b",
    ) == [
        "Document title",
        "2. GUARANTEES",
        "MADDE 75",
        "(b) Thirty-percent reduction",
    ]


def test_authoritative_article_repairs_empty_legacy_path() -> None:
    assert normalize_regulatory_heading_path([], article_no="4A") == ["MADDE 4A"]


def test_matches_case_and_diacritic_variants() -> None:
    path = [
        "Document title",
        "GEÇİCİ MADDE 1 - Earlier",
        "MÜKERRER MADDE 2 - Current",
    ]

    assert normalize_regulatory_heading_path(path, article_no="MUKERRER 2") == [
        "Document title",
        "MÜKERRER MADDE 2 - Current",
    ]


@pytest.mark.parametrize(
    "heading",
    ["4A Maddesi:", "4a MADDESİ.", "4A maddesı"],
)
def test_exact_reverse_article_heading_is_structural(heading: str) -> None:
    parsed = parse_regulatory_article_heading(heading)

    assert parsed is not None
    assert parsed.article_no == "4A"
    assert parsed.is_reverse is True


def test_reverse_article_heading_overrides_stale_parent_metadata() -> None:
    path = [
        "Document title",
        "MADDE 4 - Earlier rule",
        "4A Maddesi:",
        "First paragraph",
    ]

    assert normalize_regulatory_heading_path(path, article_no="4") == [
        "Document title",
        "4A Maddesi:",
        "First paragraph",
    ]


def test_reverse_article_reference_prose_is_not_structural() -> None:
    assert (
        parse_regulatory_article_heading("4A maddesi uyarınca işlem yapılır.") is None
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Bazel Sözleşmesi Madde 8 yeniden ithal", ("8", None)),
        ("Basel md4a ihracat yasağı", ("4A", None)),
        ("Convention Article 6 notification", ("6", None)),
        ("4A maddesi kapsamındaki yasak", ("4A", None)),
        ("Geçici Madde 2 izin", ("2", "gecici")),
        ("Mükerrer Madde 3 yükümlülük", ("3", "mukerrer")),
        ("8. madde yeniden ithal", ("8", None)),
        ("8'inci madde yeniden ithal", ("8", None)),
        ("8 inci madde yeniden ithal", ("8", None)),
        ("8'inci maddede belirtilen koşullar", ("8", None)),
        ("8 inci maddenin uygulanması", ("8", None)),
        ("4A maddesinde öngörülen yasak", ("4A", None)),
        ("6. maddeden doğan sonuç", ("6", None)),
        ("Transit Rejimi Tebliği Seri No 4 Madde 12", ("12", None)),
        ("Transit Rejimi Tebliği Seri No 14 Madde 9", ("9", None)),
    ],
)
def test_extracts_one_explicit_query_provision(
    query: str,
    expected: tuple[str, str | None],
) -> None:
    assert extract_single_regulatory_provision_reference(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Madde 6 ile Madde 8 karşılaştırması",
        "ADR 8.2.2.8 sürücü eğitimi",
        "Madde 8.2.2.8 sürücü eğitimi",
        "yeniden ithal yükümlülüğü",
        "Transit Rejimi Tebliği Seri No 4 madde hükümleri",
    ],
)
def test_query_provision_extractor_rejects_ambiguous_or_non_article_text(
    query: str,
) -> None:
    assert extract_single_regulatory_provision_reference(query) is None


def test_extracts_ordered_deduplicated_references_from_clean_chunk_text() -> None:
    assert extract_regulatory_provision_references(
        "Madde 4A kapsamı, Article 6 ve yeniden Madde 4A ile birlikte "
        "Geçici Madde 2 uyarınca değerlendirilir."
    ) == (
        RegulatoryProvisionReference(article_no="4A", qualifier=None),
        RegulatoryProvisionReference(article_no="6", qualifier=None),
        RegulatoryProvisionReference(article_no="2", qualifier="gecici"),
    )


@pytest.mark.parametrize(
    "text",
    [
        "Ek VII ve paragraf 4 birlikte uygulanır.",
        "ADR 8.2.2.8 eğitim şartıdır.",
        "14.03.2026 tarihinde 2207.10 kodu kullanıldı.",
    ],
)
def test_reference_extractor_rejects_non_article_numbers(text: str) -> None:
    assert extract_regulatory_provision_references(text) == ()


def test_query_provision_phrases_cover_compact_and_reverse_headings() -> None:
    phrases = regulatory_provision_heading_phrases(
        RegulatoryProvisionReference(article_no="4A", qualifier=None)
    )

    assert "MADDE 4A" in phrases
    assert "MADDE4A" in phrases
    assert "4A MADDESİ" in phrases
    assert "ARTICLE 4A" in phrases


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Ortak Transit Sözleşmesi Ek IV Madde 15 zamanaşımı",
            ("EK IV", "ANNEX IV"),
        ),
        (
            "Transit Rejimi Tebliği Seri No: 4 Madde 12 teminat",
            ("SERİ NO 4", "SERI NO 4", "SERIES NO 4"),
        ),
        (
            "Transit Rules Series No. 14 Article 9 authorization",
            ("SERİ NO 14", "SERI NO 14", "SERIES NO 14"),
        ),
        ("Basel Sözleşmesi Madde 8", ()),
    ],
)
def test_extracts_explicit_structural_scope_as_soft_ranking_phrases(
    query: str,
    expected: tuple[str, ...],
) -> None:
    assert regulatory_query_scope_heading_phrases(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Basel Sözleşmesi Madde 8 yeniden ithal", "basel sozlesmesi"),
        ("Ortak Transit Sözleşmesi Article 112 borç", "ortak transit sozlesmesi"),
        ("Basel md4a ihracat yasağı", "basel"),
        ("Madde 8 Basel yeniden ithal", None),
        ("Madde 6 ve Madde 8", None),
    ],
)
def test_extracts_bounded_source_hint_before_explicit_provision(
    query: str,
    expected: str | None,
) -> None:
    assert extract_regulatory_provision_source_hint(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Basel Sözleşmesi yasadışı trafik tanımı", "Basel Sözleşmesi"),
        ("Bazel sözleşmesinde geri alma", "Bazel sözleşmesinde"),
        (
            "Ortak Transit Rejimine İlişkin Sözleşme borç tahsili",
            "Ortak Transit Rejimine İlişkin Sözleşme",
        ),
        ("Basel Convention illegal traffic", "Basel Convention"),
        ("Madde 9 Basel Sözleşmesi geri alma", "Basel Sözleşmesi"),
        ("Gümrük Kanunu Madde 184 borç", "Gümrük Kanunu"),
        (
            "Kurtarılan yükün gönderimi Basel Sözleşmesi bakımından mümkün mü",
            "Basel Sözleşmesi",
        ),
        (
            "Olay Ortak Transit Rejimine İlişkin Sözleşme bakımından incelensin",
            "Ortak Transit Rejimine İlişkin Sözleşme",
        ),
    ],
)
def test_extracts_named_instrument_hint_without_requiring_provision(
    query: str,
    expected: str,
) -> None:
    assert extract_regulatory_instrument_source_hint(query) == expected
    assert extract_regulatory_source_hint(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Sözleşme Madde 8",
        "bu sözleşme Madde 8",
        "yeniden ithal yükümlülüğü",
        "Basel Sözleşmesi ve Ortak Transit Sözleşmesi karşılaştırması",
    ],
)
def test_instrument_hint_rejects_anonymous_absent_or_multiple_sources(
    query: str,
) -> None:
    assert extract_regulatory_instrument_source_hint(query) is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Basel Convention Article 8", "Basel"),
        ("Basel Sözleşmesi Madde 8", "Basel"),
        ("Ortak Transit Sözleşmesi borç tahsili", "Ortak Transit"),
        ("Gümrük Kanunu Madde 184", "Gümrük"),
        ("ADR Madde 8", "adr"),
        ("bu sözleşme Madde 8", None),
    ],
)
def test_extracts_distinctive_source_terms_without_legal_form(
    query: str,
    expected: str | None,
) -> None:
    assert extract_regulatory_distinctive_source_hint(query) == expected


def test_source_hint_keeps_acronym_fallback_for_explicit_provision() -> None:
    assert extract_regulatory_source_hint("ADR Madde 8 sürücü eğitimi") == "adr"


def test_structural_path_match_uses_first_article_anchor() -> None:
    reference = RegulatoryProvisionReference(article_no="5", qualifier=None)

    assert not regulatory_heading_path_matches_reference(
        [
            "Bazel Sözleşmesi",
            "MADDE 13 - Bilgi Aktarımı",
            "MADDE 5 uyarınca tayin edilen makamlar",
        ],
        reference,
    )
    assert regulatory_heading_path_matches_reference(
        ["Bazel Sözleşmesi", "MADDE 5", "(1)"],
        reference,
    )
