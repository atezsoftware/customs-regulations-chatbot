from onyx.regulatory.exact_search_fields import extract_legal_exact_fields


def test_extracts_turkish_legal_identifiers() -> None:
    fields = extract_legal_exact_fields(
        "Geçici Madde 2, 2024/17 sayılı karar, 06.08.2026"
    )

    assert fields.provision_identifiers == ["geçici madde 2"]
    assert fields.decision_numbers == ["2024/17"]
    assert fields.legal_dates == ["2026-08-06"]


def test_extracts_metadata_and_deduplicates_in_source_order() -> None:
    fields = extract_legal_exact_fields(
        "MADDE 4/A uyarınca 2023/9 sayılı karar uygulanır.",
        "article_no: Madde 4/A; karar no: 2023/9; tarih: 1/2/2024",
    )

    assert fields.provision_identifiers == ["madde 4/a"]
    assert fields.decision_numbers == ["2023/9"]
    assert fields.legal_dates == ["2024-02-01"]


def test_extracts_abbreviated_m_provisions_with_optional_spacing() -> None:
    fields = extract_legal_exact_fields("m. 5 ve m.6 hükümleri birlikte uygulanır.")

    assert fields.provision_identifiers == ["madde 5", "madde 6"]


def test_invalid_calendar_dates_are_not_indexed() -> None:
    fields = extract_legal_exact_fields(
        "Madde 7; 2022/31 sayılı karar; 31.02.2024 ve 2024-13-01"
    )

    assert fields.provision_identifiers == ["madde 7"]
    assert fields.decision_numbers == ["2022/31"]
    assert fields.legal_dates == []
