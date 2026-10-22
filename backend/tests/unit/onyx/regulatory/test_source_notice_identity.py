from datetime import date

import pytest

from onyx.regulatory.source_notice_identity import (
    NoticeIdentity,
    normalize_notice_alias,
    notice_identity_queries,
)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("2012/7 sayılı Genelge", ["circular:2012:7"]),
        ("2012/07 Genelge", ["circular:2012:7"]),
        ("2012/7 sayılı Genelgenin", ["circular:2012:7"]),
        ("2012/7 sayılı Gümrük Genelgesi", ["circular:2012:7"]),
        ("2012/7 sayılı Özel Tüketim Vergisi Tebliği", ["communique:2012:7"]),
        ("Genelge (2012/7 - 01.01.2012)", ["circular:2012:7"]),
        ("genelge_2012-7_tasima", ["circular:2012:7"]),
        ("2024/9366 sayılı Karar", ["decision:2024:9366"]),
        ("BKK 1986-11318", ["decision:1986:11318"]),
        ("2012/7 sayılı Tebliğ", ["communique:2012:7"]),
        ("2908 sayılı Dernekler Kanunu", ["law:2908"]),
        ("765 sayılı Türk Ceza Kanununun 1. maddesi", ["law:765"]),
        ("7 sayılı Genelgenin Kanuna atfı", []),
        ("Kanun (2908 - 01.01.1983)", ["law:2908"]),
        (
            "10.03.2022 tarihli 72720073 sayılı yazı",
            ["administrative_letter:2022-03-10:72720073"],
        ),
        ("31.02.2022 tarihli 72720073 sayılı yazı", []),
        ("Sadece 2012/7 veya 2908", []),
        ("7 sayılı Genelge ile Kanun değişikliği", []),
        (
            "2012/7 sayılı Genelge; 2012/7 sayılı Karar",
            ["circular:2012:7", "decision:2012:7"],
        ),
        (
            "circular:2012:7 decision:2024:9366 law:2908 administrative_letter:2022-03-10:72720073",
            [
                "circular:2012:7",
                "decision:2024:9366",
                "law:2908",
                "administrative_letter:2022-03-10:72720073",
            ],
        ),
    ],
)
def test_strict_typed_reference_identity(reference: str, expected: list[str]) -> None:
    assert notice_identity_queries(reference) == expected


def test_law_and_same_named_successor_are_distinct() -> None:
    assert NoticeIdentity("law", "2908").key != NoticeIdentity("law", "5253").key


def test_date_is_part_of_letter_identity() -> None:
    first = NoticeIdentity(
        "administrative_letter", "123", instrument_date=date(2020, 1, 1)
    )
    second = NoticeIdentity(
        "administrative_letter", "123", instrument_date=date(2021, 1, 1)
    )
    assert first.key != second.key
    assert notice_identity_queries(first.aliases[1]) == [first.key]


def test_filename_folding_preserves_tokens() -> None:
    assert (
        normalize_notice_alias("KANUNLAR/Türk_Vatandaşlığı_Kanunu.md")
        == "kanunlar turk vatandasligi kanunu md"
    )
