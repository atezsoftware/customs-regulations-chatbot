"""Numbered non-statutory references cannot consume an inner statute identity."""

import pytest

from onyx.asv3.authority import native_named_authority_gap, statute_references
from onyx.asv3.models import RunContext
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original


@pytest.mark.parametrize(
    ("prefix", "legacy_number"),
    [
        ("Mahkemenin E.2024/37, K.2025/81 sayılı kararı ile ", "81"),
        ("6789 sayılı Cumhurbaşkanı Kararı ile ", "6789"),
        ("8765 sayılı Uygulama Yönetmeliği ve ", "8765"),
    ],
)
def test_numbered_reference_rescans_actual_inner_statute(
    prefix: str, legacy_number: str
) -> None:
    answer = prefix + "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uygulanır [1]."
    ledger = ledger_with(original("8917 sayılı Faaliyet Kanunu", "27"))

    references = statute_references(answer, strict_reference_boundaries=True)
    assert [(reference.number, reference.article) for reference in references] == [
        ("8917", "27")
    ]
    assert (
        native_named_authority_gap(answer, ledger, strict_reference_boundaries=True)
        is None
    )

    # The opt-in does not alter the existing normal/deep reference parser.
    assert statute_references(answer) == statute_references(
        answer, strict_reference_boundaries=False
    )
    baseline = native_named_authority_gap(answer, ledger)
    assert baseline is not None
    gaps = baseline["named_authority_gaps"]
    assert isinstance(gaps, list)
    assert any(
        isinstance(gap, dict) and gap["instrument_number"] == legacy_number
        for gap in gaps
    )


def test_inner_statute_still_needs_its_own_adjacent_original() -> None:
    answer = (
        "Mahkemenin K.2025/81 sayılı kararı ile 8917 sayılı Faaliyet Kanunu'nun "
        "27. maddesi uygulanır [1]."
    )
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ", text=answer))
    gap = native_named_authority_gap(answer, ledger, strict_reference_boundaries=True)
    assert gap is not None
    entries = gap["named_authority_gaps"]
    assert isinstance(entries, list) and entries
    assert all(
        isinstance(entry, dict)
        and entry["instrument_number"] == "8917"
        and entry["article"] == "27"
        and entry["matching_original_evidence"] == []
        for entry in entries
    )

    ledger.add([original("8917 sayılı Faaliyet Kanunu", "27")], RunContext())
    elsewhere = answer + "\n\nBaşvuru ayrıca değerlendirilir [2]."
    assert native_named_authority_gap(
        elsewhere, ledger, strict_reference_boundaries=True
    )
    assert (
        native_named_authority_gap(
            answer.replace("[1]", "[1][2]"),
            ledger,
            strict_reference_boundaries=True,
        )
        is None
    )


def test_court_case_suffix_is_not_a_numbered_law_identity() -> None:
    answer = (
        "Mahkemenin E.2024/37, K.2025/81 sayılı kararı ile Faaliyet Kanunu'nun "
        "27. maddesi incelenmiştir [1]."
    )
    ledger = ledger_with(original("8917 sayılı Faaliyet Kanunu", "27"))
    assert statute_references(answer, strict_reference_boundaries=True) == ()
    assert (
        native_named_authority_gap(answer, ledger, strict_reference_boundaries=True)
        is None
    )


def test_legitimate_statute_title_can_contain_decision_word() -> None:
    answer = (
        "8917 sayılı Yargı Kararlarının Yerine Getirilmesi Kanunu'nun "
        "27. maddesi uygulanır [1]."
    )
    ledger = ledger_with(
        original("8917 sayılı Yargı Kararlarının Yerine Getirilmesi Kanunu", "27")
    )
    assert statute_references(answer, strict_reference_boundaries=True) == (
        statute_references(answer)
    )
    assert (
        native_named_authority_gap(answer, ledger, strict_reference_boundaries=True)
        is None
    )


def test_separate_statutes_keep_distinct_local_article_requirements() -> None:
    answer = (
        "8917 sayılı Faaliyet Kanunu m. 27 [1] ve 7251 sayılı Veri Kanunu m. 45 [2] "
        "uyarınca işlem yapılır."
    )
    ledger = ledger_with(
        original("8917 sayılı Faaliyet Kanunu", "27"),
        original("7251 sayılı Veri Kanunu", "45"),
    )
    assert (
        native_named_authority_gap(answer, ledger, strict_reference_boundaries=True)
        is None
    )
    assert native_named_authority_gap(
        answer.replace("[2]", "[1]"), ledger, strict_reference_boundaries=True
    )
    assert statute_references(answer, strict_reference_boundaries=True) == (
        statute_references(answer)
    )


def test_unrelated_lower_source_reference_creates_no_answer_obligation() -> None:
    ledger = ledger_with(
        original(
            "Uygulama Tebliği",
            "3",
            kind="tebliğ",
            text=(
                "6789 sayılı Cumhurbaşkanı Kararı ile 8917 sayılı Faaliyet "
                "Kanunu'nun 27. maddesine bakınız."
            ),
        )
    )
    assert (
        native_named_authority_gap(
            "Uygulama Tebliği'nin 3. maddesi kapsamında belge verilir [1].",
            ledger,
            strict_reference_boundaries=True,
        )
        is None
    )


def test_english_numbered_reference_remains_unchanged() -> None:
    text = "Act no. 8917 article 27 requires an application."
    assert statute_references(text, strict_reference_boundaries=True) == (
        statute_references(text)
    )
