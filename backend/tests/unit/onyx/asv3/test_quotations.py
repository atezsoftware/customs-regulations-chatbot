from onyx.asv3.llm_adapter import QuotationVerification
from onyx.asv3.publication import publication_gap
from onyx.asv3.quotations import unmatched_quoted_terms
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported


def test_literal_mismatch_cannot_be_approved_by_generic_supported_status() -> None:
    ledger, _context = original_ledger()
    answer = 'Belge adı "Different source title" olarak seçilir [1].'
    terms = unmatched_quoted_terms(answer, "", ledger)
    assert terms[0]["term"] == "Different source title"
    review = supported([1])
    gap = publication_gap(
        answer, review, ["question"], ledger, require_quotation_checks=True
    )
    assert gap is not None and gap.data["unmatched_quoted_terms"]
    review.quotation_checks = [
        QuotationVerification(
            term_id="qt0",
            kind="literal",
            evidence_number=1,
            source_quote="Complete source 1",
            explanation="Literal supported.",
        )
    ]
    # Even an explicit but wrong literal approval cannot change the original wording.
    assert (
        publication_gap(
            answer, review, ["question"], ledger, require_quotation_checks=True
        )
        is not None
    )


def test_translation_needs_a_witness_from_its_own_inline_original() -> None:
    ledger, _context = original_ledger()
    answer = 'Kaynak koşulu çevirisi "koşul ve istisna" olarak açıklanır [1].'
    review = supported([1])
    review.quotation_checks = [
        QuotationVerification(
            term_id="qt0",
            kind="translation",
            evidence_number=1,
            source_quote="condition AND exception",
            explanation="Turkish translation preserves both requirements.",
        )
    ]
    assert (
        publication_gap(
            answer, review, ["question"], ledger, require_quotation_checks=True
        )
        is None
    )
    review.quotation_checks[0].source_quote = "Invented quotation."
    assert (
        publication_gap(
            answer, review, ["question"], ledger, require_quotation_checks=True
        )
        is not None
    )
    review.quotation_checks[0].source_quote = "condition AND exception"
    review.quotation_checks[0].evidence_number = 2
    assert (
        publication_gap(
            answer, review, ["question"], ledger, require_quotation_checks=True
        )
        is not None
    )


def test_original_phrases_and_scenario_quotes_need_no_extra_check() -> None:
    ledger, _context = original_ledger()
    answer = 'Metinde "condition AND exception" bulunur [1].\n\nOlaydaki iddia "user statement" şeklindedir [2].'
    assert (
        unmatched_quoted_terms(answer, 'Kullanıcı "user statement" diyor.', ledger)
        == []
    )
    assert (
        publication_gap(
            answer,
            supported([1, 2]),
            ["question"],
            ledger,
            require_quotation_checks=True,
            scenario='Kullanıcı "user statement" diyor.',
        )
        is None
    )
