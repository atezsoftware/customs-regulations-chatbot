import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import QuotationVerification
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.publication import publication_gap
from onyx.asv3.quotations import unmatched_quoted_terms
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported


def test_verified_canonical_heading_is_literal_context_without_becoming_a_rule() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    label = "OPERATIVE SCHEDULE"
    ledger.add(
        [
            EvidenceItem(
                source_id="source",
                chunk_id="clause",
                text="a) Category A.",
                metadata={
                    "heading_path": [label, "a) Category A."],
                    "canonical_metadata": {
                        "document_id": "source",
                        "regulatory_chunk_id": "clause",
                        "heading_path": [label, "a) Category A."],
                    },
                },
            )
        ],
        context,
    )
    assert (
        unmatched_quoted_terms('Listed under "Operative Schedule" [1].', "", ledger)
        == []
    )
    assert unmatched_quoted_terms('The rule is "Category A is exempt" [1].', "", ledger)


@pytest.mark.parametrize("defect", ["source", "chunk", "path", "derived", "truncated"])
def test_unbound_or_incomplete_heading_cannot_approve_quoted_wording(
    defect: str,
) -> None:
    context, ledger = RunContext(), EvidenceLedger()
    label = "Operative Schedule"
    canonical: dict[str, JsonValue] = {
        "document_id": "source",
        "regulatory_chunk_id": "clause",
        "heading_path": [label],
    }
    if defect == "source":
        canonical["document_id"] = "other"
    elif defect == "chunk":
        canonical["regulatory_chunk_id"] = "other"
    elif defect == "path":
        canonical["heading_path"] = ["Other Schedule"]
    elif defect == "truncated":
        label += "..."
        canonical["heading_path"] = [label]
    ledger.add(
        [
            EvidenceItem(
                source_id="source",
                chunk_id="clause",
                text="a) Category A.",
                metadata={
                    "heading_path": [label],
                    "canonical_metadata": canonical,
                    "derived": defect == "derived",
                },
            )
        ],
        context,
    )
    assert unmatched_quoted_terms('Listed under "Operative Schedule" [1].', "", ledger)


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
