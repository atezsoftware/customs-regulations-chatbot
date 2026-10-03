"""Substantive negative review survives transport without becoming approval."""

import json

import pytest

from onyx.asv3.assertions import AssertionVerification
from onyx.asv3.llm_adapter import QuotationVerification, ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.publication import publication_gap
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response


def test_uncertain_negative_assessment_retains_full_explanation_without_repair() -> (
    None
):
    llm = scripted_model()
    explanation = "This passage cannot establish the stated consequence. " * 30
    body = {
        "status": "incomplete",
        "safe_to_publish": False,
        "explanation": "The stated consequence does not follow from its original.",
        "required_conditions": [],
        "missing_conditions": ["Find the operative consequence or correct this claim."],
        "unsupported_claims": ["The proposed automatic consequence"],
        "evidence_numbers": [1],
        "question_results": [],
        "need_results": [],
        "assertion_results": [
            {
                "unit_id": "au0",
                "status": "uncertain",
                "witnesses": [],
                "missing_conditions": ["Operative effect unresolved"],
                "explanation": explanation,
            }
        ],
        "omitted_material_source_details": [],
        "quotation_checks": [
            {
                "term_id": "qt0",
                "kind": "uncertain",
                "evidence_number": 1,
                "source_quote": "condition AND exception",
                "explanation": "The application may change the operative meaning.",
            }
        ],
    }
    llm.invoke.return_value = text_response(body)
    result = ResearchModel(llm, RunContext()).invoke_verification(
        "Assess originals",
        json.dumps({"claim": "Consequence", "assertion_units": [{"unit_id": "au0"}]}),
    )
    assert llm.invoke.call_count == 1
    assert result.format_error is None
    assert not result.safe_to_publish
    assert result.assertion_results[0].explanation == explanation
    assert result.quotation_checks[0].kind == "uncertain"
    assert result.unsupported_claims == body["unsupported_claims"]
    assert result.missing_conditions == body["missing_conditions"]


def test_uncertain_quotation_cannot_authorize_publication_with_a_literal_witness() -> (
    None
):
    ledger, _context = original_ledger()
    answer = 'Kaynak "değişmiş bir koşul" getirir [1].'
    review = supported([1])
    review.quotation_checks = [
        QuotationVerification(
            term_id="qt0",
            kind="uncertain",
            evidence_number=1,
            source_quote="condition AND exception",
            explanation="The asserted application has not been established.",
        )
    ]
    gap = publication_gap(
        answer, review, ["question"], ledger, require_quotation_checks=True
    )
    assert gap is not None
    assert gap.data["unmatched_quoted_terms"]


def test_unknown_quotation_category_and_unbounded_explanation_remain_invalid() -> None:
    with pytest.raises(ValueError):
        QuotationVerification.model_validate(
            {
                "term_id": "qt0",
                "kind": "maybe_approved",
                "evidence_number": 1,
                "source_quote": "condition",
                "explanation": "No supported classification",
            }
        )
    with pytest.raises(ValueError):
        AssertionVerification(unit_id="au0", status="uncertain", explanation="x" * 4001)
