"""Exercise omissions that a broad topic-level approval previously concealed."""

import pytest

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    DeterminationVerification,
    QuestionVerification,
    VerificationResult,
)
from onyx.asv3.models import EvidenceItem, RunContext, ToolOutcome
from onyx.asv3.publication import publication_gap
from onyx.asv3.scenario import question_determinations
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc

QUESTION = "Is the new authorization exempt? Can the earlier payment be refunded?"
ANSWER = (
    "- New authorization requires a fee [1].\n- Earlier payments may be refunded [2]."
)
ORIGINALS = {
    1: "New authorization requires a fee.",
    2: "Earlier payments may be refunded.",
}


@pytest.fixture
def ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    for number, text in ORIGINALS.items():
        source = f"source-{number}"
        ledger.add(
            [
                EvidenceItem(
                    source_id=source,
                    text=text,
                    search_doc=SearchDoc(
                        document_id=source,
                        chunk_ind=0,
                        semantic_identifier=source,
                        blurb=text,
                        source_type=DocumentSource.USER_FILE,
                        boost=0,
                        hidden=False,
                        metadata={},
                        match_highlights=[],
                    ),
                )
            ],
            RunContext(),
        )
    return ledger


def approval(answer: str = ANSWER) -> VerificationResult:
    units = assertion_inventory(answer)
    return VerificationResult(
        status="supported",
        explanation="Each requested outcome has its own operative support.",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=[1, 2],
        safe_to_publish=True,
        assertion_results=[
            AssertionVerification(
                unit_id=unit["unit_id"],
                status="supported",
                explanation="Its own original supports the outcome.",
                witnesses=[
                    AssertionWitness(citation=n, source_quote=ORIGINALS[n])
                    for n in unit["evidence_numbers"]
                ],
            )
            for unit in units
        ],
        question_results=[
            QuestionVerification(
                question_id="q0",
                status="supported",
                evidence_numbers=[1, 2],
                missing_conditions=[],
                determinations=[
                    DeterminationVerification(
                        determination_id=part["determination_id"],
                        status="supported",
                        answer_unit_ids=[units[index]["unit_id"]],
                        evidence_numbers=[index + 1],
                        missing_conditions=[],
                    )
                    for index, part in enumerate(question_determinations([QUESTION]))
                ],
            )
        ],
    )


def check(
    answer: str,
    review: VerificationResult,
    ledger: EvidenceLedger,
    *,
    allow_explicit_gaps: bool = False,
) -> ToolOutcome | None:
    return publication_gap(
        answer,
        review,
        [QUESTION],
        ledger,
        scenario="Twenty items cost five units each.",
        require_assertion_checks=True,
        require_determination_checks=True,
        allow_explicit_gaps=allow_explicit_gaps,
    )


def test_uncited_legal_paragraph_cannot_disappear_from_a_supported_topic(
    ledger: EvidenceLedger,
) -> None:
    answer = ANSWER + "\n\nA free invoice eliminates all other charges."
    review = approval(answer)
    gap = check(answer, review, ledger)
    assert gap is not None
    gaps = gap.data["assertion_gaps"]
    assert isinstance(gaps, list) and len(gaps) == 1
    assert isinstance(gaps[0], dict) and "free invoice" in str(gaps[0]["text"])


def test_missing_independent_part_is_not_approved_by_the_parent(
    ledger: EvidenceLedger,
) -> None:
    review = approval()
    review.question_results[0].determinations.pop()
    gap = check(ANSWER, review, ledger)
    assert gap is not None and "determination_gaps" in gap.data


def test_independent_part_cannot_borrow_another_blocks_citations(
    ledger: EvidenceLedger,
) -> None:
    review = approval()
    review.question_results[0].determinations[0].answer_unit_ids = [
        assertion_inventory(ANSWER)[1]["unit_id"]
    ]
    gap = check(ANSWER, review, ledger)
    assert gap is not None
    gaps = gap.data["determination_gaps"]
    assert isinstance(gaps, list) and len(gaps) == 1
    assert isinstance(gaps[0], dict) and gaps[0]["determination_id"] == "q0:d0"


def test_supported_parent_cannot_hide_negative_sibling_assessment(
    ledger: EvidenceLedger,
) -> None:
    review = approval()
    review.question_results[0].determinations[1].status = "incomplete"
    review.question_results[0].determinations[1].missing_conditions = [
        "The relevant refund procedure was not examined."
    ]
    gap = check(ANSWER, review, ledger)
    assert gap is not None and "determination_gaps" in gap.data


def test_presentation_and_literal_facts_do_not_create_legal_research(
    ledger: EvidenceLedger,
) -> None:
    answer = ANSWER + "\n\n### Given figures\n\nThe stated total is one hundred."
    review = approval(answer)
    heading, facts = review.assertion_results[-2:]
    heading.basis = "presentation"
    facts.basis = "scenario"
    facts.scenario_quotes = ["Twenty items cost five units each."]
    assert check(answer, review, ledger) is None
    facts.scenario_quotes = ["The user said there is no duty."]
    assert check(answer, review, ledger) is not None


def test_uncited_legal_prose_cannot_be_laundered_as_presentation(
    ledger: EvidenceLedger,
) -> None:
    answer = ANSWER + "\n\nThe new authorization is also exempt from every surcharge."
    review = approval(answer)
    review.assertion_results[-1].basis = "presentation"
    assert check(answer, review, ledger) is not None


def test_legal_heading_still_needs_its_original_when_assessed_as_a_rule(
    ledger: EvidenceLedger,
) -> None:
    answer = ANSWER + "\n\n### All fees are automatically refunded"
    assert check(answer, approval(answer), ledger) is not None


def test_precise_partial_answer_preserves_supported_part(
    ledger: EvidenceLedger,
) -> None:
    answer = "New authorization requires a fee [1].\n\nThe earlier payment's refund procedure was not established."
    units = assertion_inventory(answer)
    review = approval()
    review.status = "incomplete"
    review.missing_conditions = ["refund procedure"]
    review.evidence_numbers = [1]
    review.assertion_results = [
        AssertionVerification(
            unit_id=units[0]["unit_id"],
            status="supported",
            explanation="Operative original.",
            witnesses=[AssertionWitness(citation=1, source_quote=ORIGINALS[1])],
        ),
        AssertionVerification(
            unit_id=units[1]["unit_id"],
            status="uncertain",
            basis="evidence_gap",
            missing_conditions=["refund procedure"],
            explanation="Disclosed unresolved procedure, without an asserted answer.",
        ),
    ]
    parent = review.question_results[0]
    parent.status = "incomplete"
    parent.missing_conditions = ["refund procedure"]
    parent.evidence_numbers = [1]
    parent.determinations[0].answer_unit_ids = [units[0]["unit_id"]]
    parent.determinations[1].status = "incomplete"
    parent.determinations[1].answer_unit_ids = []
    parent.determinations[1].evidence_numbers = []
    parent.determinations[1].missing_conditions = ["refund procedure"]
    assert check(answer, review, ledger, allow_explicit_gaps=True) is None
    assert check(answer, review, ledger) is not None
