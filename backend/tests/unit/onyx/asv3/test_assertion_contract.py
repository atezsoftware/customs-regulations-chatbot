"""A broad approval cannot substitute for source-specific assertion checks."""

import pytest

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import QuestionVerification, VerificationResult
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.publication import publication_gap
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def original_ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    for source_id, text in (
        ("permission", "Temporary release requires security."),
        ("settlement", "The office sends the result to the authority."),
    ):
        ledger.add(
            [
                EvidenceItem(
                    source_id=source_id,
                    text=text,
                    search_doc=SearchDoc(
                        document_id=source_id,
                        chunk_ind=0,
                        semantic_identifier=source_id,
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


def broad_approval() -> VerificationResult:
    return VerificationResult(
        status="supported",
        explanation="The topic is discussed in the supplied originals.",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=[1, 2],
        safe_to_publish=True,
        question_results=[
            QuestionVerification(
                question_id="q0",
                status="supported",
                evidence_numbers=[1, 2],
                missing_conditions=[],
            )
        ],
    )


ANSWER = "- Temporary release requires security [1].\n- Security is automatically returned on notification [2]."


def checks() -> list[AssertionVerification]:
    units = assertion_inventory(ANSWER)
    return [
        AssertionVerification(
            unit_id=str(units[0]["unit_id"]),
            status="supported",
            witnesses=[
                AssertionWitness(
                    citation=1, source_quote="Temporary release requires security."
                )
            ],
            explanation="Operative permission and prerequisite.",
        ),
        AssertionVerification(
            unit_id=str(units[1]["unit_id"]),
            status="unsupported",
            witnesses=[],
            missing_conditions=[
                "Notification does not establish automatic settlement."
            ],
            explanation="This source only requires notification.",
        ),
    ]


def test_broad_approval_without_local_assessments_cannot_publish() -> None:
    gap = publication_gap(
        ANSWER,
        broad_approval(),
        ["Release and later settlement?"],
        original_ledger(),
        require_assertion_checks=True,
    )
    assert gap is not None
    gaps = gap.data["assertion_gaps"]
    assert isinstance(gaps, list) and len(gaps) == 2


def test_local_unsupported_outcome_overrides_supported_question_and_need_flags() -> (
    None
):
    review = broad_approval().model_copy(update={"assertion_results": checks()})
    gap = publication_gap(
        ANSWER,
        review,
        ["question"],
        original_ledger(),
        require_assertion_checks=True,
        allow_explicit_gaps=True,
    )
    assert gap is not None
    gaps = gap.data["assertion_gaps"]
    assert isinstance(gaps, list) and isinstance(gaps[0], dict)
    assert str(gaps[0]["text"]).endswith("notification [2].")


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_inline_source",
        "invented_quote",
        "stale_unit",
        "duplicate_unit",
        "missing_condition",
        "missing_witness",
    ],
)
def test_invalid_witness_or_block_identity_cannot_become_approval(defect: str) -> None:
    answer = "Temporary release requires security [1]."
    unit = assertion_inventory(answer)[0]
    check = AssertionVerification(
        unit_id=str(unit["unit_id"]),
        status="supported",
        witnesses=[
            AssertionWitness(
                citation=1, source_quote="Temporary release requires security."
            )
        ],
        explanation="Original operative wording.",
    )
    if defect == "wrong_inline_source":
        check.witnesses = [
            AssertionWitness(
                citation=2, source_quote="The office sends the result to the authority."
            )
        ]
    elif defect == "invented_quote":
        check.witnesses[0].source_quote = "Security is automatically returned."
    elif defect == "stale_unit":
        check.unit_id = str(
            assertion_inventory("Different conclusion [1].")[0]["unit_id"]
        )
    elif defect == "missing_condition":
        check.missing_conditions = ["Required permission remains unproved."]
    elif defect == "missing_witness":
        check.witnesses = []
    review = broad_approval().model_copy(
        update={
            "evidence_numbers": [1],
            "question_results": [
                QuestionVerification(
                    question_id="q0",
                    status="supported",
                    evidence_numbers=[1],
                    missing_conditions=[],
                )
            ],
            "assertion_results": [check, check]
            if defect == "duplicate_unit"
            else [check],
        }
    )
    assert (
        publication_gap(
            answer,
            review,
            ["question"],
            original_ledger(),
            require_assertion_checks=True,
        )
        is not None
    )


def test_every_current_block_can_use_its_own_literal_originals() -> None:
    answer = "- Temporary release requires security [1].\n- The office sends the result to the authority [2]."
    units = assertion_inventory(answer)
    review = broad_approval().model_copy(
        update={
            "assertion_results": [
                AssertionVerification(
                    unit_id=str(unit["unit_id"]),
                    status="supported",
                    witnesses=[AssertionWitness(citation=number, source_quote=quote)],
                    explanation="Own operative original.",
                )
                for unit, number, quote in zip(
                    units,
                    [1, 2],
                    [
                        "Temporary release requires security.",
                        "The office sends the result to the authority.",
                    ],
                )
            ]
        }
    )
    assert (
        publication_gap(
            answer,
            review,
            ["question"],
            original_ledger(),
            require_assertion_checks=True,
        )
        is None
    )


def test_missing_block_stays_a_gap_and_grouped_inline_sources_all_need_witnesses() -> (
    None
):
    answer = "Combined procedure [1, 2]."
    unit = assertion_inventory(answer)[0]
    review = broad_approval().model_copy(
        update={
            "assertion_results": [
                AssertionVerification(
                    unit_id=str(unit["unit_id"]),
                    status="supported",
                    witnesses=[
                        AssertionWitness(
                            citation=1,
                            source_quote="Temporary release requires security.",
                        )
                    ],
                    explanation="Only one inline original was assessed.",
                )
            ]
        }
    )
    assert (
        publication_gap(
            answer,
            review,
            ["question"],
            original_ledger(),
            require_assertion_checks=True,
        )
        is not None
    )


@pytest.mark.parametrize(
    ("original", "scenario", "rejected"),
    [
        ("Dangerous goods use compliant vehicles.", "", True),
        ("UN 1170 is subject to the applicable vehicle requirements.", "", False),
        ("UN-1170 is subject to the applicable vehicle requirements.", "", False),
        (
            "Dangerous goods use compliant vehicles.",
            "The consignment is UN1170.",
            False,
        ),
        ("UN11700 has different requirements.", "", True),
    ],
)
def test_technical_code_requires_its_literal_inline_original_or_supplied_fact(
    original: str, scenario: str, rejected: bool
) -> None:
    from onyx.asv3.assertions import assertion_support_defect

    answer = "UN1170 uses compliant vehicles [1]."
    unit = assertion_inventory(answer)[0]
    check = AssertionVerification(
        unit_id=unit["unit_id"],
        status="supported",
        witnesses=[AssertionWitness(citation=1, source_quote=original)],
    )
    defect = assertion_support_defect(unit, check, {1: original}, scenario)
    assert (defect is not None) == rejected
