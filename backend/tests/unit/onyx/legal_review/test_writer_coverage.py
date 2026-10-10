from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from onyx.legal_review.drafting import GeneratedDraft
from onyx.legal_review.engine import LegalReviewEngine
from onyx.legal_review.writer_coverage import (
    FindingBindingContract,
    FindingBindingError,
    WriterCoverageContract,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_review.test_drafting import draft_payload
from tests.unit.onyx.legal_review.test_engine import (
    FakeReviewer,
    prepared_engine,
    reading,
)


def contract() -> WriterCoverageContract:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    return WriterCoverageContract(workflow.requirements)


def test_every_material_finding_needs_an_explicit_writer_disposition() -> None:
    coverage = contract()
    payload = draft_payload()
    with pytest.raises(ValidationError, match="finding_coverage"):
        coverage.response_model.model_validate(payload)
    payload["finding_coverage"] = {}
    with pytest.raises(ValidationError, match="f0001"):
        coverage.response_model.model_validate(payload)


@pytest.mark.parametrize("claim_ids", [[], ["unknown"]])
def test_covered_finding_cannot_point_to_missing_prose(
    claim_ids: list[str],
) -> None:
    coverage = contract()
    payload = {
        **draft_payload(),
        "finding_coverage": {
            "f0001": {
                "disposition": "covered",
                "claim_ids": claim_ids,
                "reason": "The material effect is in the answer.",
            }
        },
    }
    with pytest.raises(ValueError):
        coverage.compile(coverage.response_model.model_validate(payload))


def test_writer_can_explain_non_applicability_without_forcing_a_source_citation() -> (
    None
):
    coverage = contract()
    payload = {
        **draft_payload(),
        "finding_coverage": {
            "f0001": {
                "disposition": "not_applicable",
                "claim_ids": [],
                "reason": "This alternative requires a fact absent from the requested scenario.",
            }
        },
    }
    generated, audit = coverage.compile(coverage.response_model.model_validate(payload))
    assert len(generated.blocks) == len(payload["blocks"])
    assert audit[0]["requirement_id"] == "r1"
    assert audit[0]["disposition"] == "not_applicable"


def accounted_payload(claim_ids: list[str]) -> dict[str, Any]:
    return {
        **draft_payload(),
        "finding_coverage": {
            "f0001": {
                "disposition": "covered",
                "claim_ids": claim_ids,
                "reason": "The material effect is in the literal answer.",
            }
        },
    }


def test_repeated_known_selection_is_idempotent() -> None:
    coverage = contract()
    response = coverage.response_model.model_validate(
        accounted_payload(["refund", "refund"])
    )
    _, rows = coverage.compile(response)
    assert rows[0]["claim_ids"] == ["refund"]
    assert response.model_dump()["finding_coverage"]["f0001"]["claim_ids"] == [
        "refund",
        "refund",
    ]


def test_correction_schema_selects_only_frozen_claim_ids() -> None:
    coverage = contract()
    response = coverage.response_model.model_validate(accounted_payload(["unknown"]))
    with pytest.raises(FindingBindingError) as error:
        coverage.compile(response)
    binding = FindingBindingContract(coverage, response, error.value.invalid_slots)
    replacement = {
        "f0001": {
            "disposition": "covered",
            "claim_ids": ["refund"],
            "reason": "The conditional result covers this effect.",
        }
    }
    repaired = binding.response_model.model_validate(replacement)
    rows = binding.compile(repaired)
    assert rows[0]["claim_ids"] == ["refund"]
    replacement["f0001"]["claim_ids"] = ["another-invented-name"]
    with pytest.raises(ValidationError):
        binding.response_model.model_validate(replacement)
    assert (
        binding.state()["frozen_blocks"] == response.model_dump(mode="json")["blocks"]
    )


def writer_fixture() -> tuple[
    LegalReviewEngine, WriterCoverageContract, GeneratedDraft
]:
    workflow = prepared_engine()
    workflow._accept_reading(reading())
    coverage = WriterCoverageContract(workflow.requirements)
    payload = accounted_payload(["misspelled-claim"])
    payload["blocks"] = [
        {
            "block_id": "b1",
            "text": "Başvuru belge koşuluna tabidir. [1]",
            "claims": [
                {
                    **payload["blocks"][1]["claims"][0],
                    "claim_id": "actual-claim",
                    "issue_ids": ["i1"],
                    "supports": [{"citation": 1, "span_number": 1}],
                }
            ],
        }
    ]
    payload["unresolved_issue_ids"] = []
    return workflow, coverage, coverage.response_model.model_validate(payload)


def test_binding_failure_preserves_draft_and_continues_independent_review() -> None:
    workflow, coverage, response = writer_fixture()
    binding = FindingBindingContract(coverage, response, ["f0001"])
    repair = binding.response_model.model_validate(
        {
            "f0001": {
                "disposition": "covered",
                "claim_ids": ["actual-claim"],
                "reason": "The literal procedure claim contains the document condition.",
            }
        }
    )
    workflow.gateway = Mock(complete=Mock(return_value=repair))
    workflow.reviewer = reviewer = FakeReviewer(["pass"])
    draft = workflow._accept_writer_draft(response, coverage)
    result = workflow._finish_draft("Başvuru şartı nedir?", "", draft)
    assert result.answer == "Başvuru belge koşuluna tabidir. [1]"
    assert len(reviewer.states) == 1
    assert (
        workflow.gateway.complete.call_args.args[3]
        is LLMFlow.LEGAL_REVIEW_FINDING_BINDING
    )
    assert workflow.writer_coverage[0]["claim_ids"] == ["actual-claim"]


@pytest.mark.parametrize(
    "error", [TimeoutError("deadline"), ValueError("invalid correction")]
)
def test_unfinished_binding_correction_remains_partial_without_losing_prose(
    error: Exception,
) -> None:
    workflow, coverage, response = writer_fixture()
    workflow.gateway = Mock(complete=Mock(side_effect=error))
    workflow.reviewer = FakeReviewer(["pass"])
    draft = workflow._accept_writer_draft(response, coverage)
    result = workflow._finish_draft("Başvuru şartı nedir?", "", draft)
    assert result.status == "partial" and result.answer == draft.answer
    assert workflow.gateway.complete.call_count == 1
    assert workflow.writer_coverage[0]["disposition"] == "unresolved"
    assert workflow.writer_coverage[0]["claim_ids"] == []
    assert "Finding-to-claim accounting remains incomplete" in result.gaps


def test_invalid_canonical_source_cannot_be_saved_by_binding_repair() -> None:
    workflow, coverage, response = writer_fixture()
    response.blocks[0].claims[0].supports[0] = (
        response.blocks[0].claims[0].supports[0].model_copy(update={"span_number": 999})
    )
    workflow.gateway = Mock()
    with pytest.raises(ValueError):
        workflow._accept_writer_draft(response, coverage)
    assert workflow.last_draft is None
    workflow.gateway.complete.assert_not_called()
