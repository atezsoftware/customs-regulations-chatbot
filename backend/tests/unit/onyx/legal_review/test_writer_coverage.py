import pytest
from pydantic import ValidationError

from onyx.legal_review.writer_coverage import WriterCoverageContract
from tests.unit.onyx.legal_review.test_drafting import draft_payload
from tests.unit.onyx.legal_review.test_engine import prepared_engine, reading


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


@pytest.mark.parametrize("claim_ids", [[], ["unknown"], ["refund", "refund"]])
def test_covered_finding_cannot_point_to_missing_or_duplicate_prose(
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
