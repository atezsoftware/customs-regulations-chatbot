import pytest
from pydantic import ValidationError

from onyx.legal_review.planning import InitialPlanContract, request_units
from tests.unit.onyx.legal_review.test_initial_plan import valid_plan


def test_literal_requests_survive_grouped_issues_and_incomplete_model_inventory() -> (
    None
):
    contract = InitialPlanContract(
        "Olay olguları.\n1. Başvuru şartı?\n2. Süre ve alternatif yol?"
    )
    data = valid_plan().model_dump()
    data["request_coverage"] = {
        "q0001": {"requested_result": "Başvuru", "issue_ids": ["i1"]},
        "q0002": {"requested_result": "Süre ve alternatif", "issue_ids": ["i1"]},
    }
    result = contract.compile(contract.response_model.model_validate(data))
    assert len(result.issues) == 1
    assert result.requested_outcomes[0].request == "Başvuru şartı?"
    assert result.requested_outcomes[1].request == "Süre ve alternatif yol?"
    del data["request_coverage"]["q0002"]
    with pytest.raises(ValidationError, match="q0002"):
        contract.response_model.model_validate(data)


def test_unknown_coverage_identity_is_a_provider_schema_validation_error() -> None:
    contract = InitialPlanContract("Soru")
    data = {
        **valid_plan().model_dump(),
        "request_coverage": {
            "q0001": {"requested_result": "Soru", "issue_ids": ["invented"]}
        },
    }
    with pytest.raises(ValidationError, match="known issues"):
        contract.response_model.model_validate(data)


def test_plain_question_is_retained_without_inventing_issue_boundaries() -> None:
    assert request_units("İki sonuç arasındaki ilişki nedir?") == {
        "q0001": "İki sonuç arasındaki ilişki nedir?"
    }
