"""Experimental research remains an explicit, mutually exclusive ASv3 profile."""

import pytest
from pydantic import JsonValue, ValidationError

from onyx.server.query_and_chat.models import SendMessageRequest


def test_experimental_profile_requires_explicit_asv3_opt_in() -> None:
    with pytest.raises(ValidationError, match="explicit ASv3 selection"):
        SendMessageRequest.model_validate(
            {"message": "Question", "asv3_research_profile": "experimental"}
        )
    selected = SendMessageRequest.model_validate(
        {
            "message": "Question",
            "atez_search_v3": True,
            "asv3_research_profile": "experimental",
            "asv3_allow_external": True,
            "asv3_resume_message_id": 42,
        }
    )
    assert selected.asv3_research_profile == "experimental"
    assert selected.deep_research is False
    assert selected.asv3_allow_external is True
    assert selected.asv3_resume_message_id == 42


@pytest.mark.parametrize(
    "additional",
    [
        {"deep_research": True},
        {"atez_search": True},
        {"atez_search_v2": True},
        {
            "llm_overrides": [
                {"model_provider_type": "vertex_ai", "model_version": "selected-model"},
                {"model_provider_type": "vertex_ai", "model_version": "another-model"},
            ]
        },
    ],
)
def test_experimental_profile_cannot_combine_workflows_or_coordinator_models(
    additional: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Question",
                "atez_search_v3": True,
                "asv3_research_profile": "experimental",
                **additional,
            }
        )


def test_existing_profile_defaults_and_requests_remain_unchanged() -> None:
    default = SendMessageRequest.model_validate({"message": "Question"})
    assert default.atez_search_v3 is False
    assert default.deep_research is False
    assert default.asv3_research_profile == "deep"
    normal = SendMessageRequest.model_validate(
        {
            "message": "Question",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
        }
    )
    assert normal.asv3_research_profile == "normal"
    deep = SendMessageRequest.model_validate(
        {"message": "Question", "deep_research": True}
    )
    assert deep.atez_search_v3 is False
    assert deep.asv3_research_profile == "deep"


def test_experimental_profile_is_not_activated_by_deep_research_alone() -> None:
    with pytest.raises(ValidationError, match="explicit ASv3 selection"):
        SendMessageRequest.model_validate(
            {
                "message": "Question",
                "deep_research": True,
                "asv3_research_profile": "experimental",
            }
        )


def test_guarded_experimental_requires_explicit_normal_asv3_selection() -> None:
    selected = SendMessageRequest.model_validate(
        {
            "message": "Question",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guarded_experimental": True,
        }
    )
    assert selected.asv3_guarded_experimental is True

    for invalid in (
        {"asv3_research_profile": "experimental"},
        {"asv3_parallel_research": True},
        {"deep_research": True},
        {"atez_search_v3": False},
    ):
        with pytest.raises(ValidationError):
            SendMessageRequest.model_validate(
                {
                    "message": "Question",
                    "atez_search_v3": True,
                    "asv3_research_profile": "normal",
                    "asv3_guarded_experimental": True,
                    **invalid,
                }
            )
