"""Parallel research is a separate Experimental opt-in, never a default."""

import pytest
from pydantic import JsonValue, ValidationError

from onyx.server.query_and_chat.models import SendMessageRequest


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"deep_research": True},
        {"atez_search_v3": True, "asv3_research_profile": "normal"},
        {"atez_search_v3": True, "asv3_research_profile": "deep"},
        {"atez_search_v3": True, "asv3_research_profile": "experimental"},
    ],
)
def test_existing_requests_do_not_enable_parallel_research(
    selection: dict[str, JsonValue],
) -> None:
    request = SendMessageRequest.model_validate({"message": "Question", **selection})
    assert request.asv3_parallel_research is False


def test_parallel_experimental_request_preserves_resume_and_external_permission() -> (
    None
):
    request = SendMessageRequest.model_validate(
        {
            "message": "Question",
            "atez_search_v3": True,
            "asv3_research_profile": "experimental",
            "asv3_parallel_research": True,
            "asv3_resume_message_id": 42,
            "asv3_allow_external": True,
            "llm_override": {
                "model_provider_type": "vertex_ai",
                "model_version": "selected-model",
            },
        }
    )
    assert request.asv3_parallel_research is True
    assert request.asv3_research_profile == "experimental"
    assert request.deep_research is False
    assert request.asv3_resume_message_id == 42
    assert request.asv3_allow_external is True
    assert request.llm_override is not None
    assert request.llm_override.model_version == "selected-model"


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"deep_research": True},
        {"atez_search_v3": True},
        {"atez_search_v3": True, "asv3_research_profile": "normal"},
        {"atez_search_v3": True, "asv3_research_profile": "deep"},
    ],
)
def test_parallel_flag_requires_explicit_experimental_profile(
    selection: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError, match="requires the experimental ASv3 profile"):
        SendMessageRequest.model_validate(
            {"message": "Question", "asv3_parallel_research": True, **selection}
        )


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
def test_parallel_experimental_cannot_bypass_existing_mode_constraints(
    additional: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Question",
                "atez_search_v3": True,
                "asv3_research_profile": "experimental",
                "asv3_parallel_research": True,
                **additional,
            }
        )
