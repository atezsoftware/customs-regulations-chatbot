"""Which research workflow produced an answer, derived from the request flags.

The value is shown under each answer, so every workflow the input bar offers
must map to its own identifier — collapsing two of them would mislabel answers.
"""

from typing import Any

import pytest

from onyx.chat.response_workflow import response_workflow
from onyx.server.query_and_chat.models import SendMessageRequest


def _request(**flags: Any) -> SendMessageRequest:
    # Flag combinations are validated where the request is parsed; here only the
    # mapping is under test.
    return SendMessageRequest.model_construct(**flags)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"legal_review": True}, "legal_review"),
        ({"supersearch": True}, "supersearch"),
        ({"legal_composite": True}, "legal_composite"),
        ({"deep_research": True}, "deep_research"),
        (
            {"atez_search_v3": True, "asv3_research_profile": "normal"},
            "asv3",
        ),
        (
            {
                "atez_search_v3": True,
                "asv3_research_profile": "experimental",
                "asv3_parallel_research": True,
            },
            "experimental_asv3",
        ),
        (
            {"atez_search_v3": True, "asv3_research_profile": "experimental"},
            "experimental",
        ),
        (
            {
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_exp_cropped": True,
            },
            "asv3_exp_cropped",
        ),
        (
            {
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_guarded_experimental": True,
            },
            "experimental_guardrails",
        ),
        (
            {
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_guardrails_v2": True,
            },
            "experimental_guardrails_v2",
        ),
        (
            {
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_guardrails_v3": True,
            },
            "experimental_guardrails_v3",
        ),
        ({"atez_search_v2": True}, "asv2"),
        ({"atez_search": True}, "asv1"),
    ],
)
def test_each_workflow_gets_its_own_identifier(
    flags: dict[str, Any], expected: str
) -> None:
    assert response_workflow(_request(**flags)) == expected


def test_a_request_with_no_workflow_flags_has_no_workflow() -> None:
    assert response_workflow(_request()) is None
