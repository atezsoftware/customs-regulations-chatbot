from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from onyx.configs.constants import MessageType
from onyx.secondary_llm_flows.query_expansion import (
    keyword_query_expansion,
    semantic_query_rephrase,
)
from onyx.tools.models import ChatMinimalTextMessage


def _history() -> list[ChatMinimalTextMessage]:
    return [
        ChatMinimalTextMessage(
            message="Find the controlling provision.",
            message_type=MessageType.USER,
        )
    ]


def _llm_returning(content: str) -> MagicMock:
    response = MagicMock()
    response.choice.message.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


def test_semantic_query_rephrase_bounds_llm_output() -> None:
    llm = _llm_returning("standalone controlling provision")

    with (
        patch(
            "onyx.secondary_llm_flows.query_expansion.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.secondary_llm_flows.query_expansion.record_llm_response"),
    ):
        assert (
            semantic_query_rephrase(_history(), llm)
            == "standalone controlling provision"
        )

    assert llm.invoke.call_args.kwargs["max_tokens"] == 512


def test_keyword_query_expansion_bounds_llm_output() -> None:
    llm = _llm_returning("first query\nsecond query\nthird query")

    with (
        patch(
            "onyx.secondary_llm_flows.query_expansion.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.secondary_llm_flows.query_expansion.record_llm_response"),
    ):
        assert keyword_query_expansion(_history(), llm) == [
            "first query",
            "second query",
            "third query",
        ]

    assert llm.invoke.call_args.kwargs["max_tokens"] == 256
