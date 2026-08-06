import pytest

from onyx.chat.models import ChatMessageSimple, ToolCallSimple
from onyx.configs.constants import MessageType
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolCallKickoff
from onyx.tools.tool_runner import (
    DEFAULT_SEARCH_MAX_LLM_CHUNKS,
    PARALLEL_SEARCH_MIN_LLM_CHUNKS_PER_CALL,
    PARALLEL_SEARCH_TARGET_TOTAL_LLM_CHUNKS,
    _has_completed_internal_search_in_current_turn,
    _max_llm_chunks_per_search_call,
    _merge_tool_calls,
    _parallel_search_filter_queries,
    _search_filter_message_history,
    _search_input_context,
    _should_skip_search_query_expansion,
)


def _make_tool_call(
    tool_name: str,
    tool_args: dict,
    tool_call_id: str = "call_1",
    turn_index: int = 0,
    tab_index: int = 0,
) -> ToolCallKickoff:
    """Helper to create a ToolCallKickoff for testing."""
    return ToolCallKickoff(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
        placement=Placement(turn_index=turn_index, tab_index=tab_index),
    )


def _chat_message(
    message: str,
    message_type: MessageType,
    *,
    tool_call_id: str | None = None,
    tool_calls: list[ToolCallSimple] | None = None,
) -> ChatMessageSimple:
    return ChatMessageSimple(
        message=message,
        token_count=max(1, len(message) // 4),
        message_type=message_type,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
    )


def _internal_search_message(tool_call_id: str) -> ChatMessageSimple:
    return _chat_message(
        "",
        MessageType.ASSISTANT,
        tool_calls=[
            ToolCallSimple(
                tool_call_id=tool_call_id,
                tool_name="internal_search",
                tool_arguments={},
            )
        ],
    )


def test_model_written_internal_search_uses_only_focused_context() -> None:
    call = _make_tool_call(
        tool_name="internal_search",
        tool_args={
            "queries": ["focused legal issue"],
            "coverage_item": "Exact named mechanism in the user item",
        },
    )

    original_query, history = _search_input_context(
        tool_call=call,
        last_user_message="long multi-part user request",
    )

    assert original_query == "focused legal issue"
    assert [message.message for message in history] == [
        "Exact named mechanism in the user item"
    ]


def test_single_internal_search_also_uses_only_focused_context() -> None:
    call = _make_tool_call(
        tool_name="internal_search",
        tool_args={"queries": ["short query"]},
    )

    original_query, history = _search_input_context(
        tool_call=call,
        last_user_message="full user request",
    )

    assert original_query == "short query"
    assert [message.message for message in history] == ["short query"]


def test_first_search_filter_keeps_latest_user_intent_without_old_results() -> None:
    latest_user_request = (
        "Use the records valid on 2021-05-01 and only the selected source type."
    )
    message_history = [
        _chat_message("earlier request", MessageType.USER),
        _internal_search_message("old_search"),
        _chat_message(
            "very large old tool result",
            MessageType.TOOL_CALL_RESPONSE,
            tool_call_id="old_search",
        ),
        _chat_message(latest_user_request, MessageType.USER),
        _internal_search_message("current_search"),
    ]
    current_call = _make_tool_call(
        tool_name="internal_search",
        tool_args={
            "queries": ["focused current issue"],
            "coverage_item": "Current issue",
        },
        tool_call_id="current_search",
    )

    completed_search = _has_completed_internal_search_in_current_turn(message_history)
    filter_history = _search_filter_message_history(
        tool_calls=[current_call],
        last_user_message=latest_user_request,
        has_completed_internal_search=completed_search,
    )

    assert not completed_search
    assert [message.message for message in filter_history] == [latest_user_request]
    assert "very large old tool result" not in str(filter_history)


def test_later_single_search_filter_uses_only_current_coverage_context() -> None:
    last_user_message = "A long request containing several independent issues."
    message_history = [
        _chat_message(last_user_message, MessageType.USER),
        _internal_search_message("prior_search"),
        _chat_message(
            "very large prior retrieval payload",
            MessageType.TOOL_CALL_RESPONSE,
            tool_call_id="prior_search",
        ),
        _internal_search_message("current_search"),
    ]
    current_call = _make_tool_call(
        tool_name="internal_search",
        tool_args={
            "queries": ["second focused query"],
            "coverage_item": "Unresolved second issue",
        },
        tool_call_id="current_search",
    )

    completed_search = _has_completed_internal_search_in_current_turn(message_history)
    filter_history = _search_filter_message_history(
        tool_calls=[current_call],
        last_user_message=last_user_message,
        has_completed_internal_search=completed_search,
    )

    assert completed_search
    assert [message.message for message in filter_history] == [
        "Unresolved second issue"
    ]
    assert "very large prior retrieval payload" not in str(filter_history)


def test_later_parallel_search_filter_is_bounded_to_current_batch() -> None:
    calls = [
        _make_tool_call(
            tool_name="internal_search",
            tool_args={
                "queries": ["first focused query"],
                "coverage_item": "First unresolved issue",
            },
            tool_call_id="call_1",
        ),
        _make_tool_call(
            tool_name="internal_search",
            tool_args={
                "queries": ["second focused query"],
                "coverage_item": "Second unresolved issue",
            },
            tool_call_id="call_2",
        ),
        _make_tool_call(
            tool_name="internal_search",
            tool_args={
                "queries": ["duplicate coverage query"],
                "coverage_item": "First unresolved issue",
            },
            tool_call_id="call_3",
        ),
    ]

    filter_history = _search_filter_message_history(
        tool_calls=calls,
        last_user_message="long original request",
        has_completed_internal_search=True,
    )

    assert [message.message for message in filter_history] == [
        "First unresolved issue",
        "Second unresolved issue",
    ]
    assert len(filter_history) == 2


def test_model_written_internal_searches_keep_bounded_query_expansion() -> None:
    assert not _should_skip_search_query_expansion(
        skip_requested=False,
        internal_search_call_count=7,
    )
    assert not _should_skip_search_query_expansion(
        skip_requested=False,
        internal_search_call_count=1,
    )
    assert _should_skip_search_query_expansion(
        skip_requested=True,
        internal_search_call_count=1,
    )
    assert not _should_skip_search_query_expansion(
        skip_requested=False,
        internal_search_call_count=0,
    )


def test_parallel_filter_decision_receives_every_model_written_query() -> None:
    calls = [
        _make_tool_call(
            tool_name="internal_search",
            tool_args={"queries": [" first focused query "]},
            tool_call_id="call_1",
        ),
        _make_tool_call(
            tool_name="internal_search",
            tool_args={"queries": ["second focused query"]},
            tool_call_id="call_2",
        ),
        _make_tool_call(
            tool_name="internal_search",
            tool_args={"queries": ["first focused query"]},
            tool_call_id="call_3",
        ),
    ]

    assert _parallel_search_filter_queries(calls) == [
        "first focused query",
        "second focused query",
    ]


def test_parallel_internal_searches_share_a_bounded_llm_chunk_budget() -> None:
    assert DEFAULT_SEARCH_MAX_LLM_CHUNKS == 25
    assert PARALLEL_SEARCH_TARGET_TOTAL_LLM_CHUNKS == 84
    assert PARALLEL_SEARCH_MIN_LLM_CHUNKS_PER_CALL == 8

    assert _max_llm_chunks_per_search_call(1) == 25
    assert _max_llm_chunks_per_search_call(2) == 25
    assert _max_llm_chunks_per_search_call(3) == 25
    assert _max_llm_chunks_per_search_call(4) == 21
    assert _max_llm_chunks_per_search_call(7) == 12
    assert _max_llm_chunks_per_search_call(10) == 8
    assert _max_llm_chunks_per_search_call(14) == 8
    assert _max_llm_chunks_per_search_call(1, per_call_cap=6) == 6
    assert _max_llm_chunks_per_search_call(7, per_call_cap=6) == 6
    assert _max_llm_chunks_per_search_call(14, per_call_cap=4) == 4

    with pytest.raises(ValueError, match="per_call_cap must be positive"):
        _max_llm_chunks_per_search_call(7, per_call_cap=0)


class TestMergeToolCalls:
    """Tests for _merge_tool_calls function."""

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        result = _merge_tool_calls([])
        assert result == []

    def test_single_search_tool_call_not_merged(self) -> None:
        """A single SearchTool call is returned as-is (no merging needed)."""
        call = _make_tool_call(
            tool_name="internal_search",
            tool_args={"queries": ["query1"]},
            tool_call_id="call_1",
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "internal_search"
        assert result[0].tool_args == {"queries": ["query1"]}
        assert result[0].tool_call_id == "call_1"

    def test_single_web_search_tool_call_not_merged(self) -> None:
        """A single WebSearchTool call is returned as-is."""
        call = _make_tool_call(
            tool_name="web_search",
            tool_args={"queries": ["web query"]},
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "web_search"
        assert result[0].tool_args == {"queries": ["web query"]}

    def test_single_open_url_tool_call_not_merged(self) -> None:
        """A single OpenURLTool call is returned as-is."""
        call = _make_tool_call(
            tool_name="open_url",
            tool_args={"urls": ["https://example.com"]},
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "open_url"
        assert result[0].tool_args == {"urls": ["https://example.com"]}

    def test_multiple_search_tool_calls_remain_separate(self) -> None:
        """Independent internal searches retain independent result sets."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["query1"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["query3"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 2
        assert [call.tool_args["queries"] for call in result] == [
            ["query1"],
            ["query3"],
        ]
        assert [call.tool_call_id for call in result] == ["call_1", "call_2"]

    def test_multiple_web_search_tool_calls_merged(self) -> None:
        """Multiple WebSearchTool calls have their queries merged."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web1"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web2", "web3"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_name == "web_search"
        assert result[0].tool_args["queries"] == ["web1", "web2", "web3"]

    def test_multiple_open_url_tool_calls_merged(self) -> None:
        """Multiple OpenURLTool calls have their urls merged."""
        calls = [
            _make_tool_call(
                tool_name="open_url",
                tool_args={"urls": ["https://a.com"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="open_url",
                tool_args={"urls": ["https://b.com", "https://c.com"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_name == "open_url"
        assert result[0].tool_args["urls"] == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]

    def test_non_mergeable_tool_not_merged(self) -> None:
        """Non-mergeable tools (e.g., python) are returned as separate calls."""
        calls = [
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "print(1)"},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "print(2)"},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 2
        assert result[0].tool_args["code"] == "print(1)"
        assert result[1].tool_args["code"] == "print(2)"

    def test_mixed_mergeable_and_non_mergeable(self) -> None:
        """Mix of mergeable and non-mergeable tools handles correctly."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"]},
                tool_call_id="search_1",
            ),
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "x = 1"},
                tool_call_id="python_1",
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q2"]},
                tool_call_id="search_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 3

        tool_names = {r.tool_name for r in result}
        assert tool_names == {"internal_search", "run_python"}

        search_results = [r for r in result if r.tool_name == "internal_search"]
        assert [r.tool_args["queries"] for r in search_results] == [["q1"], ["q2"]]

        python_result = next(r for r in result if r.tool_name == "run_python")
        assert python_result.tool_args["code"] == "x = 1"

    def test_multiple_different_mergeable_tools(self) -> None:
        """Multiple different mergeable tools each get merged separately."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["search1"]},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web1"]},
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["search2"]},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web2"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 3

        search_results = [r for r in result if r.tool_name == "internal_search"]
        assert [r.tool_args["queries"] for r in search_results] == [
            ["search1"],
            ["search2"],
        ]

        web_result = next(r for r in result if r.tool_name == "web_search")
        assert web_result.tool_args["queries"] == ["web1", "web2"]

    def test_preserves_first_call_placement(self) -> None:
        """Merged call uses the placement from the first call."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q1"]},
                turn_index=1,
                tab_index=2,
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q2"]},
                turn_index=3,
                tab_index=4,
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].placement.turn_index == 1
        assert result[0].placement.tab_index == 2

    def test_preserves_other_args_from_first_call(self) -> None:
        """Merged call preserves non-merge-field args from the first call."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q1"], "other_param": "value1"},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q2"], "other_param": "value2"},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1", "q2"]
        # Other params from first call are preserved
        assert result[0].tool_args["other_param"] == "value1"

    def test_handles_empty_queries_list(self) -> None:
        """Handles calls with empty queries lists."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": []},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q1"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1"]

    def test_handles_missing_merge_field(self) -> None:
        """Handles calls where the merge field is missing entirely."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={},  # No queries field
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q1"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1"]

    def test_handles_string_value_instead_of_list(self) -> None:
        """Handles edge case where merge field is a string instead of list."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": "single_query"},  # String instead of list
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["q2"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        # String should be converted to list item
        assert result[0].tool_args["queries"] == ["single_query", "q2"]
