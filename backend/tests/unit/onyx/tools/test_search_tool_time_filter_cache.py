from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from onyx.secondary_llm_flows.time_filter import (
    DocumentTimeField,
    TimeFilter,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def _make_tool() -> SearchTool:
    tool = SearchTool(
        tool_id=1,
        emitter=MagicMock(),
        user=MagicMock(),
        persona_search_info=MagicMock(),
        llm=MagicMock(),
        document_index=MagicMock(),
        user_selected_filters=None,
        project_id_filter=None,
    )
    tool._scope_decision_settled = True  # skip the source-scope job
    return tool


def test_time_filter_runs_once_and_caches_for_the_turn() -> None:
    tool = _make_tool()
    decided = TimeFilter(
        field=DocumentTimeField.UPDATED_AT,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    scheduled: list[list] = []

    def fake_parallel(jobs: list) -> list:
        funcs = [job[0] for job in jobs]
        scheduled.append(funcs)
        return [
            (
                decided
                if getattr(func, "__func__", None) is SearchTool._decide_time_filter
                else None
            )
            for func in funcs
        ]

    with patch(
        "onyx.tools.tool_implementations.search.search_tool."
        "run_functions_tuples_in_parallel",
        side_effect=fake_parallel,
    ):
        first = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[],
            user_info=None,
            memories=[],
            decide_args=(),
        )
        second = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[],
            user_info=None,
            memories=[],
            decide_args=(),
        )

    # Only the first cycle schedules the time-filter job; the second has
    # nothing to run.
    assert len(scheduled) == 1
    assert any(
        getattr(func, "__func__", None) is SearchTool._decide_time_filter
        for func in scheduled[0]
    )

    assert first.time_filter == decided
    assert second.time_filter == decided
    assert tool._time_filter_computed is True


def test_time_filter_skipped_when_auto_detect_disabled() -> None:
    tool = _make_tool()
    tool.auto_detect_filters = False

    scheduled: list[list] = []

    def fake_parallel(jobs: list) -> list:
        scheduled.append([job[0] for job in jobs])
        return [None for _ in jobs]

    with patch(
        "onyx.tools.tool_implementations.search.search_tool."
        "run_functions_tuples_in_parallel",
        side_effect=fake_parallel,
    ):
        result = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[],
            user_info=None,
            memories=[],
            decide_args=(),
        )

    assert scheduled == [] or all(
        all(
            getattr(func, "__func__", None) is not SearchTool._decide_time_filter
            for func in funcs
        )
        for funcs in scheduled
    )
    assert result.time_filter is None
    assert tool._time_filter_computed is False
