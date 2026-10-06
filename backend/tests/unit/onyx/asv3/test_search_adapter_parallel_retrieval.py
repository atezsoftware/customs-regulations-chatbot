from datetime import date
from typing import Any

import pytest

from onyx.asv3 import search_adapter
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.search_adapter import build_search_adapter
from onyx.context.search.retrieval.parallel_retrieval_scope import (
    parallel_retrieval_enabled,
)
from tests.unit.onyx.asv3.test_search_adapter import (
    search_boundaries,
    tool_and_broker,
    user_message,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize(
    "profile,parallel,diagnostics,owner,depth,expected",
    [
        ("experimental", True, False, None, 0, True),
        ("experimental", False, True, "owned-subquestion", 0, True),
        ("experimental", False, False, None, 0, False),
        ("normal", True, False, None, 0, False),
        ("deep", True, False, None, 0, False),
        ("experimental", False, True, None, 0, False),
        ("experimental", False, 1, "owned-subquestion", 0, False),
        ("experimental", False, True, "owned-subquestion", 1, False),
    ],
)
def test_actual_dispatch_binds_only_owned_parallel_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    parallel: bool,
    diagnostics: bool | int,
    owner: str | None,
    depth: int,
    expected: bool,
) -> None:
    tool, broker, selected = tool_and_broker()
    broker.filters.as_of_date = date(2025, 1, 1)
    adapter = build_search_adapter(
        tool,
        "full unchanged scenario",
        broker,
        message_history=lambda _: [user_message("full unchanged scenario")],
    )
    context = RunContext(
        depth=depth,
        services={
            "lean_native_mode": True,
            "research_profile": profile,
            "experimental_parallel": parallel,
            "serial_session_diagnostics": diagnostics,
        },
    )
    if owner is not None:
        context.services["task_id"] = owner
    dispatched: list[dict[str, Any]] = []
    actual_dispatch = search_adapter.run_tool_calls

    def capture(**arguments: Any) -> Any:
        assert parallel_retrieval_enabled() is expected
        dispatched.append(arguments)
        return actual_dispatch(**arguments)

    monkeypatch.setattr(search_adapter, "run_tool_calls", capture)
    with search_boundaries() as (pipeline, _scope, _time):
        actual = adapter(
            {"query": "literal operative anchor", "mode": "keyword"}, context
        )
    assert not parallel_retrieval_enabled()
    assert actual.status == OutcomeStatus.NOT_FOUND
    assert len(dispatched) == 1
    assert dispatched[0]["tool_calls"][0].tool_args == {
        "queries": ["literal operative anchor"],
        "search_mode": "keyword",
    }
    assert dispatched[0]["search_rerank_context"] == "full unchanged scenario"
    selected.invoke.assert_not_called()
    pipeline.assert_called_once()
    request = pipeline.call_args.kwargs["chunk_search_request"]
    assert request.query == "literal operative anchor"
    assert request.hybrid_alpha == 0.0
    assert request.user_selected_filters.tenant_id == "tenant-under-test"
    assert request.user_selected_filters.access_control_list == ["user:authorized"]
    assert request.user_selected_filters.as_of_date == date(2025, 1, 1)
    assert request.user_selected_filters.asv3_document_set_id == 15
    assert request.user_selected_filters.forced_document_set == ["PC Külliyatı"]


def test_dispatch_failure_restores_retrieval_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool, broker, _selected = tool_and_broker()
    adapter = build_search_adapter(
        tool,
        "full unchanged scenario",
        broker,
        message_history=lambda _: [user_message("full unchanged scenario")],
    )
    context = RunContext(
        services={"research_profile": "experimental", "experimental_parallel": True}
    )

    def fail(**_arguments: Any) -> Any:
        assert parallel_retrieval_enabled()
        raise RuntimeError("synthetic dispatcher failure")

    monkeypatch.setattr(search_adapter, "run_tool_calls", fail)
    with search_boundaries(), pytest.raises(RuntimeError, match="dispatcher failure"):
        adapter({"query": "literal operative anchor", "mode": "keyword"}, context)
    assert not parallel_retrieval_enabled()
