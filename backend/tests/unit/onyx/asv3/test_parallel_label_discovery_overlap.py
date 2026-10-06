import json
from contextlib import nullcontext
from datetime import date
from threading import Event
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.context.search.retrieval.parallel_retrieval_scope import (
    experimental_parallel_retrieval,
    parallel_retrieval_enabled,
)
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.regulatory.labeling import search_runtime
from onyx.regulatory.labeling.search_models import LabelSearchOverlay
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.search import search_tool
from tests.unit.onyx.regulatory.labeling.test_search_overlay import (
    chunk,
    evidence,
    snapshot,
)
from tests.unit.onyx.tools.tool_implementations.search.test_search_tool_run import (
    _make_tool,
    _run,
)


@pytest.mark.parametrize("discovery_fails", [False, True])
def test_prelaunch_discovery_overlaps_base_without_changing_candidates_or_rank(
    monkeypatch: pytest.MonkeyPatch, discovery_fails: bool
) -> None:
    baseline = [chunk(str(index)) for index in range(1, 5)]
    extra = chunk("5")
    state = snapshot()
    overlay = LabelSearchOverlay(
        candidate_ids=("5",),
        evidence_by_chunk={
            "2": (evidence("2", "SUB.TAX.VAT"),),
            "5": (evidence("5", "SUB.TAX.VAT"),),
        },
        source_texts={"2": baseline[1].content, "5": extra.content},
    )
    monkeypatch.setattr(search_tool, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "label_read_session", MagicMock())

    def execute(parallel: bool) -> dict[str, Any]:
        tool = _make_tool(
            IndexFilters(
                access_control_list=["authorized-owner"],
                asv3_document_set_id=73,
                source_type=[DocumentSource.USER_FILE],
                regulatory_chunks_only=True,
                regulatory_label_search_enabled=True,
                as_of_date=date(2025, 9, 17),
            ),
            auto_detect_filters=False,
        )
        monkeypatch.setattr(tool, "get_label_search_snapshot", lambda: state)
        base_entered, label_entered, base_finished = Event(), Event(), Event()
        queries: list[tuple[str, dict[str, Any], int]] = []

        def acquire(*arguments: Any) -> list[InferenceChunk]:
            query, filters = arguments[0], arguments[7]
            assert isinstance(query, str)
            assert isinstance(filters, IndexFilters)
            queries.append((query, filters.model_dump(mode="json"), arguments[3]))
            assert parallel_retrieval_enabled() is parallel
            if query == "neutral workflow request":
                base_entered.set()
                if parallel:
                    assert label_entered.wait(2), "label acquisition waited for base"
                base_finished.set()
                return baseline
            if parallel:
                assert base_entered.wait(2)
                assert not base_finished.is_set(), "label acquisition was serialized"
            else:
                assert base_finished.is_set()
            label_entered.set()
            if discovery_fails:
                raise RuntimeError("optional discovery unavailable")
            return [baseline[1], extra, extra]

        def rank(
            *, query: str, chunks: list[InferenceChunk], config: RerankerRuntimeConfig
        ) -> RerankResult:
            assert query == "neutral workflow request"
            assert not config.enabled
            return RerankResult(
                ordered_chunks=chunks,
                scores_by_chunk={},
                submitted_count=len(chunks),
                result_count=len(chunks),
                outcome=RerankOutcome.SUCCESS,
                fallback_used=False,
            )

        responses: list[ToolResponse] = []
        reranks: list[MagicMock] = []
        rrf: list[MagicMock] = []
        load_overlay = MagicMock(return_value=overlay)
        scope = (
            experimental_parallel_retrieval(check_active=lambda: None)
            if parallel
            else nullcontext()
        )
        with (
            scope,
            patch.object(tool, "_run_search_for_query", side_effect=acquire),
            patch.object(search_runtime, "load_label_overlay", load_overlay),
        ):
            _run(
                tool,
                connected_sources=[DocumentSource.USER_FILE],
                query="neutral workflow request",
                skip_query_expansion=True,
                label_hint={"label_ids": ["SUB.TAX.VAT"]},
                fused_chunks=baseline,
                rerank_behavior=rank,
                rerank_sink=reranks,
                rrf_sink=rrf,
                response_sink=responses,
            )
        assert base_finished.is_set() and label_entered.is_set()
        assert not parallel_retrieval_enabled()
        assert len(queries) == 2
        return {
            "response": json.loads(responses[0].llm_facing_response),
            "queries": sorted(queries),
            "ranked_chunks": reranks[0].call_args.kwargs["chunks"],
            "base_lanes": rrf[0].call_args.kwargs["ranked_results"],
            "overlay_arguments": (
                load_overlay.call_args.kwargs if load_overlay.called else None
            ),
        }

    serial = execute(False)
    parallel = execute(True)
    assert parallel == serial
    assert parallel["base_lanes"] == [baseline]
    if discovery_fails:
        assert parallel["ranked_chunks"] == baseline
        assert parallel["overlay_arguments"] is None
    else:
        ranked = parallel["ranked_chunks"]
        assert {item.regulatory_chunk_id for item in ranked} == {
            "1",
            "2",
            "3",
            "4",
            "5",
        }
        assert ranked[0] == baseline[1]
        assert extra in ranked


def test_no_subject_hint_does_not_add_discovery_to_parallel_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = snapshot()
    baseline = [chunk("1"), chunk("2")]
    tool = _make_tool(
        IndexFilters(
            access_control_list=["authorized-owner"],
            asv3_document_set_id=73,
            regulatory_chunks_only=True,
            regulatory_label_search_enabled=True,
        ),
        auto_detect_filters=False,
    )
    monkeypatch.setattr(tool, "get_label_search_snapshot", lambda: state)
    monkeypatch.setattr(search_tool, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    responses: list[ToolResponse] = []
    with (
        experimental_parallel_retrieval(check_active=lambda: None),
        patch.object(tool, "_run_search_for_query", return_value=baseline) as acquire,
        patch.object(search_runtime, "load_label_overlay") as overlay,
        patch.object(search_runtime.LabelDiscoveryAcquisition, "capture") as discover,
    ):
        _run(
            tool,
            connected_sources=[DocumentSource.USER_FILE],
            query="unclassified neutral request",
            skip_query_expansion=True,
            fused_chunks=baseline,
            response_sink=responses,
        )
    assert acquire.call_count == 1
    discover.assert_not_called()
    overlay.assert_not_called()
    assert json.loads(responses[0].llm_facing_response)["results"]
    assert not parallel_retrieval_enabled()
