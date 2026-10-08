import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.chat.emitter import NullEmitter
from onyx.configs.constants import DocumentSource
from onyx.context.search import pipeline
from onyx.context.search.models import (
    ChunkIndexRequest,
    IndexFilters,
    InferenceChunk,
    PersonaSearchInfo,
    TimeRange,
)
from onyx.db.models import User
from onyx.document_index.elasticsearch.search import DocumentQuery
from onyx.document_index.interfaces_new import DocumentIndex, TenantState
from onyx.legal_composite.search import CompositeSearchTool
from onyx.legal_composite.source_lanes import source_lane_filters
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import SearchToolOverrideKwargs, ToolResponse
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def parent_tool() -> SearchTool:
    floor = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return SearchTool(
        tool_id=1,
        emitter=NullEmitter(),
        user=cast(User, SimpleNamespace(id=uuid4(), is_anonymous=False)),
        persona_search_info=PersonaSearchInfo(
            document_set_names=["PC Külliyatı", "Other broad knowledge"],
            search_start_date=floor,
            attached_document_ids=[str(uuid4())],
            hierarchy_node_ids=[123],
        ),
        llm=cast(LLM, MagicMock()),
        document_index=cast(DocumentIndex, MagicMock()),
        user_selected_filters=IndexFilters(
            access_control_list=["captured-acl"],
            source_type=[DocumentSource.USER_FILE],
            document_set=["PC Külliyatı"],
            forced_document_set=["PC Külliyatı"],
            asv3_document_set_id=15,
            regulatory_chunks_only=True,
            as_of_date=date(2022, 1, 1),
            updated_at_range=TimeRange(start=datetime(2019, 1, 1, tzinfo=timezone.utc)),
        ),
        project_id_filter=7,
        persona_id_filter=4,
        auto_detect_filters=False,
    )


def test_lane_ids_reach_real_pipeline_before_top_k_without_persona_or_widening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = parent_tool()
    original_persona = parent.persona_search_info.model_dump()
    original_filters = cast(IndexFilters, parent.user_selected_filters).model_dump()
    captured: list[ChunkIndexRequest] = []

    def retrieve(**kwargs: Any) -> list[InferenceChunk]:
        request = cast(ChunkIndexRequest, kwargs["query_request"])
        captured.append(request)
        return []

    monkeypatch.setattr(pipeline, "search_chunks", retrieve)
    monkeypatch.setattr(
        pipeline,
        "fetch_ee_implementation_or_noop",
        lambda _module, _name, _default: (
            lambda **kwargs: cast(list[InferenceChunk], kwargs["chunks"])
        ),
    )

    def shared_run(
        owned: SearchTool,
        _placement: Placement,
        _overrides: SearchToolOverrideKwargs,
        **_kwargs: Any,
    ) -> ToolResponse:
        # Keep the real SearchTool -> pipeline -> _build_index_filters call path.
        owned._run_search_for_query(
            query="legal source question",
            hybrid_alpha=0.5,
            high_term_coverage=False,
            num_hits=100,
            acl_filters=["fresh-user-acl"],
            embedding_model=cast(EmbeddingModel, MagicMock()),
            federated_retrieval_infos=[],
            effective_filters=owned.user_selected_filters,
        )
        return ToolResponse(rich_response=None, llm_facing_response="{}")

    monkeypatch.setattr(SearchTool, "run", shared_run)
    owned = CompositeSearchTool.from_fork(parent.fork_for_independent_context())
    lane_ids = [(uuid4(), uuid4()), (uuid4(),)]
    for identifiers in lane_ids:
        fork = owned.fork_for_independent_context()
        fork.user_selected_filters = source_lane_filters(
            cast(IndexFilters, parent.user_selected_filters), identifiers
        )
        fork.run(
            Placement(turn_index=0), SearchToolOverrideKwargs(starting_citation_num=1)
        )

    assert len(captured) == 2
    for request, identifiers in zip(captured, lane_ids, strict=True):
        filters = request.filters
        assert filters.attached_document_ids == [str(value) for value in identifiers]
        assert filters.document_set == []
        assert filters.hierarchy_node_ids is None
        assert filters.project_id_filter is filters.persona_id_filter is None
        assert filters.access_control_list == ["fresh-user-acl"]
        assert filters.source_type == [DocumentSource.USER_FILE]
        assert filters.forced_document_set == ["PC Külliyatı"]
        assert filters.asv3_document_set_id == 15
        assert filters.as_of_date == date(2022, 1, 1)
        assert filters.updated_at_range is not None
        assert (
            filters.updated_at_range.start
            == parent.persona_search_info.search_start_date
        )
        clauses = DocumentQuery._get_search_filters(
            tenant_state=TenantState(tenant_id="public", multitenant=False),
            include_hidden=False,
            access_control_list=filters.access_control_list,
            source_types=filters.source_type or [],
            tags=[],
            document_sets=filters.document_set or [],
            project_id_filter=filters.project_id_filter,
            persona_id_filter=filters.persona_id_filter,
            created_at_range=filters.created_at_range,
            updated_at_range=filters.updated_at_range,
            min_chunk_index=None,
            max_chunk_index=None,
            as_of_date=filters.as_of_date,
            regulatory_chunks_only=filters.regulatory_chunks_only,
            attached_document_ids=filters.attached_document_ids,
            hierarchy_node_ids=filters.hierarchy_node_ids,
            forced_document_sets=filters.forced_document_set,
        )
        knowledge = next(
            clause["bool"]
            for clause in clauses
            if "bool" in clause
            and any(
                "document_id" in item.get("terms", {})
                for item in clause["bool"].get("should", [])
            )
        )
        assert knowledge == {
            "should": [
                {"terms": {"document_id": [str(value) for value in identifiers]}}
            ],
            "minimum_should_match": 1,
        }
        assert {"terms": {"document_sets": ["PC Külliyatı"]}} in clauses
        assert "Other broad knowledge" not in json.dumps(clauses)
    assert parent.persona_search_info.model_dump() == original_persona
    assert (
        cast(IndexFilters, parent.user_selected_filters).model_dump()
        == original_filters
    )
    assert parent.project_id_filter == 7 and parent.persona_id_filter == 4


@pytest.mark.parametrize("bypass", [False, True])
def test_empty_lane_or_acl_bypass_fails_before_shared_search(
    monkeypatch: pytest.MonkeyPatch, bypass: bool
) -> None:
    owned = CompositeSearchTool.from_fork(parent_tool().fork_for_independent_context())
    filters = cast(IndexFilters, owned.user_selected_filters)
    owned.user_selected_filters = filters.model_copy(
        update={"attached_document_ids": [str(uuid4())] if bypass else []}
    )
    owned.bypass_acl = bypass
    shared = MagicMock()
    monkeypatch.setattr(SearchTool, "run", shared)
    with pytest.raises(PermissionError, match="captured IDs and ACLs"):
        owned.run(
            Placement(turn_index=0), SearchToolOverrideKwargs(starting_citation_num=1)
        )
    shared.assert_not_called()
