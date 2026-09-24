import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.regulatory.provision_lookup import ProvisionLookupResult
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.tool_implementations.regulatory_provision.regulatory_provision_tool import (
    ProvisionToolOverrideKwargs,
    RegulatoryProvisionTool,
)


def test_tool_reports_empty_results_without_running_another_tool() -> None:
    filters = IndexFilters(access_control_list=["user:1"], as_of_date=date(2026, 9, 24))
    reader = MagicMock()
    tool = RegulatoryProvisionTool(
        tool_id=99,
        emitter=MagicMock(),
        document_index=reader,
        filters_provider=lambda: filters,
    )
    with patch(
        "onyx.tools.tool_implementations.regulatory_provision.regulatory_provision_tool.lookup_provision",
        return_value=ProvisionLookupResult("not_found_in_scope", "Try another search"),
    ) as lookup:
        response = tool.run(
            Placement(turn_index=0, tab_index=0),
            ProvisionToolOverrideKwargs(starting_citation_num=201),
            source="5434 sayılı Kanun",
            article_number="72",
        )
    assert json.loads(response.llm_facing_response)["status"] == "not_found_in_scope"
    assert response.rich_response.search_docs == []
    assert lookup.call_count == 1
    assert not reader.method_calls


def test_invalid_arguments_do_not_read_the_index() -> None:
    provider = MagicMock()
    tool = RegulatoryProvisionTool(
        tool_id=99,
        emitter=MagicMock(),
        document_index=MagicMock(),
        filters_provider=provider,
    )
    response = tool.run(
        Placement(turn_index=0, tab_index=0),
        ProvisionToolOverrideKwargs(),
        source="Kanun",
        article_number="1 or 2",
    )
    assert json.loads(response.llm_facing_response)["status"] == "invalid_reference"
    provider.assert_not_called()


def test_definition_is_optional_capability_without_forced_first_search() -> None:
    tool = RegulatoryProvisionTool(
        tool_id=99,
        emitter=MagicMock(),
        document_index=MagicMock(),
        filters_provider=MagicMock(),
    )
    definition = tool.tool_definition()
    assert definition["function"]["name"] == "get_regulatory_provision"
    assert definition["function"]["parameters"]["required"] == [
        "source",
        "article_number",
    ]
    description = definition["function"]["description"].lower()
    assert "optional" in description
    assert "first" not in description
    assert "must use" not in description


def test_tool_shares_internal_search_availability() -> None:
    from onyx.tools.tool_implementations.search.search_tool import SearchTool

    for available in (False, True):
        with patch.object(SearchTool, "is_available", return_value=available):
            assert RegulatoryProvisionTool.is_available(MagicMock()) is available


def test_constructor_exposes_optional_tool_with_same_source_scope_and_disable_controls() -> (
    None
):
    from onyx.db.models import Tool as DbTool
    from onyx.tools.models import SearchToolUsage
    from onyx.tools.tool_constructor import SearchToolConfig, _construct_tools_impl
    from onyx.tools.tool_implementations.search.search_tool import SearchTool

    persona = MagicMock()
    persona.tools = [
        DbTool(
            id=1, name="internal_search", in_code_tool_id="SearchTool", enabled=True
        ),
        DbTool(
            id=99,
            name="get_regulatory_provision",
            in_code_tool_id="RegulatoryProvisionTool",
            enabled=True,
        ),
    ]
    persona.document_sets = [MagicMock(name="set")]
    persona.document_sets[0].name = "Scoped corpus"
    persona.attached_documents = []
    persona.hierarchy_nodes = []
    persona.search_start_date = None
    user = MagicMock()
    user.oauth_accounts = []
    config = SearchToolConfig(
        user_selected_filters=None, enable_slack_search=False, project_id_filter=7
    )
    with (
        patch("onyx.tools.tool_constructor.get_default_document_index"),
        patch("onyx.tools.tool_constructor.get_current_search_settings"),
        patch.object(SearchTool, "is_available", return_value=True),
    ):
        arguments = dict(
            persona=persona,
            db_session=MagicMock(),
            emitter=MagicMock(),
            user=user,
            llm=MagicMock(),
            search_tool_config=config,
        )
        tools = _construct_tools_impl(**arguments)
        assert {1, 99} <= set(tools)
        assert isinstance(tools[99][0], RegulatoryProvisionTool)
        assert not (
            {1, 99}
            & set(
                _construct_tools_impl(
                    **arguments, search_usage_forcing_setting=SearchToolUsage.DISABLED
                )
            )
        )
        assert 99 not in _construct_tools_impl(**arguments, allowed_tool_ids=[99])
        with (
            patch(
                "onyx.tools.tool_constructor.get_session_with_current_tenant_if_none"
            ),
            patch(
                "onyx.tools.tool_constructor._build_index_filters",
                return_value=IndexFilters(access_control_list=["user:1"]),
            ) as build_filters,
        ):
            tools[99][0].filters_provider()
        assert build_filters.call_args.kwargs["project_id_filter"] == 7
        assert build_filters.call_args.kwargs["persona_document_sets"] == [
            "Scoped corpus"
        ]
        assert build_filters.call_args.kwargs["bypass_acl"] is False
        persona.tools[0].enabled = False
        assert 99 not in _construct_tools_impl(**arguments)


@pytest.mark.parametrize("visible_ids", [{"rc-1", "rc-2"}, {"rc-1"}, set()])
def test_publication_visibility_preserves_only_verified_citations(
    visible_ids: set[str],
) -> None:
    file_id = "b8070ecf-7750-4bd7-98e5-ef6f550ecbcf"
    rows = [
        InferenceChunk(
            document_id=file_id,
            chunk_id=number,
            content=f"Canonical text {number}",
            source_type=DocumentSource.USER_FILE,
            semantic_identifier="5434 sayılı Kanun",
            title="5434 sayılı Kanun",
            boost=1,
            score=1,
            hidden=False,
            metadata={},
            match_highlights=[],
            doc_summary="",
            chunk_context="",
            updated_at=None,
            image_file_id=None,
            source_links={},
            section_continuation=False,
            blurb="Canonical text",
            regulatory_chunk_id=f"rc-{number}",
            heading_path=["5434 sayılı Kanun", "MADDE 72"],
        )
        for number in (1, 2)
    ]
    tool = RegulatoryProvisionTool(
        tool_id=99,
        emitter=MagicMock(),
        document_index=MagicMock(),
        filters_provider=lambda: IndexFilters(access_control_list=["user:1"]),
    )
    module = (
        "onyx.tools.tool_implementations.regulatory_provision.regulatory_provision_tool"
    )
    with (
        patch(
            f"{module}.lookup_provision",
            return_value=ProvisionLookupResult(
                "found", "Verified", rows, as_of_date=date(2026, 9, 24)
            ),
        ),
        patch(f"{module}.get_session_with_current_tenant"),
        patch(
            f"{module}.get_visible_regulatory_chunk_ids", return_value=visible_ids
        ) as visibility,
    ):
        response = tool.run(
            Placement(turn_index=0, tab_index=0),
            ProvisionToolOverrideKwargs(starting_citation_num=101),
            source="5434 sayılı Kanun",
            article_number="72",
        )
    payload = json.loads(response.llm_facing_response)
    assert payload["status"] == (
        "found"
        if len(visible_ids) == 2
        else "partial"
        if visible_ids
        else "unavailable"
    )
    assert {
        json.loads(result["metadata"])["regulatory_chunk_id"]
        for result in payload["results"]
    } == visible_ids
    assert response.rich_response.citation_chunk_mapping == {
        101 + index: row.chunk_id
        for index, row in enumerate(
            row for row in rows if row.regulatory_chunk_id in visible_ids
        )
    }
    assert visibility.call_args.kwargs["as_of_date"] == date(2026, 9, 24)
