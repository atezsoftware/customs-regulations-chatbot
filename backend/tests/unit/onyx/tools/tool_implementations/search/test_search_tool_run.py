from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from threading import Barrier
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, InferenceChunk, InferenceSection
from onyx.context.search.utils import (
    convert_inference_sections_to_search_docs,
    inference_section_from_single_chunk,
)
from onyx.regulatory.heading_path import RegulatoryProvisionReference
from onyx.regulatory.provision_retrieval import (
    RegulatoryProvisionNavigation,
    RegulatoryProvisionNavigationEntry,
)
from onyx.secondary_llm_flows.time_filter import DocumentTimeField, TimeFilter
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import SearchToolFilterDelta
from onyx.tools.models import (
    ChatMinimalTextMessage,
    SearchToolOverrideKwargs,
    ToolCallException,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.search_tool import (
    REGULATORY_MAX_SEARCH_QUERY_CHARS,
    QueryExpansionAndScope,
    SearchTool,
    _add_regulatory_provision_navigation,
    _add_search_receipt,
    _backfill_ranked_regulatory_sections,
    _can_use_ranked_regulatory_selection,
    _diversify_focused_regulatory_retrieval_lanes,
    _filter_visible_regulatory_sections,
    _normalize_exact_search_query,
    _prepare_search_query,
    _regulatory_provision_family,
    _regulatory_reference_expansion_limit,
    _reserve_ranked_regulatory_seeds,
    _rich_response_sections,
    _validate_coverage_item,
    _validate_evidence_target,
    _validate_search_mode,
    _validate_search_queries,
)

MODULE = "onyx.tools.tool_implementations.search.search_tool"

# What decide_search_scope returns: the scope to apply now (or None for everything).
ScopeDecision = list[DocumentSource] | None


def test_regulatory_internal_search_accepts_one_focused_query() -> None:
    assert _validate_search_queries(
        {"queries": ["  exact provision  "]},
        regulatory_chunks_only=True,
    ) == ["exact provision"]


def test_regulatory_internal_search_rejects_oversized_query_at_runtime() -> None:
    with pytest.raises(ToolCallException, match="focused-query limit") as exc_info:
        _validate_search_queries(
            {"queries": ["x" * (REGULATORY_MAX_SEARCH_QUERY_CHARS + 1)]},
            regulatory_chunks_only=True,
        )

    assert "Do not paste the full user narrative" in str(
        exc_info.value.llm_facing_message
    )


def test_non_regulatory_internal_search_keeps_existing_query_length_behavior() -> None:
    long_query = "x" * (REGULATORY_MAX_SEARCH_QUERY_CHARS + 1)

    assert _validate_search_queries(
        {"queries": [long_query]}, regulatory_chunks_only=False
    ) == [long_query]


def test_non_regulatory_internal_search_accepts_multiple_queries() -> None:
    queries = ["first query", "second query"]

    assert (
        _validate_search_queries(
            {"queries": queries},
            regulatory_chunks_only=False,
        )
        == queries
    )


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "full_text"])
def test_internal_search_accepts_explicit_search_modes(mode: str) -> None:
    assert (
        _validate_search_mode(
            {"search_mode": mode},
            regulatory_chunks_only=True,
        )
        == mode
    )


def test_internal_search_requires_explicit_search_mode() -> None:
    with pytest.raises(ToolCallException, match="Invalid internal_search mode"):
        _validate_search_mode({}, regulatory_chunks_only=True)


def test_non_regulatory_internal_search_defaults_to_hybrid_mode() -> None:
    assert _validate_search_mode({}, regulatory_chunks_only=False) == "hybrid"


def test_internal_search_requires_explicit_coverage_item() -> None:
    with pytest.raises(ToolCallException, match="non-empty coverage_item"):
        _validate_coverage_item({})

    assert _validate_coverage_item({"coverage_item": "  exact user item  "}) == (
        "exact user item"
    )


def test_internal_search_requires_model_written_evidence_target() -> None:
    with pytest.raises(ToolCallException, match="non-empty evidence_target"):
        _validate_evidence_target({})

    assert (
        _validate_evidence_target(
            {"evidence_target": "  whether the exception applies  "}
        )
        == "whether the exception applies"
    )


def test_regulatory_internal_search_schema_requires_mode_selection() -> None:
    tool = _make_tool(BaseFilters(regulatory_chunks_only=True))
    definition = tool.tool_definition()
    parameters = definition["function"]["parameters"]

    assert "regulatory chunks" in definition["function"]["description"]
    assert "search_mode" in parameters["required"]
    assert "coverage_item" not in parameters["properties"]
    assert "evidence_target" not in parameters["properties"]
    assert parameters["properties"]["search_mode"]["enum"] == [
        "hybrid",
        "keyword",
        "full_text",
    ]
    assert (
        "retain that identifier verbatim"
        in parameters["properties"]["queries"]["description"]
    )
    assert (
        "used without secondary term expansion"
        in parameters["properties"]["queries"]["description"]
    )
    mode_description = parameters["properties"]["search_mode"]["description"]
    assert "high proportion" in mode_description
    assert "every significant analyzed term must occur" not in mode_description


def test_non_regulatory_internal_search_preserves_original_schema_and_description() -> (
    None
):
    definition = _make_tool().tool_definition()
    function = definition["function"]
    parameters = function["parameters"]
    query_schema = parameters["properties"]["queries"]

    assert function["description"] == "Search connected applications for information."
    assert parameters["required"] == ["queries"]
    assert "search_mode" not in parameters["properties"]
    assert "minItems" not in query_schema
    assert "maxItems" not in query_schema
    assert "Query expansion" in query_schema["description"]


def test_search_receipt_preserves_results_and_citation_ids() -> None:
    result = {
        "document": 7,
        "title": "MADDE 7",
        "content": "Controlling provision",
    }
    response = _add_search_receipt(
        json.dumps({"results": [result], "note": "scope note"}),
        coverage_item="User-requested authorization effect",
        evidence_target="whether suspension precedes cancellation",
    )

    payload = json.loads(response)
    assert list(payload) == ["receipt", "results", "note"]
    assert payload["receipt"] == {
        "coverage_item": "User-requested authorization effect",
        "evidence_target": "whether suspension precedes cancellation",
    }
    assert payload["results"] == [result]
    assert payload["note"] == "scope note"


def test_regulatory_navigation_is_metadata_only_and_preserves_results() -> None:
    result = {
        "document": 7,
        "title": "MADDE 7",
        "content": "Controlling provision",
    }
    navigation = RegulatoryProvisionNavigation(
        document_title="Generic Convention",
        entries=(
            RegulatoryProvisionNavigationEntry(
                article_key="madde:10a",
                heading_label="MADDE 10A — A bounded topic hint",
            ),
        ),
    )

    response = _add_regulatory_provision_navigation(
        json.dumps({"results": [result], "note": "scope note"}),
        navigation,
    )

    payload = json.loads(response)
    assert payload["results"] == [result]
    assert payload["note"] == "scope note"
    assert payload["regulatory_provision_navigation"] == {
        "type": "regulatory_provision_heading_navigation",
        "document_title": "Generic Convention",
        "usage_note": (
            "Headings and title/topic hints are navigation leads only; they are "
            "not legal evidence, and an omitted heading is not evidence that a "
            "provision is absent. A lead that could materially change a requested "
            "conclusion remains unresolved until its operative text is retrieved "
            "or the resulting source gap is expressly qualified."
        ),
        "headings": [
            {
                "article_key": "madde:10a",
                "heading_label": "MADDE 10A — A bounded topic hint",
            }
        ],
    }
    assert "content" not in payload["regulatory_provision_navigation"]
    assert "document" not in payload["regulatory_provision_navigation"]


def test_no_regulatory_navigation_keeps_response_byte_identical() -> None:
    response = json.dumps({"results": []}, indent=2)

    assert _add_regulatory_provision_navigation(response, None) == response


def test_focused_regulatory_search_can_use_ranked_selection() -> None:
    section = MagicMock()
    section.center_chunk.regulatory_chunk_id = "rc_exact"

    assert _can_use_ranked_regulatory_selection([section], focused_search=True) is True
    assert (
        _can_use_ranked_regulatory_selection([section], focused_search=False) is False
    )


def test_ranked_selection_requires_every_section_to_be_regulatory() -> None:
    regulatory_section = MagicMock()
    regulatory_section.center_chunk.regulatory_chunk_id = "rc_exact"
    ordinary_section = MagicMock()
    ordinary_section.center_chunk.regulatory_chunk_id = None

    assert (
        _can_use_ranked_regulatory_selection(
            [regulatory_section, ordinary_section], focused_search=True
        )
        is False
    )
    assert _can_use_ranked_regulatory_selection([], focused_search=True) is False


def _regulatory_section(document_id: str, chunk_id: int) -> InferenceSection:
    section = MagicMock()
    section.center_chunk.document_id = document_id
    section.center_chunk.chunk_id = chunk_id
    section.center_chunk.regulatory_chunk_id = f"rc-{document_id}-{chunk_id}"
    return cast(InferenceSection, section)


def _regulatory_chunk(document_id: str, chunk_id: int) -> InferenceChunk:
    chunk = MagicMock()
    chunk.document_id = document_id
    chunk.chunk_id = chunk_id
    chunk.regulatory_chunk_id = f"rc-{document_id}-{chunk_id}"
    chunk.unique_id = f"{document_id}__{chunk_id}"
    return cast(InferenceChunk, chunk)


def _real_regulatory_section(
    document_id: str,
    chunk_id: int,
    *,
    heading_path: list[str],
) -> InferenceSection:
    chunk = InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content="authoritative content",
        source_type=DocumentSource.USER_FILE,
        semantic_identifier="Convention",
        title="Convention",
        boost=1,
        score=0.9,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        image_file_id=None,
        source_links={},
        section_continuation=False,
        blurb="authoritative content",
        file_id="stored-file-id",
        regulatory_chunk_id=f"rc-{document_id}-{chunk_id}",
        heading_path=heading_path,
    )
    return inference_section_from_single_chunk(chunk)


def test_ranked_regulatory_selection_reserves_half_for_provision_siblings() -> None:
    sections = [_regulatory_section("doc", index) for index in range(6)]

    assert _reserve_ranked_regulatory_seeds(sections, 4) == sections[:2]
    assert _reserve_ranked_regulatory_seeds(sections, 1) == sections[:1]


@pytest.mark.parametrize(
    ("selected_count", "max_total", "expected_limit"),
    [
        (4, 8, 6),
        (13, 25, 19),
        (1, 2, 2),
        (4, 4, 4),
        (4, 0, 0),
    ],
)
def test_reference_expansion_leaves_bounded_room_for_seed_provision_families(
    selected_count: int,
    max_total: int,
    expected_limit: int,
) -> None:
    assert (
        _regulatory_reference_expansion_limit(selected_count, max_total)
        == expected_limit
    )


def test_ranked_regulatory_backfill_is_exact_chunk_deduplicated() -> None:
    first = _regulatory_section("doc", 1)
    sibling = _regulatory_section("doc", 2)
    fallback = _regulatory_section("other", 3)

    assert _backfill_ranked_regulatory_sections(
        [first, sibling], [first, fallback], 3
    ) == [first, sibling, fallback]


def test_ranked_regulatory_backfill_never_exceeds_section_cap() -> None:
    selected = [_regulatory_section("doc", index) for index in range(3)]
    fallback = _regulatory_section("other", 4)

    assert (
        _backfill_ranked_regulatory_sections(selected, [fallback], max_total_sections=3)
        == selected
    )


def test_regulatory_rich_response_uses_authoritative_repaired_selected_chunk() -> None:
    stale = _real_regulatory_section(
        "doc", 454, heading_path=["Convention", "2. Amount"]
    )
    repaired = _real_regulatory_section(
        "doc", 454, heading_path=["Convention", "MADDE 75", "2. Amount"]
    )

    merged = _rich_response_sections(
        [stale],
        [repaired],
        authoritative_selected=True,
    )

    assert merged == [repaired]
    search_docs = convert_inference_sections_to_search_docs(merged)
    assert search_docs[0].metadata["regulatory_heading_path"] == [
        "Convention",
        "MADDE 75",
        "2. Amount",
    ]


def test_non_regulatory_rich_response_preserves_existing_rich_chunk_behavior() -> None:
    rich = _regulatory_section("doc", 1)
    selected = _regulatory_section("doc", 1)

    merged = _rich_response_sections(
        [rich],
        [selected],
        authoritative_selected=False,
    )

    assert merged == [rich]


def test_regulatory_snapshot_filter_drops_stale_or_missing_index_hits() -> None:
    visible = _regulatory_section("doc", 1)
    stale = _regulatory_section("doc", 2)

    assert _filter_visible_regulatory_sections(
        [visible, stale],
        {visible.center_chunk.regulatory_chunk_id},
    ) == [visible]


def test_focused_regulatory_lane_heads_prevent_one_document_flood() -> None:
    first = _regulatory_chunk("doc-a", 1)
    same_document_second = _regulatory_chunk("doc-a", 2)
    same_document_third = _regulatory_chunk("doc-a", 3)
    semantic_alternative = _regulatory_chunk("doc-b", 4)
    lexical_alternative = _regulatory_chunk("doc-c", 5)
    fused = [
        first,
        same_document_second,
        same_document_third,
        semantic_alternative,
        lexical_alternative,
    ]

    diversified = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        [
            [first, same_document_second, semantic_alternative],
            [semantic_alternative, lexical_alternative],
        ],
        max_chunks=4,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert diversified == [
        first,
        semantic_alternative,
        lexical_alternative,
        same_document_second,
    ]


def test_retrieval_lane_diversity_does_not_touch_non_regulatory_search() -> None:
    first = _regulatory_chunk("doc-a", 1)
    second = _regulatory_chunk("doc-b", 2)
    fused = [first, second]

    assert (
        _diversify_focused_regulatory_retrieval_lanes(
            fused,
            [[second], [first]],
            max_chunks=1,
            focused_search=True,
            regulatory_chunks_only=False,
        )
        == fused
    )


def test_regulatory_retrieval_lane_diversity_is_bounded_and_deterministic() -> None:
    fused = [_regulatory_chunk(f"doc-{index}", index) for index in range(6)]
    lanes = [[fused[2], fused[3]], [fused[4], fused[5]]]

    first_run = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        lanes,
        max_chunks=3,
        focused_search=True,
        regulatory_chunks_only=True,
    )
    second_run = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        lanes,
        max_chunks=3,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert first_run == second_run == [fused[0], fused[2], fused[4]]


def test_single_focused_lane_does_not_displace_multiple_top_ranked_provisions() -> None:
    first = _regulatory_chunk("doc-a", 1)
    same_document_second = _regulatory_chunk("doc-a", 2)
    same_document_third = _regulatory_chunk("doc-a", 3)
    alternative = _regulatory_chunk("doc-b", 4)
    another_alternative = _regulatory_chunk("doc-c", 5)
    fused = [
        first,
        same_document_second,
        same_document_third,
        alternative,
        another_alternative,
    ]

    diversified = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        [[first, same_document_second, alternative, another_alternative]],
        max_chunks=4,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert diversified == [
        first,
        alternative,
        same_document_second,
        same_document_third,
    ]


def test_focused_regulatory_lane_tail_does_not_displace_bounded_fused_hits() -> None:
    first = _regulatory_chunk("doc-a", 1)
    same_document_second = _regulatory_chunk("doc-a", 2)
    same_document_third = _regulatory_chunk("doc-a", 3)
    same_document_fourth = _regulatory_chunk("doc-a", 4)
    deep_lane_alternative = _regulatory_chunk("doc-b", 5)
    fused = [
        first,
        same_document_second,
        same_document_third,
        same_document_fourth,
        deep_lane_alternative,
    ]

    diversified = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        [
            [
                first,
                same_document_second,
                same_document_third,
                same_document_fourth,
                deep_lane_alternative,
            ]
        ],
        max_chunks=3,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert diversified == [first, same_document_second, same_document_third]


def test_regulatory_provision_families_survive_same_document_chunk_flood() -> None:
    annex_hit = _regulatory_chunk("instrument", 1)
    annex_hit.heading_path = ["Generic Convention", "ANNEX VII"]
    supplementary_hits = [
        _regulatory_chunk("supplement", index) for index in range(2, 62)
    ]
    for hit in supplementary_hits:
        hit.heading_path = ["Membership table", f"Row {hit.chunk_id}"]

    article_headings = ["MADDE 4", "MADDE 6", "MADDE 2", "MADDE 4A"]
    article_hits = [
        _regulatory_chunk("instrument", index)
        for index in range(70, 70 + len(article_headings))
    ]
    for hit, heading in zip(article_hits, article_headings):
        hit.heading_path = ["Generic Convention", heading]

    unrelated_article = _regulatory_chunk("unrepresented", 90)
    unrelated_article.heading_path = ["Unrelated Act", "MADDE 99"]
    fused = [annex_hit, *supplementary_hits, *article_hits, unrelated_article]

    diversified = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        [fused],
        max_chunks=50,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert diversified[:6] == [
        annex_hit,
        supplementary_hits[0],
        *article_hits,
    ]
    assert len(diversified) == 50
    assert unrelated_article not in diversified


def test_regulatory_provision_family_seed_deduplicates_paragraphs() -> None:
    annex_hit = _regulatory_chunk("instrument", 1)
    annex_hit.heading_path = ["Generic Convention", "ANNEX"]
    first_paragraph = _regulatory_chunk("instrument", 70)
    first_paragraph.heading_path = ["Generic Convention", "MADDE 4", "1. paragraph"]
    second_paragraph = _regulatory_chunk("instrument", 71)
    second_paragraph.heading_path = ["Generic Convention", "MADDE 4", "2. paragraph"]
    next_article = _regulatory_chunk("instrument", 72)
    next_article.heading_path = ["Generic Convention", "MADDE 4A"]
    fused = [annex_hit, first_paragraph, second_paragraph, next_article]

    diversified = _diversify_focused_regulatory_retrieval_lanes(
        fused,
        [fused],
        max_chunks=4,
        focused_search=True,
        regulatory_chunks_only=True,
    )

    assert diversified == [annex_hit, first_paragraph, next_article, second_paragraph]


def test_regulatory_provision_family_uses_current_anchor_after_legacy_stale_scope() -> (
    None
):
    polluted = _regulatory_chunk("instrument", 75)
    polluted.heading_path = [
        "Generic Convention",
        "MADDE 74",
        "Reference heading",
        "MADDE 75",
        "First paragraph",
    ]
    clean = _regulatory_chunk("instrument", 76)
    clean.heading_path = [
        "Generic Convention",
        "MADDE 75",
        "Second paragraph",
    ]

    assert _regulatory_provision_family(polluted) == _regulatory_provision_family(clean)


def test_exact_search_removes_boolean_syntax_without_dropping_identifiers() -> None:
    assert (
        _normalize_exact_search_query(
            '"named mechanism" OR "ABC" OR "XYZ" AND "1234.56"'
        )
        == "named mechanism ABC XYZ 1234.56"
    )


def test_keyword_search_uses_only_the_model_written_query() -> None:
    assert (
        _prepare_search_query(
            '"named mechanism" OR "exact-code"',
            "keyword",
        )
        == "named mechanism exact-code"
    )


def test_full_text_search_keeps_the_requested_phrase_narrow() -> None:
    assert (
        _prepare_search_query(
            '"yasadışı trafik"',
            "full_text",
        )
        == "yasadışı trafik"
    )


def test_hybrid_search_removes_unparsed_boolean_syntax() -> None:
    assert (
        _prepare_search_query(
            '"TIR Karnesi" AND ("2207.10" OR "etil alkol")',
            "hybrid",
        )
        == "TIR Karnesi 2207.10 etil alkol"
    )


def test_internal_search_rejects_unknown_search_mode() -> None:
    with pytest.raises(ToolCallException, match="Invalid internal_search mode"):
        _validate_search_mode({"search_mode": "exactish"})


def test_parallel_search_fork_has_isolated_cycle_state() -> None:
    tool = _make_tool()
    tool._search_cycles.append(MagicMock())

    forked = tool.fork_for_parallel_call()

    assert forked is not tool
    assert forked.document_index is tool.document_index
    assert forked.llm is tool.llm
    assert forked._search_cycles == []


def test_independent_context_fork_shares_no_mutable_search_decisions() -> None:
    tool = _make_tool()
    tool._search_cycles.append(MagicMock())
    tool._cached_expansion = ("prior semantic query", ["prior keyword query"])
    tool._scope_decision_settled = True
    tool._time_filter_computed = True

    forked = tool.fork_for_independent_context()

    assert forked is not tool
    assert forked.document_index is tool.document_index
    assert forked.llm is tool.llm
    assert forked._search_cycles == []
    assert forked._cached_expansion is None
    assert not forked._scope_decision_settled
    assert not forked._time_filter_computed
    assert forked._shared_time_filter_decision is not tool._shared_time_filter_decision


def test_parallel_search_batch_shares_scope_and_turn_time_decisions() -> None:
    tool = _make_tool()
    forks = tool.fork_for_parallel_calls(2)
    history = [
        ChatMinimalTextMessage(
            message="Search the named source for documents updated after 2025.",
            message_type=MessageType.USER,
        )
    ]
    batch_queries = ["first focused query", "second focused query"]
    connected_sources = [DocumentSource.CONFLUENCE, DocumentSource.GITHUB]
    decided_time = TimeFilter(
        field=DocumentTimeField.UPDATED_AT,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    decide_scope = MagicMock(return_value=[DocumentSource.CONFLUENCE])
    decide_time = MagicMock(return_value=decided_time)
    start_barrier = Barrier(len(forks))

    def run_fork(index: int, fork: SearchTool) -> QueryExpansionAndScope:
        start_barrier.wait()
        return fork._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[
                ChatMinimalTextMessage(
                    message=f"focused history {index}",
                    message_type=MessageType.USER,
                )
            ],
            filter_message_history=history,
            user_info=None,
            memories=[],
            decide_args=(
                history,
                tool.llm,
                connected_sources,
                [],
                batch_queries,
            ),
        )

    with (
        patch(f"{MODULE}.decide_search_scope", decide_scope),
        patch(f"{MODULE}.decide_time_filter", decide_time),
    ):
        with ThreadPoolExecutor(max_workers=len(forks)) as executor:
            fork_results = list(executor.map(run_fork, range(len(forks)), forks))

        # A later, non-parallel search may reroute sources, but the turn-level
        # document-time decision remains shared.
        direct_result = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=history,
            user_info=None,
            memories=[],
            decide_args=(
                history,
                tool.llm,
                connected_sources,
                [],
                ["later retry query"],
            ),
        )

    assert decide_scope.call_count == 2
    assert decide_time.call_count == 1
    assert all(
        result.plan_scope == [DocumentSource.CONFLUENCE]
        and result.time_filter == decided_time
        for result in fork_results
    )
    assert direct_result.time_filter == decided_time


@pytest.mark.parametrize(
    "queries",
    [[], ["one", "two"], [""], "one"],
)
def test_internal_search_rejects_non_focused_query_payloads(queries: Any) -> None:
    with pytest.raises(ToolCallException, match="exactly one non-empty query"):
        _validate_search_queries({"queries": queries})


def _make_tool(
    user_selected_filters: BaseFilters | None = None,
    auto_detect_filters: bool = True,
) -> SearchTool:
    """Instantiate SearchTool with non-DB deps mocked; DB/LLM calls are patched in _run."""
    return SearchTool(
        tool_id=1,
        emitter=MagicMock(),
        user=MagicMock(is_anonymous=False),
        persona_search_info=MagicMock(document_set_names=[]),
        llm=MagicMock(),
        document_index=MagicMock(),
        user_selected_filters=user_selected_filters,
        project_id_filter=None,
        enable_slack_search=False,
        auto_detect_filters=auto_detect_filters,
    )


def _run(
    tool: SearchTool,
    *,
    connected_sources: list[DocumentSource],
    decision: ScopeDecision = None,
    decide_mock: MagicMock | None = None,
    skip_query_expansion: bool = False,
    filter_message_history: list[ChatMinimalTextMessage] | None = None,
    filter_queries: list[str] | None = None,
    response_sink: list[ToolResponse] | None = None,
    query: str = "ticket",
    queries: list[str] | None = None,
    search_mode: str | None = "hybrid",
    expanded_keyword_queries: list[str] | None = None,
) -> MagicMock:
    """Run tool.run() with all DB/LLM deps mocked; returns the search_pipeline mock.

    decide_search_scope is replaced by `decide_mock` when given (so its call args
    can be inspected), otherwise by a stub returning `decision`. search_pipeline
    returns no chunks, so run() takes the empty-results early return.
    """
    mock_search_pipeline = MagicMock(return_value=[])
    decide = (
        decide_mock if decide_mock is not None else MagicMock(return_value=decision)
    )
    with (
        patch(f"{MODULE}.get_session_with_current_tenant") as mock_session_ctx,
        patch(f"{MODULE}.build_access_filters_for_user", return_value=[]),
        patch(f"{MODULE}.get_current_search_settings", return_value=MagicMock()),
        patch(f"{MODULE}.EmbeddingModel"),
        patch(f"{MODULE}.get_federated_retrieval_functions", return_value=[]),
        patch(
            f"{MODULE}.fetch_unique_document_sources", return_value=connected_sources
        ),
        patch(f"{MODULE}.semantic_query_rephrase", return_value="rephrased query"),
        patch(
            f"{MODULE}.keyword_query_expansion",
            return_value=expanded_keyword_queries or [],
        ),
        patch(f"{MODULE}.decide_search_scope", decide),
        patch(f"{MODULE}.decide_time_filter", MagicMock(return_value=None)),
        patch(f"{MODULE}.weighted_reciprocal_rank_fusion", return_value=[]),
        patch(f"{MODULE}.search_pipeline", mock_search_pipeline),
    ):
        mock_session_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session_ctx.return_value.__exit__ = MagicMock(return_value=False)
        tool_kwargs: dict[str, Any] = {
            "queries": queries if queries is not None else [query],
            "coverage_item": "resolve the ticket",
            "evidence_target": "whether the ticket can be resolved",
        }
        if search_mode is not None:
            tool_kwargs["search_mode"] = search_mode
        response = tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=SearchToolOverrideKwargs(
                starting_citation_num=1,
                original_query="resolve the ticket",
                message_history=[
                    ChatMinimalTextMessage(
                        message="resolve the ticket",
                        message_type=MessageType.USER,
                    )
                ],
                filter_message_history=filter_message_history,
                filter_queries=filter_queries,
                skip_query_expansion=skip_query_expansion,
            ),
            **tool_kwargs,
        )
        if response_sink is not None:
            response_sink.append(response)
    return mock_search_pipeline


def _filters_passed_to_search(mock_search_pipeline: MagicMock) -> list[Any]:
    return [
        call.kwargs["chunk_search_request"].user_selected_filters
        for call in mock_search_pipeline.call_args_list
    ]


def _queries_sent(mock_search_pipeline: MagicMock) -> list[str]:
    return [
        call.kwargs["chunk_search_request"].query
        for call in mock_search_pipeline.call_args_list
    ]


def _query_modes_sent(
    mock_search_pipeline: MagicMock,
) -> list[tuple[str, float | None]]:
    return [
        (
            call.kwargs["chunk_search_request"].query,
            call.kwargs["chunk_search_request"].hybrid_alpha,
        )
        for call in mock_search_pipeline.call_args_list
    ]


def _query_requests_sent(
    mock_search_pipeline: MagicMock,
) -> list[tuple[str, float | None, bool, int | None]]:
    return [
        (
            request.query,
            request.hybrid_alpha,
            request.high_term_coverage,
            request.limit,
        )
        for call in mock_search_pipeline.call_args_list
        if (request := call.kwargs["chunk_search_request"])
    ]


def test_non_regulatory_hybrid_search_preserves_original_query_expansion_lanes() -> (
    None
):
    tool = _make_tool(auto_detect_filters=False)
    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
        search_mode=None,
        expanded_keyword_queries=["expanded keyword query"],
    )

    query_modes = _query_modes_sent(mock_search_pipeline)
    assert ("ticket", None) in query_modes
    assert ("expanded keyword query", 0.2) in query_modes
    assert all(alpha != 0.0 for _, alpha in query_modes)


def test_non_regulatory_run_accepts_multiple_queries_without_mode_or_normalization() -> (
    None
):
    queries = [
        '"first exact phrase" AND code',
        "second independent query",
    ]
    mock_search_pipeline = _run(
        _make_tool(auto_detect_filters=False),
        connected_sources=[DocumentSource.USER_FILE],
        skip_query_expansion=True,
        queries=queries,
        search_mode=None,
    )

    requests = _query_requests_sent(mock_search_pipeline)
    assert any(
        query == queries[0] and alpha is None and not strict
        for query, alpha, strict, _ in requests
    )
    assert any(
        query == queries[1] and alpha is None and not strict
        for query, alpha, strict, _ in requests
    )


def test_regulatory_explicit_hybrid_query_keeps_broad_and_structural_lanes() -> None:
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )
    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
        skip_query_expansion=True,
        query="Basel Sözleşmesi Madde 8 tamamlanamayan hareket",
    )

    requests = _query_requests_sent(mock_search_pipeline)
    assert any(alpha is None and not strict for _, alpha, strict, _ in requests)
    assert any(
        alpha == 0.0 and not strict and limit == 128
        for _, alpha, strict, limit in requests
    )


def test_search_mode_schema_distinguishes_single_and_multiple_exact_anchors() -> None:
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )

    mode_description = tool.tool_definition()["function"]["parameters"]["properties"][
        "search_mode"
    ]["description"]

    assert "literal provision identifier" in mode_description
    assert "numeric value" in mode_description
    assert "remaining terms may be optional" in mode_description
    assert "literal alternatives may occur in different chunks" in mode_description
    assert "multiple exact anchors" in mode_description
    assert "must co-occur lexically in the same chunk" in mode_description
    assert "terms whose co-occurrence is actually required" in mode_description


@pytest.mark.parametrize(
    ("search_mode", "high_term_coverage"),
    [("keyword", False), ("full_text", True)],
)
def test_regulatory_explicit_lexical_mode_keeps_structural_and_strictness_separate(
    search_mode: str,
    high_term_coverage: bool,
) -> None:
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )
    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
        query="Basel md4a ihracat yasağı",
        search_mode=search_mode,
    )

    assert _query_requests_sent(mock_search_pipeline) == [
        ("Basel md4a ihracat yasağı", 0.0, high_term_coverage, 128)
    ]


def test_regulatory_concept_search_overfetches_candidates_without_more_llm_chunks() -> (
    None
):
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )
    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
        query="export restriction to states outside a treaty annex",
        search_mode="keyword",
    )

    assert _query_requests_sent(mock_search_pipeline) == [
        ("export restriction to states outside a treaty annex", 0.0, False, 128)
    ]


def test_non_regulatory_skip_expansion_uses_only_ordinary_semantic_lanes() -> None:
    mock_search_pipeline = _run(
        _make_tool(auto_detect_filters=False),
        connected_sources=[DocumentSource.USER_FILE],
        skip_query_expansion=True,
        query="Convention Article 8 unfinished movement",
    )

    requests = _query_requests_sent(mock_search_pipeline)
    assert any(alpha is None and not strict for _, alpha, strict, _ in requests)
    assert all(
        alpha is None and not strict and limit != 128
        for _, alpha, strict, limit in requests
    )


def test_exact_provision_lane_rejects_descendant_cross_reference() -> None:
    exact = _regulatory_chunk("basel", 64)
    exact.heading_path = ["Bazel Sözleşmesi", "MADDE 8"]
    cross_reference = _regulatory_chunk("other", 83)
    cross_reference.heading_path = [
        "Başka Sözleşme",
        "MADDE 13 - Bilgi Aktarımı",
        "MADDE 8 uyarınca yapılan bildirim",
    ]
    no_heading = _regulatory_chunk("legacy", 1)
    no_heading.heading_path = None
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )

    with patch(
        f"{MODULE}.search_pipeline",
        return_value=[cross_reference, no_heading, exact],
    ) as search_pipeline_mock:
        results = tool._run_search_for_query(
            query="Basel Sözleşmesi Madde 8 tamamlanamayan hareket",
            hybrid_alpha=0.0,
            high_term_coverage=False,
            num_hits=8,
            acl_filters=[],
            embedding_model=MagicMock(),
            federated_retrieval_infos=[],
            effective_filters=BaseFilters(regulatory_chunks_only=True),
            provision_reference=RegulatoryProvisionReference("8", None),
        )

    assert results == [exact]
    request = search_pipeline_mock.call_args.kwargs["chunk_search_request"]
    assert request.limit == 32
    assert request.high_term_coverage is False


@pytest.mark.parametrize(
    ("num_hits", "expected_candidate_limit"),
    [(8, 32), (32, 128), (50, 128), (160, 160)],
)
def test_regulatory_candidate_overfetch_is_bounded_without_shrinking_output_contract(
    num_hits: int,
    expected_candidate_limit: int,
) -> None:
    tool = _make_tool(
        BaseFilters(regulatory_chunks_only=True),
        auto_detect_filters=False,
    )

    with patch(f"{MODULE}.search_pipeline", return_value=[]) as search_pipeline_mock:
        tool._run_search_for_query(
            query="focused unknown provision relationship",
            hybrid_alpha=0.0,
            high_term_coverage=False,
            num_hits=num_hits,
            acl_filters=[],
            embedding_model=MagicMock(),
            federated_retrieval_infos=[],
            effective_filters=BaseFilters(regulatory_chunks_only=True),
        )

    request = search_pipeline_mock.call_args.kwargs["chunk_search_request"]
    assert request.limit == expected_candidate_limit


def test_non_regulatory_candidate_search_keeps_requested_limit() -> None:
    tool = _make_tool(auto_detect_filters=False)

    with patch(f"{MODULE}.search_pipeline", return_value=[]) as search_pipeline_mock:
        tool._run_search_for_query(
            query="ordinary connected knowledge",
            hybrid_alpha=None,
            high_term_coverage=False,
            num_hits=8,
            acl_filters=[],
            embedding_model=MagicMock(),
            federated_retrieval_infos=[],
            effective_filters=BaseFilters(),
        )

    request = search_pipeline_mock.call_args.kwargs["chunk_search_request"]
    assert request.limit == 8


def test_evidence_target_is_not_added_to_retrieval_queries() -> None:
    tool = _make_tool(auto_detect_filters=False)
    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
    )

    assert _queries_sent(mock_search_pipeline)
    assert all(
        "whether the ticket can be resolved" not in query
        for query in _queries_sent(mock_search_pipeline)
    )


def test_run_returns_receipt_without_changing_empty_results_shape() -> None:
    responses: list[ToolResponse] = []
    _run(
        _make_tool(
            BaseFilters(regulatory_chunks_only=True),
            auto_detect_filters=False,
        ),
        connected_sources=[DocumentSource.USER_FILE],
        response_sink=responses,
    )

    assert len(responses) == 1
    payload = json.loads(responses[0].llm_facing_response)
    assert payload["receipt"] == {
        "coverage_item": "resolve the ticket",
        "evidence_target": "whether the ticket can be resolved",
    }
    assert payload["results"] == []


def test_non_regulatory_empty_response_does_not_add_regulatory_receipt() -> None:
    responses: list[ToolResponse] = []
    _run(
        _make_tool(auto_detect_filters=False),
        connected_sources=[DocumentSource.USER_FILE],
        response_sink=responses,
        search_mode=None,
    )

    assert len(responses) == 1
    assert "receipt" not in json.loads(responses[0].llm_facing_response)


def _emitted_filter_sources(tool: SearchTool) -> list[list[str]]:
    """Sources of each SearchToolFilterDelta the tool emitted to the UI."""
    emit_mock = cast(MagicMock, tool.emitter.emit)
    emitted = [call.args[0].obj for call in emit_mock.call_args_list]
    return [obj.sources for obj in emitted if isinstance(obj, SearchToolFilterDelta)]


def test_decided_scope_is_passed_to_search() -> None:
    """When the filter flow decides a source, every search runs scoped to it."""
    tool = _make_tool()
    mock_search_pipeline = _run(
        tool,
        decision=[DocumentSource.CONFLUENCE],
        connected_sources=[
            DocumentSource.SLACK,
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
        ],
    )

    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is not None
        assert applied.source_type == [DocumentSource.CONFLUENCE]


def test_filter_decision_uses_full_history_and_all_parallel_queries() -> None:
    tool = _make_tool()
    decide_mock = MagicMock(return_value=[DocumentSource.CONFLUENCE])
    full_history = [
        ChatMinimalTextMessage(
            message="Use Confluence for this request.",
            message_type=MessageType.USER,
        ),
        ChatMinimalTextMessage(
            message="Compare both independent issues.",
            message_type=MessageType.USER,
        ),
    ]
    batch_queries = ["first focused query", "second focused query"]

    _run(
        tool,
        connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB],
        decide_mock=decide_mock,
        filter_message_history=full_history,
        filter_queries=batch_queries,
    )

    assert decide_mock.call_args.args[0] == full_history
    assert decide_mock.call_args.args[4] == batch_queries


def test_parallel_fork_preserves_explicit_regulatory_as_of_date() -> None:
    as_of_date = date(2024, 7, 1)
    tool = _make_tool(
        BaseFilters(as_of_date=as_of_date),
        auto_detect_filters=False,
    ).fork_for_parallel_calls(2)[0]

    mock_search_pipeline = _run(
        tool,
        connected_sources=[DocumentSource.USER_FILE],
    )

    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters
    assert all(applied is not None for applied in filters)
    assert all(applied.as_of_date == as_of_date for applied in filters if applied)


def test_filter_delta_emitted_for_a_subset_scope() -> None:
    """A scope narrower than the connected sources surfaces a filter to the UI."""
    tool = _make_tool()
    _run(
        tool,
        decision=[DocumentSource.CONFLUENCE],
        connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB],
    )
    assert _emitted_filter_sources(tool) == [["confluence"]]


def test_no_filter_delta_when_scope_covers_all_sources() -> None:
    """Scoping to every connected source is equivalent to an unscoped search, so
    no filter is surfaced (the UI keeps its default 'internal documents' label)."""
    tool = _make_tool()
    connected = [DocumentSource.CONFLUENCE, DocumentSource.GITHUB]
    _run(tool, decision=connected, connected_sources=connected)
    assert _emitted_filter_sources(tool) == []


def test_no_decided_scope_leaves_search_unscoped() -> None:
    """A no-scope decision applies no source filter."""
    tool = _make_tool()
    mock_search_pipeline = _run(
        tool,
        decision=None,
        connected_sources=[DocumentSource.SLACK, DocumentSource.CONFLUENCE],
    )

    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is None or applied.source_type is None


def test_persona_restriction_is_refined_by_the_decision() -> None:
    """A persona source restriction is the outer bound; the decision refines
    WITHIN it (here, down to a single source)."""
    tool = _make_tool(
        BaseFilters(
            source_type=[
                DocumentSource.CONFLUENCE,
                DocumentSource.GITHUB,
                DocumentSource.SLACK,
            ]
        )
    )
    mock_search_pipeline = _run(
        tool,
        decision=[DocumentSource.CONFLUENCE],
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
    )

    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is not None
        assert applied.source_type == [DocumentSource.CONFLUENCE]


def test_persona_restriction_applies_when_decision_does_not_route() -> None:
    """With a persona restriction and a no-scope decision, the search stays scoped
    to the restriction (never broadens to everything)."""
    restriction = [DocumentSource.CONFLUENCE, DocumentSource.GITHUB]
    tool = _make_tool(BaseFilters(source_type=restriction))
    mock_search_pipeline = _run(
        tool,
        decision=None,
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
    )

    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is not None
        assert applied.source_type == restriction


def test_cached_expansion_is_reused_on_a_new_filter_not_a_repeat() -> None:
    """The first call expands (and caches). A repeat call on a NOT-yet-searched
    source reuses the cached expansion; a repeat on an already-searched source
    does not (the agent is expected to vary terms there)."""
    tool = _make_tool()
    connected = [DocumentSource.ZENDESK, DocumentSource.ASANA]

    # Call 1: first search (expansion runs) scoped to Zendesk -> caches expansion.
    _run(tool, decision=[DocumentSource.ZENDESK], connected_sources=connected)

    # Call 2: repeat call, walk advanced to Asana (new) -> reuse cached expansion.
    new_filter = _run(
        tool,
        decision=[DocumentSource.ASANA],
        connected_sources=connected,
        skip_query_expansion=True,
    )
    assert "rephrased query" in _queries_sent(new_filter), (
        "cached expansion should be reused when searching a new source"
    )

    # Call 3: repeat call on Asana again (already searched) -> no reuse.
    repeat = _run(
        tool,
        decision=[DocumentSource.ASANA],
        connected_sources=connected,
        skip_query_expansion=True,
    )
    assert "rephrased query" not in _queries_sent(repeat), (
        "a same-source repeat should not re-apply the cached expansion"
    )


def test_no_scope_decision_is_not_repeated_within_a_turn() -> None:
    """Once a cycle's scope decision comes back unscoped, the conversation has no
    source directive (which can't change this turn), so later cycles skip the
    decision instead of burning another LLM call."""
    tool = _make_tool()
    connected = [DocumentSource.ZENDESK, DocumentSource.CONFLUENCE]
    decide_mock = MagicMock(return_value=None)

    _run(tool, decide_mock=decide_mock, connected_sources=connected)
    _run(tool, decide_mock=decide_mock, connected_sources=connected)

    assert decide_mock.call_count == 1, (
        "decide_search_scope should run once, then latch off after a no-scope result"
    )


def test_scope_decision_keeps_running_while_a_directive_is_present() -> None:
    """A routed decision does not latch the skip — the walk must keep deciding on
    later cycles (e.g. to advance a backoff sequence to the next source)."""
    tool = _make_tool()
    connected = [DocumentSource.ZENDESK, DocumentSource.CONFLUENCE]
    decide_mock = MagicMock(
        side_effect=[[DocumentSource.ZENDESK], [DocumentSource.CONFLUENCE]]
    )

    _run(tool, decide_mock=decide_mock, connected_sources=connected)
    _run(tool, decide_mock=decide_mock, connected_sources=connected)

    assert decide_mock.call_count == 2


def test_prior_cycles_accumulate_across_calls_for_the_walk() -> None:
    """A backoff sequence advances: the first call's queries + resolved scope are
    passed back to decide_search_scope as previous_cycles on the second."""
    tool = _make_tool()
    connected = [DocumentSource.ZENDESK, DocumentSource.CONFLUENCE]

    # Mimic the walk: first call routes to Zendesk, second to Confluence.
    decide_mock = MagicMock(
        side_effect=[[DocumentSource.ZENDESK], [DocumentSource.CONFLUENCE]]
    )
    _run(tool, decide_mock=decide_mock, connected_sources=connected)
    _run(tool, decide_mock=decide_mock, connected_sources=connected)

    # previous_cycles is the 4th positional arg.
    first_cycles = decide_mock.call_args_list[0].args[3]
    second_cycles = decide_mock.call_args_list[1].args[3]
    assert first_cycles == []
    assert len(second_cycles) == 1
    assert second_cycles[0].searched_sources == ["zendesk"]
    assert second_cycles[0].queries == ["ticket"]
    assert second_cycles[0].cycle_number == 1


def test_auto_detect_disabled_skips_scope_decision() -> None:
    """With auto-detect off, no scope decision runs and the search stays unscoped."""
    tool = _make_tool(auto_detect_filters=False)
    connected = [DocumentSource.ZENDESK, DocumentSource.CONFLUENCE]
    decide_mock = MagicMock(return_value=[DocumentSource.ZENDESK])

    mock_search_pipeline = _run(
        tool, decide_mock=decide_mock, connected_sources=connected
    )

    decide_mock.assert_not_called()
    assert _emitted_filter_sources(tool) == []
    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is None or applied.source_type is None


def test_auto_detect_disabled_keeps_user_selected_filters() -> None:
    """With auto-detect off, user/persona-selected filters are still applied."""
    restriction = [DocumentSource.CONFLUENCE, DocumentSource.GITHUB]
    tool = _make_tool(BaseFilters(source_type=restriction), auto_detect_filters=False)
    decide_mock = MagicMock(return_value=[DocumentSource.CONFLUENCE])

    mock_search_pipeline = _run(
        tool,
        decide_mock=decide_mock,
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
    )

    decide_mock.assert_not_called()
    filters = _filters_passed_to_search(mock_search_pipeline)
    assert filters, "search_pipeline was never called"
    for applied in filters:
        assert applied is not None
        assert applied.source_type == restriction
