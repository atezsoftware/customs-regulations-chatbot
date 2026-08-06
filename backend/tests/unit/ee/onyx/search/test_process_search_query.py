from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ee.onyx.search.process_search_query import stream_search_query
from ee.onyx.server.query_and_chat.models import SendSearchQueryRequest
from ee.onyx.server.query_and_chat.streaming_models import SearchQueriesPacket
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.reranking.models import RerankOutcome, RerankResult

MODULE = "ee.onyx.search.process_search_query"


def _chunk(document_id: str, chunk_id: int) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=chunk_id,
        blurb=f"{document_id}-{chunk_id}",
        content=f"{document_id}-{chunk_id}",
        source_links=None,
        image_file_id=None,
        section_continuation=False,
        document_id=document_id,
        source_type=DocumentSource.USER_FILE,
        semantic_identifier=document_id,
        title=document_id,
        boost=0,
        score=1.0,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )


def test_end_user_search_enables_query_expansion_by_default() -> None:
    assert SendSearchQueryRequest(search_query="hukuki soru").run_query_expansion


def test_search_ui_builds_bounded_semantic_and_turkish_legal_lanes() -> None:
    original_query = (
        'Yargıtayın 01.01.2024 tarihli kararında "mücbir sebep" ve TBK m. 136'
    )
    semantic_query = (
        'Yargıtayın 01.01.2024 tarihli kararında "mücbir sebep" kavramının '
        "TBK m. 136 kapsamında uygulanması"
    )
    keyword_queries = [
        '"mücbir sebep" TBK m. 136',
        "Yargıtay 01.01.2024",
        "borcun ifasının imkânsızlaşması",
        "kusursuz sonraki imkânsızlık",
    ]
    rerank_result = RerankResult(
        ordered_chunks=[],
        scores_by_chunk={},
        submitted_count=0,
        result_count=0,
        outcome=RerankOutcome.DISABLED,
        fallback_used=True,
    )

    with (
        patch(f"{MODULE}.get_current_search_settings", return_value=MagicMock()),
        patch(f"{MODULE}.get_default_document_index", return_value=MagicMock()),
        patch(f"{MODULE}.get_reranker_configuration", return_value=MagicMock()),
        patch(f"{MODULE}.get_default_llm", return_value=MagicMock()),
        patch(
            f"{MODULE}.expand_search_queries",
            return_value=(semantic_query, keyword_queries),
        ),
        patch(f"{MODULE}._run_single_search", return_value=[]) as run_search,
        patch(f"{MODULE}.rerank_chunks", return_value=rerank_result),
    ):
        packets = list(
            stream_search_query(
                SendSearchQueryRequest(search_query=original_query),
                MagicMock(is_anonymous=True),
                MagicMock(),
            )
        )

    queries_packet = next(
        packet for packet in packets if isinstance(packet, SearchQueriesPacket)
    )
    assert queries_packet.all_executed_queries == [
        original_query,
        semantic_query,
        *keyword_queries[:3],
    ]
    hybrid_alpha_by_query = {
        call.args[0]: call.args[6] for call in run_search.call_args_list
    }
    assert hybrid_alpha_by_query == {
        original_query: None,
        semantic_query: None,
        keyword_queries[0]: 0.2,
        keyword_queries[1]: 0.2,
        keyword_queries[2]: 0.2,
    }


def test_search_ui_reranks_before_merge_and_truncation() -> None:
    fused_chunks = [_chunk("doc", index) for index in range(4)]
    reranked_chunks = list(reversed(fused_chunks))
    rerank_result = RerankResult(
        ordered_chunks=reranked_chunks,
        scores_by_chunk={
            (chunk.document_id, chunk.chunk_id): 1.0 - (rank * 0.1)
            for rank, chunk in enumerate(reranked_chunks)
        },
        submitted_count=4,
        result_count=4,
        outcome=RerankOutcome.SUCCESS,
        fallback_used=False,
    )
    user = MagicMock(is_anonymous=True)

    with (
        patch(f"{MODULE}.get_current_search_settings", return_value=MagicMock()),
        patch(f"{MODULE}.get_default_document_index", return_value=MagicMock()),
        patch(f"{MODULE}.get_reranker_configuration", return_value=MagicMock()),
        patch(f"{MODULE}._run_single_search", return_value=fused_chunks),
        patch(f"{MODULE}.rerank_chunks", return_value=rerank_result) as rerank,
        patch(
            f"{MODULE}.apply_soft_diversity", return_value=reranked_chunks
        ) as diversify,
        patch(f"{MODULE}.merge_individual_chunks", return_value=[]) as merge,
    ):
        list(
            stream_search_query(
                SendSearchQueryRequest(
                    search_query="asıl hukuki soru",
                    run_query_expansion=False,
                    num_hits=2,
                ),
                user,
                MagicMock(),
            )
        )

    assert rerank.call_args.kwargs["query"] == "asıl hukuki soru"
    assert rerank.call_args.kwargs["chunks"] == fused_chunks
    assert diversify.call_args.kwargs["chunks"] == reranked_chunks
    assert merge.call_args.args[0] == reranked_chunks
