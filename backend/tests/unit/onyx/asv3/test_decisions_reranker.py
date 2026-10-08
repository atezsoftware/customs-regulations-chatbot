from unittest.mock import MagicMock, patch

import pytest

from onyx.context.search.models import InferenceChunk

MODULE = "onyx.asv3.decisions_reranker"


def _chunks(count: int = 20) -> list[InferenceChunk]:
    return [
        InferenceChunk(
            document_id=f"document-{index}",
            chunk_id=index,
            content=f"Operative source passage {index}",
            source_type="user_file",
            semantic_identifier=f"Source {index}",
            title=f"Source {index}",
            boost=1,
            score=1.0,
            hidden=False,
            metadata={},
            match_highlights=[],
            doc_summary="",
            chunk_context="",
            updated_at=None,
            image_file_id=None,
            source_links=None,
            section_continuation=False,
            blurb=f"Source {index}",
            file_id=f"document-{index}",
        )
        for index in range(count)
    ]


def test_disabled_decisions_keeps_deterministic_order_without_creating_client() -> None:
    from onyx.asv3.decisions_reranker import promote_guarded_boundary_candidates

    chunks = _chunks()
    with (
        patch(f"{MODULE}.guarded_decisions_enabled", return_value=False),
        patch(f"{MODULE}._create_client") as create_client,
    ):
        result = promote_guarded_boundary_candidates(
            query="Which source governs the exception?",
            ordered_chunks=chunks,
            baseline_limit=4,
        )

    assert result == chunks
    create_client.assert_not_called()


def test_decisions_promotes_bounded_boundary_candidates_without_demoting_baseline() -> (
    None
):
    from onyx.asv3.decisions_reranker import promote_guarded_boundary_candidates

    chunks = _chunks()
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value.json.return_value = {
        "answers": [
            {"name": "candidate_0", "type": "predicate", "probability": 0.1},
            {"name": "candidate_2", "type": "predicate", "probability": 0.95},
            {"name": "candidate_4", "type": "predicate", "probability": 0.9},
            {"name": "candidate_6", "type": "predicate", "probability": 0.85},
            {"name": "candidate_8", "type": "predicate", "probability": 0.8},
            {"name": "candidate_10", "type": "predicate", "probability": 0.75},
        ]
    }
    with (
        patch(f"{MODULE}.guarded_decisions_enabled", return_value=True),
        patch(f"{MODULE}._create_client", return_value=client),
    ):
        result = promote_guarded_boundary_candidates(
            query="Which source governs the exception?",
            ordered_chunks=chunks,
            baseline_limit=4,
        )

    assert result[:4] == chunks[:4]
    assert result[4:8] == [chunks[6], chunks[8], chunks[10], chunks[12]]
    assert {(chunk.document_id, chunk.chunk_id) for chunk in result} == {
        (chunk.document_id, chunk.chunk_id) for chunk in chunks
    }
    post_args = client.post.call_args
    assert post_args.args == ("https://api.openai.com/v1/decisions",)
    assert post_args.kwargs["json"]["model"] == "gpt-6-luna"
    assert len(post_args.kwargs["json"]["questions"]) == 12
    client.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "response",
    [
        TimeoutError("deadline"),
        {"answers": [{"name": "candidate_0", "type": "refusal"}]},
        {"answers": [{"name": "unknown", "type": "predicate"}]},
    ],
)
def test_decisions_failures_keep_deterministic_order(response: object) -> None:
    from onyx.asv3.decisions_reranker import promote_guarded_boundary_candidates

    chunks = _chunks()
    client = MagicMock()
    if isinstance(response, Exception):
        client.post.side_effect = response
    else:
        client.post.return_value.json.return_value = response
    with (
        patch(f"{MODULE}.guarded_decisions_enabled", return_value=True),
        patch(f"{MODULE}._create_client", return_value=client),
    ):
        result = promote_guarded_boundary_candidates(
            query="Which source governs the exception?",
            ordered_chunks=chunks,
            baseline_limit=4,
        )

    assert result == chunks
