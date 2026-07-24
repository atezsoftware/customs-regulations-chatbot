"""Tests for the cross-encoder rerank step in the amendment candidate finder.

`find_candidates` sends up to `limit*3` merged hybrid-search candidates
through `_rank` before `matcher.py`'s `confirm_match` sends the surviving
ones' *full* chunk text to the LLM — this is the highest-value rerank
target in the codebase (see `rerank.py`'s module docstring), so it's worth
covering both the success path and the heuristic fallback directly.
"""

from unittest.mock import Mock, patch

from fs_explorer_api.amendments.candidate_finder import _rank
from fs_explorer_api.amendments.ranker import CandidateChunk


def _candidate(chunk_id: str, **overrides) -> CandidateChunk:
    defaults = dict(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        relative_path=f"{chunk_id}.txt",
        text=f"text for {chunk_id}",
    )
    defaults.update(overrides)
    return CandidateChunk(**defaults)


def test_rank_uses_reranker_ordering_when_available() -> None:
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    fake_reranker = Mock(rerank=Mock(return_value=[(2, 0.9), (0, 0.5)]))

    with patch(
        "fs_explorer_api.amendments.candidate_finder.get_reranker",
        return_value=fake_reranker,
    ):
        result = _rank("instruction text", candidates, limit=2)

    assert [c.chunk_id for c in result] == ["c", "a"]


def test_rank_falls_back_to_heuristic_when_reranker_returns_none() -> None:
    candidates = [
        _candidate("a", semantic_score=0.1),
        _candidate("b", semantic_score=0.9),
    ]
    fake_reranker = Mock(rerank=Mock(return_value=None))

    with patch(
        "fs_explorer_api.amendments.candidate_finder.get_reranker",
        return_value=fake_reranker,
    ):
        result = _rank("instruction text", candidates, limit=2)

    # Falls back to rank_candidates' combined_score heuristic: higher
    # semantic_score wins.
    assert [c.chunk_id for c in result] == ["b", "a"]


def test_rank_falls_back_to_heuristic_when_reranker_unavailable() -> None:
    candidates = [
        _candidate("a", semantic_score=0.1),
        _candidate("b", semantic_score=0.9),
    ]

    with patch(
        "fs_explorer_api.amendments.candidate_finder.get_reranker",
        return_value=None,
    ):
        result = _rank("instruction text", candidates, limit=2)

    assert [c.chunk_id for c in result] == ["b", "a"]


def test_rank_single_candidate_skips_reranker_call() -> None:
    candidates = [_candidate("solo")]
    fake_reranker = Mock(rerank=Mock())

    with patch(
        "fs_explorer_api.amendments.candidate_finder.get_reranker",
        return_value=fake_reranker,
    ):
        result = _rank("instruction text", candidates, limit=5)

    fake_reranker.rerank.assert_not_called()
    assert [c.chunk_id for c in result] == ["solo"]
