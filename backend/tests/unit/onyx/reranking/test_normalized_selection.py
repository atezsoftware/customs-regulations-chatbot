from collections.abc import Sequence

import pytest

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankPayloadLimits, RerankResult
from onyx.reranking.normalized_selection import (
    normalize_rerank_scores,
    select_normalized_rerank_candidates,
)
from onyx.reranking.payload import serialize_rerank_candidates
from onyx.reranking.staged import rerank_regulatory_candidates_in_batches
from shared_configs.enums import RerankerProvider
from tests.unit.onyx.regulatory.labeling.test_search_overlay import chunk


def _key(candidate: InferenceChunk) -> tuple[str, int]:
    return candidate.document_id, candidate.chunk_id


def _config() -> RerankerRuntimeConfig:
    return RerankerRuntimeConfig(
        enabled=True,
        provider_type=RerankerProvider.SILICONFLOW,
        model_name="Qwen/Qwen3-Reranker-8B",
        api_key=None,
        configuration_generation="test-generation",
    )


def _result(
    candidates: Sequence[InferenceChunk],
    scores: dict[tuple[str, int], float],
    *,
    outcome: RerankOutcome = RerankOutcome.SUCCESS,
) -> RerankResult:
    return RerankResult(
        ordered_chunks=list(reversed(candidates)),
        scores_by_chunk=scores,
        submitted_count=len(candidates),
        result_count=len(scores),
        outcome=outcome,
        fallback_used=outcome is not RerankOutcome.SUCCESS,
    )


@pytest.mark.parametrize("raw_scores", [[0.0, 0.2, 0.9, 1.0], [0.2, 0.24, 0.38, 0.4]])
def test_normalization_uses_the_complete_pool_and_inclusive_threshold(
    raw_scores: list[float],
) -> None:
    candidates = [chunk(str(index)) for index in range(4)]
    scores = dict(zip(map(_key, candidates), raw_scores, strict=True))

    selected = select_normalized_rerank_candidates(
        chunks=candidates, scores=scores, baseline_limit=1
    )

    assert selected.ordered_chunks == [candidates[0], candidates[2], candidates[3]]
    assert selected.qualified_chunk_ids == [_key(candidates[2]), _key(candidates[3])]
    assert selected.normalized_scores_by_chunk[_key(candidates[1])] == pytest.approx(
        0.2
    )
    assert selected.normalized_scores_by_chunk[_key(candidates[2])] == pytest.approx(
        0.9
    )
    assert selected.threshold == 0.90


def test_selection_preserves_baseline_and_all_30_high_score_ties() -> None:
    candidates = [chunk(str(index)) for index in range(60)]
    scores = {
        _key(item): 0.0 if index < 30 else 1.0 for index, item in enumerate(candidates)
    }

    selected = select_normalized_rerank_candidates(
        chunks=candidates, scores=scores, baseline_limit=25
    )

    assert selected.ordered_chunks == [*candidates[:25], *candidates[30:]]
    assert len(selected.ordered_chunks) == 55
    assert selected.qualified_chunk_ids == [_key(item) for item in candidates[30:]]
    assert len({_key(item) for item in selected.ordered_chunks}) == 55


def test_equal_positive_scores_preserve_every_tied_candidate() -> None:
    candidates = [chunk(str(index)) for index in range(30)]
    scores = {_key(item): 0.35 for item in candidates}

    selected = select_normalized_rerank_candidates(
        chunks=candidates, scores=scores, baseline_limit=25
    )

    assert selected.ordered_chunks == candidates
    assert set(selected.normalized_scores_by_chunk.values()) == {1.0}
    assert selected.qualified_chunk_ids == list(scores)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_uninformative_scores_do_not_expand_the_baseline(value: float) -> None:
    candidates = [chunk(str(index)) for index in range(30)]

    selected = select_normalized_rerank_candidates(
        chunks=candidates,
        scores={_key(item): value for item in candidates},
        baseline_limit=25,
    )

    assert selected.ordered_chunks == candidates[:25]
    assert selected.qualified_chunk_ids == []
    assert all(score == 0.0 for score in selected.normalized_scores_by_chunk.values())


def test_unknown_and_nonfinite_scores_cannot_become_threshold_candidates() -> None:
    candidates = [chunk(str(index)) for index in range(6)]
    scores = {
        _key(candidates[0]): 0.0,
        _key(candidates[1]): 1.0,
        _key(candidates[3]): float("nan"),
        _key(candidates[4]): float("inf"),
        ("unrelated-document", 999): 100.0,
    }

    selected = select_normalized_rerank_candidates(
        chunks=candidates, scores=scores, baseline_limit=2
    )

    assert selected.ordered_chunks == candidates[:2]
    assert selected.qualified_chunk_ids == [_key(candidates[1])]


def test_normalization_does_not_mutate_provider_scores() -> None:
    original = {("source", 1): 2.0, ("source", 2): 4.0}

    assert normalize_rerank_scores(original) == {("source", 1): 0.0, ("source", 2): 1.0}
    assert original == {("source", 1): 2.0, ("source", 2): 4.0}


def test_extreme_finite_scores_do_not_overflow_the_normalized_band() -> None:
    scores = {("source", 1): -1e308, ("source", 2): 0.0, ("source", 3): 1e308}

    assert normalize_rerank_scores(scores) == {
        ("source", 1): 0.0,
        ("source", 2): 0.5,
        ("source", 3): 1.0,
    }


def test_finite_negative_scores_participate_in_global_normalization() -> None:
    assert normalize_rerank_scores(
        {("source", 1): -2.0, ("source", 2): -1.0, ("source", 3): 0.0}
    ) == {("source", 1): 0.0, ("source", 2): 0.5, ("source", 3): 1.0}


@pytest.mark.parametrize("score_offset", [0.0, -1.0])
def test_batched_rerank_preserves_all_96_scores_without_a_finalist_call(
    score_offset: float,
) -> None:
    candidates = [chunk(str(index)) for index in range(96)]
    scores = {_key(item): item.chunk_id / 100 + score_offset for item in candidates}
    call_sizes: list[int] = []
    submitted_ids: list[tuple[str, int]] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query"
        assert config.enabled
        call_sizes.append(len(chunks))
        submitted_ids.extend(map(_key, chunks))
        return _result(chunks, {_key(item): scores[_key(item)] for item in chunks})

    result = rerank_regulatory_candidates_in_batches(
        query="focused query", chunks=candidates, config=_config(), rerank=rerank
    )

    assert call_sizes == [32, 32, 32]
    assert submitted_ids == list(scores)
    assert result.used_external
    assert result.scores_by_chunk == scores
    assert result.submitted_count == 96
    assert result.result_count == 96
    assert result.ordered_chunks == list(reversed(candidates))


def test_later_batch_high_score_is_compared_globally() -> None:
    candidates = [chunk(str(index)) for index in range(64)]
    scores = {_key(item): 0.2 for item in candidates}
    scores[_key(candidates[0])] = 0.0
    scores[_key(candidates[33])] = 1.0

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query" and config.enabled
        return _result(chunks, {_key(item): scores[_key(item)] for item in chunks})

    result = rerank_regulatory_candidates_in_batches(
        query="focused query", chunks=candidates, config=_config(), rerank=rerank
    )
    selected = select_normalized_rerank_candidates(
        chunks=candidates, scores=result.scores_by_chunk, baseline_limit=25
    )

    assert selected.ordered_chunks == [*candidates[:25], candidates[33]]
    assert selected.qualified_chunk_ids == [_key(candidates[33])]
    assert selected.normalized_scores_by_chunk[_key(candidates[31])] == pytest.approx(
        0.2
    )


@pytest.mark.parametrize("failure", ["missing_score", "nonfinite_score", "timeout"])
def test_incomplete_batch_restores_complete_baseline_without_guessed_scores(
    failure: str,
) -> None:
    candidates = [chunk(str(index)) for index in range(96)]
    call_sizes: list[int] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query" and config.enabled
        call_sizes.append(len(chunks))
        scores = {_key(item): 1.0 for item in chunks}
        outcome = RerankOutcome.SUCCESS
        if len(call_sizes) == 2:
            if failure == "missing_score":
                scores.pop(_key(chunks[-1]))
            elif failure == "nonfinite_score":
                scores[_key(chunks[-1])] = float("nan")
            else:
                outcome = RerankOutcome.TIMEOUT
        return _result(chunks, scores, outcome=outcome)

    result = rerank_regulatory_candidates_in_batches(
        query="focused query", chunks=candidates, config=_config(), rerank=rerank
    )

    assert call_sizes == [32, 32]
    assert result.ordered_chunks == candidates
    assert result.fallback_used
    assert not result.used_external
    assert result.scores_by_chunk == {}


def test_payload_budgets_can_require_more_than_two_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [chunk(str(index)) for index in range(7)]
    for item in candidates:
        item.content = "A source passage with substantial operative detail. " * 30
    limits = RerankPayloadLimits(
        max_candidates=32,
        max_document_bytes=512,
        max_document_tokens=600,
        max_total_bytes=512,
        max_total_tokens=600,
    )
    monkeypatch.setattr(
        "onyx.reranking.staged.payload_limits_for_reranker", lambda _model: limits
    )
    submitted_ids: list[tuple[str, int]] = []
    call_sizes: list[int] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query" and config.enabled
        payload = serialize_rerank_candidates(chunks, limits=limits)
        assert payload.unsent_chunks == []
        assert payload.utf8_bytes <= limits.max_total_bytes
        assert payload.estimated_tokens <= limits.max_total_tokens
        call_sizes.append(len(chunks))
        submitted_ids.extend(map(_key, chunks))
        return _result(chunks, {_key(item): 1.0 for item in chunks})

    result = rerank_regulatory_candidates_in_batches(
        query="focused query", chunks=candidates, config=_config(), rerank=rerank
    )

    assert call_sizes == [1] * 7
    assert submitted_ids == [_key(item) for item in candidates]
    assert result.used_external
    assert len(result.scores_by_chunk) == 7
