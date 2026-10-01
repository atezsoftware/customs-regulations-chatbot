from collections.abc import Sequence

from onyx.context.search.models import InferenceChunk
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.reranking.staged import rerank_regulatory_candidates_in_stages
from tests.unit.onyx.regulatory.labeling.test_search_overlay import chunk


def _config() -> RerankerRuntimeConfig:
    return RerankerRuntimeConfig(
        enabled=True,
        provider_type=None,
        model_name=None,
        api_key=None,
        configuration_generation="test-generation",
    )


def _result(
    chunks: Sequence[InferenceChunk],
    *,
    outcome: RerankOutcome = RerankOutcome.SUCCESS,
) -> RerankResult:
    return RerankResult(
        ordered_chunks=list(chunks),
        scores_by_chunk={},
        submitted_count=len(chunks),
        result_count=len(chunks) if outcome is RerankOutcome.SUCCESS else 0,
        outcome=outcome,
        fallback_used=outcome is not RerankOutcome.SUCCESS,
    )


def test_staged_rerank_exposes_96_candidates_in_bounded_calls() -> None:
    candidates = [chunk(str(index)) for index in range(96)]
    call_sizes: list[int] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query"
        assert config.enabled
        call_sizes.append(len(chunks))
        return _result(list(reversed(chunks)))

    result = rerank_regulatory_candidates_in_stages(
        query="focused query",
        chunks=candidates,
        config=_config(),
        rerank=rerank,
    )

    assert call_sizes == [32, 32, 32, 48]
    assert result.used_external
    assert len(result.ordered_chunks) == 96
    assert {item.regulatory_chunk_id for item in result.ordered_chunks} == {
        item.regulatory_chunk_id for item in candidates
    }
    assert result.ordered_chunks[0].regulatory_chunk_id == "80"


def test_staged_rerank_keeps_complete_baseline_after_batch_failure() -> None:
    candidates = [chunk(str(index)) for index in range(96)]
    call_sizes: list[int] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query"
        assert config.enabled
        call_sizes.append(len(chunks))
        if len(call_sizes) == 2:
            return _result(chunks, outcome=RerankOutcome.TIMEOUT)
        return _result(list(reversed(chunks)))

    result = rerank_regulatory_candidates_in_stages(
        query="focused query",
        chunks=candidates,
        config=_config(),
        rerank=rerank,
    )

    assert call_sizes == [32, 32]
    assert result.fallback_used
    assert result.outcome is RerankOutcome.TIMEOUT
    assert result.ordered_chunks == candidates


def test_staged_rerank_divides_smaller_final_budget_across_all_batches() -> None:
    candidates = [chunk(str(index)) for index in range(96)]
    call_sizes: list[int] = []

    def rerank(
        *,
        query: str,
        chunks: Sequence[InferenceChunk],
        config: RerankerRuntimeConfig,
    ) -> RerankResult:
        assert query == "focused query"
        assert config.enabled
        call_sizes.append(len(chunks))
        return _result(chunks)

    result = rerank_regulatory_candidates_in_stages(
        query="focused query",
        chunks=candidates,
        config=_config(),
        rerank=rerank,
        final_limit=32,
    )

    assert call_sizes == [32, 32, 32, 32]
    assert {item.regulatory_chunk_id for item in result.ordered_chunks[:32]} & {
        "0",
        "32",
        "64",
    } == {"0", "32", "64"}


def test_staged_rerank_uses_one_call_for_small_pool() -> None:
    candidates = [chunk(str(index)) for index in range(40)]
    call_sizes: list[int] = []

    def rerank(
        *, query: str, chunks: Sequence[InferenceChunk], config: RerankerRuntimeConfig
    ) -> RerankResult:
        assert query == "focused query"
        assert config.enabled
        call_sizes.append(len(chunks))
        return _result(chunks)

    result = rerank_regulatory_candidates_in_stages(
        query="focused query",
        chunks=candidates,
        config=_config(),
        rerank=rerank,
    )

    assert call_sizes == [40]
    assert result.ordered_chunks == candidates
