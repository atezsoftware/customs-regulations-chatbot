from unittest.mock import Mock, patch

import pytest

from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.server.features.regulatory.library_rerank import rerank_library_candidates
from shared_configs.enums import RerankerProvider
from tests.unit.onyx.regulatory.labeling.test_search_overlay import chunk


def test_library_scores_every_candidate_in_the_provider_supported_batch() -> None:
    candidates = [chunk(str(index)) for index in range(100)]
    config = RerankerRuntimeConfig(
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="voyageai/rerank-3",
        api_key=None,
        configuration_generation="test",
    )
    calls = []

    def score(**kwargs):
        assert kwargs["config"].model_name == "cohere/rerank-5-pro"
        batch = kwargs["chunks"]
        calls.append([item.regulatory_chunk_id for item in batch])
        return RerankResult(
            ordered_chunks=batch,
            scores_by_chunk={
                (item.document_id, item.chunk_id): float(index)
                for index, item in enumerate(batch)
            },
            submitted_count=len(batch),
            result_count=len(batch),
            outcome=RerankOutcome.SUCCESS,
            fallback_used=False,
        )

    result = rerank_library_candidates(
        query="İthalat ihtisas", chunks=candidates, config=config, rerank=score
    )
    assert calls == [[item.regulatory_chunk_id for item in candidates]]
    assert len(result.scores_by_chunk) == 100
    assert config.model_name == "voyageai/rerank-3"
    assert {item.regulatory_chunk_id for item in result.ordered_chunks} == set(calls[0])


def test_library_respects_provider_cooldown_and_retries_the_same_batch() -> None:
    candidates = [chunk("1"), chunk("2")]
    config = Mock(model_name="voyageai/rerank-3", enabled=True)
    limited = RerankResult(
        ordered_chunks=candidates,
        scores_by_chunk={},
        submitted_count=2,
        result_count=0,
        outcome=RerankOutcome.RATE_LIMITED,
        fallback_used=True,
        retry_after_seconds=2,
    )
    success = limited.model_copy(
        update={
            "outcome": RerankOutcome.SUCCESS,
            "fallback_used": False,
            "result_count": 2,
            "scores_by_chunk": {
                (item.document_id, item.chunk_id): float(index)
                for index, item in enumerate(candidates)
            },
        }
    )
    score = Mock(side_effect=[limited, success])
    with patch("onyx.server.features.regulatory.library_rerank.time.sleep") as sleep:
        result = rerank_library_candidates(
            query="İthalat ihtisas", chunks=candidates, config=config, rerank=score
        )
    assert result.used_external
    sleep.assert_called_once()
    assert 2 <= sleep.call_args.args[0] < 3
    assert score.call_count == 2
    assert (
        score.call_args_list[0].kwargs["chunks"]
        == score.call_args_list[1].kwargs["chunks"]
    )


@pytest.mark.parametrize("cooldown", [60, 300])
def test_library_leaves_a_long_provider_cooldown_as_an_explicit_failure(
    cooldown: int,
) -> None:
    candidates = [chunk("1")]
    config = Mock(model_name="voyageai/rerank-3", enabled=True)
    limited = RerankResult(
        ordered_chunks=candidates,
        scores_by_chunk={},
        submitted_count=1,
        result_count=0,
        outcome=RerankOutcome.RATE_LIMITED,
        fallback_used=True,
        retry_after_seconds=cooldown,
    )
    score = Mock(return_value=limited)
    with patch("onyx.server.features.regulatory.library_rerank.time.sleep") as sleep:
        result = rerank_library_candidates(
            query="İthalat ihtisas", chunks=candidates, config=config, rerank=score
        )
    assert not result.used_external
    score.assert_called_once()
    sleep.assert_not_called()


def test_library_rejects_more_than_100_candidates_before_calling_provider() -> None:
    score = Mock()
    with pytest.raises(ValueError, match="100"):
        rerank_library_candidates(
            query="İthalat ihtisas",
            chunks=[chunk(str(index)) for index in range(101)],
            config=Mock(),
            rerank=score,
        )
    score.assert_not_called()
