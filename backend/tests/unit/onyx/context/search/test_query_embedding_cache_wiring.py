from unittest.mock import MagicMock, patch

from onyx.context.search.utils import (
    get_query_embeddings,
    prime_query_embedding_cache,
)
from shared_configs.enums import EmbeddingProvider


def test_prefetched_embedding_model_uses_its_search_settings_cache_identity() -> None:
    model = MagicMock()
    model.search_settings_id = 17
    model.provider_type = EmbeddingProvider.GOOGLE
    cached_embedding = [0.1, 0.2, 0.3]

    with (
        patch(
            "onyx.context.search.utils.get_cached_query_embeddings",
            return_value=[cached_embedding],
        ) as cache_get,
        patch("onyx.context.search.utils.QUERY_EMBEDDING_CACHE_ENABLED", True),
    ):
        result = get_query_embeddings(["focused query"], embedding_model=model)

    assert result == [cached_embedding]
    model.encode.assert_not_called()
    assert cache_get.call_args.kwargs["search_settings_id"] == 17


def test_gemini_embedding_two_prime_uses_parallel_single_content_calls() -> None:
    model = MagicMock()
    model.provider_type = EmbeddingProvider.GOOGLE
    model.model_name = "gemini-embedding-2"

    with (
        patch(
            "onyx.context.search.utils.get_current_search_settings",
            return_value=MagicMock(),
        ),
        patch(
            "onyx.context.search.utils.EmbeddingModel.from_db_model",
            return_value=model,
        ),
        patch("onyx.context.search.utils.run_functions_tuples_in_parallel") as parallel,
    ):
        prime_query_embedding_cache(
            ["first", "second", "third"],
            db_session=MagicMock(),
            max_workers=8,
        )

    assert [args[1][0] for args in parallel.call_args.args[0]] == [
        "first",
        "second",
        "third",
    ]
    assert parallel.call_args.kwargs["max_workers"] == 8
