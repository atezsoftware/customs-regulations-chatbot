from unittest.mock import patch

import pytest

from onyx.natural_language_processing import utils
from onyx.natural_language_processing.utils import TiktokenTokenizer, get_tokenizer
from shared_configs.enums import EmbeddingProvider


@pytest.fixture(autouse=True)
def clear_tokenizer_caches() -> None:
    utils._TOKENIZER_CACHE.clear()
    TiktokenTokenizer._instances.clear()


def test_openrouter_embedding_model_uses_unqualified_tiktoken_name() -> None:
    with patch.object(
        utils,
        "HuggingFaceTokenizer",
        side_effect=AssertionError("cloud tokenizer must not use Hugging Face"),
    ):
        tokenizer = get_tokenizer(
            "openai/text-embedding-3-small", EmbeddingProvider.LITELLM
        )

    assert isinstance(tokenizer, TiktokenTokenizer)
    assert tokenizer.encoder.name == "cl100k_base"


def test_unknown_cloud_model_uses_offline_tiktoken_fallback() -> None:
    with patch.object(
        utils,
        "HuggingFaceTokenizer",
        side_effect=AssertionError("cloud tokenizer must not use Hugging Face"),
    ):
        tokenizer = get_tokenizer(
            "router/vendor-embedding-model-without-a-known-tokenizer",
            EmbeddingProvider.LITELLM,
        )

    assert isinstance(tokenizer, TiktokenTokenizer)
    assert tokenizer.encoder.name == "cl100k_base"
