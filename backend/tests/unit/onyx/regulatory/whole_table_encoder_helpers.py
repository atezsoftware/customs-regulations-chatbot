"""Whole-table requests preserve exact text through freezing, encoding, and replay."""

from onyx.natural_language_processing.search_nlp_models import (
    EmbeddingModel,
)
from onyx.natural_language_processing.utils import TiktokenTokenizer
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)


def _model() -> EmbeddingModel:
    model = _durable_embedding_model()
    model.provider_type = EmbeddingProvider.GOOGLE
    model.model_name = "gemini-embedding-2"
    model.tokenizer = TiktokenTokenizer(
        "gemini-embedding-2", fallback_encoding_name="cl100k_base"
    )
    return model


def _table() -> str:
    return (
        "| 8544 | " + "Özgün menşe koşulu aynen korunur. " * 70 + "SON_SATIR_KORUNDU |"
    )
