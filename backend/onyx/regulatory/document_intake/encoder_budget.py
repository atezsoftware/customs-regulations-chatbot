"""Use frozen intake table budgets at every encoder and publication boundary."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from onyx.regulatory.document_intake.embedding_policy import (
    TABLE_EMBEDDING_TOKEN_LIMIT,
    embedding_token_limit_for_metadata,
    preparation_policy_for_model,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel


class EncoderBudgetOptions(TypedDict, total=False):
    max_seq_length: int
    strict_full_input: bool


def row_encoder_options(
    row: "RegulatoryChunk", model: "EmbeddingModel"
) -> EncoderBudgetOptions:
    if row.chunk_metadata.get("intake_chunking_version") == "4" and (
        model.provider_type is None or model.model_name is None
    ):
        raise ValueError("The frozen whole-table encoder is unavailable")
    limit = embedding_token_limit_for_metadata(
        row.chunk_metadata,
        row.chunk_type or "",
        provider=model.provider_type.value if model.provider_type else None,
        model_name=model.model_name,
    )
    return (
        {"max_seq_length": limit, "strict_full_input": True}
        if limit > DOC_EMBEDDING_CONTEXT_SIZE
        else {}
    )


def frozen_encoder_options(configuration: Mapping[str, object]) -> EncoderBudgetOptions:
    if "strict_full_input" not in configuration:
        return {}
    provider = configuration.get("provider")
    policy = preparation_policy_for_model(
        "google" if provider in ("google", "EmbeddingProvider.GOOGLE") else None,
        str(configuration.get("model")),
    )
    if (
        configuration["strict_full_input"] is not True
        or configuration.get("max_sequence_length") != TABLE_EMBEDDING_TOKEN_LIMIT
        or policy is None
    ):
        raise ValueError("The frozen whole-table encoder configuration is invalid")
    return {"max_seq_length": TABLE_EMBEDDING_TOKEN_LIMIT, "strict_full_input": True}
