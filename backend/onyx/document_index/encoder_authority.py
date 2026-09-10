"""Versioned interpretation of retained encoder receipts; no evidence rewriting."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from onyx.document_index.publication_models import (
    PublicationEncoderAuthority,
    PublicationEncoderReceipt,
    publication_digest,
)
from shared_configs.enums import EmbeddingProvider

_SYNC_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
_BATCH_ENDPOINT = "https://openrouter.ai/api/beta/batches"
_BATCH_KEYS = frozenset(
    {
        "provider",
        "model",
        "dimension",
        "endpoint_sha256",
        "transport",
        "embedding_endpoint",
        "formatter",
    }
)
_SYNC_KEYS = frozenset(
    {
        "provider",
        "model",
        "dimension",
        "reduced_dimension",
        "normalize",
        "passage_prefix",
        "retrim_content",
        "endpoint_sha256",
        "api_version",
        "deployment_name",
        "tokenizer",
        "max_sequence_length",
        "formatter",
        "text_type",
    }
)


class EffectiveEncoderAuthority(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal["configured-v1", "openrouter-native-v1"]
    authority: PublicationEncoderAuthority


def _native(authority: PublicationEncoderAuthority) -> EffectiveEncoderAuthority:
    return EffectiveEncoderAuthority(
        version="openrouter-native-v1",
        authority=authority.model_copy(
            update={
                "provider": EmbeddingProvider.OPENROUTER.value,
                "endpoint_sha256": publication_digest(_SYNC_ENDPOINT),
                "normalize": None,
                "passage_prefix": None,
                "deployment_name": None,
                "api_version": None,
            }
        ),
    )


def effective_receipt_authority(
    receipt: PublicationEncoderReceipt,
) -> EffectiveEncoderAuthority:
    """Only the inspected native sync and official batch payloads prove equivalence."""
    authority = receipt.authority
    config = json.loads(receipt.configuration_json)
    if authority.provider in (
        EmbeddingProvider.OPENROUTER.value,
        str(EmbeddingProvider.OPENROUTER),
    ):
        if (
            set(config) == _BATCH_KEYS
            and config.get("transport") == "openrouter_batch"
            and config.get("embedding_endpoint") == "/v1/embeddings"
            and authority.endpoint_sha256 == publication_digest(_BATCH_ENDPOINT)
        ):
            return _native(authority)
        transport = config.get("transport")
        keys = set(config) - (
            {"transport"} if transport == "synchronous_encoder" else set()
        )
        if (
            keys == _SYNC_KEYS
            and transport in (None, "synchronous_encoder")
            and config.get("text_type") == "passage"
            and authority.passage_prefix in (None, "")
            and authority.deployment_name is None
            and authority.api_version is None
            and authority.endpoint_sha256
            in {publication_digest(value) for value in (None, "", _SYNC_ENDPOINT)}
        ):
            return _native(authority)
    return EffectiveEncoderAuthority(version="configured-v1", authority=authority)


def effective_runtime_authority(
    authority: PublicationEncoderAuthority,
    *,
    query_prefix: str | None,
) -> EffectiveEncoderAuthority:
    """Actual CloudEmbedding OpenRouter query execution has native, prefix-free output."""
    if (
        authority.provider
        in (EmbeddingProvider.OPENROUTER.value, str(EmbeddingProvider.OPENROUTER))
        and query_prefix in (None, "")
        and authority.passage_prefix in (None, "")
        and authority.deployment_name is None
        and authority.api_version is None
        and authority.endpoint_sha256
        in {publication_digest(value) for value in (None, "", _SYNC_ENDPOINT)}
    ):
        return _native(authority)
    return EffectiveEncoderAuthority(version="configured-v1", authority=authority)
