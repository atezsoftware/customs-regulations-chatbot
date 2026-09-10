"""Configured receipts stay exact while proven OpenRouter transports share space."""

import json

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt


def _sync():
    return dict(
        provider="EmbeddingProvider.OPENROUTER",
        model="openai/text-embedding-3-large",
        dimension=3,
        reduced_dimension=3,
        normalize=True,
        passage_prefix="",
        retrim_content=False,
        endpoint_sha256=context_hash(None),
        api_version=None,
        deployment_name=None,
        tokenizer="test.Tokenizer",
        max_sequence_length=512,
        formatter="durable-context-before-text-v1",
        text_type="passage",
    )


def _batch():
    return dict(
        provider="openrouter",
        model="openai/text-embedding-3-large",
        dimension=3,
        endpoint_sha256=context_hash("https://openrouter.ai/api/beta/batches"),
        transport="openrouter_batch",
        embedding_endpoint="/v1/embeddings",
        formatter="durable-context-before-text-v1",
    )


def test_native_authority_retains_original_receipt_bytes():
    sync, batch = [
        _encoder_receipt(config, resolution="1" * 64) for config in (_sync(), _batch())
    ]
    retained = sync.model_dump_json(), batch.model_dump_json()
    assert sync.authority != batch.authority
    assert sync.effective_authority() == batch.effective_authority()
    assert sync.effective_authority().version == "openrouter-native-v1"
    assert retained == (sync.model_dump_json(), batch.model_dump_json())
    assert json.loads(sync.configuration_json)["normalize"] is True
    assert "normalize" not in json.loads(batch.configuration_json)


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint_sha256": context_hash("https://custom.example/api/beta/batches")},
        {"model": "other-model"},
        {"dimension": 4},
        {"embedding_endpoint": "/other"},
        {"input_type": "query"},
        {"provider_preferences": "custom"},
    ],
)
def test_unproven_batch_space_is_not_equivalent(change):
    sync = _encoder_receipt(_sync(), resolution="1" * 64)
    batch = _encoder_receipt(_batch() | change, resolution="1" * 64)
    assert sync.effective_authority() != batch.effective_authority()


@pytest.mark.parametrize(
    "change",
    [
        {"passage_prefix": "passage: "},
        {"endpoint_sha256": context_hash("https://custom.example")},
        {"input_type": "query"},
        {"text_type": "query"},
    ],
)
def test_unproven_sync_space_is_not_equivalent(change):
    sync = _encoder_receipt(_sync() | change, resolution="1" * 64)
    batch = _encoder_receipt(_batch(), resolution="1" * 64)
    assert sync.effective_authority() != batch.effective_authority()


def test_legacy_provider_spelling_retains_both_exact_temporal_selectors():
    from onyx.document_index.publication_models import PublicationIndexSnapshot

    def index(config):
        receipt = _encoder_receipt(config, resolution="1" * 64)
        assert receipt.authority.provider is not None
        return PublicationIndexSnapshot(
            index_name="owned",
            index_uuid="physical",
            search_settings_id=1,
            model_provider=receipt.authority.provider,
            model_name=receipt.authority.model,
            vector_dimension=3,
            embedding_config_sha256=context_hash(config),
            multitenant=True,
            encoder_authority=receipt.authority,
            encoder_receipts=(receipt,),
        )

    sync, batch = index(_sync()), index(_batch())
    retained = sync.model_dump_json(), batch.model_dump_json()
    assert sync.temporal_lookup_identity() != batch.temporal_lookup_identity()
    assert sync.matches_temporal_index(batch)
    assert sync.temporal_lookup_identity() in batch.temporal_lookup_identities()
    assert retained == (sync.model_dump_json(), batch.model_dump_json())
    for change in (
        {"index_uuid": "replaced"},
        {"model_name": "other"},
        {"model_provider": "cohere"},
    ):
        assert not sync.matches_temporal_index(batch.model_copy(update=change))
