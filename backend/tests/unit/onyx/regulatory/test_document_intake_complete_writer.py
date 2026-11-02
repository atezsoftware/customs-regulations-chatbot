"""Whole-table requests preserve exact text through freezing, encoding, and replay."""

import json
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document
from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.document_index.publication_models import FrozenPublicationProjection
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.models import DocAwareChunk
from onyx.llm.interfaces import LLMConfig
from onyx.regulatory import projection, writer_projection
from onyx.regulatory.document_intake.models import IntakePreparationPolicy
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)
from tests.unit.onyx.regulatory.whole_table_encoder_helpers import _model, _table


@pytest.mark.parametrize("concurrent", [False, True])
def test_actual_writer_retains_both_encoder_receipts_and_sends_table_whole(
    concurrent: bool,
) -> None:
    file_id = uuid4()
    authority = OwnedAuthority(file_id)
    table, body = (
        canonical_row(file_id, 0, _table()),
        canonical_row(file_id, 1, "Short body"),
    )
    table.chunk_type = "table"
    for row in (table, body):
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_chunking_version": "4",
            "intake_preparation_policy": IntakePreparationPolicy().model_dump(
                mode="json"
            ),
        }
    inputs = writer_inputs(file_id, [table, body])
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name="whole-table",
        model_name="gemini-embedding-2",
        provider_type=EmbeddingProvider.GOOGLE,
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    inputs = replace(inputs, settings=[settings])
    model = _model()
    client = MagicMock()
    client.indices.get.return_value = {
        settings.index_name: {"settings": {"index": {"uuid": "physical-test"}}}
    }
    with (
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
        ),
        patch.object(
            writer_projection, "PublicationStore", side_effect=authority.for_scope
        ),
        patch.object(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            return_value=MagicMock(embedding_model=model),
        ),
        patch.object(
            writer_projection, "resolve_review_context_llm", return_value=None
        ),
        patch.object(
            model,
            "encode",
            side_effect=lambda texts, **_kwargs: [[1.0, 0.0, 0.0] for _ in texts],
        ) as encode,
    ):
        manifest = writer_projection.prepare_owned_correction(
            authority.owner,
            client,
            inputs,
            inputs.canonical,
            changed_id=None,
            before_embedding_request=(lambda: None) if concurrent else None,
        )
    table_binding = next(
        binding
        for binding in manifest.bindings
        if isinstance(binding.projection, FrozenPublicationProjection)
        and _table() in binding.projection.embedding_inputs
    )
    assert isinstance(table_binding.projection, FrozenPublicationProjection)
    config = json.loads(table_binding.projection.embedding_config_json)
    assert config["max_sequence_length"] == 2048 and config["strict_full_input"] is True
    assert table_binding.index.accepts_encoder_configuration(config)
    assert manifest.indexes[0].accepts_encoder_configuration(config)
    table_call = next(
        call for call in encode.call_args_list if _table() in call.kwargs["texts"]
    )
    assert (
        table_call.kwargs["max_seq_length"] == 2048
        and table_call.kwargs["strict_full_input"] is True
    )
    body_call = next(
        call for call in encode.call_args_list if "Short body" in call.kwargs["texts"]
    )
    assert "strict_full_input" not in body_call.kwargs


@pytest.mark.parametrize("version", ["1", "2", "3", "4"])
def test_disabled_context_cannot_create_a_whole_table_generation(version: str) -> None:
    def generation() -> str:
        return compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.GOOGLE,
            embedding_model_name="gemini-embedding-2",
            enable_contextual_rag=False,
            intake_chunking_version=version,
            intake_preparation_policy=IntakePreparationPolicy()
            if version == "4"
            else None,
        )

    if version == "4":
        with pytest.raises(ValueError, match="requires contextual retrieval"):
            generation()
    else:
        assert len(generation()) == 64


def test_generic_indexing_cannot_silently_retrim_a_whole_table_document() -> None:
    embedder = DefaultIndexingEmbedder(
        model_name="gemini-embedding-2",
        normalize=True,
        query_prefix=None,
        passage_prefix=None,
        provider_type=EmbeddingProvider.GOOGLE,
    )
    document = Document(
        id="source",
        source=DocumentSource.USER_FILE,
        semantic_identifier="table",
        sections=[],
        metadata={},
        doc_metadata={"regulatory_document_intake": {"chunking_version": "4"}},
    )
    chunk = DocAwareChunk.model_construct(source_document=document)
    with patch.object(embedder.embedding_model, "encode") as encode:
        with pytest.raises(ValueError, match="scoped regulatory writer"):
            embedder.embed_chunks([chunk])
        encode.assert_not_called()


def test_context_generation_keeps_full_table_and_fits_generated_fields_to_2048() -> (
    None
):
    file_id = uuid4()
    table, body = (
        canonical_row(file_id, 0, _table()),
        canonical_row(file_id, 1, "Short body"),
    )
    table.chunk_type = "table"
    for row in (table, body):
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_chunking_version": "4",
            "intake_preparation_policy": IntakePreparationPolicy().model_dump(
                mode="json"
            ),
        }
    inputs = writer_inputs(file_id, [table, body])
    model = _model()
    settings = SearchSettings(
        id=1,
        model_name="gemini-embedding-2",
        provider_type=EmbeddingProvider.GOOGLE,
        model_dim=3,
        normalize=True,
        enable_contextual_rag=True,
    )
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=100000,
    )

    def generated(**kwargs: Any) -> None:
        assert kwargs["chunk_token_limit"] == 2048
        assert any(chunk.content == _table() for chunk in kwargs["chunks"])
        for chunk in kwargs["chunks"]:
            chunk.doc_summary = "Belgenin tam satırları korunur. " * 100
            chunk.chunk_context = (
                "Tarife kodu ürün tanımıyla ve menşe koşuluyla birlikte değerlendirilir. "
                * 100
            )

    with (
        patch.object(projection, "effective_contextual_rag_enabled", return_value=True),
        patch.object(
            projection, "_get_contextual_tokenizer", return_value=model.tokenizer
        ),
        patch(
            "onyx.indexing.indexing_pipeline.add_contextual_summaries",
            side_effect=generated,
        ),
    ):
        view = projection.prepare_normal_context_view(
            rows=[table, body],
            user_file=inputs.file,
            search_settings=settings,
            embedder=MagicMock(embedding_model=model),
            llm=llm,
        )
    complete = next(
        context
        for context in view.projections
        if context.canonical_chunk_id == table.id
    )
    assert _table() in complete.embedding_texts[0]
    assert len(model.tokenizer.encode(complete.embedding_texts[0])) <= 2048
    assert complete.doc_summary.strip() or complete.chunk_context.strip()


def test_legacy_rows_keep_the_configured_encoder_without_policy_inference() -> None:
    from onyx.regulatory.document_intake import encoder_budget

    row = canonical_row(uuid4(), 0, "Legacy source")
    with patch.object(encoder_budget, "DOC_EMBEDDING_CONTEXT_SIZE", 256):
        assert encoder_budget.row_encoder_options(row, _model()) == {}
