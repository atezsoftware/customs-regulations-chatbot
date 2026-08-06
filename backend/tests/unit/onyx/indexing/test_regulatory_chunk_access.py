from unittest.mock import MagicMock, patch

from onyx.access.models import DocumentAccess
from onyx.indexing.adapters.user_file_indexing_adapter import UserFileChunkEnricher


def test_regulatory_user_file_chunks_use_file_acl_and_document_sets() -> None:
    private_access = DocumentAccess.build(
        user_emails=["owner@example.com"],
        user_groups=[],
        external_user_emails=[],
        external_user_group_ids=[],
        is_public=False,
    )
    enricher = UserFileChunkEnricher(
        user_file_id_to_access={"file-id": private_access},
        user_file_id_to_project_ids={},
        user_file_id_to_persona_ids={},
        user_file_id_to_document_set_names={"file-id": ["Private regulations"]},
        doc_id_to_previous_chunk_cnt={},
        doc_id_to_new_chunk_cnt={},
        user_file_id_to_raw_text={},
        user_file_id_to_token_count={},
        no_access=private_access,
        tenant_id="tenant",
    )
    chunk = MagicMock()
    chunk.source_document.id = "file-id"
    chunk.regulatory_chunk_id = "regulatory-id"

    with patch(
        "onyx.indexing.adapters.user_file_indexing_adapter."
        "DocMetadataAwareIndexChunk.from_index_chunk"
    ) as from_index_chunk:
        enricher.enrich_chunk(chunk, 1.0)

    assert from_index_chunk.call_args.kwargs["access"].is_public is False
    assert from_index_chunk.call_args.kwargs["document_sets"] == {"Private regulations"}
