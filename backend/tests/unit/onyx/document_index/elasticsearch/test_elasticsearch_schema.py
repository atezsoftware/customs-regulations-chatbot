from unittest.mock import MagicMock

import pytest

from onyx.configs.app_configs import ELASTICSEARCH_TEXT_ANALYZER
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchSchemaMigrationRequiredError,
    ensure_current_schema,
)
from onyx.document_index.elasticsearch.schema import (
    CONTENT_FIELD_NAME,
    CONTENT_VECTOR_FIELD_NAME,
    DECISION_NUMBERS_FIELD_NAME,
    HEADING_PATH_FIELD_NAME,
    LEGAL_DATES_FIELD_NAME,
    PROVISION_IDENTIFIERS_FIELD_NAME,
    TITLE_FIELD_NAME,
    TITLE_VECTOR_FIELD_NAME,
    DocumentSchema,
)


def test_vector_fields_use_indexed_elasticsearch_hnsw_dense_vectors() -> None:
    properties = DocumentSchema.get_document_schema(
        vector_dimension=768, multitenant=False
    )["properties"]

    for field_name in (TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME):
        vector_mapping = properties[field_name]
        assert vector_mapping == {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
            "index_options": {
                "type": "hnsw",
                "ef_construction": 256,
                "m": 32,
            },
        }


def test_turkish_legal_text_mapping_has_exact_subfields() -> None:
    properties = DocumentSchema.get_document_schema(
        vector_dimension=768, multitenant=False
    )["properties"]

    assert ELASTICSEARCH_TEXT_ANALYZER == "turkish"
    assert properties[TITLE_FIELD_NAME]["analyzer"] == "turkish"
    assert properties[CONTENT_FIELD_NAME]["analyzer"] == "turkish"
    assert properties[HEADING_PATH_FIELD_NAME] == {
        "type": "text",
        "analyzer": "turkish",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    }
    assert properties[PROVISION_IDENTIFIERS_FIELD_NAME] == {"type": "keyword"}
    assert properties[DECISION_NUMBERS_FIELD_NAME] == {"type": "keyword"}
    assert properties[LEGAL_DATES_FIELD_NAME] == {
        "type": "date",
        "format": "strict_date",
    }


def test_schema_honors_explicit_custom_text_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.document_index.elasticsearch.schema.ELASTICSEARCH_TEXT_ANALYZER",
        "custom_legal_analyzer",
    )

    properties = DocumentSchema.get_document_schema(
        vector_dimension=768, multitenant=False
    )["properties"]

    assert properties[TITLE_FIELD_NAME]["analyzer"] == "custom_legal_analyzer"
    assert properties[CONTENT_FIELD_NAME]["analyzer"] == "custom_legal_analyzer"
    assert properties[HEADING_PATH_FIELD_NAME]["analyzer"] == ("custom_legal_analyzer")


def _mismatched_index_client(indexed_chunk_count: int) -> MagicMock:
    client = MagicMock()
    client.get_index_mapping.return_value = {
        "properties": {CONTENT_FIELD_NAME: {"type": "text", "analyzer": "english"}}
    }
    client.count_by_query.return_value = indexed_chunk_count
    client.delete_index.return_value = True
    return client


def test_existing_empty_index_is_recreated_for_turkish_mapping() -> None:
    client = _mismatched_index_client(indexed_chunk_count=0)
    expected_mappings = DocumentSchema.get_document_schema(768, multitenant=False)
    index_settings = DocumentSchema.get_index_settings_based_on_environment()

    ensure_current_schema(
        index_client=client,
        expected_mappings=expected_mappings,
        index_settings=index_settings,
        database_has_indexed_documents=False,
    )

    client.delete_index.assert_called_once_with()
    client.create_index.assert_called_once_with(
        mappings=expected_mappings,
        settings=index_settings,
    )
    client.put_mapping.assert_not_called()


@pytest.mark.parametrize(
    ("database_has_indexed_documents", "indexed_chunk_count"),
    [(True, 0), (False, 1), (True, 1)],
)
def test_mismatched_nonempty_index_is_never_destroyed(
    database_has_indexed_documents: bool,
    indexed_chunk_count: int,
) -> None:
    client = _mismatched_index_client(indexed_chunk_count=indexed_chunk_count)

    with pytest.raises(
        ElasticsearchSchemaMigrationRequiredError,
        match="Create a new search index and reindex",
    ):
        ensure_current_schema(
            index_client=client,
            expected_mappings=DocumentSchema.get_document_schema(
                768, multitenant=False
            ),
            index_settings=DocumentSchema.get_index_settings_based_on_environment(),
            database_has_indexed_documents=database_has_indexed_documents,
        )

    client.delete_index.assert_not_called()
    client.create_index.assert_not_called()


def test_compatible_index_receives_additive_mapping_update() -> None:
    client = MagicMock()
    expected_mappings = DocumentSchema.get_document_schema(768, multitenant=False)
    client.get_index_mapping.return_value = expected_mappings

    ensure_current_schema(
        index_client=client,
        expected_mappings=expected_mappings,
        index_settings=DocumentSchema.get_index_settings_based_on_environment(),
        database_has_indexed_documents=True,
    )

    client.put_mapping.assert_called_once_with(expected_mappings)
    client.count_by_query.assert_not_called()
    client.delete_index.assert_not_called()
