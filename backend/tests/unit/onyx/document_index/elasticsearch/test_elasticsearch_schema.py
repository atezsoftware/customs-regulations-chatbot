from onyx.document_index.elasticsearch.schema import (
    CONTENT_VECTOR_FIELD_NAME,
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
