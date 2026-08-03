import datetime
from unittest.mock import MagicMock

import pytest

from onyx.document_index.interfaces_new import TenantState
from onyx.document_index.opensearch.client import (
    OpenSearchDocumentMissingError,
    OpenSearchIndexClient,
    OpenSearchUpdateError,
)
from onyx.document_index.opensearch.opensearch_document_index import (
    OpenSearchDocumentIndex,
)
from onyx.document_index.opensearch.schema import (
    CHUNK_INDEX_FIELD_NAME,
    DOCUMENT_ID_FIELD_NAME,
    REGULATORY_CHUNK_ID_FIELD_NAME,
    VALIDITY_END_DATE_FIELD_NAME,
    VALIDITY_START_DATE_FIELD_NAME,
    get_opensearch_doc_chunk_id,
)
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA


def _document_index() -> tuple[OpenSearchDocumentIndex, MagicMock, TenantState]:
    tenant_state = TenantState(
        tenant_id=POSTGRES_DEFAULT_SCHEMA,
        multitenant=False,
    )
    index = OpenSearchDocumentIndex.__new__(OpenSearchDocumentIndex)
    client = MagicMock()
    index._client = client
    index._tenant_state = tenant_state
    index._index_name = "test-index"
    return index, client, tenant_state


def test_regulatory_validity_patch_preflights_exact_ids_and_updates_metadata() -> None:
    index, client, tenant_state = _document_index()

    index.update_regulatory_validity(
        document_id="doc-1",
        expected_regulatory_chunk_ids=["rc-a", "rc-b"],
        previous_start_date=None,
        previous_end_date=None,
        updated_start_date=datetime.date(2025, 1, 1),
        updated_end_date=None,
    )

    chunk_ids = [
        get_opensearch_doc_chunk_id(tenant_state, "doc-1", chunk_index)
        for chunk_index in range(2)
    ]
    client.validate_regulatory_chunk_projection.assert_called_once_with(
        {
            chunk_ids[0]: ("doc-1", 0, "rc-a"),
            chunk_ids[1]: ("doc-1", 1, "rc-b"),
        },
        validity_start_date=None,
        validity_end_date=None,
    )
    client.bulk_update_documents.assert_called_once_with(
        document_chunk_ids=chunk_ids,
        properties_to_update={
            VALIDITY_START_DATE_FIELD_NAME: 1_735_689_600,
            VALIDITY_END_DATE_FIELD_NAME: None,
        },
        surface_document_missing=True,
    )


def test_regulatory_validity_bulk_failure_is_compensated() -> None:
    index, client, _ = _document_index()
    client.bulk_update_documents.side_effect = [RuntimeError("write failed"), None]

    with pytest.raises(RuntimeError, match="write failed"):
        index.update_regulatory_validity(
            document_id="doc-1",
            expected_regulatory_chunk_ids=["rc-a"],
            previous_start_date=datetime.date(2024, 1, 1),
            previous_end_date=None,
            updated_start_date=datetime.date(2025, 1, 1),
            updated_end_date=None,
        )

    assert client.bulk_update_documents.call_count == 2
    rollback_call = client.bulk_update_documents.call_args_list[1]
    assert rollback_call.kwargs["properties_to_update"] == {
        VALIDITY_START_DATE_FIELD_NAME: 1_704_067_200,
        VALIDITY_END_DATE_FIELD_NAME: None,
    }


def test_regulatory_validity_preflight_failure_writes_nothing() -> None:
    index, client, _ = _document_index()
    client.validate_regulatory_chunk_projection.side_effect = OpenSearchUpdateError(
        "mismatch"
    )

    with pytest.raises(OpenSearchUpdateError, match="mismatch"):
        index.update_regulatory_validity(
            document_id="doc-1",
            expected_regulatory_chunk_ids=["rc-a"],
            previous_start_date=None,
            previous_end_date=None,
            updated_start_date=datetime.date(2025, 1, 1),
            updated_end_date=None,
        )

    client.bulk_update_documents.assert_not_called()


def _index_client() -> tuple[OpenSearchIndexClient, MagicMock]:
    client = OpenSearchIndexClient.__new__(OpenSearchIndexClient)
    raw_client = MagicMock()
    raw_client.count.return_value = {"count": 1, "_shards": {"failed": 0}}
    client._client = raw_client
    client._index_name = "test-index"
    return client, raw_client


def test_projection_preflight_fetches_only_identity_and_date_fields() -> None:
    client, raw_client = _index_client()
    raw_client.mget.return_value = {
        "docs": [
            {
                "_id": "os-0",
                "found": True,
                "_source": {
                    DOCUMENT_ID_FIELD_NAME: "doc-1",
                    CHUNK_INDEX_FIELD_NAME: 0,
                    REGULATORY_CHUNK_ID_FIELD_NAME: "rc-a",
                },
            }
        ]
    }

    client.validate_regulatory_chunk_projection(
        {"os-0": ("doc-1", 0, "rc-a")},
        validity_start_date=None,
        validity_end_date=None,
    )

    source_fields = raw_client.mget.call_args.kwargs["body"]["docs"][0]["_source"]
    assert source_fields == [
        DOCUMENT_ID_FIELD_NAME,
        CHUNK_INDEX_FIELD_NAME,
        REGULATORY_CHUNK_ID_FIELD_NAME,
        VALIDITY_START_DATE_FIELD_NAME,
        VALIDITY_END_DATE_FIELD_NAME,
    ]


def test_projection_preflight_rejects_stale_extra_chunk_before_identity_read() -> None:
    client, raw_client = _index_client()
    raw_client.count.return_value = {"count": 2, "_shards": {"failed": 0}}

    with pytest.raises(OpenSearchUpdateError, match="chunk count mismatch"):
        client.validate_regulatory_chunk_projection(
            {"os-0": ("doc-1", 0, "rc-a")},
            validity_start_date=None,
            validity_end_date=None,
        )

    raw_client.count.assert_called_once_with(
        index="test-index",
        body={
            "query": {
                "term": {DOCUMENT_ID_FIELD_NAME: {"value": "doc-1"}},
            }
        },
    )
    raw_client.mget.assert_not_called()


def test_projection_preflight_surfaces_missing_and_misaligned_chunks() -> None:
    client, raw_client = _index_client()
    raw_client.mget.return_value = {"docs": [{"_id": "os-0", "found": False}]}
    with pytest.raises(OpenSearchDocumentMissingError):
        client.validate_regulatory_chunk_projection(
            {"os-0": ("doc-1", 0, "rc-a")},
            validity_start_date=None,
            validity_end_date=None,
        )

    raw_client.mget.return_value = {
        "docs": [
            {
                "_id": "os-0",
                "found": True,
                "_source": {
                    DOCUMENT_ID_FIELD_NAME: "doc-1",
                    CHUNK_INDEX_FIELD_NAME: 0,
                    REGULATORY_CHUNK_ID_FIELD_NAME: "wrong-row",
                },
            }
        ]
    }
    with pytest.raises(OpenSearchUpdateError, match="mismatch"):
        client.validate_regulatory_chunk_projection(
            {"os-0": ("doc-1", 0, "rc-a")},
            validity_start_date=None,
            validity_end_date=None,
        )
